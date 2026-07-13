"""Entrenamiento de los 6 modelos de regresión con tracking en MLflow.

Replica el flujo del notebook (KFold CV + métricas RMSE/MAE/R² sobre test)
y añade el pipeline de MLOps:
- una corrida de MLflow por modelo (hiperparámetros + métricas + artefacto),
- importancia de variables por impureza y por permutación,
- gate de calidad (sección 5 de CLAUDE.md):
    * R² > 0.90  -> `requiere_revision_manual = true`, el modelo NO se
      promueve automáticamente (hallazgo de fuga de datos del notebook),
    * el RMSE del candidato no puede empeorar más del umbral (5% por
      defecto) frente al modelo registrado como stage="Production".
- el mejor modelo que pasa el gate se guarda como pipeline completo
  (preprocesamiento + modelo) con joblib y se registra como candidato.

Uso:
    python -m src.train --data data/bank_transactions_data_2_augmented_clean_2.csv
    python -m src.train --sample-frac 0.1   # modo validación (CI)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import RANDOM_STATE, build_preprocessor, load_dataset

EXPERIMENT_NAME = "transacciones-bancarias-regresion"
REGISTERED_MODEL_NAME = "bank-transaction-regressor"

R2_ALERT_THRESHOLD = 0.90  # por encima de esto: sospecha de fuga de datos
RMSE_DEGRADATION_THRESHOLD = 0.05  # 5% frente al modelo en Production


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_models() -> dict[str, object]:
    return {
        "Baseline (media)": DummyRegressor(strategy="mean"),
        "Regresión Lineal": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(max_iter=3000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }


def log_feature_importances(pipe: Pipeline, X_test, y_test, n_repeats: int) -> None:
    """Loguea en MLflow la importancia por impureza (si aplica) y por permutación."""
    model = pipe.named_steps["modelo"]
    with tempfile.TemporaryDirectory() as tmp:
        if hasattr(model, "feature_importances_"):
            nombres = pipe.named_steps["preprocess"].get_feature_names_out()
            imp = (
                pd.Series(model.feature_importances_, index=nombres)
                .sort_values(ascending=False)
            )
            path = Path(tmp) / "importancia_impureza.json"
            path.write_text(json.dumps(imp.round(6).to_dict(), indent=2))
            mlflow.log_artifact(str(path))

        perm = permutation_importance(
            pipe, X_test, y_test, n_repeats=n_repeats,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        imp_perm = (
            pd.Series(perm.importances_mean, index=X_test.columns)
            .sort_values(ascending=False)
        )
        path = Path(tmp) / "importancia_permutacion.json"
        path.write_text(json.dumps(imp_perm.round(6).to_dict(), indent=2))
        mlflow.log_artifact(str(path))


def production_rmse(client: mlflow.MlflowClient) -> float | None:
    """RMSE en test del modelo actualmente en stage='Production', si existe."""
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        return None
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get("test_rmse")


def train(args: argparse.Namespace) -> int:
    X, y = load_dataset(args.data)
    if args.sample_frac < 1.0:
        X = X.sample(frac=args.sample_frac, random_state=RANDOM_STATE)
        y = y.loc[X.index]
        print(f"[modo validación] usando {len(X)} filas ({args.sample_frac:.0%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": make_scorer(rmse, greater_is_better=False),
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.MlflowClient()
    resultados = []

    with mlflow.start_run(run_name="comparacion-modelos") as parent_run:
        mlflow.log_params(
            {
                "cv_folds": args.cv_folds,
                "sample_frac": args.sample_frac,
                "test_size": 0.2,
                "random_state": RANDOM_STATE,
                "n_rows": len(X),
            }
        )

        for nombre, modelo in build_models().items():
            with mlflow.start_run(run_name=nombre, nested=True) as run:
                pipe = Pipeline([("preprocess", build_preprocessor()), ("modelo", modelo)])

                cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)

                metricas = {
                    "cv_rmse": float(-cv_res["test_rmse"].mean()),
                    "cv_mae": float(-cv_res["test_mae"].mean()),
                    "cv_r2": float(cv_res["test_r2"].mean()),
                    "test_rmse": rmse(y_test, pred),
                    "test_mae": float(mean_absolute_error(y_test, pred)),
                    "test_r2": float(r2_score(y_test, pred)),
                }
                mlflow.log_params(modelo.get_params())
                mlflow.log_metrics(metricas)
                mlflow.sklearn.log_model(pipe, artifact_path="model")
                log_feature_importances(pipe, X_test, y_test, n_repeats=args.perm_repeats)

                # Gate 1: R² sospechosamente alto => posible fuga de datos.
                sospechoso = metricas["test_r2"] > args.r2_alert
                mlflow.set_tag("requiere_revision_manual", str(sospechoso).lower())
                if sospechoso:
                    print(
                        f"⚠️  {nombre}: R²={metricas['test_r2']:.3f} > {args.r2_alert} "
                        "-> requiere_revision_manual=true (posible fuga de datos, "
                        "no se promueve automáticamente)"
                    )

                resultados.append(
                    {"modelo": nombre, "run_id": run.info.run_id,
                     "pipeline": pipe, "sospechoso": sospechoso, **metricas}
                )

        tabla = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "pipeline"} for r in resultados]
        ).sort_values("test_rmse")
        print("\n=== Resultados (ordenados por RMSE en test) ===")
        print(tabla.drop(columns=["run_id"]).to_string(index=False))

        # Candidato: mejor RMSE entre modelos NO marcados como sospechosos
        # (el baseline no se promueve nunca).
        elegibles = [
            r for r in resultados
            if not r["sospechoso"] and "Baseline" not in r["modelo"]
        ]
        if not elegibles:
            print("\n❌ Ningún modelo pasa el gate de R²: todos requieren revisión manual.")
            mlflow.set_tag("gate_resultado", "sin_candidato")
            return 1

        candidato = min(elegibles, key=lambda r: r["test_rmse"])
        print(f"\nCandidato a producción: {candidato['modelo']} "
              f"(RMSE test = {candidato['test_rmse']:.2f})")

        # Gate 2: no empeorar el RMSE más del umbral vs. Production.
        rmse_prod = production_rmse(client)
        if rmse_prod is not None:
            degradacion = (candidato["test_rmse"] - rmse_prod) / rmse_prod
            mlflow.log_metric("rmse_degradacion_vs_produccion", degradacion)
            print(f"RMSE en Production: {rmse_prod:.2f} | degradación: {degradacion:+.2%}")
            if degradacion > args.rmse_threshold:
                print(
                    f"❌ Gate de RMSE: el candidato empeora {degradacion:.2%} "
                    f"(> {args.rmse_threshold:.0%}). No se registra."
                )
                mlflow.set_tag("gate_resultado", "rmse_degradado")
                return 1
        else:
            print("No hay modelo en stage='Production'; se omite la comparación de RMSE.")

        mlflow.set_tag("gate_resultado", "aprobado")

        # Registrar el candidato en el Model Registry y guardar el .joblib.
        version = None
        if args.register:
            mv = mlflow.register_model(
                f"runs:/{candidato['run_id']}/model", REGISTERED_MODEL_NAME
            )
            version = mv.version
            print(f"Registrado como {REGISTERED_MODEL_NAME} v{version} (candidato).")

        bundle = {
            "pipeline": candidato["pipeline"],
            "model_name": candidato["modelo"],
            "model_version": str(version) if version else candidato["run_id"][:8],
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {k: candidato[k] for k in
                        ("cv_rmse", "cv_mae", "cv_r2", "test_rmse", "test_mae", "test_r2")},
        }
        out = Path(args.model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out)
        print(f"Pipeline completo guardado en {out}")

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/bank_transactions_data_2_augmented_clean_2.csv")
    p.add_argument("--model-out", default="models/model.joblib")
    p.add_argument("--sample-frac", type=float, default=1.0,
                   help="fracción del dataset (modo validación en CI, ej. 0.1)")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--perm-repeats", type=int, default=3)
    p.add_argument("--r2-alert", type=float, default=R2_ALERT_THRESHOLD)
    p.add_argument("--rmse-threshold", type=float, default=RMSE_DEGRADATION_THRESHOLD)
    p.add_argument("--no-register", dest="register", action="store_false",
                   help="no registrar el candidato en el Model Registry")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(train(parse_args()))
