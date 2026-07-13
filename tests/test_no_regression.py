"""Test de no-regresión: el RMSE del modelo nuevo no debe empeorar
más de un umbral configurable frente al modelo de referencia.

Variables de entorno:
    CANDIDATE_MODEL_PATH       modelo recién entrenado (default: models/model.joblib)
    REFERENCE_MODEL_PATH       modelo vigente (default: models/reference.joblib)
    RMSE_REGRESSION_THRESHOLD  degradación máxima permitida (default: 0.05 = 5%)

Si no existe modelo de referencia, se compara contra un baseline
(DummyRegressor con la media), que el candidato siempre debería superar.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.evaluate import evaluate_model
from src.preprocess import RANDOM_STATE, load_dataset

DATA_PATH = "data/bank_transactions_data_2_augmented_clean_2.csv"
CANDIDATE = os.getenv("CANDIDATE_MODEL_PATH", "models/model.joblib")
REFERENCE = os.getenv("REFERENCE_MODEL_PATH", "models/reference.joblib")
THRESHOLD = float(os.getenv("RMSE_REGRESSION_THRESHOLD", "0.05"))


def _baseline_rmse() -> float:
    """RMSE del baseline de referencia (media) sobre el mismo set de test."""
    X, y = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    return float(np.sqrt(mean_squared_error(y_test, dummy.predict(X_test))))


def test_rmse_no_empeora_mas_del_umbral():
    if not Path(CANDIDATE).exists():
        pytest.skip(
            f"No existe el modelo candidato en {CANDIDATE}; "
            "ejecuta primero `python -m src.train`."
        )

    rmse_nuevo = evaluate_model(CANDIDATE, DATA_PATH)["rmse"]

    if Path(REFERENCE).exists():
        rmse_ref = evaluate_model(REFERENCE, DATA_PATH)["rmse"]
        origen = f"modelo vigente ({REFERENCE})"
    else:
        rmse_ref = _baseline_rmse()
        origen = "baseline de referencia (media)"

    limite = rmse_ref * (1 + THRESHOLD)
    assert rmse_nuevo <= limite, (
        f"RMSE del candidato ({rmse_nuevo:.2f}) empeora más del "
        f"{THRESHOLD:.0%} frente al {origen} ({rmse_ref:.2f}); "
        f"límite permitido: {limite:.2f}"
    )
