"""Evaluación de un pipeline serializado sobre el set de test.

Reconstruye el mismo split train/test del entrenamiento (mismo
random_state) y reporta RMSE, MAE y R².

Uso:
    python -m src.evaluate --model models/model.joblib \
        --data data/bank_transactions_data_2_augmented_clean_2.csv
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.preprocess import RANDOM_STATE, load_dataset


def evaluate_model(model_path: str, data_path: str) -> dict[str, float]:
    """Devuelve las métricas del pipeline guardado sobre el set de test."""
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"] if isinstance(bundle, dict) else bundle

    X, y = load_dataset(data_path)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pred = pipe.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/model.joblib")
    p.add_argument("--data", default="data/bank_transactions_data_2_augmented_clean_2.csv")
    args = p.parse_args(argv)

    metricas = evaluate_model(args.model, args.data)
    print(json.dumps(metricas, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
