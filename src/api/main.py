"""API de servicio del modelo de regresión de transacciones bancarias.

- Carga el pipeline serializado (.joblib) al arrancar (lifespan).
- POST /predict: predice el monto de una transacción.
- GET /health: estado + versión del modelo para el orquestador de despliegue.
- Cada predicción se registra en un archivo JSON-lines local.

Variables de entorno:
    MODEL_PATH           ruta del artefacto (default: models/model.joblib)
    MODEL_VERSION        sobreescribe la versión reportada del modelo
    PREDICTIONS_LOG_PATH ruta del log de predicciones (default: logs/predictions.jsonl)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictionOutput, TransactionInput

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
PREDICTIONS_LOG_PATH = os.getenv("PREDICTIONS_LOG_PATH", "logs/predictions.jsonl")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        _state["pipeline"] = bundle["pipeline"]
        _state["model_version"] = os.getenv(
            "MODEL_VERSION", bundle.get("model_version", "unknown")
        )
        _state["model_name"] = bundle.get("model_name", "unknown")
    else:  # artefacto antiguo: solo el pipeline
        _state["pipeline"] = bundle
        _state["model_version"] = os.getenv("MODEL_VERSION", "unknown")
        _state["model_name"] = "unknown"
    yield
    _state.clear()


app = FastAPI(title="Bank Transaction Amount API", lifespan=lifespan)


def _to_model_frame(t: TransactionInput) -> pd.DataFrame:
    """Convierte el input de la API a las columnas que espera el pipeline."""
    fecha = t.transaction_date
    dow = fecha.weekday()
    return pd.DataFrame(
        [
            {
                "TransactionType": t.transaction_type,
                "Location": t.location,
                "Channel": t.channel,
                "CustomerOccupation": t.customer_occupation,
                "CustomerAge": t.customer_age,
                "TransactionDuration": t.transaction_duration,
                "LoginAttempts": t.login_attempts,
                "AccountBalance": t.account_balance,
                "trans_year": fecha.year,
                "trans_month": fecha.month,
                "trans_dow": dow,
                "trans_hour": fecha.hour,
                "es_fin_de_semana": int(dow >= 5),
            }
        ]
    )


def _log_prediction(entrada: TransactionInput, salida: PredictionOutput) -> None:
    """Registra input/output/versión/timestamp en JSON-lines.

    Para producción real, reemplazar esta escritura local por un insert en
    una base de datos (p. ej. PostgreSQL) o un topic de eventos: este es el
    único punto de conexión que hay que cambiar.
    """
    registro = {
        "timestamp": salida.prediction_timestamp.isoformat(),
        "model_version": salida.model_version,
        "input": entrada.model_dump(mode="json"),
        "output": {"predicted_amount": salida.predicted_amount},
    }
    path = Path(PREDICTIONS_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(registro) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": _state.get("model_version", "unknown")}


@app.post("/predict", response_model=PredictionOutput)
def predict(transaccion: TransactionInput) -> PredictionOutput:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    df = _to_model_frame(transaccion)
    monto = float(pipeline.predict(df)[0])

    salida = PredictionOutput(
        predicted_amount=monto,
        model_version=_state["model_version"],
        prediction_timestamp=datetime.now(timezone.utc),
    )
    _log_prediction(transaccion, salida)
    return salida
