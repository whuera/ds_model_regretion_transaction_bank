"""Contrato de entrada/salida de la API (sección 6 de CLAUDE.md)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class TransactionInput(BaseModel):
    transaction_type: Literal["Credit", "Debit"]
    channel: Literal["ATM", "Online", "Branch"]
    location: str
    customer_occupation: Literal["Doctor", "Engineer", "Student", "Retired"]
    customer_age: int
    transaction_duration: int
    login_attempts: int
    account_balance: float
    transaction_date: datetime


class PredictionOutput(BaseModel):
    # "model_version" colisiona con el namespace protegido "model_" de
    # pydantic v2; se desactiva para conservar el contrato de la sección 6.
    model_config = ConfigDict(protected_namespaces=())

    predicted_amount: float
    model_version: str
    prediction_timestamp: datetime
