"""Limpieza y preprocesamiento de transacciones bancarias.

Traducción productiva de la lógica del notebook
`proyecto_kaggle_transacciones_regresion_v2.ipynb`:
- parseo de fechas mixtas (`format="mixed"`),
- variables derivadas de la fecha,
- descarte de identificadores de alta cardinalidad,
- ColumnTransformer con imputación + escalado (numéricas) y
  imputación + One-Hot con `handle_unknown="ignore"` (categóricas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

TARGET = "TransactionAmount"

# Identificadores de alta cardinalidad sin valor predictivo generalizable.
ID_COLUMNS = ["TransactionID", "AccountID", "DeviceID", "IP Address", "MerchantID"]

DATE_COLUMN = "TransactionDate"

DERIVED_DATE_COLUMNS = [
    "trans_year",
    "trans_month",
    "trans_dow",
    "trans_hour",
    "es_fin_de_semana",
]

CATEGORICAL_COLUMNS = ["TransactionType", "Location", "Channel", "CustomerOccupation"]

NUMERIC_COLUMNS = [
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
] + DERIVED_DATE_COLUMNS

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_raw(csv_path: str) -> pd.DataFrame:
    """Carga el CSV crudo del dataset."""
    return pd.read_csv(csv_path)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parsea `TransactionDate` (formatos mixtos) y deriva variables temporales."""
    df = df.copy()
    fechas = pd.to_datetime(df[DATE_COLUMN], format="mixed")
    df["trans_year"] = fechas.dt.year
    df["trans_month"] = fechas.dt.month
    df["trans_dow"] = fechas.dt.dayofweek  # 0=lunes, 6=domingo
    df["trans_hour"] = fechas.dt.hour
    df["es_fin_de_semana"] = (df["trans_dow"] >= 5).astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica derivación de fechas y descarta identificadores y la fecha original."""
    df = add_date_features(df)
    to_drop = [c for c in ID_COLUMNS + [DATE_COLUMN] if c in df.columns]
    return df.drop(columns=to_drop)


def split_features_target(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (en el orden canónico de FEATURE_COLUMNS) e y."""
    y = df_clean[TARGET]
    X = df_clean[FEATURE_COLUMNS]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer idéntico al del notebook."""
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("cat", cat_pipe, CATEGORICAL_COLUMNS),
            ("num", num_pipe, NUMERIC_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )


def load_dataset(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Carga, limpia y devuelve (X, y) listos para entrenar."""
    df = clean(load_raw(csv_path))
    return split_features_target(df)
