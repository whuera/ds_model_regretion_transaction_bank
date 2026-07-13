"""Tests del preprocesamiento: categorías nuevas, nulos y forma de salida."""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_preprocessor,
    clean,
    load_raw,
)

DATA_PATH = "data/bank_transactions_data_2_augmented_clean_2.csv"


@pytest.fixture(scope="module")
def sample_X() -> pd.DataFrame:
    df = clean(load_raw(DATA_PATH).head(500))
    return df[FEATURE_COLUMNS]


def test_clean_deriva_fechas_y_elimina_ids(sample_X):
    for col in ("trans_year", "trans_month", "trans_dow", "trans_hour", "es_fin_de_semana"):
        assert col in sample_X.columns
    for col in ("TransactionID", "AccountID", "DeviceID", "IP Address",
                "MerchantID", "TransactionDate"):
        assert col not in sample_X.columns
    assert set(sample_X["es_fin_de_semana"].unique()) <= {0, 1}


def test_categoria_no_vista_no_falla(sample_X):
    """handle_unknown='ignore': una Location nueva produce fila one-hot en cero."""
    pre = build_preprocessor()
    pre.fit(sample_X)

    fila_nueva = sample_X.iloc[[0]].copy()
    fila_nueva["Location"] = "Ciudad Inexistente"
    salida = pre.transform(fila_nueva)

    assert salida.shape[0] == 1
    assert not np.isnan(salida).any()

    # todas las columnas one-hot de Location deben quedar en 0
    nombres = pre.get_feature_names_out()
    idx_location = [i for i, n in enumerate(nombres) if n.startswith("Location_")]
    assert len(idx_location) > 0
    assert np.all(salida[0, idx_location] == 0)


def test_sin_nulos_tras_transformar(sample_X):
    """La imputación elimina los nulos tanto numéricos como categóricos."""
    X = sample_X.copy()
    X.loc[X.index[:5], "AccountBalance"] = np.nan
    X.loc[X.index[5:10], "Location"] = None

    pre = build_preprocessor()
    salida = pre.fit_transform(X)
    assert not np.isnan(salida).any()


def test_forma_de_salida_esperada(sample_X):
    """Salida = una columna por categoría (one-hot) + las numéricas."""
    pre = build_preprocessor()
    salida = pre.fit_transform(sample_X)

    n_onehot = sum(sample_X[c].nunique() for c in CATEGORICAL_COLUMNS)
    esperado = n_onehot + len(NUMERIC_COLUMNS)

    assert salida.shape == (len(sample_X), esperado)
    assert len(pre.get_feature_names_out()) == esperado
