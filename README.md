# Regresión de transacciones bancarias — MLOps

Predicción de `TransactionAmount` sobre 50.000 transacciones bancarias simuladas,
productivizada a partir del notebook
[`proyecto_kaggle_transacciones_regresion_v2.ipynb`](notebooks/proyecto_kaggle_transacciones_regresion_v2.ipynb):
entrenamiento reproducible, tracking con MLflow, gate de calidad, API FastAPI,
Docker y CI/CD con GitHub Actions.

> ⚠️ **Hallazgo de fuga de datos (documentado, no oculto):** Random Forest
> alcanza R² ≈ 0.994 porque `AccountBalance` (junto con `Location`,
> `TransactionDuration` y `CustomerAge`) determina casi por completo el monto —
> una relación no realista, probablemente inyectada al aumentar el dataset
> sintético. Por eso el pipeline marca automáticamente cualquier modelo con
> R² > 0.90 como `requiere_revision_manual=true` y **no** lo promueve.

## Instalación

Requiere Python 3.11 o 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenamiento

```bash
python -m src.train --data data/bank_transactions_data_2_augmented_clean_2.csv
```

Entrena los 6 modelos (Baseline, Lineal, Ridge, Lasso, Random Forest,
Gradient Boosting) con validación cruzada KFold, loguea hiperparámetros,
métricas (RMSE/MAE/R²) e importancias de variables (impureza y permutación)
en MLflow, aplica el gate de calidad y guarda el pipeline completo
(preprocesamiento + mejor modelo elegible) en `models/model.joblib`.

Opciones útiles:

- `--sample-frac 0.1` — modo validación rápido (el que usa CI).
- `--no-register` — no registrar el candidato en el Model Registry.
- `--r2-alert 0.90` / `--rmse-threshold 0.05` — umbrales del gate.

Para explorar las corridas: `mlflow ui` y abrir <http://localhost:5000>.

### Gate de calidad

1. **R² > 0.90 ⇒ sospecha de fuga de datos.** El modelo se etiqueta
   `requiere_revision_manual=true` en MLflow y queda excluido de la
   promoción automática.
2. **No-regresión de RMSE.** El candidato no puede empeorar más del 5%
   (configurable) frente al modelo registrado como `stage="Production"`.
   Si ningún modelo pasa el gate, `train.py` termina con exit code 1
   (y el workflow de CI falla).

## Tests

```bash
python -m pytest
```

- `tests/test_preprocess.py` — categorías no vistas (`handle_unknown="ignore"`),
  ausencia de nulos tras transformar y forma de salida esperada.
- `tests/test_no_regression.py` — el RMSE del candidato no empeora más del
  umbral (`RMSE_REGRESSION_THRESHOLD`, default 5%) frente al modelo vigente
  (`REFERENCE_MODEL_PATH`) o, en su defecto, frente al baseline de la media.

## API

```bash
uvicorn src.api.main:app --reload
```

Variables de entorno: `MODEL_PATH` (default `models/model.joblib`),
`MODEL_VERSION`, `PREDICTIONS_LOG_PATH` (default `logs/predictions.jsonl`,
JSON-lines con input/output/versión/timestamp de cada predicción).

Ejemplo de request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_type": "Debit",
    "channel": "ATM",
    "location": "Houston",
    "customer_occupation": "Doctor",
    "customer_age": 68,
    "transaction_duration": 141,
    "login_attempts": 1,
    "account_balance": 13758.91,
    "transaction_date": "2023-06-27T16:44:00"
  }'
```

Respuesta:

```json
{"predicted_amount": 263.79, "model_version": "1", "prediction_timestamp": "..."}
```

`GET /health` devuelve `{"status": "ok", "model_version": "..."}`.

## Docker

```bash
python -m src.train          # el modelo debe existir antes del build
docker build -t bank-transaction-api .
docker run -p 8000:8000 bank-transaction-api
```

## CI/CD

- [`ci.yml`](.github/workflows/ci.yml) — en cada push/PR: instala dependencias,
  entrena en modo validación aplicando el gate de calidad (falla el workflow
  si no pasa) y corre `pytest` incluyendo el test de no-regresión.
- [`cd.yml`](.github/workflows/cd.yml) — si CI pasa en `main`: entrena el
  modelo, construye la imagen Docker, la publica en GitHub Container Registry
  y deja plantillas de despliegue (Render / Fly.io / Cloud Run) activables con
  secretos del repositorio — sin credenciales hardcodeadas.

## Estructura

```
├── notebooks/            # notebook exploratorio original
├── data/                 # dataset (50k transacciones simuladas)
├── src/
│   ├── preprocess.py     # limpieza + ColumnTransformer
│   ├── train.py          # 6 modelos + MLflow + gate de calidad
│   ├── evaluate.py       # métricas sobre el set de test
│   └── api/              # FastAPI (main.py, schemas.py)
├── tests/                # preprocesamiento y no-regresión
├── .github/workflows/    # ci.yml, cd.yml
├── Dockerfile
└── requirements.txt
```
