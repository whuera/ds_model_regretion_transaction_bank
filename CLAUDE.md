# Instrucciones para Claude Code — Productivizar el modelo de regresión de transacciones bancarias

Este documento es el brief de proyecto para **Claude Code** (la CLI de Anthropic). Colócalo como `CLAUDE.md` en la raíz de tu repositorio para que Claude lo lea automáticamente al iniciar sesión en la carpeta, o pégalo como primer mensaje si prefieres no crear el repo todavía.

---

## 1. Contexto

Este proyecto parte de un notebook de Google Colab (`proyecto_kaggle_transacciones_regresion_v2.ipynb`) que:

- Analiza 50.000 transacciones bancarias simuladas (`bank_transactions_data_2_augmented_clean_2.csv`).
- Entrena y compara 6 modelos de regresión (Baseline, Regresión Lineal, Ridge, Lasso, Random Forest, Gradient Boosting) para predecir `TransactionAmount`.
- Usa un `Pipeline` + `ColumnTransformer` de scikit-learn para el preprocesamiento (imputación, escalado, One-Hot Encoding).
- Detecta y documenta un caso de **fuga de datos**: Random Forest alcanza R² ≈ 0.994 porque `AccountBalance` (junto con `Location`, `TransactionDuration` y `CustomerAge`) determina casi por completo el monto — una relación no realista en datos bancarios reales, probablemente inyectada al "aumentar" el dataset sintético.

El objetivo de esta fase es **convertir ese notebook exploratorio en un proyecto de Python productivo**, con un pipeline de MLOps automatizado: entrenamiento reproducible, tracking de experimentos, registro de modelos, una API de servicio, contenerización, CI/CD y monitoreo básico.

**Importante — no ocultar el hallazgo de fuga de datos.** El código productivo debe mantener la validación que detecta un R² sospechosamente alto (ver sección 5, gate de calidad) en lugar de simplemente desplegar el modelo con mejor métrica sin más.

---

## 2. Material de partida (colócalo en el repo antes de empezar)

```
notebooks/proyecto_kaggle_transacciones_regresion_v2.ipynb
data/bank_transactions_data_2_augmented_clean_2.csv
```

Si no tienes el notebook a mano, Claude puede reconstruir la lógica de preprocesamiento y modelado a partir de la sección 3 de este documento — pero es preferible partir del notebook real para no perder detalles de limpieza (parseo de fechas con `format="mixed"`, columnas descartadas por alta cardinalidad, etc.).

---

## 3. Estructura de repositorio a crear

```
proyecto/
├── notebooks/
│   └── proyecto_kaggle_transacciones_regresion_v2.ipynb
├── data/
│   └── bank_transactions_data_2_augmented_clean_2.csv
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # ColumnTransformer (numéricas + categóricas)
│   ├── train.py              # entrena los 6 modelos, aplica el gate, registra en MLflow
│   ├── evaluate.py           # calcula RMSE/MAE/R² sobre el set de test
│   └── api/
│       ├── __init__.py
│       ├── main.py           # app FastAPI
│       └── schemas.py        # TransactionInput / PredictionOutput (Pydantic)
├── tests/
│   ├── test_preprocess.py    # columnas esperadas, categorías nuevas, nulos
│   └── test_no_regression.py # RMSE nuevo no debe empeorar vs. el modelo vigente
├── .github/
│   └── workflows/
│       ├── ci.yml            # test + train + gate de calidad
│       └── cd.yml            # build de imagen + deploy
├── Dockerfile
├── requirements.txt
├── mlruns/                   # (o configuración de MLflow remoto)
├── README.md
└── CLAUDE.md                 # este archivo
```

---

## 4. Tareas a ejecutar, en orden

Pide a Claude Code que las resuelva **una por una**, verificando cada una antes de pasar a la siguiente (correr tests, revisar output) — no todo de una sola vez.

1. **Inicializar el repositorio.** `git init`, crear la estructura de carpetas de la sección 3, `requirements.txt` con versiones fijadas (`scikit-learn`, `pandas`, `numpy`, `fastapi`, `uvicorn`, `pydantic`, `mlflow`, `pytest`, `joblib`).

2. **Extraer `src/preprocess.py`.** Traducir la limpieza y el `ColumnTransformer` del notebook: parseo de fechas mixtas, variables derivadas (`trans_year`, `trans_month`, `trans_dow`, `trans_hour`, `es_fin_de_semana`), columnas descartadas (`TransactionID`, `AccountID`, `DeviceID`, `IP Address`, `MerchantID`), imputación + escalado numérico, imputación + One-Hot categórico.

3. **Extraer `src/train.py`.** Entrena los 6 modelos con `KFold` de validación cruzada, calcula RMSE/MAE/R² con un scorer personalizado (igual que en el notebook), y guarda el pipeline completo (preprocesamiento + mejor modelo) con `joblib`.

4. **Instrumentar con MLflow.** Cada corrida de `train.py` debe loguear: hiperparámetros, métricas por modelo, y el artefacto serializado. Usa `mlflow.sklearn.log_model`.

5. **Implementar el gate de calidad** (ver sección 5) dentro de `train.py` o como paso separado antes de registrar el modelo como candidato a producción.

6. **Escribir `tests/test_preprocess.py`.** Verifica que el `ColumnTransformer` maneja una categoría de `Location` no vista en entrenamiento (gracias a `handle_unknown="ignore"`), que no quedan nulos tras la transformación, y que la forma de salida es la esperada.

7. **Escribir `tests/test_no_regression.py`.** Carga el modelo vigente (o un baseline de referencia) y el modelo recién entrenado; falla si el RMSE en el set de test empeora más de un umbral configurable (ej. 5%).

8. **Construir `src/api/schemas.py` y `src/api/main.py`** siguiendo el contrato de la sección 6. La API carga el `.joblib` registrado al iniciar (`@app.on_event("startup")` o `lifespan`), expone `POST /predict` y `GET /health`.

9. **Escribir el `Dockerfile`.** Imagen base `python:3.11-slim`, copia `requirements.txt` e instala con `--no-cache-dir`, copia `src/`, expone el puerto de `uvicorn`, `CMD` arranca la API.

10. **Escribir `.github/workflows/ci.yml`.** En cada push/PR: instala dependencias, corre `pytest`, corre `train.py` en modo validación, aplica el gate de calidad (falla el workflow si no pasa).

11. **Escribir `.github/workflows/cd.yml`.** Si `ci.yml` pasa en la rama `main`: construye la imagen Docker, la publica en un container registry (GitHub Container Registry), y dispara el despliegue (Render/Fly.io/Cloud Run — dejar el paso de deploy como plantilla con variables de entorno para las credenciales, sin hardcodear secretos).

12. **Añadir logging de predicciones** en `main.py`: cada llamada a `/predict` registra input, output, versión del modelo (variable de entorno o metadata del artefacto) y timestamp — a un archivo JSON-lines local por defecto, con un comentario indicando dónde conectar una base de datos real.

13. **Escribir `README.md`** con instrucciones de instalación, cómo correr los tests, cómo levantar la API localmente (`uvicorn src.api.main:app --reload`) y un ejemplo de request con `curl`.

14. **Ejecutar una verificación final end-to-end**: build de la imagen Docker, levantar el contenedor, hacer un `curl` real a `/predict` con un payload de ejemplo, y confirmar que la respuesta cumple el esquema de la sección 6.

---

## 5. Gate de calidad (no negociable)

Antes de registrar cualquier modelo como candidato a producción, `train.py` debe aplicar estas reglas:

- Si el R² del modelo candidato supera **0.90**, el pipeline debe marcarlo automáticamente como `requiere_revision_manual = true` en el log de MLflow y **no** promoverlo solo, en lugar de publicarlo automáticamente. Esto replica el hallazgo de fuga de datos documentado en el notebook: un resultado sospechosamente bueno se investiga, no se celebra.
- El RMSE del modelo candidato no puede empeorar más de un 5% respecto al modelo actualmente en producción (comparación contra el artefacto registrado como `stage="Production"` en MLflow).
- Registrar siempre, junto a las métricas, la importancia de variables (impureza y `permutation_importance`) para poder auditar después qué está impulsando cada predicción.

---

## 6. Contrato de la API (`POST /predict`)

```python
# src/api/schemas.py
from pydantic import BaseModel
from typing import Literal
from datetime import datetime

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
    predicted_amount: float
    model_version: str
    prediction_timestamp: datetime
```

`GET /health` debe devolver `{"status": "ok", "model_version": "..."}` para que el orquestador de despliegue pueda verificar que el contenedor está listo antes de recibir tráfico.

---

## 7. Definición de terminado

- [ ] `pytest` pasa en local y en CI, sin excepciones.
- [ ] `docker build` genera la imagen sin errores y el contenedor arranca.
- [ ] `curl -X POST /predict` con un payload de ejemplo devuelve un JSON válido según el esquema de la sección 6.
- [ ] El workflow de CI falla intencionalmente si se degrada el RMSE más del umbral (pruébalo bajando el umbral a un valor irreal para confirmar que el gate funciona).
- [ ] MLflow tiene al menos una corrida registrada con métricas y artefacto.
- [ ] `README.md` permite a otra persona levantar el proyecto sin más contexto que ese archivo.

---