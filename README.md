# eduhamuy-ai

Backend y notebooks para el MVP de búsqueda documental de EduHamuy.

El notebook explora la extracción de texto desde PDFs, la construcción de
índices y la evaluación de búsquedas híbridas. El backend FastAPI expone los
endpoints `/health` y `/search`.

## Estructura

```text
app/          API y lógica de búsqueda
indexer/      Proceso batch para construir el índice
notebooks/    Exploraciones y experimentos reproducibles
tests/        Pruebas automatizadas
```

Los PDFs y artefactos generados permanecen en Azure Blob Storage. 
No se subiran secretos, archivos `.env`, PDFs ni archivos `.joblib` o `.npz`.

## Ejecución local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

El servicio queda disponible en `http://localhost:8000`.

La documentación interactiva está disponible en `http://localhost:8000/docs`.
Una búsqueda se realiza con:

```bash
curl 'http://localhost:8000/search?q=educacion%20superior&limit=5'
```

El endpoint devuelve `503` hasta que las variables de Azure estén configuradas.
Durante la migración, sin `ARTIFACT_PREFIX`, el backend conserva compatibilidad
con los tres artefactos TF-IDF planos en `ai-artifacts`:

```text
tfidf_vectorizer.joblib
X_tfidf.npz
processed_documents.csv
```

Con `ARTIFACT_PREFIX`, el backend carga una versión híbrida publicada, verifica
sus hashes desde `artifact_manifest.json` y lee el contrato de despliegue. La
validación actual de DEV usa:

```text
AZURE_ARTIFACT_CONTAINER=ai-artifacts-dev
ARTIFACT_PREFIX=experiments/2026-10-07-v1
```

## Pruebas

```bash
python -m pytest -q
```

`requirements.txt` contiene solo el runtime del servicio y fija las versiones
compatibles con los artefactos híbridos publicados. `requirements-dev.txt`
agrega las herramientas de prueba; las dependencias de exploración de los
notebooks no forman parte de la imagen del backend.

En Colab se usan Secrets para `AZURE_STORAGE_ACCOUNT` y `AZURE_STORAGE_SAS`.
