# eduhamuy-ai

Backend y notebooks para el MVP de búsqueda documental de EduHamuy.

El notebook explora la extracción de texto desde PDFs, la construcción de un
índice TF-IDF y la carga de artefactos desde Azure Blob Storage. El backend
FastAPI expone los endpoints `/health` y `/search`.

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
pip install -r requirements.txt
uvicorn app.main:app --reload
```

El servicio queda disponible en `http://localhost:8000`.

La documentación interactiva está disponible en `http://localhost:8000/docs`.
Una búsqueda se realiza con:

```bash
curl 'http://localhost:8000/search?q=educacion%20superior&limit=5'
```

El endpoint devuelve `503` hasta que las variables de Azure estén configuradas
y los tres artefactos estén publicados en el contenedor `ai-artifacts`:

```text
tfidf_vectorizer.joblib
X_tfidf.npz
processed_documents.csv
```

## Pruebas

```bash
python -m pytest -q
```

En Colab se usan Secrets para `AZURE_STORAGE_ACCOUNT` y `AZURE_STORAGE_SAS`.
