# eduhamuy-ai

Backend y notebooks para el MVP de búsqueda documental de EduHamuy.

El notebook explora la extracción de texto desde PDFs, la construcción de un
índice TF-IDF y la carga de artefactos desde Azure Blob Storage. El backend
FastAPI se encuentra inicialmente preparado con el endpoint `/health`.

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

## Pruebas

```bash
pytest
```

En Colab se usan Secrets para `AZURE_STORAGE_ACCOUNT` y `AZURE_STORAGE_SAS`.
