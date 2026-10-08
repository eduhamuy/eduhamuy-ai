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

## Construcción reproducible de índices

El notebook permanece como entorno de exploración. La construcción aprobable se
ejecuta con `indexer/build_index.py` y nunca sobrescribe un prefijo existente:

```text
builds/hybrid-tfidf-embeddings/<BUILD_VERSION>/   # salida evaluada
indexes/hybrid-tfidf-embeddings/<BUILD_VERSION>/ # misma salida, aprobada
```

El comando `build` descarga los PDFs y la suite desde Azure, normaliza rutas
Unicode, comprueba los documentos protegidos por la suite, ajusta `alpha` solo
con consultas `dev`, evalúa sobre `test`, y publica manifiestos SHA-256 junto
con el contrato de despliegue. El comando `promote` verifica cada hash y copia
los mismos bytes de `builds/` a `indexes/`; no recalcula el índice.

Para una ejecución local, instalar las dependencias del indexador y usar Azure
CLI (`az login`) o un SAS local:

```bash
pip install -r requirements-indexer.txt
python -m indexer.build_index build \
  --target-env dev \
  --corpus-version 2026-10-07-v1 \
  --suite-version 2026-10-07-v1 \
  --build-version dev-2026-10-07-v1-local \
  --account-name "$AZURE_STORAGE_ACCOUNT" \
  --output-dir /tmp/eduhamuy-index
```

Agregar `--publish` publica en:

```text
ai-artifacts-dev/builds/hybrid-tfidf-embeddings/<BUILD_VERSION>/
```

## Workflows y Azure OIDC

Los workflows manuales están separados para evitar publicar o desplegar una
construcción sin revisión:

1. **Build AI Index** crea y evalúa `builds/.../<BUILD_VERSION>`.
2. **Promote AI Index** verifica hashes y crea `indexes/.../<BUILD_VERSION>`.
3. **Deploy AI Index** verifica el índice aprobado y abre un PR de GitOps que
   cambia `ARTIFACT_PREFIX`.

Actualmente las opciones se limitan a `dev` porque solo existe el overlay AI
para DEV. TEST y PROD se habilitarán cuando cuenten con sus contenedores,
secretos y overlays de `eduhamuy-ai`.

Antes de ejecutar los workflows, configurar un App Registration de Microsoft
Entra con una credencial federada para el entorno GitHub `dev`:

```text
issuer:  https://token.actions.githubusercontent.com
subject: repo:eduhamuy/eduhamuy-ai:environment:dev
audience: api://AzureADTokenExchange
```

Asignar al principal el rol **Storage Blob Data Contributor** en la cuenta de
almacenamiento (o en los tres contenedores DEV). En el Environment `dev` de
GitHub configurar:

```text
Secrets: AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID, GITOPS_TOKEN
Variable: AZURE_STORAGE_ACCOUNT=steduhamuyshared
```

`GITOPS_TOKEN` requiere acceso de escritura de contenidos y Pull Requests en
`eduhamuy/eduhamuy-gitops`. No se usan SAS ni connection strings en GitHub
Actions; el acceso a Blob Storage usa OIDC.
