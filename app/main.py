"""HTTP entry point for the EduHamuy AI service."""

from pathlib import PurePosixPath
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.artifacts import ArtifactError, load_artifacts, source_container
from app.search import search_documents

app = FastAPI(
    title="EduHamuy AI",
    description="Backend de búsqueda documental para EduHamuy.",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""

    return {"status": "ok"}


@app.get("/search", tags=["search"])
def search(
    q: str = Query(..., min_length=1, description="Texto que se desea buscar"),
    limit: int = Query(5, ge=1, le=50, description="Número máximo de resultados"),
) -> dict[str, object]:
    """Search the current TF-IDF index."""

    try:
        artifacts = load_artifacts()
    except ArtifactError as exc:
        raise HTTPException(
            status_code=503,
            detail="Search index is temporarily unavailable",
        ) from exc

    return {"query": q, "results": search_documents(q, artifacts, limit)}


def _artifacts_or_503():
    try:
        return load_artifacts()
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail="Search index is temporarily unavailable") from exc


def _catalog_documents(artifacts):
    documents = artifacts.documents
    required = {"document_id", "document_title"}
    if not required.issubset(documents.columns):
        raise HTTPException(
            status_code=503,
            detail="The active index does not support the document catalog yet",
        )
    return documents.sort_values(
        "document_title", key=lambda titles: titles.astype(str).str.casefold(), kind="stable"
    )


@app.get("/documents", tags=["documents"])
def documents(
    limit: int = Query(20, ge=1, le=50, description="Número de documentos por página"),
    offset: int = Query(0, ge=0, description="Posición inicial dentro del catálogo"),
) -> dict[str, object]:
    """Return a stable, public page of the PDFs included in the active index."""

    catalog = _catalog_documents(_artifacts_or_503())
    page = catalog.iloc[offset : offset + limit]
    return {
        "total": int(len(catalog)),
        "offset": offset,
        "limit": limit,
        "documents": [
            {"document_id": str(row.document_id), "document_title": str(row.document_title)}
            for row in page.itertuples(index=False)
        ],
    }


@app.get("/documents/{document_id}/preview", tags=["documents"])
def preview_document(document_id: str):
    """Stream one indexed PDF without exposing storage credentials or paths."""

    artifacts = _artifacts_or_503()
    catalog = _catalog_documents(artifacts)
    matching = catalog[catalog["document_id"].astype(str) == document_id]
    if matching.empty:
        raise HTTPException(status_code=404, detail="Document not found")
    if "source_blob_path" not in matching.columns:
        raise HTTPException(status_code=503, detail="The active index does not support PDF preview yet")

    blob_path = str(matching.iloc[0]["source_blob_path"])
    path = PurePosixPath(blob_path)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=503, detail="Document preview metadata is invalid")
    try:
        downloader = source_container().get_blob_client(blob_path).download_blob()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document preview is temporarily unavailable") from exc

    filename = quote(path.name)
    return StreamingResponse(
        downloader.chunks(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{filename}",
            "Cache-Control": "public, max-age=3600",
        },
    )
