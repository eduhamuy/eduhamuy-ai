"""HTTP entry point for the EduHamuy AI service."""

from fastapi import FastAPI, HTTPException, Query

from app.artifacts import ArtifactError, load_artifacts
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
