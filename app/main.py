"""HTTP entry point for the EduHamuy AI service."""

from fastapi import FastAPI

app = FastAPI(
    title="EduHamuy AI",
    description="Backend de búsqueda documental para EduHamuy.",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""

    return {"status": "ok"}
