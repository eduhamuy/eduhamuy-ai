import pandas as pd
from fastapi.testclient import TestClient
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import app.main as main_module
from app.artifacts import ArtifactBundle
from app.search import preprocess_text


class FakeEmbeddingModel:
    def encode(self, queries, normalize_embeddings=True):
        assert normalize_embeddings is True
        return [[1.0, 0.0] for _ in queries]


def test_search_returns_matching_documents(monkeypatch) -> None:
    documents = pd.DataFrame(
        {
            "document_title": ["Educación superior", "Salud comunitaria"],
            "document_path": ["education.pdf", "health.pdf"],
        }
    )
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(
        [preprocess_text("educación superior"), preprocess_text("salud comunitaria")]
    )
    bundle = ArtifactBundle(vectorizer, sparse.csr_matrix(matrix), documents)
    monkeypatch.setattr(main_module, "load_artifacts", lambda: bundle)

    response = TestClient(main_module.app).get(
        "/search", params={"q": "educación", "limit": 3}
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["document_title"] == "Educación superior"


def test_search_returns_service_unavailable_without_artifacts(monkeypatch) -> None:
    def fail_to_load():
        from app.artifacts import ArtifactConfigurationError

        raise ArtifactConfigurationError("missing configuration")

    monkeypatch.setattr(main_module, "load_artifacts", fail_to_load)

    response = TestClient(main_module.app).get("/search", params={"q": "educación"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Search index is temporarily unavailable"}


def test_hybrid_search_combines_tfidf_and_embeddings() -> None:
    documents = pd.DataFrame({
        "document_title": ["Educación superior", "Salud comunitaria"],
        "document_path": ["education.pdf", "health.pdf"],
    })
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(["educación superior", "salud comunitaria"])
    bundle = ArtifactBundle(
        vectorizer, sparse.csr_matrix(matrix), documents,
        method="hybrid_tfidf_embeddings", alpha=0.3,
        document_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        embedding_model=FakeEmbeddingModel(),
    )

    from app.search import search_documents

    assert search_documents("educación", bundle, 1)[0]["document_path"] == "education.pdf"
