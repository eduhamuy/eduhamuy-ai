"""Load versioned search artifacts from Azure Blob Storage."""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import PurePosixPath
from typing import Any

import joblib
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from scipy import sparse


class ArtifactError(RuntimeError):
    pass


class ArtifactConfigurationError(ArtifactError):
    pass


class ArtifactValidationError(ArtifactError):
    pass


@dataclass(frozen=True)
class ArtifactBundle:
    vectorizer: Any
    matrix: sparse.spmatrix
    documents: pd.DataFrame
    method: str = "tfidf"
    alpha: float = 1.0
    document_embeddings: np.ndarray | None = None
    embedding_model: Any | None = None


def _required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ArtifactConfigurationError(f"Required setting {name} is not configured")
    return value


def _storage_container(account: str, credential: str, container_name: str) -> Any:
    return BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net", credential=credential
    ).get_container_client(container_name)


def _blob_name(prefix: str, relative_name: str) -> str:
    path = PurePosixPath(relative_name)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactValidationError(f"Invalid artifact path: {relative_name}")
    return f"{prefix}/{path.as_posix()}" if prefix else path.as_posix()


def _download(container: Any, blob_name: str) -> bytes:
    try:
        return container.get_blob_client(blob_name).download_blob().readall()
    except Exception as exc:
        raise ArtifactError(f"Unable to download artifact {blob_name}") from exc


def _download_verified(container: Any, prefix: str, name: str, files: dict[str, Any]) -> bytes:
    expected = files.get(name, {}).get("sha256")
    if not isinstance(expected, str):
        raise ArtifactValidationError(f"Artifact is absent from artifact_manifest.json: {name}")
    data = _download(container, _blob_name(prefix, name))
    if sha256(data).hexdigest() != expected:
        raise ArtifactValidationError(f"SHA-256 mismatch for artifact: {name}")
    return data


def _parse_tfidf(vectorizer_bytes: bytes, matrix_bytes: bytes, documents_bytes: bytes) -> ArtifactBundle:
    try:
        vectorizer = joblib.load(BytesIO(vectorizer_bytes))
        matrix = sparse.load_npz(BytesIO(matrix_bytes))
        documents = pd.read_csv(BytesIO(documents_bytes))
    except Exception as exc:
        raise ArtifactValidationError("Downloaded TF-IDF artifacts could not be read") from exc
    required = {"document_path", "document_title"}
    if not required.issubset(documents.columns):
        raise ArtifactValidationError("Document metadata is incomplete")
    if matrix.shape[0] != len(documents) or not hasattr(vectorizer, "transform"):
        raise ArtifactValidationError("TF-IDF artifacts are incompatible")
    return ArtifactBundle(vectorizer, matrix, documents)


def _load_embedding_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        raise ArtifactValidationError(f"Unable to load embedding model {model_name!r}") from exc


@lru_cache(maxsize=1)
def load_artifacts() -> ArtifactBundle:
    """Load the selected version and validate every artifact consumed by it."""

    account = _required_setting("AZURE_STORAGE_ACCOUNT")
    sas = _required_setting("AZURE_STORAGE_SAS").lstrip("?")
    container_name = os.getenv("AZURE_ARTIFACT_CONTAINER", "ai-artifacts").strip()
    prefix = os.getenv("ARTIFACT_PREFIX", "").strip().strip("/")
    container = _storage_container(account, sas, container_name)

    # Compatibility mode while the deployed TF-IDF layout is still in service.
    if not prefix:
        return _parse_tfidf(
            _download(container, "tfidf_vectorizer.joblib"),
            _download(container, "X_tfidf.npz"),
            _download(container, "processed_documents.csv"),
        )

    try:
        manifest = json.loads(_download(container, _blob_name(prefix, "artifact_manifest.json")))
        files = manifest["files"]
        contract = json.loads(_download_verified(container, prefix, "deployment_contract.json", files))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Invalid versioned artifact manifest or contract") from exc

    required = {
        "tfidf/tfidf_vectorizer.joblib", "tfidf/X_tfidf.npz", "tfidf/document_metadata.csv",
        "embeddings/document_embeddings.npy", "embeddings/embedding_config.json",
        "hybrid_tfidf_embeddings/hybrid_config.json",
    }
    if contract.get("backend_method") != "hybrid_tfidf_embeddings" or set(contract.get("artifacts", [])) != required:
        raise ArtifactValidationError("Unexpected hybrid deployment contract")

    base = _parse_tfidf(
        _download_verified(container, prefix, "tfidf/tfidf_vectorizer.joblib", files),
        _download_verified(container, prefix, "tfidf/X_tfidf.npz", files),
        _download_verified(container, prefix, "tfidf/document_metadata.csv", files),
    )
    try:
        embeddings = np.load(BytesIO(_download_verified(container, prefix, "embeddings/document_embeddings.npy", files)), allow_pickle=False)
        embedding_config = json.loads(_download_verified(container, prefix, "embeddings/embedding_config.json", files))
        hybrid_config = json.loads(_download_verified(container, prefix, "hybrid_tfidf_embeddings/hybrid_config.json", files))
        alpha = float(hybrid_config["alpha"])
        model_name = embedding_config["model_name"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Invalid hybrid configuration") from exc
    if embeddings.ndim != 2 or embeddings.shape[0] != len(base.documents) or not 0 <= alpha <= 1:
        raise ArtifactValidationError("Hybrid artifacts are incompatible")
    if hybrid_config.get("normalization") != "minmax":
        raise ArtifactValidationError("Unsupported hybrid normalization")
    return ArtifactBundle(base.vectorizer, base.matrix, base.documents, "hybrid_tfidf_embeddings", alpha, embeddings, _load_embedding_model(model_name))


def source_container() -> Any:
    """Return the private source container used only for PDF previewing.

    A separate read-only SAS is preferred. Falling back to the existing
    storage SAS keeps local development compatible while deployments migrate
    to a source-container-scoped credential.
    """

    account = _required_setting("AZURE_STORAGE_ACCOUNT")
    sas = os.getenv("AZURE_SOURCE_SAS", "").strip().lstrip("?")
    if not sas:
        sas = _required_setting("AZURE_STORAGE_SAS").lstrip("?")
    container_name = os.getenv("AZURE_SOURCE_CONTAINER", "").strip()
    if not container_name:
        raise ArtifactConfigurationError("Required setting AZURE_SOURCE_CONTAINER is not configured")
    return _storage_container(account, sas, container_name)
