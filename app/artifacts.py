"""Loading of TF-IDF artifacts from Azure Blob Storage.

The MVP stores generated artifacts in the private ``ai-artifacts`` container.
Authentication values must be supplied through environment variables or a
secret manager, never committed to this repository.
"""

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import os
from typing import Any

import joblib
import pandas as pd
from azure.storage.blob import BlobServiceClient
from scipy import sparse


class ArtifactError(RuntimeError):
    """Base error for artifact configuration, access, or validation failures."""


class ArtifactConfigurationError(ArtifactError):
    """Raised when required Azure configuration is missing."""


class ArtifactValidationError(ArtifactError):
    """Raised when downloaded artifacts do not describe the same index."""


@dataclass(frozen=True)
class ArtifactBundle:
    """The fitted vectorizer, sparse matrix, and document metadata."""

    vectorizer: Any
    matrix: sparse.spmatrix
    documents: pd.DataFrame


def _required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ArtifactConfigurationError(f"Required setting {name} is not configured")
    return value


def _download(container: Any, blob_name: str) -> bytes:
    try:
        return container.get_blob_client(blob_name).download_blob().readall()
    except Exception as exc:  # Azure SDK exceptions vary by transport/version.
        raise ArtifactError(f"Unable to download artifact {blob_name}") from exc


@lru_cache(maxsize=1)
def load_artifacts() -> ArtifactBundle:
    """Download and validate the current TF-IDF artifacts.

    The result is cached for the lifetime of the process because the backend
    should not download the same files for every search request. Restart the
    pod (or call ``load_artifacts.cache_clear()``) after publishing a new index.
    """

    account = _required_setting("AZURE_STORAGE_ACCOUNT")
    sas = _required_setting("AZURE_STORAGE_SAS").lstrip("?")
    container_name = os.getenv("AZURE_ARTIFACT_CONTAINER", "ai-artifacts").strip()

    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=sas,
    )
    container = service.get_container_client(container_name)

    vectorizer_bytes = _download(container, "tfidf_vectorizer.joblib")
    matrix_bytes = _download(container, "X_tfidf.npz")
    documents_bytes = _download(container, "processed_documents.csv")

    try:
        vectorizer = joblib.load(BytesIO(vectorizer_bytes))
        matrix = sparse.load_npz(BytesIO(matrix_bytes))
        documents = pd.read_csv(BytesIO(documents_bytes))
    except Exception as exc:
        raise ArtifactValidationError("Downloaded artifacts could not be read") from exc

    required_columns = {"document_path", "document_title"}
    if not required_columns.issubset(documents.columns):
        missing = ", ".join(sorted(required_columns - set(documents.columns)))
        raise ArtifactValidationError(f"Document metadata is missing: {missing}")

    if matrix.shape[0] != len(documents):
        raise ArtifactValidationError(
            "TF-IDF matrix rows do not match document metadata rows"
        )

    if not hasattr(vectorizer, "transform"):
        raise ArtifactValidationError("TF-IDF vectorizer does not support transform()")

    return ArtifactBundle(vectorizer, matrix, documents)
