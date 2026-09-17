import pytest

from app.artifacts import ArtifactConfigurationError, load_artifacts


def test_artifacts_require_azure_configuration(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_SAS", raising=False)
    load_artifacts.cache_clear()

    with pytest.raises(ArtifactConfigurationError):
        load_artifacts()

    load_artifacts.cache_clear()
