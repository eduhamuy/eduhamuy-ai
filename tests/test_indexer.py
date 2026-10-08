import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from indexer.build_index import (
    BuildConfig,
    REQUIRED_CONTRACT_ARTIFACTS,
    normalize_reference,
    write_artifacts,
)


def test_normalize_reference_handles_decomposed_unicode() -> None:
    assert normalize_reference("Raciolingu\u0308ismo.pdf") == "Raciolingüismo.pdf"


def test_write_artifacts_emits_hybrid_contract_and_hashes(tmp_path) -> None:
    documents = pd.DataFrame({
        "document_path": ["one.pdf", "two.pdf"],
        "document_title": ["One", "Two"],
        "document_words": [2, 2],
        "text_content": ["educacion superior", "salud comunitaria"],
    })
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents["text_content"])
    config = BuildConfig(
        target_env="dev", corpus_version="corpus-v1", suite_version="suite-v1",
        source_container="source", evaluation_container="evaluation", artifact_container="artifacts",
        build_version="build-v1", account_name="storage", corpus_prefix="corpora/corpus-v1/pdfs/",
        evaluation_blob="suites/suite-v1/evaluation_queries.csv",
        artifact_prefix="builds/hybrid-tfidf-embeddings/build-v1",
    )
    suite = pd.DataFrame({"query_id": ["q1"], "query": ["educacion"], "split": ["dev"],
                          "relevant_documents": ["one.pdf"], "query_type": ["title"]})
    detail = pd.DataFrame({"method": ["hybrid_tfidf_embeddings"], "query_id": ["q1"],
                           "split": ["dev"], "precision_at_5": [0.2], "recall_at_5": [1.0],
                           "mrr": [1.0], "ndcg_at_5": [1.0]})
    summary = detail.groupby("split")[["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5"]].mean()

    write_artifacts(
        tmp_path, config, documents, vectorizer, sparse.csr_matrix(matrix), np.eye(2), 0.3,
        suite, {"document_count": 2}, {"query_count": 1}, [], [],
        pd.DataFrame({"alpha": [0.3], "mean_ndcg_at_5": [1.0]}), detail, summary,
    )

    contract = json.loads((tmp_path / "deployment_contract.json").read_text())
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert contract["artifacts"] == REQUIRED_CONTRACT_ARTIFACTS
    assert contract["selected_method"] == "hybrid_tfidf_embeddings"
    assert set(contract["artifacts"]).issubset(manifest["files"])
