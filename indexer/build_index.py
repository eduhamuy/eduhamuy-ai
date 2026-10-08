"""Build an auditable hybrid TF-IDF + embeddings index from Azure Blob Storage.

The module is deliberately separate from the FastAPI process: it reads a
versioned PDF corpus, verifies the evaluation suite, writes immutable build
artifacts, and optionally publishes them. Serving code only reads an approved
artifact prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import nltk
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pypdf import PdfReader
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SELECTED_METHOD = "hybrid_tfidf_embeddings"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ALPHA_CANDIDATES = tuple(round(value / 10, 1) for value in range(11))
REQUIRED_EVALUATION_COLUMNS = {
    "query_id", "query", "split", "relevant_documents", "query_type",
}
REQUIRED_CONTRACT_ARTIFACTS = [
    "tfidf/tfidf_vectorizer.joblib",
    "tfidf/X_tfidf.npz",
    "tfidf/document_metadata.csv",
    "embeddings/document_embeddings.npy",
    "embeddings/embedding_config.json",
    "hybrid_tfidf_embeddings/hybrid_config.json",
]


class IndexerError(RuntimeError):
    """Raised when source data or an artifact cannot be trusted."""


@dataclass(frozen=True)
class BuildConfig:
    target_env: str
    corpus_version: str
    suite_version: str
    source_container: str
    evaluation_container: str
    artifact_container: str
    build_version: str
    account_name: str
    corpus_prefix: str
    evaluation_blob: str
    artifact_prefix: str


def normalize_reference(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def title_from_path(path: str) -> str:
    return normalize_reference(path).removesuffix(".pdf").replace("+", " ").strip()


def ensure_nltk_resources() -> None:
    for resource_path, resource_name in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(resource_name, quiet=True):
                raise IndexerError(f"Could not download NLTK resource {resource_name}")


def preprocess_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.lower()
    normalized = re.sub(r"\d+", "", normalized)
    normalized = re.sub(r"[^a-záéíóúüñ\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    try:
        tokens = word_tokenize(normalized, language="spanish")
    except LookupError:
        tokens = normalized.split()
    return " ".join(token for token in tokens if token not in set(stopwords.words("spanish")))


def credential() -> object:
    sas = os.getenv("AZURE_STORAGE_SAS", "").strip().lstrip("?")
    if sas:
        return sas
    try:
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential(exclude_interactive_browser_credential=True)
    except Exception as exc:  # pragma: no cover - package/configuration error
        raise IndexerError("Azure authentication is not configured") from exc


def blob_service(account_name: str) -> BlobServiceClient:
    if not account_name:
        raise IndexerError("AZURE_STORAGE_ACCOUNT is required")
    return BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=credential(),
    )


def download(container: Any, name: str) -> bytes:
    try:
        return container.get_blob_client(name).download_blob().readall()
    except Exception as exc:
        raise IndexerError(f"Unable to download {container.container_name}/{name}") from exc


def read_evaluation_suite(container: Any, blob_name: str) -> tuple[pd.DataFrame, bytes]:
    raw = download(container, blob_name)
    suite = pd.read_csv(io.BytesIO(raw)).fillna("")
    missing = REQUIRED_EVALUATION_COLUMNS - set(suite.columns)
    if missing:
        raise IndexerError(f"Evaluation suite is missing columns: {sorted(missing)}")
    if suite["query_id"].duplicated().any():
        raise IndexerError("Evaluation suite has duplicate query_id values")
    suite["relevant_documents"] = suite["relevant_documents"].map(
        lambda value: "|".join(
            normalize_reference(item) for item in str(value).split("|") if item
        )
    )
    invalid_splits = sorted(set(suite["split"]) - {"dev", "test"})
    if invalid_splits:
        raise IndexerError(f"Unsupported evaluation splits: {invalid_splits}")
    return suite, raw


def extract_and_audit_corpus(
    container: Any, corpus_prefix: str, suite: pd.DataFrame, suite_bytes: bytes, config: BuildConfig
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    references: dict[str, list[dict[str, Any]]] = {}
    skipped_no_text: list[str] = []
    skipped_errors: list[dict[str, str]] = []

    blobs = [blob for blob in container.list_blobs(name_starts_with=corpus_prefix) if blob.name.lower().endswith(".pdf")]
    for position, blob in enumerate(blobs, start=1):
        print(f"[{position}/{len(blobs)}] Extracting {blob.name}", flush=True)
        raw_pdf = download(container, blob.name)
        reference = normalize_reference(blob.name[len(corpus_prefix):])
        audit = {
            # The path participates in the identifier so two intentionally
            # separate files with identical bytes remain separate documents.
            "document_id": f"doc-{sha256_bytes(blob.name.encode('utf-8') + b'\\0' + raw_pdf)[:16]}",
            "blob_path": blob.name,
            "sha256": sha256_bytes(raw_pdf),
            "bytes": len(raw_pdf),
        }
        audit_records.append(audit)
        references.setdefault(reference, []).append(audit)
        try:
            reader = PdfReader(io.BytesIO(raw_pdf))
            text = "\n".join(
                page_text for page in reader.pages
                if (page_text := page.extract_text() or "").strip()
            ).strip()
            if not text:
                skipped_no_text.append(blob.name)
                continue
            records.append({
                "document_id": audit["document_id"],
                "document_path": reference,
                "source_blob_path": blob.name,
                "document_title": title_from_path(reference),
                "text_content": text,
                "document_words": len(text.split()),
            })
        except Exception as exc:
            skipped_errors.append({"document_path": blob.name, "error": type(exc).__name__})

    if not audit_records:
        raise IndexerError(f"No PDFs found at {container.container_name}/{corpus_prefix}")
    if not records:
        raise IndexerError("No PDFs had extractable text")

    required_by_reference: dict[str, list[str]] = {}
    for row in suite.itertuples(index=False):
        for reference in filter(None, str(row.relevant_documents).split("|")):
            required_by_reference.setdefault(reference, []).append(str(row.query_id))
    missing = sorted(reference for reference in required_by_reference if reference not in references)
    ambiguous = sorted(
        reference for reference in required_by_reference if len(references.get(reference, [])) != 1
    )
    if missing or ambiguous:
        raise IndexerError(json.dumps({"missing": missing, "ambiguous": ambiguous}, ensure_ascii=False))

    corpus_manifest = {
        "environment": config.target_env,
        "corpus_version": config.corpus_version,
        "source_container": config.source_container,
        "source_prefix": config.corpus_prefix,
        "document_count": len(audit_records),
        "documents": sorted(audit_records, key=lambda item: item["blob_path"]),
    }
    suite_manifest = {
        "environment": config.target_env,
        "suite_version": config.suite_version,
        "corpus_version": config.corpus_version,
        "evaluation_blob": config.evaluation_blob,
        "evaluation_file_sha256": sha256_bytes(suite_bytes),
        "query_count": int(len(suite)),
        "required_documents": [
            {**references[reference][0], "query_ids": sorted(query_ids)}
            for reference, query_ids in sorted(required_by_reference.items())
        ],
    }
    return pd.DataFrame(records), corpus_manifest, suite_manifest, skipped_no_text, skipped_errors


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    low, high = scores.min(), scores.max()
    return np.zeros_like(scores) if high == low else (scores - low) / (high - low)


def parse_relevant(value: object) -> set[str]:
    return {normalize_reference(item) for item in str(value).split("|") if str(item).strip()}


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    gains = [1 if path in relevant else 0 for path in retrieved[:k]]
    dcg = sum(gain / np.log2(position + 2) for position, gain in enumerate(gains))
    ideal = sum(gain / np.log2(position + 2) for position, gain in enumerate(sorted(gains, reverse=True)))
    return float(dcg / ideal) if ideal else 0.0


def metric_row(retrieved: list[str], relevant: set[str], k: int = 5) -> dict[str, float]:
    hits = len(set(retrieved[:k]) & relevant)
    reciprocal = next((1.0 / position for position, path in enumerate(retrieved, 1) if path in relevant), 0.0)
    return {
        "precision_at_5": hits / k,
        "recall_at_5": hits / len(relevant) if relevant else 0.0,
        "mrr": reciprocal,
        "ndcg_at_5": ndcg_at_k(retrieved, relevant, k),
    }


def rank_paths(scores: np.ndarray, paths: list[str], k: int = 5) -> list[str]:
    return [paths[index] for index in np.argsort(scores)[::-1][:k] if scores[index] > 0]


def build_embeddings(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return np.asarray(model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True))


def evaluate_hybrid(
    suite: pd.DataFrame, paths: list[str], vectorizer: TfidfVectorizer,
    matrix: sparse.spmatrix, embeddings: np.ndarray,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def scores(query: str, alpha: float) -> np.ndarray:
        cleaned = preprocess_text(query)
        lexical = cosine_similarity(vectorizer.transform([cleaned]), matrix).ravel() if cleaned else np.zeros(len(paths))
        semantic = embeddings @ model.encode([query], normalize_embeddings=True)[0]
        return alpha * minmax(lexical) + (1 - alpha) * minmax(semantic)

    dev = suite[suite["split"] == "dev"]
    if dev.empty:
        raise IndexerError("Evaluation suite has no dev queries for alpha tuning")
    alpha_scores: list[dict[str, float]] = []
    for alpha in ALPHA_CANDIDATES:
        values = [ndcg_at_k(rank_paths(scores(row.query, alpha), paths), parse_relevant(row.relevant_documents)) for row in dev.itertuples(index=False)]
        alpha_scores.append({"alpha": alpha, "mean_ndcg_at_5": float(np.mean(values))})
    alpha = float(pd.DataFrame(alpha_scores).sort_values("mean_ndcg_at_5", ascending=False).iloc[0]["alpha"])

    detail: list[dict[str, Any]] = []
    for row in suite.itertuples(index=False):
        relevant = parse_relevant(row.relevant_documents)
        if not relevant:
            continue
        result = metric_row(rank_paths(scores(row.query, alpha), paths), relevant)
        detail.append({"method": SELECTED_METHOD, "query_id": row.query_id, "split": row.split, **result})
    detail_frame = pd.DataFrame(detail)
    if detail_frame.empty or "test" not in set(detail_frame["split"]):
        raise IndexerError("Evaluation suite has no annotated test queries")
    summaries = detail_frame.groupby("split")[["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5"]].mean()
    return alpha, pd.DataFrame(alpha_scores), detail_frame, summaries


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_artifacts(
    output_dir: Path, config: BuildConfig, documents: pd.DataFrame, vectorizer: TfidfVectorizer,
    matrix: sparse.spmatrix, embeddings: np.ndarray, alpha: float, suite: pd.DataFrame,
    corpus_manifest: dict[str, Any], suite_manifest: dict[str, Any], skipped_no_text: list[str],
    skipped_errors: list[dict[str, str]], alpha_tuning: pd.DataFrame,
    evaluation_detail: pd.DataFrame, evaluation_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the public identifier and the audited source location together with
    # the vector metadata. The serving API uses the identifier to select a
    # document; it never accepts an arbitrary blob path from a browser.
    metadata = documents[[
        "document_id", "document_path", "source_blob_path", "document_title", "document_words",
    ]].copy()
    (output_dir / "tfidf").mkdir(exist_ok=True)
    joblib.dump(vectorizer, output_dir / "tfidf" / "tfidf_vectorizer.joblib")
    sparse.save_npz(output_dir / "tfidf" / "X_tfidf.npz", matrix)
    metadata.to_csv(output_dir / "tfidf" / "document_metadata.csv", index=False)
    (output_dir / "embeddings").mkdir(exist_ok=True)
    np.save(output_dir / "embeddings" / "document_embeddings.npy", embeddings)
    write_json(output_dir / "embeddings" / "embedding_config.json", {
        "model_name": EMBEDDING_MODEL_NAME,
        "normalize_embeddings": True,
        "document_representation": "title_plus_full_text",
        "corpus_version": config.corpus_version,
    })
    (output_dir / "hybrid_tfidf_embeddings").mkdir(exist_ok=True)
    (output_dir / "evaluation").mkdir(exist_ok=True)
    write_json(output_dir / "hybrid_tfidf_embeddings" / "hybrid_config.json", {
        "method": SELECTED_METHOD,
        "lexical_method": "tfidf",
        "semantic_method": EMBEDDING_MODEL_NAME,
        "alpha": alpha,
        "normalization": "minmax",
        "corpus_version": config.corpus_version,
    })
    write_json(output_dir / "deployment_contract.json", {
        "corpus_version": config.corpus_version,
        "selected_method": SELECTED_METHOD,
        "backend_method": SELECTED_METHOD,
        "artifacts": REQUIRED_CONTRACT_ARTIFACTS,
    })
    write_json(output_dir / "corpus_manifest.json", corpus_manifest)
    write_json(output_dir / "suite_manifest.json", suite_manifest)
    write_json(output_dir / "build_metadata.json", {
        "build_version": config.build_version,
        "target_env": config.target_env,
        "corpus_version": config.corpus_version,
        "suite_version": config.suite_version,
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    suite.to_csv(output_dir / "evaluation" / "evaluation_queries_used.csv", index=False)
    alpha_tuning.to_csv(output_dir / "evaluation" / "hybrid_alpha_tuning.csv", index=False)
    evaluation_detail.to_csv(output_dir / "evaluation" / "evaluation_detail.csv", index=False)
    evaluation_summary.to_csv(output_dir / "evaluation" / "evaluation_summary.csv")
    write_json(output_dir / "evaluation" / "extraction_report.json", {
        "pdfs_without_extractable_text": skipped_no_text,
        "pdfs_with_errors": skipped_errors,
        "document_count": int(len(documents)),
    })
    manifest_path = output_dir / "artifact_manifest.json"
    files = {
        path.relative_to(output_dir).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.rglob("*")) if path.is_file() and path != manifest_path
    }
    write_json(manifest_path, {
        "target_env": config.target_env,
        "corpus_version": config.corpus_version,
        "suite_version": config.suite_version,
        "build_version": config.build_version,
        "selected_method": SELECTED_METHOD,
        "document_count": int(len(documents)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    })


def publish_tree(container: Any, local_root: Path, prefix: str) -> None:
    if list(container.list_blobs(name_starts_with=f"{prefix}/")):
        raise IndexerError(f"Artifact prefix already exists and is immutable: {prefix}")
    for path in sorted(local_root.rglob("*")):
        if path.is_file():
            name = f"{prefix}/{path.relative_to(local_root).as_posix()}"
            with path.open("rb") as handle:
                container.upload_blob(name=name, data=handle, overwrite=False)


def build(config: BuildConfig, output_dir: Path, publish: bool) -> Path:
    ensure_nltk_resources()
    service = blob_service(config.account_name)
    source = service.get_container_client(config.source_container)
    evaluation = service.get_container_client(config.evaluation_container)
    suite, suite_bytes = read_evaluation_suite(evaluation, config.evaluation_blob)
    documents, corpus_manifest, suite_manifest, skipped_no_text, skipped_errors = extract_and_audit_corpus(
        source, config.corpus_prefix, suite, suite_bytes, config
    )
    documents["cleaned_text"] = documents["text_content"].map(preprocess_text)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents["cleaned_text"])
    embeddings = build_embeddings((documents["document_title"] + ". " + documents["text_content"]).tolist())
    alpha, alpha_tuning, evaluation_detail, evaluation_summary = evaluate_hybrid(
        suite, documents["document_path"].tolist(), vectorizer, matrix, embeddings
    )
    write_artifacts(output_dir, config, documents, vectorizer, matrix, embeddings, alpha, suite,
                    corpus_manifest, suite_manifest, skipped_no_text, skipped_errors, alpha_tuning,
                    evaluation_detail, evaluation_summary)
    if publish:
        publish_tree(service.get_container_client(config.artifact_container), output_dir, config.artifact_prefix)
    return output_dir


def promote(account_name: str, artifact_container: str, build_version: str) -> str:
    source_prefix = f"builds/{SELECTED_METHOD}/{build_version}"
    target_prefix = f"indexes/{SELECTED_METHOD}/{build_version}"
    container = blob_service(account_name).get_container_client(artifact_container)
    if list(container.list_blobs(name_starts_with=f"{target_prefix}/")):
        raise IndexerError(f"Index prefix already exists and is immutable: {target_prefix}")
    manifest = json.loads(download(container, f"{source_prefix}/artifact_manifest.json"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise IndexerError("Build artifact_manifest.json is invalid")
    with tempfile.TemporaryDirectory(prefix="eduhamuy-index-promote-") as temporary:
        root = Path(temporary)
        for relative, expected in sorted(files.items()):
            if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
                raise IndexerError(f"Invalid manifest entry: {relative}")
            destination = root / PurePosixPath(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = download(container, f"{source_prefix}/{relative}")
            if sha256_bytes(data) != expected["sha256"]:
                raise IndexerError(f"Build hash mismatch: {relative}")
            destination.write_bytes(data)
        (root / "artifact_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        publish_tree(container, root, target_prefix)
    return target_prefix


def parse_build_args(args: argparse.Namespace) -> BuildConfig:
    account_name = args.account_name or os.getenv("AZURE_STORAGE_ACCOUNT", "").strip()
    environment = args.target_env
    return BuildConfig(
        target_env=environment,
        corpus_version=args.corpus_version,
        suite_version=args.suite_version,
        source_container=args.source_container or f"ai-source-{environment}",
        evaluation_container=args.evaluation_container or f"ai-evaluation-{environment}",
        artifact_container=args.artifact_container or f"ai-artifacts-{environment}",
        build_version=args.build_version,
        account_name=account_name,
        corpus_prefix=f"corpora/{args.corpus_version}/pdfs/",
        evaluation_blob=f"suites/{args.suite_version}/evaluation_queries.csv",
        artifact_prefix=f"builds/{SELECTED_METHOD}/{args.build_version}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Build and optionally publish an immutable index")
    build_parser.add_argument("--target-env", required=True)
    build_parser.add_argument("--corpus-version", required=True)
    build_parser.add_argument("--suite-version", required=True)
    build_parser.add_argument("--build-version", required=True)
    build_parser.add_argument("--account-name")
    build_parser.add_argument("--source-container")
    build_parser.add_argument("--evaluation-container")
    build_parser.add_argument("--artifact-container")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--publish", action="store_true")
    promote_parser = commands.add_parser("promote", help="Copy a verified build to an immutable index prefix")
    promote_parser.add_argument("--account-name", default=os.getenv("AZURE_STORAGE_ACCOUNT", ""))
    promote_parser.add_argument("--artifact-container", required=True)
    promote_parser.add_argument("--build-version", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            output = build(parse_build_args(args), args.output_dir, args.publish)
            print(f"Build completed: {output}")
        else:
            print(f"Index promoted: {promote(args.account_name, args.artifact_container, args.build_version)}")
    except IndexerError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
