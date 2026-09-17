"""TF-IDF search logic extracted from the exploratory notebook."""

import re
from typing import Any

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


def _spanish_stop_words() -> set[str]:
    """Return NLTK's Spanish stopwords when its optional data is available."""

    try:
        return set(stopwords.words("spanish"))
    except LookupError:
        # The vectorizer vocabulary already excludes the notebook's stopwords,
        # so an empty fallback still produces valid queries in a fresh runtime.
        return set()


def preprocess_text(text: str) -> str:
    """Apply the same basic normalization used by the notebook."""

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

    stop_words = _spanish_stop_words()
    return " ".join(token for token in tokens if token not in stop_words)


def search_documents(
    query: str,
    artifacts: Any,
    top_n: int = 5,
) -> list[dict[str, object]]:
    """Return the most similar indexed documents for ``query``.

    Documents with a zero cosine score are omitted because they have no terms
    in common with the query. Results are ordered from most to least similar.
    """

    if top_n < 1:
        raise ValueError("top_n must be greater than zero")

    cleaned_query = preprocess_text(query)
    if not cleaned_query:
        return []

    query_vector = artifacts.vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vector, artifacts.matrix).ravel()
    candidate_indices = scores.argsort()[::-1][:top_n]

    results: list[dict[str, object]] = []
    for index in candidate_indices:
        score = float(scores[index])
        if score <= 0:
            continue

        row = artifacts.documents.iloc[int(index)]
        results.append(
            {
                "document_title": str(row["document_title"]),
                "document_path": str(row["document_path"]),
                "similarity_score": round(score, 6),
            }
        )

    return results
