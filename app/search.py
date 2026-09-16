"""TF-IDF search logic.

The search implementation will be extracted from the exploratory notebook in
a subsequent change. Keeping it in its own module prevents the HTTP layer from
being coupled to the indexing details.
"""


def search_documents(query: str, top_n: int = 5) -> list[dict[str, object]]:
    """Search indexed documents.

    This is intentionally a placeholder for the initial repository bootstrap.
    """

    if not query.strip():
        return []

    raise NotImplementedError("TF-IDF search is not implemented yet")
