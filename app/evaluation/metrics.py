def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate the fraction of relevant documents retrieved in top-k."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def mrr_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate reciprocal rank of the first relevant result in top-k."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    for rank, document_id in enumerate(retrieved_ids[:k], start=1):
        if document_id in relevant_ids:
            return 1.0 / rank

    return 0.0