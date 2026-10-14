from collections.abc import Callable
from dataclasses import dataclass

from app.evaluation.metrics import mrr_at_k, recall_at_k


@dataclass(frozen=True)
class EvaluationCase:
    """A query and the IDs of documents known to be relevant."""

    query: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate retrieval metrics for an evaluation dataset."""

    total_queries: int
    recall_at_k: float
    mrr_at_k: float
    k: int


def evaluate_retriever(
    cases: list[EvaluationCase],
    retrieve: Callable[[str], list[str]],
    k: int = 5,
) -> EvaluationResult:
    """Evaluate a retrieval function against known relevant document IDs."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not cases:
        raise ValueError("Evaluation cases must not be empty.")

    recall_scores: list[float] = []
    reciprocal_rank_scores: list[float] = []

    for case in cases:
        retrieved_ids = retrieve(case.query)

        recall_scores.append(
            recall_at_k(
                retrieved_ids,
                set(case.relevant_ids),
                k,
            )
        )
        reciprocal_rank_scores.append(
            mrr_at_k(
                retrieved_ids,
                set(case.relevant_ids),
                k,
            )
        )

    return EvaluationResult(
        total_queries=len(cases),
        recall_at_k=sum(recall_scores) / len(recall_scores),
        mrr_at_k=sum(reciprocal_rank_scores) / len(reciprocal_rank_scores),
        k=k,
    )