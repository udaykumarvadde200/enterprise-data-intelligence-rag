import pytest

from app.evaluation.runner import EvaluationCase, evaluate_retriever


def test_evaluator_calculates_average_metrics():
    cases = [
        EvaluationCase(
            query="customer security",
            relevant_ids=frozenset({"doc-1"}),
        ),
        EvaluationCase(
            query="employee leave",
            relevant_ids=frozenset({"doc-3"}),
        ),
    ]

    results = {
        "customer security": ["doc-1", "doc-2"],
        "employee leave": ["doc-4", "doc-3"],
    }

    result = evaluate_retriever(
        cases,
        retrieve=lambda query: results[query],
        k=2,
    )

    assert result.total_queries == 2
    assert result.k == 2
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.mrr_at_k == pytest.approx(0.75)


def test_evaluator_rejects_empty_cases():
    with pytest.raises(ValueError, match="Evaluation cases must not be empty"):
        evaluate_retriever([], retrieve=lambda query: [], k=5)


@pytest.mark.parametrize("k", [0, -1])
def test_evaluator_rejects_nonpositive_k(k):
    case = EvaluationCase(
        query="test",
        relevant_ids=frozenset({"doc-1"}),
    )

    with pytest.raises(ValueError, match="k must be greater than zero"):
        evaluate_retriever(
            [case],
            retrieve=lambda query: ["doc-1"],
            k=k,
        )