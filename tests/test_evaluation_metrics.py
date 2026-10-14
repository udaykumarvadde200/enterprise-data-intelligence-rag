import pytest

from app.evaluation.metrics import mrr_at_k, recall_at_k


def test_recall_at_k_counts_relevant_documents():
    retrieved = ["doc-1", "doc-2", "doc-3"]
    relevant = {"doc-2", "doc-3", "doc-4"}

    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3)


def test_recall_at_k_returns_one_when_all_relevant_found():
    assert recall_at_k(
        ["doc-1", "doc-2"],
        {"doc-1", "doc-2"},
        k=2,
    ) == pytest.approx(1.0)


def test_recall_at_k_returns_zero_for_no_relevant_results():
    assert recall_at_k(["doc-1"], {"doc-2"}, k=1) == 0.0


def test_recall_at_k_returns_zero_for_empty_relevance_set():
    assert recall_at_k(["doc-1"], set(), k=1) == 0.0


def test_mrr_at_k_uses_first_relevant_rank():
    retrieved = ["doc-1", "doc-2", "doc-3"]
    relevant = {"doc-2", "doc-3"}

    assert mrr_at_k(retrieved, relevant, k=3) == pytest.approx(0.5)


def test_mrr_at_k_returns_zero_when_no_relevant_result_found():
    assert mrr_at_k(["doc-1", "doc-2"], {"doc-3"}, k=2) == 0.0


def test_mrr_at_k_returns_one_when_first_result_is_relevant():
    assert mrr_at_k(["doc-2", "doc-1"], {"doc-2"}, k=2) == 1.0


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_nonpositive_k(k):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        recall_at_k(["doc-1"], {"doc-1"}, k)

    with pytest.raises(ValueError, match="k must be greater than zero"):
        mrr_at_k(["doc-1"], {"doc-1"}, k)