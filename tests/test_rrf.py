import pytest
from langchain_core.documents import Document

from app.rag.rrf import ReciprocalRankFusion


def doc(text: str, source: str = "test.txt") -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_document_ranked_high_in_both_lists_is_favored():
    fusion = ReciprocalRankFusion(k=60)

    shared = doc("shared relevant document")
    dense_only = doc("dense result")
    bm25_only = doc("keyword result")

    results = fusion.fuse(
        [
            [shared, dense_only],
            [shared, bm25_only],
        ],
        top_k=3,
    )

    assert results[0].page_content == "shared relevant document"


def test_fusion_removes_duplicate_documents():
    fusion = ReciprocalRankFusion()

    shared = doc("same content")

    results = fusion.fuse(
        [
            [shared, shared],
            [shared],
        ],
        top_k=5,
    )

    assert len(results) == 1


def test_same_text_from_different_sources_stays_distinct():
    fusion = ReciprocalRankFusion()

    results = fusion.fuse(
        [
            [
                doc("identical text", "source_a.txt"),
                doc("identical text", "source_b.txt"),
            ]
        ],
        top_k=5,
    )

    assert len(results) == 2


def test_top_k_limits_results():
    fusion = ReciprocalRankFusion()

    results = fusion.fuse(
        [[doc("first"), doc("second"), doc("third")]],
        top_k=2,
    )

    assert len(results) == 2


def test_empty_ranked_lists_return_empty_results():
    fusion = ReciprocalRankFusion()

    assert fusion.fuse([[], []]) == []


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_fusion_constant_raises_error(k):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        ReciprocalRankFusion(k=k)


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises_error(top_k):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        ReciprocalRankFusion().fuse([], top_k=top_k)