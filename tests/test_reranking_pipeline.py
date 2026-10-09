import pytest
from langchain_core.documents import Document

from app.rag.context_filter import ContextFilter
from app.rag.reranker import CrossEncoderReranker
from app.rag.retrieval_pipeline import RetrievalPipeline


def doc(text, source="policy.txt"):
    return Document(page_content=text, metadata={"source": source})


class FakeCrossEncoder:
    def predict(self, pairs):
        # Higher scores should rank higher.
        return [0.2, 0.9, 0.5][:len(pairs)]


class FakeHybridRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.last_query = None
        self.last_top_k = None

    def search(self, query, top_k=5):
        self.last_query = query
        self.last_top_k = top_k
        return self.documents[:top_k]


def test_reranker_orders_documents_by_score():
    documents = [doc("First"), doc("Second"), doc("Third")]
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    results = reranker.rerank("test query", documents, top_k=3)

    assert [item.page_content for item in results] == [
        "Second", "Third", "First"
    ]
    assert results[0].metadata["rerank_score"] == pytest.approx(0.9)


def test_reranker_limits_results():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    results = reranker.rerank(
        "test query",
        [doc("A"), doc("B"), doc("C")],
        top_k=2,
    )

    assert len(results) == 2


def test_reranker_empty_documents():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    assert reranker.rerank("test query", []) == []


@pytest.mark.parametrize("query", ["", "   "])
def test_reranker_rejects_empty_query(query):
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    with pytest.raises(ValueError, match="Query must not be empty"):
        reranker.rerank(query, [doc("Example")])


def test_context_filter_removes_duplicates_and_empty_chunks():
    context_filter = ContextFilter()
    documents = [
        doc("Confidential information."),
        doc("Confidential information."),
        doc("   "),
        doc("Password policy."),
    ]

    results = context_filter.filter(documents)

    assert [item.page_content for item in results] == [
        "Confidential information.",
        "Password policy.",
    ]


def test_context_filter_respects_character_budget():
    context_filter = ContextFilter(max_chunks=5, max_characters=10)

    results = context_filter.filter([
        doc("12345678"),
        doc("abcdefgh"),
    ])

    assert "".join(item.page_content for item in results) == "12345678ab"
    assert sum(len(item.page_content) for item in results) <= 10


def test_context_filter_respects_chunk_limit():
    context_filter = ContextFilter(max_chunks=2)

    results = context_filter.filter([
        doc("One"),
        doc("Two"),
        doc("Three"),
    ])

    assert len(results) == 2


def test_pipeline_connects_retrieval_reranking_and_filtering():
    documents = [
        doc("First candidate"),
        doc("Second candidate"),
        doc("Third candidate"),
    ]
    hybrid = FakeHybridRetriever(documents)
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        context_filter=ContextFilter(max_chunks=2),
        candidate_k=3,
        rerank_k=3,
    )

    results = pipeline.search("test query")

    assert hybrid.last_query == "test query"
    assert hybrid.last_top_k == 3
    assert [item.page_content for item in results] == [
        "Second candidate",
        "Third candidate",
    ]


def test_pipeline_rejects_empty_query():
    pipeline = RetrievalPipeline(
        hybrid_retriever=FakeHybridRetriever([]),
        reranker=CrossEncoderReranker(model=FakeCrossEncoder()),
    )

    with pytest.raises(ValueError, match="Query must not be empty"):
        pipeline.search("  ")