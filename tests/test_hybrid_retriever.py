import pytest
from langchain_core.documents import Document

from app.rag.hybrid_retriever import HybridRetriever


class FakeVectorStore:
    """Return predetermined dense-search results without Ollama."""

    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_k = None

    def search(self, query: str, k: int = 4) -> list[Document]:
        self.last_query = query
        self.last_k = k
        return self.results[:k]


def make_doc(text: str, source: str) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source},
    )


@pytest.fixture
def setup_retriever():
    shared = make_doc(
        "Confidential customer security policy.",
        "security.txt",
    )
    dense_only = make_doc(
        "Semantic result about protecting information.",
        "semantic.txt",
    )
    sparse_only = make_doc(
        "Confidential information must remain private.",
        "confidential.txt",
    )
    unrelated = make_doc(
        "Employees can request annual leave.",
        "leave.txt",
    )

    corpus = [shared, dense_only, sparse_only, unrelated]

    vector_store = FakeVectorStore([shared, dense_only])

    retriever = HybridRetriever(
        documents=corpus,
        vector_store=vector_store,
        dense_k=2,
        sparse_k=4,
    )

    return retriever, vector_store, shared, dense_only


def test_hybrid_search_returns_ranked_documents(setup_retriever):
    retriever, _, shared, _ = setup_retriever

    results = retriever.search("confidential customer security", top_k=3)

    assert results
    assert results[0].page_content == shared.page_content


def test_dense_retriever_receives_query_and_k(setup_retriever):
    retriever, vector_store, _, _ = setup_retriever

    retriever.search("confidential customer security", top_k=2)

    assert vector_store.last_query == "confidential customer security"
    assert vector_store.last_k == 2


def test_hybrid_results_are_deduplicated(setup_retriever):
    retriever, _, _, _ = setup_retriever

    results = retriever.search("confidential customer security", top_k=10)
    identities = [
        (doc.metadata.get("source"), doc.page_content)
        for doc in results
    ]

    assert len(identities) == len(set(identities))


def test_top_k_limits_results(setup_retriever):
    retriever, _, _, _ = setup_retriever

    results = retriever.search("confidential", top_k=1)

    assert len(results) <= 1


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_raises_error(setup_retriever, query):
    retriever, _, _, _ = setup_retriever

    with pytest.raises(ValueError, match="Search query must not be empty"):
        retriever.search(query)


def test_empty_documents_raise_error():
    with pytest.raises(ValueError, match="Documents must not be empty"):
        HybridRetriever(
            documents=[],
            vector_store=FakeVectorStore([]),
        )


def test_nonpositive_retrieval_k_raises_error():
    with pytest.raises(ValueError, match="Retrieval k values"):
        HybridRetriever(
            documents=[make_doc("Example content.", "example.txt")],
            vector_store=FakeVectorStore([]),
            dense_k=0,
        )


def test_nonpositive_top_k_raises_error(setup_retriever):
    retriever, _, _, _ = setup_retriever

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.search("confidential", top_k=0)