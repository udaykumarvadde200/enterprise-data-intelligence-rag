import pytest
from langchain_core.documents import Document

from app.rag.bm25_retriever import BM25DocumentRetriever


@pytest.fixture
def documents():
    return [
        Document(
            page_content=(
                "Employees must protect confidential "
                "customer information and personal data."
            ),
            metadata={"source": "security_policy.txt"},
        ),
        Document(
            page_content=(
                "Employees can request annual leave "
                "through the human resources department."
            ),
            metadata={"source": "leave_policy.txt"},
        ),
        Document(
            page_content=(
                "The database stores customer orders "
                "and transaction records."
            ),
            metadata={"source": "database_guide.txt"},
        ),
    ]


def test_keyword_search_ranks_relevant_document_first(documents):
    retriever = BM25DocumentRetriever(documents)

    results = retriever.search("annual leave", k=3)

    assert results
    assert results[0].metadata["source"] == "leave_policy.txt"


def test_search_preserves_document_metadata(documents):
    retriever = BM25DocumentRetriever(documents)

    results = retriever.search("confidential customer information", k=1)

    assert results[0].metadata["source"] == "security_policy.txt"


def test_search_respects_k(documents):
    retriever = BM25DocumentRetriever(documents)

    results = retriever.search("employees customer", k=1)

    assert len(results) <= 1


def test_unmatched_query_returns_no_documents(documents):
    retriever = BM25DocumentRetriever(documents)

    assert retriever.search("volcanic astronomy", k=3) == []


def test_empty_corpus_returns_no_documents():
    retriever = BM25DocumentRetriever([])

    assert retriever.search("customer data") == []


def test_empty_documents_are_ignored():
    retriever = BM25DocumentRetriever(
        [
            Document(page_content="   ", metadata={"source": "empty.txt"}),
            Document(
                page_content="Customer records are confidential.",
                metadata={"source": "records.txt"},
            ),
        ]
    )

    results = retriever.search("customer records", k=2)

    assert len(results) == 1
    assert results[0].metadata["source"] == "records.txt"


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_raises_error(query, documents):
    retriever = BM25DocumentRetriever(documents)

    with pytest.raises(ValueError, match="Search query must not be empty"):
        retriever.search(query)


def test_nonpositive_k_raises_error(documents):
    retriever = BM25DocumentRetriever(documents)

    with pytest.raises(ValueError, match="k must be greater than zero"):
        retriever.search("customer", k=0)