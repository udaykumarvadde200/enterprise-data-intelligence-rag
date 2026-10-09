import pytest
from langchain_core.documents import Document

from app.rag.answer_generator import AnswerResult
from app.rag.rag_service import RAGService


class FakeRetrievalPipeline:
    def __init__(self, documents):
        self.documents = documents
        self.last_query = None

    def search(self, query):
        self.last_query = query
        return self.documents


class FakeAnswerGenerator:
    def __init__(self):
        self.last_query = None
        self.last_documents = None

    def generate(self, query, documents):
        self.last_query = query
        self.last_documents = documents
        return AnswerResult(answer="Answer grounded in [S1].", citation_ids=["S1"])


def test_service_connects_retrieval_and_generation():
    documents = [
        Document(
            page_content="Employees must protect customer data.",
            metadata={"source": "security.txt"},
        )
    ]
    retriever = FakeRetrievalPipeline(documents)
    generator = FakeAnswerGenerator()
    service = RAGService(retriever, generator)

    result = service.ask("How should customer data be handled?")

    assert result.answer == "Answer grounded in [S1]."
    assert retriever.last_query == "How should customer data be handled?"
    assert generator.last_documents == documents
    assert generator.last_query == retriever.last_query


def test_service_rejects_empty_query():
    service = RAGService(
        FakeRetrievalPipeline([]),
        FakeAnswerGenerator(),
    )

    with pytest.raises(ValueError, match="Query must not be empty"):
        service.ask(" ")