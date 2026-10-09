import hashlib
import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.vector_store import ChromaVectorStore


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for isolated tests."""

    dimensions = 16

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))

        if magnitude:
            vector = [value / magnitude for value in vector]

        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def store(tmp_path):
    """Create a fresh, isolated Chroma collection for each test."""
    return ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedding_model="unused-in-tests",
    )


@pytest.fixture(autouse=True)
def use_fake_embeddings(monkeypatch):
    """Prevent tests from creating a real Ollama embedding client."""
    monkeypatch.setattr(
        "app.rag.vector_store.OllamaEmbeddings",
        lambda **kwargs: FakeEmbeddings(),
    )


def test_index_and_count_documents(store):
    documents = [
        Document(
            page_content="Protect confidential customer information.",
            metadata={"file_type": "txt"},
        )
    ]

    indexed = store.index_documents(
        documents,
        source="security_policy.txt",
    )

    assert indexed == 1
    assert store.count() == 1


def test_indexed_source_metadata_is_preserved(store):
    documents = [
        Document(
            page_content="Employees must protect customer data.",
            metadata={"file_type": "txt"},
        )
    ]

    store.index_documents(documents, source="security_policy.txt")
    results = store.search("protect customer data", k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "security_policy.txt"
    assert results[0].metadata["file_type"] == "txt"


def test_reindex_replaces_old_chunks_for_same_source(store):
    first_version = [
        Document(page_content="Old policy text.", metadata={}),
        Document(page_content="Old additional policy text.", metadata={}),
    ]

    updated_version = [
        Document(page_content="Updated policy text.", metadata={}),
    ]

    store.index_documents(first_version, source="policy.txt")
    assert store.count() == 2

    store.index_documents(updated_version, source="policy.txt")

    assert store.count() == 1
    results = store.search("updated policy", k=1)
    assert results[0].page_content == "Updated policy text."


def test_different_sources_are_kept_separate(store):
    store.index_documents(
        [Document(page_content="Customer security rules.", metadata={})],
        source="security.txt",
    )
    store.index_documents(
        [Document(page_content="Annual leave rules.", metadata={})],
        source="leave.txt",
    )

    assert store.count() == 2

    results = store.search("annual leave", k=2)
    sources = {document.metadata["source"] for document in results}

    assert sources == {"security.txt", "leave.txt"}


@pytest.mark.parametrize("source", ["", "   "])
def test_empty_source_raises_error(store, source):
    with pytest.raises(ValueError, match="Source must not be empty"):
        store.index_documents(
            [Document(page_content="Some text.", metadata={})],
            source=source,
        )


def test_empty_document_list_raises_error(store):
    with pytest.raises(ValueError, match="Cannot index an empty document list"):
        store.index_documents([], source="policy.txt")


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_search_query_raises_error(store, query):
    with pytest.raises(ValueError, match="Search query must not be empty"):
        store.search(query)


def test_nonpositive_k_raises_error(store):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        store.search("customer data", k=0)