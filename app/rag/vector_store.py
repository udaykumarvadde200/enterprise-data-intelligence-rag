import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


class ChromaVectorStore:
    """Persistent vector storage for document chunks."""

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "enterprise_knowledge",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.persist_directory = str(Path(persist_directory))

        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
        )

        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    @staticmethod
    def _document_id(document: Document) -> str:
        """Generate a deterministic ID for a chunk."""
        metadata = document.metadata

        identity = "|".join(
            str(value)
            for value in (
                metadata.get("source", ""),
                metadata.get("page", ""),
                metadata.get("start_index", ""),
                document.page_content,
            )
        )

        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def index_documents(
        self,
        documents: list[Document],
        source: str,
    ) -> int:
        """Replace the indexed chunks for one source document."""
        if not source.strip():
            raise ValueError("Source must not be empty.")

        if not documents:
            raise ValueError("Cannot index an empty document list.")

        chunks = []

        for document in documents:
            metadata = dict(document.metadata)
            metadata["source"] = source

            chunks.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        # Remove stale chunks from an earlier version of this source.
        self.store.delete(where={"source": source})

        ids = [
            self._document_id(document)
            for document in chunks
        ]

        self.store.add_documents(
            documents=chunks,
            ids=ids,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Return the most semantically similar chunks."""
        if not query.strip():
            raise ValueError("Search query must not be empty.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        return self.store.similarity_search(query, k=k)

    def count(self) -> int:
        """Return the number of chunks in this collection."""
        return self.store._collection.count()