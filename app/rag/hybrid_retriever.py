from langchain_core.documents import Document

from app.rag.bm25_retriever import BM25DocumentRetriever
from app.rag.rrf import ReciprocalRankFusion
from app.rag.vector_store import ChromaVectorStore


class HybridRetriever:
    """Combine dense retrieval and BM25 using RRF."""

    def __init__(
        self,
        documents: list[Document],
        vector_store: ChromaVectorStore,
        dense_k: int = 10,
        sparse_k: int = 10,
        fusion_k: int = 60,
    ) -> None:
        if not documents:
            raise ValueError("Documents must not be empty.")

        if dense_k <= 0 or sparse_k <= 0:
            raise ValueError("Retrieval k values must be greater than zero.")

        self.vector_store = vector_store
        self.sparse_retriever = BM25DocumentRetriever(documents)
        self.fusion = ReciprocalRankFusion(k=fusion_k)
        self.dense_k = dense_k
        self.sparse_k = sparse_k

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve candidates from both systems and fuse their rankings."""
        if not query.strip():
            raise ValueError("Search query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        dense_results = self.vector_store.search(
            query,
            k=self.dense_k,
        )

        sparse_results = self.sparse_retriever.search(
            query,
            k=self.sparse_k,
        )

        return self.fusion.fuse(
            [dense_results, sparse_results],
            top_k=top_k,
        )