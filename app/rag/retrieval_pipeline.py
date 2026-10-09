from langchain_core.documents import Document

from app.rag.context_filter import ContextFilter
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.ollama_reranker import OllamaRelevanceReranker


class RetrievalPipeline:
    """Retrieve, rerank, and filter relevant document chunks."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: OllamaRelevanceReranker,
        context_filter: ContextFilter | None = None,
        candidate_k: int = 10,
        rerank_k: int = 5,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")
        if rerank_k <= 0:
            raise ValueError("rerank_k must be greater than zero.")

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_filter = context_filter or ContextFilter()
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k

    def search(self, query: str) -> list[Document]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        candidates = self.hybrid_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        if not candidates:
            return []

        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=self.rerank_k,
        )

        return self.context_filter.filter(reranked)