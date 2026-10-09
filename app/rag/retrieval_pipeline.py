from langchain_core.documents import Document

from app.rag.context_filter import ContextFilter
from app.rag.reranker import CrossEncoderReranker


class RetrievalPipeline:
    """Hybrid retrieval followed by reranking and context filtering."""

    def __init__(
        self,
        hybrid_retriever,
        reranker: CrossEncoderReranker,
        context_filter: ContextFilter | None = None,
        candidate_k: int = 10,
        rerank_k: int = 5,
    ) -> None:
        if candidate_k <= 0 or rerank_k <= 0:
            raise ValueError("Retrieval candidate counts must be positive.")

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

        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=self.rerank_k,
        )

        return self.context_filter.filter(reranked)