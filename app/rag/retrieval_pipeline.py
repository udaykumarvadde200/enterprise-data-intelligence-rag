from langchain_core.documents import Document

from app.rag.context_filter import ContextFilter
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.multi_query_generator import MultiQueryGenerator
from app.rag.ollama_reranker import OllamaRelevanceReranker
from app.rag.rrf import ReciprocalRankFusion


class RetrievalPipeline:
    """Generate queries, retrieve, fuse, rerank, and filter evidence."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: OllamaRelevanceReranker,
        context_filter: ContextFilter | None = None,
        candidate_k: int = 10,
        rerank_k: int = 5,
        query_generator: MultiQueryGenerator | None = None,
        query_fusion: ReciprocalRankFusion | None = None,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")
        if rerank_k <= 0:
            raise ValueError("rerank_k must be greater than zero.")

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_filter = context_filter or ContextFilter()
        self.query_generator = query_generator or MultiQueryGenerator()
        self.query_fusion = query_fusion or ReciprocalRankFusion()
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k

    def search(self, query: str) -> list[Document]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        queries = self.query_generator.generate(query)

        ranked_lists = [
            self.hybrid_retriever.search(
                generated_query,
                top_k=self.candidate_k,
            )
            for generated_query in queries
        ]

        candidates = self.query_fusion.fuse(
            ranked_lists,
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