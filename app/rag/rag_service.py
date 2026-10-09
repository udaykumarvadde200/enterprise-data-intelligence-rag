from app.rag.answer_generator import AnswerResult, GroundedAnswerGenerator
from app.rag.retrieval_pipeline import RetrievalPipeline


class RAGService:
    """Coordinate retrieval and grounded answer generation."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        answer_generator: GroundedAnswerGenerator,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.answer_generator = answer_generator

    def ask(self, query: str) -> AnswerResult:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        documents = self.retrieval_pipeline.search(query)

        return self.answer_generator.generate(
            query=query,
            documents=documents,
        )