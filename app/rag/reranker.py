from langchain_core.documents import Document


class CrossEncoderReranker:
    """Rerank retrieved documents using query-document relevance scores."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model=None,
    ) -> None:
        # Injection lets us test ranking without downloading a model.
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers to use cross-encoder reranking."
                ) from exc

            model = CrossEncoder(model_name)

        self.model = model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not documents:
            return []

        pairs = [
            (query, document.page_content)
            for document in documents
        ]
        scores = self.model.predict(pairs)

        if len(scores) != len(documents):
            raise ValueError("The reranker returned an unexpected number of scores.")

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results = []
        for document, score in ranked[:top_k]:
            metadata = dict(document.metadata)
            metadata["rerank_score"] = float(score)

            results.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return results