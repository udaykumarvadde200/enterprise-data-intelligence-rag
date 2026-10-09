from collections import defaultdict

from langchain_core.documents import Document


class ReciprocalRankFusion:
    """Combine ranked document lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        self.k = k

    @staticmethod
    def _document_key(document: Document) -> tuple[str, str]:
        """Identify a chunk using its source metadata and text."""
        return (
            str(document.metadata.get("source", "")),
            document.page_content,
        )

    def fuse(
        self,
        ranked_lists: list[list[Document]],
        top_k: int = 4,
    ) -> list[Document]:
        """Fuse ranked lists and return the top-k unique documents."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        scores: dict[tuple[str, str], float] = defaultdict(float)
        unique_documents: dict[tuple[str, str], Document] = {}

        for ranked_documents in ranked_lists:
            seen_in_list = set()

            for rank, document in enumerate(ranked_documents, start=1):
                key = self._document_key(document)

                # A duplicate within one list should not gain extra weight.
                if key in seen_in_list:
                    continue

                seen_in_list.add(key)
                unique_documents.setdefault(key, document)
                scores[key] += 1.0 / (self.k + rank)

        ranked_keys = sorted(
            scores,
            key=lambda key: (-scores[key], key),
        )

        return [
            unique_documents[key]
            for key in ranked_keys[:top_k]
        ]