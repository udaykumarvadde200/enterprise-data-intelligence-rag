import math
from numbers import Real

from langchain_core.documents import Document


class ContextFilter:
    """Deduplicate and limit retrieved evidence before generation."""

    def __init__(
        self,
        max_chunks: int = 5,
        max_characters: int = 6000,
        min_score: float | None = None,
    ) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero.")
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero.")
        if min_score is not None and (
            isinstance(min_score, bool)
            or not isinstance(min_score, Real)
            or not math.isfinite(min_score)
        ):
            raise ValueError("min_score must be a finite number or None.")

        self.max_chunks = max_chunks
        self.max_characters = max_characters
        # Reranker-score floor. Scores are model-specific (the MiniLM
        # cross-encoder emits raw logits), so this value must be tuned for
        # the reranker in use. None disables score filtering.
        self.min_score = min_score

    def _below_min_score(self, document: Document) -> bool:
        if self.min_score is None:
            return False

        score = (document.metadata or {}).get("rerank_score")

        # Unscored documents cannot be judged here; keep them. The grounding
        # verifier still gates any answer built from them.
        if isinstance(score, bool) or not isinstance(score, Real):
            return False

        return not math.isfinite(score) or score < self.min_score

    def filter(self, documents: list[Document]) -> list[Document]:
        selected = []
        seen = set()
        remaining = self.max_characters

        for document in documents:
            content = document.page_content.strip()
            if not content:
                continue

            if self._below_min_score(document):
                continue

            # Deduplicate identical text regardless of source filename.
            identity = content
            if identity in seen:
                continue
            seen.add(identity)

            if remaining <= 0:
                break

            # Preserve the context budget even for oversized chunks.
            content = content[:remaining]
            if not content:
                break

            selected.append(
                Document(
                    page_content=content,
                    metadata=dict(document.metadata),
                )
            )
            remaining -= len(content)

            if len(selected) >= self.max_chunks:
                break

        return selected