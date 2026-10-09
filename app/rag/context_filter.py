from langchain_core.documents import Document


class ContextFilter:
    """Deduplicate and limit retrieved evidence before generation."""

    def __init__(
        self,
        max_chunks: int = 5,
        max_characters: int = 6000,
    ) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero.")
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero.")

        self.max_chunks = max_chunks
        self.max_characters = max_characters

    def filter(self, documents: list[Document]) -> list[Document]:
        selected = []
        seen = set()
        remaining = self.max_characters

        for document in documents:
            content = document.page_content.strip()
            if not content:
                continue

            identity = (
                document.metadata.get("source", ""),
                document.metadata.get("page", ""),
                document.metadata.get("start_index", ""),
                content,
            )
            if identity in seen:
                continue
            seen.add(identity)

            if remaining <= 0:
                break

            # Preserve the context budget even if a single chunk is oversized.
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