from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Split documents into overlapping chunks while preserving metadata."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Return chunks with their original metadata and start positions."""
        if not documents:
            return []

        return self.splitter.split_documents(documents)