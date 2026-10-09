import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25DocumentRetriever:
    """Keyword-based retrieval over document chunks using BM25."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = []
        tokenized_documents = []

        for document in documents:
            tokens = self._tokenize(document.page_content)

            # Ignore empty or whitespace-only chunks.
            if tokens:
                self.documents.append(document)
                tokenized_documents.append(tokens)

        self.bm25 = (
            BM25Okapi(tokenized_documents)
            if tokenized_documents
            else None
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into lowercase word tokens."""
        return re.findall(r"\w+", text.casefold())

    def search(self, query: str, k: int = 4) -> list[Document]:
        """Return the top-k documents matching the query keywords."""
        if not query.strip():
            raise ValueError("Search query must not be empty.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        query_tokens = self._tokenize(query)

        if self.bm25 is None or not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Stable ordering: preserve corpus order when scores are tied.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )

        # Do not return documents with no keyword match.
                # Return documents containing at least one query term.
        # A matching document can have a zero BM25 score when all
        # query terms occur in every document in a tiny corpus.
        matching_indices = [
            index
            for index, document in enumerate(self.documents)
            if set(query_tokens).intersection(
                self._tokenize(document.page_content)
            )
        ]

        matching_indices.sort(
            key=lambda index: (-scores[index], index)
        )

        return [
            self.documents[index]
            for index in matching_indices[:k]
        ]