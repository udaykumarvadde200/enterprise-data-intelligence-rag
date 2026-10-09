import re


class QueryRouter:
    """Route greetings, document questions, database questions, or hybrid queries."""

    GREETING_PATTERNS = (
        r"hi",
        r"hello",
        r"hey",
        r"hi there",
        r"hello there",
        r"good morning",
        r"good afternoon",
        r"good evening",
        r"how are you",
        r"thanks",
        r"thank you",
        r"bye",
        r"goodbye",
    )

    HYBRID_KEYWORDS = (
        "and how many",
        "and how much",
        "combine",
        "according to the document",
        "based on the policy",
        "document and database",
        "document and sql",
        "compare the policy",
    )

    RAG_KEYWORDS = (
        "resume",
        "resumes",
        "cv",
        "curriculum vitae",
        "uploaded file",
        "uploaded document",
        "document",
        "documents",
        "pdf",
        "policy",
        "policies",
        "guidelines",
        "manual",
        "contract",
        "report",
        "according to the file",
        "according to the document",
        "summarize",
        "summarise",
        "explain the document",
        "what is in the file",
        "what is in the document",
        "what is in the resume",
        "what does the file say",
        "what does the document say",
    )

    SQL_KEYWORDS = (
        "customer",
        "customers",
        "order",
        "orders",
        "database",
        "database records",
        "total sales",
        "revenue",
        "count",
        "average",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "greater than",
        "less than",
        "above",
        "below",
        "how many",
        "how much",
        "show me",
        "list all",
    )

    @classmethod
    def _contains_phrase(cls, question: str, phrases: tuple[str, ...]) -> bool:
        """Match phrases without accidentally matching inside other words."""
        return any(
            re.search(
                rf"(?<!\w){re.escape(phrase)}(?!\w)",
                question,
            )
            is not None
            for phrase in phrases
        )

    def route(self, question: str) -> str:
        """Return general, rag, sql, or hybrid."""
        normalized = " ".join(question.casefold().split()).strip()

        if not normalized:
            return "general"

        # Greetings and conversational messages should never query the database.
        if normalized.strip(" \t!.,?") in self.GREETING_PATTERNS:            
            return "general"

        # Cross-source questions must take priority over single-source routing.
        if self._contains_phrase(normalized, self.HYBRID_KEYWORDS):
            return "hybrid"

        has_rag_intent = self._contains_phrase(normalized, self.RAG_KEYWORDS)
        has_sql_intent = self._contains_phrase(normalized, self.SQL_KEYWORDS)

        # Explicit document intent wins over incidental database-related words.
        if has_rag_intent and has_sql_intent:
            return "hybrid"

        if has_rag_intent:
            return "rag"

        if has_sql_intent:
            return "sql"

        # Do not guess that an unrecognized question is a database query.
        return "general"