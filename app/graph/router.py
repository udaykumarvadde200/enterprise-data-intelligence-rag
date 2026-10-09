class QueryRouter:
    """Routes user questions to SQL, RAG, or hybrid processing."""

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
        "document",
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

    def route(self, question: str) -> str:
        normalized = question.lower().strip()

        # Questions requiring both documents and structured data
        if any(
            keyword in normalized
            for keyword in self.HYBRID_KEYWORDS
        ):
            return "hybrid"

        has_rag_intent = any(
            keyword in normalized
            for keyword in self.RAG_KEYWORDS
        )

        has_sql_intent = any(
            keyword in normalized
            for keyword in self.SQL_KEYWORDS
        )

        if has_rag_intent and has_sql_intent:
            return "hybrid"

        if has_rag_intent:
            return "rag"

        if has_sql_intent:
            return "sql"

        # Default to SQL for now. We'll improve routing and
        # ambiguity detection as the full workflow develops.
        return "sql"