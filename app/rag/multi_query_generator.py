from app.llm.ollama_client import OllamaClient


class MultiQueryGenerator:
    """Generate alternative search queries using a local LLM."""

    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        max_alternatives: int = 0,
    ) -> None:
        if max_alternatives < 0:
            raise ValueError("max_alternatives must not be negative.")

        self.llm_client = llm_client or OllamaClient()
        self.max_alternatives = max_alternatives

    def generate(self, question: str) -> list[str]:
        """Return the original query plus distinct alternative queries."""
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        queries = [question]

        if self.max_alternatives == 0:
            return queries

        prompt = f"""
Generate up to {self.max_alternatives} alternative search queries for
an enterprise knowledge retrieval system.

Rules:
- Preserve the original question's meaning and intent.
- Use different wording or useful search terminology.
- Do not answer the question.
- Do not invent facts, names, database tables, columns, or filters.
- Return only the alternative queries, one per line.
- Do not number the lines or add explanations.
- Do not repeat the original question.

Original question:
{question}
""".strip()

        try:
            response = self.llm_client.invoke(prompt)
            output = (
                response.content
                if hasattr(response, "content")
                else response
            )

            if not isinstance(output, str):
                return queries

            for line in output.splitlines():
                candidate = line.strip()
                # Accept common numbered or bulleted model output.
                while candidate and candidate[0] in "-*•":
                    candidate = candidate[1:].strip()

                if candidate[:2].rstrip(".").isdigit():
                    candidate = candidate[2:].strip()
                elif (
                    len(candidate) > 2
                    and candidate[0].isdigit()
                    and candidate[1] in ".)"
                ):
                    candidate = candidate[2:].strip()

                candidate = candidate.strip("\"' ").strip()

                if not candidate:
                    continue

                if candidate.casefold() not in {
                    query.casefold() for query in queries
                }:
                    queries.append(candidate)

                if len(queries) >= self.max_alternatives + 1:
                    break

        except Exception:
            # Search must remain usable if the local LLM is unavailable.
            return [question]

        return queries