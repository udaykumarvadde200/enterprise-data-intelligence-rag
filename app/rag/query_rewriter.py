from langchain_core.messages import BaseMessage

from app.llm.ollama_client import OllamaClient


class QueryRewriter:
    """Rewrite follow-up questions into standalone search queries."""

    def __init__(self, llm_client: OllamaClient | None = None):
        self.llm_client = llm_client or OllamaClient()

    @staticmethod
    def _history_text(
        conversation_history: list[dict[str, str]],
        max_turns: int = 5,
    ) -> str:
        recent_turns = conversation_history[-max_turns:]

        if not recent_turns:
            return "(No previous conversation.)"

        return "\n".join(
            f"User: {turn.get('question', '')}\n"
            f"Assistant: {turn.get('answer', '')}"
            for turn in recent_turns
        )

    def rewrite(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Return a standalone query grounded in recent conversation."""
        question = question.strip()
        if not question:
            return question

        history = conversation_history or []
        if not history:
            return question

        prompt = f"""
You rewrite follow-up questions for an enterprise knowledge platform.

Convert the latest user question into a standalone query that can be
understood without the conversation history.

Rules:
- Preserve the user's intent and all explicit constraints.
- Resolve references such as "it", "that", "same", "previous year", and
  "what about 2024" using the conversation when possible.
- Do not answer the question.
- Do not invent facts, table names, column names, filters, or values.
- If the reference cannot be resolved confidently, return the original
  question unchanged.
- Output only the rewritten question, with no explanation or quotes.

Recent conversation:
{self._history_text(history)}

Latest question:
{question}

Standalone query:
""".strip()

        try:
            response = self.llm_client.invoke(prompt)

            if isinstance(response, BaseMessage):
                rewritten = str(response.content).strip()
            else:
                rewritten = str(response).strip()

            # Guard against empty or unexpectedly verbose model output.
            if not rewritten or "\n" in rewritten:
                return question

            rewritten = rewritten.strip('"').strip("'").strip()
            return rewritten or question
        except Exception:
            # Query rewriting is an enhancement, not a single point of failure.
            return question
