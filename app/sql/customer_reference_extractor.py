from app.llm.ollama_client import OllamaClient


class CustomerReferenceExtractor:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def extract(self, question: str) -> str:
        prompt = f"""
Identify the customer name mentioned in the user's question.

Rules:
1. Return ONLY the customer name.
2. Do not include explanations.
3. If no customer is mentioned, return NONE.

USER QUESTION:
{question}
"""

        return self.llm.invoke(prompt).strip()