from app.llm.ollama_client import OllamaClient


class SQLGenerator:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def generate(self, question: str, schema: str) -> str:
        prompt = f"""
You are a PostgreSQL SQL generation assistant.

Your task is to convert the user's natural language question
into a valid PostgreSQL SQL query.

Use ONLY the tables and columns provided in the schema.

Do NOT invent tables or columns.

Return ONLY the SQL query.
Do not include explanations.
Do not use markdown code fences.

DATABASE SCHEMA:

{schema}

USER QUESTION:

{question}
"""

        return self.llm.invoke(prompt).strip()