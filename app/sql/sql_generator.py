import sqlglot
from sqlglot import exp

from app.llm.ollama_client import OllamaClient


class SQLGenerator:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def generate(
        self,
        question: str,
        schema: str,
        resolved_customer_id: int | None = None,
    ) -> str:

        prompt = f"""
You are a PostgreSQL SQL generation assistant.

Convert the user's natural language question into a valid
PostgreSQL SQL query.

Rules:
1. Use ONLY tables and columns provided in the schema.
2. Do NOT invent tables or columns.
3. Use PostgreSQL syntax.
4. If the question cannot be answered using the schema,
   clearly state that it cannot be answered.
5. Return ONLY the SQL query.
6. Do NOT use markdown code fences.
7. Do NOT include explanations.
8. Generate READ-ONLY SQL only.

DATABASE SCHEMA:

{schema}

USER QUESTION:

{question}
"""

        sql = self.llm.invoke(prompt).strip()

        # If a customer was resolved by the deterministic
        # clarification system, enforce the exact customer_id.
        #
        # We do NOT rely on the LLM to obey this constraint.
        if resolved_customer_id is not None:
            sql = self._enforce_customer_id(
                sql,
                resolved_customer_id,
            )

        return sql

    def _enforce_customer_id(
        self,
        sql: str,
        customer_id: int,
    ) -> str:

        parsed = sqlglot.parse_one(
            sql,
            dialect="postgres",
        )

        customer_condition = exp.EQ(
            this=exp.Column(
                this=exp.Identifier(
                    this="customer_id",
                    quoted=False,
                ),
                table=exp.Identifier(
                    this="c",
                    quoted=False,
                ),
            ),
            expression=exp.Literal.number(customer_id),
        )

        where = parsed.args.get("where")

        if where is not None:
            existing_condition = where.this

            parsed.set(
                "where",
                exp.Where(
                    this=exp.And(
                        this=existing_condition,
                        expression=customer_condition,
                    )
                ),
            )
        else:
            parsed.set(
                "where",
                exp.Where(
                    this=customer_condition,
                ),
            )

        return parsed.sql(
            dialect="postgres",
        )