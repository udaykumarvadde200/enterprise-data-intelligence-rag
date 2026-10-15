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

        if resolved_customer_id is not None:
            sql = self._enforce_customer_id(
                sql,
                resolved_customer_id,
            )

        return sql

    def _enforce_customer_id(self, sql: str, customer_id: int) -> str:
        parsed = sqlglot.parse_one(sql, dialect="postgres")

        if not isinstance(parsed, exp.Select):
            raise ValueError("Only SELECT queries are allowed.")

        from_clause = parsed.args.get("from_")
        joins = parsed.args.get("joins") or []

        outer_sources = []

        if from_clause is not None:
            outer_sources.append(from_clause.this)

        outer_sources.extend(join.this for join in joins)

        orders_table = next(
            (
                source
                for source in outer_sources
                if isinstance(source, exp.Table)
                and source.name == "orders"
            ),
            None,
        )

        if orders_table is None:
            raise ValueError(
                "Cannot safely enforce customer ID: "
                "the outer query must reference the orders table."
            )

        orders_reference = orders_table.alias_or_name

        customer_condition = exp.EQ(
            this=exp.Column(
                this=exp.Identifier(this="customer_id"),
                table=exp.Identifier(this=orders_reference),
            ),
            expression=exp.Literal.number(customer_id),
        )

        where = parsed.args.get("where")

        if where is not None:
            existing_condition = where.this

            def flatten_and(expression):
                """Split only top-level AND-connected predicates."""
                if isinstance(expression, exp.And):
                    return (
                        flatten_and(expression.this)
                        + flatten_and(expression.expression)
                    )
                return [expression]

            def is_customer_subquery_predicate(expression):
                """Identify a direct customer_id comparison with a subquery."""
                if not isinstance(expression, exp.EQ):
                    return False

                left = expression.this
                right = expression.expression

                left_is_customer_id = (
                    isinstance(left, exp.Column)
                    and left.name.lower() == "customer_id"
                )
                right_is_customer_id = (
                    isinstance(right, exp.Column)
                    and right.name.lower() == "customer_id"
                )

                left_has_subquery = any(
                    isinstance(node, exp.Subquery)
                    for node in left.walk()
                )
                right_has_subquery = any(
                    isinstance(node, exp.Subquery)
                    for node in right.walk()
                )

                return (
                    (left_is_customer_id and right_has_subquery)
                    or (right_is_customer_id and left_has_subquery)
                )

            predicates = flatten_and(existing_condition)
            safe_predicates = []

            for predicate in predicates:
                contains_subquery = any(
                    isinstance(node, exp.Subquery)
                    for node in predicate.walk()
                )

                if not contains_subquery:
                    safe_predicates.append(predicate.copy())
                elif is_customer_subquery_predicate(predicate):
                    # Remove only the customer-resolution subquery.
                    continue
                else:
                    # Do not silently change the meaning of a complex query.
                    raise ValueError(
                        "Cannot safely enforce customer ID in a query "
                        "containing a complex or unrelated subquery predicate."
                    )

            safe_predicates.append(customer_condition)

            combined_condition = safe_predicates[0]
            for predicate in safe_predicates[1:]:
                combined_condition = exp.and_(
                    combined_condition,
                    predicate,
                )

            parsed.set("where", exp.Where(this=combined_condition))

        else:
            parsed.set("where", exp.Where(this=customer_condition))

        return parsed.sql(dialect="postgres")