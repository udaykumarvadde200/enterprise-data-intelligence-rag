from app.db.connection import DatabaseConnection
from app.db.schema import DatabaseSchemaInspector
from app.llm.ollama_client import OllamaClient
from app.sql.ambiguity_checker import AmbiguityChecker
from app.sql.customer_reference_extractor import CustomerReferenceExtractor
from app.sql.schema_formatter import SchemaFormatter
from app.sql.sql_executor import SQLExecutor
from app.sql.sql_generator import SQLGenerator
from app.sql.sql_validator import SQLValidator


class SQLPipeline:
    def __init__(self):
        self.db = DatabaseConnection()

        self.schema_inspector = DatabaseSchemaInspector(self.db)
        self.schema_formatter = SchemaFormatter()

        self.llm = OllamaClient()

        self.sql_generator = SQLGenerator(self.llm)
        self.sql_validator = SQLValidator()
        self.sql_executor = SQLExecutor(self.db)

        self.customer_reference_extractor = (
            CustomerReferenceExtractor(self.llm)
        )
        self.ambiguity_checker = AmbiguityChecker(self.db)

    def run(self, question: str):
        # 1. Check whether the question contains a customer reference
        customer_reference = self.customer_reference_extractor.extract(
            question
        )

        # 2. If a customer is mentioned, check whether the reference
        #    uniquely identifies a customer.
        if customer_reference != "NONE":
            ambiguity_result = self.ambiguity_checker.check_customer_name(
                customer_reference
            )

            if ambiguity_result["status"] == "ambiguous":
                matches = ambiguity_result["matches"]

                options = "\n".join(
                    f"- {name} — {city}"
                    for _customer_id, name, city in matches
                )

                return {
                    "status": "clarification_required",
                    "message": (
                        f"Multiple customers match "
                        f"'{customer_reference}'.\n"
                        f"Please specify which customer you mean:\n"
                        f"{options}"
                    ),
                }

            if ambiguity_result["status"] == "not_found":
                return {
                    "status": "not_found",
                    "message": (
                        f"No customer was found matching "
                        f"'{customer_reference}'."
                    ),
                }

        # 3. Get database schema
        schema = self.schema_inspector.get_schema()

        # 4. Format schema for the LLM
        formatted_schema = self.schema_formatter.format_schema(schema)

        # 5. Generate SQL
        sql = self.sql_generator.generate(
            question,
            formatted_schema,
        )

        # 6. Validate generated SQL
        if not self.sql_validator.validate_syntax(sql):
            raise ValueError("Generated SQL has invalid syntax.")

        if not self.sql_validator.validate_schema(sql, schema):
            raise ValueError(
                "Generated SQL references an invalid schema."
            )

        if not self.sql_validator.validate_read_only(sql):
            raise ValueError(
                "Only read-only SQL queries are allowed."
            )

        # 7. Execute validated SQL
        results = self.sql_executor.execute(sql)

        return {
            "status": "success",
            "sql": sql,
            "results": results,
        }