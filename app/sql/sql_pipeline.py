from app.db.connection import DatabaseConnection
from app.db.schema import DatabaseSchemaInspector
from app.sql.schema_formatter import SchemaFormatter
from app.llm.ollama_client import OllamaClient
from app.sql.sql_generator import SQLGenerator
from app.sql.sql_validator import SQLValidator
from app.sql.sql_executor import SQLExecutor


class SQLPipeline:
    def __init__(self):
        self.db = DatabaseConnection()

        self.schema_inspector = DatabaseSchemaInspector(self.db)
        self.schema_formatter = SchemaFormatter()

        self.llm = OllamaClient()
        self.sql_generator = SQLGenerator(self.llm)

        self.sql_validator = SQLValidator()
        self.sql_executor = SQLExecutor(self.db)

    def run(self, question: str):
        # 1. Get database schema
        schema = self.schema_inspector.get_schema()

        # 2. Format schema for the LLM
        formatted_schema = self.schema_formatter.format_schema(schema)

        # 3. Generate SQL
        sql = self.sql_generator.generate(
            question,
            formatted_schema,
        )

        # 4. Validate generated SQL
        if not self.sql_validator.validate_syntax(sql):
            raise ValueError("Generated SQL has invalid syntax.")

        if not self.sql_validator.validate_schema(sql, schema):
            raise ValueError("Generated SQL references an invalid schema.")

        if not self.sql_validator.validate_read_only(sql):
            raise ValueError("Only read-only SQL queries are allowed.")

        # 5. Execute validated SQL
        results = self.sql_executor.execute(sql)

        return sql, results