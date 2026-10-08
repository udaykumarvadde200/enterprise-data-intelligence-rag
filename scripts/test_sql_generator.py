from app.db.connection import DatabaseConnection
from app.db.schema import DatabaseSchemaInspector
from app.llm.ollama_client import OllamaClient
from app.sql.schema_formatter import SchemaFormatter
from app.sql.sql_generator import SQLGenerator


def main():
    # Database
    db = DatabaseConnection()

    # Get database schema
    inspector = DatabaseSchemaInspector(db)
    schema = inspector.get_schema()

    # Convert schema into LLM-readable text
    formatter = SchemaFormatter()
    formatted_schema = formatter.format_schema(schema)

    # Create Ollama client
    llm = OllamaClient()

    # Create SQL generator
    generator = SQLGenerator(llm)

    # User question
    question = "Show me the average salary of customers in Bangalore."    
    sql = generator.generate(
        question,
        formatted_schema,
    )

    print("USER QUESTION:")
    print(question)

    print("\nGENERATED SQL:")
    print(sql)


if __name__ == "__main__":
    main()