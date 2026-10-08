from app.db.connection import DatabaseConnection
from app.db.schema import DatabaseSchemaInspector
from app.sql.schema_formatter import SchemaFormatter


def main():
    db = DatabaseConnection()

    inspector = DatabaseSchemaInspector(db)
    schema = inspector.get_schema()

    formatter = SchemaFormatter()
    formatted_schema = formatter.format_schema(schema)

    print(formatted_schema)


if __name__ == "__main__":
    main()