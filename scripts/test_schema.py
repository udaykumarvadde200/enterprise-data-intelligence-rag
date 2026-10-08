from pprint import pprint

from app.db.connection import DatabaseConnection
from app.db.schema import DatabaseSchemaInspector


def main():
    db = DatabaseConnection()
    inspector = DatabaseSchemaInspector(db)

    schema = inspector.get_schema()

    pprint(schema)


if __name__ == "__main__":
    main()