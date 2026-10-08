from app.db.connection import DatabaseConnection
from app.db.schema import DatabaseSchemaInspector
from app.sql.sql_validator import SQLValidator


def main():
    db = DatabaseConnection()

    inspector = DatabaseSchemaInspector(db)
    schema = inspector.get_schema()

    validator = SQLValidator()

    valid_sql = """
    SELECT name
    FROM customers
    WHERE city = 'Bangalore';
    """

    invalid_sql = """
    SELECT AVG(amount)
    FROM orders
    WHERE city = 'Bangalore';
    """

    join_sql = """
    SELECT c.name
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE c.city = 'Bangalore'
      AND o.amount > 50000;
    """

    print("Test 1 - Valid SQL:")
    print(validator.validate_schema(valid_sql, schema))

    print("\nTest 2 - Invalid column:")
    print(validator.validate_schema(invalid_sql, schema))

    print("\nTest 3 - JOIN:")
    print(validator.validate_schema(join_sql, schema))
    
    select_sql = """
    SELECT name
    FROM customers;
    """

    delete_sql = """
    DELETE FROM customers
    WHERE customer_id = 1;
    """

    update_sql = """
    UPDATE customers
    SET city = 'Hyderabad'
    WHERE customer_id = 1;
    """

    print("\nTest 4 - SELECT:")
    print(validator.validate_read_only(select_sql))

    print("\nTest 5 - DELETE:")
    print(validator.validate_read_only(delete_sql))

    print("\nTest 6 - UPDATE:")
    print(validator.validate_read_only(update_sql))


if __name__ == "__main__":
    main()