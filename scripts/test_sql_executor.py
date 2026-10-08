from app.db.connection import DatabaseConnection
from app.sql.sql_executor import SQLExecutor


def main():
    db = DatabaseConnection()
    executor = SQLExecutor(db)

    sql = """
    SELECT name, city
    FROM customers
    WHERE city = 'Bangalore';
    """

    results = executor.execute(sql)

    print("QUERY RESULT:")

    for row in results:
        print(row)


if __name__ == "__main__":
    main()