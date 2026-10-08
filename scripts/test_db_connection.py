from app.db.connection import DatabaseConnection


def main():
    db = DatabaseConnection()

    with db.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database();")
            database_name = cursor.fetchone()[0]

            print(f"Connected to database: {database_name}")


if __name__ == "__main__":
    main()

