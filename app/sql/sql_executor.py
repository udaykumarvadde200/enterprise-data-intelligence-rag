from app.db.connection import DatabaseConnection


class SQLExecutor:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    def execute(self, sql: str):
        with self.db.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)

                return cursor.fetchall()