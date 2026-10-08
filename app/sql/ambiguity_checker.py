from app.db.connection import DatabaseConnection


class AmbiguityChecker:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    def find_customers_by_name(self, name: str):
        sql = """
        SELECT customer_id, name, city
        FROM customers
        WHERE name ILIKE %s;
        """

        with self.db.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (name,))
                return cursor.fetchall()

    def check_customer_name(self, name: str):
        matches = self.find_customers_by_name(name)

        if len(matches) == 0:
            return {
                "status": "not_found",
                "matches": [],
            }

        if len(matches) == 1:
            return {
                "status": "unique",
                "matches": matches,
            }

        return {
            "status": "ambiguous",
            "matches": matches,
        }