import logging

from app.db.connection import DatabaseConnection


logger = logging.getLogger(__name__)


class SQLExecutor:
    MAX_ROWS = 500
    STATEMENT_TIMEOUT_MS = 10_000

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def execute(self, sql: str):
        with self.db.connect() as connection:
            # Defense in depth: even if a statement slipped past validation,
            # PostgreSQL itself rejects writes inside a read-only transaction.
            connection.read_only = True

            with connection.cursor() as cursor:
                # Bound the runtime of this transaction only.
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(self.STATEMENT_TIMEOUT_MS),),
                )
                cursor.execute(sql)

                rows = cursor.fetchmany(self.MAX_ROWS + 1)

        if len(rows) > self.MAX_ROWS:
            logger.warning(
                "SQL result truncated to %d rows.", self.MAX_ROWS
            )
            rows = rows[: self.MAX_ROWS]

        return rows