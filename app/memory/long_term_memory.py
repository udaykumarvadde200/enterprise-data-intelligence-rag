
from typing import Any

from app.db.connection import DatabaseConnection


class LongTermMemory:
    """Persistent, user-scoped memory stored in PostgreSQL."""

    def __init__(self, db: DatabaseConnection | None = None):
        self.db = db or DatabaseConnection()

    def initialize(self) -> None:
        """Create the memory table if it does not already exist."""
        query = """
            CREATE TABLE IF NOT EXISTS long_term_memories (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                memory_key VARCHAR(200) NOT NULL,
                memory_value TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (user_id, memory_key)
            );

            CREATE INDEX IF NOT EXISTS idx_long_term_memories_user
                ON long_term_memories (user_id);
        """

        with self.db.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)

    def save(
        self,
        user_id: str,
        memory_key: str,
        memory_value: str,
    ) -> None:
        """Insert a memory or update that user's existing memory key."""
        user_id = user_id.strip()
        memory_key = memory_key.strip()
        memory_value = memory_value.strip()

        if not user_id or not memory_key or not memory_value:
            raise ValueError(
                "user_id, memory_key, and memory_value must not be empty."
            )

        query = """
            INSERT INTO long_term_memories (
                user_id, memory_key, memory_value
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, memory_key)
            DO UPDATE SET
                memory_value = EXCLUDED.memory_value,
                updated_at = NOW();
        """

        with self.db.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (user_id, memory_key, memory_value),
                )

    def search(
        self,
        user_id: str,
        query_text: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search only the specified user's memories."""
        user_id = user_id.strip()
        terms = query_text.split()

        if not user_id or not terms:
            return []

        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50.")

        conditions = " OR ".join(
            "(memory_key ILIKE %s OR memory_value ILIKE %s)"
            for _ in terms
        )
        parameters: list[Any] = [user_id]

        for term in terms:
            pattern = f"%{term}%"
            parameters.extend([pattern, pattern])

        parameters.append(limit)

        sql = f"""
            SELECT memory_key, memory_value, updated_at
            FROM long_term_memories
            WHERE user_id = %s AND ({conditions})
            ORDER BY updated_at DESC
            LIMIT %s;
        """

        with self.db.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, parameters)
                rows = cursor.fetchall()

        return [
            {
                "memory_key": row[0],
                "memory_value": row[1],
                "updated_at": row[2],
            }
            for row in rows
        ]

    def list_memories(self, user_id: str) -> list[dict[str, Any]]:
        """List memories belonging to one user."""
        user_id = user_id.strip()

        if not user_id:
            raise ValueError("user_id must not be empty.")

        query = """
            SELECT memory_key, memory_value, created_at, updated_at
            FROM long_term_memories
            WHERE user_id = %s
            ORDER BY updated_at DESC;
        """

        with self.db.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (user_id,))
                rows = cursor.fetchall()

        return [
            {
                "memory_key": row[0],
                "memory_value": row[1],
                "created_at": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]

    def delete(self, user_id: str, memory_key: str) -> bool:
        """Delete one memory owned by the specified user."""
        user_id = user_id.strip()
        memory_key = memory_key.strip()

        if not user_id or not memory_key:
            raise ValueError("user_id and memory_key must not be empty.")

        query = """
            DELETE FROM long_term_memories
            WHERE user_id = %s AND memory_key = %s;
        """

        with self.db.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (user_id, memory_key))
                return cursor.rowcount > 0
