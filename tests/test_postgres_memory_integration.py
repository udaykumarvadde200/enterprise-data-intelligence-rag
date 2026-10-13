
import os
from uuid import uuid4

import pytest

from app.memory.long_term_memory import LongTermMemory


@pytest.fixture
def postgres_memory():
    """Run only when real PostgreSQL integration testing is enabled."""
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip(
            "Set RUN_POSTGRES_INTEGRATION=1 to enable PostgreSQL tests."
        )

    memory = LongTermMemory()
    memory.initialize()

    user_id = f"pytest-user-{uuid4().hex}"
    memory_key = f"pytest-memory-{uuid4().hex}"

    try:
        yield memory, user_id, memory_key
    finally:
        memory.delete(user_id, memory_key)


def test_postgres_memory_persists_and_updates(postgres_memory):
    memory, user_id, memory_key = postgres_memory
    marker = uuid4().hex

    memory.save(user_id, memory_key, f"{marker} original")

    results = memory.search(user_id, marker)
    matching = [
        item for item in results if item["memory_key"] == memory_key
    ]

    assert len(matching) == 1
    assert matching[0]["memory_value"] == f"{marker} original"

    memory.save(user_id, memory_key, f"{marker} updated")

    results = memory.search(user_id, marker)
    matching = [
        item for item in results if item["memory_key"] == memory_key
    ]

    assert len(matching) == 1
    assert matching[0]["memory_value"] == f"{marker} updated"


def test_postgres_memory_is_isolated_between_users(postgres_memory):
    memory, user_a, memory_key = postgres_memory
    user_b = f"pytest-user-{uuid4().hex}"
    marker = uuid4().hex

    try:
        memory.save(user_a, memory_key, f"{marker} private value")

        user_a_results = memory.search(user_a, marker)
        user_b_results = memory.search(user_b, marker)

        assert any(
            item["memory_key"] == memory_key
            for item in user_a_results
        )
        assert not any(
            item["memory_key"] == memory_key
            for item in user_b_results
        )
    finally:
        memory.delete(user_b, memory_key)


def test_postgres_memory_deletes_record(postgres_memory):
    memory, user_id, memory_key = postgres_memory
    marker = uuid4().hex

    memory.save(user_id, memory_key, f"{marker} temporary value")

    assert memory.delete(user_id, memory_key) is True
    assert not any(
        item["memory_key"] == memory_key
        for item in memory.search(user_id, marker)
    )
