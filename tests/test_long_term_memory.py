
from unittest.mock import MagicMock

import pytest

from app.memory.long_term_memory import LongTermMemory


@pytest.fixture
def memory():
    db = MagicMock()
    return LongTermMemory(db=db)


def test_save_rejects_empty_fields(memory):
    with pytest.raises(ValueError):
        memory.save("", "project", "RAG")

    with pytest.raises(ValueError):
        memory.save("user-1", " ", "RAG")

    with pytest.raises(ValueError):
        memory.save("user-1", "project", " ")


def test_save_uses_upsert(memory):
    memory.save("user-1", "project", "Advanced RAG")

    memory.db.connect.assert_called_once()
    connection = memory.db.connect.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value

    cursor.execute.assert_called_once()
    sql, parameters = cursor.execute.call_args.args

    assert "ON CONFLICT" in sql
    assert parameters == ("user-1", "project", "Advanced RAG")


def test_search_returns_matching_rows(memory):
    connection = memory.db.connect.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        ("project", "Building an Advanced RAG system", "2026-10-12")
    ]

    results = memory.search("user-1", "Advanced RAG")

    assert len(results) == 1
    assert results[0]["memory_key"] == "project"
    assert results[0]["memory_value"] == "Building an Advanced RAG system"

    sql, parameters = cursor.execute.call_args.args
    assert "user_id = %s" in sql
    assert parameters[0] == "user-1"
    assert parameters[-1] == 5


def test_search_validates_limit(memory):
    with pytest.raises(ValueError):
        memory.search("user-1", "project", limit=0)

    with pytest.raises(ValueError):
        memory.search("user-1", "project", limit=51)


def test_search_empty_input_returns_no_results(memory):
    assert memory.search("", "project") == []
    assert memory.search("user-1", "   ") == []
    memory.db.connect.assert_not_called()


def test_list_memories_is_scoped_to_user(memory):
    connection = memory.db.connect.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []

    assert memory.list_memories("user-2") == []

    sql, parameters = cursor.execute.call_args.args
    assert "WHERE user_id = %s" in sql
    assert parameters == ("user-2",)


def test_delete_is_scoped_to_user(memory):
    connection = memory.db.connect.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.rowcount = 1

    assert memory.delete("user-1", "project") is True

    sql, parameters = cursor.execute.call_args.args
    assert "WHERE user_id = %s AND memory_key = %s" in sql
    assert parameters == ("user-1", "project")


def test_delete_returns_false_when_memory_does_not_exist(memory):
    connection = memory.db.connect.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.rowcount = 0

    assert memory.delete("user-1", "missing-key") is False
