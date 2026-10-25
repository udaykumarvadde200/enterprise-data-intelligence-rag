import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.graph.workflow import EnterpriseWorkflow
from app.sql.sql_executor import SQLExecutor

format_result = EnterpriseWorkflow._format_sql_result

ORDER_COLUMNS = ["order_id", "customer_id", "product", "amount", "order_date"]
ORDER_ROW = (1, 1, "iPhone 15", Decimal("65000.00"), datetime.date(2026, 9, 10))


def test_table_shows_column_names_and_plain_values():
    answer = format_result(
        {"sql": "SELECT * FROM orders", "results": [ORDER_ROW]},
        ORDER_COLUMNS,
    )

    assert answer.startswith("Query completed successfully.")
    assert "SQL:\nSELECT * FROM orders" in answer
    assert "| order_id | customer_id | product | amount | order_date |" in answer
    assert "| --- | --- | --- | --- | --- |" in answer
    assert "| 1 | 1 | iPhone 15 | 65000.00 | 2026-09-10 |" in answer
    assert "Decimal(" not in answer
    assert "datetime.date" not in answer


def test_null_values_and_pipes_are_rendered_safely():
    answer = format_result(
        {"sql": "SELECT a, b FROM t", "results": [(None, "x|y\nz")]},
        ["a", "b"],
    )

    assert "| NULL | x\\|y z |" in answer


def test_count_query_gets_a_readable_header():
    answer = format_result(
        {"sql": "SELECT COUNT(*) FROM customers", "results": [(10,)]},
        ["count"],
    )

    assert "| count |" in answer
    assert "| 10 |" in answer


def test_without_columns_the_original_format_is_kept():
    answer = format_result({"sql": "SELECT 1", "results": [(1,), (2,)]})

    assert answer == (
        "Query completed successfully.\n\n"
        "SQL:\nSELECT 1\n\n"
        "Results:\n(1,)\n(2,)"
    )


def test_row_width_mismatch_falls_back_to_original_format():
    answer = format_result(
        {"sql": "SELECT 1", "results": [(1, 2)]},
        ["only_one"],
    )

    assert "Results:\n(1, 2)" in answer
    assert "---" not in answer


def test_empty_result_message_is_unchanged():
    assert format_result({"sql": "SELECT 1", "results": []}, ["a"]) == (
        "The query completed successfully, but no records matched."
    )


def test_executor_columns_helper_is_tolerant():
    ready = SimpleNamespace(sql_executor=SimpleNamespace(last_columns=["a", "b"]))

    assert EnterpriseWorkflow._executor_columns(ready) == ["a", "b"]
    assert EnterpriseWorkflow._executor_columns(MagicMock()) == []
    assert EnterpriseWorkflow._executor_columns(object()) == []


def test_executor_records_column_names_and_resets_them():
    db = MagicMock()
    connection = db.connect.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchmany.return_value = [(1, "a")]
    executor = SQLExecutor(db)

    cursor.description = [SimpleNamespace(name="id"), SimpleNamespace(name="x")]
    executor.execute("SELECT id, x FROM t")
    assert executor.last_columns == ["id", "x"]

    cursor.description = None
    executor.execute("SELECT 1")
    assert executor.last_columns == []
