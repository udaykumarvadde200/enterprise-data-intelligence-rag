from unittest.mock import MagicMock

import pytest

from app.sql.sql_executor import SQLExecutor
from app.sql.sql_validator import SQLValidator

SCHEMA = {
    "columns": [
        ("customers", "customer_id", "integer", "NO", None, 1),
        ("customers", "name", "text", "NO", None, 2),
        ("customers", "city", "text", "YES", None, 3),
        ("orders", "order_id", "integer", "NO", None, 1),
        ("orders", "customer_id", "integer", "NO", None, 2),
        ("orders", "total", "numeric", "YES", None, 3),
        ("orders", "created_at", "timestamp", "YES", None, 4),
    ]
}


@pytest.fixture
def validator():
    return SQLValidator()


# --- validator: write statements hidden inside a SELECT ------------------

@pytest.mark.parametrize(
    "sql",
    [
        "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
        "WITH u AS (UPDATE orders SET total = 0 RETURNING order_id) "
        "SELECT order_id FROM u",
        "WITH i AS (INSERT INTO orders (customer_id) VALUES (1) "
        "RETURNING order_id) SELECT order_id FROM i",
    ],
)
def test_read_only_rejects_data_modifying_cte(validator, sql):
    assert validator.validate_read_only(sql) is False


# --- validator: side-effecting / file-reaching functions -----------------

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(30)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT nextval('orders_order_id_seq')",
        "SELECT set_config('statement_timeout', '0', false)",
        "SELECT lo_import('/etc/passwd')",
        "SELECT * FROM pg_ls_dir('/')",
        "SELECT name FROM customers WHERE city = (SELECT pg_sleep(5)::text)",
    ],
)
def test_read_only_rejects_unrecognized_functions(validator, sql):
    assert validator.validate_read_only(sql) is False


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM orders",
        "SELECT SUM(total), AVG(total), MIN(total), MAX(total) FROM orders",
        "SELECT COALESCE(total, 0) FROM orders",
        "SELECT LOWER(name), UPPER(city) FROM customers",
        "SELECT ROUND(total, 2) FROM orders",
        "SELECT DATE_TRUNC('month', created_at) FROM orders",
        "SELECT EXTRACT(YEAR FROM created_at) FROM orders",
        "SELECT AGE(created_at) FROM orders",
        "SELECT c.name, o.total FROM customers c "
        "JOIN orders o ON o.customer_id = c.customer_id",
        "WITH t AS (SELECT customer_id FROM orders) SELECT * FROM t",
    ],
)
def test_read_only_still_accepts_ordinary_queries(validator, sql):
    assert validator.validate_read_only(sql) is True
    assert validator.validate_schema(sql, SCHEMA) is True


# --- validator: schema checks --------------------------------------------

def test_schema_rejects_invented_column(validator):
    assert validator.validate_schema(
        "SELECT fake_col FROM customers", SCHEMA
    ) is False


def test_schema_rejects_unknown_table(validator):
    assert validator.validate_schema("SELECT * FROM pg_user", SCHEMA) is False


# --- validator: model prose / fenced SQL never raises --------------------

@pytest.mark.parametrize(
    "text",
    [
        "I can't answer this question with the schema.",
        "I cannot answer this question.",
        "```sql\nSELECT * FROM customers\n```",
        "SELECT 'unterminated",
        "",
    ],
)
def test_invalid_text_is_rejected_without_raising(validator, text):
    assert validator.validate_syntax(text) is False
    assert validator.validate_schema(text, SCHEMA) is False
    assert validator.validate_read_only(text) is False


# --- executor -------------------------------------------------------------

def make_executor(rows):
    db = MagicMock()
    connection = db.connect.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchmany.return_value = rows
    return SQLExecutor(db), connection, cursor


def test_executor_runs_in_read_only_transaction_with_timeout():
    executor, connection, cursor = make_executor([(1,)])

    result = executor.execute("SELECT 1")

    assert connection.read_only is True
    executed = [call.args[0] for call in cursor.execute.call_args_list]
    assert "statement_timeout" in executed[0]
    assert executed[-1] == "SELECT 1"
    assert result == [(1,)]


def test_executor_caps_rows_and_requests_one_extra_to_detect_truncation():
    rows = [(i,) for i in range(SQLExecutor.MAX_ROWS + 1)]
    executor, _, cursor = make_executor(rows)

    result = executor.execute("SELECT * FROM orders")

    cursor.fetchmany.assert_called_once_with(SQLExecutor.MAX_ROWS + 1)
    assert len(result) == SQLExecutor.MAX_ROWS


def test_executor_returns_all_rows_below_cap():
    rows = [(i,) for i in range(3)]
    executor, _, _ = make_executor(rows)

    assert executor.execute("SELECT * FROM orders") == rows