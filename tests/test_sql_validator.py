import pytest

from app.sql.sql_validator import SQLValidator


@pytest.fixture
def validator():
    return SQLValidator()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
    ],
)
def test_read_only_rejects_multiple_statements(validator, sql):
    assert validator.validate_read_only(sql) is False


def test_read_only_rejects_select_into(validator):
    assert validator.validate_read_only(
        "SELECT customer_id INTO copied_customers FROM orders"
    ) is False


def test_read_only_accepts_select(validator):
    assert validator.validate_read_only(
        "SELECT customer_id FROM orders"
    ) is True
