import pytest
import sqlglot

from app.sql.sql_generator import SQLGenerator


@pytest.fixture
def generator():
    return SQLGenerator(llm=None)


def test_resolved_customer_removes_ambiguous_name_subquery(generator):
    sql = """
        SELECT * FROM orders
        WHERE customer_id = (
            SELECT customer_id
            FROM customers
            WHERE name = 'John Smith'
        )
    """

    result = generator._enforce_customer_id(sql, 4)
    parsed = sqlglot.parse_one(result, dialect="postgres")

    assert "orders.customer_id = 4" in result
    assert "John Smith" not in result
    assert parsed.args["where"] is not None


def test_resolved_customer_supports_orders_alias(generator):
    sql = "SELECT * FROM orders AS o WHERE o.amount > 1000"

    result = generator._enforce_customer_id(sql, 4)

    assert "o.customer_id = 4" in result


def test_rejects_query_without_outer_orders_table(generator):
    sql = "SELECT * FROM customers"

    with pytest.raises(ValueError, match="orders table"):
        generator._enforce_customer_id(sql, 4)


def test_rejects_non_select_query(generator):
    sql = "DELETE FROM orders"

    with pytest.raises(ValueError, match="SELECT"):
        generator._enforce_customer_id(sql, 4)


def test_preserves_existing_non_subquery_filter(generator):
    sql = """
        SELECT *
        FROM orders
        WHERE amount > 1000
    """

    result = generator._enforce_customer_id(sql, 4)

    assert "amount > 1000" in result
    assert "orders.customer_id = 4" in result

def test_preserves_date_filter_when_removing_customer_subquery(generator):
    sql = """
        SELECT *
        FROM orders
        WHERE order_date >= '2026-01-01'
          AND customer_id = (
              SELECT customer_id
              FROM customers
              WHERE name = 'John Smith'
          )
    """

    result = generator._enforce_customer_id(sql, 4)

    assert "2026-01-01" in result
    assert "John Smith" not in result
    assert "customer_id = 4" in result


def test_rejects_unsafe_or_condition_with_subquery(generator):
    sql = """
        SELECT *
        FROM orders
        WHERE customer_id = (
            SELECT customer_id
            FROM customers
            WHERE name = 'John Smith'
        )
        OR amount > 1000
    """

    with pytest.raises(ValueError, match="safely enforce"):
        generator._enforce_customer_id(sql, 4)