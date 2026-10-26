from datetime import date
from decimal import Decimal

import pytest

from app.writes.parser import (
    MissingAmountError,
    WriteRequestError,
    looks_like_amount,
    looks_like_purchase,
    parse_amount_only,
    parse_purchase,
)

TODAY = date(2026, 10, 25)


def parse(text):
    return parse_purchase(text, today=TODAY)


def test_basic_sentence():
    request = parse("Uday bought iPhone for 80k")

    assert request.customer_name == "Uday"
    assert request.product == "iPhone"
    assert request.amount == Decimal("80000.00")
    assert request.order_date == TODAY
    assert request.ignored_text == ""


def test_lowercase_input_gets_a_tidy_name():
    request = parse("uday bought iphone for 80k")

    assert request.customer_name == "Uday"
    assert request.product == "iphone"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Uday bought iPhone for 80000", "80000.00"),
        ("Uday bought iPhone for 80,000", "80000.00"),
        ("Uday bought iPhone for 1,20,000", "120000.00"),
        ("Uday bought iPhone for Rs. 35,000", "35000.00"),
        ("Uday bought iPhone for \u20b924900.50", "24900.50"),
        ("Uday bought iPhone for 80K", "80000.00"),
        ("Uday bought iPhone for 80 thousand", "80000.00"),
        ("Uday bought a laptop worth 1.5 lakh", "150000.00"),
        ("Uday bought a car for 8 lakhs", "800000.00"),
        ("Uday bought a house for 2 crore", "20000000.00"),
        ("Uday bought iPhone for 80000 rupees", "80000.00"),
    ],
)
def test_amount_formats(text, expected):
    assert parse(text).amount == Decimal(expected)


@pytest.mark.parametrize("verb", ["bought", "purchased", "ordered"])
def test_verbs(verb):
    assert parse(f"Priya {verb} a tablet for 30k").customer_name == "Priya"


def test_multi_word_name_and_article_removed():
    request = parse("Uday Kumar purchased a MacBook Air for 1,20,000")

    assert request.customer_name == "Uday Kumar"
    assert request.product == "MacBook Air"


def test_product_may_contain_the_word_for():
    request = parse("Sneha bought a gift for mom for 2k")

    assert request.product == "gift for mom"
    assert request.amount == Decimal("2000.00")


def test_dates():
    assert parse("Uday bought iPhone for 80k yesterday").order_date == date(
        2026, 10, 24
    )
    assert parse("Uday bought iPhone for 80k today").order_date == TODAY
    assert parse("Uday bought iPhone for 80k on 2026-10-01").order_date == date(
        2026, 10, 1
    )
    assert parse("Uday bought iPhone for 80k on 01/10/2026").order_date == date(
        2026, 10, 1
    )


def test_unrecognized_trailing_text_is_reported_not_silently_dropped():
    assert parse("Uday bought iPhone for 80k from Amazon").ignored_text == (
        "from Amazon"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Uday bought iPhone for 80k",
        "uday bought iphone",
        "I bought iPhone for 80k",
    ],
)
def test_purchase_statements_are_recognized(text):
    assert looks_like_purchase(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "hi",
        "How many orders are there?",
        "Show me all customers",
        "Who bought the iPhone for 80k?",
        "Show me what Uday bought for 80k",
        "Did Uday buy iPhone for 80k",
        "What is the project status?",
        "Uday's order was for 80k?",
        "",
    ],
)
def test_questions_and_other_text_are_not_purchases(text):
    assert looks_like_purchase(text) is False


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("Uday bought iPhone", "amount"),
        ("Uday bought rice for 80kg", "amount"),
        ("I bought iPhone for 80k", "customer's name"),
        ("Uday bought iPhone for 0", "greater than zero"),
        ("Uday bought iPhone for 80k on 2026-13-45", "date"),
        ("Uday bought iPhone for 80k on 2999-01-01", "future"),
        ("Uday bought " + "x" * 150 + " for 5", "too long"),
        ("Uday bought iPhone for 99999999999999 crore", "too large"),
    ],
)
def test_invalid_requests_give_a_clear_message(text, fragment):
    with pytest.raises(WriteRequestError) as error:
        parse(text)

    assert fragment in str(error.value)


def test_non_purchase_text_raises():
    with pytest.raises(WriteRequestError):
        parse("How many orders are there?")


def test_missing_amount_keeps_what_was_understood():
    with pytest.raises(MissingAmountError) as error:
        parse("uday bought iphone for")

    partial = error.value.partial
    assert partial.customer_name == "Uday"
    assert partial.product == "iphone"
    assert partial.order_date == TODAY
    assert "amount" in str(error.value)


def test_missing_amount_keeps_numbers_that_belong_to_the_product():
    with pytest.raises(MissingAmountError) as error:
        parse("Uday bought iPhone 15")

    assert error.value.partial.product == "iPhone 15"


def test_unreadable_amount_is_not_treated_as_missing():
    with pytest.raises(WriteRequestError) as error:
        parse("Uday bought rice for 80kg")

    assert not isinstance(error.value, MissingAmountError)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("80k", "80000.00"),
        ("1325465k", "1325465000.00"),
        ("1,20,000", "120000.00"),
        ("Rs. 5000", "5000.00"),
        ("\u20b92,500.50", "2500.50"),
        ("1.5 lakh", "150000.00"),
        (" 75000 ", "75000.00"),
    ],
)
def test_amount_only_messages(text, expected):
    assert looks_like_amount(text) is True
    assert parse_amount_only(text) == Decimal(expected)


@pytest.mark.parametrize(
    "text", ["hello", "80 apples", "", "uday bought iphone for 80k", "no", "2 or 3"]
)
def test_other_messages_are_not_amount_only(text):
    assert looks_like_amount(text) is False


def test_zero_amount_only_message_is_rejected():
    with pytest.raises(WriteRequestError, match="greater than zero"):
        parse_amount_only("0")