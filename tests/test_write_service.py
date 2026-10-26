from datetime import date
from decimal import Decimal

import pytest

from app.writes.parser import MissingAmountError, PurchaseRequest, parse_purchase
from app.writes.service import PENDING_TTL_SECONDS, PurchaseWriter

TODAY = date(2026, 10, 25)


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.db.executed.append((sql, params))

        if self.db.fail_on and self.db.fail_on in sql:
            raise RuntimeError("boom")

        if sql.lstrip().upper().startswith("SELECT"):
            self._rows = list(self.db.customers)
        elif "INSERT INTO customers" in sql:
            self.db.next_customer_id += 1
            self._rows = [(self.db.next_customer_id,)]
        elif "INSERT INTO orders" in sql:
            self.db.next_order_id += 1
            self._rows = [(self.db.next_order_id,)]

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


class FakeConnection:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *args):
        if exc_type is None:
            self.db.commits += 1
        else:
            self.db.rollbacks += 1
        return False

    def cursor(self):
        return FakeCursor(self.db)


class FakeDB:
    def __init__(self, customers=(), fail_on=None):
        self.customers = list(customers)
        self.fail_on = fail_on
        self.executed = []
        self.commits = 0
        self.rollbacks = 0
        self.next_customer_id = 100
        self.next_order_id = 500

    def connect(self):
        return FakeConnection(self)

    def inserts(self):
        return [
            (sql, params)
            for sql, params in self.executed
            if sql.lstrip().upper().startswith("INSERT")
        ]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def request(text="Uday bought iPhone for 80k"):
    return parse_purchase(text, today=TODAY)


def test_single_match_asks_to_confirm_and_writes_nothing_yet():
    db = FakeDB([(7, "Uday Kumar", "Hyderabad")])
    reply = PurchaseWriter(db).prepare(request())

    assert reply.status == "confirmation_required"
    assert "Uday Kumar (id 7)" in reply.message
    assert "80,000.00" in reply.message
    assert "2026-10-25" in reply.message
    assert "Reply **yes**" in reply.message
    assert db.inserts() == []


def test_yes_saves_exactly_one_order_for_the_matched_customer():
    db = FakeDB([(7, "Uday Kumar", "Hyderabad")])
    writer = PurchaseWriter(db)
    pending = writer.prepare(request()).pending

    reply = writer.respond(pending, "yes")

    assert reply.status == "saved"
    assert reply.pending is None
    assert reply.details["order_id"] == 501
    assert "Saved order **#501**" in reply.message

    inserts = db.inserts()
    assert len(inserts) == 1
    assert "INSERT INTO orders" in inserts[0][0]
    assert inserts[0][1] == (7, "iPhone", Decimal("80000.00"), TODAY)
    assert db.commits >= 1


def test_unknown_customer_is_created_with_the_order_in_one_transaction():
    db = FakeDB([])
    writer = PurchaseWriter(db)
    prepared = writer.prepare(request())

    assert "a new one will be created" in prepared.message
    assert db.inserts() == []

    reply = writer.respond(prepared.pending, "yes")

    inserts = db.inserts()
    assert [("customers" in sql) for sql, _ in inserts] == [True, False]
    assert inserts[0][1] == ("Uday",)
    assert inserts[1][1][0] == 101  # the new customer's id
    assert reply.details["customer_created"] is True
    assert "Created customer **Uday** (id 101)" in reply.message


def test_names_match_whole_words_only():
    db = FakeDB([(1, "Sudayan Raj", "Chennai"), (2, "Uday Kumar", None)])

    matches = PurchaseWriter(db).find_customers("Uday")

    assert [row[1] for row in matches] == ["Uday Kumar"]


def test_exact_name_wins_over_partial_matches():
    db = FakeDB([(1, "Uday", None), (2, "Uday Kumar", None)])

    matches = PurchaseWriter(db).find_customers("Uday")

    assert [row[0] for row in matches] == [1]


def test_several_matches_ask_which_one_and_accept_a_number_or_a_city():
    rows = [(1, "Rahul Verma", "Hyderabad"), (2, "Rahul Mehta", "Pune")]
    writer = PurchaseWriter(FakeDB(rows))
    asked = writer.prepare(request("Rahul bought a laptop for 1.5 lakh"))

    assert asked.status == "clarification_required"
    assert "1. Rahul Verma - Hyderabad (id 1)" in asked.message
    assert "2. Rahul Mehta - Pune (id 2)" in asked.message

    by_number = writer.respond(asked.pending, "2")
    assert by_number.status == "confirmation_required"
    assert "Rahul Mehta (id 2)" in by_number.message

    by_city = writer.respond(asked.pending, "hyderabad")
    assert "Rahul Verma (id 1)" in by_city.message


def test_bad_choices_are_not_understood():
    rows = [(1, "Rahul Verma", "Hyderabad"), (2, "Rahul Mehta", "Pune")]
    writer = PurchaseWriter(FakeDB(rows))
    pending = writer.prepare(request("Rahul bought a laptop for 1k")).pending

    for text in ["3", "0", "banana", "yes"]:
        assert writer.classify(pending, text) is None
        assert writer.respond(pending, text) is None


@pytest.mark.parametrize("text", ["no", "No.", "cancel", "stop", "Never mind"])
def test_cancel_writes_nothing(text):
    db = FakeDB([(7, "Uday Kumar", None)])
    writer = PurchaseWriter(db)
    pending = writer.prepare(request()).pending

    reply = writer.respond(pending, text)

    assert reply.status == "cancelled"
    assert reply.pending is None
    assert db.inserts() == []


@pytest.mark.parametrize("text", ["banana", "what about tomorrow", "maybe"])
def test_other_replies_are_not_confirmation(text):
    db = FakeDB([(7, "Uday Kumar", None)])
    writer = PurchaseWriter(db)
    pending = writer.prepare(request()).pending

    assert writer.classify(pending, text) is None
    assert writer.respond(pending, text) is None
    assert db.inserts() == []


@pytest.mark.parametrize("text", ["yes", "Yes!", "ok", "Sure", "go ahead"])
def test_confirmation_words(text):
    writer = PurchaseWriter(FakeDB([(7, "Uday Kumar", None)]))
    pending = writer.prepare(request()).pending

    assert writer.classify(pending, text) == "confirm"


def test_pending_requests_expire():
    clock = Clock()
    writer = PurchaseWriter(FakeDB([(7, "Uday Kumar", None)]), clock=clock)
    pending = writer.prepare(request()).pending

    assert writer.is_expired(pending) is False
    clock.now = PENDING_TTL_SECONDS + 1
    assert writer.is_expired(pending) is True


def test_database_failure_reports_error_and_keeps_nothing_pending():
    db = FakeDB([], fail_on="INSERT INTO orders")
    writer = PurchaseWriter(db)
    pending = writer.prepare(request()).pending

    reply = writer.respond(pending, "yes")

    assert reply.status == "error"
    assert reply.pending is None
    assert "nothing was changed" in reply.message
    assert db.rollbacks == 1


def test_user_text_is_only_ever_sent_as_parameters():
    db = FakeDB([(7, "Uday Kumar", None)])
    writer = PurchaseWriter(db)
    hostile = PurchaseRequest(
        "Uday",
        "kit'); DROP TABLE orders;--",
        Decimal("5.00"),
        TODAY,
    )
    pending = writer.prepare(hostile).pending
    writer.respond(pending, "yes")

    assert all("DROP" not in sql for sql, _ in db.executed)
    assert db.inserts()[0][1][1] == "kit'); DROP TABLE orders;--"


def partial_of(text):
    with pytest.raises(MissingAmountError) as error:
        parse_purchase(text, today=TODAY)

    return error.value.partial


def test_missing_amount_is_remembered_and_asked_for():
    db = FakeDB([(7, "Uday Kumar", None)])
    writer = PurchaseWriter(db)

    reply = writer.prepare_partial(partial_of("uday bought iphone for"))

    assert reply.status == "amount_required"
    assert reply.pending.stage == "need_amount"
    assert "How much was it?" in reply.message
    assert "iphone" in reply.message
    assert db.executed == []


def test_an_amount_reply_completes_the_order_then_asks_to_confirm():
    db = FakeDB([(7, "Uday Kumar", None)])
    writer = PurchaseWriter(db)
    pending = writer.prepare_partial(partial_of("uday bought iphone for")).pending

    assert writer.classify(pending, "80k") == "amount"

    reply = writer.respond(pending, "80k")

    assert reply.status == "confirmation_required"
    assert "Uday Kumar (id 7)" in reply.message
    assert "80,000.00" in reply.message
    assert db.inserts() == []

    saved = writer.respond(reply.pending, "yes")
    assert saved.status == "saved"
    assert db.inserts()[0][1] == (7, "iphone", Decimal("80000.00"), TODAY)


@pytest.mark.parametrize("text", ["banana", "show me orders", "5 apples", "yes"])
def test_other_replies_do_not_complete_a_missing_amount(text):
    writer = PurchaseWriter(FakeDB([]))
    pending = writer.prepare_partial(partial_of("uday bought iphone for")).pending

    assert writer.classify(pending, text) is None
    assert writer.respond(pending, text) is None


def test_zero_amount_keeps_asking():
    writer = PurchaseWriter(FakeDB([]))
    pending = writer.prepare_partial(partial_of("uday bought iphone for")).pending

    reply = writer.respond(pending, "0")

    assert reply.status == "invalid_request"
    assert "greater than zero" in reply.message
    assert reply.pending is pending


def test_cancel_while_waiting_for_an_amount():
    db = FakeDB([])
    writer = PurchaseWriter(db)
    pending = writer.prepare_partial(partial_of("uday bought iphone for")).pending

    reply = writer.respond(pending, "no")

    assert reply.status == "cancelled"
    assert reply.pending is None
    assert db.executed == []