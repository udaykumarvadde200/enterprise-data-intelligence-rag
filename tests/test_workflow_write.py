from unittest.mock import MagicMock, patch

import pytest

from app.graph.workflow import EnterpriseWorkflow
from app.writes.service import PurchaseWriter


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

        if sql.lstrip().upper().startswith("SELECT"):
            self._rows = list(self.db.customers)
        elif "INSERT INTO customers" in sql:
            self._rows = [(900,)]
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

    def __exit__(self, *args):
        return False

    def cursor(self):
        return FakeCursor(self.db)


class FakeDB:
    def __init__(self, customers):
        self.customers = list(customers)
        self.executed = []
        self.next_order_id = 40

    def connect(self):
        return FakeConnection(self)

    def inserts(self):
        return [
            (sql, params)
            for sql, params in self.executed
            if sql.lstrip().upper().startswith("INSERT")
        ]


def make_workflow(customers=((7, "Uday Kumar", "Hyderabad"),)):
    db = FakeDB(customers)
    memory = MagicMock()
    memory.search.return_value = []

    workflow = EnterpriseWorkflow(
        long_term_memory=memory,
        purchase_writer=PurchaseWriter(db),
    )
    # Follow-up rewriting must never see data-entry messages.
    workflow.query_rewriter = MagicMock()
    workflow.query_rewriter.rewrite.side_effect = lambda question, history: question

    return workflow, db


def test_sentence_asks_first_then_yes_saves_one_order():
    workflow, db = make_workflow()

    first = workflow.ask("Uday bought iPhone for 80k", thread_id="t1")

    assert first["result"]["route"] == "write"
    assert first["result"]["status"] == "confirmation_required"
    assert "Uday Kumar (id 7)" in first["answer"]
    assert db.inserts() == []

    second = workflow.ask("yes", thread_id="t1")

    assert second["result"]["status"] == "saved"
    assert second["result"]["order_id"] == 41
    assert len(db.inserts()) == 1
    assert workflow.pending_writes == {}


def test_a_second_yes_does_not_save_again():
    workflow, db = make_workflow()
    workflow.ask("Uday bought iPhone for 80k", thread_id="t1")
    workflow.ask("yes", thread_id="t1")

    again = workflow.ask("yes", thread_id="t1")

    assert again["result"]["status"] == "nothing_pending"
    assert len(db.inserts()) == 1


def test_no_cancels_and_saves_nothing():
    workflow, db = make_workflow()
    workflow.ask("Uday bought iPhone for 80k", thread_id="t1")

    reply = workflow.ask("no", thread_id="t1")

    assert reply["result"]["status"] == "cancelled"
    assert db.inserts() == []
    assert workflow.pending_writes == {}


def test_ambiguous_customer_flow_through_the_graph():
    workflow, db = make_workflow(
        [(1, "Rahul Verma", "Hyderabad"), (2, "Rahul Mehta", "Pune")]
    )

    asked = workflow.ask("Rahul bought a laptop for 55k", thread_id="t2")
    assert asked["result"]["status"] == "clarification_required"

    chosen = workflow.ask("2", thread_id="t2")
    assert chosen["result"]["status"] == "confirmation_required"
    assert "Rahul Mehta (id 2)" in chosen["answer"]
    assert db.inserts() == []

    saved = workflow.ask("yes", thread_id="t2")
    assert saved["result"]["status"] == "saved"
    assert db.inserts()[0][1][0] == 2


def test_unknown_customer_is_created_after_confirmation():
    workflow, db = make_workflow(customers=[])

    asked = workflow.ask("Meera bought a tablet for 30k", thread_id="t3")
    assert "a new one will be created" in asked["answer"]

    saved = workflow.ask("yes", thread_id="t3")
    assert saved["result"]["customer_created"] is True
    assert len(db.inserts()) == 2


def test_other_messages_abandon_the_pending_request():
    workflow, db = make_workflow()
    workflow.ask("Uday bought iPhone for 80k", thread_id="t1")

    hello = {"question": "hi", "thread_id": "t1", "user_id": "local-user"}
    assert workflow.route_node(hello)["route"] == "general"
    assert workflow.pending_writes == {}

    yes = {"question": "yes", "thread_id": "t1", "user_id": "local-user"}
    assert workflow.write_node(yes)["result"]["status"] == "nothing_pending"
    assert db.inserts() == []


def test_pending_requests_are_separate_per_user_and_thread():
    workflow, db = make_workflow()
    workflow.ask("Uday bought iPhone for 80k", thread_id="t1", user_id="alice")

    other_user = {"question": "yes", "thread_id": "t1", "user_id": "bob"}
    other_thread = {"question": "yes", "thread_id": "t9", "user_id": "alice"}

    for state in (other_user, other_thread):
        assert workflow.write_node(state)["result"]["status"] == "nothing_pending"

    assert ("alice", "t1") in workflow.pending_writes
    assert db.inserts() == []


def test_write_messages_bypass_the_query_rewriter():
    workflow, _ = make_workflow()

    workflow.ask("Uday bought iPhone for 80k", thread_id="t1")
    workflow.ask("yes", thread_id="t1")

    workflow.query_rewriter.rewrite.assert_not_called()


def test_incomplete_sentence_asks_for_the_amount_and_saves_nothing():
    workflow, db = make_workflow()

    reply = workflow.ask("Uday bought iPhone", thread_id="t1")

    assert reply["result"]["route"] == "write"
    assert reply["result"]["status"] == "amount_required"
    assert "How much was it?" in reply["answer"]
    assert db.inserts() == []


def test_replying_with_just_the_amount_completes_the_order():
    # The exact sequence from a real session.
    workflow, db = make_workflow()

    asked = workflow.ask("uday bought iphone for", thread_id="t1")
    assert asked["result"]["status"] == "amount_required"

    confirm = workflow.ask("1325465k", thread_id="t1")
    assert confirm["result"]["route"] == "write"
    assert confirm["result"]["status"] == "confirmation_required"
    assert "1,325,465,000.00" in confirm["answer"]
    assert db.inserts() == []

    saved = workflow.ask("yes", thread_id="t1")
    assert saved["result"]["status"] == "saved"
    assert len(db.inserts()) == 1


def test_amount_reply_is_not_a_database_question_when_nothing_is_waiting():
    workflow, _ = make_workflow()

    assert workflow._is_write_turn("80k", ("local-user", "t1")) is False


def test_cancel_and_other_messages_while_waiting_for_an_amount():
    workflow, db = make_workflow()
    workflow.ask("uday bought iphone for", thread_id="t1")

    assert workflow.ask("no", thread_id="t1")["result"]["status"] == "cancelled"

    workflow.ask("uday bought iphone for", thread_id="t2")
    hello = {"question": "hi", "thread_id": "t2", "user_id": "local-user"}
    assert workflow.route_node(hello)["route"] == "general"
    assert workflow.pending_writes == {}
    assert db.inserts() == []


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "off", "no"])
def test_writes_can_be_switched_off(monkeypatch, value):
    monkeypatch.setenv("ALLOW_WRITES", value)
    workflow, db = make_workflow()

    reply = workflow.ask("Uday bought iPhone for 80k", thread_id="t1")

    assert reply["result"]["status"] == "disabled"
    assert "ALLOW_WRITES" in reply["answer"]
    assert db.executed == []


def test_writes_are_on_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_WRITES", raising=False)

    assert EnterpriseWorkflow.writes_enabled() is True


@pytest.mark.parametrize(
    "question",
    [
        "How many orders are there?",
        "Show me all customers",
        "Who bought the iPhone for 80k?",
        "What is the project status and target completion date?",
    ],
)
def test_ordinary_questions_are_not_treated_as_data_entry(question):
    workflow, _ = make_workflow()
    state = {"question": question, "thread_id": "t1", "user_id": "local-user"}

    assert workflow._is_write_turn(question, ("local-user", "t1")) is False


def test_yes_after_a_cancel_gets_a_clear_message_and_saves_nothing():
    # The exact sequence from a real session: sentence, "no", then "yes".
    workflow, db = make_workflow(customers=[])
    workflow.ask("uday bought iphone for 80k", thread_id="t1")
    workflow.ask("no", thread_id="t1")

    reply = workflow.ask("yes", thread_id="t1")

    assert reply["result"]["route"] == "write"
    assert reply["result"]["status"] == "nothing_pending"
    assert "nothing waiting to be saved" in reply["answer"]
    assert db.inserts() == []


def test_yes_with_no_earlier_request_is_answered_without_the_model():
    workflow, db = make_workflow()

    reply = workflow.ask("yes", thread_id="fresh")

    assert reply["result"]["status"] == "nothing_pending"
    assert db.executed == []


@pytest.mark.parametrize("text", ["ok thanks", "sure", "okay", "yes it is raining"])
def test_weaker_or_longer_replies_stay_normal_conversation(text):
    workflow, _ = make_workflow()

    assert workflow._is_write_turn(text, ("local-user", "t1")) is False


def test_bare_yes_is_normal_conversation_when_writes_are_off(monkeypatch):
    monkeypatch.setenv("ALLOW_WRITES", "false")
    workflow, _ = make_workflow()

    assert workflow._is_write_turn("yes", ("local-user", "t1")) is False


PREVIOUS_TOPIC = [
    {
        "question": "Show orders for Uday",
        "answer": "No customer was found matching 'Uday'.",
    }
]


@pytest.mark.parametrize(
    "text", ["hi", "oh ok", "Thanks!", "ok", "Okay thanks", "bye", "Good morning"]
)
def test_small_talk_is_not_rewritten_into_a_database_question(text):
    # A real session: after talking about Uday, "hi" became a SQL query.
    workflow, _ = make_workflow()
    state = {
        "question": text,
        "thread_id": "t1",
        "user_id": "local-user",
        "conversation_history": PREVIOUS_TOPIC,
    }

    result = workflow.route_node(state)

    assert result["route"] == "general"
    assert result["rewritten_question"] == text
    workflow.query_rewriter.rewrite.assert_not_called()


def test_genuine_follow_ups_are_still_rewritten():
    workflow, _ = make_workflow()
    workflow.query_rewriter.rewrite.side_effect = (
        lambda question, history: "How many orders did Uday place?"
    )
    workflow._get_sql_pipeline = MagicMock(
        return_value=MagicMock(conversation_state=MagicMock(status=None))
    )
    state = {
        "question": "how many orders did he place",
        "thread_id": "t1",
        "user_id": "local-user",
        "conversation_history": PREVIOUS_TOPIC,
    }

    result = workflow.route_node(state)

    workflow.query_rewriter.rewrite.assert_called_once()
    assert result["rewritten_question"] == "How many orders did Uday place?"
    assert result["route"] == "sql"


def test_acknowledgement_gets_a_canned_reply_without_calling_the_model():
    workflow, _ = make_workflow()

    with patch("app.graph.workflow.OllamaClient") as client:
        result = workflow.general_node(
            {
                "question": "oh ok",
                "thread_id": "t1",
                "user_id": "local-user",
                "conversation_history": PREVIOUS_TOPIC,
            }
        )

    assert result["answer"].startswith("Okay!")
    client.assert_not_called()


def test_greeting_after_a_conversation_stays_a_greeting():
    workflow, db = make_workflow(customers=[])
    workflow.ask("uday bought iphone for 80k", thread_id="t5")
    workflow.ask("no", thread_id="t5")

    reply = workflow.ask("hi", thread_id="t5")

    assert reply["result"]["route"] == "general"
    assert "documents" in reply["answer"]
    assert db.inserts() == []