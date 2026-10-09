from app.graph.router import QueryRouter
from app.graph.workflow import EnterpriseWorkflow


def test_greeting_routes_to_general():
    assert QueryRouter().route("hi") == "general"


def test_resume_question_routes_to_rag():
    assert QueryRouter().route("What is in the resume?") == "rag"


def test_customer_question_routes_to_sql():
    assert QueryRouter().route("Show me all customers") == "sql"


def test_unknown_question_does_not_default_to_sql():
    assert QueryRouter().route("Can you help me?") == "general"


def test_greeting_workflow_does_not_query_database():
    workflow = EnterpriseWorkflow()
    result = workflow.ask("hi", thread_id="greeting-test")

    assert result["result"]["route"] == "general"
    assert "uploaded documents" in result["answer"]