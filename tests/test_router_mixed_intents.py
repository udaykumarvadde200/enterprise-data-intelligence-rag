import pytest

from app.graph.router import QueryRouter


@pytest.mark.parametrize(
    "question",
    [
        "What is the company refund policy for enterprise customers?",
        "refund policy for enterprise customers",
        "What does the contract say about customer orders?",
        "Show me the policy",
        "Show me the sales report",
        "List all guidelines for handling orders",
    ],
)
def test_document_questions_with_incidental_database_nouns_route_to_rag(question):
    assert QueryRouter().route(question) == "rag"


@pytest.mark.parametrize(
    "question",
    [
        "Does the policy match the database records?",
        "How many customers are covered by the refund policy?",
        "Does the contract total match the revenue in the orders table?",
        "Compare the policy with the database",
        "According to the document, and how many orders were placed?",
    ],
)
def test_clear_cross_source_questions_still_route_to_hybrid(question):
    assert QueryRouter().route(question) == "hybrid"


@pytest.mark.parametrize(
    "question",
    [
        "Show me all customers",
        "Show me all orders",
        "How many orders are there?",
        "List all customers in Kochi",
        "What is the average order amount?",
        "Who has the highest order?",
        "How many customers are there?",
    ],
)
def test_database_questions_still_route_to_sql(question):
    assert QueryRouter().route(question) == "sql"


@pytest.mark.parametrize(
    "question",
    [
        "What is in the resume?",
        "Summarize the uploaded document",
        "What is the project status and target completion date?",
    ],
)
def test_document_only_questions_still_route_to_rag(question):
    assert QueryRouter().route(question) == "rag"


@pytest.mark.parametrize("question", ["hi", "Can you help me?", "thanks", "oh ok"])
def test_conversation_still_routes_to_general(question):
    assert QueryRouter().route(question) == "general"
