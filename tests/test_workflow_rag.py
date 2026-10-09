from types import SimpleNamespace

from app.graph.workflow import EnterpriseWorkflow


class FakeRAGService:
    def ask(self, query):
        return SimpleNamespace(
            answer=f"Grounded answer for: {query} [S1]",
            sources=[
                {
                    "id": "S1",
                    "source": "policy.txt",
                    "content": "Employees must protect confidential information.",
                }
            ],
            citation_ids=["S1"],
        )


class FakeSQLPipeline:
    def __init__(self):
        self.conversation_state = SimpleNamespace(status="idle")

    def run(self, question):
        return {
            "status": "success",
            "sql": "SELECT COUNT(*) FROM orders",
            "results": [{"count": 5}],
        }


def make_workflow(monkeypatch):
    workflow = EnterpriseWorkflow(rag_service=FakeRAGService())
    fake_sql = FakeSQLPipeline()
    monkeypatch.setattr(
        workflow,
        "_get_sql_pipeline",
        lambda thread_id: fake_sql,
    )
    return workflow


def test_rag_question_uses_rag_service(monkeypatch):
    workflow = make_workflow(monkeypatch)

    result = workflow.ask(
        "Summarize the policy document",
        thread_id="rag-test",
    )

    assert result["route"] == "rag"
    assert result["result"]["status"] == "success"
    assert result["result"]["sources"][0]["source"] == "policy.txt"
    assert "[S1]" in result["answer"]


def test_hybrid_question_returns_sql_and_document_results(monkeypatch):
    workflow = make_workflow(monkeypatch)

    result = workflow.ask(
        "According to the document, how many orders are there?",
        thread_id="hybrid-test",
    )

    assert result["route"] == "hybrid"
    assert result["result"]["route"] == "hybrid"
    assert result["result"]["sql"]["status"] == "success"
    assert result["result"]["rag"]["status"] == "success"
    assert "DATABASE RESULTS" in result["answer"]
    assert "DOCUMENT EVIDENCE" in result["answer"]


def test_empty_question_returns_validation_message(monkeypatch):
    workflow = make_workflow(monkeypatch)

    result = workflow.ask("   ", thread_id="empty-test")

    assert result["answer"] == "Please enter a question."
    assert result["error"] == "Empty question."
