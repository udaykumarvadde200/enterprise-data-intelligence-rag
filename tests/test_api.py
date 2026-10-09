from fastapi.testclient import TestClient

from app.api import main


class FakeWorkflow:
    def __init__(self):
        self.refreshed = False

    def ask(self, question: str, thread_id: str = "default"):
        return {
            "answer": f"Processed: {question}",
            "result": {"status": "success", "route": "rag"},
        }

    def refresh_documents(self):
        self.refreshed = True


def test_health_endpoint():
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_uses_workflow(monkeypatch):
    fake = FakeWorkflow()
    monkeypatch.setattr(main, "workflow", fake)
    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={"question": "Explain the security policy", "thread_id": "test-session"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Processed: Explain the security policy"
    assert response.json()["result"]["route"] == "rag"


def test_ask_rejects_blank_question():
    client = TestClient(main.app)

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_upload_document_and_refresh_workflow(tmp_path, monkeypatch):
    fake = FakeWorkflow()
    monkeypatch.setattr(main, "workflow", fake)
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/documents/upload",
        files={"file": ("policy.txt", b"Employees must protect customer data.", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "policy.txt"
    assert (tmp_path / "policy.txt").read_text() == (
        "Employees must protect customer data."
    )
    assert fake.refreshed is True


def test_upload_rejects_unsupported_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/documents/upload",
        files={"file": ("script.exe", b"not allowed", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert list(tmp_path.iterdir()) == []


def test_upload_rejects_duplicate_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    client = TestClient(main.app)
    upload = {
        "file": ("policy.txt", b"Policy content.", "text/plain"),
    }

    first = client.post("/documents/upload", files=upload)
    second = client.post("/documents/upload", files=upload)

    assert first.status_code == 200
    assert second.status_code == 409


def test_list_documents(tmp_path, monkeypatch):
    (tmp_path / "policy.txt").write_text("Policy content.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("a,b", encoding="utf-8")
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    client = TestClient(main.app)

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["documents"][0]["filename"] == "policy.txt"