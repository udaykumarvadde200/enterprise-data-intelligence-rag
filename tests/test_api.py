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
    (tmp_path / "policy.txt").write_text(
        "Policy content.", encoding="utf-8"
    )
    (tmp_path / "ignored.csv").write_text(
        "a,b", encoding="utf-8"
    )

    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    client = TestClient(main.app)

    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()
    assert data["count"] == 2

    filenames = {
        item["filename"] for item in data["documents"]
    }
    assert filenames == {"policy.txt", "ignored.csv"}


def test_upload_csv_document(tmp_path, monkeypatch):
    fake = FakeWorkflow()
    monkeypatch.setattr(main, "workflow", fake)
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    client = TestClient(main.app)

    csv_content = (
        "customer_id,name,city\n"
        "CUST-901,Aarav Hyderabad,Hyderabad\n"
        "CUST-902,Meera Bengaluru,Bengaluru\n"
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "customers.csv",
                csv_content.encode("utf-8"),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "customers.csv"
    assert (tmp_path / "customers.csv").read_text(encoding="utf-8") == csv_content
    assert fake.refreshed is True


def test_upload_rejects_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/documents/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_upload_sanitizes_filename(tmp_path, monkeypatch):
    fake = FakeWorkflow()
    monkeypatch.setattr(main, "workflow", fake)
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                r"..\nested\customers.csv",
                b"customer_id,name\nCUST-901,Aarav\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "customers.csv"
    assert (tmp_path / "customers.csv").is_file()
    assert not (tmp_path / "nested" / "customers.csv").exists()