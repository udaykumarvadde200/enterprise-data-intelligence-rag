import pytest
from langchain_core.documents import Document

from app.ingestion.document_loader import DocumentLoader
from app.ingestion.text_splitter import TextChunker


def test_load_txt_file(tmp_path):
    file_path = tmp_path / "policy.txt"
    file_path.write_text(
        "Employees must protect confidential information.",
        encoding="utf-8",
    )

    documents = DocumentLoader().load_file(file_path)

    assert len(documents) == 1
    assert "confidential information" in documents[0].page_content
    assert documents[0].metadata["source"] == "policy.txt"
    assert documents[0].metadata["file_type"] == "txt"


def test_empty_txt_file_raises_error(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No extractable text"):
        DocumentLoader().load_file(file_path)


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocumentLoader().load_file(tmp_path / "missing.txt")


def test_unsupported_file_type_raises_error(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("id,name\n1,Alice", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentLoader().load_file(file_path)


def test_chunking_preserves_metadata_and_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    document = Document(
        page_content=text,
        metadata={"source": "policy.txt", "file_type": "txt"},
    )

    chunks = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    ).split_documents([document])

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "policy.txt" for chunk in chunks)
    assert all(chunk.metadata["file_type"] == "txt" for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)


def test_empty_document_list_returns_empty_chunks():
    assert TextChunker().split_documents([]) == []


def test_invalid_chunk_configuration_raises_error():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_load_pdf_preserves_page_metadata(tmp_path, monkeypatch):
    file_path = tmp_path / "policy.pdf"
    file_path.write_bytes(b"test fixture")

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [
                FakePage("First page content."),
                FakePage("Second page content."),
            ]

    monkeypatch.setattr(
        "app.ingestion.document_loader.PdfReader",
        FakeReader,
    )

    documents = DocumentLoader().load_file(file_path)

    assert len(documents) == 2
    assert documents[0].page_content == "First page content."
    assert documents[0].metadata["source"] == "policy.pdf"
    assert documents[0].metadata["page"] == 1
    assert documents[1].metadata["page"] == 2
    assert all(doc.metadata["file_type"] == "pdf" for doc in documents)