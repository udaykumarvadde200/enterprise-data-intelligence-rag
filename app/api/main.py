from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.graph.workflow import EnterpriseWorkflow
from app.ingestion.document_loader import DocumentLoader

app = FastAPI(
    title="Enterprise Knowledge Intelligence API",
    description="Ask questions across enterprise documents and structured data.",
    version="0.1.0",
)

workflow = EnterpriseWorkflow()
workflow_lock = Lock()

DOCUMENTS_DIR = Path("data/documents")
SUPPORTED_EXTENSIONS = DocumentLoader.SUPPORTED_EXTENSIONS
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MiB


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="default", min_length=1, max_length=128)


class AskResponse(BaseModel):
    answer: str
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def _supported_extensions_message() -> str:
    extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return f"Supported file types: {extensions}."


@app.get("/health")
def health() -> dict[str, str]:
    """Check whether the API process is responding."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Route a question to SQL, document RAG, or both."""
    question = request.question.strip()
    thread_id = request.thread_id.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    if not thread_id:
        raise HTTPException(status_code=422, detail="Thread ID must not be empty.")

    try:
        with workflow_lock:
            output = workflow.ask(question, thread_id=thread_id)

        return AskResponse(
            answer=output.get("answer", "No answer was returned."),
            result=output.get("result", {}),
            error=output.get("error"),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The question could not be processed.",
        ) from exc


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    """List supported files currently stored in the document directory."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = [
        {
            "filename": path.relative_to(DOCUMENTS_DIR).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(DOCUMENTS_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return {"count": len(documents), "documents": documents}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Save a supported document and invalidate the cached RAG service."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    # Strip Windows- or Unix-style directory components.
    filename = file.filename.replace("\\", "/").split("/")[-1].strip()

    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. {_supported_extensions_message()}",
        )

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 15 MiB upload limit.",
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCUMENTS_DIR / filename

    # Exclusive creation prevents accidental overwrites.
    try:
        with destination.open("xb") as destination_file:
            destination_file.write(contents)
    except FileExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail="A document with that filename already exists.",
        ) from exc

    # The next document question will rebuild the cached RAG service.
    with workflow_lock:
        workflow.refresh_documents()

    return {
        "status": "uploaded",
        "filename": filename,
        "size_bytes": len(contents),
        "message": (
            "Saved successfully. The document index will refresh "
            "on the next question."
        ),
    }