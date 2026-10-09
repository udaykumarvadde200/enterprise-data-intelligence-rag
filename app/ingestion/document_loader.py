from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


class DocumentLoader:
    """Load PDF and TXT files into standardized documents."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def load_file(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension or '(no extension)'}. "
                "Supported types: PDF and TXT."
            )

        if extension == ".pdf":
            documents = self._load_pdf(path)
        else:
            documents = self._load_txt(path)

        documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        if not documents:
            raise ValueError(f"No extractable text found in: {path.name}")

        return documents

    @staticmethod
    def _load_pdf(path: Path) -> list[Document]:
        reader = PdfReader(str(path))
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_number,
                        "file_type": "pdf",
                    },
                )
            )

        return documents

    @staticmethod
    def _load_txt(path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8-sig")

        return [
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "file_type": "txt",
                },
            )
        ]