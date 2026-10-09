from langchain_core.documents import Document

from app.ingestion.text_splitter import TextChunker
from app.rag.vector_store import ChromaVectorStore


def main():
    store = ChromaVectorStore(
        persist_directory="chroma_db_test",
        collection_name="vector_store_smoke_test",
    )

    chunker = TextChunker(
        chunk_size=200,
        chunk_overlap=30,
    )

    security_document = Document(
        page_content=(
            "Employees must protect confidential "
            "customer information and personal data."
        ),
        metadata={"file_type": "txt"},
    )

    leave_document = Document(
        page_content=(
            "Employees can request annual leave "
            "through the human resources department."
        ),
        metadata={"file_type": "txt"},
    )

    # Index each source independently to preserve its metadata.
    security_chunks = chunker.split_documents([security_document])
    leave_chunks = chunker.split_documents([leave_document])

    store.index_documents(
        security_chunks,
        source="security_policy.txt",
    )

    store.index_documents(
        leave_chunks,
        source="leave_policy.txt",
    )

    print(f"Total stored chunks: {store.count()}")

    results = store.search(
        "How should we keep customer data safe?",
        k=2,
    )

    print("\nSearch results:")

    for result in results:
        print(f"\nSource: {result.metadata.get('source')}")
        print(result.page_content)


if __name__ == "__main__":
    main()