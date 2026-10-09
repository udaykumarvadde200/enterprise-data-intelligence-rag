from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document

from app.rag.hybrid_retriever import HybridRetriever
from app.rag.vector_store import ChromaVectorStore


def main() -> None:
    documents = [
        Document(
            page_content=(
                "Employees must protect confidential customer "
                "information and personal data."
            ),
            metadata={"source": "security_policy.txt"},
        ),
        Document(
            page_content=(
                "Employees can request annual leave through "
                "the human resources department."
            ),
            metadata={"source": "leave_policy.txt"},
        ),
        Document(
            page_content=(
                "Passwords must be strong, unique, and never "
                "shared with other employees."
            ),
            metadata={"source": "password_policy.txt"},
        ),
    ]

    with TemporaryDirectory(prefix="hybrid_rag_") as temp_dir:
        vector_store = ChromaVectorStore(
            persist_directory=str(Path(temp_dir) / "chroma"),
            collection_name="hybrid_smoke_test",
            embedding_model="nomic-embed-text",
        )

        try:
            for document in documents:
                vector_store.index_documents(
                    documents=[document],
                    source=document.metadata["source"],
                )

            retriever = HybridRetriever(
                documents=documents,
                vector_store=vector_store,
                dense_k=3,
                sparse_k=3,
            )

            query = "How should confidential customer data be protected?"
            results = retriever.search(query, top_k=3)

            print(f"Query: {query}")
            print(f"Indexed chunks: {vector_store.count()}")
            print(f"Hybrid results: {len(results)}")

            for rank, document in enumerate(results, start=1):
                print(f"\nRank: {rank}")
                print(f"Source: {document.metadata.get('source')}")
                print(f"Content: {document.page_content}")

        finally:
            # Release ChromaDB's persistent client before directory cleanup.
            vector_store.store = None
            vector_store.embeddings = None


if __name__ == "__main__":
    main()