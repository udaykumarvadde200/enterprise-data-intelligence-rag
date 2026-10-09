from langchain_core.documents import Document

from app.rag.answer_generator import GroundedAnswerGenerator
from app.rag.context_filter import ContextFilter
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.ollama_reranker import OllamaRelevanceReranker
from app.rag.rag_service import RAGService
from app.rag.retrieval_pipeline import RetrievalPipeline
from app.rag.vector_store import ChromaVectorStore


def main() -> None:
    documents = [
        Document(
            page_content=(
                "Employees must protect confidential customer "
                "information and personal data. Access should be "
                "limited to authorized personnel."
            ),
            metadata={"file_type": "txt", "source": "security_policy.txt"},
        ),
        Document(
            page_content=(
                "Passwords must be strong, unique, and never shared "
                "with other employees."
            ),
            metadata={"file_type": "txt", "source": "password_policy.txt"},
        ),
        Document(
            page_content=(
                "Employees can request annual leave through the "
                "human resources department."
            ),
            metadata={"file_type": "txt", "source": "leave_policy.txt"},
        ),
    ]

    vector_store = ChromaVectorStore(
        persist_directory="chroma_db_rag_smoke",
        collection_name="end_to_end_rag_smoke",
        embedding_model="nomic-embed-text",
    )

    try:
        print("STAGE 1: INDEX DOCUMENTS")
        for document in documents:
            source = document.metadata["source"]
            count = vector_store.index_documents([document], source=source)
            print(f"Indexed {count} chunk(s): {source}")

        print(f"Collection count: {vector_store.count()}")

        hybrid_retriever = HybridRetriever(
            documents=documents,
            vector_store=vector_store,
            dense_k=3,
            sparse_k=3,
        )

        retrieval_pipeline = RetrievalPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=OllamaRelevanceReranker(),
            context_filter=ContextFilter(
                max_chunks=3,
                max_characters=4000,
            ),
            candidate_k=3,
            rerank_k=3,
        )

        service = RAGService(
            retrieval_pipeline=retrieval_pipeline,
            answer_generator=GroundedAnswerGenerator(),
        )

        query = "How should confidential customer information be protected?"

        print("\nSTAGE 2: RETRIEVE, RERANK, FILTER, AND GENERATE")
        result = service.ask(query)

        print(f"\nQuestion: {query}")
        print(f"\nAnswer:\n{result.answer}")
        print("\nSources:")

        for source in result.sources:
            page = source.get("page")
            location = f", page {page}" if page is not None else ""
            print(f"[{source['id']}] {source['source']}{location}")

        print(f"\nRecognized citation IDs: {result.citation_ids}")

        if not result.answer.strip():
            raise RuntimeError("RAG returned an empty answer.")

        if not result.sources:
            raise RuntimeError("RAG returned no source information.")

        if result.citation_ids:
            print("\nPASS: End-to-end RAG smoke test completed.")
        else:
            print(
                "\nWARNING: No valid citation IDs were recognized. "
                "Inspect the generated answer."
            )

    finally:
        vector_store.store = None
        vector_store.embeddings = None


if __name__ == "__main__":
    main()
