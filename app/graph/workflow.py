from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.router import QueryRouter
from app.graph.state import WorkflowState
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.text_splitter import TextChunker
from app.llm.ollama_client import OllamaClient
from app.memory.long_term_memory import LongTermMemory
from app.rag.answer_generator import GroundedAnswerGenerator
from app.rag.context_filter import ContextFilter
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.ollama_reranker import OllamaRelevanceReranker
from app.rag.rag_service import RAGService
from app.rag.retrieval_pipeline import RetrievalPipeline
from app.rag.vector_store import ChromaVectorStore
from app.sql.sql_pipeline import SQLPipeline


class EnterpriseWorkflow:
    """Route enterprise questions to general conversation, SQL, RAG, or both."""

    def __init__(
        self,
        rag_service: RAGService | None = None,
        long_term_memory: LongTermMemory | None = None,
    ):
        self.router = QueryRouter()
        self.sql_pipelines: dict[str, SQLPipeline] = {}
        self.checkpointer = MemorySaver()
        self._rag_service = rag_service
        self.long_term_memory = long_term_memory or LongTermMemory()

        builder = StateGraph(WorkflowState)

        # Register every graph node.
        builder.add_node("route", self.route_node)
        builder.add_node("general", self.general_node)
        builder.add_node("sql", self.sql_node)
        builder.add_node("rag", self.rag_node)
        builder.add_node("hybrid", self.hybrid_node)
        builder.add_node("record_turn", self.record_turn_node)

        # Start by routing the user's question.
        builder.add_edge(START, "route")

        builder.add_conditional_edges(
            "route",
            self.select_route,
            {
                "general": "general",
                "sql": "sql",
                "rag": "rag",
                "hybrid": "hybrid",
            },
        )

        # Record the result of every route before ending the turn.
        builder.add_edge("general", "record_turn")
        builder.add_edge("sql", "record_turn")
        builder.add_edge("rag", "record_turn")
        builder.add_edge("hybrid", "record_turn")
        builder.add_edge("record_turn", END)

        self.graph = builder.compile(checkpointer=self.checkpointer)

    def refresh_documents(self) -> None:
        """Invalidate the cached RAG service after documents change."""
        self._rag_service = None

    def _get_sql_pipeline(self, thread_id: str) -> SQLPipeline:
        if thread_id not in self.sql_pipelines:
            self.sql_pipelines[thread_id] = SQLPipeline()
        return self.sql_pipelines[thread_id]

    def _get_rag_service(self) -> RAGService:
        if self._rag_service is not None:
            return self._rag_service

        documents_dir = Path("data/documents")
        if not documents_dir.is_dir():
            raise FileNotFoundError(
                f"Document directory not found: {documents_dir.resolve()}"
            )

        loader = DocumentLoader()
        chunker = TextChunker()
        all_chunks: list[Document] = []

        files = sorted(
            path
            for path in documents_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in loader.SUPPORTED_EXTENSIONS
        )

        if not files:
            raise FileNotFoundError(
                "No supported documents found in data/documents."
            )

        vector_store = ChromaVectorStore(
            persist_directory="chroma_db",
            collection_name="enterprise_knowledge",
            embedding_model="nomic-embed-text",
        )

        try:
            for path in files:
                source = path.relative_to(documents_dir).as_posix()
                loaded = loader.load_file(path)
                chunks = chunker.split_documents(loaded)

                for chunk in chunks:
                    chunk.metadata["source"] = source

                vector_store.index_documents(chunks, source=source)
                all_chunks.extend(chunks)

            hybrid_retriever = HybridRetriever(
                documents=all_chunks,
                vector_store=vector_store,
                dense_k=10,
                sparse_k=10,
                fusion_k=60,
            )

            retrieval_pipeline = RetrievalPipeline(
                hybrid_retriever=hybrid_retriever,
                reranker=OllamaRelevanceReranker(),
                context_filter=ContextFilter(
                    max_chunks=5,
                    max_characters=6000,
                ),
                candidate_k=10,
                rerank_k=5,
            )

            self._rag_service = RAGService(
                retrieval_pipeline=retrieval_pipeline,
                answer_generator=GroundedAnswerGenerator(
                    llm=OllamaClient(),
                ),
            )

            return self._rag_service

        except Exception:
            vector_store.store = None
            vector_store.embeddings = None
            raise

    def route_node(self, state: WorkflowState) -> dict[str, Any]:
        """Retrieve user-specific memories and select a processing route."""
        question = state["question"]
        thread_id = state["thread_id"]
        user_id = state.get("user_id", "local-user")

        try:
            memories = self.long_term_memory.search(
                user_id=user_id,
                query_text=question,
                limit=5,
            )
        except Exception:
            # Memory is supplementary; failures must not break SQL or RAG.
            memories = []

        route = self.router.route(question)

        # Ordinary conversation must not trigger SQL or RAG initialization.
        if route == "general":
            return {
                "route": "general",
                "memories": memories,
            }

        pipeline = self._get_sql_pipeline(thread_id)

        # Preserve an outstanding SQL clarification across turns.
        if pipeline.conversation_state.status == "awaiting_clarification":
            route = "sql"

        return {
            "route": route,
            "memories": memories,
        }

    def select_route(self, state: WorkflowState) -> str:
        return state["route"]

    def record_turn_node(self, state: WorkflowState) -> dict[str, Any]:
        """Persist the latest turn in checkpointed conversation state."""
        history = list(state.get("conversation_history", []))

        history.append(
            {
                "question": state.get("question", ""),
                "answer": state.get("answer", ""),
            }
        )

        return {"conversation_history": history[-10:]}

    def general_node(self, state: WorkflowState) -> dict[str, Any]:
        """Respond to general requests using relevant saved user context."""
        normalized = " ".join(
            state["question"].casefold().strip(" \t!.,?").split()
        )

        if normalized in {
            "hi",
            "hello",
            "hey",
            "hi there",
            "hello there",
            "good morning",
            "good afternoon",
            "good evening",
        }:
            answer = (
                "Hi! I can help you search uploaded documents and answer "
                "questions about enterprise database records. What would "
                "you like to know?"
            )
        elif normalized in {"thanks", "thank you"}:
            answer = "You're welcome! What would you like to explore next?"
        elif normalized in {"bye", "goodbye"}:
            answer = "Goodbye! Come back whenever you need help."
        else:

            memories = state.get("memories", [])
            history = state.get("conversation_history", [])

            if not memories and not history:
                answer = (
                    "I can help with uploaded documents and enterprise database "
                    "questions. Please mention the document, resume, customer, "
                    "order, or information you're looking for."
                )
            else:
                memory_context = "\n".join(
                    f"- {item['memory_key']}: {item['memory_value']}"
                    for item in memories
                ) or "(No saved memories retrieved.)"

                recent_history = history[-10:]
                history_context = "\n".join(
                    f"User: {turn['question']}\n"
                    f"Assistant: {turn['answer']}"
                    for turn in recent_history
                ) or "(No previous conversation turns.)"

                try:
                    llm = OllamaClient().get_chat_model()
                    response = llm.invoke(
                        "You are an assistant for an enterprise knowledge "
                        "platform. Answer the user's general question clearly. "
                        "Use recent conversation history to understand follow-up "
                        "questions and saved memories when relevant. Treat both "
                        "as context, not verified evidence about enterprise "
                        "records or documents. Never claim to have queried a "
                        "database or searched documents unless that actually "
                        "occurred.\n\n"
                        f"Recent conversation:\n{history_context}\n\n"
                        f"Saved user context:\n{memory_context}\n\n"
                        f"Current user question:\n{state['question']}"
                    )
                    answer = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                except Exception:
                    answer = (
                        "I can help with uploaded documents and enterprise "
                        "database questions. Please tell me what you need."
                    )

        return {
            "answer": answer,
            "result": {
                "status": "success",
                "route": "general",
            },
        }

    def sql_node(self, state: WorkflowState) -> dict[str, Any]:
        pipeline = self._get_sql_pipeline(state["thread_id"])

        try:
            result = pipeline.run(state["question"])

            if result["status"] in {"clarification_required", "not_found"}:
                answer = result["message"]
            elif result["status"] == "success":
                answer = self._format_sql_result(result)
            else:
                answer = str(result)

            return {
                "result": {
                    **result,
                    "route": "sql",
                },
                "answer": answer,
            }

        except Exception:
            return {
                "error": "Database query failed.",
                "answer": (
                    "I couldn't complete the database query. "
                    "Please check the query and try again."
                ),
                "result": {"status": "error", "route": "sql"},
            }

    def rag_node(self, state: WorkflowState) -> dict[str, Any]:
        try:
            answer_result = self._get_rag_service().ask(state["question"])

            result = {
                "status": (
                    "success"
                    if answer_result.sources
                    else "insufficient_evidence"
                ),
                "route": "rag",
                "sources": answer_result.sources,
                "citation_ids": answer_result.citation_ids,
            }

            return {
                "answer": answer_result.answer,
                "result": result,
            }

        except Exception:
            return {
                "answer": (
                    "I couldn't search the document collection. "
                    "Please check that supported files exist in "
                    "data/documents and that Ollama is running."
                ),
                "error": "Document retrieval failed. Check local logs for details.",
                "result": {"status": "error", "route": "rag"},
            }

    def hybrid_node(self, state: WorkflowState) -> dict[str, Any]:
        question = state["question"]

        try:
            sql_result = self._get_sql_pipeline(
                state["thread_id"]
            ).run(question)

            if sql_result["status"] == "success":
                sql_answer = self._format_sql_result(sql_result)
            elif sql_result["status"] in {
                "clarification_required",
                "not_found",
            }:
                sql_answer = sql_result["message"]
            else:
                sql_answer = str(sql_result)

            sql_part = {
                "status": sql_result.get("status", "unknown"),
                "answer": sql_answer,
                "result": sql_result,
            }

        except Exception:
            sql_part = {
                "status": "error",
                "answer": "The database portion could not be completed.",
                "result": {"status": "error"},
            }

        try:
            answer_result = self._get_rag_service().ask(question)

            rag_part = {
                "status": (
                    "success"
                    if answer_result.sources
                    else "insufficient_evidence"
                ),
                "answer": answer_result.answer,
                "sources": answer_result.sources,
                "citation_ids": answer_result.citation_ids,
            }

        except Exception:
            rag_part = {
                "status": "error",
                "answer": "The document portion could not be completed.",
                "sources": [],
                "citation_ids": [],
            }

        answer = (
            "DATABASE RESULTS\n"
            f"{sql_part['answer']}\n\n"
            "DOCUMENT EVIDENCE\n"
            f"{rag_part['answer']}"
        )

        return {
            "answer": answer,
            "result": {
                "status": (
                    "success"
                    if sql_part["status"] != "error"
                    or rag_part["status"] == "success"
                    else "error"
                ),
                "route": "hybrid",
                "sql": sql_part,
                "rag": rag_part,
            },
        }

    @staticmethod
    def _format_sql_result(result: dict[str, Any]) -> str:
        rows = result.get("results", [])

        if not rows:
            return "The query completed successfully, but no records matched."

        formatted_rows = "\n".join(str(row) for row in rows)

        return (
            "Query completed successfully.\n\n"
            f"SQL:\n{result['sql']}\n\n"
            f"Results:\n{formatted_rows}"
        )

    def ask(
        self,
        question: str,
        thread_id: str = "default",
        user_id: str = "local-user",
    ) -> dict[str, Any]:
        """Run the workflow with conversation and user-specific memory."""
        question = question.strip()
        thread_id = thread_id.strip()
        user_id = user_id.strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "error": "Empty question.",
            }

        if not thread_id or not user_id:
            return {
                "answer": "A valid thread ID and user ID are required.",
                "error": "Empty thread ID or user ID.",
            }

        state: WorkflowState = {
            "question": question,
            "thread_id": thread_id,
            "user_id": user_id,
        }

        # Keep checkpointed conversation state separate by user.
        conversation_id = f"{user_id}:{thread_id}"

        return self.graph.invoke(
            state,
            config={"configurable": {"thread_id": conversation_id}},
        )
