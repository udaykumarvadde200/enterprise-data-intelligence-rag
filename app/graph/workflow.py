import hashlib
import logging
import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
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
from app.rag.reranker import CrossEncoderReranker
from app.rag.query_rewriter import QueryRewriter
from app.rag.rag_service import RAGService
from app.rag.retrieval_pipeline import RetrievalPipeline
from app.rag.vector_store import ChromaVectorStore
from app.sql.sql_pipeline import SQLPipeline
from app.writes.parser import (
    MissingAmountError,
    WriteRequestError,
    looks_like_purchase,
    parse_purchase,
)
from app.writes.service import PendingPurchase, PurchaseWriter

# Make ALLOW_WRITES from .env visible before the first chat message.
load_dotenv()

logger = logging.getLogger(__name__)

# A bare "yes" with nothing waiting should get a clear answer, not chat.
# Weaker words such as "ok" or "sure" are left to normal conversation.
BARE_CONFIRMATIONS = frozenset(
    {"yes", "yes please", "confirm", "save it", "do it", "go ahead"}
)

# Small talk is never rewritten into a search query: the small local model
# would otherwise turn "hi" into a question about the previous topic.
GREETINGS = frozenset(
    {
        "hi", "hello", "hey", "hi there", "hello there", "good morning",
        "good afternoon", "good evening",
    }
)
ACKNOWLEDGEMENTS = frozenset(
    {
        "ok", "okay", "oh ok", "oh okay", "ok thanks", "okay thanks",
        "ok thank you", "got it", "cool", "great", "nice", "alright",
        "fine", "i see", "understood", "perfect",
    }
)
SMALL_TALK = (
    GREETINGS
    | ACKNOWLEDGEMENTS
    | frozenset({"thanks", "thank you", "thanks a lot", "bye", "goodbye"})
)


class EnterpriseWorkflow:
    """Route enterprise questions to general conversation, SQL, RAG, or both."""

    def __init__(
        self,
        rag_service: RAGService | None = None,
        long_term_memory: LongTermMemory | None = None,
        purchase_writer: PurchaseWriter | None = None,
    ):
        self.router = QueryRouter()
        self.sql_pipelines: dict[tuple[str, str], SQLPipeline] = {}
        self.checkpointer = MemorySaver()
        self._rag_service = rag_service
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.query_rewriter = QueryRewriter()
        self.purchase_writer = purchase_writer or PurchaseWriter()
        # Unsaved data-entry requests waiting for the user's yes/no,
        # keyed by (user_id, thread_id).
        self.pending_writes: dict[tuple[str, str], PendingPurchase] = {}

        builder = StateGraph(WorkflowState)

        builder.add_node("route", self.route_node)
        builder.add_node("general", self.general_node)
        builder.add_node("sql", self.sql_node)
        builder.add_node("rag", self.rag_node)
        builder.add_node("hybrid", self.hybrid_node)
        builder.add_node("write", self.write_node)
        builder.add_node("record_turn", self.record_turn_node)

        builder.add_edge(START, "route")

        builder.add_conditional_edges(
            "route",
            self.select_route,
            {
                "general": "general",
                "sql": "sql",
                "rag": "rag",
                "hybrid": "hybrid",
                "write": "write",
            },
        )

        builder.add_edge("general", "record_turn")
        builder.add_edge("sql", "record_turn")
        builder.add_edge("rag", "record_turn")
        builder.add_edge("hybrid", "record_turn")
        builder.add_edge("write", "record_turn")
        builder.add_edge("record_turn", END)

        self.graph = builder.compile(checkpointer=self.checkpointer)

    def refresh_documents(self) -> None:
        """Invalidate the cached RAG service after documents change."""
        self._rag_service = None

    def _get_sql_pipeline(
        self,
        thread_id: str,
        user_id: str = "local-user",
    ) -> SQLPipeline:
        """Return a SQL pipeline isolated by user and conversation."""
        cache_key = (user_id, thread_id)

        if cache_key not in self.sql_pipelines:
            self.sql_pipelines[cache_key] = SQLPipeline()

        return self.sql_pipelines[cache_key]

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
    	reranker=CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    ),
    context_filter=ContextFilter(
        max_chunks=5,
        max_characters=6000,
        min_score=-10.0,
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
        """Rewrite follow-ups, retrieve memories, and select a route."""
        question = state["question"]
        thread_id = state["thread_id"]
        user_id = state.get("user_id", "local-user")
        history = state.get("conversation_history", [])

        # Data entry is decided on the raw message, before any rewriting.
        if self._is_write_turn(question, (user_id, thread_id)):
            return {
                "route": "write",
                "memories": [],
                "rewritten_question": question,
            }

        if self._is_small_talk(question):
            rewritten_question = question
        else:
            rewritten_question = self.query_rewriter.rewrite(
                question,
                history,
            )

        try:
            memories = self.long_term_memory.search(
                user_id=user_id,
                query_text=rewritten_question,
                limit=5,
            )
        except Exception:
            memories = []

        route = self.router.route(rewritten_question)

        if route == "general":
            return {
                "route": "general",
                "memories": memories,
                "rewritten_question": rewritten_question,
            }

        pipeline = self._get_sql_pipeline(
            thread_id,
            user_id,
        )

        if pipeline.conversation_state.status == "awaiting_clarification":
            route = "sql"

        return {
            "route": route,
            "memories": memories,
            "rewritten_question": rewritten_question,
        }

    @staticmethod
    def writes_enabled() -> bool:
        """Chat data entry is on unless ALLOW_WRITES is set to false."""
        value = os.getenv("ALLOW_WRITES", "true").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _live_pending(self, key: tuple[str, str]) -> PendingPurchase | None:
        pending = self.pending_writes.get(key)

        if pending is not None and self.purchase_writer.is_expired(pending):
            del self.pending_writes[key]
            return None

        return pending

    def _is_write_turn(self, question: str, key: tuple[str, str]) -> bool:
        pending = self._live_pending(key)

        if pending is not None:
            if self.purchase_writer.classify(pending, question) is not None:
                return True

            # Any other message abandons the unsaved request. Nothing was
            # written, so nothing needs to be undone.
            del self.pending_writes[key]

        if looks_like_purchase(question):
            return True

        return self.writes_enabled() and self._is_bare_confirmation(question)

    @staticmethod
    def _is_small_talk(text: str) -> bool:
        words = " ".join(text.casefold().strip(" \t!.,?").split())
        return words in SMALL_TALK

    @staticmethod
    def _is_bare_confirmation(text: str) -> bool:
        words = " ".join(re.sub(r"[^\w' ]", " ", text.casefold()).split())
        return words in BARE_CONFIRMATIONS

    def write_node(self, state: WorkflowState) -> dict[str, Any]:
        """Handle 'NAME bought PRODUCT for AMOUNT' with confirm-then-save."""
        question = state["question"]
        key = (state.get("user_id", "local-user"), state["thread_id"])

        if not self.writes_enabled():
            self.pending_writes.pop(key, None)
            return {
                "answer": (
                    "Adding data through chat is turned off "
                    "(ALLOW_WRITES=false)."
                ),
                "result": {"status": "disabled", "route": "write"},
            }

        try:
            pending = self._live_pending(key)
            reply = (
                self.purchase_writer.respond(pending, question)
                if pending is not None
                else None
            )

            if reply is None and not looks_like_purchase(question):
                self.pending_writes.pop(key, None)
                return {
                    "answer": (
                        "There is nothing waiting to be saved. To add an "
                        "order, write a sentence such as: "
                        "Uday bought iPhone for 80k"
                    ),
                    "result": {"status": "nothing_pending", "route": "write"},
                }

            if reply is None:
                try:
                    reply = self.purchase_writer.prepare(parse_purchase(question))
                except MissingAmountError as missing:
                    reply = self.purchase_writer.prepare_partial(missing.partial)

        except WriteRequestError as exc:
            self.pending_writes.pop(key, None)
            return {
                "answer": str(exc),
                "result": {"status": "invalid_request", "route": "write"},
            }
        except Exception:
            logger.exception("Chat data entry failed.")
            self.pending_writes.pop(key, None)
            return {
                "answer": (
                    "I couldn't process that data entry. Nothing was saved."
                ),
                "error": "Data entry failed. Check local logs for details.",
                "result": {"status": "error", "route": "write"},
            }

        if reply.pending is None:
            self.pending_writes.pop(key, None)
        else:
            self.pending_writes[key] = reply.pending

        return {
            "answer": reply.message,
            "result": {
                "status": reply.status,
                "route": "write",
                **reply.details,
            },
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
        elif normalized in ACKNOWLEDGEMENTS:
            answer = (
                "Okay! Ask about your documents or database, or add an order "
                "such as: Uday bought iPhone for 80k"
            )
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
        pipeline = self._get_sql_pipeline(
            state["thread_id"],
            state.get("user_id", "local-user"),
        )

        if pipeline.conversation_state.status == "awaiting_clarification":
            question = state["question"]
        else:
            question = state.get("rewritten_question") or state["question"]

        try:
            result = pipeline.run(question)

            if result["status"] in {"clarification_required", "not_found"}:
                answer = result["message"]
            elif result["status"] == "success":
                answer = self._format_sql_result(
                    result,
                    self._executor_columns(pipeline),
                )
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
            answer_result = self._get_rag_service().ask(
                state.get("rewritten_question") or state["question"]
            )

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
        question = state.get("rewritten_question") or state["question"]

        try:
            sql_pipeline = self._get_sql_pipeline(
                state["thread_id"],
                state.get("user_id", "local-user"),
            )
            sql_result = sql_pipeline.run(question)

            if sql_result["status"] == "success":
                sql_answer = self._format_sql_result(
                    sql_result,
                    self._executor_columns(sql_pipeline),
                )
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
    def _executor_columns(pipeline: Any) -> list[str]:
        """Column names of the pipeline's last query, or [] if unknown."""
        columns = getattr(
            getattr(pipeline, "sql_executor", None),
            "last_columns",
            None,
        )

        if isinstance(columns, (list, tuple)):
            return [str(column) for column in columns]

        return []

    @staticmethod
    def _format_cell(value: Any) -> str:
        """Render one database value as readable table text."""
        if value is None:
            return "NULL"

        text = format(value, "f") if isinstance(value, Decimal) else str(value)

        return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _format_sql_result(
        result: dict[str, Any],
        columns: list[str] | None = None,
    ) -> str:
        rows = result.get("results", [])

        if not rows:
            return "The query completed successfully, but no records matched."

        columns = list(columns or [])

        # With known column names, show a readable table.
        if columns and all(len(row) == len(columns) for row in rows):
            cell = EnterpriseWorkflow._format_cell
            header = "| " + " | ".join(cell(name) for name in columns) + " |"
            divider = "| " + " | ".join("---" for _ in columns) + " |"
            body = "\n".join(
                "| " + " | ".join(cell(value) for value in row) + " |"
                for row in rows
            )

            return (
                "Query completed successfully.\n\n"
                f"SQL:\n{result['sql']}\n\n"
                f"Results:\n\n{header}\n{divider}\n{body}"
            )

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

        remember_prefix = "remember that "

        if question.casefold().startswith(remember_prefix):
            fact = question[len(remember_prefix):].strip().rstrip(" .")

            if not fact:
                return {
                    "answer": "Please tell me what you'd like me to remember.",
                    "error": "Empty memory.",
                }

            sensitive_terms = (
                "password",
                "api key",
                "secret key",
                "access token",
                "credit card",
                "social security number",
            )

            if any(term in fact.casefold() for term in sensitive_terms):
                return {
                    "answer": (
                        "I can't save passwords, keys, tokens, or "
                        "payment or identity secrets as memories."
                    ),
                    "error": "Sensitive memory rejected.",
                }

            memory_key = (
                "user_note_"
                + hashlib.sha256(
                    fact.casefold().encode("utf-8")
                ).hexdigest()[:16]
            )

            try:
                self.long_term_memory.save(
                    user_id=user_id,
                    memory_key=memory_key,
                    memory_value=fact,
                )
            except Exception:
                return {
                    "answer": (
                        "I couldn't save that memory because persistent "
                        "memory is currently unavailable. Please try again."
                    ),
                    "error": "Memory persistence failed.",
                }

            return {
                "answer": f"I'll remember that: {fact}",
                "result": {
                    "status": "success",
                    "route": "memory",
                },
            }

        state: WorkflowState = {
            "question": question,
            "thread_id": thread_id,
            "user_id": user_id,
        }

        conversation_id = f"{user_id}:{thread_id}"

        return self.graph.invoke(
            state,
            config={"configurable": {"thread_id": conversation_id}},
        )