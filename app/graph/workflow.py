from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.router import QueryRouter
from app.graph.state import WorkflowState
from app.sql.sql_pipeline import SQLPipeline


class EnterpriseWorkflow:
    """LangGraph workflow for SQL, RAG, and hybrid questions."""

    def __init__(self):
        self.router = QueryRouter()

        # Maintain a separate SQL pipeline per conversation.
        self.sql_pipelines: dict[str, SQLPipeline] = {}

        # Keeps graph state between turns for each thread_id.
        self.checkpointer = MemorySaver()

        builder = StateGraph(WorkflowState)

        builder.add_node("route", self.route_node)
        builder.add_node("sql", self.sql_node)
        builder.add_node("rag", self.rag_node)
        builder.add_node("hybrid", self.hybrid_node)

        builder.add_edge(START, "route")

        builder.add_conditional_edges(
            "route",
            self.select_route,
            {
                "sql": "sql",
                "rag": "rag",
                "hybrid": "hybrid",
            },
        )

        builder.add_edge("sql", END)
        builder.add_edge("rag", END)
        builder.add_edge("hybrid", END)

        self.graph = builder.compile(
            checkpointer=self.checkpointer
        )

    def _get_sql_pipeline(self, thread_id: str) -> SQLPipeline:
        if thread_id not in self.sql_pipelines:
            self.sql_pipelines[thread_id] = SQLPipeline()

        return self.sql_pipelines[thread_id]

    def route_node(self, state: WorkflowState) -> dict[str, Any]:
        question = state["question"]
        thread_id = state["thread_id"]

        pipeline = self._get_sql_pipeline(thread_id)

        # A clarification reply must return to SQL processing,
        # even if the reply itself contains no SQL keywords.
        if (
            pipeline.conversation_state.status
            == "awaiting_clarification"
        ):
            route = "sql"
        else:
            route = self.router.route(question)

        return {"route": route}

    def select_route(self, state: WorkflowState) -> str:
        return state["route"]

    def sql_node(self, state: WorkflowState) -> dict[str, Any]:
        pipeline = self._get_sql_pipeline(state["thread_id"])

        try:
            result = pipeline.run(state["question"])

            if result["status"] == "clarification_required":
                answer = result["message"]

            elif result["status"] == "not_found":
                answer = result["message"]

            elif result["status"] == "success":
                answer = self._format_sql_result(result)

            else:
                answer = str(result)

            return {
                "result": result,
                "answer": answer,
            }

        except Exception as exc:
            return {
                "error": str(exc),
                "answer": (
                    "I couldn't complete the database query. "
                    "Please check the query and try again."
                ),
            }

    def rag_node(self, state: WorkflowState) -> dict[str, Any]:
        # Implemented in the next RAG milestone.
        return {
            "answer": (
                "The document retrieval pipeline is not connected yet. "
                "We'll implement ingestion, hybrid retrieval, and "
                "grounded generation next."
            ),
            "result": {
                "status": "not_implemented",
                "route": "rag",
            },
        }

    def hybrid_node(self, state: WorkflowState) -> dict[str, Any]:
        # Implemented after both SQL and RAG nodes are working.
        return {
            "answer": (
                "Hybrid SQL + document answering is not connected yet. "
                "We'll enable it after the RAG pipeline is ready."
            ),
            "result": {
                "status": "not_implemented",
                "route": "hybrid",
            },
        }

    @staticmethod
    def _format_sql_result(result: dict[str, Any]) -> str:
        rows = result.get("results", [])

        if not rows:
            return "The query completed successfully, but no records matched."

        formatted_rows = "\n".join(
            str(row) for row in rows
        )

        return (
            f"Query completed successfully.\n\n"
            f"SQL:\n{result['sql']}\n\n"
            f"Results:\n{formatted_rows}"
        )

    def ask(
        self,
        question: str,
        thread_id: str = "default",
    ) -> dict[str, Any]:
        question = question.strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "error": "Empty question.",
            }

        state: WorkflowState = {
            "question": question,
            "thread_id": thread_id,
        }

        return self.graph.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                }
            },
        )