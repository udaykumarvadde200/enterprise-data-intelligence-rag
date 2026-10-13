
from unittest.mock import MagicMock, patch

from app.graph.workflow import EnterpriseWorkflow
from langgraph.graph import END, START, StateGraph
from app.graph.state import WorkflowState

from langgraph.graph import END, START, StateGraph
from app.graph.state import WorkflowState

def test_record_turn_keeps_only_last_ten_turns():
    workflow = EnterpriseWorkflow(long_term_memory=MagicMock())

    history = [
        {"question": f"Question {i}", "answer": f"Answer {i}"}
        for i in range(12)
    ]

    result = workflow.record_turn_node(
        {
            "question": "Question 12",
            "answer": "Answer 12",
            "conversation_history": history,
        }
    )

    assert len(result["conversation_history"]) == 10
    assert result["conversation_history"][0]["question"] == "Question 3"
    assert result["conversation_history"][-1]["question"] == "Question 12"


def test_general_response_includes_conversation_history():
    workflow = EnterpriseWorkflow(long_term_memory=MagicMock())

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="Your project uses PostgreSQL."
    )

    with patch("app.graph.workflow.OllamaClient") as ollama_client:
        ollama_client.return_value.get_chat_model.return_value = llm

        result = workflow.general_node(
            {
                "question": "Which database does it use?",
                "memories": [],
                "conversation_history": [
                    {
                        "question": "What database did we choose?",
                        "answer": "We chose PostgreSQL.",
                    }
                ],
            }
        )

    prompt = llm.invoke.call_args.args[0]

    assert "We chose PostgreSQL." in prompt
    assert "Which database does it use?" in prompt
    assert result["answer"] == "Your project uses PostgreSQL."


def test_empty_history_does_not_break_general_response():
    workflow = EnterpriseWorkflow(long_term_memory=MagicMock())

    result = workflow.general_node(
        {
            "question": "Hello",
            "memories": [],
            "conversation_history": [],
        }
    )

    assert result["answer"].startswith("Hi!")


def test_graph_restores_conversation_history_for_same_thread():
    memory = MagicMock()
    memory.search.return_value = []
    workflow = EnterpriseWorkflow(long_term_memory=memory)

    recorded_histories = []

    def fake_general_node(state):
        history = state.get("conversation_history", [])
        recorded_histories.append(list(history))

        return {
            "answer": (
                "First answer"
                if state["question"] == "Remember this fact"
                else f"History count: {len(history)}"
            )
        }

    # Replace only the general-response node so the test checks
    # graph checkpointing rather than the behavior of a real LLM.
    workflow.graph = (
        workflow.graph.builder.compile(checkpointer=workflow.checkpointer)
        if hasattr(workflow.graph, "builder")
        else workflow.graph
    )

    workflow.general_node = fake_general_node
    # Rebuild the graph using the patched node.
    
    builder = StateGraph(WorkflowState)
    builder.add_node("route", workflow.route_node)
    builder.add_node("general", fake_general_node)
    builder.add_node("sql", workflow.sql_node)
    builder.add_node("rag", workflow.rag_node)
    builder.add_node("hybrid", workflow.hybrid_node)
    builder.add_node("record_turn", workflow.record_turn_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        workflow.select_route,
        {
            "general": "general",
            "sql": "sql",
            "rag": "rag",
            "hybrid": "hybrid",
        },
    )
    for route in ("general", "sql", "rag", "hybrid"):
        builder.add_edge(route, "record_turn")
    builder.add_edge("record_turn", END)

    workflow.graph = builder.compile(checkpointer=workflow.checkpointer)

    first = workflow.ask(
        "Remember this fact",
        thread_id="memory-thread",
        user_id="memory-user",
    )
    second = workflow.ask(
        "What did I ask before?",
        thread_id="memory-thread",
        user_id="memory-user",
    )

    assert first["answer"] == "First answer"
    assert len(recorded_histories[1]) == 1
    assert recorded_histories[1][0]["question"] == "Remember this fact"
    assert recorded_histories[1][0]["answer"] == "First answer"
    assert second["answer"] == "History count: 1"




def test_different_threads_do_not_share_conversation_history():
    memory = MagicMock()
    memory.search.return_value = []
    workflow = EnterpriseWorkflow(long_term_memory=memory)

    observed_histories = []

    def fake_general_node(state):
        history = state.get("conversation_history", [])
        observed_histories.append(list(history))
        return {"answer": f"History count: {len(history)}"}

   

    builder = StateGraph(WorkflowState)
    builder.add_node("route", workflow.route_node)
    builder.add_node("general", fake_general_node)
    builder.add_node("sql", workflow.sql_node)
    builder.add_node("rag", workflow.rag_node)
    builder.add_node("hybrid", workflow.hybrid_node)
    builder.add_node("record_turn", workflow.record_turn_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        workflow.select_route,
        {
            "general": "general",
            "sql": "sql",
            "rag": "rag",
            "hybrid": "hybrid",
        },
    )
    for route in ("general", "sql", "rag", "hybrid"):
        builder.add_edge(route, "record_turn")
    builder.add_edge("record_turn", END)

    workflow.graph = builder.compile(checkpointer=workflow.checkpointer)

    workflow.ask(
        "First thread question",
        thread_id="thread-A",
        user_id="isolation-user",
    )
    workflow.ask(
        "Second thread question",
        thread_id="thread-B",
        user_id="isolation-user",
    )

    assert observed_histories[0] == []
    assert observed_histories[1] == []
