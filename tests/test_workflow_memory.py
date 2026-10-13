
from unittest.mock import MagicMock, patch

from app.graph.workflow import EnterpriseWorkflow


def make_workflow():
    memory = MagicMock()
    workflow = EnterpriseWorkflow(long_term_memory=memory)
    return workflow, memory


def test_route_retrieves_memories_for_requested_user():
    workflow, memory = make_workflow()
    memory.search.return_value = [
        {
            "memory_key": "project",
            "memory_value": "Building an Advanced RAG system",
        }
    ]

    result = workflow.route_node(
        {
            "question": "Tell me about my project",
            "thread_id": "thread-1",
            "user_id": "user-1",
        }
    )

    memory.search.assert_called_once_with(
        user_id="user-1",
        query_text="Tell me about my project",
        limit=5,
    )
    assert result["memories"][0]["memory_key"] == "project"


def test_memory_failure_does_not_break_general_routing():
    workflow, memory = make_workflow()
    memory.search.side_effect = RuntimeError("Database unavailable")

    result = workflow.route_node(
        {
            "question": "hello",
            "thread_id": "thread-1",
            "user_id": "user-1",
        }
    )

    assert result["route"] == "general"
    assert result["memories"] == []


def test_general_response_uses_retrieved_memory():
    workflow, _ = make_workflow()

    response = MagicMock()
    response.content = "Your project is an Advanced RAG system."

    llm = MagicMock()
    llm.invoke.return_value = response

    with patch(
        "app.graph.workflow.OllamaClient"
    ) as ollama_client:
        ollama_client.return_value.get_chat_model.return_value = llm

        result = workflow.general_node(
            {
                "question": "What is my project?",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "memories": [
                    {
                        "memory_key": "project",
                        "memory_value": "Building an Advanced RAG system",
                    }
                ],
            }
        )

    assert result["answer"] == (
        "Your project is an Advanced RAG system."
    )

    prompt = llm.invoke.call_args.args[0]
    assert "Building an Advanced RAG system" in prompt
    assert "What is my project?" in prompt
