from unittest.mock import MagicMock

from app.rag.query_rewriter import QueryRewriter


def test_no_history_returns_original_question():
    llm = MagicMock()
    rewriter = QueryRewriter(llm_client=llm)

    result = rewriter.rewrite("What about 2024?", [])

    assert result == "What about 2024?"
    llm.invoke.assert_not_called()


def test_rewrites_follow_up_using_conversation_history():
    llm = MagicMock()
    llm.invoke.return_value = (
        "Show me the total orders in 2024."
    )
    rewriter = QueryRewriter(llm_client=llm)

    result = rewriter.rewrite(
        "What about 2024?",
        [
            {
                "question": "Show me the total orders in 2025.",
                "answer": "There were 120 orders in 2025.",
            }
        ],
    )

    assert result == "Show me the total orders in 2024."
    prompt = llm.invoke.call_args.args[0]
    assert "Show me the total orders in 2025." in prompt
    assert "What about 2024?" in prompt


def test_empty_question_returns_empty_string():
    rewriter = QueryRewriter(llm_client=MagicMock())

    assert rewriter.rewrite("   ", []) == ""


def test_llm_failure_returns_original_question():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("Ollama unavailable")
    rewriter = QueryRewriter(llm_client=llm)

    result = rewriter.rewrite(
        "What about 2024?",
        [
            {
                "question": "Show me the total orders in 2025.",
                "answer": "There were 120 orders in 2025.",
            }
        ],
    )

    assert result == "What about 2024?"


def test_multiline_llm_output_falls_back_to_original_question():
    llm = MagicMock()
    llm.invoke.return_value = (
        "Reason: prior context found\n"
        "Show me the total orders in 2024."
    )
    rewriter = QueryRewriter(llm_client=llm)

    result = rewriter.rewrite(
        "What about 2024?",
        [{"question": "Show orders in 2025.", "answer": "120 orders."}],
    )

    assert result == "What about 2024?"