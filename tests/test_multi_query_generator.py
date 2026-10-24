from unittest.mock import MagicMock

import pytest

from app.rag.multi_query_generator import MultiQueryGenerator


def test_returns_original_and_distinct_alternatives():
    llm = MagicMock()
    llm.invoke.return_value = (
        "sales revenue decline\n"
        "Why did sales revenue decline?\n"
        "sales revenue decline"
    )
    generator = MultiQueryGenerator(
        llm_client=llm,
        max_alternatives=3,
    )

    result = generator.generate("Why did sales revenue decline?")

    assert result == [
        "Why did sales revenue decline?",
        "sales revenue decline",
    ]


def test_removes_numbering_and_bullets():
    llm = MagicMock()
    llm.invoke.return_value = (
        "1. Customer order trends\n"
        "2) Order volume analysis\n"
        "- Sales performance"
    )
    generator = MultiQueryGenerator(
    llm_client=llm,
    max_alternatives=3,
)
    result = generator.generate("Show sales")

    assert result == [
        "Show sales",
        "Customer order trends",
        "Order volume analysis",
        "Sales performance",
    ]


def test_limits_number_of_alternatives():
    llm = MagicMock()
    llm.invoke.return_value = "Alternative A\nAlternative B\nAlternative C"
    generator = MultiQueryGenerator(
        llm_client=llm,
        max_alternatives=2,
    )

    result = generator.generate("Original query")

    assert result == ["Original query", "Alternative A", "Alternative B"]


def test_llm_failure_falls_back_to_original():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("Ollama unavailable")
    generator = MultiQueryGenerator(llm_client=llm)

    assert generator.generate("Original query") == ["Original query"]


@pytest.mark.parametrize("question", ["", "   "])
def test_rejects_empty_question(question):
    generator = MultiQueryGenerator(llm_client=MagicMock())

    with pytest.raises(ValueError, match="Question must not be empty"):
        generator.generate(question)


def test_zero_alternatives_does_not_call_llm():
    llm = MagicMock()
    generator = MultiQueryGenerator(
        llm_client=llm,
        max_alternatives=0,
    )

    assert generator.generate("Original query") == ["Original query"]
    llm.invoke.assert_not_called()