import pytest
from langchain_core.documents import Document

from app.rag.answer_generator import GroundedAnswerGenerator


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""

    def invoke(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def test_generates_answer_with_valid_citation():
    llm = FakeLLM("Customer data must be protected. [S1]")
    generator = GroundedAnswerGenerator(llm=llm)

    result = generator.generate(
        "How should data be protected?",
        [
            Document(
                page_content="Employees must protect customer data.",
                metadata={"source": "security.txt", "page": 2},
            )
        ],
    )

    assert result.answer == "Customer data must be protected. [S1]"
    assert result.citation_ids == ["S1"]
    assert result.sources[0]["source"] == "security.txt"
    assert result.sources[0]["page"] == 2


def test_removes_unknown_citation():
    llm = FakeLLM("Data must be protected. [S1] [S99]")
    generator = GroundedAnswerGenerator(llm=llm)

    result = generator.generate(
        "How should data be protected?",
        [Document(page_content="Protect customer data.")],
    )

    assert "[S1]" in result.answer
    assert "[S99]" not in result.answer
    assert result.citation_ids == ["S1"]


def test_returns_abstention_when_no_documents():
    llm = FakeLLM("This should not be called.")
    generator = GroundedAnswerGenerator(llm=llm)

    result = generator.generate("What is our policy?", [])

    assert "don't have enough information" in result.answer
    assert result.sources == []
    assert result.citation_ids == []
    assert llm.last_prompt == ""


def test_rejects_empty_query():
    generator = GroundedAnswerGenerator(llm=FakeLLM("unused"))

    with pytest.raises(ValueError, match="Query must not be empty"):
        generator.generate("  ", [Document(page_content="Evidence")])


def test_prompt_includes_evidence_and_grounding_rules():
    llm = FakeLLM("The policy protects customer data. [S1]")
    generator = GroundedAnswerGenerator(llm=llm)

    generator.generate(
        "What does the policy protect?",
        [
            Document(
                page_content="The policy protects customer data.",
                metadata={"source": "policy.txt"},
            )
        ],
    )

    assert "policy protects customer data" in llm.last_prompt
    assert "Do not invent facts" in llm.last_prompt
    assert "[S1]" in llm.last_prompt


def test_empty_model_response_uses_fallback():
    generator = GroundedAnswerGenerator(llm=FakeLLM(""))

    result = generator.generate(
        "What does the document say?",
        [Document(page_content="Some evidence.", metadata={"source": "a.txt"})],
    )

    assert "couldn't generate an answer" in result.answer