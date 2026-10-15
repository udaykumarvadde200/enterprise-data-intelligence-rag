
import pytest
from langchain_core.documents import Document

from app.rag.answer_generator import GroundedAnswerGenerator


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("No fake response configured")
        return self.responses.pop(0)


def make_generator(answer, verdict='{"supported": true}'):
    llm = FakeLLM([answer])
    verifier = FakeLLM([verdict])
    return GroundedAnswerGenerator(llm=llm, verifier=verifier), llm, verifier


def test_generates_answer_with_valid_citation():
    generator, llm, verifier = make_generator(
        "Employees must protect customer data. [S1]"
    )

    result = generator.generate(
        "How should data be protected?",
        [
            Document(
                page_content="Employees must protect customer data.",
                metadata={"source": "security.txt", "page": 2},
            )
        ],
    )

    assert result.answer == "Employees must protect customer data. [S1]"
    assert result.citation_ids == ["S1"]
    assert result.sources[0]["source"] == "security.txt"
    assert result.sources[0]["page"] == 2
    assert len(llm.prompts) == 1
    assert len(verifier.prompts) == 1


def test_removes_unknown_citation():
    generator, _, _ = make_generator(
        "Data must be protected. [S1] [S99]"
    )

    result = generator.generate(
        "How should data be protected?",
        [Document(page_content="Protect customer data.")],
    )

    assert "[S1]" in result.answer
    assert "[S99]" not in result.answer
    assert result.citation_ids == ["S1"]


def test_returns_abstention_when_no_documents():
    llm = FakeLLM(["unused"])
    verifier = FakeLLM(['{"supported": true}'])
    generator = GroundedAnswerGenerator(llm=llm, verifier=verifier)

    result = generator.generate("What is our policy?", [])

    assert "don't have enough information" in result.answer
    assert result.sources == []
    assert result.citation_ids == []
    assert llm.prompts == []
    assert verifier.prompts == []


def test_rejects_empty_query():
    generator = GroundedAnswerGenerator(llm=FakeLLM(["unused"]))

    with pytest.raises(ValueError, match="Query must not be empty"):
        generator.generate("  ", [Document(page_content="Evidence")])


def test_prompt_includes_evidence_and_grounding_rules():
    generator, llm, _ = make_generator(
        "The policy protects customer data. [S1]"
    )

    generator.generate(
        "What does the policy protect?",
        [
            Document(
                page_content="The policy protects customer data.",
                metadata={"source": "policy.txt"},
            )
        ],
    )

    assert "policy protects customer data" in llm.prompts[0]
    assert "Use only facts explicitly supported" in llm.prompts[0]
    assert "[S1]" in llm.prompts[0]


def test_empty_model_response_uses_fallback():
    generator = GroundedAnswerGenerator(
        llm=FakeLLM([""]),
        verifier=FakeLLM(['{"supported": true}']),
    )

    result = generator.generate(
        "What does the document say?",
        [Document(page_content="Some evidence.", metadata={"source": "a.txt"})],
    )

    assert "couldn't generate an answer" in result.answer


def test_prompt_requires_evidence_for_every_factual_claim():
    generator, llm, verifier = make_generator(
        "Employees must protect customer information. [S1]"
    )

    generator.generate(
        "How should customer information be protected?",
        [
            Document(
                page_content=(
                    "Employees must protect confidential customer "
                    "information and personal data."
                ),
                metadata={"source": "security_policy.txt"},
            )
        ],
    )

    assert "Use only facts explicitly supported" in llm.prompts[0]
    assert "EVERY factual claim" in verifier.prompts[0]
    assert "citation existing is not proof" in verifier.prompts[0].lower()


def test_unsupported_answer_is_rejected():
    generator, _, _ = make_generator(
        "Employees must protect customer data. Access is limited "
        "to authorized personnel. [S1]",
        '{"supported": false}',
    )

    result = generator.generate(
        "How should customer data be protected?",
        [
            Document(
                page_content="Employees must protect customer data.",
                metadata={"source": "security.txt"},
            )
        ],
    )

    assert result.answer == generator.ABSTENTION
    assert result.citation_ids == []


def test_invalid_verifier_response_fails_closed():
    generator, _, _ = make_generator(
        "Employees must protect customer data. [S1]",
        "I think this is supported.",
    )

    result = generator.generate(
        "How should customer data be protected?",
        [Document(page_content="Employees must protect customer data.")],
    )

    assert result.answer == generator.ABSTENTION


def test_verifier_service_failure_fails_closed():
    generator = GroundedAnswerGenerator(
        llm=FakeLLM(["Employees must protect customer data. [S1]"]),
        verifier=FakeLLM([]),
    )

    result = generator.generate(
        "How should customer data be protected?",
        [Document(page_content="Employees must protect customer data.")],
    )

    assert result.answer == generator.ABSTENTION


def test_answer_without_valid_citations_is_rejected():
    generator, _, verifier = make_generator(
        "Employees must protect customer data. [S99]"
    )

    result = generator.generate(
        "How should customer data be protected?",
        [Document(page_content="Employees must protect customer data.")],
    )

    assert result.answer == generator.ABSTENTION
    assert result.citation_ids == []
    # No verification call is needed after all citations are removed.
    assert verifier.prompts == []
