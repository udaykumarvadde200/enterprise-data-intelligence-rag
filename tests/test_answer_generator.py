"""
Regression tests for per-claim grounding verification.

The scripted verifier below tests the PLUMBING: every claim is checked,
each claim sees only its cited evidence, any rejection abstains, and bad
verifier output fails closed. It cannot prove the real model judges
paraphrases correctly; that is covered by the opt-in live test at the
bottom (set RUN_LIVE_OLLAMA=1 with Ollama running).
"""
import json
import os
import re

import pytest
from langchain_core.documents import Document

from app.rag.answer_generator import GroundedAnswerGenerator

APOLLO_TEXT = (
    "Project: Apollo Enterprise\n"
    "Project Manager: Priya Sharma\n"
    "Project Location: Hyderabad\n"
    "Project Status: In Progress\n"
    "Target Completion Date: 30 December 2026\n"
    "Current Risk: Delayed delivery of the reporting module."
)

GOOD_ANSWER = (
    "According to the evidence, the project manager of Project Apollo "
    "Enterprise is Priya Sharma [S1]. The project is located in Hyderabad "
    "[S1]. The current risk is the delayed delivery of the reporting "
    "module [S1]."
)


def docs():
    return [
        Document(
            page_content=APOLLO_TEXT,
            metadata={"source": "rag_smoke_test.txt"},
        ),
        Document(
            page_content="Skills: Python, SQL, LangChain.",
            metadata={"source": "RESUME.pdf", "page": 1},
        ),
    ]


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, prompt):
        return self.answer


class ScriptedVerifier:
    """Parses CLAIM/EVIDENCE out of the prompt and applies a judge."""

    def __init__(self, judge=lambda claim, evidence: True):
        self.judge = judge
        self.calls = []

    def invoke(self, prompt):
        evidence = re.search(
            r"EVIDENCE:\n(.*?)\n\nCLAIM:", prompt, re.S
        ).group(1)
        claim = re.search(
            r"CLAIM: (.*?)\n\nIs the claim", prompt, re.S
        ).group(1)
        self.calls.append((claim, evidence))
        return json.dumps(
            {"reason": "scripted", "supported": self.judge(claim, evidence)}
        )


class RawVerifier:
    """Returns a fixed raw response, or raises it if it is an Exception."""

    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if isinstance(self.raw, Exception):
            raise self.raw
        return self.raw


def run(answer, verifier):
    generator = GroundedAnswerGenerator(
        llm=FakeLLM(answer), verifier=verifier
    )
    return generator.generate("Tell me about Apollo", docs())


ABSTAIN = GroundedAnswerGenerator.ABSTENTION


# --- accepted ---------------------------------------------------------

def test_supported_answer_is_accepted_and_citations_kept():
    verifier = ScriptedVerifier()
    result = run(GOOD_ANSWER, verifier)

    assert result.answer == GOOD_ANSWER
    assert result.citation_ids == ["S1"]
    assert len(result.sources) == 2


def test_each_claim_checked_separately_preamble_not_a_claim():
    verifier = ScriptedVerifier()
    run(GOOD_ANSWER, verifier)

    assert len(verifier.calls) == 3
    assert not verifier.calls[0][0].lower().startswith("according")
    assert "Priya Sharma" in verifier.calls[0][0]


def test_paraphrased_answer_is_accepted_when_judge_approves():
    answer = (
        "Priya Sharma manages Apollo Enterprise [S1]. "
        "It is based in Hyderabad [S1]."
    )
    result = run(answer, ScriptedVerifier())

    assert result.answer == answer
    assert result.citation_ids == ["S1"]


def test_citation_after_period_stays_with_its_claim():
    answer = "The project is located in Hyderabad. [S1]"
    verifier = ScriptedVerifier()
    result = run(answer, verifier)

    assert result.citation_ids == ["S1"]
    assert len(verifier.calls) == 1
    assert "Source location: rag_smoke_test.txt" in verifier.calls[0][1]


# --- rejected: content ------------------------------------------------

def test_unsupported_fact_is_rejected():
    answer = GOOD_ANSWER.replace(
        "the delayed delivery of the reporting module",
        "a cybersecurity breach",
    )
    verifier = ScriptedVerifier(lambda c, e: "cybersecurity" not in c)

    result = run(answer, verifier)

    assert result.answer == ABSTAIN
    assert result.citation_ids == []


def test_contradiction_is_rejected():
    answer = GOOD_ANSWER.replace("Hyderabad", "Delhi")
    verifier = ScriptedVerifier(lambda c, e: "Delhi" not in c)

    assert run(answer, verifier).answer == ABSTAIN


def test_mixed_supported_and_unsupported_claims_are_rejected():
    answer = (
        "Priya Sharma is the project manager [S1]. "
        "The project is located in Hyderabad [S1]. "
        "The budget is 5 crore [S1]."
    )
    verifier = ScriptedVerifier(lambda c, e: "budget" not in c)

    assert run(answer, verifier).answer == ABSTAIN


def test_claim_is_verified_only_against_its_own_cited_source():
    answer = (
        "Priya Sharma is the project manager [S1]. "
        "The candidate knows Python [S2]."
    )
    verifier = ScriptedVerifier()
    run(answer, verifier)

    assert "Priya Sharma" in verifier.calls[0][1]
    assert "RESUME.pdf" not in verifier.calls[0][1]
    assert "RESUME.pdf" in verifier.calls[1][1]
    assert "rag_smoke_test.txt" not in verifier.calls[1][1]


# --- rejected: citations ----------------------------------------------

def test_missing_citation_is_rejected_without_calling_verifier():
    verifier = ScriptedVerifier()
    result = run("The project manager is Priya Sharma.", verifier)

    assert result.answer == ABSTAIN
    assert verifier.calls == []


def test_invalid_citation_id_is_removed_valid_one_kept():
    answer = (
        "Priya Sharma is the project manager [S1]. "
        "The project is located in Hyderabad [S99]."
    )
    result = run(answer, ScriptedVerifier())

    assert "[S99]" not in result.answer
    assert result.citation_ids == ["S1"]


def test_answer_with_only_invalid_citations_is_rejected():
    verifier = ScriptedVerifier()
    result = run("Priya Sharma is the project manager [S99].", verifier)

    assert result.answer == ABSTAIN
    assert verifier.calls == []


# --- fail closed ------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        RuntimeError("ollama down"),
        "not json at all",
        "[]",
        "{}",
        '{"supported": "true"}',
        '{"supported": 1}',
        '{"supported": null}',
        '{"supported": true, "extra": 1}',
        '{"reason": 5, "supported": true}',
        None,
    ],
)
def test_verifier_errors_and_malformed_verdicts_fail_closed(raw):
    verifier = RawVerifier(raw)

    assert run(GOOD_ANSWER, verifier).answer == ABSTAIN
    assert verifier.calls >= 1


def test_well_formed_true_verdicts_are_accepted():
    for raw in ('{"supported": true}',
                '{"reason": "ok", "supported": true}'):
        assert run(GOOD_ANSWER, RawVerifier(raw)).answer == GOOD_ANSWER


def test_verifier_false_verdict_abstains():
    assert run(GOOD_ANSWER, RawVerifier('{"supported": false}')).answer == (
        ABSTAIN
    )


# --- claim splitting --------------------------------------------------

def test_split_claims_handles_bullets_and_preamble():
    claims = GroundedAnswerGenerator._split_claims(
        "According to the evidence, the manager is Priya [S1].\n"
        "- The location is Hyderabad [S1]\n"
        "- The status is In Progress [S1]"
    )

    assert claims == [
        "the manager is Priya [S1].",
        "The location is Hyderabad [S1]",
        "The status is In Progress [S1]",
    ]


def test_answer_with_no_claims_fails_closed():
    generator = GroundedAnswerGenerator(
        llm=FakeLLM("x"), verifier=ScriptedVerifier()
    )
    assert generator._verify_answer("[S1].", "[S1]\nEvidence:\nx") is False


# --- opt-in live check against the real local model ---------------------

@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_OLLAMA"),
    reason="set RUN_LIVE_OLLAMA=1 with Ollama + llama3.2:3b running",
)
@pytest.mark.parametrize(
    "answer, expected",
    [
        (GOOD_ANSWER, True),
        (GOOD_ANSWER.replace(
            "the delayed delivery of the reporting module",
            "a cybersecurity breach"), False),
        (GOOD_ANSWER.replace("Hyderabad", "Delhi"), False),
    ],
)
def test_live_verifier_apollo(answer, expected):
    generator = GroundedAnswerGenerator(llm=FakeLLM(answer))
    result = generator.generate("Tell me about Apollo", docs())

    assert (result.answer != ABSTAIN) is expected