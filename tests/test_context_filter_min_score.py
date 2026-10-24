import pytest
from langchain_core.documents import Document

from app.rag.context_filter import ContextFilter


def doc(text, score=None, source="a.txt"):
    metadata = {"source": source}
    if score is not None:
        metadata["rerank_score"] = score
    return Document(page_content=text, metadata=metadata)


# --- behavior that must not change when min_score is not set ------------

def test_default_keeps_everything_regardless_of_score():
    docs = [doc("one", -11.3), doc("two", 7.9), doc("three")]

    assert [d.page_content for d in ContextFilter().filter(docs)] == [
        "one",
        "two",
        "three",
    ]


def test_dedups_identical_text_across_sources():
    docs = [doc("same", source="a.txt"), doc("same", source="b.txt")]

    assert len(ContextFilter().filter(docs)) == 1


def test_respects_max_chunks_and_max_characters():
    docs = [doc("aaaa"), doc("bbbb"), doc("cccc")]

    assert len(ContextFilter(max_chunks=2).filter(docs)) == 2

    limited = ContextFilter(max_characters=6).filter(docs)
    assert sum(len(d.page_content) for d in limited) <= 6


# --- score floor ----------------------------------------------------------

def test_drops_chunks_below_floor_and_keeps_those_at_or_above():
    docs = [doc("hi", 1.0), doc("edge", -10.0), doc("low", -10.01)]

    kept = ContextFilter(min_score=-10.0).filter(docs)

    assert [d.page_content for d in kept] == ["hi", "edge"]


def test_unscored_documents_are_kept():
    docs = [doc("no score"), doc("bad", -50.0)]

    kept = ContextFilter(min_score=-10.0).filter(docs)

    assert [d.page_content for d in kept] == ["no score"]


def test_non_finite_scores_are_dropped():
    docs = [doc("nan", float("nan")), doc("inf", float("-inf")), doc("ok", 0.0)]

    kept = ContextFilter(min_score=-10.0).filter(docs)

    assert [d.page_content for d in kept] == ["ok"]


def test_kept_chunks_retain_their_scores():
    kept = ContextFilter(min_score=-10.0).filter([doc("x", 7.9)])

    assert kept[0].metadata["rerank_score"] == 7.9


# --- the scores observed on the real MiniLM reranker ---------------------

def test_apollo_query_keeps_only_the_relevant_chunk():
    docs = [
        doc("apollo", 7.887698173522949, "rag_smoke_test.txt"),
        doc("edu", -11.184951782226562, "RESUME.pdf"),
        doc("contact", -11.28866958618164, "RESUME.pdf"),
        doc("micrograd", -11.321173667907715, "RESUME.pdf"),
    ]

    kept = ContextFilter(min_score=-10.0).filter(docs)

    assert [d.page_content for d in kept] == ["apollo"]


def test_resume_query_keeps_relevant_negative_scores_drops_apollo():
    docs = [
        doc("projects", -0.38925179839134216, "RESUME.pdf"),
        doc("summary", -8.212560653686523, "RESUME.pdf"),
        doc("skills", -8.735447883605957, "RESUME.pdf"),
        doc("apollo", -10.9638671875, "rag_smoke_test.txt"),
    ]

    kept = ContextFilter(min_score=-10.0).filter(docs)

    assert [d.page_content for d in kept] == ["projects", "summary", "skills"]


def test_off_topic_query_returns_no_evidence():
    docs = [
        doc("a", -11.087944984436035),
        doc("b", -11.28174877166748),
        doc("c", -11.284297943115234),
        doc("d", -11.305734634399414),
    ]

    assert ContextFilter(min_score=-10.0).filter(docs) == []


# --- validation -----------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", True])
def test_rejects_invalid_min_score(bad):
    with pytest.raises(ValueError, match="min_score"):
        ContextFilter(min_score=bad)