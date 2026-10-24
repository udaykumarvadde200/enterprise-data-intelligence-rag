import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.rag.ollama_reranker import OllamaRelevanceReranker


class FakeOllama:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(content=self.response)


def doc(text, source="policy.txt"):
    return Document(page_content=text, metadata={"source": source})


def test_reranker_returns_documents_in_score_order():
    model = FakeOllama(json.dumps({
        "scores": [
            {"index": 0, "score": 20},
            {"index": 1, "score": 95},
            {"index": 2, "score": 60},
        ]
    }))
    reranker = OllamaRelevanceReranker(model=model)
    documents = [doc("First"), doc("Second"), doc("Third")]

    results = reranker.rerank("test query", documents, top_k=2)

    assert [item.page_content for item in results] == ["Second", "Third"]
    assert results[0].metadata["rerank_score"] == 95
    assert results[1].metadata["rerank_score"] == 60
    assert len(model.calls) == 1


def test_reranker_preserves_document_metadata():
    model = FakeOllama(json.dumps({
        "scores": [{"index": 0, "score": 80}]
    }))
    reranker = OllamaRelevanceReranker(model=model)

    result = reranker.rerank("query", [doc("Text")])[0]

    assert result.metadata["source"] == "policy.txt"
    assert result.metadata["rerank_score"] == 80


def test_reranker_returns_empty_for_no_documents():
    model = FakeOllama("{}")
    reranker = OllamaRelevanceReranker(model=model)

    assert reranker.rerank("query", []) == []
    assert model.calls == []


@pytest.mark.parametrize("query", ["", "   "])
def test_reranker_rejects_empty_query(query):
    reranker = OllamaRelevanceReranker(model=FakeOllama("{}"))

    with pytest.raises(ValueError, match="Query must not be empty"):
        reranker.rerank(query, [doc("Text")])


@pytest.mark.parametrize("response", [
    "not json",
    '{"scores": []}',
    '{"scores": [{"index": 0, "score": 101}]}',
    '{"scores": [{"index": 0, "score": 50}, {"index": 0, "score": 60}]}',
    '{"scores": [{"index": 2, "score": 50}]}',
    '{"scores": [{"index": 0, "score": "high"}]}',
])
def test_reranker_rejects_invalid_batch_response(response):
    reranker = OllamaRelevanceReranker(model=FakeOllama(response))

    with pytest.raises(ValueError):
        reranker.rerank("query", [doc("First"), doc("Second")])


def test_reranker_retries_when_model_returns_extra_score():
    model = FakeOllama(
        '{"scores": [{"index": 0, "score": 20}, '
        '{"index": 1, "score": 99}]}'
    )
    model.responses = [
        model.response,
        '{"scores": [{"index": 0, "score": 15}]}',
    ]

    def invoke(prompt):
        model.calls.append(prompt)
        response = model.responses.pop(0)
        return SimpleNamespace(content=response)

    model.invoke = invoke
    reranker = OllamaRelevanceReranker(model=model, max_retries=1)

    scores = reranker._scores("Apollo project risk?", [doc("Resume content")])

    assert scores == [15.0]
    assert len(model.calls) == 2
    assert "Correct the response" in model.calls[1]


def test_reranker_raises_after_retry_if_response_is_still_invalid():
    model = FakeOllama(
        '{"scores": [{"index": 0, "score": 20}, '
        '{"index": 1, "score": 99}]}'
    )
    reranker = OllamaRelevanceReranker(model=model, max_retries=1)

    with pytest.raises(ValueError, match="incorrect number of scores"):
        reranker._scores("Apollo project risk?", [doc("Resume content")])

    assert len(model.calls) == 2


def test_reranker_accepts_valid_single_document_response():
    model = FakeOllama('{"scores": [{"index": 0, "score": 12}]}')
    reranker = OllamaRelevanceReranker(model=model)

    assert reranker._scores("Apollo project risk?", [doc("Resume")]) == [12.0]
    assert len(model.calls) == 1
