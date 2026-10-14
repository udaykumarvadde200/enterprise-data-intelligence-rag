import json

import pytest

from app.evaluation.dataset import load_evaluation_cases


def test_loads_evaluation_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([
            {
                "query": "  Find security policy  ",
                "relevant_ids": ["security_policy"],
            }
        ]),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(path)

    assert len(cases) == 1
    assert cases[0].query == "Find security policy"
    assert cases[0].relevant_ids == frozenset({"security_policy"})


def test_rejects_empty_dataset(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty JSON list"):
        load_evaluation_cases(path)


def test_rejects_empty_query(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{"query": " ", "relevant_ids": ["doc-1"]}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty query"):
        load_evaluation_cases(path)


def test_rejects_missing_relevant_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{"query": "Find security policy", "relevant_ids": []}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty list of document IDs"):
        load_evaluation_cases(path)


def test_rejects_invalid_json(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_evaluation_cases(path)
