import json
from pathlib import Path

from app.evaluation.runner import EvaluationCase


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load and validate retrieval evaluation cases from a JSON file."""
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    cases = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Case {index} must be a JSON object.")

        query = item.get("query")
        relevant_ids = item.get("relevant_ids")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Case {index} must have a non-empty query.")

        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(
                not isinstance(doc_id, str) or not doc_id.strip()
                for doc_id in relevant_ids
            )
        ):
            raise ValueError(
                f"Case {index} must have a non-empty list of document IDs."
            )

        cases.append(
            EvaluationCase(
                query=query.strip(),
                relevant_ids=frozenset(relevant_ids),
            )
        )

    return cases
