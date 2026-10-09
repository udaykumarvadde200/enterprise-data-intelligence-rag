from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    question: str
    thread_id: str
    route: str
    answer: str
    result: dict[str, Any]
    error: str