from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    question: str
    thread_id: str
    user_id: str
    memories: list[dict[str, Any]]
    route: str
    answer: str
    result: dict[str, Any]
    error: str