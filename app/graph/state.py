from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    question: str
    thread_id: str
    user_id: str
    memories: list[dict[str, Any]]
    conversation_history: list[dict[str, str]]
    route: str
    answer: str
    result: dict[str, Any]
    error: str
