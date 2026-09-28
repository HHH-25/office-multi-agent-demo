from typing import TypedDict


class AgentLog(TypedDict):
    agent: str
    title: str
    content: str


class GraphState(TypedDict, total=False):
    question: str
    hops: int
    revision_count: int
    next_agent: str
    evidence: list[str]
    draft: str
    review_notes: str
    approved: bool
    answer: str
    logs: list[AgentLog]
