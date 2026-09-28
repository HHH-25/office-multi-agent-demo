from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.llm import get_llm
from src.retriever import search_knowledge
from src.state import AgentLog, GraphState

MAX_HOPS = 6


def _append_log(state: GraphState, agent: str, title: str, content: str) -> list[AgentLog]:
    logs = list(state.get("logs") or [])
    logs.append({"agent": agent, "title": title, "content": content})
    return logs


def _text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(item.get("text", item) if isinstance(item, dict) else item) for item in content)
    return str(content).strip()


def _parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _rule_next(state: GraphState) -> str:
    if not state.get("evidence"):
        return "researcher"
    if not state.get("draft"):
        return "writer"
    if not state.get("review_notes"):
        return "reviewer"
    if state.get("approved"):
        return "finish"
    if int(state.get("revision_count") or 0) < 1:
        return "writer"
    return "finish"


def supervisor_node(state: GraphState) -> dict[str, Any]:
    hops = int(state.get("hops") or 0) + 1
    if hops > MAX_HOPS:
        answer = state.get("draft") or "超过最大协作轮次，已停止，请缩小问题范围后重试。"
        return {
            "hops": hops,
            "next_agent": "finish",
            "answer": answer,
            "logs": _append_log(state, "supervisor", "停止", "达到最大轮次，输出当前草稿。"),
        }

    fallback = _rule_next(state)
    llm = get_llm(0.1)
    status = (
        f"已检索证据：{'是' if state.get('evidence') else '否'}\n"
        f"已有草稿：{'是' if state.get('draft') else '否'}\n"
        f"已复核：{'是' if state.get('review_notes') else '否'}\n"
        f"复核是否通过：{state.get('approved')}\n"
        f"已改写次数：{state.get('revision_count') or 0}"
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是办公助手的协调 Agent。根据当前进度决定下一个角色。"
                    "只输出 JSON：{\"next\":\"researcher|writer|reviewer|finish\",\"reason\":\"一句话\"}。"
                    "researcher=检索制度，writer=写答复，reviewer=核对是否越界，finish=可以给用户终稿。"
                    "没有证据时不要 finish。没有草稿时不要 reviewer。"
                )
            ),
            HumanMessage(content=f"用户问题：{state['question']}\n\n当前进度：\n{status}"),
        ]
    )
    parsed = _parse_json(_text(response)) or {}
    nxt = str(parsed.get("next") or fallback)
    if nxt not in {"researcher", "writer", "reviewer", "finish"}:
        nxt = fallback
    # 规则兜底：避免模型跳步导致空答复
    if nxt == "finish" and not state.get("draft"):
        nxt = fallback
    if nxt == "reviewer" and not state.get("draft"):
        nxt = fallback
    if nxt == "writer" and not state.get("evidence"):
        nxt = "researcher"

    reason = str(parsed.get("reason") or f"按进度进入 {nxt}")
    return {
        "hops": hops,
        "next_agent": nxt,
        "logs": _append_log(state, "supervisor", "分派任务", f"下一步：{nxt}\n原因：{reason}"),
    }


def researcher_node(state: GraphState) -> dict[str, Any]:
    llm = get_llm(0.1)
    query_resp = llm.invoke(
        [
            SystemMessage(content="从用户问题中抽出 3 到 6 个检索词，用空格分隔，不要解释。"),
            HumanMessage(content=state["question"]),
        ]
    )
    query = _text(query_resp) or state["question"]
    hits = search_knowledge(f"{state['question']} {query}", top_k=3)
    if not hits:
        hits = ["知识库未检索到直接对应的制度条款。"]
    summary_resp = llm.invoke(
        [
            SystemMessage(
                content="你是制度检索 Agent。只根据摘录归纳与问题相关的要点。没有的信息写「未提及」，禁止编造条款。"
            ),
            HumanMessage(content=f"问题：{state['question']}\n\n摘录：\n" + "\n\n---\n\n".join(hits)),
        ]
    )
    evidence = hits + [f"检索摘要：\n{_text(summary_resp)}"]
    return {
        "evidence": evidence,
        "next_agent": "supervisor",
        "logs": _append_log(state, "researcher", "检索制度", f"检索词：{query}\n\n" + "\n\n".join(evidence)),
    }


def writer_node(state: GraphState) -> dict[str, Any]:
    llm = get_llm(0.3)
    review_notes = state.get("review_notes") or ""
    extra = f"\n复核意见，必须按此修改：\n{review_notes}" if review_notes else ""
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是办公撰写 Agent。根据制度证据写给员工看的中文答复。"
                    "要求：1) 只使用证据里出现的规定；2) 引用制度名称；"
                    "3) 证据没有的内容明确说转人工，不要承诺可以批准或可以报销；"
                    "4) 结构为：结论 / 依据 / 建议下一步。"
                )
            ),
            HumanMessage(
                content=(
                    f"问题：{state['question']}\n\n证据：\n"
                    + "\n\n".join(state.get("evidence") or [])
                    + extra
                )
            ),
        ]
    )
    draft = _text(response)
    revision = int(state.get("revision_count") or 0)
    if review_notes:
        revision += 1
    return {
        "draft": draft,
        "review_notes": "",
        "approved": False,
        "revision_count": revision,
        "next_agent": "supervisor",
        "logs": _append_log(state, "writer", "生成草稿", draft),
    }


def reviewer_node(state: GraphState) -> dict[str, Any]:
    llm = get_llm(0.0)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是边界复核 Agent。检查草稿是否超出制度证据。"
                    "只输出 JSON："
                    '{"approved":true/false,"notes":"问题列表或通过说明"}。'
                    "出现编造条款、代替审批、断言未覆盖费用可报销时，approved 必须为 false。"
                )
            ),
            HumanMessage(
                content=(
                    f"问题：{state['question']}\n\n证据：\n"
                    + "\n\n".join(state.get("evidence") or [])
                    + f"\n\n草稿：\n{state.get('draft')}"
                )
            ),
        ]
    )
    raw = _text(response)
    parsed = _parse_json(raw) or {}
    approved = bool(parsed.get("approved"))
    notes = str(parsed.get("notes") or raw)
    result: dict[str, Any] = {
        "review_notes": notes,
        "approved": approved,
        "next_agent": "supervisor",
        "logs": _append_log(state, "reviewer", "边界复核", f"通过：{approved}\n{notes}"),
    }
    if approved:
        result["answer"] = state.get("draft") or ""
    elif int(state.get("revision_count") or 0) >= 1:
        warning = "【复核未完全通过，以下答复已限制承诺】\n\n"
        result["answer"] = warning + (state.get("draft") or "")
        result["approved"] = False
    return result


def route_from_supervisor(state: GraphState) -> str:
    nxt = state.get("next_agent") or "finish"
    return nxt if nxt in {"researcher", "writer", "reviewer", "finish"} else "finish"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "finish": END,
        },
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")
    return graph.compile()


def run_question(question: str) -> GraphState:
    app = build_graph()
    return app.invoke(
        {
            "question": question.strip(),
            "hops": 0,
            "revision_count": 0,
            "logs": [],
        }
    )
