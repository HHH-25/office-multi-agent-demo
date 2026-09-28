from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.graph import run_question
from src.retriever import load_documents

st.set_page_config(page_title="办公制度多智能体 Demo", page_icon="🧭", layout="wide")

EXAMPLES = [
    "请 3 天年假要走什么流程？需要谁审批？",
    "出差住宿超标能不能报销？",
    "帮我按制度写一份内部例会纪要必须包含哪些字段。",
]

st.title("办公制度多智能体 Demo")
st.caption("LangGraph：协调 → 检索 → 冲突检测 → 撰写 → 复核")

with st.sidebar:
    st.subheader("本地知识库")
    for doc in load_documents():
        st.markdown(f"- `{doc['path']}`")
    st.divider()
    st.markdown(
        "这是练习项目。检索用关键词，正式办公系统可换成向量库。"
        "多份制度口径不一致时会先标出冲突，不擅自采信其中一份。"
    )

question = st.selectbox("示例问题", EXAMPLES)
custom = st.text_input("或自己输入", value="")
query = custom.strip() or question

if st.button("开始协作", type="primary"):
    with st.spinner("各节点正在协作…"):
        result = run_question(query)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("协作轨迹")
        for log in result.get("logs") or []:
            with st.expander(f"{log['agent']} · {log['title']}", expanded=False):
                st.markdown(log["content"])
    with right:
        st.subheader("最终答复")
        st.markdown(result.get("answer") or result.get("draft") or "没有生成答复。")
        st.caption(f"协作轮次：{result.get('hops', '-')} ｜ 复核通过：{result.get('approved')}")
else:
    st.info("选一个问题后点击「开始协作」。页面会展示每个 Agent 做了什么。")
