# 办公制度多智能体 Demo

给面试官看的 **LangGraph 多 Agent 练习项目**。场景是企业内部制度问答：先查制度，再写答复，再检查有没有越权承诺。

> 这是独立练习项目，不是线上产品。本地 `streamlit run app.py` 即可试用。

## 四个 Agent

```text
用户问题
   │
   ▼
supervisor  协调：看进度，决定下一步
   │
   ├─ researcher  检索：从本地制度里找依据
   ├─ writer      撰写：只根据证据写答复
   └─ reviewer    复核：拦住编造条款、代替审批、乱承诺报销
   │
   ▼
最终答复（结论 / 依据 / 建议下一步）
```

| 角色 | 职责 | 为什么拆开 |
| --- | --- | --- |
| 协调 | 路由，并限制最大 6 轮 | 避免单次 Prompt 里既检索又拍板 |
| 检索 | 本地制度关键词检索 + 摘要 | 证据和生成分开，方便核对 |
| 撰写 | 按制度写给员工看的答复 | 只消费证据，不自己发明流程 |
| 复核 | JSON 判定是否通过 | 对应办公场景的边界控制 |

检索没有上向量库，是为了 **clone 就能跑**。正式项目可以把 `src/retriever.py` 换成 Chroma + BGE，图结构不用改。

## 怎么跑

需要 Python 3.10+。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

在 `.env` 里填一种模型即可：

- `LLM_PROVIDER=siliconflow`，填 `SILICONFLOW_API_KEY`
- 或 `LLM_PROVIDER=dashscope`，填 `DASHSCOPE_API_KEY`（通义千问）
- 或 `LLM_PROVIDER=deepseek`，填 `DEEPSEEK_API_KEY`
- 或 `LLM_PROVIDER=ollama`，本机已 `ollama serve` 并 pull 过对应模型

命令行：

```bash
python run_demo.py "请 3 天年假要走什么流程？需要谁审批？"
python run_demo.py --save
```

页面：

```bash
streamlit run app.py
```

浏览器打开终端提示的本地地址（一般是 http://localhost:8501）。

## 建议试的三个问题

1. 请 3 天年假要走什么流程？需要谁审批？
2. 出差住宿超标能不能报销？
3. 内部例会纪要必须包含哪些字段？

第 2 题适合看复核：制度写的是超标自理，答复里不应出现「可以报销」。

一次真实运行记录见 [`traces/sample_run.md`](traces/sample_run.md)。

## 代码入口

- `src/graph.py`：状态图和四个节点
- `src/retriever.py`：本地 `knowledge/*.md` 检索
- `knowledge/`：请假、报销、会议三份练习制度
- `app.py`：Streamlit 页面，展示每个 Agent 的输出

## 面试时可以怎么讲

1. 多 Agent 不是多个模型，而是 **同一张状态图上的分工**：共享 `question / evidence / draft / review`。
2. 协调节点用模型建议下一步，同时用规则兜底，防止没检索就结束。
3. 复核失败会让撰写再改一稿；仍不通过就带警告输出，而不是无限循环。
4. 办公场景宁可「转人工」，也不代替 OA 审批。
