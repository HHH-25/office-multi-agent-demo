import re

from src.config import KNOWLEDGE_DIR


def _tokenize(text: str) -> set[str]:
    parts = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
    tokens: set[str] = set()
    for part in parts:
        if re.fullmatch(r"[a-zA-Z0-9]+", part):
            tokens.add(part)
            continue
        tokens.add(part)
        if len(part) >= 2:
            tokens.update(part[i : i + 2] for i in range(len(part) - 1))
    return tokens


def load_documents() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        docs.append(
            {
                "title": path.stem,
                "path": str(path.relative_to(KNOWLEDGE_DIR.parent)),
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return docs


def search_knowledge(query: str, top_k: int = 3) -> list[str]:
    """本地关键词检索。练习项目刻意不用向量库，保证零部署可跑。"""
    query_tokens = _tokenize(query)
    scored: list[tuple[float, dict[str, str], str]] = []

    for doc in load_documents():
        chunks = _split_chunks(doc["text"])
        for chunk in chunks:
            overlap = query_tokens & _tokenize(chunk)
            score = float(len(overlap))
            if any(token in chunk for token in query_tokens if len(token) >= 2):
                score += 1.5
            if score <= 0:
                continue
            scored.append((score, doc, chunk.strip()))

    scored.sort(key=lambda item: item[0], reverse=True)
    evidence: list[str] = []
    seen: set[str] = set()
    for _, doc, chunk in scored:
        key = chunk[:80]
        if key in seen:
            continue
        seen.add(key)
        evidence.append(f"来源：{doc['title']}\n{chunk}")
        if len(evidence) >= top_k:
            break
    return evidence


def _split_chunks(text: str, size: int = 280) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) > size and buf:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}".strip()
    if buf:
        chunks.append(buf)
    return chunks or [text]
