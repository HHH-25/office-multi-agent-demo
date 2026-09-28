from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import run_question


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="办公制度多智能体 Demo")
    parser.add_argument("question", nargs="?", default="请 3 天年假要走什么流程？需要谁审批？")
    parser.add_argument("--save", action="store_true", help="把本次运行写入 traces/sample_run.md")
    args = parser.parse_args()

    result = run_question(args.question)
    print("=" * 60)
    print("问题：", result.get("question"))
    print("=" * 60)
    for log in result.get("logs") or []:
        print(f"\n[{log['agent']}] {log['title']}")
        print(log["content"])
        print("-" * 40)
    print("\n最终答复：")
    print(result.get("answer") or result.get("draft") or "（无输出）")

    if args.save:
        traces = ROOT / "traces"
        traces.mkdir(exist_ok=True)
        lines = [
            "# 一次真实运行记录",
            "",
            f"**问题**：{result.get('question')}",
            "",
        ]
        for log in result.get("logs") or []:
            lines.extend([f"## {log['agent']} · {log['title']}", "", log["content"], ""])
        lines.extend(["## 最终答复", "", result.get("answer") or result.get("draft") or ""])
        (traces / "sample_run.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\n已写入 {traces / 'sample_run.md'}")


if __name__ == "__main__":
    main()
