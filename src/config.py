from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KNOWLEDGE_DIR = ROOT / "knowledge"


def env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


PROVIDER = env("LLM_PROVIDER", "deepseek").lower()
