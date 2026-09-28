from langchain_openai import ChatOpenAI

from src.config import PROVIDER, env


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    if PROVIDER == "deepseek":
        api_key = env("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 后填写。")
        return ChatOpenAI(
            api_key=api_key,
            base_url=env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=env("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=temperature,
        )

    if PROVIDER == "dashscope":
        api_key = env("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY，请复制 .env.example 为 .env 后填写。")
        return ChatOpenAI(
            api_key=api_key,
            base_url=env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=env("DASHSCOPE_MODEL", "qwen-plus"),
            temperature=temperature,
        )

    if PROVIDER == "ollama":
        return ChatOpenAI(
            api_key="ollama",
            base_url=env("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
            model=env("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=temperature,
        )

    api_key = env("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 SILICONFLOW_API_KEY，请复制 .env.example 为 .env 后填写。")
    return ChatOpenAI(
        api_key=api_key,
        base_url=env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        model=env("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        temperature=temperature,
    )
