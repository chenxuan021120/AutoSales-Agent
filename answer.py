"""最终回答器。没有模型 Key 时仍能展示 RAG 检索是否正确。"""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import settings


def format_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        entity = hit["entity"]
        title = f"；标题：{entity['title']}" if entity.get("title") else ""
        source = entity.get("source", "未知来源")
        chunk = f"（片段 {entity['chunk_index']}）" if entity.get("chunk_index") is not None else ""
        blocks.append(f"[{i}] 来源：{source}{chunk}{title}\n{entity.get('text', '')}")
    return "\n\n".join(blocks)


def answer(question: str, hits: list[dict]) -> str:
    context = format_context(hits)
    if not settings.openai_api_key:
        return "未配置 OPENAI_API_KEY，以下是检索到的证据（先确认 RAG 是否召回正确）：\n\n" + context

    prompt = f"""你是星辰电动 ES9 的销售助手。只能依据参考资料回答；资料未提及就明确说明。
每个事实后标注 [编号]，不要编造参数或价格。来源为“购车计算工具”时，必须原样采用其计算结果、展示其假设和提示，不要自行改算。

参考资料：
{context}

用户问题：{question}"""
    endpoint = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    request = Request(
        endpoint,
        data=json.dumps(
            {"model": settings.openai_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        ).encode("utf-8"),
        headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    print("[LLM] 正在请求 qwen-plus…", flush=True)
    try:
        with urlopen(request, timeout=45) as http_response:
            response = json.loads(http_response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"大模型请求失败（HTTP {error.code}）：{detail}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError("大模型网络请求失败或超过 45 秒。请检查网络和 OPENAI_BASE_URL。") from error
    return response["choices"][0]["message"]["content"] or "模型未返回文本。"
