"""百炼内置联网搜索工具，复用已配置的 Qwen 账号。"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import settings

OFFICIAL_FINANCE_SITES = [
    "icbc.com.cn", "ccb.com", "boc.cn", "abchina.com", "bankcomm.com", "cmbchina.com",
]


def _verified_finance_content(content: str) -> str:
    """金融检索必须给出白名单官网链接，否则不能作为可用方案。"""
    urls = re.findall(r"https?://[^\s)>，]+", content)
    verified = [url for url in urls if any(urlparse(url).netloc.endswith(site) for site in OFFICIAL_FINANCE_SITES)]
    if verified:
        return content
    return (
        "本次联网检索未返回可验证的银行官网链接，因此不会把该结果当作金融方案提供。\n"
        "请换成“查询{城市}新能源汽车的工行、建行通用分期方案”，或以银行 App、官网、网点给出的正式报价为准。"
    )


def search(query: str, title: str = "联网检索结果", sites: list[str] | None = None) -> list[dict]:
    if not settings.openai_api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY，无法执行联网搜索。")
    endpoint = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    prompt = "请联网检索并整理最新可靠信息，优先官方来源；每条事实后附来源 URL。问题：" + query
    if sites:
        prompt = (
            "你是汽车金融信息检索助手。必须使用本次联网检索结果，不能引用知识库或模型记忆；只使用指定的银行官网。"
            "不要推荐机构、不要推测利率。先查询该城市的通用新能源汽车分期/车贷方案；车型专属活动只有在官网明确披露时才写。"
            "以表格列出机构、方案、首付/期限、年化利率或费率、适用条件、有效期、官方 URL；没有官方 URL 的条目不得输出。"
            f"用户需求：{query}"
        )
    body = {
        "model": settings.openai_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "enable_search": True,
    }
    if sites:
        body["search_options"] = {
            "forced_search": True,
            "assigned_site_list": sites,
            "enable_source": True,
            "enable_citation": True,
        }
    request = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    print("[Finance] 正在通过百炼检索官方金融方案…" if sites else "[Web] 正在通过百炼联网搜索…", flush=True)
    try:
        with urlopen(request, timeout=settings.web_search_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"百炼联网搜索失败（HTTP {error.code}）：{detail}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"百炼联网搜索失败或超过 {settings.web_search_timeout} 秒。") from error

    content = payload["choices"][0]["message"]["content"]
    if sites:
        content = _verified_finance_content(content)
    urls = re.findall(r"https?://[^\s)>，]+", content)
    return [
        {
            "entity": {"text": content, "source": "百炼官方金融检索" if sites else "百炼联网搜索", "title": title, "url": urls[0] if urls else None, "chunk_index": 0},
            "distance": 0.0,
        }
    ]


def finance_search(query: str) -> list[dict]:
    return search(query, title="官方汽车金融方案（实时检索）", sites=OFFICIAL_FINANCE_SITES)
