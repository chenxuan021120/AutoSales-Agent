"""车型图片/配置表图片识别；使用项目已有的 OpenAI 兼容端点。"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import settings


def analyze_image(image_path: str | Path, instruction: str = "识别图片中的车型、配置参数和价格；看不清的内容明确标注为无法确认。") -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")
    if not settings.openai_api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY，无法调用多模态模型。")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    endpoint = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.vision_model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]}],
        "temperature": 0.1,
    }
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}, method="POST")
    print(f"[Vision] 正在请求 {settings.vision_model}…", flush=True)
    try:
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"多模态请求失败（HTTP {error.code}）：{detail}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError("多模态请求失败或超过 60 秒，请检查网络和模型配置。") from error
    return body["choices"][0]["message"].get("content", "") or "模型未返回文本。"
