"""Embedding 是 RAG 的“语义坐标”。提供可离线运行的教学模式和 BGE 模式。"""
from __future__ import annotations

import hashlib
import json
import math
import re
from urllib.error import URLError
from urllib.request import Request, urlopen


class BGEEmbedder:
    # BGE 中文模型推荐给检索 query 加上这段指令；文档本身不用加。
    QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str) -> None:
        # 延迟导入：离线教学模式无需加载 torch / transformers。
        from sentence_transformers import SentenceTransformer

        print(f"正在加载向量模型：{model_name}（首次使用会下载模型）")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        text = self.QUERY_INSTRUCTION + query
        return self.model.encode([text], normalize_embeddings=True).tolist()[0]


class LocalHashEmbedder:
    """无需联网的轻量向量器，用于学习和验证 RAG 流程，不等同于语义模型。"""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        print("使用离线教学向量器（不下载模型）。")

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
        return tokens + ["".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokens(text):
            value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            index = value % self.dimension
            vector[index] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)


class AliyunEmbedding:
    """调用百炼 text-embedding-v4，避免本机下载 BGE 模型。"""

    BATCH_SIZE = 10  # text-embedding-v4 的 OpenAI 兼容接口每批最多 10 条。

    def __init__(self, model_name: str, base_url: str | None, dimensions: int, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError("未配置 OPENAI_API_KEY，无法调用百炼 Embedding。")
        if not base_url:
            raise RuntimeError("请先在 .env 填写 EMBEDDING_BASE_URL（百炼业务空间专属地址）。")
        # 直接调用 OpenAI 兼容 HTTP 接口，避免当前环境中 SDK 导入变慢。
        self.endpoint = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = dimensions
        print(f"使用百炼语义向量模型：{model_name}（{dimensions} 维）。")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_count = (len(texts) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        for batch_number, start in enumerate(range(0, len(texts), self.BATCH_SIZE), start=1):
            print(f"[Embedding] 正在请求百炼：第 {batch_number}/{batch_count} 批…", flush=True)
            request_body = json.dumps(
                {
                    "model": self.model_name,
                    "input": texts[start : start + self.BATCH_SIZE],
                    "dimensions": self.dimension,
                    "encoding_format": "float",
                }
            ).encode("utf-8")
            request = Request(
                self.endpoint,
                data=request_body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=30) as http_response:
                    response = json.loads(http_response.read().decode("utf-8"))
            except (URLError, TimeoutError, OSError) as error:
                raise RuntimeError(
                    "百炼 Embedding 调用失败。请检查 Workspace ID、模型调用权限和网络；"
                    "现在已设置 30 秒超时，不会无限等待。"
                ) from error
            vectors.extend(item["embedding"] for item in sorted(response["data"], key=lambda item: item["index"]))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query])[0]


def get_embedder(
    backend: str,
    model_name: str,
    base_url: str | None = None,
    dimensions: int = 512,
    api_key: str | None = None,
) -> BGEEmbedder | LocalHashEmbedder | AliyunEmbedding:
    if backend == "local":
        return LocalHashEmbedder()
    if backend == "bge":
        return BGEEmbedder(model_name)
    if backend == "aliyun":
        return AliyunEmbedding(model_name, base_url, dimensions, api_key)
    raise ValueError("RAG_EMBEDDING_BACKEND 只能是 local、bge 或 aliyun")
