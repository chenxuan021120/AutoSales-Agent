"""集中管理配置，避免把任何密钥写进代码。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    pdf_path: Path = Path(os.getenv("PDF_PATH", "../data/星辰电动ES9·未来旗舰电动SUV产品介绍.pdf"))
    vector_store_path: Path = Path(os.getenv("RAG_DB_PATH", "data/chroma"))
    collection_name: str = os.getenv("COLLECTION_NAME", "es9_knowledge")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    embedding_backend: str = os.getenv("RAG_EMBEDDING_BACKEND", "local").lower()
    embedding_base_url: str | None = os.getenv("EMBEDDING_BASE_URL") or None
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "512"))
    # P2.1-B1：默认保留 R0，候选 fusion 仅由评测脚本显式传入。
    ranking_strategy: str = os.getenv("RAG_RANKING_STRATEGY", "lexical_first").lower()
    semantic_weight: float = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.7"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    vision_model: str = os.getenv("VISION_MODEL", "qwen-vl-plus")
    serper_api_key: str | None = os.getenv("SERPER_API_KEY") or None
    web_search_timeout: int = int(os.getenv("WEB_SEARCH_TIMEOUT", "60"))

    def resolved_pdf_path(self) -> Path:
        return (PROJECT_DIR / self.pdf_path).resolve()

    def resolved_vector_store_path(self) -> Path:
        return (PROJECT_DIR / self.vector_store_path).resolve()


settings = Settings()
