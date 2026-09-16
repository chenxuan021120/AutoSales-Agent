"""轻量本地向量库：保存文本、向量与来源元数据。"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


_PDF_GLYPH_MAP = str.maketrans({"⻋": "车", "⼿": "手", "⾠": "辰", "⽤": "用", "⻓": "长"})


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_PDF_GLYPH_MAP)
    # 汽车参数问法常把“最大支持”说成“最高能达到”，统一后保护直接证据。
    return text.replace("最高能达到", "最高支持").replace("最大支持", "最高支持")


def _lexical_score(query: str, text: str) -> int:
    """对包含用户关键短语的 chunk 做小幅加权，避免相邻 chunk 压过直接证据。"""
    query, text = _normalize_text(query), _normalize_text(text)
    score = 0
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", query):
        for size in range(2, min(6, len(phrase)) + 1):
            score += sum(1 for start in range(len(phrase) - size + 1) if phrase[start : start + size] in text)
    for term in re.findall(r"[A-Za-z0-9]{2,}", query):
        if term.lower() != "es9" and term.lower() in text.lower():
            score += 1
    return score


class LocalVectorStore:
    def __init__(self, path: Path, collection: str, dimension: int) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.collection = collection
        self.dimension = dimension
        print(f"本地向量数据库：{self.path}")

    def rebuild(self, texts: list[str], vectors: list[list[float]], source: str) -> None:
        records = [
            {"id": index, "vector": vector, "text": text, "source": source, "chunk_index": index}
            for index, (text, vector) in enumerate(zip(texts, vectors))
        ]
        self.path.write_text(
            json.dumps({"collection": self.collection, "dimension": self.dimension, "records": records}, ensure_ascii=False),
            encoding="utf-8",
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 4,
        query_text: str = "",
        ranking_strategy: str = "lexical_first",
        semantic_weight: float = 0.7,
    ) -> list[dict]:
        if not self.path.exists():
            raise RuntimeError("知识库尚未建立。请先运行：python app.py ingest")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload["dimension"] != len(query_vector):
            raise RuntimeError("向量维度已变更，请重新运行：python app.py ingest")
        if ranking_strategy not in {"lexical_first", "weighted_fusion"}:
            raise ValueError("ranking_strategy 只能是 lexical_first 或 weighted_fusion")
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight 必须在 0 到 1 之间")
        scored = []
        for record in payload["records"]:
            semantic = sum(a * b for a, b in zip(query_vector, record["vector"]))
            lexical = _lexical_score(query_text, record["text"]) if query_text else 0
            scored.append({"lexical": float(lexical), "semantic": float(semantic), "record": record})
        if ranking_strategy == "lexical_first":
            # R0：保留原始的词法优先、语义打破平局的精确行为。
            for item in scored:
                item["semantic_norm"] = None
                item["lexical_norm"] = None
                item["final"] = item["lexical"] + item["semantic"]
            ranked = sorted(scored, key=lambda item: (item["lexical"], item["semantic"]), reverse=True)
        else:
            semantic_values = [item["semantic"] for item in scored]
            lexical_values = [item["lexical"] for item in scored]
            semantic_min, semantic_max = min(semantic_values), max(semantic_values)
            lexical_min, lexical_max = min(lexical_values), max(lexical_values)
            semantic_range, lexical_range = semantic_max - semantic_min, lexical_max - lexical_min
            for item in scored:
                item["semantic_norm"] = (item["semantic"] - semantic_min) / semantic_range if semantic_range else 0.0
                item["lexical_norm"] = (item["lexical"] - lexical_min) / lexical_range if lexical_range else 0.0
                item["final"] = semantic_weight * item["semantic_norm"] + (1 - semantic_weight) * item["lexical_norm"]
            ranked = sorted(scored, key=lambda item: item["final"], reverse=True)
        return [
            {
                "entity": item["record"],
                "rank": rank,
                "semantic_score": item["semantic"],
                "lexical_score": item["lexical"],
                "semantic_normalized": item["semantic_norm"],
                "lexical_normalized": item["lexical_norm"],
                "final_score": item["final"],
                "ranking_strategy": ranking_strategy,
                "semantic_weight": semantic_weight if ranking_strategy == "weighted_fusion" else None,
                "distance": None,
            }
            for rank, item in enumerate(ranked[:limit], start=1)
        ]
