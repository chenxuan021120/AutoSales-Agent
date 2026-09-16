"""步骤 1：PDF -> 文本块 -> 向量 -> Milvus Lite。"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from config import settings
from embeddings import get_embedder


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("PDF 未提取到文本。若它是扫描件，请先做 OCR。")
    return text


def chunk_text(text: str, size: int = 450, overlap: int = 80) -> list[str]:
    """按字符切块；overlap 能避免一句话正好被切开而丢失上下文。"""
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return [normalized[i : i + size] for i in range(0, len(normalized), size - overlap)]


def build_index() -> None:
    pdf_path = settings.resolved_pdf_path()
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到 PDF：{pdf_path}")

    print("[1/4] 正在读取 PDF…", flush=True)
    chunks = chunk_text(read_pdf(pdf_path))
    print(f"[1/4] 已读取 {pdf_path.name}，共切成 {len(chunks)} 个文本块。", flush=True)

    print("[2/4] 正在初始化向量器…", flush=True)
    embedder = get_embedder(
        settings.embedding_backend,
        settings.embedding_model,
        settings.embedding_base_url,
        settings.embedding_dimensions,
        settings.openai_api_key,
    )
    print("[3/4] 正在把文本块转换为向量…", flush=True)
    vectors = embedder.embed_documents(chunks)
    print("[4/4] 正在写入本地向量数据库…", flush=True)
    from vector_store import LocalVectorStore

    store = LocalVectorStore(settings.resolved_vector_store_path(), settings.collection_name, embedder.dimension)
    store.rebuild(chunks, vectors, pdf_path.name)
    print("建库完成。下一步运行：python app.py chat")


if __name__ == "__main__":
    build_index()
