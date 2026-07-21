"""文本清洗、指纹计算和来源展示工具。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


def safe_filename(filename: str) -> str:
    """移除目录信息和不安全字符，避免上传文件越界写入。"""
    name = Path(filename).name
    cleaned = re.sub(r"[^\w\-.()\u4e00-\u9fff]", "_", name, flags=re.UNICODE)
    return cleaned or "uploaded.pdf"


def normalize_financial_text(text: str) -> str:
    """保留表格换行的同时，清理 PDF 常见的多余空白和空行。"""
    text = text.replace("\u00a0", " ").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def document_fingerprint(documents: Iterable[Document]) -> str:
    """依据来源、页码与正文生成稳定指纹，用于复用本地向量集合。"""
    digest = hashlib.sha256()
    for doc in documents:
        digest.update(str(doc.metadata.get("source", "")).encode("utf-8"))
        digest.update(str(doc.metadata.get("page_number", "")).encode("utf-8"))
        digest.update(doc.page_content.encode("utf-8"))
    return digest.hexdigest()


def compact_excerpt(text: str, limit: int = 500) -> str:
    """将检索片段压缩为适合界面展示的长度。"""
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[:limit].rstrip() + "…"


def format_source_label(document: Document) -> str:
    source = document.metadata.get("source", "未知文档")
    page = document.metadata.get("page_number", "?")
    return f"{source} · 第 {page} 页"

