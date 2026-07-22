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
    """按文件哈希、页码和正文生成顺序无关的指纹，用于复用向量集合。"""
    digest = hashlib.sha256()
    fingerprint_items = []
    for document in documents:
        identity = str(
            document.metadata.get("file_hash") or document.metadata.get("source", "")
        )
        page = str(document.metadata.get("page_number", ""))
        fingerprint_items.append((identity, page, document.page_content))
    for identity, page, content in sorted(fingerprint_items):
        digest.update(identity.encode("utf-8"))
        digest.update(page.encode("utf-8"))
        digest.update(content.encode("utf-8"))
    return digest.hexdigest()


def deduplicate_documents(documents: Iterable[Document]) -> tuple[list[Document], int]:
    """按“文件哈希 + 清洗后的正文”去除重复片段，同时保留不同文档的来源。"""
    unique: list[Document] = []
    seen: set[tuple[str, str]] = set()
    duplicate_count = 0
    for document in documents:
        file_identity = str(
            document.metadata.get("file_hash") or document.metadata.get("source", "")
        )
        normalized = re.sub(r"\s+", " ", document.page_content).strip()
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        key = (file_identity, content_hash)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(document)
    return unique, duplicate_count


def compact_excerpt(text: str, limit: int = 500) -> str:
    """将检索片段压缩为适合界面展示的长度。"""
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[:limit].rstrip() + "…"


def format_source_label(document: Document) -> str:
    source = document.metadata.get("source", "未知文档")
    page = document.metadata.get("page_number", "?")
    return f"{source} · 第 {page} 页"
