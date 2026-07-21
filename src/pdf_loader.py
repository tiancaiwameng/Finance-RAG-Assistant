"""金融 PDF 解析与中文友好的文本切分。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.utils import normalize_financial_text


class PDFLoadError(RuntimeError):
    """PDF 无法读取或没有可提取文本。"""


class FinancialPDFLoader:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", ". ", " ", ""],
        )

    def load(self, file_path: str | Path, source_name: str | None = None) -> list[Document]:
        """逐页提取 PDF，元数据保留用户看到的文件名和 1-based 页码。"""
        path = Path(file_path)
        source = source_name or path.name
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise PDFLoadError(f"{source} 已加密，暂不支持解析") from exc
        except PDFLoadError:
            raise
        except Exception as exc:
            raise PDFLoadError(f"无法读取 {source}：{exc}") from exc

        pages: list[Document] = []
        for page_index, page in enumerate(reader.pages):
            try:
                text = normalize_financial_text(page.extract_text() or "")
            except Exception:
                # 单页解析失败不影响其余页面，但该页不会进入知识库。
                continue
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "page_number": page_index + 1,
                        },
                    )
                )

        if not pages:
            raise PDFLoadError(
                f"{source} 未提取到文本；如为扫描件，请先使用 OCR 转为可搜索 PDF。"
            )
        return pages

    def load_many(self, files: Sequence[tuple[str | Path, str]]) -> list[Document]:
        pages: list[Document] = []
        for file_path, source_name in files:
            pages.extend(self.load(file_path, source_name))
        return pages

    def split(self, pages: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(pages)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = index
        return chunks
