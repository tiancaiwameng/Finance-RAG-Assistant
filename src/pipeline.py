"""上传文件校验与文档处理阶段的通用数据结构。"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from src.utils import safe_filename


logger = logging.getLogger(__name__)
PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}


class UploadedFileLike(Protocol):
    """Streamlit UploadedFile 所需的最小接口，便于独立测试。"""

    name: str
    size: int
    type: str | None

    def getvalue(self) -> bytes: ...


class UploadValidationError(ValueError):
    """上传文件不满足类型、大小或内容要求。"""


@dataclass(frozen=True)
class ValidatedUpload:
    """完成安全校验并计算内容哈希后的上传文件。"""

    filename: str
    content: bytes
    file_hash: str


@dataclass(frozen=True)
class UploadBatch:
    """有效上传文件，以及因内容相同而跳过的文件名。"""

    files: list[ValidatedUpload]
    duplicate_names: list[str]


def sha256_bytes(content: bytes) -> str:
    """计算原始文件 SHA-256，用于重复文档识别。"""
    return hashlib.sha256(content).hexdigest()


def validate_uploads(
    uploaded_files: Sequence[UploadedFileLike], max_upload_mb: int
) -> UploadBatch:
    """校验 PDF 扩展名、MIME、文件头和单文件/批次大小，并按内容去重。"""
    if not uploaded_files:
        raise UploadValidationError("请先上传至少一个 PDF。")

    max_bytes = max_upload_mb * 1024 * 1024
    total_bytes = 0
    unique_files: list[ValidatedUpload] = []
    duplicate_names: list[str] = []
    seen_hashes: set[str] = set()

    for uploaded in uploaded_files:
        filename = safe_filename(uploaded.name)
        if not filename.lower().endswith(".pdf"):
            raise UploadValidationError(f"{filename} 不是 PDF 文件，请仅上传 .pdf 文件。")

        mime_type = (uploaded.type or "").lower()
        if mime_type and mime_type not in PDF_MIME_TYPES:
            raise UploadValidationError(
                f"{filename} 的文件类型为 {mime_type}，不是有效的 PDF MIME 类型。"
            )

        content = uploaded.getvalue()
        file_size = len(content)
        if file_size == 0:
            raise UploadValidationError(f"{filename} 是空文件。")
        if file_size > max_bytes:
            raise UploadValidationError(f"{filename} 超过单文件 {max_upload_mb} MB 限制。")
        if not content.startswith(b"%PDF-"):
            raise UploadValidationError(f"{filename} 缺少 PDF 文件头，文件可能已损坏或类型不符。")

        total_bytes += file_size
        if total_bytes > max_bytes:
            raise UploadValidationError(f"本批文件总大小不能超过 {max_upload_mb} MB。")

        file_hash = sha256_bytes(content)
        if file_hash in seen_hashes:
            duplicate_names.append(filename)
            logger.info("跳过重复上传文档：%s (sha256=%s)", filename, file_hash[:12])
            continue

        seen_hashes.add(file_hash)
        unique_files.append(ValidatedUpload(filename, content, file_hash))

    if not unique_files:
        raise UploadValidationError("本批上传文件均为重复文档，没有可处理的新内容。")

    logger.info(
        "上传校验完成：有效文件 %d 个，重复文件 %d 个，总大小 %.2f MB",
        len(unique_files),
        len(duplicate_names),
        total_bytes / 1024 / 1024,
    )
    return UploadBatch(unique_files, duplicate_names)
