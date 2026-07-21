"""项目配置：所有可变参数均从环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _read_int(name: str, default: int) -> int:
    """读取正整数配置；配置异常时尽早给出明确错误。"""
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值为 {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"环境变量 {name} 必须大于 0")
    return parsed


@dataclass(frozen=True)
class Settings:
    """集中管理模型、切分和检索参数。"""

    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )
    chunk_size: int = field(default_factory=lambda: _read_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _read_int("CHUNK_OVERLAP", 120))
    top_k: int = field(default_factory=lambda: _read_int("TOP_K", 5))
    max_upload_mb: int = field(default_factory=lambda: _read_int("MAX_UPLOAD_MB", 50))
    chroma_dir: Path = field(default_factory=lambda: BASE_DIR / "chroma_db")

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")

    def require_api_key(self) -> None:
        if not self.deepseek_api_key or self.deepseek_api_key == "your_deepseek_api_key_here":
            raise ValueError("未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 后填写密钥。")


settings = Settings()
settings.validate()

