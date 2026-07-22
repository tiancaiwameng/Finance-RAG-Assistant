"""Chroma 向量知识库：本地 Embedding、缓存与带分数检索。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import document_fingerprint


logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """向量模型加载、写入或检索失败。"""


@dataclass(frozen=True)
class SearchResult:
    document: Document
    relevance: float


class FinancialVectorStore:
    def __init__(self, embedding_model: str, persist_directory: str | Path) -> None:
        try:
            self.persist_directory = Path(persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception("向量数据库目录创建失败：%s", persist_directory)
            raise VectorStoreError("无法创建向量数据库目录，请检查路径和写入权限。") from exc

        embedding_options = {
            "model_name": embedding_model,
            "encode_kwargs": {"normalize_embeddings": True},
        }
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_kwargs={"device": "cpu"},
                **embedding_options,
            )
        except Exception as exc:
            logger.warning("在线加载 Embedding 失败，尝试本地缓存：%s", exc)
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_kwargs={"device": "cpu", "local_files_only": True},
                    **embedding_options,
                )
                logger.info("已从本地缓存加载 Embedding 模型：%s", embedding_model)
            except Exception as cache_exc:
                logger.exception("Embedding 模型初始化失败：%s", embedding_model)
                raise VectorStoreError(
                    "Embedding 模型加载失败；首次运行请确认网络可访问 Hugging Face。"
                ) from cache_exc
        self.store: Chroma | None = None
        self.collection_name = ""

    def build(self, documents: list[Document]) -> str:
        """按内容指纹建集合；相同文档重复上传时直接复用已有向量。"""
        if not documents:
            raise ValueError("没有可写入向量库的文本片段")

        try:
            fingerprint = document_fingerprint(documents)
            self.collection_name = f"finance_{fingerprint[:24]}"
            self.store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
                collection_metadata={"hnsw:space": "cosine"},
            )

            # 相同文档集合生成相同指纹；已有片段时直接复用，避免重复向量化和写入。
            existing_count = self.store._collection.count()  # noqa: SLF001
            if existing_count == 0:
                logger.info("开始向量化并写入 %d 个片段", len(documents))
                ids = [f"{fingerprint[:16]}_{index}" for index in range(len(documents))]
                self.store.add_documents(documents=documents, ids=ids)
                logger.info("向量知识库写入完成：%s", self.collection_name)
            else:
                logger.info(
                    "复用已有向量集合：%s（%d 个片段）", self.collection_name, existing_count
                )
            return self.collection_name
        except Exception as exc:
            logger.exception("向量知识库构建失败")
            raise VectorStoreError(
                "向量知识库创建失败，请检查磁盘空间、Embedding 模型和网络连接。"
            ) from exc

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self.store is None:
            raise RuntimeError("向量知识库尚未建立")
        try:
            pairs = self.store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.exception("向量检索失败")
            raise VectorStoreError("文档检索失败，请重新处理文档后再试。") from exc
        # cosine distance 越小越相似，转换为 0~1 的直观相关度。
        return [
            SearchResult(document=doc, relevance=max(0.0, min(1.0, 1.0 - float(distance))))
            for doc, distance in pairs
        ]
