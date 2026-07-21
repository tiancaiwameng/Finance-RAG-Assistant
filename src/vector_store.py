"""Chroma 向量知识库：本地 Embedding、缓存与带分数检索。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import document_fingerprint


@dataclass(frozen=True)
class SearchResult:
    document: Document
    relevance: float


class FinancialVectorStore:
    def __init__(self, embedding_model: str, persist_directory: str | Path) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store: Chroma | None = None
        self.collection_name = ""

    def build(self, documents: list[Document]) -> str:
        """按内容指纹建集合；相同文档重复上传时直接复用已有向量。"""
        if not documents:
            raise ValueError("没有可写入向量库的文本片段")

        fingerprint = document_fingerprint(documents)
        self.collection_name = f"finance_{fingerprint[:24]}"
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

        # Chroma 暂未暴露统一的公共 count 接口，因此从底层集合只读取数量。
        if self.store._collection.count() == 0:  # noqa: SLF001
            ids = [f"{fingerprint[:16]}_{index}" for index in range(len(documents))]
            self.store.add_documents(documents=documents, ids=ids)
        return self.collection_name

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self.store is None:
            raise RuntimeError("向量知识库尚未建立")
        pairs = self.store.similarity_search_with_score(query, k=k)
        # cosine distance 越小越相似，转换为 0~1 的直观相关度。
        return [
            SearchResult(document=doc, relevance=max(0.0, min(1.0, 1.0 - float(distance))))
            for doc, distance in pairs
        ]

