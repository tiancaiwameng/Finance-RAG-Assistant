"""Finance RAG Assistant 的 Streamlit Web 入口。"""

from __future__ import annotations

import logging
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st

from config import settings
from src.database import MySQLRepository
from src.pdf_loader import FinancialPDFLoader, PDFLoadError
from src.pipeline import UploadValidationError, ValidatedUpload, validate_uploads
from src.rag_chain import FinancialRAGChain, RAGGenerationError
from src.utils import compact_excerpt, deduplicate_documents, format_source_label
from src.vector_store import FinancialVectorStore, VectorStoreError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


st.set_page_config(page_title="Finance RAG Assistant", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.45rem;}
    .subtle {color: #667085; font-size: .95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_database() -> MySQLRepository:
    """每个 Streamlit 进程初始化一次数据库；失败时仓储会自动降级。"""
    repository = MySQLRepository(settings)
    repository.initialize()
    return repository


database = get_database()


def init_state() -> None:
    defaults = {
        "messages": [],
        "vector_store": None,
        "document_stats": None,
        "indexed_files": [],
        "document_ids": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_uploads(uploaded_files: list) -> None:
    """执行校验、解析、去重、切分、向量化及可选元数据记录。"""
    batch = validate_uploads(uploaded_files, settings.max_upload_mb)
    logger.info("开始处理文档批次：%d 个有效文件", len(batch.files))

    document_ids: dict[str, int] = {}
    for uploaded in batch.files:
        document_id = database.upsert_document(
            uploaded.filename, uploaded.file_hash, 0, "processing"
        )
        if document_id is not None:
            document_ids[uploaded.file_hash] = document_id

    loader = FinancialPDFLoader(settings.chunk_size, settings.chunk_overlap)
    try:
        with tempfile.TemporaryDirectory(prefix="finance_rag_") as temp_dir:
            file_entries: list[tuple[Path, ValidatedUpload]] = []
            for position, uploaded in enumerate(batch.files):
                # 序号避免同名文件覆盖；路径只存在于受控临时目录。
                temp_path = Path(temp_dir) / f"{position}_{uploaded.filename}"
                temp_path.write_bytes(uploaded.content)
                file_entries.append((temp_path, uploaded))

            pages = []
            for temp_path, uploaded in file_entries:
                pages.extend(loader.load(temp_path, uploaded.filename, uploaded.file_hash))
            raw_chunks = loader.split(pages)
            chunks, duplicate_chunk_count = deduplicate_documents(raw_chunks)
            if not chunks:
                raise PDFLoadError("文档切分后没有可用文本片段。")
            if duplicate_chunk_count:
                logger.info("文本片段去重完成：跳过 %d 个重复片段", duplicate_chunk_count)

        vector_store = FinancialVectorStore(settings.embedding_model, settings.chroma_dir)
        vector_store.build(chunks)
    except Exception:
        for uploaded in batch.files:
            database.update_document(uploaded.file_hash, 0, "failed")
        raise

    chunk_counts = Counter(str(chunk.metadata.get("file_hash", "")) for chunk in chunks)
    for uploaded in batch.files:
        database.update_document(
            uploaded.file_hash, chunk_counts.get(uploaded.file_hash, 0), "completed"
        )

    st.session_state.vector_store = vector_store
    st.session_state.indexed_files = [uploaded.filename for uploaded in batch.files]
    st.session_state.document_ids = document_ids
    st.session_state.document_stats = {
        "files": len(batch.files),
        "pages": len(pages),
        "chunks": len(chunks),
        "duplicate_files": len(batch.duplicate_names),
        "duplicate_chunks": duplicate_chunk_count,
    }
    st.session_state.messages = []
    logger.info(
        "文档批次处理完成：%d 页，%d 个去重片段", len(pages), len(chunks)
    )


def render_sources(sources) -> None:
    if not sources:
        return
    st.caption("回答依据 · 点击查看原文片段")
    for index, result in enumerate(sources, start=1):
        label = format_source_label(result.document)
        with st.expander(f"来源 {index}｜{label}｜相关度 {result.relevance:.0%}"):
            st.write(compact_excerpt(result.document.page_content, limit=1200))


init_state()

st.title("📊 Finance RAG Assistant")
st.markdown(
    '<p class="subtle">面向财报、公告与研报的可追溯 AI 投研问答：回答来自文档，结论附带页码证据。</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("建立研究知识库")
    uploaded_files = st.file_uploader(
        "上传金融 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"支持年报、公告、研报；单批总计不超过 {settings.max_upload_mb} MB。",
    )
    top_k = st.slider("检索片段数 Top K", min_value=3, max_value=8, value=settings.top_k)
    if st.button("处理文档", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("请先上传至少一个 PDF。")
        else:
            try:
                with st.spinner("正在解析、切分并建立向量知识库…首次运行需下载 Embedding 模型。"):
                    process_uploads(uploaded_files)
                st.success("知识库建立完成。")
            except (PDFLoadError, UploadValidationError, VectorStoreError, ValueError) as exc:
                st.error(str(exc))
            except Exception:
                logger.exception("未预期的文档处理错误")
                st.error("文档处理失败，请查看终端日志获取详细原因。")

    stats = st.session_state.document_stats
    if stats:
        st.divider()
        left, middle, right = st.columns(3)
        left.metric("文档", stats["files"])
        middle.metric("页数", stats["pages"])
        right.metric("片段", stats["chunks"])
        st.caption("已索引：" + "、".join(st.session_state.indexed_files))
        if stats["duplicate_files"] or stats["duplicate_chunks"]:
            st.caption(
                f"已跳过重复文件 {stats['duplicate_files']} 个、重复片段 "
                f"{stats['duplicate_chunks']} 个。"
            )

    st.divider()
    key_ready = bool(settings.deepseek_api_key and "your_" not in settings.deepseek_api_key)
    st.caption("DeepSeek API：" + ("✅ 已配置" if key_ready else "⚠️ 未配置 .env"))
    st.caption("MySQL：" + database.status_label)
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if st.session_state.vector_store is None:
    st.info("请在左侧上传 PDF 并点击“处理文档”。你也可以先使用 `data/example.pdf`。")
    st.subheader("适合投研的示例问题")
    cols = st.columns(3)
    examples = [
        "公司 2024 年营业收入增长的原因是什么？",
        "毛利率发生了怎样的变化？主要原因是什么？",
        "管理层披露了哪些主要风险因素？",
    ]
    for col, question in zip(cols, examples):
        col.markdown(f"> {question}")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    if question := st.chat_input("针对已上传文档提问，例如：2024 年收入增长由哪些业务驱动？"):
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("正在检索证据并生成回答…"):
                    chain = FinancialRAGChain(st.session_state.vector_store, settings)
                    response = chain.ask(question, top_k=top_k, history=history)
                st.markdown(response.answer)
                render_sources(response.sources)
                referenced_hashes = {
                    str(source.document.metadata.get("file_hash", ""))
                    for source in response.sources
                    if source.document.metadata.get("file_hash")
                }
                referenced_ids = [
                    st.session_state.document_ids[file_hash]
                    for file_hash in referenced_hashes
                    if file_hash in st.session_state.document_ids
                ]
                database.record_qa(referenced_ids, question, response.answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.answer, "sources": response.sources}
                )
            except (RAGGenerationError, VectorStoreError, ValueError) as exc:
                message = str(exc)
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
            except Exception:
                logger.exception("未预期的问答错误")
                message = "回答生成失败，请查看终端日志获取详细原因。"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
