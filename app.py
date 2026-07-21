"""Finance RAG Assistant 的 Streamlit Web 入口。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from config import settings
from src.pdf_loader import FinancialPDFLoader, PDFLoadError
from src.rag_chain import FinancialRAGChain
from src.utils import compact_excerpt, format_source_label, safe_filename
from src.vector_store import FinancialVectorStore


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


def init_state() -> None:
    defaults = {
        "messages": [],
        "vector_store": None,
        "document_stats": None,
        "indexed_files": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_uploads(uploaded_files: list) -> None:
    total_bytes = sum(file.size for file in uploaded_files)
    if total_bytes > settings.max_upload_mb * 1024 * 1024:
        raise ValueError(f"文件总大小不能超过 {settings.max_upload_mb} MB")

    loader = FinancialPDFLoader(settings.chunk_size, settings.chunk_overlap)
    with tempfile.TemporaryDirectory(prefix="finance_rag_") as temp_dir:
        file_entries: list[tuple[Path, str]] = []
        for position, uploaded in enumerate(uploaded_files):
            display_name = safe_filename(uploaded.name)
            # 序号避免同名文件覆盖；路径只存在于临时目录。
            temp_path = Path(temp_dir) / f"{position}_{display_name}"
            temp_path.write_bytes(uploaded.getvalue())
            file_entries.append((temp_path, display_name))
        pages = loader.load_many(file_entries)
        chunks = loader.split(pages)

    vector_store = FinancialVectorStore(settings.embedding_model, settings.chroma_dir)
    vector_store.build(chunks)
    st.session_state.vector_store = vector_store
    st.session_state.indexed_files = [name for _, name in file_entries]
    st.session_state.document_stats = {
        "files": len(file_entries),
        "pages": len(pages),
        "chunks": len(chunks),
    }
    st.session_state.messages = []


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
            except (PDFLoadError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"文档处理失败：{exc}")

    stats = st.session_state.document_stats
    if stats:
        st.divider()
        left, middle, right = st.columns(3)
        left.metric("文档", stats["files"])
        middle.metric("页数", stats["pages"])
        right.metric("片段", stats["chunks"])
        st.caption("已索引：" + "、".join(st.session_state.indexed_files))

    st.divider()
    key_ready = bool(settings.deepseek_api_key and "your_" not in settings.deepseek_api_key)
    st.caption("DeepSeek API：" + ("✅ 已配置" if key_ready else "⚠️ 未配置 .env"))
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
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.answer, "sources": response.sources}
                )
            except Exception as exc:
                message = f"回答生成失败：{exc}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})

