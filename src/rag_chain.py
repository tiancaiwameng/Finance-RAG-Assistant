"""检索增强生成：先检索证据，再让 DeepSeek 生成带引用的回答。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from config import Settings
from src.utils import compact_excerpt, format_source_label
from src.vector_store import FinancialVectorStore, SearchResult


SYSTEM_PROMPT = """你是一名严谨的金融研究助理。只能依据“检索证据”回答，不得补充文档外事实。
要求：
1. 先给出简洁结论，再说明关键依据；涉及金额、同比、毛利率等数字时保留原单位和期间。
2. 每项关键判断后使用 [来源1]、[来源2] 格式引用证据。
3. 区分管理层陈述、客观财务数据与风险提示，不把计划当作已实现结果。
4. 如果证据不足或不同片段矛盾，明确写“根据当前文档无法确定”，并说明还需什么资料。
5. 不提供买卖建议；最后用一句话提示本回答仅用于资料整理和研究辅助。
"""


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[SearchResult]


class FinancialRAGChain:
    def __init__(self, vector_store: FinancialVectorStore, settings: Settings) -> None:
        settings.require_api_key()
        self.vector_store = vector_store
        self.settings = settings
        self.llm = ChatDeepSeek(
            model=settings.deepseek_model,
            temperature=0.1,
            max_tokens=1400,
            api_key=settings.deepseek_api_key,
            max_retries=2,
        )

    @staticmethod
    def _format_context(results: Sequence[SearchResult]) -> str:
        blocks = []
        for index, result in enumerate(results, start=1):
            label = format_source_label(result.document)
            blocks.append(
                f"[来源{index}] {label} | 检索相关度 {result.relevance:.0%}\n"
                f"{result.document.page_content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _history_messages(history: Sequence[dict[str, str]]) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        # 仅保留最近 3 轮，控制上下文成本；来源证据每轮都会重新检索。
        for item in history[-6:]:
            content = compact_excerpt(item.get("content", ""), limit=800)
            if item.get("role") == "user":
                messages.append(HumanMessage(content=content))
            elif item.get("role") == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        history: Sequence[dict[str, str]] = (),
    ) -> RAGResponse:
        if not question.strip():
            raise ValueError("问题不能为空")
        results = self.vector_store.search(question.strip(), k=top_k or self.settings.top_k)
        if not results:
            return RAGResponse("未检索到可用证据，请重新处理文档。", [])

        context = self._format_context(results)
        user_prompt = f"""检索证据：
{context}

本轮问题：{question.strip()}

请严格依据证据回答，并确保引用编号与上面的来源编号一致。"""
        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(self._history_messages(history))
        messages.append(HumanMessage(content=user_prompt))
        response = self.llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return RAGResponse(answer=answer, sources=results)

