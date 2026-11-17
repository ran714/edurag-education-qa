"""Local dependency-free demo backend for the EduRAG web application.

It keeps the public interface used by app.py, but stores conversations in memory
and retrieves answers directly from the bundled CSV knowledge base. This makes
it possible to demonstrate the UI, REST API and WebSocket streaming without
MySQL, Redis, Milvus or an external LLM API.
"""

from __future__ import annotations

import csv
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Tuple

from base.conversation_store import ConversationStore
from base.knowledge_data import knowledge_csv


class DemoBM25Search:
    """Compatibility layer for the original BM25Search interface."""

    def __init__(self, backend: "DemoIntegratedQASystem") -> None:
        self.backend = backend

    def search(
        self,
        query: str,
        threshold: float = 0.85,
        source_filter: Optional[str] = None,
    ) -> Tuple[Optional[str], bool]:
        match = self.backend.best_match(query, source_filter)
        if match and match[0] >= min(threshold, 0.78):
            score, row = match
            return row["answer"], False
        return None, True


class DemoIntegratedQASystem:
    """CSV-backed implementation matching the production system's API."""

    # 课程 CSV 里只有学科答疑，没有技术概念条目；不内置这几条时，
    # “RAG 是什么”这类问题在演示模式下只能落到“未找到答案”。
    CONCEPT_ANSWERS = (
        (
            ("rag", "检索增强生成"),
            "**RAG（Retrieval-Augmented Generation，检索增强生成）** 指的是“先检索、再生成”。\n\n"
            "1. **检索**：把用户问题向量化，在知识库中召回最相关的若干个文档片段；\n"
            "2. **增强**：把这些片段拼进提示词，作为回答的事实依据；\n"
            "3. **生成**：大模型基于检索到的内容组织答案，而不是只凭参数里的记忆。\n\n"
            "它的价值在于：知识可以随时更新（无需重新训练模型）、回答可溯源（能看到依据的原文）、幻觉更少。\n\n"
            "本系统就是这种两级结构：先由 BM25 命中课程问答库，未命中时再用 Milvus 向量检索 + 大模型生成。",
        ),
        (
            ("llm", "大语言模型", "大模型"),
            "**大语言模型（Large Language Model，LLM）** 是基于 Transformer、用海量文本训练出的生成式模型，"
            "常见的有通义千问（Qwen）、GPT、Llama 等。\n\n"
            "它的能力来自“预测下一个 token”，所以擅长理解和生成自然语言；但它并不天然掌握你私有的课程资料，"
            "也可能编造内容（幻觉）。因此在知识问答场景里，通常把 LLM 与检索结合（即 RAG）来保证回答有依据。",
        ),
        (
            ("embedding", "向量化", "词向量"),
            "**Embedding（向量化）** 是把文本映射成高维向量，使语义相近的文本在向量空间中距离更近。\n\n"
            "它分两步使用：入库时把文档切片逐条转成向量存进向量库；查询时把问题转成向量，再按相似度召回最近的片段。"
            "本项目使用 bge-m3 生成向量、bge-reranker-large 做重排序。",
        ),
        (
            ("milvus", "向量数据库"),
            "**向量数据库** 用于存储 Embedding 向量并提供近似最近邻检索，常见的有 Milvus、FAISS、Chroma 等。\n\n"
            "它解决的是“海量文本里按语义找最相关片段”的问题：先按向量相似度粗召回，再用量化、倒排等索引结构保证查询速度。"
            "本项目的向量库是 Milvus，集合名为 edurag_final。",
        ),
    )

    def __init__(self, history_path=None) -> None:
        self.rows = self._load_knowledge_base()
        self.history_store = ConversationStore(
            history_path
            or os.getenv(
                "EDURAG_SESSION_DB",
                str(Path(__file__).resolve().parent / "data/runtime/demo_sessions.db"),
            )
        )
        sources = sorted({row["source"] for row in self.rows})
        self.config = SimpleNamespace(VALID_SOURCES=sources)
        self.bm25_search = DemoBM25Search(self)

    @staticmethod
    def _load_knowledge_base() -> List[Dict[str, str]]:
        csv_path = knowledge_csv()

        last_error: Optional[Exception] = None
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                with csv_path.open("r", encoding=encoding, newline="") as file:
                    reader = csv.DictReader(file)
                    rows = []
                    for raw in reader:
                        source = (raw.get("学科名称") or "未分类").strip()
                        question = (raw.get("问题") or "").strip()
                        answer = (raw.get("答案") or "").strip()
                        if question and answer:
                            rows.append({"source": source, "question": question, "answer": answer})
                    if rows:
                        return rows
            except UnicodeError as exc:
                last_error = exc
        raise RuntimeError(f"无法读取演示知识库：{last_error}")

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", (text or "").lower())

    @classmethod
    def _features(cls, text: str) -> set[str]:
        normalized = cls._normalize(text)
        features = set(re.findall(r"[a-zA-Z]+|\d+", normalized))
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        # 疑问句套话不能作为相关性依据，否则“什么是人工智能”会误匹配“什么是张量”。
        stop_phrases = ("什么是", "什么叫", "如何", "怎么", "请问", "请讲讲", "有哪些", "是什么")
        for phrase in stop_phrases:
            chinese = chinese.replace(phrase, "")
        features.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
        features.update(chinese)
        return features

    @classmethod
    def _core_terms(cls, text: str) -> set[str]:
        normalized = cls._normalize(text)
        normalized = re.sub(r"(什么是|什么叫|如何|怎么|请问|请讲讲|有哪些|是什么)", "", normalized)
        terms = set(re.findall(r"[a-zA-Z]+|\d+", normalized))
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        terms.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
        terms.update(chinese)
        return terms

    @classmethod
    def _score(cls, query: str, candidate: str) -> float:
        left = cls._normalize(query)
        right = cls._normalize(candidate)
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        sequence_score = SequenceMatcher(None, left, right).ratio()
        left_features = cls._features(query)
        right_features = cls._features(candidate)
        union = left_features | right_features
        token_score = len(left_features & right_features) / len(union) if union else 0.0
        core_overlap = cls._core_terms(query) & cls._core_terms(candidate)
        # 有实际主题词时，至少需要一个主题词重合；避免只凭“什么是”等套话命中。
        if cls._core_terms(query) and not core_overlap:
            return 0.0
        containment = 0.92 if left in right or right in left else 0.0
        return max(
            containment,
            sequence_score * 0.48 + token_score * 0.32 + min(len(core_overlap), 4) * 0.05,
        )

    @staticmethod
    def _concept_hit(query: str, keywords: Tuple[str, ...]) -> bool:
        lowered = (query or "").lower()
        for keyword in keywords:
            if keyword.isascii():
                # 英文缩写按词边界匹配，避免 storage 之类误命中 rag。
                if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", lowered):
                    return True
            elif keyword in query:
                return True
        return False

    def _concept_answer(self, query: str) -> Optional[str]:
        for keywords, answer in self.CONCEPT_ANSWERS:
            if self._concept_hit(query, keywords):
                return answer
        return None

    def best_match(self, query: str, source_filter: Optional[str] = None):
        candidates = self.rows
        if source_filter:
            candidates = [row for row in candidates if row["source"] == source_filter]
        if not candidates:
            return None
        row = max(candidates, key=lambda item: self._score(query, item["question"]))
        return self._score(query, row["question"]), row

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.history_store.get(session_id)

    def update_session_history(self, session_id: str, question: str, answer: str) -> None:
        self.history_store.append(session_id, question, answer)

    def clear_session_history(self, session_id: str) -> bool:
        return self.history_store.clear(session_id)

    def close(self):
        self.history_store.close()

    def _build_answer(self, query: str, source_filter: Optional[str]) -> str:
        normalized = self._normalize(query)
        filter_header = f"**检索范围：{source_filter or '全部学科'}**\n\n"
        is_python_scope = not source_filter or source_filter == "Python学科"

        # 通用示例回答只在“全部学科”下生效；Python 示例只在全部/Python 下生效。
        # 选择 JAVA 时必须继续进入已过滤的知识库检索，不能返回相同的固定答案。
        if not source_filter and ("什么是人工智能" in normalized or "人工智能是什么" in normalized):
            return filter_header + (
                "人工智能（Artificial Intelligence，AI）是让计算机模拟人类的感知、学习、推理、决策和生成等能力的技术。\n\n"
                "常见方向包括机器学习、深度学习、自然语言处理、计算机视觉和智能语音。\n\n"
                "简单来说，就是让机器能够根据数据和规则完成过去通常需要人来完成的智能任务。\n\n"
                "当前是本地演示模式，答案由 EduRAG 演示后端生成。"
            )
        if is_python_scope and ("机器学习有哪些算法" in normalized or "机器学习算法" in normalized):
            return filter_header + (
                "机器学习算法通常分为三类：\n\n"
                "1. **监督学习**：线性回归、逻辑回归、决策树、随机森林、支持向量机、神经网络；\n"
                "2. **无监督学习**：K-Means 聚类、层次聚类、主成分分析（PCA）；\n"
                "3. **强化学习**：Q-Learning、DQN 等。\n\n"
                "选择算法时要结合任务类型、数据规模、特征类型和是否有标签来判断。\n\n"
                "当前是本地演示模式，答案由 EduRAG 演示后端生成。"
            )
        if is_python_scope and ("如何学习python" in normalized or "怎么学习python" in normalized):
            return filter_header + (
                "建议按 **语法基础 → 数据结构 → 函数与面向对象 → 常用库 → 项目实战** 的顺序学习 Python。\n\n"
                "1. 先掌握变量、条件、循环、函数；\n"
                "2. 熟悉列表、字典、集合和异常处理；\n"
                "3. 完成爬虫、数据分析或 Web 小项目；\n"
                "4. 遇到问题时结合本知识库中的具体问答查漏补缺。\n\n"
                "当前是本地演示模式，答案来自内置规则与课程 CSV 知识库。"
            )

        # 通用技术概念不属于任何学科，即使选了学科也直接给出解释。
        concept = self._concept_answer(query) if not source_filter else None
        if concept:
            return filter_header + concept

        match = self.best_match(query, source_filter)
        if match and match[0] >= 0.58:
            score, row = match
            return filter_header + (
                f"我在 **{row['source']}** 知识库中找到了相关问题：\n\n"
                f"> {row['question']}\n\n{row['answer']}\n\n"
                f"_本地检索相似度：{score:.0%}；演示模式未调用外部大模型。_"
            )

        filter_tip = f"（当前筛选：{source_filter}）" if source_filter else ""
        return filter_header + (
            f"本地课程知识库暂未找到与“{query}”足够接近的答案{filter_tip}。\n\n"
            "当前是演示模式：只在课程 CSV 问答库里做字面相似度检索，不调用大模型，"
            "所以换个说法或问知识库以外的内容时就命中不到。\n\n"
            "可以这样继续：\n"
            "- 换成课程答疑里更常见的表述，或去掉学科筛选后再问一次；\n"
            "- 需要完整的检索增强生成（文档知识库 + 语义检索 + 大模型作答）时，请运行 `run_full.ps1` 启动完整模式。"
        )

    @staticmethod
    def _chunks(text: str, size: int = 18) -> Iterator[str]:
        for index in range(0, len(text), size):
            yield text[index : index + size]

    def query(
        self,
        query: str,
        source_filter: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[Tuple[str, bool]]:
        from rag_qa.core.query_rewriter import needs_context

        if needs_context(query):
            answer = "演示模式不解析多轮指代，请在问题中写明课程、模型或技术名称。"
        else:
            answer = self._build_answer(query, source_filter)
        if session_id:
            self.update_session_history(session_id, query, answer)
        chunks = list(self._chunks(answer))
        for index, chunk in enumerate(chunks):
            yield chunk, index == len(chunks) - 1
