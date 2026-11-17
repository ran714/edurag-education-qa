"""Retrieval strategies and evidence-based generation, independent of transport."""

import re
from pathlib import Path

from base.config import config
from base.llm_stream import collect_text, iter_text
from base.logger import logger

from .prompts import RAGPrompts
from .query_rewriter import QueryRewriter, RewriteResult


class RAGSystem:
    def __init__(self, vector_store, llm, *, classifier=None, selector=None, settings=None):
        self.config = settings or config
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = RAGPrompts.rag_prompt()
        self.general_prompt = RAGPrompts.general_prompt()
        self.query_rewriter = QueryRewriter(llm)
        if classifier is None:
            from .query_classifier import QueryClassifier

            classifier = QueryClassifier(
                str(Path(self.config.MODELS_DIR) / "bert_query_classifier")
            )
        if selector is None:
            from .strategy_selector import StrategySelector

            selector = StrategySelector()
        self.query_classifier = classifier
        self.strategy_selector = selector

    def _search(self, query, source_filter):
        return self.vector_store.hybrid_search_with_rerank(
            query, k=self.config.RETRIEVAL_K, source_filter=source_filter
        )

    def _retrieve_with_hyde(self, query, source_filter=None):
        hypothesis = collect_text(self.llm(RAGPrompts.hyde_prompt().format(query=query)))
        return self._search(hypothesis, source_filter)

    def _retrieve_with_subqueries(self, query, source_filter=None):
        text = collect_text(self.llm(RAGPrompts.subquery_prompt().format(query=query)))
        queries = list(
            dict.fromkeys(
                re.sub(r"^\s*(?:[-*]|\d+[.)、])\s*", "", line).strip()
                for line in text.splitlines()
                if line.strip()
            )
        )[:4]
        branches = [self._search(subquery, source_filter) for subquery in queries if subquery]
        return self._fuse(branches, self.config.CANDIDATE_M)

    @staticmethod
    def _identity(doc):
        return (
            doc.metadata.get("source"),
            doc.metadata.get("file_name"),
            doc.metadata.get("parent_id") or doc.page_content,
        )

    @classmethod
    def _fuse(cls, branches, limit):
        """Round-robin unique parents so no query consumes the context budget."""
        queues = [iter(branch) for branch in branches if branch]
        selected, seen = [], set()
        while queues and len(selected) < limit:
            remaining = []
            for queue in queues:
                for doc in queue:
                    key = cls._identity(doc)
                    if key not in seen:
                        seen.add(key)
                        selected.append(doc)
                        break
                else:
                    continue
                remaining.append(queue)
                if len(selected) == limit:
                    break
            queues = remaining
        return selected

    def _retrieve_with_backtracking(self, query, source_filter=None):
        simplified = collect_text(self.llm(RAGPrompts.backtracking_prompt().format(query=query)))
        return self._search(simplified, source_filter)

    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        strategy = strategy or self.strategy_selector.select_strategy(query)
        handlers = {
            "假设问题检索": self._retrieve_with_hyde,
            "子查询检索": self._retrieve_with_subqueries,
            "回溯问题检索": self._retrieve_with_backtracking,
        }
        # Always retain the original question; nonempty transformed results can
        # still omit the very facts requested by the user.
        original = self._search(query, source_filter)
        handler = handlers.get(strategy)
        if handler:
            try:
                docs = handler(query, source_filter=source_filter)
            except Exception as exc:
                logger.warning("检索增强失败，回退到直接检索：%s", type(exc).__name__)
                docs = []
        else:
            docs = []
        return self._fuse([original, docs], self.config.CANDIDATE_M)

    def resolve_query(self, query, source_filter=None, history=None):
        if not getattr(self.config, "QUERY_REWRITE_ENABLED", True):
            return RewriteResult(query)
        return self.query_rewriter.resolve(query, history, source_filter)

    def generate_answer(self, query, source_filter=None, history=None, *, resolve_context=True):
        if resolve_context:
            resolved = self.resolve_query(query, source_filter, history)
            if resolved.clarification:
                yield resolved.clarification
                return
            query = resolved.query
        valid_history = [
            h for h in (history or []) if isinstance(h, dict) and "question" in h and "answer" in h
        ][-5:]
        history_context = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in valid_history)
        requests_documents = bool(
            re.search(r"(?:根据|依据|按照|结合|参照).{0,24}(?:资料|文档|课件|知识库)", query)
        )
        requests_course_facts = bool(
            re.search(r"课程|本课|这门课|培训班|开班", query)
            and re.search(r"学费|价格|退款|退费|开班|班期|师资|承诺|安排", query)
        )
        category = (
            "专业咨询"
            if source_filter or requests_documents or requests_course_facts
            else self.query_classifier.predict_category(query)
        )
        logger.info(
            "问答分流：category=%s scope=%s grounded_request=%s",
            category,
            source_filter or "全部学科",
            requests_documents or requests_course_facts,
        )
        docs = []
        if category != "通用知识":
            docs = self.retrieve_and_merge(query, source_filter=source_filter)
            if not docs:
                yield "当前知识库中没有找到可支持回答的资料，请补充课程资料或调整问题。"
                return
        context = "\n\n".join(
            f"[{i}] {doc.metadata.get('file_name') or doc.metadata.get('source', '课程资料')}\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
        )
        if category == "通用知识":
            prompt = self.general_prompt.format(history=history_context, question=query)
        else:
            prompt = self.rag_prompt.format(
                context=context, history=history_context, question=query
            )
        has_text = False
        for text in iter_text(self.llm(prompt)):
            has_text = True
            yield text
        if not has_text:
            raise ValueError("语言模型返回了空内容")
        if docs:
            references = []
            for i, doc in enumerate(docs, 1):
                name = doc.metadata.get("file_name") or f"{doc.metadata.get('source', '课程')}资料"
                parent = doc.metadata.get("parent_id", "unknown")
                references.append(f"- [{i}] {name}（片段：{parent}）")
            yield "\n\n**本次检索资料**\n" + "\n".join(references)
