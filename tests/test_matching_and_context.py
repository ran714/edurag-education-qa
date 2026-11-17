"""Matching, cache boundaries and question resolution behavior without external services."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from evals.qa_fixtures import MemoryCache, QuestionDatabase
from mysql_qa.retrieval.bm25_search import BM25Search
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.query_rewriter import QueryRewriter, needs_context


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.database = QuestionDatabase(
            [
                {
                    "source": "Python学科",
                    "question": "ubuntu 中如何安装Selenium",
                    "answer": "install-selenium",
                },
                {"source": "Python学科", "question": "正则中的^符号", "answer": "regex-start"},
                {
                    "source": "Python学科",
                    "question": "如何退出vim\n关于vim退出",
                    "answer": "quit-vim",
                },
                {
                    "source": "Python学科",
                    "question": "mysql5.7 忘记 root密码怎么办？",
                    "answer": "mysql57-reset",
                },
                {
                    "source": "JAVA学科",
                    "question": "申请令牌测试的时候报错401",
                    "answer": "token-401",
                },
                {
                    "source": "Python学科",
                    "question": "datetime与字符串相互转化",
                    "answer": "convert-date",
                },
                {"source": "Python学科", "question": "代码中的!=符号", "answer": "not-equal"},
            ]
        )
        self.cache = MemoryCache()
        self.now = 0
        self.system = BM25Search(self.cache, self.database, clock=lambda: self.now)

    def test_same_meaning_rephrased(self):
        self.assertEqual(
            self.system.search("datetime 和字符串如何互相转换？", source_filter="Python学科"),
            ("convert-date", False),
        )

    def test_complete_multiline_question_and_each_alias(self):
        for query in ["如何退出vim\n关于vim退出", "如何退出vim", "关于vim退出"]:
            self.assertEqual(
                self.system.search(query, source_filter="Python学科"), ("quit-vim", False)
            )

    def test_opposite_action_abstains(self):
        self.assertEqual(
            self.system.search("ubuntu 中如何卸载Selenium", source_filter="Python学科"),
            (None, True),
        )

    def test_regex_symbol_is_not_lost(self):
        self.assertEqual(
            self.system.search("正则中的$符号", source_filter="Python学科"), (None, True)
        )
        self.assertEqual(
            self.system.search("正则中的^符号", source_filter="Python学科"), ("regex-start", False)
        )

    def test_comparison_operators_are_not_interchangeable(self):
        self.assertEqual(
            self.system.search("代码中的==符号", source_filter="Python学科"), (None, True)
        )

    def test_version_change_does_not_reuse_reset_steps(self):
        self.assertEqual(
            self.system.search("mysql8.4 忘记 root密码怎么办？", source_filter="Python学科"),
            (None, True),
        )

    def test_error_code_change_abstains(self):
        self.assertEqual(
            self.system.search("申请令牌测试的时候报错403", source_filter="JAVA学科"), (None, True)
        )

    def test_document_java_scope_does_not_turn_into_standard_qa(self):
        self.assertEqual(
            self.system.search("申请令牌测试的时候报错401", source_filter="java"), (None, True)
        )

    def test_threshold_is_part_of_cache_contract(self):
        self.assertEqual(
            self.system.search("正则中的^符号", source_filter="Python学科"), ("regex-start", False)
        )
        self.assertEqual(
            self.system.search("正则中的^符号", threshold=1.1, source_filter="Python学科"),
            (None, True),
        )

    def test_old_cache_cannot_bypass_new_policy(self):
        self.cache.data["answer:v2:Python学科:正则中的$符号"] = "incorrect"
        self.assertEqual(
            self.system.search("正则中的$符号", source_filter="Python学科"), (None, True)
        )

    def test_subject_index_reused_then_refreshed(self):
        self.system.rank("怎样安装 Selenium", "Python学科")
        count = self.database.fetch_count
        self.system.rank("怎样安装 Selenium", "Python学科")
        self.assertEqual(self.database.fetch_count, count)
        self.now = 61
        self.system.rank("怎样安装 Selenium", "Python学科")
        self.assertEqual(self.database.fetch_count, count + 1)

    def test_unknown_topic_abstains(self):
        self.assertEqual(
            self.system.search("这个东西坏了怎么办", source_filter="Python学科"), (None, True)
        )


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.llm = Mock(
            return_value=json.dumps(
                {"query": "BERT 有哪些特点？", "needs_clarification": False}, ensure_ascii=False
            )
        )
        self.resolver = QueryRewriter(self.llm)
        self.history = [
            {"question": "什么是 BERT？", "answer": "**检索范围：ai**\n\nBERT 是预训练语言模型。"}
        ]

    def test_standalone_question_costs_no_rewrite_call(self):
        result = self.resolver.resolve("BERT 有哪些特点", self.history, "ai")
        self.assertFalse(result.changed)
        self.llm.assert_not_called()

    def test_missing_context_asks_instead_of_searching_pronoun(self):
        result = self.resolver.resolve("它有哪些特点？")
        self.assertIsNotNone(result.clarification)
        self.llm.assert_not_called()

    def test_resolves_follow_up(self):
        result = self.resolver.resolve("它有哪些特点？", self.history, "ai")
        self.assertEqual(result.query, "BERT 有哪些特点？")
        self.assertTrue(result.changed)

    def test_streamed_json_is_supported(self):
        self.llm.return_value = iter(['{"query":"BERT 的特点",', '"needs_clarification":false}'])
        self.assertEqual(self.resolver.resolve("它的特点", self.history, "ai").query, "BERT 的特点")

    def test_scope_switch_does_not_reuse_previous_subject(self):
        result = self.resolver.resolve("它有哪些特点", self.history, "java")
        self.assertEqual(result.reason, "scope_changed")
        self.llm.assert_not_called()

    def test_ambiguous_subjects_ask(self):
        self.llm.return_value = '{"query":"它有哪些特点", "needs_clarification":true}'
        self.assertIsNotNone(
            self.resolver.resolve("它有哪些特点", self.history, "ai").clarification
        )

    def test_malformed_json_asks(self):
        self.llm.return_value = "Use this unstructured answer instead"
        self.assertIsNotNone(
            self.resolver.resolve("它有哪些特点", self.history, "ai").clarification
        )

    def test_provider_failure_asks(self):
        self.llm.side_effect = RuntimeError("provider failed")
        self.assertEqual(
            self.resolver.resolve("它有哪些特点", self.history, "ai").reason, "rewrite_unavailable"
        )

    def test_invented_entity_rejected(self):
        self.llm.return_value = '{"query":"GPT-99 的特点", "needs_clarification":false}'
        self.assertIsNotNone(
            self.resolver.resolve("它有哪些特点", self.history, "ai").clarification
        )

    def test_current_version_and_exclusion_must_be_preserved(self):
        self.llm.return_value = '{"query":"BERT 的特点", "needs_clarification":false}'
        self.assertIsNotNone(
            self.resolver.resolve("它在 Python3.12 中能用吗", self.history, "ai").clarification
        )
        self.assertIsNotNone(
            self.resolver.resolve("它能否不依赖 BERT 实现", self.history, "ai").clarification
        )

    def test_explicit_error_name_does_not_require_context(self):
        self.assertFalse(needs_context("这个 ModuleNotFoundError 怎么解决？"))

    def test_word_order_and_position_do_not_require_conversation_context(self):
        questions = [
            "unigram、bigram、trigram 对前面词语的依赖有什么区别？",
            "bigram 是否只依赖前面一个词？",
            "如何在链表前面插入节点？",
            "Python 列表前面的元素如何删除？",
        ]
        for question in questions:
            with self.subTest(question=question):
                result = self.resolver.resolve(question, source_filter="ai")
                self.assertEqual(result.query, question)
                self.assertIsNone(result.clarification)
                self.assertEqual(result.reason, "standalone")
        self.llm.assert_not_called()

    def test_prior_discussion_still_requires_context(self):
        for question in [
            "前面提到的模型有哪些特点？",
            "前面讨论的模型有哪些特点？",
            "前面讲过的方法适用什么场景？",
            "前面的模型有哪些特点？",
            "前面那个模型有哪些特点？",
        ]:
            with self.subTest(question=question):
                result = self.resolver.resolve(question)
                self.assertEqual(result.reason, "missing_context")
                self.assertIsNotNone(result.clarification)
        self.llm.assert_not_called()

    def test_prior_discussion_resolves_and_respects_scope_change(self):
        question = "前面提到的模型有哪些特点？"
        result = self.resolver.resolve(question, self.history, "ai")
        self.assertEqual(result.query, "BERT 有哪些特点？")
        self.llm.reset_mock()
        result = self.resolver.resolve(question, self.history, "java")
        self.assertEqual(result.reason, "scope_changed")
        self.llm.assert_not_called()

    def test_named_target_does_not_hide_another_unresolved_reference(self):
        question = "它和这个 BERT 有什么区别？"
        result = self.resolver.resolve(question)
        self.assertEqual(result.reason, "missing_context")
        self.assertIsNotNone(result.clarification)
        self.llm.assert_not_called()

    def test_resolved_query_actually_reaches_retriever(self):
        documents = [
            Document(
                page_content="BERT 的特点是双向编码。",
                metadata={"source": "ai", "file_name": "资料.md", "parent_id": "bert"},
            )
        ]
        store = SimpleNamespace(hybrid_search_with_rerank=Mock(return_value=documents))
        llm = Mock(
            side_effect=[
                json.dumps(
                    {"query": "BERT 有哪些特点？", "needs_clarification": False}, ensure_ascii=False
                ),
                "BERT 使用双向编码。[1]",
            ]
        )
        rag = RAGSystem(
            store,
            llm,
            classifier=SimpleNamespace(predict_category=lambda _: "专业咨询"),
            selector=SimpleNamespace(select_strategy=lambda _: "直接检索"),
            settings=SimpleNamespace(RETRIEVAL_K=10, CANDIDATE_M=3, CUSTOMER_SERVICE_PHONE=""),
        )
        answer = "".join(rag.generate_answer("它有哪些特点？", "ai", self.history))
        store.hybrid_search_with_rerank.assert_called_once_with(
            "BERT 有哪些特点？", k=10, source_filter="ai"
        )
        self.assertIn("双向编码", answer)
        self.assertEqual(llm.call_count, 2)


if __name__ == "__main__":
    unittest.main()
