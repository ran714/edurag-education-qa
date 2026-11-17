"""FAQ fallback boundaries without provider calls or mutable databases."""

import json
import unittest
from unittest.mock import Mock

from evals.qa_fixtures import MemoryCache, QuestionDatabase
from mysql_qa.retrieval.answer_quality import is_usable_answer
from mysql_qa.retrieval.bm25_search import BM25Search
from mysql_qa.retrieval.semantic_fallback import SemanticFAQFallback, candidate_conflict


class FAQFallbackTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "source": "Python学科",
                "question": "如何用上下文管理器实现函数运行时间的计算?",
                "answer": "使用 contextlib.contextmanager，进入时记录 time.time()，yield 后计算耗时。",
            },
            {
                "source": "JAVA学科",
                "question": "如何用上下文管理器实现函数运行时间的计算?",
                "answer": "other-subject-answer",
            },
        ]
        self.search = BM25Search(MemoryCache(), QuestionDatabase(self.rows))
        self.llm = Mock(return_value='{"candidate_id":"faq1","answerable":true}')
        self.selector = SemanticFAQFallback(self.search, self.llm)
        self.query = "怎样通过 Python 上下文管理器统计函数执行耗时？"

    def test_selects_only_existing_body_in_requested_subject(self):
        self.assertIsNone(self.search.search(self.query, source_filter="Python学科")[0])
        result = self.selector.select(self.query, "Python学科")
        self.assertEqual(result.answer, self.rows[0]["answer"])
        self.assertEqual(result.question, self.rows[0]["question"])
        self.assertEqual(result.trace["selected_id"], "faq1")
        self.assertNotIn("other-subject-answer", self.llm.call_args.args[0])

    def test_global_scope_cannot_mix_subjects(self):
        self.assertEqual(
            self.selector.select(self.query).trace["reason"], "explicit_qa_scope_required"
        )
        self.llm.assert_not_called()

    def test_document_scope_does_not_expand_to_qa_scope(self):
        self.assertIsNone(self.selector.select(self.query, "java").answer)
        self.llm.assert_not_called()

    def test_strict_schema_and_unknown_ids_fail_safely(self):
        for output in [
            '{"candidate_id":"faq99","answerable":true}',
            '{"candidate_id":"faq1","answerable":"true"}',
            '{"candidate_id":"faq1","answerable":true,"answer":"invented"}',
            '{"candidate_id":"faq1","answerable":false}',
            '```json\n{"candidate_id":"faq1","answerable":true}\n```',
        ]:
            with self.subTest(output=output):
                self.llm.return_value = output
                self.assertIsNone(self.selector.select(self.query, "Python学科").answer)

    def test_model_abstention_and_failure_preserve_fallback(self):
        self.llm.return_value = '{"candidate_id":null,"answerable":false}'
        self.assertEqual(
            self.selector.select(self.query, "Python学科").trace["reason"],
            "answer_not_supported",
        )
        self.llm.side_effect = RuntimeError("private-provider-information")
        result = self.selector.select(self.query, "Python学科")
        self.assertIsNone(result.answer)
        self.assertNotIn("private-provider-information", json.dumps(result.trace))

    def test_streaming_callback_is_supported(self):
        self.llm.return_value = iter(['{"candidate_id":"faq1",', '"answerable":true}'])
        self.assertEqual(
            self.selector.select(self.query, "Python学科").answer, self.rows[0]["answer"]
        )

    def test_full_alias_row_preserves_entity_coverage(self):
        self.assertIsNone(
            candidate_conflict(
                "使用 scrapy genspider 创建爬虫时要注意哪些问题？",
                "scrapy genspider name domain\n创建spider有哪些注意点",
            )
        )

    def test_versions_normalize_spacing_without_changing_values(self):
        self.assertIsNone(
            candidate_conflict("忘了 MySQL 5.7 的 root 密码", "mysql5.7 忘记 root密码怎么办？")
        )
        self.assertIsNotNone(
            candidate_conflict("MySQL 8.4 忘记 root 密码", "mysql5.7 忘记 root密码怎么办？")
        )

    def test_hard_negative_constraints_cannot_reach_model(self):
        cases = [
            ("ubuntu 中如何卸载Selenium", "ubuntu 中如何安装Selenium"),
            ("正则表达式中的 $ 符号代表什么？", "正则中的^符号"),
            ("不使用上下文管理器怎样统计函数运行时间？", self.rows[0]["question"]),
            ("忘记 PostgreSQL 数据库的 root 密码", "mysql5.7 忘记 root密码怎么办？"),
            ("申请令牌时服务返回 403", "申请令牌测试的时候报错401"),
            ("如何删除 Vim 编辑器，而不是退出编辑模式？", "关于vim 退出"),
            ("微信退款回调 jar 包怎样安装？", "微信支付jar包安装步骤"),
            ("JavaScript 被称为胶水语言的原因", "为什么说python是一种胶水语言"),
            ("NumPy 的 __qualname__ 与 fit_transform 有什么区别？", "__name__与__qualname__的区别"),
        ]
        for query, question in cases:
            with self.subTest(query=query):
                search = BM25Search(
                    MemoryCache(),
                    QuestionDatabase(
                        [{"source": "Python学科", "question": question, "answer": "已有解答正文"}]
                    ),
                )
                llm = Mock(return_value='{"candidate_id":"faq1","answerable":true}')
                self.assertIsNone(
                    SemanticFAQFallback(search, llm).select(query, "Python学科").answer
                )
                llm.assert_not_called()

    def test_oversized_bodies_are_not_partially_shown_then_returned(self):
        self.rows[0]["answer"] = "长答案" * 3000
        self.assertIsNone(self.selector.select(self.query, "Python学科").answer)
        self.llm.assert_not_called()

    def test_placeholder_exact_answer_is_not_returned_or_cached(self):
        self.rows[0]["answer"] = (
            '{"answerText":"d310928c-9dba-45ea-91fc-e058f0758c4d","answerType":"NEWS"}'
        )
        self.assertIsNone(
            self.search.search(self.rows[0]["question"], source_filter="Python学科")[0]
        )
        self.assertEqual(self.search.redis_client.data, {})
        self.assertIsNone(self.selector.select(self.query, "Python学科").answer)
        self.llm.assert_not_called()


class AnswerQualityTests(unittest.TestCase):
    def test_missing_bodies_are_rejected(self):
        for body in [
            "",
            None,
            "d310928c-9dba-45ea-91fc-e058f0758c4d",
            "具体参考下:https://example.invalid/path",
            "详情请看这篇文章：https://example.invalid/path",
            "https://example.invalid/path",
            "看视频一步一步操作一下.",
        ]:
            with self.subTest(body=body):
                self.assertFalse(is_usable_answer(body))

    def test_substantive_answer_may_contain_reference_link(self):
        self.assertTrue(
            is_usable_answer("先检查防火墙设置，再核对服务端口。参考：https://example.invalid")
        )
        self.assertTrue(
            is_usable_answer("到 PyCharm 官网选择 Community 版本下载：https://example.invalid")
        )


if __name__ == "__main__":
    unittest.main()
