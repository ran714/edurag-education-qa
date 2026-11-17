"""Production FAQ routing, provenance and persistence without external services."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from new_main import IntegratedQASystem
from rag_qa.core.query_rewriter import RewriteResult


class IntegratedFAQTests(unittest.TestCase):
    def setUp(self):
        self.qa = IntegratedQASystem.__new__(IntegratedQASystem)
        self.qa.logger = Mock()
        self.qa.get_session_history = Mock(return_value=[])
        self.qa.update_session_history = Mock()
        self.qa.rag_system = SimpleNamespace(
            resolve_query=Mock(return_value=RewriteResult("independent question")),
            generate_answer=Mock(return_value=iter(["document answer"])),
        )
        self.qa.bm25_search = SimpleNamespace(search=Mock(return_value=(None, True)))
        self.qa.faq_fallback = SimpleNamespace(
            select=Mock(return_value=SimpleNamespace(answer="stored answer", trace={"id": "faq1"}))
        )

    def test_selected_answer_is_original_and_model_use_is_disclosed(self):
        result = list(self.qa.query("question", "Python学科", "session"))
        self.assertEqual(len(result), 1)
        answer, complete = result[0]
        self.assertTrue(complete)
        self.assertIn("stored answer", answer)
        self.assertIn("模型匹配校验", answer)
        self.assertNotIn("未调用大模型", answer)
        self.qa.faq_fallback.select.assert_called_once_with("independent question", "Python学科")
        self.qa.rag_system.generate_answer.assert_not_called()
        self.qa.update_session_history.assert_called_once_with("session", "question", answer)

    def test_direct_answer_skips_model_selection(self):
        self.qa.bm25_search.search.return_value = ("exact answer", False)
        answer, complete = list(self.qa.query("question", "Python学科"))[0]
        self.assertTrue(complete)
        self.assertIn("未调用大模型", answer)
        self.qa.faq_fallback.select.assert_not_called()

    def test_rewritten_direct_answer_does_not_claim_no_model(self):
        self.qa.rag_system.resolve_query.return_value = RewriteResult(
            "resolved query", changed=True, reason="resolved"
        )
        self.qa.bm25_search.search.return_value = ("stored answer", False)
        answer, _ = list(self.qa.query("it?", "Python学科"))[0]
        self.assertIn("模型解析追问", answer)
        self.assertNotIn("未调用大模型", answer)

    def test_selection_abstention_preserves_document_rag_and_scope(self):
        self.qa.faq_fallback.select.return_value = SimpleNamespace(
            answer=None, trace={"reason": "abstain"}
        )
        result = list(self.qa.query("question", "ai", "session"))
        self.assertEqual(result[-1], ("", True))
        self.qa.rag_system.generate_answer.assert_called_once_with(
            "independent question", source_filter="ai", history=[], resolve_context=False
        )
        self.qa.update_session_history.assert_called_once()

    def test_clarification_clears_previous_trace_and_skips_retrieval(self):
        self.qa.last_faq_trace = {"id": "old"}
        self.qa.rag_system.resolve_query.return_value = RewriteResult(
            "it?", clarification="clarify"
        )
        list(self.qa.query("it?", "ai"))
        self.assertIsNone(self.qa.last_faq_trace)
        self.qa.bm25_search.search.assert_not_called()
        self.qa.faq_fallback.select.assert_not_called()


if __name__ == "__main__":
    unittest.main()
