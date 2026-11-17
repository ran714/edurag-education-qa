"""Offline regression tests. No provider keys, Docker or model downloads."""

import os

os.environ["EDURAG_MODE"] = "demo"
os.environ["EDURAG_SESSION_DB"] = ":memory:"

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import create_app
from base.config import Config, parse_sources
from base.llm_stream import collect_text, iter_text
from demo_backend import DemoIntegratedQASystem
from rag_qa.core.new_rag_system import RAGSystem


class StreamTests(unittest.TestCase):
    def test_string_is_one_chunk(self):
        self.assertEqual(list(iter_text("完整回答")), ["完整回答"])

    def test_collects_generator(self):
        self.assertEqual(collect_text(iter([" a", "b "])), "ab")

    def test_empty_result_is_failure(self):
        with self.assertRaises(ValueError):
            collect_text(iter([]))

    def test_invalid_chunk_is_failure(self):
        with self.assertRaises(TypeError):
            list(iter_text([{"text": "wrong contract"}]))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.document = Document(
            page_content="课程覆盖 Python 和 RAG。",
            metadata={"source": "ai", "file_name": "课程.md", "parent_id": "p1"},
        )
        self.store = SimpleNamespace(hybrid_search_with_rerank=Mock(return_value=[self.document]))
        self.llm = Mock(side_effect=lambda _: iter(["生成", "内容"]))
        self.rag = RAGSystem(
            self.store,
            self.llm,
            classifier=SimpleNamespace(predict_category=lambda _: "专业咨询"),
            selector=SimpleNamespace(select_strategy=lambda _: "直接检索"),
            settings=SimpleNamespace(RETRIEVAL_K=10, CANDIDATE_M=3, CUSTOMER_SERVICE_PHONE=""),
        )

    def test_streaming_hyde_preserves_scope(self):
        self.rag.retrieve_and_merge("问题", "ai", "假设问题检索")
        calls = self.store.hybrid_search_with_rerank.call_args_list
        self.assertEqual([item.args[0] for item in calls], ["问题", "生成内容"])
        self.assertTrue(all(item.kwargs["source_filter"] == "ai" for item in calls))
        self.assertTrue(all(item.kwargs["k"] == 10 for item in calls))

    def test_streaming_backtracking_preserves_scope(self):
        self.rag.retrieve_and_merge("问题", "ai", "回溯问题检索")
        calls = self.store.hybrid_search_with_rerank.call_args_list
        self.assertEqual([item.args[0] for item in calls], ["问题", "生成内容"])
        self.assertTrue(all(item.kwargs["source_filter"] == "ai" for item in calls))
        self.assertTrue(all(item.kwargs["k"] == 10 for item in calls))

    def test_subqueries_are_bounded_deduplicated_and_scoped(self):
        self.llm.side_effect = lambda _: iter(["1. A\n2. B\n3. A\n4. C\n5. D\n6. E"])
        docs = self.rag.retrieve_and_merge("问题", "ai", "子查询检索")
        self.assertEqual(len(docs), 1)
        self.assertEqual(self.store.hybrid_search_with_rerank.call_count, 5)
        self.assertEqual(
            [item.args[0] for item in self.store.hybrid_search_with_rerank.call_args_list],
            ["问题", "A", "B", "C", "D"],
        )
        self.assertTrue(
            all(
                call.kwargs["source_filter"] == "ai"
                for call in self.store.hybrid_search_with_rerank.call_args_list
            )
        )

    def test_strategy_failure_falls_back_to_original_query(self):
        self.llm.side_effect = RuntimeError("provider failed")
        self.rag.retrieve_and_merge("原问题", "ai", "假设问题检索")
        calls = self.store.hybrid_search_with_rerank.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0], "原问题")
        self.assertEqual(calls[0].kwargs["source_filter"], "ai")

    def test_no_evidence_refuses_without_calling_llm(self):
        self.store.hybrid_search_with_rerank.return_value = []
        answer = "".join(self.rag.generate_answer("价格是多少", "ai"))
        self.assertIn("没有找到", answer)
        self.llm.assert_not_called()

    def test_answer_has_document_references(self):
        answer = "".join(self.rag.generate_answer("问题", "ai"))
        self.assertIn("课程.md", answer)
        self.assertIn("p1", answer)
        self.assertIn("[1]", self.llm.call_args.args[0])

    def test_mid_stream_failure_propagates(self):
        def failing(_):
            yield "部分回答"
            raise RuntimeError("provider failed")

        self.llm.side_effect = failing
        with self.assertRaises(RuntimeError):
            list(self.rag.generate_answer("问题", "ai"))

    def test_string_provider_is_not_split_into_characters(self):
        self.llm.side_effect = None
        self.llm.return_value = "完整回答"
        self.assertEqual(next(self.rag.generate_answer("问题", "ai")), "完整回答")

    def test_explicit_scope_always_uses_retrieval(self):
        self.rag.query_classifier.predict_category = lambda _: "通用知识"
        list(self.rag.generate_answer("问题", "ai"))
        self.store.hybrid_search_with_rerank.assert_called_once()

    def test_general_question_uses_general_knowledge_without_retrieval(self):
        self.rag.query_classifier.predict_category = lambda _: "通用知识"
        answer = "".join(self.rag.generate_answer("机器学习有哪些算法？"))
        self.assertEqual(answer, "生成内容")
        self.store.hybrid_search_with_rerank.assert_not_called()
        prompt = self.llm.call_args.args[0]
        self.assertIn("通用知识说明", prompt)
        self.assertNotIn("电话", prompt.replace("客服电话", ""))
        self.assertNotIn("本次检索资料", answer)

    def test_explicit_document_request_overrides_general_classifier(self):
        self.rag.query_classifier.predict_category = lambda _: "通用知识"
        answer = "".join(self.rag.generate_answer("请根据课程资料解释机器学习"))
        self.store.hybrid_search_with_rerank.assert_called_once()
        self.assertIn("课程.md", answer)

    def test_course_price_cannot_fall_back_to_general_knowledge(self):
        self.rag.query_classifier.predict_category = lambda _: "通用知识"
        self.store.hybrid_search_with_rerank.return_value = []
        answer = "".join(self.rag.generate_answer("本课程的学费和退款安排是什么？"))
        self.assertIn("没有找到", answer)
        self.llm.assert_not_called()

    def test_explicit_scope_uses_grounded_prompt(self):
        self.rag.query_classifier.predict_category = lambda _: "通用知识"
        list(self.rag.generate_answer("机器学习有哪些算法？", "ai"))
        self.assertIn("课程覆盖 Python 和 RAG", self.llm.call_args.args[0])
        self.assertNotIn("不需要课程资料", self.llm.call_args.args[0])


class APITests(unittest.TestCase):
    def setUp(self):
        self.backend = DemoIntegratedQASystem(history_path=":memory:")
        self.client = TestClient(create_app(self.backend, mode="demo"))
        self.row = self.backend.rows[0]
        self.session = "regression-session"

    def tearDown(self):
        self.client.close()
        self.backend.close()

    def test_rest_standard_answer_saved_once(self):
        response = self.client.post(
            "/api/query",
            json={
                "query": self.row["question"],
                "source_filter": self.row["source"],
                "session_id": self.session,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["is_streaming"])
        history = self.client.get("/api/history/" + self.session).json()["history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["answer"], response.json()["answer"])

    def test_mixed_greeting_reaches_question(self):
        response = self.client.post(
            "/api/query", json={"query": "你好，请解释 RAG 是什么", "stream": False}
        )
        self.assertIn("Retrieval-Augmented Generation", response.json()["answer"])

    def test_pure_greeting_is_saved(self):
        self.client.post("/api/query", json={"query": "你好！", "session_id": self.session})
        self.assertEqual(len(self.backend.get_session_history(self.session)), 1)

    def test_whitespace_question_rejected(self):
        self.assertEqual(self.client.post("/api/query", json={"query": "   "}).status_code, 422)

    def test_unknown_scope_rejected(self):
        self.assertEqual(
            self.client.post(
                "/api/query", json={"query": "问题", "source_filter": "ai' or true"}
            ).status_code,
            422,
        )

    def test_session_id_limit_matches_storage_contract(self):
        self.assertEqual(
            self.client.post(
                "/api/query", json={"query": "你好", "session_id": "s" * 128}
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.post(
                "/api/query", json={"query": "你好", "session_id": "s" * 129}
            ).status_code,
            422,
        )

    def test_websocket_reconstructs_full_answer_and_saves(self):
        with self.client.websocket_connect("/api/stream") as socket:
            socket.send_json({"query": self.row["question"], "session_id": self.session})
            chunks = []
            while True:
                event = socket.receive_json()
                if event["type"] == "token":
                    chunks.append(event["token"])
                elif event["type"] == "end":
                    self.assertEqual(event["full_answer"], "".join(chunks))
                    self.assertIn(self.row["answer"], event["full_answer"])
                    break
                elif event["type"] == "error":
                    self.fail(event["error"])
        self.assertEqual(len(self.backend.get_session_history(self.session)), 1)

    def test_websocket_bad_payload_is_explicit_error(self):
        with self.client.websocket_connect("/api/stream") as socket:
            socket.send_json({"query": None})
            self.assertEqual(socket.receive_json()["type"], "error")

    def test_greeting_websocket_includes_full_answer(self):
        with self.client.websocket_connect("/api/stream") as socket:
            socket.send_json({"query": "hello"})
            self.assertEqual(socket.receive_json()["type"], "start")
            token = socket.receive_json()["token"]
            self.assertEqual(socket.receive_json()["full_answer"], token)

    def test_other_subject_does_not_expand_to_global(self):
        python_row = next(
            r
            for r in self.backend.rows
            if r["source"] == "Python学科"
            and self.backend.best_match(r["question"], "JAVA学科")[0] < 0.58
        )
        answer = self.backend._build_answer(python_row["question"], "JAVA学科")
        self.assertNotIn("已扩展", answer)
        self.assertNotIn("**Python学科**", answer)

    def test_history_clear_is_session_scoped(self):
        self.backend.update_session_history("a", "q", "a")
        self.backend.update_session_history("b", "q", "b")
        self.client.delete("/api/history/a")
        self.assertEqual(self.backend.get_session_history("a"), [])
        self.assertEqual(len(self.backend.get_session_history("b")), 1)

    def test_health_reports_real_mode(self):
        self.assertEqual(self.client.get("/health").json()["mode"], "demo")
        self.assertEqual(self.client.get("/ready").status_code, 200)

    def test_rest_provider_failure_is_not_success_or_secret(self):
        def failing(*args, **kwargs):
            raise RuntimeError("private-provider-details")
            yield "", True

        self.backend.query = failing
        response = self.client.post("/api/query", json={"query": "问题", "stream": False})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-provider-details", response.text)

    def test_stream_failure_has_error_event_and_no_saved_answer(self):
        def failing(*args, **kwargs):
            yield "部分内容", False
            raise RuntimeError("private-provider-details")

        self.backend.query = failing
        with self.client.websocket_connect("/api/stream") as socket:
            socket.send_json({"query": "问题", "session_id": self.session})
            self.assertEqual(socket.receive_json()["type"], "start")
            self.assertEqual(socket.receive_json()["type"], "token")
            event = socket.receive_json()
            self.assertEqual(event["type"], "error")
            self.assertNotIn("private-provider-details", event["error"])
        self.assertEqual(self.backend.get_session_history(self.session), [])

    def test_health_responds_while_backend_is_waiting(self):
        entered, release = Event(), Event()

        def slow(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timeout")
            yield "answer", True

        self.backend.query = slow
        results = []
        worker = Thread(
            target=lambda: results.append(
                self.client.post("/api/query", json={"query": "问题", "stream": False}).status_code
            )
        )
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(self.client.get("/health").status_code, 200)
        finally:
            release.set()
            worker.join(timeout=5)
        self.assertEqual(results, [200])

    def test_retrieval_ablation_is_not_simulated_in_demo(self):
        self.assertEqual(self.client.post("/api/retrieve", json={"query": "课程"}).status_code, 409)


class PersistenceTests(unittest.TestCase):
    def test_restart_retains_last_five_turns(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.db"
            first = DemoIntegratedQASystem(history_path=path)
            for i in range(7):
                first.update_session_history("session", str(i), str(i))
            first.close()
            second = DemoIntegratedQASystem(history_path=path)
            self.assertEqual(
                [h["question"] for h in second.get_session_history("session")],
                ["2", "3", "4", "5", "6"],
            )
            second.close()


class ConfigurationTests(unittest.TestCase):
    def test_rerank_setting_and_environment_override(self):
        self.assertFalse(Config("config.example.ini").RERANK_ENABLED)
        with patch.dict(os.environ, {"RETRIEVAL_RERANK_ENABLED": "true"}):
            self.assertTrue(Config("config.example.ini").RERANK_ENABLED)

    def test_legacy_and_example_source_formats(self):
        self.assertEqual(parse_sources('["ai", "java"]'), ["ai", "java"])
        self.assertEqual(parse_sources("ai, java, ai"), ["ai", "java"])
        self.assertIn("ai", Config("config.example.ini").VALID_SOURCES)

    def test_malicious_list_is_not_executed(self):
        with self.assertRaises((SyntaxError, ValueError)):
            parse_sources("[__import__('os').system('should-never-run')]")


if __name__ == "__main__":
    unittest.main()
