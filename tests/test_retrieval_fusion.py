"""Evidence coverage regressions for original and expanded query retrieval."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from rag_qa.core.new_rag_system import RAGSystem


def document(parent_id, content, source="ai", file_name="课程.md"):
    metadata = {"source": source, "file_name": file_name}
    if parent_id is not None:
        metadata["parent_id"] = parent_id
    return Document(page_content=content, metadata=metadata)


class RetrievalFusionTests(unittest.TestCase):
    def make_system(self, results, expansion="扩展查询", limit=3):
        search = Mock(side_effect=lambda query, **_: list(results.get(query, [])))
        llm = Mock(return_value=expansion)
        system = RAGSystem(
            SimpleNamespace(hybrid_search_with_rerank=search),
            llm,
            classifier=SimpleNamespace(predict_category=lambda _: "专业咨询"),
            selector=SimpleNamespace(select_strategy=lambda _: "直接检索"),
            settings=SimpleNamespace(RETRIEVAL_K=10, CANDIDATE_M=limit, CUSTOMER_SERVICE_PHONE=""),
        )
        return system, search

    def test_nonempty_expansion_cannot_replace_original_evidence(self):
        target = document("target", "原问题所需的具体版本、框架和限制。")
        broad = [document(f"b{i}", f"宽泛介绍{i}") for i in range(3)]
        for strategy in ["假设问题检索", "回溯问题检索"]:
            with self.subTest(strategy=strategy):
                system, search = self.make_system({"原问题": [target], "扩展查询": broad})
                docs = system.retrieve_and_merge("原问题", "ai", strategy)
                self.assertIn("target", [doc.metadata["parent_id"] for doc in docs])
                self.assertTrue(any(doc.metadata["parent_id"].startswith("b") for doc in docs))
                self.assertLessEqual(len(docs), 3)
                self.assertTrue(
                    all(call.kwargs["source_filter"] == "ai" for call in search.call_args_list)
                )

    def test_later_subqueries_contribute_before_first_subquery_fills_budget(self):
        system, _ = self.make_system(
            {
                "原问题": [],
                "第一问": [document(f"a{i}", f"第一方面内容{i}") for i in range(3)],
                "第二问": [document("b", "第二方面的唯一依据。")],
                "第三问": [document("c", "第三方面的唯一依据。")],
            },
            expansion="1. 第一问\n2. 第二问\n3. 第三问",
        )
        docs = system.retrieve_and_merge("原问题", "ai", "子查询检索")
        self.assertEqual({doc.metadata["parent_id"] for doc in docs}, {"a0", "b", "c"})

    def test_repeated_parent_does_not_consume_context_budget(self):
        repeated = document("same", "同一个父块的完整内容。")
        system, _ = self.make_system(
            {
                "原问题": [repeated, document("same", "同一个父块的完整内容。\n")],
                "扩展查询": [repeated, document("other", "补充问题所需的另一父块。")],
            }
        )
        docs = system.retrieve_and_merge("原问题", "ai", "假设问题检索")
        self.assertEqual([doc.metadata["parent_id"] for doc in docs].count("same"), 1)
        self.assertEqual({doc.metadata["parent_id"] for doc in docs}, {"same", "other"})

    def test_same_text_and_parent_id_in_distinct_sources_keeps_provenance(self):
        originals = [
            document("shared-id", "相同文字", source="ai", file_name="资料甲.md"),
            document("shared-id", "相同文字", source="java", file_name="资料甲.md"),
            document("shared-id", "相同文字", source="ai", file_name="资料乙.md"),
        ]
        system, _ = self.make_system({"原问题": originals})
        docs = system.retrieve_and_merge("原问题", strategy="直接检索")
        self.assertEqual(
            {(doc.metadata["source"], doc.metadata["file_name"]) for doc in docs},
            {("ai", "资料甲.md"), ("java", "资料甲.md"), ("ai", "资料乙.md")},
        )

    def test_missing_parent_ids_deduplicate_only_within_same_source_and_file(self):
        system, _ = self.make_system(
            {
                "原问题": [
                    document(None, "相同文字", file_name="资料甲.md"),
                    document(None, "相同文字", file_name="资料甲.md"),
                    document(None, "相同文字", file_name="资料乙.md"),
                ]
            }
        )
        docs = system.retrieve_and_merge("原问题", "ai", "直接检索")
        self.assertEqual(len(docs), 2)
        self.assertEqual({doc.metadata["file_name"] for doc in docs}, {"资料甲.md", "资料乙.md"})


if __name__ == "__main__":
    unittest.main()
