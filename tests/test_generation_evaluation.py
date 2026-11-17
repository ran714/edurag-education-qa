"""Ensure the evaluation preserves answers and reports missing scores honestly."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_generation import (
    append_jsonl,
    audit_scores,
    clean_answer,
    latest_scores,
    load_cases,
    read_jsonl,
    summarize,
)


class GenerationEvaluationTests(unittest.TestCase):
    def test_invalid_reference_is_flagged_without_overwriting_raw_evidence(self):
        rows = [
            {"id": "a", "metric": "factual_correctness", "status": "ok", "score": 1.0},
            {"id": "a", "metric": "faithfulness", "status": "ok", "score": 1.0},
        ]
        audited = audit_scores(rows, {"a"})
        self.assertEqual(rows[0]["score"], 1.0)
        self.assertEqual(audited[0]["status"], "invalid_reference")
        self.assertEqual(audited[0]["raw_score"], 1.0)
        self.assertNotIn("score", audited[0])
        self.assertEqual(audited[1]["score"], 1.0)
        self.assertTrue(audited[1]["context_noninformative"])

    def test_retry_does_not_double_count_a_sample(self):
        rows = [
            {"id": "a", "metric": "faithfulness", "status": "metric_failed"},
            {"id": "a", "metric": "faithfulness", "status": "ok", "score": 0.75},
        ]
        self.assertEqual(latest_scores(rows), [rows[-1]])

    def test_answer_body_keeps_code_but_removes_transport_metadata(self):
        raw = "**检索范围：ai**\n\n正文[1]\n```python\nprint('__name__')\n```\n\n**本次检索资料**\n- [1] 资料.md（片段：p1）"
        body = clean_answer(raw)
        self.assertIn("print('__name__')", body)
        self.assertIn("正文[1]", body)
        self.assertNotIn("片段：p1", body)
        self.assertNotIn("检索范围", body)

    def test_standard_answer_marker_is_not_part_of_scored_content(self):
        raw = "**检索范围：Python学科**\n\n标准答案\n\n_来源：课程标准问答库（BM25 检索），未调用大模型。_"
        self.assertEqual(clean_answer(raw), "标准答案")

    def test_fixed_cases_cover_prior_fallbacks_and_boundaries(self):
        if not (Path(__file__).resolve().parents[1] / "evals/generation_cases.json").exists():
            self.skipTest("源码发布版不包含原课程题组；该项仅验证本地历史评测资料")
        cases, _ = load_cases()
        self.assertEqual(len(cases), 62)
        self.assertEqual(len({case["id"] for case in cases}), 62)
        self.assertEqual(sum(case["category"] == "faq_paraphrase" for case in cases), 36)
        self.assertTrue(any(case.get("setup") for case in cases))
        self.assertTrue(all(case["reference"].strip() for case in cases))
        self.assertTrue(
            {"answer", "refuse", "clarify"}.issubset({case["expected_behavior"] for case in cases})
        )

    def test_missing_and_failed_scores_are_not_converted_to_zero(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text(json.dumps({"case_count": 3}), encoding="utf-8")
            for index in range(3):
                append_jsonl(
                    root / "responses.jsonl",
                    {"id": str(index), "status": "ok", "disposition_heuristic": "refusal"},
                )
            for value in [
                {
                    "id": "0",
                    "category": "document",
                    "metric": "faithfulness",
                    "status": "ok",
                    "score": 0.8,
                },
                {
                    "id": "1",
                    "category": "document",
                    "metric": "faithfulness",
                    "status": "not_applicable",
                    "reason": "no_context",
                },
                {
                    "id": "2",
                    "category": "document",
                    "metric": "faithfulness",
                    "status": "metric_failed",
                    "error_type": "TimeoutError",
                },
            ]:
                append_jsonl(root / "scores.jsonl", value)
            result = summarize(root)
            row = result["metrics"]["document/faithfulness"]
            self.assertEqual(row["mean_score"], 0.8)
            self.assertEqual(row["valid_cases"], 1)
            self.assertEqual(row["total_cases"], 3)

    def test_jsonl_checkpoint_round_trip(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            append_jsonl(path, {"answer": "中文资料"})
            append_jsonl(path, {"score": 0.0})
            self.assertEqual(read_jsonl(path), [{"answer": "中文资料"}, {"score": 0.0}])


if __name__ == "__main__":
    unittest.main()
