"""Check comparable denominators, frozen evidence and safe real-input tracing."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.evaluate_generation import append_jsonl, read_jsonl
from scripts.evaluate_improvement import (
    ROOT,
    RuntimeTrace,
    audited_invalid_ids,
    compare_metric,
    context_view,
    file_digest,
    frozen_cases,
    summarize,
    validate_same_cases,
)


def sample(key="a", category="faq_paraphrase", reference="原始答案"):
    return {
        "id": key,
        "category": category,
        "query": "原始问题",
        "source_filter": "Python学科",
        "expected_behavior": "answer",
        "reference": reference,
        "status": "ok",
    }


class ImprovementEvaluationTests(unittest.TestCase):
    def test_changed_question_reference_or_setup_cannot_be_compared(self):
        old = sample()
        for changed in ({"query": "另一个问题"}, {"reference": "另一个答案"}, {"setup": []}):
            with self.assertRaisesRegex(ValueError, "changed_case_or_reference"):
                validate_same_cases([old], [{**old, **changed}])

    def test_paired_mean_uses_same_ids_without_zero_filling(self):
        before = [
            {"id": "a", "status": "ok", "score": 0.8},
            {"id": "b", "status": "not_applicable"},
            {"id": "c", "status": "ok", "score": 0.2},
        ]
        after = [
            {"id": "a", "status": "ok", "score": 0.9},
            {"id": "b", "status": "ok", "score": 1.0},
            {"id": "c", "status": "metric_failed"},
        ]
        result = compare_metric(before, after, {"a", "b", "c", "d"})
        self.assertEqual(result["paired_case_ids"], ["a"])
        self.assertAlmostEqual(result["paired_mean_difference"], 0.1)
        self.assertEqual(result["newly_valid_ids"], ["b"])
        self.assertEqual(result["lost_valid_ids"], ["c"])
        self.assertEqual(result["current"]["status_counts"]["missing"], 1)
        self.assertAlmostEqual(result["current"]["mean_score"], 0.95)

    def test_context_only_comes_from_actual_citations_or_returned_answer(self):
        parents = {"p1": {"parent_content": "实际资料"}}
        raw = "回答[1]\n\n**本次检索资料**\n- [1] 文档（片段：p1）\n- [2] 缺失（片段：p2）"
        view = context_view(raw, parents, [{"answer": "不能注入的参考答案"}])
        self.assertEqual(view["retrieved_contexts"], ["实际资料"])
        self.assertEqual(view["missing_parent_ids"], ["p2"])
        answer = "实际标答\n\n_来源：课程标准问答库（现有答案复用）。_"
        view = context_view(answer, {}, [{"answer": "实际标答"}, {"answer": "其他答案"}])
        self.assertEqual(view["retrieved_contexts"], ["实际标答"])
        self.assertEqual(context_view("未命中", {}, [{"answer": "参考"}])["retrieved_contexts"], [])

    def test_audit_applies_in_new_directory_but_rejects_changed_reference(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            baseline = directory / "baseline"
            baseline.mkdir()
            row = sample(reference="opaque-id")
            original = baseline / "responses.jsonl"
            append_jsonl(original, row)
            audit = directory / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "source_report": str(original),
                        "source_report_sha256": file_digest(original),
                        "cases": [{"id": "a", "evidence": {"reference_original": "opaque-id"}}],
                    }
                ),
                encoding="utf-8",
            )
            invalid, evidence = audited_invalid_ids(baseline, [dict(row)], audit)
            self.assertEqual(invalid, {"a"})
            self.assertIn("a", evidence["reference_sha256"])
            with self.assertRaisesRegex(ValueError, "changed_case_or_reference"):
                audited_invalid_ids(baseline, [{**row, "reference": "补造的新参考"}], audit)

    def test_faq_scope_does_not_claim_full_pipeline_cases(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, current = root / "baseline", root / "current"
            baseline.mkdir()
            current.mkdir()
            faq, document = sample(), sample("d", "document")
            for row in (faq, document):
                append_jsonl(baseline / "responses.jsonl", row)
            append_jsonl(current / "responses.jsonl", faq)
            metric = {"id": "a", "category": "faq_paraphrase", "metric": "factual_correctness"}
            append_jsonl(baseline / "scores.jsonl", {**metric, "status": "ok", "score": 0.0})
            append_jsonl(current / "scores.jsonl", {**metric, "status": "ok", "score": 0.75})
            (current / "manifest.json").write_text(
                json.dumps(
                    {
                        "evaluation_scope": "faq_only",
                        "scope_case_ids": ["a"],
                        "limitations": ["CSV组件实验，不代表HTTP端到端。"],
                    }
                ),
                encoding="utf-8",
            )
            audit = root / "audit.json"
            source = baseline / "responses.jsonl"
            audit.write_text(
                json.dumps(
                    {
                        "source_report": str(source),
                        "source_report_sha256": file_digest(source),
                        "cases": [],
                    }
                ),
                encoding="utf-8",
            )
            result = summarize(current, baseline, audit)
            self.assertEqual(result["baseline_cases"], 1)
            self.assertEqual(result["expected_metric_results"], 3)
            self.assertEqual(result["comparison"]["all/factual_correctness"]["paired_cases"], 1)
            self.assertNotIn("document/factual_correctness", result["comparison"])
            self.assertIn("CSV组件实验", (current / "report.md").read_text(encoding="utf-8"))
            self.assertEqual(read_jsonl(source), [faq, document])

    def test_model_trace_preserves_messages_without_recording_credentials(self):
        replies = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="结果"))]
        )
        completions = SimpleNamespace(create=lambda **kwargs: replies)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions), api_key="secret")
        trace = RuntimeTrace()
        trace.wrap_model(client, "selector")
        response = client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "真实提示"}],
            extra_headers={"Authorization": "secret"},
        )
        self.assertIs(response, replies)
        self.assertEqual(trace.events[0]["request"]["messages"][0]["content"], "真实提示")
        self.assertNotIn("secret", json.dumps(trace.events))

    def test_original_62_cases_are_frozen_and_include_36_faqs(self):
        if not (ROOT / "reports/generation_quality/cases.json").exists():
            self.skipTest("源码发布版不包含原课程题组；该项仅验证本地历史评测资料")
        cases = frozen_cases(ROOT / "reports/generation_quality")
        self.assertEqual(len(cases), 62)
        self.assertEqual(sum(row["category"] == "faq_paraphrase" for row in cases), 36)


if __name__ == "__main__":
    unittest.main()
