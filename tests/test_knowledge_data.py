"""Authored public examples must support a source-only installation."""

import csv
import unittest
from pathlib import Path
from unittest.mock import patch

from base.knowledge_data import knowledge_csv
from demo_backend import DemoIntegratedQASystem

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples/faq_sample.csv"


class ExampleDataTests(unittest.TestCase):
    def test_example_csv_has_complete_rows_in_both_subjects(self):
        with SAMPLE.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["学科名称"] for row in rows}, {"Python学科", "JAVA学科"})
        self.assertTrue(all(row["问题"] and row["答案"] for row in rows))

    def test_demo_can_use_explicit_example_without_course_files(self):
        with patch.dict("os.environ", {"EDURAG_FAQ_CSV": "examples/faq_sample.csv"}):
            self.assertEqual(knowledge_csv(), SAMPLE)
            backend = DemoIntegratedQASystem(history_path=":memory:")
            try:
                question = "Python 列表和元组有什么区别？"
                answer, needs_rag = backend.bm25_search.search(question, source_filter="Python学科")
                self.assertFalse(needs_rag)
                self.assertIn("列表支持修改", answer)
            finally:
                backend.close()

    def test_explicit_missing_file_is_not_silently_replaced(self):
        with patch.dict("os.environ", {"EDURAG_FAQ_CSV": "examples/not-present.csv"}):
            with self.assertRaises(FileNotFoundError):
                knowledge_csv()
