"""Read-only CSV fixtures shared by offline evaluators and retrieval regressions."""

import csv

from base.knowledge_data import knowledge_csv


class MemoryCache:
    def __init__(self):
        self.data = {}

    def get_data(self, key):
        return self.data.get(key)

    def set_data(self, key, value):
        self.data[key] = value


class QuestionDatabase:
    def __init__(self, rows=None):
        self.fetch_count = 0
        if rows is not None:
            self.rows = rows
            return
        path = knowledge_csv()
        with path.open(encoding="utf-8-sig", newline="") as file:
            self.rows = [
                {
                    "source": row["学科名称"].strip(),
                    "question": row["问题"].strip(),
                    "answer": row["答案"].strip(),
                }
                for row in csv.DictReader(file)
                if row["问题"].strip() and row["答案"].strip()
            ]

    def fetch_questions(self, source_filter=None):
        self.fetch_count += 1
        return [
            (r["question"],) for r in self.rows if not source_filter or r["source"] == source_filter
        ]

    def fetch_answer(self, question, source_filter=None):
        return next(
            (
                r["answer"]
                for r in self.rows
                if r["question"] == question and (not source_filter or r["source"] == source_filter)
            ),
            None,
        )
