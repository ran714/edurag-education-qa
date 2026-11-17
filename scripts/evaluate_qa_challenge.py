"""Fixed development/test cases: compare improved matching with the preserved v2 baseline."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from evals.baselines.bm25_v2 import BM25Search as Baseline
    from evals.qa_fixtures import MemoryCache, QuestionDatabase
    from mysql_qa.retrieval.bm25_search import BM25Search

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    args = parser.parse_args()
    database = QuestionDatabase()
    label_path = ROOT / "evals/qa_challenge.json"
    cases = json.loads(label_path.read_text(encoding="utf-8"))["cases"]
    cases = [case for case in cases if args.split == "all" or case["split"] == args.split]
    variants = {
        "v2_baseline": Baseline(MemoryCache(), database),
        "v3_candidates": BM25Search(MemoryCache(), database),
    }
    details, grouped = [], defaultdict(list)
    for case in cases:
        gold = (
            database.fetch_answer(case["expected_question"].strip(), case["subject"])
            if case["kind"] == "positive"
            else None
        )
        if case["kind"] == "positive" and not gold:
            raise ValueError("CSV 与标注不一致：" + case["id"])
        for name, system in variants.items():
            answer, _ = system.search(case["query"], source_filter=case["subject"])
            item = {
                **case,
                "variant": name,
                "answered": bool(answer),
                "correct": answer == gold if gold else not bool(answer),
                "wrong_direct_answer": bool(answer and answer != gold),
            }
            if isinstance(system, BM25Search):
                ranked = system.rank(case["query"], case["subject"])
                item["top_candidates"] = [
                    {
                        "question": m.question,
                        "score": round(m.score, 4),
                        "coverage": round(m.coverage, 4),
                        "reason": m.reason,
                    }
                    for m in ranked[:3]
                ]
            details.append(item)
            grouped[(case["split"], name)].append(item)
    summary = {}
    for (split, name), items in grouped.items():
        positives = [i for i in items if i["kind"] == "positive"]
        negatives = [i for i in items if i["kind"] == "negative"]
        answered = [i for i in items if i["answered"]]
        summary[f"{split}/{name}"] = {
            "positive_cases": len(positives),
            "positive_correct_rate": sum(i["correct"] for i in positives) / len(positives),
            "negative_cases": len(negatives),
            "negative_rejection_rate": sum(i["correct"] for i in negatives) / len(negatives),
            "wrong_direct_answers": sum(i["wrong_direct_answer"] for i in items),
            "answered_precision": sum(i["correct"] for i in answered) / len(answered)
            if answered
            else None,
        }
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(
            next((ROOT / "mysql_qa/data").glob("*.csv")).read_bytes()
        ).hexdigest(),
        "implementation_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "mysql_qa/retrieval/bm25_search.py",
                ROOT / "mysql_qa/retrieval/lexical_match.py",
            ]
        },
        "summary": summary,
        "details": details,
        "limitations": "小型人工回归集，开发组/验证组在本轮实现前固定；没有外部盲测。负例表示不应直接套用现有标准答案，不表示通用模型不能回答。不能将结果当作完整系统的生成回答正确率。",
    }
    path = ROOT / f"reports/qa_challenge_{args.split}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(path)


if __name__ == "__main__":
    main()
