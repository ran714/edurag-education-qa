"""Reproducible retrieval checks. Never treats retrieval scores as answer accuracy."""

import argparse
import hashlib
import json
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def qa_evaluation():
    from demo_backend import DemoIntegratedQASystem
    from mysql_qa.retrieval.bm25_search import BM25Search

    backend = DemoIntegratedQASystem(history_path=":memory:")
    rows = backend.rows

    class Cache:
        def __init__(self):
            self.data = {}

        def get_data(self, key):
            return self.data.get(key)

        def set_data(self, key, value):
            self.data[key] = value

    class Database:
        def fetch_questions(self, source_filter=None):
            return [
                (r["question"],) for r in rows if not source_filter or r["source"] == source_filter
            ]

        def fetch_answer(self, question, source_filter=None):
            return next(
                (
                    r["answer"]
                    for r in rows
                    if r["question"] == question
                    and (not source_filter or r["source"] == source_filter)
                ),
                None,
            )

    bm25 = BM25Search(Cache(), Database())
    paraphrases = json.loads((ROOT / "evals/qa_paraphrases.json").read_text(encoding="utf-8"))
    # Original questions are smoke checks; authored paraphrases are reported separately.
    exact = [
        {"row": i, "query": rows[i]["question"], "kind": "exact"} for i in range(0, len(rows), 19)
    ]
    cases = exact + [dict(c, kind="paraphrase") for c in paraphrases]
    details = []
    for case in cases:
        gold = rows[case["row"]]
        match = backend.best_match(case["query"], gold["source"])
        answer, _ = bm25.search(case["query"], source_filter=gold["source"])
        details.append(
            {
                **case,
                "expected_question": gold["question"],
                "subject": gold["source"],
                "lexical_top1_correct": match[1]["question"] == gold["question"],
                "bm25_answered": bool(answer),
                "bm25_correct": bool(answer and answer == gold["answer"]),
                "lexical_score": round(match[0], 4),
            }
        )
    summary = {}
    for kind in ["exact", "paraphrase"]:
        selected = [d for d in details if d["kind"] == kind]
        summary[kind] = {
            "cases": len(selected),
            "lexical_top1_rate": sum(d["lexical_top1_correct"] for d in selected) / len(selected),
            "bm25_correct_answer_rate": sum(d["bm25_correct"] for d in selected) / len(selected),
            "bm25_wrong_answers": sum(
                d["bm25_answered"] and not d["bm25_correct"] for d in selected
            ),
            "bm25_abstained": sum(not d["bm25_answered"] for d in selected),
        }
    backend.close()
    return {
        "scope": "bundled_qa_retrieval_only",
        "records": len(rows),
        "summary": summary,
        "details": details,
        "limitations": "源自现有知识库的原题与人工编写改写，用于回归和错误分析；不是独立盲测，不包含大模型回答质量。CSV 行序改变后必须重新核对标注。",
    }


def retrieval_evaluation(base_url):
    import httpx

    cases = json.loads((ROOT / "evals/retrieval_cases.json").read_text(encoding="utf-8"))
    variants = [("dense", False), ("sparse", False), ("hybrid", False), ("hybrid", True)]
    details, grouped = [], defaultdict(list)
    with httpx.Client(base_url=base_url, timeout=120) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json().get("mode") != "full":
            raise RuntimeError("检索评测要求真实完整模式，演示模式不能替代")
        for case in cases:
            for mode, rerank in variants:
                variant = mode + ("+rerank" if rerank else "")
                start = time.perf_counter()
                response = client.post(
                    "/api/retrieve",
                    json={
                        "query": case["query"],
                        "source_filter": case["source_filter"],
                        "retrieval_mode": mode,
                        "rerank": rerank,
                        "top_k": 3,
                    },
                )
                response.raise_for_status()
                documents = response.json()["documents"]
                ranks = [
                    i
                    for i, d in enumerate(documents, 1)
                    if all(
                        normalize(term) in normalize(d["text"]) for term in case["evidence_terms"]
                    )
                ]
                item = {
                    "case": case["id"],
                    "query": case["query"],
                    "variant": variant,
                    "hit_at_1": bool(ranks and ranks[0] == 1),
                    "hit_at_3": bool(ranks),
                    "reciprocal_rank_at_3": 1 / ranks[0] if ranks else 0,
                    "seconds": time.perf_counter() - start,
                    "returned_parents": [d["parent_id"] for d in documents],
                    "matched_parents": [documents[i - 1]["parent_id"] for i in ranks],
                }
                details.append(item)
                grouped[variant].append(item)
                print(f"{case['id']} {variant}: hit@3={bool(ranks)}", flush=True)
    summary = {
        variant: {
            "cases": len(items),
            "evidence_hit_at_1": statistics.mean(i["hit_at_1"] for i in items),
            "evidence_hit_at_3": statistics.mean(i["hit_at_3"] for i in items),
            "evidence_mrr_at_3": statistics.mean(i["reciprocal_rank_at_3"] for i in items),
            "median_seconds": statistics.median(i["seconds"] for i in items),
        }
        for variant, items in grouped.items()
    }
    return {
        "scope": "real_milvus_retrieval_ablation",
        "base_url": base_url,
        "summary": summary,
        "details": details,
        "limitations": "16 道人工编写的小样本；以同一父块中同时含标注证据词判断相关性，含 NFKC 规范化，不是穷尽相关性标注，不计算 Recall。结果不代表大模型正确率。串行运行、包含冷热缓存差异，耗时不是并发压测。",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["qa", "retrieval"], default="qa")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output) if args.output else ROOT / "reports" / f"{args.mode}_evaluation.json"
    result = qa_evaluation() if args.mode == "qa" else retrieval_evaluation(args.base_url)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    label_file = ROOT / (
        "evals/qa_paraphrases.json" if args.mode == "qa" else "evals/retrieval_cases.json"
    )
    result["labels_sha256"] = hashlib.sha256(label_file.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
