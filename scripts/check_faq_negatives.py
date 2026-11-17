"""Check existing FAQ hard negatives with real model selection and CSV data."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def selection_failed(selection):
    return selection.trace.get("reason") in {
        "selection_unavailable",
        "recall_unavailable",
        "invalid_selection_schema",
        "unknown_candidate_id",
    }


def main():
    from openai import OpenAI

    from base.config import Config
    from evals.qa_fixtures import MemoryCache, QuestionDatabase
    from mysql_qa.retrieval.bm25_search import BM25Search
    from mysql_qa.retrieval.semantic_fallback import SemanticFAQFallback
    from rag_qa.core.query_rewriter import QueryRewriter
    from scripts.evaluate_improvement import RuntimeTrace, code_hashes, digest, file_digest

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/faq_negative_improvement.json"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.limit and args.output == ROOT / "reports/faq_negative_improvement.json":
        args.output = ROOT / "reports/faq_negative_improvement.smoke.json"
    if args.output.exists():
        parser.error("output exists; choose another path to preserve previous evidence")
    label_path = ROOT / "evals/qa_challenge.json"
    cases = [
        case
        for case in json.loads(label_path.read_text(encoding="utf-8"))["cases"]
        if case["kind"] == "negative"
    ]
    if args.limit:
        cases = cases[: args.limit]
    settings = Config()
    database = QuestionDatabase()
    search = BM25Search(MemoryCache(), database)
    trace = RuntimeTrace()
    client = OpenAI(
        api_key=settings.DASHSCOPE_API_KEY, base_url=settings.DASHSCOPE_BASE_URL, max_retries=1
    )
    trace.wrap_model(client, "negative_faq_selection")

    def llm(prompt):
        completion = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            timeout=30,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    selector = SemanticFAQFallback(search, llm)
    rewriter = QueryRewriter(llm)
    details = []
    hashes = code_hashes()
    try:
        for case in cases:
            trace.reset()
            started = time.perf_counter()
            item = dict(case)
            try:
                resolved = rewriter.resolve(case["query"], [], case["subject"])
                selection = None
                if resolved.clarification:
                    answer, route = None, "clarification"
                else:
                    answer, _ = search.search(resolved.query, source_filter=case["subject"])
                    route = "direct" if answer else "abstain"
                    if not answer:
                        selection = selector.select(resolved.query, case["subject"])
                        if selection_failed(selection):
                            raise RuntimeError("selection_failed")
                        answer = selection.answer
                        route = "semantic" if answer else "abstain"
                item.update(
                    status="ok",
                    returned_answer=answer,
                    route=route,
                    rejected=not bool(answer),
                    selection_trace=selection.trace if selection else None,
                )
            except Exception as exc:
                item.update(status="failed", error_type=type(exc).__name__, rejected=None)
            item.update(actual_trace=trace.events, seconds=time.perf_counter() - started)
            details.append(item)
            print(f"negative {case['id']}: {item['status']} {item.get('route', '')}", flush=True)
    finally:
        client.close()
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transport": "local_faq_components_csv_mirror_real_llm",
        "model": settings.LLM_MODEL,
        "generator_temperature": "provider_default_unspecified",
        "labels_sha256": file_digest(label_path),
        "runner_sha256": file_digest(Path(__file__)),
        "csv_rows_sha256": digest(database.rows),
        "code_sha256": hashes,
        "cases": len(details),
        "rejected": sum(item.get("rejected") is True for item in details),
        "wrong_existing_answer_returned": sum(item.get("rejected") is False for item in details),
        "failed": sum(item["status"] != "ok" for item in details),
        "details": details,
        "limitations": "既有20题难负例已用于开发，非独立盲测；仅检查不应套用已有标准答案的行为，不代表模型不能回答问题。使用CSV镜像、内存缓存和真实模型，没有验证HTTP/MySQL/Redis/Milvus全链路。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ["cases", "rejected", "wrong_existing_answer_returned", "failed"]
            }
        )
    )


if __name__ == "__main__":
    main()
