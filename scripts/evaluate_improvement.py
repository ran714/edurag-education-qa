"""Evaluate current code against the frozen generation run without restarting services.

The isolated runner loads one real backend and exercises its HTTP API in process.
It captures real model messages and retrieval calls, while RAGAS receives the same
citation-material view used by the original run. References never become context.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("EDURAG_CPU_THREADS", "2")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from scripts.evaluate_generation import (
    append_jsonl,
    audit_scores,
    clean_answer,
    disposition,
    latest_scores,
    read_jsonl,
    score,
)

METRICS = ("faithfulness", "factual_correctness", "expected_behavior")
CASE_FIELDS = (
    "id",
    "category",
    "query",
    "source_filter",
    "expected_behavior",
    "reference",
    "setup",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(case):
    return digest({key: case.get(key) for key in CASE_FIELDS})


def unique_rows(rows, label):
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate_ids_in_{label}")
    return result


def frozen_cases(baseline):
    cases = json.loads((baseline / "cases.json").read_text(encoding="utf-8"))
    responses = unique_rows(read_jsonl(baseline / "responses.jsonl"), "baseline_responses")
    unique_rows(cases, "baseline_cases")
    for case in cases:
        if case["id"] not in responses or signature(case) != signature(responses[case["id"]]):
            raise ValueError("baseline_case_and_response_disagree")
    return cases


def validate_same_cases(baseline_rows, current_rows):
    before = unique_rows(baseline_rows, "baseline_responses")
    after = unique_rows(current_rows, "current_responses")
    for key, row in after.items():
        if key not in before or signature(row) != signature(before[key]):
            raise ValueError(f"changed_case_or_reference:{key}")


def audited_invalid_ids(baseline, current_rows, audit_path):
    """Bind audit to its original evidence, then apply identical references by hash."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    original = ROOT / audit["source_report"]
    if file_digest(original) != audit["source_report_sha256"]:
        raise ValueError("reference_audit_source_hash_changed")
    original_rows = unique_rows(read_jsonl(original), "audit_source")
    baseline_rows = read_jsonl(baseline / "responses.jsonl")
    validate_same_cases(list(original_rows.values()), baseline_rows)
    validate_same_cases(baseline_rows, current_rows)
    invalid = set()
    reference_hashes = {}
    for case in audit["cases"]:
        key = case["id"]
        expected = case["evidence"]["reference_original"]
        if key not in original_rows or digest(expected) != digest(original_rows[key]["reference"]):
            raise ValueError(f"reference_audit_original_disagrees:{key}")
        invalid.add(key)
        reference_hashes[key] = digest(expected)
    return invalid, {
        "audit_sha256": file_digest(audit_path),
        "original_responses_sha256": file_digest(original),
        "invalid_case_ids": sorted(invalid),
        "reference_sha256": reference_hashes,
        "rule": "Same frozen case and reference hashes; raw evidence remains unchanged.",
    }


def context_view(raw, parents, standard_rows):
    """Reconstruct the original citation view, never using a gold answer as context."""
    body = clean_answer(raw)
    ids = list(dict.fromkeys(re.findall(r"（片段：([^）]+)）", raw)))
    contexts = [parents[key]["parent_content"] for key in ids if key in parents]
    source = "returned_citation_parent_ids"
    if "来源：课程标准问答库" in raw:
        contexts = list(
            dict.fromkeys(row["answer"] for row in standard_rows if row["answer"].strip() == body)
        )
        source = "actual_returned_standard_answer_matched_to_csv"
    return {
        "response": body,
        "retrieved_contexts": contexts,
        "returned_parent_ids": ids,
        "missing_parent_ids": [key for key in ids if key not in parents],
        "context_source": source,
    }


class RuntimeTrace:
    """Capture whitelisted API inputs rather than credentials or arbitrary clients."""

    def __init__(self):
        self.events = []

    def reset(self):
        self.events = []

    def wrap_model(self, client, stage):
        resource = client.chat.completions
        original = resource.create

        def create(*args, **kwargs):
            event = {
                "kind": "model",
                "stage": stage,
                "request": {
                    key: kwargs[key]
                    for key in (
                        "model",
                        "messages",
                        "temperature",
                        "stream",
                        "max_tokens",
                        "timeout",
                    )
                    if key in kwargs
                },
            }
            self.events.append(event)
            started = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                event.update(error_type=type(exc).__name__, seconds=time.perf_counter() - started)
                raise
            if not kwargs.get("stream"):
                event["response_text"] = [choice.message.content for choice in result.choices]
                event["seconds"] = time.perf_counter() - started
                return result

            def stream():
                pieces = []
                try:
                    for chunk in result:
                        for choice in chunk.choices:
                            if choice.delta.content:
                                pieces.append(choice.delta.content)
                        yield chunk
                except Exception as exc:
                    event["error_type"] = type(exc).__name__
                    raise
                finally:
                    event.update(
                        response_text="".join(pieces), seconds=time.perf_counter() - started
                    )
                    if hasattr(result, "close"):
                        result.close()

            return stream()

        resource.create = create

    def wrap_call(self, owner, name, kind, encode=lambda value: value):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            event = {"kind": kind, "method": name, "args": list(args), "kwargs": kwargs}
            self.events.append(event)
            started = time.perf_counter()
            try:
                result = original(*args, **kwargs)
                event["result"] = encode(result)
                return result
            except Exception as exc:
                event["error_type"] = type(exc).__name__
                raise
            finally:
                event["seconds"] = time.perf_counter() - started

        setattr(owner, name, wrapped)

    def attach(self, backend):
        self.wrap_model(backend.client, "pipeline")
        selector = backend.rag_system.strategy_selector
        if hasattr(selector, "client"):
            self.wrap_model(selector.client, "retrieval_strategy")
        self.wrap_call(
            backend.vector_store,
            "hybrid_search_with_rerank",
            "document_retrieval",
            lambda docs: [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs],
        )
        self.wrap_call(backend.bm25_search, "search", "faq_search")
        self.wrap_call(
            backend.bm25_search,
            "rank",
            "faq_candidates",
            lambda values: [
                asdict(value) if is_dataclass(value) else str(value) for value in values
            ],
        )
        self.wrap_call(
            backend.rag_system,
            "resolve_query",
            "query_resolution",
            lambda value: asdict(value) if is_dataclass(value) else str(value),
        )


@contextmanager
def isolated_client():
    """One real backend plus an in-memory demo import; no network server is created."""
    os.environ["EDURAG_MODE"] = "demo"
    os.environ["EDURAG_SESSION_DB"] = ":memory:"
    import torch
    from fastapi.testclient import TestClient

    from app import app, create_app
    from new_main import IntegratedQASystem

    torch.set_num_threads(2)
    backend = IntegratedQASystem()
    trace = RuntimeTrace()
    trace.attach(backend)
    try:
        with TestClient(create_app(backend, mode="full")) as client:
            yield client, backend, trace
    finally:
        backend.close()
        selector = backend.rag_system.strategy_selector
        if hasattr(selector, "client"):
            selector.client.close()
        app.state.backend.close()


def code_hashes():
    paths = [ROOT / "app.py", ROOT / "new_main.py", Path(__file__)]
    paths.extend((ROOT / "rag_qa/core").glob("*.py"))
    paths.extend((ROOT / "mysql_qa/retrieval").glob("*.py"))
    return {str(path.relative_to(ROOT)): file_digest(path) for path in sorted(set(paths))}


def initialize_run(directory, baseline, cases, backend, parents):
    settings = backend.config
    previous = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transport": "in_process_http_real_dependencies",
        "case_count": len(cases),
        "cases_sha256": digest(cases),
        "baseline_responses_sha256": file_digest(baseline / "responses.jsonl"),
        "baseline_cases_sha256": file_digest(baseline / "cases.json"),
        "code_sha256": code_hashes(),
        "corpus_parents_sha256": digest(parents),
        "collection": settings.MILVUS_COLLECTION_NAME,
        "retrieval_k": settings.RETRIEVAL_K,
        "candidate_m": settings.CANDIDATE_M,
        "rerank_enabled": settings.RERANK_ENABLED,
        "generator_model": settings.LLM_MODEL,
        "judge_model": settings.LLM_MODEL,
        "generator_temperature": "provider_default_unspecified",
        "judge_temperature": 1e-8,
        "configured_chat_temperature": 0,
        "baseline_generator_model": previous.get("generator_model"),
        "baseline_judge_model": previous.get("judge_model"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("ragas", "langchain-openai", "openai")
        },
        "context_view": "returned_citation_parent_ids_or_actual_returned_standard_answer",
        "evaluation_scope": "full_pipeline",
    }
    path = directory / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for key in manifest:
            if key != "created_at" and old.get(key) != manifest[key]:
                raise ValueError(f"run_input_changed_use_new_directory:{key}")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (directory / "cases.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def collect(directory, baseline, cases):
    from evals.qa_fixtures import QuestionDatabase

    database = QuestionDatabase()
    existing_rows = read_jsonl(directory / "responses.jsonl")
    validate_same_cases(cases, existing_rows)
    existing = {row["id"] for row in existing_rows}
    with isolated_client() as (client, backend, trace):
        health = client.get("/health")
        health.raise_for_status()
        if health.json()["mode"] != "full":
            raise ValueError("requires_real_full_backend")
        client.get("/ready").raise_for_status()
        if (
            int(
                backend.vector_store.client.get_collection_stats(
                    backend.vector_store.collection_name
                )["row_count"]
            )
            > 1000
        ):
            raise ValueError("evaluation_context_snapshot_requires_small_corpus")
        records = backend.vector_store.client.query(
            collection_name=backend.vector_store.collection_name,
            filter="",
            output_fields=["parent_id", "parent_content", "source", "file_name"],
            limit=1000,
        )
        parents = {record["parent_id"]: record for record in records}
        initialize_run(directory, baseline, cases, backend, parents)
        for case in cases:
            if case["id"] in existing:
                continue
            session = "quality-improvement-" + uuid.uuid4().hex
            row = dict(
                case,
                captured_at=datetime.now(timezone.utc).isoformat(),
                service=health.json(),
                setup_outputs=[],
            )
            started = time.perf_counter()
            trace.reset()
            try:
                for setup in case.get("setup", []):
                    reply = client.post(
                        "/api/query", json={**setup, "session_id": session, "stream": False}
                    )
                    reply.raise_for_status()
                    row["setup_outputs"].append(
                        {"query": setup["query"], "answer": reply.json()["answer"]}
                    )
                row["setup_trace"] = trace.events
                trace.reset()
                reply = client.post(
                    "/api/query",
                    json={
                        "query": case["query"],
                        "source_filter": case.get("source_filter"),
                        "session_id": session,
                        "stream": False,
                    },
                )
                reply.raise_for_status()
                raw = reply.json()["answer"]
                view = context_view(raw, parents, database.rows)
                row.update(
                    status="ok",
                    raw_response=raw,
                    disposition_heuristic=disposition(view["response"]),
                    **view,
                )
            except Exception as exc:
                row.update(status="request_failed", error_type=type(exc).__name__)
            finally:
                row["actual_trace"] = trace.events
                row["seconds"] = time.perf_counter() - started
                try:
                    client.delete(f"/api/history/{session}").raise_for_status()
                except Exception as exc:
                    row["session_cleanup_error"] = type(exc).__name__
            append_jsonl(directory / "responses.jsonl", row)
            print(f"capture {case['id']}: {row['status']}", flush=True)


def collect_faq(directory, baseline, cases, scope_cases):
    """Run production FAQ components over a CSV mirror; this is not a live API test."""
    from openai import OpenAI

    from base.config import Config
    from evals.qa_fixtures import MemoryCache, QuestionDatabase
    from mysql_qa.retrieval.bm25_search import BM25Search
    from mysql_qa.retrieval.semantic_fallback import SemanticFAQFallback
    from rag_qa.core.query_rewriter import QueryRewriter

    settings = Config()
    database = QuestionDatabase()
    search = BM25Search(MemoryCache(), database)
    trace = RuntimeTrace()
    client = OpenAI(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        max_retries=1,
    )
    trace.wrap_model(client, "faq_selector_or_rewrite")

    def llm(prompt):
        completion = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": prompt},
            ],
            timeout=30,
            stream=True,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    fallback = SemanticFAQFallback(search, llm)
    rewriter = QueryRewriter(llm)

    def encode(value):
        return asdict(value) if is_dataclass(value) else str(value)

    trace.wrap_call(search, "search", "faq_search")
    for method in ("rank", "recall_candidates"):
        trace.wrap_call(
            search, method, "faq_candidates", lambda values: [encode(v) for v in values]
        )
    trace.wrap_call(fallback, "select", "faq_selection", encode)
    trace.wrap_call(rewriter, "resolve", "query_resolution", encode)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transport": "local_faq_components_csv_mirror_real_llm",
        "evaluation_scope": "faq_only",
        "scope_case_ids": [case["id"] for case in scope_cases],
        "case_count": len(cases),
        "cases_sha256": digest(cases),
        "baseline_responses_sha256": file_digest(baseline / "responses.jsonl"),
        "baseline_cases_sha256": file_digest(baseline / "cases.json"),
        "code_sha256": code_hashes(),
        "csv_rows_sha256": digest(database.rows),
        "csv_rows": len(database.rows),
        "generator_model": settings.LLM_MODEL,
        "selector_model": settings.LLM_MODEL,
        "judge_model": settings.LLM_MODEL,
        "generator_temperature": "provider_default_unspecified",
        "judge_temperature": 1e-8,
        "configured_chat_temperature": 0,
        "answer_text_origin": "Existing CSV answer; model can select an ID or abstain.",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("ragas", "langchain-openai", "openai")
        },
        "context_view": "actual_returned_standard_answer_matched_to_csv",
        "limitations": [
            "不调用 HTTP/MySQL/Redis/Milvus；CSV 为随附标准问答数据镜像，缓存为进程内缓存。",
            "调用真实生产 BM25Search、SemanticFAQFallback 和 QueryRewriter，选择器使用真实模型。",
            "未命中时返回既有无资料拒答；此组件实验不验证文档 RAG 或 API 集成。",
        ],
    }
    try:
        path = directory / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            for key in manifest:
                if key != "created_at" and old.get(key) != manifest[key]:
                    raise ValueError(f"run_input_changed_use_new_directory:{key}")
        else:
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            (directory / "cases.json").write_text(
                json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        existing_rows = read_jsonl(directory / "responses.jsonl")
        validate_same_cases(cases, existing_rows)
        existing = {row["id"] for row in existing_rows}
        for case in cases:
            if case["id"] in existing:
                continue
            trace.reset()
            started = time.perf_counter()
            row = dict(
                case,
                captured_at=datetime.now(timezone.utc).isoformat(),
                service={"mode": "faq_components_csv_real_llm"},
                setup_outputs=[],
            )
            try:
                resolved = rewriter.resolve(case["query"], [], case.get("source_filter"))
                scope = f"**检索范围：{case.get('source_filter') or '全部学科'}**\n\n"
                if resolved.clarification:
                    raw = scope + resolved.clarification
                    row["route"] = "clarification"
                else:
                    answer, _ = search.search(
                        resolved.query, threshold=0.85, source_filter=case.get("source_filter")
                    )
                    row["route"] = "direct_standard_answer"
                    if not answer:
                        selected = fallback.select(resolved.query, case.get("source_filter"))
                        answer = selected.answer
                        row["route"] = (
                            "semantic_selected_answer" if answer else "no_document_fallback"
                        )
                    if answer:
                        raw = scope + answer + "\n\n_来源：课程标准问答库（现有答案复用）。_"
                    else:
                        raw = (
                            scope
                            + "当前知识库中没有找到可支持回答的资料，请补充课程资料或调整问题。"
                        )
                view = context_view(raw, {}, database.rows)
                row.update(
                    status="ok",
                    raw_response=raw,
                    disposition_heuristic=disposition(view["response"]),
                    **view,
                )
            except Exception as exc:
                row.update(status="request_failed", error_type=type(exc).__name__)
            row["actual_trace"] = trace.events
            row["seconds"] = time.perf_counter() - started
            append_jsonl(directory / "responses.jsonl", row)
            print(f"capture {case['id']}: {row['status']} {row.get('route', '')}", flush=True)
    finally:
        client.close()


def metric_summary(rows, ids):
    by_id = {row["id"]: row for row in rows}
    statuses = Counter(by_id[key]["status"] if key in by_id else "missing" for key in ids)
    valid = [by_id[key]["score"] for key in ids if key in by_id and by_id[key]["status"] == "ok"]
    return {
        "total_cases": len(ids),
        "valid_cases": len(valid),
        "mean_score": statistics.mean(valid) if valid else None,
        "status_counts": dict(statuses),
    }


def compare_metric(before, after, ids):
    old = {row["id"]: row for row in before}
    new = {row["id"]: row for row in after}
    old_valid = {key for key in ids if key in old and old[key]["status"] == "ok"}
    new_valid = {key for key in ids if key in new and new[key]["status"] == "ok"}
    paired = sorted(old_valid & new_valid)
    old_mean = statistics.mean(old[key]["score"] for key in paired) if paired else None
    new_mean = statistics.mean(new[key]["score"] for key in paired) if paired else None
    return {
        "baseline": metric_summary(before, ids),
        "current": metric_summary(after, ids),
        "paired_case_ids": paired,
        "paired_cases": len(paired),
        "paired_baseline_mean": old_mean,
        "paired_current_mean": new_mean,
        "paired_mean_difference": new_mean - old_mean if paired else None,
        "newly_valid_ids": sorted(new_valid - old_valid),
        "lost_valid_ids": sorted(old_valid - new_valid),
        "unpaired_case_ids": sorted(set(ids) - set(paired)),
    }


def render_report(result):
    def number(value):
        return "—" if value is None else f"{value:.3f}"

    lines = [
        "# 固定生成评测集优化对比",
        "",
        f"本轮采集 {result['captured_cases']}/{result['baseline_cases']} 个既有场景，"
        f"评分记录 {result['unique_metric_results']}/{result['expected_metric_results']} 项。",
        "",
        "该题组已用于开发与失败分析，结果属于回归集改进，不是独立盲测。"
        "请求成功不等于回答正确，模型评分不等于人工准确率。",
        "",
        "| 指标 | 旧有效数 | 新有效数 | 配对数 | 配对旧均分 | 配对新均分 | 绝对差值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in METRICS:
        row = result["comparison"][f"all/{name}"]
        lines.append(
            f"| {name} | {row['baseline']['valid_cases']} | {row['current']['valid_cases']} | "
            f"{row['paired_cases']} | {number(row['paired_baseline_mean'])} | "
            f"{number(row['paired_current_mean'])} | {number(row['paired_mean_difference'])} |"
        )
    lines.extend(["", "## 分组与异常", ""])
    for key, row in result["comparison"].items():
        if key.startswith("all/"):
            continue
        lines.append(
            f"- {key}：旧 {row['baseline']['valid_cases']}/{row['baseline']['total_cases']}，"
            f"新 {row['current']['valid_cases']}/{row['current']['total_cases']}；"
            f"配对 {row['paired_cases']}，差值 {number(row['paired_mean_difference'])}。"
        )
    lines.extend(["", "## 口径与限制", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend(
        [
            "",
            "完整配对 ID、新增/失去有效评分、缺失与异常计数见 [report.json](report.json)。",
            "原始逐题回答、实际模型消息、FAQ 候选与检索调用见 [responses.jsonl](responses.jsonl)。",
            "原评分尝试保留在 scores.jsonl，统一参考审计后的结果另存 scores_audited.json。",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(directory, baseline, audit_path=ROOT / "evals/generation_reference_audit.json"):
    before_responses = read_jsonl(baseline / "responses.jsonl")
    responses = read_jsonl(directory / "responses.jsonl")
    validate_same_cases(before_responses, responses)
    invalid, audit = audited_invalid_ids(baseline, responses, audit_path)
    before_scores = audit_scores(latest_scores(read_jsonl(baseline / "scores.jsonl")), invalid)
    scores = audit_scores(latest_scores(read_jsonl(directory / "scores.jsonl")), invalid)
    captured_ids = {row["id"] for row in responses}
    if any(row["id"] not in captured_ids or row["metric"] not in METRICS for row in scores):
        raise ValueError("scores_must_match_captured_responses_and_known_metrics")
    (directory / "scores_audited.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    scope_ids = manifest.get("scope_case_ids")
    if scope_ids is not None:
        before_responses = [row for row in before_responses if row["id"] in set(scope_ids)]
    ids = {row["id"] for row in before_responses}
    comparisons = {}
    groups = {"all": ids}
    for category in sorted({row["category"] for row in before_responses}):
        groups[category] = {row["id"] for row in before_responses if row["category"] == category}
    for category, group in groups.items():
        for metric in METRICS:
            comparisons[f"{category}/{metric}"] = compare_metric(
                [row for row in before_scores if row["metric"] == metric],
                [row for row in scores if row["metric"] == metric],
                group,
            )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": manifest,
        "baseline_cases": len(before_responses),
        "captured_cases": len(responses),
        "capture_status_counts": dict(Counter(row["status"] for row in responses)),
        "disposition_counts_heuristic": dict(
            Counter(row.get("disposition_heuristic", "request_failed") for row in responses)
        ),
        "unique_metric_results": len(scores),
        "expected_metric_results": len(before_responses) * len(METRICS),
        "reference_audit": audit,
        "comparison": comparisons,
        "limitations": [
            "固定题组已用于开发，不是外部独立盲测，不能据此宣称泛化准确率。",
            "按相同题目、参考与评分规则比较。各指标的配对均分只用两轮均有效的相同 ID；"
            "新增覆盖、失去有效评分、不适用、未定义、失败、未采集/未评分均单列，不补零。",
            "Faithfulness 仍采用与旧版一致的引用资料视图，而非完整生成输入。"
            "新轮次额外保存实际模型消息和检索轨迹，旧轮次没有相同追踪，不能补造。",
            "生成与评判使用同型号模型，存在同源偏差；单轮配对差值未证明统计显著性。",
            "faq-15/faq-25 的占位标识与纯链接参考按既定审计标无效；"
            "两版均保留原记录。Faithfulness 原值保留并标记非信息性上下文，不能解释为有用答案。",
            "改进可能同时影响召回、生成和覆盖，本轮组合对比不能归因于某个单独组件。",
        ]
        + manifest.get("limitations", []),
    }
    (directory / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "report.md").write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {"captured": len(responses), "metric_results": len(scores), "report": str(directory)},
            ensure_ascii=False,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "score", "report", "all"), default="collect")
    parser.add_argument("--isolated-real", action="store_true")
    parser.add_argument("--faq-only", action="store_true")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "reports/generation_quality")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--metric-timeout", type=int, default=180)
    args = parser.parse_args()
    default_output = ROOT / (
        "reports/faq_improvement" if args.faq_only else "reports/quality_improvement"
    )
    directory, baseline = (args.output_dir or default_output).resolve(), args.baseline_dir.resolve()
    if directory == baseline or baseline in directory.parents:
        raise ValueError("use_separate_directory_preserve_baseline")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正整数")
    directory.mkdir(parents=True, exist_ok=True)
    if args.mode in {"collect", "all"}:
        if args.isolated_real and args.faq_only:
            parser.error("--faq-only 与 --isolated-real 不能同时使用")
        if not args.isolated_real and not args.faq_only:
            parser.error("采集当前代码须显式提供 --isolated-real；不会启动或停止现有服务")
        cases = frozen_cases(baseline)
        if args.faq_only:
            cases = [case for case in cases if case["category"] == "faq_paraphrase"]
            collect_faq(directory, baseline, cases[: args.limit] if args.limit else cases, cases)
        else:
            collect(directory, baseline, cases[: args.limit] if args.limit else cases)
    if args.mode in {"score", "all"}:
        if not (directory / "responses.jsonl").exists():
            raise ValueError("collect_responses_before_scoring")
        asyncio.run(score(directory, args.workers, args.retry_failed, args.metric_timeout))
    summarize(directory, baseline)


if __name__ == "__main__":
    main()
