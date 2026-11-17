"""Verify the new pipeline in an isolated process or against an approved running service."""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from base.version import VERSION

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def verify(client, retrieval_calls=None):
    health = client.get("/health")
    health.raise_for_status()
    assert health.json()["version"] == VERSION and health.json()["mode"] == "full"
    client.get("/ready").raise_for_status()
    checks = [{"check": "full_mode_version_and_readiness", "passed": True}]
    session = "round2-" + uuid.uuid4().hex
    empty_session = "round2-empty-" + uuid.uuid4().hex
    try:
        response = client.post(
            "/api/query",
            json={
                "query": "如何在 Ubuntu 系统上安装 Selenium？",
                "source_filter": "Python学科",
                "session_id": session,
            },
        )
        response.raise_for_status()
        assert not response.json()["is_streaming"] and "标准问答库" in response.json()["answer"]
        checks.append(
            {"check": "rephrased_standard_qa", "passed": True, "answer": response.json()["answer"]}
        )
        # Use a separate session so the previous subject does not contaminate the follow-up.
        client.delete(f"/api/history/{session}").raise_for_status()
        first = client.post(
            "/api/query",
            json={
                "query": "BERT 属于哪一种语言模型？",
                "source_filter": "ai",
                "session_id": session,
                "stream": False,
            },
        )
        first.raise_for_status()
        assert "本次检索资料" in first.json()["answer"] and "BERT" in first.json()["answer"]
        checks.append(
            {
                "check": "real_first_turn_with_evidence",
                "passed": True,
                "answer": first.json()["answer"],
            }
        )
        if retrieval_calls is not None:
            retrieval_calls.clear()
        start = time.perf_counter()
        follow = client.post(
            "/api/query",
            json={
                "query": "它采用什么预训练方式？",
                "source_filter": "ai",
                "session_id": session,
                "stream": False,
            },
        )
        follow.raise_for_status()
        assert "本次检索资料" in follow.json()["answer"]
        item = {
            "check": "real_contextual_follow_up",
            "passed": True,
            "seconds": time.perf_counter() - start,
            "answer": follow.json()["answer"],
        }
        if retrieval_calls is not None:
            assert retrieval_calls and any(
                "bert" in call["query"].casefold() for call in retrieval_calls
            )
            assert all(call["source_filter"] == "ai" for call in retrieval_calls)
            item["actual_retrieval_queries"] = list(retrieval_calls)
        checks.append(item)
        missing = client.post(
            "/api/query",
            json={
                "query": "它有哪些特点？",
                "source_filter": "ai",
                "session_id": empty_session,
                "stream": False,
            },
        )
        missing.raise_for_status()
        assert "请补充" in missing.json()["answer"]
        checks.append(
            {
                "check": "missing_context_clarifies",
                "passed": True,
                "answer": missing.json()["answer"],
            }
        )
        switched = client.post(
            "/api/query",
            json={
                "query": "它有哪些特点？",
                "source_filter": "java",
                "session_id": session,
                "stream": False,
            },
        )
        switched.raise_for_status()
        assert "已切换学科" in switched.json()["answer"]
        checks.append(
            {
                "check": "changed_subject_clarifies",
                "passed": True,
                "answer": switched.json()["answer"],
            }
        )
    finally:
        client.delete(f"/api/history/{session}").raise_for_status()
        client.delete(f"/api/history/{empty_session}").raise_for_status()
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    args = parser.parse_args()
    if args.isolated:
        # Import the API without initializing a second full backend at module scope.
        os.environ["EDURAG_MODE"] = "demo"
        os.environ["EDURAG_SESSION_DB"] = ":memory:"
        from fastapi.testclient import TestClient

        from app import app, create_app
        from new_main import IntegratedQASystem

        backend = IntegratedQASystem()
        original = backend.vector_store.hybrid_search_with_rerank
        calls = []

        def traced(query, *arguments, **kwargs):
            calls.append({"query": query, "source_filter": kwargs.get("source_filter")})
            return original(query, *arguments, **kwargs)

        backend.vector_store.hybrid_search_with_rerank = traced
        try:
            with TestClient(create_app(backend, mode="full")) as client:
                checks = verify(client, calls)
        finally:
            backend.close()
            app.state.backend.close()
    else:
        import httpx

        with httpx.Client(base_url=args.base_url, timeout=120) as client:
            checks = verify(client)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transport": "in_process_real_dependencies" if args.isolated else "http",
        "base_url": None if args.isolated else args.base_url,
        "mode": "full",
        "checks": checks,
        "limitations": "少量真实依赖功能验收，不代表大规模多轮正确率；独立验证进程不切换或中断用户正在使用的服务。",
    }
    path = ROOT / (
        "reports/round2_isolated_acceptance.json"
        if args.isolated
        else "reports/round2_live_acceptance.json"
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed_checks": len(checks), "report": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
