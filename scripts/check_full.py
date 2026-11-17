"""HTTP/WebSocket acceptance against an already running full backend."""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import httpx
    from websockets.sync.client import connect

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    args = parser.parse_args()
    checks = []
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        deadline = time.monotonic() + 30
        while True:
            try:
                client.get("/health").raise_for_status()
                break
            except httpx.ConnectError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)
        for endpoint in ["/health", "/ready"]:
            response = client.get(endpoint)
            response.raise_for_status()
            assert response.json()["mode"] == "full", "要求完整模式"
            checks.append({"check": endpoint, "passed": True})
        session = "acceptance-" + uuid.uuid4().hex
        response = client.post(
            "/api/query",
            json={
                "query": "用上下文管理器实现函数运行时间的计算?",
                "source_filter": "Python学科",
                "session_id": session,
            },
        )
        response.raise_for_status()
        assert not response.json()["is_streaming"]
        assert len(client.get(f"/api/history/{session}").json()["history"]) == 1
        checks.append(
            {
                "check": "standard_qa_and_history",
                "passed": True,
                "answer": response.json()["answer"],
            }
        )
        websocket_url = (
            args.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/api/stream"
        )
        with connect(websocket_url, open_timeout=10, close_timeout=5) as websocket:
            started = time.perf_counter()
            websocket.send(
                json.dumps(
                    {
                        "query": "你好，请根据课程资料说明人工智能开发课程包含哪些学习阶段",
                        "source_filter": "ai",
                        "session_id": session,
                    },
                    ensure_ascii=False,
                )
            )
            chunks, first_token = [], None
            while True:
                event = json.loads(websocket.recv(timeout=120))
                if event["type"] == "token":
                    if first_token is None:
                        first_token = time.perf_counter() - started
                    chunks.append(event["token"])
                elif event["type"] == "error":
                    raise RuntimeError(event["error"])
                elif event["type"] == "end":
                    answer = event["full_answer"]
                    assert answer == "".join(chunks)
                    assert "本次检索资料" in answer
                    assert ".md" in answer or ".pdf" in answer
                    assert len(client.get(f"/api/history/{session}").json()["history"]) == 2
                    checks.append(
                        {
                            "check": "real_rag_websocket_and_citations",
                            "passed": True,
                            "first_token_seconds": first_token,
                            "total_seconds": time.perf_counter() - started,
                            "answer": answer,
                        }
                    )
                    break
        refusal = client.post(
            "/api/query",
            json={
                "query": "请提供本课程今年的精确学费和退款承诺",
                "source_filter": "java",
                "stream": False,
            },
        )
        refusal.raise_for_status()
        assert "没有找到" in refusal.json()["answer"]
        checks.append(
            {"check": "empty_subject_refusal", "passed": True, "answer": refusal.json()["answer"]}
        )
        invalid = client.post("/api/query", json={"query": "  "})
        assert invalid.status_code == 422
        checks.append({"check": "invalid_input", "passed": True})
        for index in range(6):
            result = client.post("/api/query", json={"query": "你好", "session_id": session})
            result.raise_for_status()
        retained = client.get(f"/api/history/{session}").json()["history"]
        assert len(retained) == 5
        checks.append({"check": "mysql_history_retains_five_turns", "passed": True})
        client.delete(f"/api/history/{session}").raise_for_status()
        assert client.get(f"/api/history/{session}").json()["history"] == []
        checks.append({"check": "session_cleanup", "passed": True})
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "mode": "full",
        "checks": checks,
        "limitations": "单次功能验收，耗时只代表本机此次请求，非大规模正确率或并发性能。",
    }
    target = ROOT / "reports/full_acceptance.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed_checks": len(checks), "report": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
