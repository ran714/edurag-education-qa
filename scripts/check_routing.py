"""Real-model routing checks; retain outcomes, not raw course excerpts."""

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base.version import VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing_acceptance.json")
    args = parser.parse_args()
    checks = []
    sessions = []
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["version"] == VERSION and health.json()["mode"] == "full"
        client.get("/ready").raise_for_status()
        checks.append({"check": "updated_full_service_ready", "passed": True})
        try:
            for query in ["什么是人工智能？", "机器学习有哪些算法？", "如何学习Python？"]:
                session = "routing-" + uuid.uuid4().hex
                sessions.append(session)
                response = client.post(
                    "/api/query", json={"query": query, "session_id": session, "stream": False}
                )
                response.raise_for_status()
                answer = response.json()["answer"]
                assert "通用知识说明" in answer, answer
                assert "12345678" not in answer and "信息不足，无法回答" not in answer, answer
                assert "本次检索资料" not in answer, answer
                checks.append({"check": "general_question", "query": query, "passed": True})

            session = "routing-" + uuid.uuid4().hex
            sessions.append(session)
            ws_url = args.base_url.replace("http://", "ws://").replace("https://", "wss://")
            with connect(ws_url + "/api/stream", open_timeout=10, close_timeout=5) as socket:
                socket.send(json.dumps({"query": "机器学习有哪些算法？", "session_id": session}))
                tokens = []
                while True:
                    event = json.loads(socket.recv(timeout=120))
                    if event["type"] == "token":
                        tokens.append(event["token"])
                    elif event["type"] == "end":
                        assert event["full_answer"] == "".join(tokens)
                        assert "通用知识说明" in event["full_answer"]
                        break
                    elif event["type"] == "error":
                        raise RuntimeError(event["error"])
            assert len(client.get(f"/api/history/{session}").json()["history"]) == 1
            checks.append({"check": "general_websocket_and_history", "passed": True})

            for query, scope, marker in [
                ("请根据课程资料说明课程包含哪些学习阶段", None, "本次检索资料"),
                ("请提供本课程今年的精确学费和退款承诺", "java", "没有找到"),
            ]:
                session = "routing-" + uuid.uuid4().hex
                sessions.append(session)
                response = client.post(
                    "/api/query",
                    json={
                        "query": query,
                        "source_filter": scope,
                        "session_id": session,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                answer = response.json()["answer"]
                assert marker in answer and "12345678" not in answer, answer
                checks.append(
                    {"check": "grounded_question", "query": query, "scope": scope, "passed": True}
                )
        finally:
            for session in sessions:
                client.delete(f"/api/history/{session}").raise_for_status()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
        "checks": checks,
        "code_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["rag_qa/core/new_rag_system.py", "rag_qa/core/prompts.py"]
        },
        "scope": "small_functional_check_with_real_dependencies_and_model_not_accuracy_evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({"passed_checks": len(checks), "report": str(args.output)}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
