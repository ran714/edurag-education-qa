"""Verify the public sample over REST/WebSocket without keys, Docker or models."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["EDURAG_MODE"] = "demo"
os.environ["EDURAG_SESSION_DB"] = ":memory:"
os.environ["EDURAG_FAQ_CSV"] = "examples/faq_sample.csv"

from fastapi.testclient import TestClient

from app import app


def main():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["mode"] == "demo"
        assert client.get("/ready").status_code == 200
        response = client.post(
            "/api/query",
            json={
                "query": "Python 列表和元组有什么区别？",
                "source_filter": "Python学科",
                "session_id": "public-example-check",
                "stream": False,
            },
        )
        response.raise_for_status()
        assert "列表支持修改" in response.json()["answer"]
        with client.websocket_connect("/api/stream") as socket:
            socket.send_json({"query": "RAG 是什么？", "session_id": "public-example-check"})
            tokens = []
            while True:
                event = socket.receive_json()
                if event["type"] == "token":
                    tokens.append(event["token"])
                elif event["type"] == "end":
                    assert event["full_answer"] == "".join(tokens)
                    assert "Retrieval-Augmented Generation" in event["full_answer"]
                    break
                elif event["type"] == "error":
                    raise RuntimeError(event["error"])
        assert len(client.get("/api/history/public-example-check").json()["history"]) == 2
        client.delete("/api/history/public-example-check").raise_for_status()
    print("Public demo: homepage, health, readiness, FAQ, WebSocket and history passed.")


if __name__ == "__main__":
    main()
