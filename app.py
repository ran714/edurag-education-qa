"""EduRAG HTTP and WebSocket application.

Synchronous model/database work is executed in worker threads. A per-backend
lock protects the current single MySQL connection; this is a local demo service,
not a claim of production concurrency.
"""

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool

from base.version import VERSION
from rag_qa.core.query_rewriter import needs_context

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
logger = logging.getLogger("edurag.api")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    source_filter: str | None = Field(default=None, max_length=50)
    session_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    stream: bool = True

    @field_validator("query")
    @classmethod
    def clean_query(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空")
        return value


class RetrievalRequest(QueryRequest):
    retrieval_mode: str = Field(default="hybrid", pattern=r"^(dense|sparse|hybrid)$")
    rerank: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


def check_greeting(query):
    text = re.sub(r"[\s，。！？,.!?、~～]+", "", query).lower()
    greetings = {
        "你好",
        "您好",
        "hi",
        "hello",
        "在吗",
        "在不在",
        "有人吗",
        "你是谁",
        "您是谁",
        "你叫什么",
        "你的名字",
        "whoareyou",
    }
    if text in greetings:
        return "你好！我是 EduRAG 学习助手，可以查询课程资料和解答编程学习问题。"
    return None


def load_backend(mode):
    if mode not in {"demo", "full", "auto"}:
        raise ValueError("EDURAG_MODE 只能是 demo、full 或 auto")
    if mode != "demo":
        try:
            from new_main import IntegratedQASystem

            return IntegratedQASystem(), "full", None
        except Exception as exc:
            if mode == "full":
                raise
            logger.warning("完整后端初始化失败，启用演示模式：%s", type(exc).__name__)
            fallback = type(exc).__name__
    else:
        fallback = None
    from demo_backend import DemoIntegratedQASystem

    return DemoIntegratedQASystem(), "demo", fallback


def create_app(backend=None, mode=None):
    owns_backend = backend is None
    requested = mode or os.getenv("EDURAG_MODE", "auto")
    if backend is None:
        backend, actual_mode, fallback = load_backend(requested)
    else:
        actual_mode, fallback = requested, None

    @asynccontextmanager
    async def lifespan(application):
        yield
        if owns_backend and hasattr(backend, "close"):
            await run_in_threadpool(backend.close)

    application = FastAPI(title="EduRAG 教育问答系统", version=VERSION, lifespan=lifespan)
    application.state.backend = backend
    application.state.mode = actual_mode
    application.state.lock = Lock()
    application.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

    def locked(function, *args, **kwargs):
        with application.state.lock:
            return function(*args, **kwargs)

    def scope(value):
        value = (value or "").strip() or None
        aliases = {"python": "Python学科", "python学科": "Python学科", "java学科": "JAVA学科"}
        value = aliases.get(value.lower(), value) if value else None
        if value is not None and value not in backend.config.VALID_SOURCES:
            raise ValueError("不支持的学科，请从学科列表选择")
        return value

    def greeting_answer(request, session_id):
        answer = check_greeting(request.query)
        if answer:
            backend.update_session_history(session_id, request.query, answer)
        return answer

    def complete(request, source, session_id):
        greeting = greeting_answer(request, session_id)
        if greeting:
            return greeting, False
        if request.stream:
            if needs_context(request.query):
                return "请使用WebSocket接口获取流式响应", True
            answer, needs_rag = backend.bm25_search.search(
                request.query, threshold=0.85, source_filter=source
            )
            if needs_rag:
                return "请使用WebSocket接口获取流式响应", True
        return "".join(
            token
            for token, _ in backend.query(
                request.query, source_filter=source, session_id=session_id
            )
        ), False

    @application.get("/")
    def index():
        return FileResponse(ROOT / "static/index.html")

    @application.get("/health")
    async def health():
        return {
            "status": "healthy",
            "application": "edurag",
            "version": VERSION,
            "mode": actual_mode,
            "fallback_reason": fallback,
        }

    @application.get("/ready")
    def ready():
        def check():
            if actual_mode == "full":
                backend.mysql_client.connection.ping(reconnect=True)
                backend.redis_client.client.ping()
                backend.vector_store.client.get_collection_stats(
                    backend.vector_store.collection_name
                )
            else:
                backend.get_session_history("readiness")

        try:
            locked(check)
            return {"status": "ready", "mode": actual_mode}
        except Exception:
            return JSONResponse({"status": "unavailable", "mode": actual_mode}, status_code=503)

    @application.get("/api/sources")
    async def sources():
        return {"sources": backend.config.VALID_SOURCES}

    @application.post("/api/create_session")
    async def create_session():
        return {"session_id": str(uuid.uuid4())}

    @application.get("/api/history/{session_id}")
    def history(session_id: str):
        return {
            "session_id": session_id,
            "history": locked(backend.get_session_history, session_id),
        }

    @application.delete("/api/history/{session_id}")
    def clear_history(session_id: str):
        if not locked(backend.clear_session_history, session_id):
            raise HTTPException(500, "清除历史失败")
        return {"status": "success", "message": "历史记录已清除"}

    @application.post("/api/query")
    def query(request: QueryRequest):
        start = time.perf_counter()
        session_id = request.session_id or str(uuid.uuid4())
        try:
            source = scope(request.source_filter)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            answer, streaming = locked(complete, request, source, session_id)
        except Exception as exc:
            logger.error("回答失败：%s", type(exc).__name__)
            raise HTTPException(503, "回答服务暂时不可用，请稍后重试") from exc
        return {
            "answer": answer,
            "is_streaming": streaming,
            "session_id": session_id,
            "mode": actual_mode,
            "processing_time": time.perf_counter() - start,
        }

    @application.post("/api/retrieve")
    def retrieve(request: RetrievalRequest):
        """Inspect retrieved evidence without generating an answer or storing history."""
        if actual_mode != "full":
            raise HTTPException(409, "文档检索需要完整模式")
        try:
            source = scope(request.source_filter)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        start = time.perf_counter()
        try:
            docs = locked(
                backend.vector_store.hybrid_search_with_rerank,
                request.query,
                k=max(10, request.top_k),
                source_filter=source,
                mode=request.retrieval_mode,
                rerank=request.rerank,
                limit=request.top_k,
            )
        except Exception as exc:
            logger.error("检索失败：%s", type(exc).__name__)
            raise HTTPException(503, "文档检索暂时不可用") from exc
        return {
            "documents": [
                {
                    "text": d.page_content,
                    "source": d.metadata.get("source"),
                    "parent_id": d.metadata.get("parent_id"),
                    "file_name": d.metadata.get("file_name"),
                }
                for d in docs
            ],
            "processing_time": time.perf_counter() - start,
            "mode": actual_mode,
        }

    @application.websocket("/api/stream")
    async def stream(websocket: WebSocket):
        await websocket.accept()
        iterator = None
        try:
            request = QueryRequest.model_validate(await websocket.receive_json())
            source = scope(request.source_filter)
            session_id = request.session_id or str(uuid.uuid4())
            start = time.perf_counter()
            await websocket.send_json(
                {"type": "start", "session_id": session_id, "mode": actual_mode}
            )
            greeting = await run_in_threadpool(locked, greeting_answer, request, session_id)
            answer, first_token = "", None
            if greeting:
                await websocket.send_json(
                    {"type": "token", "token": greeting, "session_id": session_id}
                )
                answer, first_token = greeting, time.perf_counter() - start
            else:
                iterator = backend.query(request.query, source_filter=source, session_id=session_id)

                def step():
                    with application.state.lock:
                        return next(iterator, None)

                while True:
                    item = await run_in_threadpool(step)
                    if item is None:
                        break
                    token, is_complete = item
                    if token:
                        first_token = (
                            first_token if first_token is not None else time.perf_counter() - start
                        )
                        answer += token
                        await websocket.send_json(
                            {"type": "token", "token": token, "session_id": session_id}
                        )
                    if is_complete:
                        break
            await websocket.send_json(
                {
                    "type": "end",
                    "session_id": session_id,
                    "full_answer": answer,
                    "is_complete": True,
                    "processing_time": time.perf_counter() - start,
                    "first_token_seconds": first_token,
                    "mode": actual_mode,
                }
            )
        except (ValidationError, ValueError, TypeError):
            await websocket.send_json({"type": "error", "error": "请求格式、问题或学科无效"})
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("流式回答失败：%s", type(exc).__name__)
            try:
                await websocket.send_json(
                    {"type": "error", "error": "回答服务暂时不可用，请稍后重试"}
                )
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            if iterator is not None:
                await run_in_threadpool(locked, iterator.close)
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Passing the instance avoids loading every model twice when run as a script.
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8088")))
