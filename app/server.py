"""Backend cho giao diện chatbot web. Chạy: python app/server.py  (mở http://localhost:8000)

Cùng pipeline và cấu hình với app/chat.py (Streamlit vẫn giữ làm phương án dự phòng):
NEORAG_PIPELINE="module.path:ClassName" đổi pipeline, NEORAG_CHAT_CONFIG đổi cấu hình.

POST /api/chat trả về luồng Server-Sent Events, mỗi sự kiện là `event: <tên>` + `data: <json>`:
  status   {"step": "followup" | "retrieve" | "generate"}   bước đang chạy (hiện trạng thái)
  rewrite  {"standalone": "..."}                           câu hỏi nối tiếp đã được viết lại
  sources  {"docs": [...]}                                 tài liệu đưa vào prompt, gửi ngay khi truy xuất xong
  token    {"text": "..."}                                 từng đoạn chữ của câu trả lời
  done     {"no_info": bool, "timings": {...}, "total": s} kết thúc; no_info = tài liệu không có đáp án
  error    {"message": "...", "detail": "..."}             lỗi gateway/mạng, câu báo dễ hiểu
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from neorag.chat_errors import friendly_error
from neorag.config import CONFIG_DIR
from neorag.data import load_corpus
from neorag.pipelines import load_pipeline
from neorag.pipelines.followup import condense_question, needs_context
from neorag.pipelines.retrieval import ScoredDoc

PIPELINE = os.getenv("NEORAG_PIPELINE", "neorag.pipelines.my_pipeline:MyPipeline")
# Như app/chat.py: không để NEORAG_PIPELINE_CONFIG còn sót từ thí nghiệm làm chatbot chạy sai cấu hình.
os.environ["NEORAG_PIPELINE_CONFIG"] = os.getenv(
    "NEORAG_CHAT_CONFIG", str(CONFIG_DIR / "demo.toml")
)

STATIC_DIR = Path(__file__).parent / "static"
NO_INFO_PREFIX = "Tài liệu hiện có không có thông tin"
SNIPPET_CHARS = 160
CLUB_LABELS = {
    "CLB Lập Trình PTIT": "ProPTIT",
    "CLB Multimedia PTIT": "MPC",
    "CLB Nhà Sáng Tạo Game PTIT (GCC)": "GCC",
    "Cộng đồng S2B": "S2B",
}
EXAMPLE_QUESTIONS = [
    "CLB Lập Trình PTIT thành lập ngày nào?",
    "MPC và GCC khác nhau thế nào?",
    "lam sao de dang ky vao proptit",
    "Học phí của PTIT là bao nhiêu?",
]

state: dict = {}
# Pipeline (model PyTorch, cache SQLite) không chắc an toàn khi nhiều luồng cùng chạy:
# demo chỉ có một người dùng nên cho các câu hỏi chạy lần lượt.
pipeline_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    corpus = load_corpus()
    pipeline = load_pipeline(PIPELINE)
    pipeline.prepare()
    pipeline.index(corpus)
    reranker = getattr(pipeline, "reranker", None)
    if reranker is not None:  # lần suy luận đầu của PyTorch chậm: chạy trước khi nhận câu hỏi
        reranker.rerank("khởi động", [ScoredDoc(corpus[0], 0.0)])
    state["pipeline"] = pipeline
    state["docs"] = {d.id: d for d in corpus}
    yield
    state.clear()


app = FastAPI(title="NeoRAG Chatbot", lifespan=lifespan)


class Turn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = []  # các lượt trước (đã bỏ lượt lỗi), cũ trước mới sau


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def snippet(text: str) -> str:
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


def doc_payload(doc) -> dict:
    return {
        "id": doc.id,
        "club": doc.club,
        "label": CLUB_LABELS.get(doc.club, doc.club),
        "topic": doc.topic,
        "snippet": snippet(doc.text),
        "text": doc.text,
    }


def used_doc_count(pipeline) -> int:
    """Chỉ các doc đầu được đưa vào prompt mới là nguồn thật của câu trả lời."""
    config = getattr(pipeline, "config", None)
    return config.generation.top_k if config is not None else 5


def answer_stream(pipeline, question: str, timings: dict[str, float]):
    """(retrieved_ids, iterator chữ); pipeline không có run_stream thì trả cả câu một lần."""
    if hasattr(pipeline, "run_stream"):
        return pipeline.run_stream(question, timings=timings)
    if "timings" in inspect.signature(pipeline.run).parameters:
        out = pipeline.run(question, timings=timings)
    else:
        out = pipeline.run(question)
    return out.retrieved_ids, iter([out.answer])


def chat_events(req: ChatRequest) -> Iterator[str]:
    """Generator đồng bộ: StreamingResponse tự chạy nó trong threadpool nên không chặn server."""
    pipeline, docs = state["pipeline"], state["docs"]
    question = req.question.strip()
    history = [t.model_dump() for t in req.history]
    timings: dict[str, float] = {}
    start = time.perf_counter()
    with pipeline_lock:
        try:
            query = question
            llm = getattr(pipeline, "llm", None)
            if history and llm is not None and needs_context(question):
                yield sse("status", {"step": "followup"})
                t0 = time.perf_counter()
                model = getattr(pipeline, "llm_model", "claude-haiku-5-5")
                query = condense_question(llm, model, history, question)
                timings["followup"] = time.perf_counter() - t0
                if query.strip() != question:
                    yield sse("rewrite", {"standalone": query})

            yield sse("status", {"step": "retrieve"})
            retrieved_ids, pieces = answer_stream(pipeline, query, timings)
            sources = [doc_payload(docs[i]) for i in retrieved_ids[: used_doc_count(pipeline)]]
            yield sse("sources", {"docs": sources})

            yield sse("status", {"step": "generate"})
            answer = ""
            for piece in pieces:
                if not answer:
                    timings["first_piece"] = time.perf_counter() - start
                answer += piece
                yield sse("token", {"text": piece})
        except Exception as exc:  # lỗi gateway/mạng: câu báo dễ hiểu, chi tiết để mở xem
            yield sse(
                "error", {"message": friendly_error(exc), "detail": f"{type(exc).__name__}: {exc}"}
            )
            return
    yield sse(
        "done",
        {
            "no_info": answer.startswith(NO_INFO_PREFIX),
            "timings": {k: round(v, 2) for k, v in timings.items()},
            "total": round(time.perf_counter() - start, 2),
        },
    )


async def closing_stream(events: Iterator[str]) -> AsyncIterator[str]:
    """Chạy generator đồng bộ trong threadpool và luôn đóng nó khi kết thúc.

    Người dùng tải lại trang giữa lúc đang trả lời thì Starlette dừng gửi nhưng không đóng
    generator đồng bộ: `pipeline_lock` bị giữ mãi, mọi câu hỏi sau đó treo. `close()` ném
    GeneratorExit vào chỗ đang dừng, khối `with pipeline_lock` thoát và nhả khóa.
    """
    try:
        async for chunk in iterate_in_threadpool(events):
            yield chunk
    finally:
        events.close()


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    # X-Accel-Buffering: tắt buffer nếu sau này đặt sau nginx, để chữ vẫn hiện dần.
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(
        closing_stream(chat_events(req)), media_type="text/event-stream", headers=headers
    )


@app.get("/api/info")
def info() -> dict:
    """Thông tin cho giao diện: câu hỏi mẫu, số tài liệu, cấu hình đang chạy."""
    pipeline = state["pipeline"]
    config = getattr(pipeline, "config", None)
    return {
        "examples": EXAMPLE_QUESTIONS,
        "doc_count": len(state["docs"]),
        "pipeline": type(pipeline).__name__,
        "config": os.path.basename(os.environ["NEORAG_PIPELINE_CONFIG"]),
        "llm": config.llm.model if config is not None else None,
        "clubs": CLUB_LABELS,
    }


# Không có Cache-Control, trình duyệt tự đoán thời hạn cache và có thể ghép index.html cũ
# với app.js mới sau khi sửa giao diện. "no-cache" = vẫn cache nhưng hỏi lại server mỗi lần
# (dựa vào ETag, file không đổi thì server chỉ trả 304, gần như không tốn gì).
NO_CACHE = {"Cache-Control": "no-cache"}


class RevalidatedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


if STATIC_DIR.exists():
    app.mount("/static", RevalidatedStaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
