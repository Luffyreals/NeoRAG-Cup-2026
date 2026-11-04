"""Chatbot hỏi đáp cộng đồng S2B. Chạy: python -m streamlit run app/chat.py

Mặc định dùng MyPipeline với configs/demo.toml (reranker top-10: nhanh hơn cấu hình benchmark).
Đổi pipeline bằng NEORAG_PIPELINE="module.path:ClassName", đổi cấu hình bằng NEORAG_PIPELINE_CONFIG.
"""

from __future__ import annotations

import inspect
import os
import time

import streamlit as st

from neorag.config import CONFIG_DIR
from neorag.data import load_corpus
from neorag.pipelines import load_pipeline
from neorag.pipelines.followup import condense_question
from neorag.pipelines.retrieval import ScoredDoc

PIPELINE = os.getenv("NEORAG_PIPELINE", "neorag.pipelines.my_pipeline:MyPipeline")
os.environ.setdefault("NEORAG_PIPELINE_CONFIG", str(CONFIG_DIR / "demo.toml"))

# Tên các bước theo thứ tự chạy: "followup" do chatbot đo, còn lại do MyPipeline.run(..., timings).
STEP_LABELS = {
    "followup": "Hiểu ngữ cảnh",
    "rewrite": "Viết lại câu hỏi",
    "search": "Tìm kiếm",
    "rerank": "Xếp hạng lại",
    "generate": "Sinh câu trả lời",
}

# Nhãn ngắn và màu cho từng CLB khi hiện nguồn tham khảo.
CLUB_BADGES = {
    "CLB Lập Trình PTIT": ("ProPTIT", "blue"),
    "CLB Multimedia PTIT": ("Multimedia", "violet"),
    "CLB Nhà Sáng Tạo Game PTIT (GCC)": ("GCC", "orange"),
    "Cộng đồng S2B": ("S2B", "green"),
}
SHOWN_SOURCES = 3  # số nguồn hiện sẵn; các nguồn còn lại đưa vào prompt nằm trong mục "Thêm"
SNIPPET_CHARS = 160
# Câu mở đầu prompt v2 dùng khi tài liệu không có đáp án: khi đó không hiện nguồn (nguồn không liên quan).
NO_INFO_PREFIX = "Tài liệu hiện có không có thông tin"

st.set_page_config(page_title="NeoRAG Chatbot - S2B", page_icon="💬")
st.title("💬 Hỏi đáp cộng đồng S2B")
st.caption("CLB Lập Trình PTIT · CLB Multimedia PTIT · CLB Nhà Sáng Tạo Game PTIT")


@st.cache_resource(show_spinner="Đang nạp model và xây index (lần đầu có thể mất vài phút)...")
def get_pipeline():
    corpus = load_corpus()
    pipeline = load_pipeline(PIPELINE)
    pipeline.prepare()
    pipeline.index(corpus)
    # Chạy thử reranker một lần: lần suy luận đầu của PyTorch chậm hơn hẳn các lần sau,
    # nên làm trước để câu hỏi đầu tiên của người dùng không phải chịu.
    reranker = getattr(pipeline, "reranker", None)
    if reranker is not None:
        reranker.rerank("khởi động", [ScoredDoc(corpus[0], 0.0)])
    return pipeline, {d.id: d for d in corpus}


def answer_stream(pipeline, question: str, timings: dict[str, float]):
    """Trả (retrieved_ids, iterator các đoạn chữ của câu trả lời).

    Pipeline có `run_stream` (MyPipeline) thì chữ hiện dần ngay khi model sinh; pipeline khác
    (ví dụ baseline) thì chạy `run` như cũ và trả cả câu một lần.
    """
    if hasattr(pipeline, "run_stream"):
        return pipeline.run_stream(question, timings=timings)
    if "timings" in inspect.signature(pipeline.run).parameters:
        out = pipeline.run(question, timings=timings)
    else:
        out = pipeline.run(question)
    return out.retrieved_ids, iter([out.answer])


def mark_first_piece(pieces, start: float, timings: dict[str, float]):
    """Ghi lại lúc chữ đầu tiên hiện ra: đây là thời gian người dùng thực sự phải chờ."""
    for i, piece in enumerate(pieces):
        if i == 0:
            timings["first_piece"] = time.perf_counter() - start
        yield piece


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Đoạn đầu của doc, cắt ở ranh giới từ."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def used_doc_count(pipeline) -> int:
    """Số doc đầu được đưa vào prompt sinh câu trả lời (chỉ những doc này mới là nguồn thật)."""
    config = getattr(pipeline, "config", None)
    return config.generation.top_k if config is not None else 5


def render_source(doc) -> None:
    label, color = CLUB_BADGES.get(doc.club, (doc.club, "gray"))
    with st.container(border=True):
        st.markdown(f":{color}-badge[{label}] **{doc.topic}**")
        st.caption(snippet(doc.text))
        if len(doc.text) > SNIPPET_CHARS:
            with st.expander("Xem toàn văn"):
                st.markdown(doc.text)


def render_sources(doc_ids: list[int]) -> None:
    if not doc_ids:
        return
    st.caption("Nguồn tham khảo")
    for doc_id in doc_ids[:SHOWN_SOURCES]:
        render_source(docs[doc_id])
    if rest := doc_ids[SHOWN_SOURCES:]:
        with st.expander(f"Thêm {len(rest)} nguồn khác"):
            for doc_id in rest:
                d = docs[doc_id]
                label = CLUB_BADGES.get(d.club, (d.club, "gray"))[0]
                st.markdown(f"**{label} · {d.topic}**\n\n{d.text}")


def standalone_question(pipeline, history: list[dict], question: str, timings: dict[str, float]):
    """Câu hỏi nối tiếp → câu tự đủ nghĩa (cần lịch sử và pipeline có LLM; không thì giữ nguyên)."""
    llm = getattr(pipeline, "llm", None)
    if not history or llm is None:
        return question
    start = time.perf_counter()
    model = getattr(pipeline, "llm_model", "claude-haiku-5-5")
    result = condense_question(llm, model, history, question)
    timings["followup"] = time.perf_counter() - start
    return result


def show_standalone(standalone: str | None) -> None:
    if standalone:
        st.caption(f"Hiểu câu hỏi là: _{standalone}_")


def format_timings(timings: dict[str, float], total: float) -> str:
    parts = [f"{label} {timings[k]:.1f}s" for k, label in STEP_LABELS.items() if k in timings]
    if "first_piece" in timings:
        parts.append(f"Chữ đầu tiên sau {timings['first_piece']:.1f}s")
    return " · ".join(parts + [f"Tổng {total:.1f}s"])


pipeline, docs = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        show_standalone(msg.get("standalone"))
        st.markdown(msg["content"])
        if msg.get("timing"):
            st.caption(msg["timing"])
        render_sources(msg.get("sources", []))

if question := st.chat_input("Hỏi gì đó về ProPTIT, Multimedia hoặc GCC..."):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        timings: dict[str, float] = {}
        with st.spinner("Đang tìm tài liệu liên quan..."):
            query = standalone_question(pipeline, history, question, timings)
            # Chỉ hiện khi câu hỏi thực sự được viết lại, để người dùng biết chatbot hiểu đúng.
            standalone = query if query.strip() != question.strip() else None
            show_standalone(standalone)
            retrieved_ids, pieces = answer_stream(pipeline, query, timings)
        answer = st.write_stream(mark_first_piece(pieces, start, timings))
        timing = format_timings(timings, time.perf_counter() - start)
        st.caption(timing)
        sources = (
            [] if answer.startswith(NO_INFO_PREFIX) else retrieved_ids[: used_doc_count(pipeline)]
        )
        render_sources(sources)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "timing": timing,
            "sources": sources,
            "standalone": standalone,
        }
    )
