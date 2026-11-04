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
from neorag.pipelines.retrieval import ScoredDoc

PIPELINE = os.getenv("NEORAG_PIPELINE", "neorag.pipelines.my_pipeline:MyPipeline")
os.environ.setdefault("NEORAG_PIPELINE_CONFIG", str(CONFIG_DIR / "demo.toml"))

# Tên các bước do MyPipeline.run(..., timings) đo, theo thứ tự chạy.
STEP_LABELS = {
    "rewrite": "Viết lại câu hỏi",
    "search": "Tìm kiếm",
    "rerank": "Xếp hạng lại",
    "generate": "Sinh câu trả lời",
}

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
        st.markdown(msg["content"])
        if msg.get("timing"):
            st.caption(msg["timing"])

if question := st.chat_input("Hỏi gì đó về ProPTIT, Multimedia hoặc GCC..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        timings: dict[str, float] = {}
        with st.spinner("Đang tìm tài liệu liên quan..."):
            retrieved_ids, pieces = answer_stream(pipeline, question, timings)
        answer = st.write_stream(mark_first_piece(pieces, start, timings))
        timing = format_timings(timings, time.perf_counter() - start)
        st.caption(timing)
        with st.expander("Nguồn tham khảo"):
            for doc_id in retrieved_ids:
                d = docs[doc_id]
                st.markdown(f"**[{d.id}] {d.club}** · _{d.topic}_\n\n{d.text}")
    st.session_state.messages.append({"role": "assistant", "content": answer, "timing": timing})
