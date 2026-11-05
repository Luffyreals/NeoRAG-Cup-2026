"""Chatbot hỏi đáp cộng đồng S2B. Chạy: python -m streamlit run app/chat.py

Mặc định dùng MyPipeline với configs/demo.toml (reranker top-10: nhanh hơn cấu hình benchmark).
Đổi pipeline bằng NEORAG_PIPELINE="module.path:ClassName", đổi cấu hình bằng NEORAG_CHAT_CONFIG.
"""

from __future__ import annotations

import inspect
import os
import time

import streamlit as st

from neorag.chat_errors import friendly_error
from neorag.config import CONFIG_DIR
from neorag.data import load_corpus
from neorag.pipelines import load_pipeline
from neorag.pipelines.followup import condense_question, needs_context
from neorag.pipelines.retrieval import ScoredDoc

PIPELINE = os.getenv("NEORAG_PIPELINE", "neorag.pipelines.my_pipeline:MyPipeline")
# Biến riêng cho chatbot: NEORAG_PIPELINE_CONFIG còn sót lại sau khi chạy thí nghiệm (thường là
# config chỉ đo retrieval, tắt sinh câu trả lời) không được làm chatbot chạy sai cấu hình.
os.environ["NEORAG_PIPELINE_CONFIG"] = os.getenv(
    "NEORAG_CHAT_CONFIG", str(CONFIG_DIR / "demo.toml")
)

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

# Câu hỏi mẫu ở sidebar: mỗi câu cho thấy một khả năng của chatbot.
EXAMPLE_QUESTIONS = [
    "CLB Lập Trình PTIT thành lập ngày nào?",  # câu hỏi một ý, có số liệu
    "MPC và GCC khác nhau thế nào?",  # so sánh hai CLB
    "lam sao de dang ky vao proptit",  # gõ không dấu
    "Chủ nhiệm hiện tại của MPC và GCC là ai?",  # nhiều CLB trong một câu
    "Học phí của PTIT là bao nhiêu?",  # không có trong tài liệu: chatbot phải từ chối
]

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
    if not history or llm is None or not needs_context(question):
        return question  # không đo: bước này không chạy
    start = time.perf_counter()
    model = getattr(pipeline, "llm_model", "claude-haiku-5-5")
    result = condense_question(llm, model, history, question)
    timings["followup"] = time.perf_counter() - start
    return result


def show_standalone(standalone: str | None) -> None:
    if standalone:
        st.caption(f"Hiểu câu hỏi là: _{standalone}_")


def config_summary(pipeline) -> list[str]:
    """Các dòng mô tả cấu hình đang chạy (để biết chắc app đã nạp đúng config)."""
    lines = [f"Pipeline: `{type(pipeline).__name__}`"]
    config = getattr(pipeline, "config", None)
    if config is None:
        return lines
    r, rr, g = config.retrieval, config.reranker, config.generation
    lines.append(f"Config: `{os.path.basename(os.environ['NEORAG_PIPELINE_CONFIG'])}`")
    lines.append(
        f"Truy xuất: {r.method} ({r.embedding_model} + BM25), viết lại câu hỏi: "
        f"{'bật' if config.rewrite.enabled else 'tắt'}"
    )
    if rr.enabled:
        extra = ", int8" if rr.quantize else ""
        if rr.multi_query != "none":
            extra += f", theo câu hỏi con ({rr.multi_query})"
        lines.append(f"Xếp hạng lại: {rr.model.split('/')[-1]}, top-{rr.top_n}{extra}")
    lines.append(f"Sinh: {config.llm.model}, prompt {g.prompt}, {g.top_k} tài liệu")
    if device := getattr(pipeline, "device", None):
        lines.append(f"Thiết bị: {device}")
    return lines


def format_timings(timings: dict[str, float], total: float) -> str:
    parts = [f"{label} {timings[k]:.1f}s" for k, label in STEP_LABELS.items() if k in timings]
    if "first_piece" in timings:
        parts.append(f"Chữ đầu tiên sau {timings['first_piece']:.1f}s")
    return " · ".join(parts + [f"Tổng {total:.1f}s"])


pipeline, docs = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Câu hỏi mẫu")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example
    st.divider()
    if st.button("🗑️ Xóa hội thoại", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.pop("pending_question", None)
        st.rerun()
    config = getattr(pipeline, "config", None)
    if config is not None and not config.generation.enabled:
        st.warning("Cấu hình đang tắt sinh câu trả lời (chỉ đo retrieval).", icon="⚠️")
    with st.expander("Cấu hình đang chạy"):
        st.markdown("\n".join(f"- {line}" for line in config_summary(pipeline)))
    st.caption(f"Trả lời dựa trên {len(docs)} tài liệu về cộng đồng S2B.")

# Ô nhập luôn nằm cuối trang dù gọi ở đâu; đọc câu hỏi trước để ẩn lời chào khi đã có câu hỏi.
typed = st.chat_input("Hỏi gì đó về ProPTIT, Multimedia hoặc GCC...")
question = typed or st.session_state.pop("pending_question", None)

if not st.session_state.messages and not question:
    st.info(
        "Chào bạn! Mình trả lời các câu hỏi về CLB Lập Trình PTIT, CLB Multimedia PTIT và "
        "CLB Nhà Sáng Tạo Game PTIT: lịch sử, tuyển thành viên, hoạt động, liên hệ... "
        "Bạn có thể gõ không dấu, hỏi nhiều ý một lúc, hoặc hỏi tiếp câu trước. "
        "Chọn một câu hỏi mẫu ở thanh bên để bắt đầu."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        show_standalone(msg.get("standalone"))
        if msg.get("error"):
            st.error(msg["content"], icon="⚠️")
        else:
            st.markdown(msg["content"])
        if msg.get("timing"):
            st.caption(msg["timing"])
        render_sources(msg.get("sources", []))

if question:
    # Lịch sử cho câu hỏi nối tiếp: bỏ các lượt bị lỗi (không có câu trả lời thật).
    history = [m for m in st.session_state.messages if not m.get("error")]
    if history and history[-1]["role"] == "user":
        history = history[:-1]  # câu hỏi trước bị lỗi, không có câu trả lời đi kèm
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        timings: dict[str, float] = {}
        standalone = None
        try:
            with st.spinner("Đang tìm tài liệu liên quan..."):
                query = standalone_question(pipeline, history, question, timings)
                # Chỉ hiện khi câu hỏi thực sự được viết lại, để người dùng biết chatbot hiểu đúng.
                standalone = query if query.strip() != question.strip() else None
                show_standalone(standalone)
                retrieved_ids, pieces = answer_stream(pipeline, query, timings)
            answer = st.write_stream(mark_first_piece(pieces, start, timings))
        except Exception as exc:  # lỗi gateway/mạng: báo dễ hiểu thay vì traceback
            message = friendly_error(exc)
            st.error(message, icon="⚠️")
            with st.expander("Chi tiết lỗi"):
                st.code(f"{type(exc).__name__}: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": message, "error": True, "standalone": standalone}
            )
            st.stop()
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
