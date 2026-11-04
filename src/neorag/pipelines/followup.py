"""Câu hỏi nối tiếp trong chatbot: viết lại thành câu tự đủ nghĩa dựa trên lịch sử hội thoại.

Ví dụ sau "GCC thành lập năm nào?", câu "thế chủ nhiệm là ai?" → "Chủ nhiệm GCC là ai?".
Pipeline xử lý từng câu độc lập, nên chatbot gọi bước này trước khi chạy pipeline.
Chỉ dùng cho chatbot; benchmark không có lịch sử nên không gọi.
"""

from __future__ import annotations

PROMPT = """Bạn giúp chatbot hỏi đáp về cộng đồng S2B (CLB Lập Trình PTIT - ProPTIT, CLB Multimedia \
PTIT - MPC, CLB Nhà Sáng Tạo Game PTIT - GCC) hiểu câu hỏi nối tiếp.

LỊCH SỬ HỘI THOẠI:
{history}

CÂU HỎI MỚI: {question}

Nếu câu hỏi mới phụ thuộc vào lịch sử (dùng "thế", "còn ... thì sao", "CLB đó", "nó", "ở đó", \
hoặc thiếu chủ thể đang được nói tới), hãy viết lại thành một câu hỏi tự hiểu được mà không cần \
lịch sử: nêu rõ chủ thể, giữ nguyên ý và cách hỏi. Nếu câu hỏi mới đã tự đủ nghĩa hoặc hỏi sang \
chủ đề khác, chép lại nguyên văn. Không trả lời câu hỏi. Chỉ in ra đúng một câu hỏi."""

# Lịch sử dài làm chậm và dễ kéo nhầm chủ thể cũ: chỉ lấy vài lượt gần nhất, cắt câu trả lời.
MAX_TURNS = 2
MAX_ANSWER_CHARS = 300


def format_history(messages: list[dict]) -> str:
    """`messages` dạng [{"role": "user" | "assistant", "content": ...}], cũ trước mới sau."""
    recent = messages[-2 * MAX_TURNS :]
    lines = []
    for msg in recent:
        if msg["role"] == "user":
            lines.append(f"Người dùng: {msg['content']}")
        else:
            text = msg["content"]
            if len(text) > MAX_ANSWER_CHARS:
                text = text[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + "…"
            lines.append(f"Trợ lý: {text}")
    return "\n".join(lines)


def condense_question(llm, model: str, messages: list[dict], question: str) -> str:
    """Câu hỏi tự đủ nghĩa; không có lịch sử hoặc LLM lỗi thì trả nguyên câu hỏi."""
    if not messages:
        return question
    prompt = PROMPT.format(history=format_history(messages), question=question)
    try:
        text = llm.chat([{"role": "user", "content": prompt}], model=model)
    except Exception:
        return question
    lines = [line.strip().strip('"') for line in text.splitlines() if line.strip()]
    return lines[0] if lines else question
