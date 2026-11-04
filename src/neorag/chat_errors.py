"""Thông báo lỗi dễ hiểu cho chatbot thay vì traceback."""

from __future__ import annotations


def friendly_error(exc: BaseException) -> str:
    """Câu thông báo tiếng Việt theo loại lỗi khi gọi gateway LLM."""
    import openai

    if isinstance(exc, openai.AuthenticationError):
        return (
            "Khóa API không hợp lệ hoặc đã bị thu hồi. Kiểm tra `OPENAI_API_KEY` trong file `.env`."
        )
    if isinstance(exc, openai.RateLimitError):
        return (
            "Máy chủ mô hình đang nhận quá nhiều yêu cầu hoặc đã hết hạn mức. "
            "Bạn đợi một lát rồi hỏi lại nhé."
        )
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return "Không kết nối được tới máy chủ mô hình. Kiểm tra mạng rồi hỏi lại nhé."
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code in (402, 403):
            return "Khóa API đã hết hạn mức hoặc không có quyền dùng mô hình này."
        if exc.status_code >= 500:
            return "Máy chủ mô hình đang gặp sự cố. Bạn thử hỏi lại sau ít phút nhé."
    return "Đã có lỗi khi tạo câu trả lời. Bạn thử hỏi lại nhé."
