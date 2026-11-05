"""Viết lại câu hỏi bằng LLM trước khi truy xuất: thêm dấu, mở rộng viết tắt, tách ý.

Chỉ dùng cho retrieval; câu hỏi gốc vẫn được đưa vào prompt sinh câu trả lời.
Giữ nguyên prompt giữa các lần chạy để trúng cache LLM.
"""

from __future__ import annotations

import re

PROMPTS: dict[str, str] = {
    "v1": """Bạn chuẩn hóa câu hỏi cho công cụ tìm kiếm tài liệu về cộng đồng S2B của Học viện Công nghệ Bưu chính Viễn thông (PTIT), gồm 3 câu lạc bộ: CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

Viết lại câu hỏi bên dưới thành tiếng Việt chuẩn, có đầy đủ dấu:
- Thêm dấu nếu câu gõ không dấu; sửa lỗi chính tả và teencode.
- Mở rộng viết tắt, ví dụ: clb → câu lạc bộ, lt → Lập Trình, hv → Học viện, tv → thành viên, bcn → ban chủ nhiệm, ko/k → không, mpc → CLB Multimedia PTIT, gcc → CLB Nhà Sáng Tạo Game PTIT.
- Ghi rõ tên đầy đủ của CLB được nhắc tới.
- Giữ nguyên số, ngày, năm, tên riêng. Không thêm thông tin, không trả lời câu hỏi.
- Nếu câu hỏi có nhiều ý hoặc so sánh nhiều CLB, viết thêm mỗi ý (mỗi CLB) thành một câu hỏi con.

Chỉ in ra các câu hỏi, mỗi câu một dòng, dòng đầu là câu viết lại đầy đủ, tối đa 4 dòng.

Câu hỏi: {query}""",
}

_BULLET = re.compile(r"^\s*(?:\d+[.)]|[-*•+])\s*")


def parse_queries(text: str, max_queries: int) -> list[str]:
    """Mỗi dòng không rỗng là một câu hỏi; bỏ ký hiệu đầu dòng và dòng trùng."""
    lines = [_BULLET.sub("", line).strip().strip('"') for line in text.splitlines()]
    return list(dict.fromkeys(line for line in lines if line))[:max_queries]


class QueryRewriter:
    def __init__(self, llm, model: str, prompt: str = "v1", max_queries: int = 4):
        if prompt not in PROMPTS:
            raise ValueError(f"rewrite.prompt không hợp lệ: {prompt!r}")
        self.llm, self.model = llm, model
        self.template = PROMPTS[prompt]
        self.max_queries = max_queries

    def rewrite(self, query: str) -> list[str]:
        """Danh sách câu hỏi dùng để truy xuất; dòng đầu là câu viết lại. Lỗi API thì trả []."""
        messages = [{"role": "user", "content": self.template.format(query=query)}]
        try:
            text = self.llm.chat(messages, model=self.model)
        except Exception:
            return []
        return parse_queries(text, self.max_queries)
