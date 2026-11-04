"""Sinh câu trả lời từ các doc đã truy xuất."""

from __future__ import annotations

from neorag.data import Document
from neorag.pipelines import baseline

# Template prompt theo tên; config chọn bằng generation.prompt.
# "baseline" giữ nguyên prompt của BaselinePipeline để dùng lại cache.
PROMPTS: dict[str, str] = {
    "baseline": baseline.PROMPT,
    # v2: đáp án chuẩn ngắn (trung vị ~200 ký tự) và nêu nguyên văn số liệu, còn baseline trả lời
    # dài gấp đôi, có markdown và lời dẫn → mất ROUGE-L và Response Relevancy.
    "v2": """Bạn là trợ lý hỏi đáp về cộng đồng S2B gồm CLB Lập Trình PTIT (ProPTIT), \
CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

TÀI LIỆU:
{context}

CÂU HỎI: {query}

Trả lời câu hỏi chỉ dựa trên TÀI LIỆU, theo các quy tắc:
- Đi thẳng vào ý chính trong 1-3 câu văn xuôi tiếng Việt có dấu. Không dùng markdown (không in \
đậm, không gạch đầu dòng, không tiêu đề), không mở đầu bằng "Dựa vào tài liệu" hay nhắc lại câu hỏi.
- Câu đầu nêu rõ chủ thể được hỏi (tên CLB, sự kiện, chương trình) rồi trả lời ngay.
- Chép nguyên văn tên riêng, ngày tháng, số liệu, khẩu hiệu, tên kênh và đường link như trong \
tài liệu.
- Câu hỏi có nhiều ý thì trả lời đủ từng ý. Câu hỏi có/không thì mở đầu bằng "Có." hoặc "Không.".
- Không thêm nhận xét, lời khuyên hay thông tin không được hỏi; bỏ qua tài liệu không liên quan.
- Nếu tài liệu không có thông tin để trả lời, nói ngắn gọn trong một câu: "Tài liệu hiện có \
không có thông tin về ...", không đoán.""",
}


class Generator:
    def __init__(self, llm, model: str, prompt: str = "baseline", top_k: int = 5):
        if prompt not in PROMPTS:
            raise ValueError(f"generation.prompt không hợp lệ: {prompt!r}")
        self.llm, self.model = llm, model
        self.template = PROMPTS[prompt]
        self.top_k = top_k  # số doc đưa vào prompt

    def build_context(self, docs: list[Document]) -> str:
        return "\n\n".join(d.text for d in docs[: self.top_k])

    def build_messages(self, query: str, docs: list[Document]) -> list[dict]:
        content = self.template.format(context=self.build_context(docs), query=query)
        return [{"role": "user", "content": content}]

    def generate(self, query: str, docs: list[Document]) -> str:
        return self.llm.chat(self.build_messages(query, docs), model=self.model)
