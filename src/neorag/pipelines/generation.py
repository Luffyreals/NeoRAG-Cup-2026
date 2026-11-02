"""Sinh câu trả lời từ các doc đã truy xuất."""

from __future__ import annotations

from neorag.data import Document
from neorag.pipelines import baseline

# Template prompt theo tên; config chọn bằng generation.prompt.
# "baseline" giữ nguyên prompt của BaselinePipeline để dùng lại cache.
PROMPTS: dict[str, str] = {
    "baseline": baseline.PROMPT,
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
