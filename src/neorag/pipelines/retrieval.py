"""Các bộ truy xuất dùng trong MyPipeline.

Mỗi retriever có cùng interface `index(corpus)` / `search(query, k)`, nên có thể thay hoặc
ghép (hybrid, rerank) mà không đụng tới phần sinh câu trả lời.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from neorag.data import Document


@dataclass(frozen=True)
class ScoredDoc:
    doc: Document
    score: float


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Trả về ma trận (len(texts), dim) đã chuẩn hóa L2."""


class Retriever(Protocol):
    def index(self, corpus: list[Document]) -> None: ...

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        """Trả về tối đa k doc, xếp hạng từ liên quan nhất."""


class DenseRetriever:
    """Cosine similarity giữa embedding câu hỏi và embedding nguyên văn từng doc."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus
        self.doc_vectors = self.embedder.embed([d.text for d in corpus])

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        q = self.embedder.embed([query])[0]
        scores = self.doc_vectors @ q
        ranking = np.argsort(-scores)[:k]
        return [ScoredDoc(self.docs[i], float(scores[i])) for i in ranking]


def build_retriever(method: str, embedder: Embedder) -> Retriever:
    if method == "dense":
        return DenseRetriever(embedder)
    raise ValueError(f"retrieval.method không hợp lệ: {method!r}")
