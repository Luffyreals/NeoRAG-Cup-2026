"""Các bộ truy xuất dùng trong MyPipeline.

Mỗi retriever có cùng interface `index(corpus)` / `search(query, k)`, nên có thể thay hoặc
ghép (hybrid, rerank) mà không đụng tới phần sinh câu trả lời.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from neorag.data import Document
from neorag.pipelines.vietnamese import bm25_tokens


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


class BM25Retriever:
    """BM25 Okapi trên từ ghép tiếng Việt, kèm âm tiết bỏ dấu để khớp câu hỏi không dấu."""

    def __init__(
        self, tokenizer: str = "underthesea", fold: bool = True, k1: float = 1.5, b: float = 0.75
    ):
        self.tokenizer, self.fold, self.k1, self.b = tokenizer, fold, k1, b

    def tokens(self, text: str) -> list[str]:
        return bm25_tokens(text, self.tokenizer, self.fold)

    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus
        self.term_freqs = [Counter(self.tokens(d.text)) for d in corpus]
        self.doc_lens = np.array([sum(tf.values()) for tf in self.term_freqs], dtype=np.float32)
        self.avg_len = float(self.doc_lens.mean()) if corpus else 0.0
        df = Counter(t for tf in self.term_freqs for t in tf)
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.docs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_lens / max(self.avg_len, 1e-9))
        for term in set(self.tokens(query)):
            if term not in self.idf:
                continue
            tf = np.array([f.get(term, 0) for f in self.term_freqs], dtype=np.float32)
            scores += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        scores = self.scores(query)
        ranking = np.argsort(-scores, kind="stable")[:k]
        return [ScoredDoc(self.docs[i], float(scores[i])) for i in ranking if scores[i] > 0]


def rrf_fuse(
    rankings: Sequence[list[ScoredDoc]], weights: Sequence[float] | None = None, k: int = 60
) -> list[ScoredDoc]:
    """Reciprocal Rank Fusion: điểm = Σ w / (k + hạng). Chỉ dùng thứ hạng, không cần cùng thang điểm."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    docs: dict[int, Document] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.doc.id] = scores.get(hit.doc.id, 0.0) + w / (k + rank)
            docs[hit.doc.id] = hit.doc
    order = sorted(scores, key=lambda i: -scores[i])
    return [ScoredDoc(docs[i], scores[i]) for i in order]


class HybridRetriever:
    """Chạy nhiều retriever, mỗi cái lấy `depth` doc, rồi gộp bằng RRF có trọng số."""

    def __init__(
        self, retrievers: list[Retriever], weights: list[float], rrf_k: int = 60, depth: int = 30
    ):
        self.retrievers, self.weights, self.rrf_k, self.depth = retrievers, weights, rrf_k, depth

    def index(self, corpus: list[Document]) -> None:
        for r in self.retrievers:
            r.index(corpus)

    def rankings(self, query: str) -> list[list[ScoredDoc]]:
        return [r.search(query, self.depth) for r in self.retrievers]

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        return rrf_fuse(self.rankings(query), self.weights, self.rrf_k)[:k]


def build_retriever(
    method: str,
    embedder: Embedder,
    bm25: dict | None = None,
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    depth: int = 30,
) -> Retriever:
    """`method`: dense | bm25 | hybrid. `bm25` là tham số của BM25Retriever."""
    if method == "dense":
        return DenseRetriever(embedder)
    if method == "bm25":
        return BM25Retriever(**(bm25 or {}))
    if method == "hybrid":
        weights = weights or {}
        return HybridRetriever(
            [DenseRetriever(embedder), BM25Retriever(**(bm25 or {}))],
            [weights.get("dense", 1.0), weights.get("bm25", 1.0)],
            rrf_k,
            depth,
        )
    raise ValueError(f"retrieval.method không hợp lệ: {method!r}")
