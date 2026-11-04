"""Rerank ứng viên bằng cross-encoder chạy local (mặc định BAAI/bge-reranker-v2-m3).

Cần cài thêm: python scripts/install.py --local
"""

from __future__ import annotations

import threading
from functools import lru_cache

from neorag.embeddings import _silence_libraries, is_downloaded
from neorag.pipelines.retrieval import ScoredDoc

_load_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(model_name: str, max_length: int):
    _silence_libraries()
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name, max_length=max_length, local_files_only=is_downloaded(model_name)
    )


def fuse_scores(matrix: list[list[float]], fusion: str = "max", rrf_k: int = 60) -> list[float]:
    """Gộp điểm cross-encoder `matrix[câu hỏi][doc]` thành một điểm cho mỗi doc.

    max: điểm cao nhất qua các câu hỏi (doc chỉ cần khớp tốt một ý).
    rrf: cộng 1/(k + hạng) của doc trong xếp hạng theo từng câu hỏi.
    """
    if fusion == "max":
        return [max(col) for col in zip(*matrix, strict=True)]
    if fusion == "rrf":
        fused = [0.0] * len(matrix[0])
        for row in matrix:
            order = sorted(range(len(row)), key=lambda i: -row[i])
            for rank, i in enumerate(order, start=1):
                fused[i] += 1.0 / (rrf_k + rank)
        return fused
    raise ValueError(f"reranker.multi_query không hợp lệ: {fusion!r}")


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 20,
        batch_size: int = 16,
        max_length: int = 512,
        num_threads: int = 0,
    ):
        self.model_name, self.top_n = model_name, top_n
        self.batch_size, self.max_length = batch_size, max_length
        self.num_threads = num_threads
        self._lock = threading.Lock()  # bộ chấm gọi từ nhiều luồng

    def load(self):
        """Nạp model (tải về nếu chưa có). Gọi nhiều lần cũng chỉ nạp một lần."""
        with _load_lock:
            if self.num_threads > 0:
                # CPU lai P-core/E-core (ví dụ i7-12800H): mặc định PyTorch chia việc cho cả E-core
                # chạy chậm hơn nên P-core phải chờ; ít luồng hơn lại nhanh hơn.
                import torch

                torch.set_num_threads(self.num_threads)
            return _load(self.model_name, self.max_length)

    @property
    def device(self) -> str:
        return str(self.load().device)

    def rerank(self, query: str, hits: list[ScoredDoc]) -> list[ScoredDoc]:
        """Chấm lại `top_n` ứng viên đầu; phần còn lại giữ nguyên thứ tự, xếp sau."""
        return self.rerank_multi([query], hits)

    def rerank_multi(
        self, queries: list[str], hits: list[ScoredDoc], fusion: str = "max", rrf_k: int = 60
    ) -> list[ScoredDoc]:
        """Như `rerank` nhưng chấm ứng viên với từng câu hỏi con rồi gộp điểm (`fuse_scores`).

        Câu nhiều ý: chấm bằng một câu duy nhất thì doc trả lời ý phụ dễ rơi khỏi top-5.
        """
        head, tail = hits[: self.top_n], hits[self.top_n :]
        if not head or not queries:
            return hits
        model = self.load()
        pairs = [(q, h.doc.text) for q in queries for h in head]
        with self._lock:
            flat = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        n = len(head)
        matrix = [[float(s) for s in flat[i * n : (i + 1) * n]] for i in range(len(queries))]
        scores = fuse_scores(matrix, fusion, rrf_k)
        reranked = sorted(
            (ScoredDoc(h.doc, s) for h, s in zip(head, scores, strict=True)),
            key=lambda h: -h.score,
        )
        floor = reranked[-1].score
        # phần đuôi nằm ngay dưới ứng viên thấp nhất, không làm giãn thang điểm
        return reranked + [ScoredDoc(h.doc, floor - 1e-6 * (i + 1)) for i, h in enumerate(tail)]
