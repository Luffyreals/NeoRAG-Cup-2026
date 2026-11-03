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


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 20,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.model_name, self.top_n = model_name, top_n
        self.batch_size, self.max_length = batch_size, max_length
        self._lock = threading.Lock()  # bộ chấm gọi từ nhiều luồng

    def load(self):
        """Nạp model (tải về nếu chưa có). Gọi nhiều lần cũng chỉ nạp một lần."""
        with _load_lock:
            return _load(self.model_name, self.max_length)

    @property
    def device(self) -> str:
        return str(self.load().device)

    def rerank(self, query: str, hits: list[ScoredDoc]) -> list[ScoredDoc]:
        """Chấm lại `top_n` ứng viên đầu; phần còn lại giữ nguyên thứ tự, xếp sau."""
        head, tail = hits[: self.top_n], hits[self.top_n :]
        if not head:
            return hits
        model = self.load()
        with self._lock:
            scores = model.predict(
                [(query, h.doc.text) for h in head],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        reranked = sorted(
            (ScoredDoc(h.doc, float(s)) for h, s in zip(head, scores, strict=True)),
            key=lambda h: -h.score,
        )
        floor = reranked[-1].score
        # phần đuôi nằm ngay dưới ứng viên thấp nhất, không làm giãn thang điểm
        return reranked + [ScoredDoc(h.doc, floor - 1e-6 * (i + 1)) for i, h in enumerate(tail)]
