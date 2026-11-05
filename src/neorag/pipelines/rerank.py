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
def _load(model_name: str, max_length: int, quantize: bool = False):
    _silence_libraries()
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        model_name, max_length=max_length, local_files_only=is_downloaded(model_name)
    )
    if quantize:
        # int8 động cho các lớp Linear: trên CPU chấm nhanh ~2.2 lần (10 cặp: 2.8 s → 1.25 s),
        # điểm lệch nhẹ so với fp32 nên phải đo lại retrieval (exp12).
        import warnings

        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
    return model


def fuse_scores(matrix: list[list[float]], fusion: str = "max", rrf_k: int = 60) -> list[float]:
    """Gộp điểm cross-encoder `matrix[câu hỏi][doc]` thành một điểm cho mỗi doc.

    max: điểm cao nhất qua các câu hỏi (doc chỉ cần khớp tốt một ý).
    rrf: cộng 1/(k + hạng) của doc trong xếp hạng theo từng câu hỏi.
    mix: ½ điểm theo câu đầu (câu đầy đủ) + ½ điểm cao nhất theo các câu còn lại (câu hỏi con),
    để một câu con khớp mạnh doc khác không một mình đẩy doc đúng của câu một ý xuống.
    """
    if fusion == "max":
        return [max(col) for col in zip(*matrix, strict=True)]
    if fusion == "mix":
        if len(matrix) == 1:
            return list(matrix[0])
        return [0.5 * col[0] + 0.5 * max(col[1:]) for col in zip(*matrix, strict=True)]
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
        quantize: bool = False,
    ):
        self.model_name, self.top_n = model_name, top_n
        self.batch_size, self.max_length = batch_size, max_length
        self.num_threads, self.quantize = num_threads, quantize
        self._lock = threading.Lock()  # bộ chấm gọi từ nhiều luồng

    def load(self):
        """Nạp model (tải về nếu chưa có). Gọi nhiều lần cũng chỉ nạp một lần."""
        with _load_lock:
            if self.num_threads > 0:
                # CPU lai P-core/E-core (ví dụ i7-12800H): mặc định PyTorch chia việc cho cả E-core
                # chạy chậm hơn nên P-core phải chờ; ít luồng hơn lại nhanh hơn.
                import torch

                torch.set_num_threads(self.num_threads)
            return _load(self.model_name, self.max_length, self.quantize)

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

    def rerank_groups(
        self, groups: list[tuple[str, list[ScoredDoc]]], hits: list[ScoredDoc]
    ) -> list[ScoredDoc]:
        """Mỗi câu hỏi chỉ chấm nhóm ứng viên của riêng nó; điểm doc = max qua các câu đã chấm nó.

        Khác `rerank_multi` (mọi câu × cùng top_n ứng viên): câu hỏi con về một CLB tự mang ứng viên
        tốt nhất của nó vào, nên doc trả lời ý phụ không bị loại chỉ vì ngoài top_n sau khi gộp RRF,
        và số cặp phải chấm ít hơn. Doc trong `hits` không thuộc nhóm nào xếp sau, giữ thứ tự.
        """
        pairs: dict[tuple[str, int], tuple[str, str]] = {}
        docs = {}
        for query, group in groups:
            for h in group:
                pairs.setdefault((query, h.doc.id), (query, h.doc.text))
                docs[h.doc.id] = h.doc
        if not pairs:
            return hits
        model = self.load()
        with self._lock:
            flat = model.predict(
                list(pairs.values()), batch_size=self.batch_size, show_progress_bar=False
            )
        best: dict[int, float] = {}
        for (_, doc_id), score in zip(pairs, flat, strict=True):
            best[doc_id] = max(best.get(doc_id, float("-inf")), float(score))
        reranked = sorted((ScoredDoc(docs[i], s) for i, s in best.items()), key=lambda h: -h.score)
        floor = reranked[-1].score
        tail = [h for h in hits if h.doc.id not in best]
        return reranked + [ScoredDoc(h.doc, floor - 1e-6 * (i + 1)) for i, h in enumerate(tail)]
