"""Pipeline dự thi: retrieval và generation tách module, tham số đọc từ configs/pipeline.toml.

Chạy: neorag run train --no-judge --pipeline neorag.pipelines.my_pipeline:MyPipeline
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from neorag.config import CONFIG_DIR, load_config
from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.pipelines.baseline import GatewayEmbedder
from neorag.pipelines.clubs import boost_clubs, detect_clubs
from neorag.pipelines.generation import Generator
from neorag.pipelines.retrieval import HybridRetriever, ScoredDoc, build_retriever, rrf_fuse
from neorag.pipelines.rewrite import QueryRewriter
from neorag.pipelines.vietnamese import accent_ratio

DEFAULT_CONFIG = CONFIG_DIR / "pipeline.toml"
# Câu trả lời khi generation.enabled = false: không gọi LLM sinh, chỉ đo retrieval.
RETRIEVAL_ONLY_ANSWER = "(retrieval-only)"

# Giá trị mặc định của các dataclass = hành vi baseline; configs/pipeline.toml bật các cải tiến.


@contextmanager
def timed(timings: dict[str, float] | None, step: str):
    """Cộng thời gian chạy (giây) của khối lệnh vào `timings[step]`; None thì không đo."""
    start = time.perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            timings[step] = timings.get(step, 0.0) + time.perf_counter() - start


@dataclass
class LLMConfig:
    model: str = "claude-haiku-5-5"
    cache_path: str = ".cache/pipeline.sqlite"


@dataclass
class RetrievalConfig:
    method: str = "dense"  # dense | bm25 | hybrid
    embedding_backend: str = "gateway"  # gateway | local
    embedding_model: str = "bge-m3"
    n_retrieved: int = 10
    depth: int = 30  # số doc mỗi retriever lấy trước khi gộp
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0


@dataclass
class BM25Config:
    tokenizer: str = "underthesea"  # underthesea | syllable
    fold_accents: bool = True  # thêm âm tiết bỏ dấu để khớp câu hỏi không dấu
    k1: float = 1.5
    b: float = 0.75


@dataclass
class RewriteConfig:
    enabled: bool = False
    prompt: str = "v1"
    max_queries: int = 4
    original_weight: float = 1.0  # trọng số RRF của câu hỏi gốc; mỗi câu viết lại có trọng số 1
    # Chỉ viết lại câu gõ không dấu (tỉ lệ âm tiết có dấu < accent_ratio): trên train, rewrite
    # chỉ có lợi với nhóm này, còn câu có dấu thì không, mà lại tốn thêm một lần gọi LLM.
    only_unaccented: bool = False
    accent_ratio: float = 0.2


@dataclass
class RerankerConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 20  # số ứng viên đầu được chấm lại
    batch_size: int = 16
    max_length: int = 512
    query: str = "rewrite"  # rewrite | original: câu đưa vào cross-encoder
    num_threads: int = 0  # số luồng PyTorch trên CPU; 0 = mặc định của PyTorch
    # none: chấm bằng một câu (`query`); max | rrf | mix: chấm với từng câu viết lại (câu đầy đủ +
    # các câu hỏi con) rồi gộp điểm, để doc trả lời ý phụ của câu nhiều ý không bị đẩy xuống.
    multi_query: str = "none"


@dataclass
class ClubBoostConfig:
    enabled: bool = False
    boost: float = 0.1  # cộng boost × (max − min điểm) cho doc đúng CLB


@dataclass
class GenerationConfig:
    enabled: bool = True  # false: không sinh câu trả lời, chỉ đo retrieval (không tốn API)
    prompt: str = "baseline"
    top_k: int = 5


@dataclass
class PipelineConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    club_boost: ClubBoostConfig = field(default_factory=ClubBoostConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_dict(cls, raw: dict) -> PipelineConfig:
        sections = {
            "llm": LLMConfig,
            "retrieval": RetrievalConfig,
            "bm25": BM25Config,
            "rewrite": RewriteConfig,
            "reranker": RerankerConfig,
            "club_boost": ClubBoostConfig,
            "generation": GenerationConfig,
        }
        if unknown := set(raw) - set(sections):
            raise TypeError(f"mục cấu hình không hợp lệ: {sorted(unknown)}")
        return cls(**{name: kind(**raw.get(name, {})) for name, kind in sections.items()})

    @classmethod
    def load(cls, path: str | Path | None = None) -> PipelineConfig:
        """Đọc từ `path`, hoặc NEORAG_PIPELINE_CONFIG, hoặc configs/pipeline.toml."""
        path = path or os.getenv("NEORAG_PIPELINE_CONFIG") or DEFAULT_CONFIG
        return cls.from_dict(load_config(path))


def build_embedder(cfg: RetrievalConfig, llm: LLMClient):
    if cfg.embedding_backend == "gateway":
        return GatewayEmbedder(llm, cfg.embedding_model)
    if cfg.embedding_backend == "local":
        from neorag.embeddings import LocalEmbedder

        return LocalEmbedder(cfg.embedding_model)
    raise ValueError(f"retrieval.embedding_backend không hợp lệ: {cfg.embedding_backend!r}")


def build_reranker(cfg: RerankerConfig):
    if not cfg.enabled:
        return None
    from neorag.pipelines.rerank import CrossEncoderReranker

    return CrossEncoderReranker(
        cfg.model, cfg.top_n, cfg.batch_size, cfg.max_length, cfg.num_threads
    )


class MyPipeline(RAGPipeline):
    """Câu hỏi → (rewrite) → dense/BM25 cho từng câu → RRF → (rerank) → (ưu tiên CLB) → sinh."""

    def __init__(
        self,
        config: PipelineConfig | str | Path | None = None,
        llm: LLMClient | None = None,
        embedder=None,
        reranker=None,
    ):
        if not isinstance(config, PipelineConfig):
            config = PipelineConfig.load(config)
        if config.reranker.query not in ("rewrite", "original"):
            raise ValueError(f"reranker.query không hợp lệ: {config.reranker.query!r}")
        if config.reranker.multi_query not in ("none", "max", "rrf", "mix"):
            raise ValueError(f"reranker.multi_query không hợp lệ: {config.reranker.multi_query!r}")
        self.config = config
        self.llm_model = config.llm.model
        self.llm = llm or LLMClient(cache_path=config.llm.cache_path)
        self.embedder = embedder or build_embedder(config.retrieval, self.llm)
        r, bm25 = config.retrieval, config.bm25
        self.retriever = build_retriever(
            r.method,
            self.embedder,
            bm25={
                "tokenizer": bm25.tokenizer,
                "fold": bm25.fold_accents,
                "k1": bm25.k1,
                "b": bm25.b,
            },
            weights={"dense": r.dense_weight, "bm25": r.bm25_weight},
            rrf_k=r.rrf_k,
            depth=r.depth,
        )
        self.rewriter = None
        if config.rewrite.enabled:
            self.rewriter = QueryRewriter(
                self.llm, self.llm_model, config.rewrite.prompt, config.rewrite.max_queries
            )
        self.reranker = reranker if reranker is not None else build_reranker(config.reranker)
        self.generator = Generator(
            self.llm, self.llm_model, config.generation.prompt, config.generation.top_k
        )

    @property
    def device(self) -> str | None:
        """Thiết bị chạy model local (reranker hoặc embedder), None nếu chỉ dùng gateway."""
        for model in (self.reranker, self.embedder):
            if model is not None and hasattr(model, "device"):
                return str(model.device)
        return None

    def prepare(self) -> None:
        for model in (self.embedder, self.reranker):
            if model is not None and hasattr(model, "load"):
                model.load()

    def index(self, corpus: list[Document]) -> None:
        self.retriever.index(corpus)

    def _rankings(self, query: str) -> tuple[list[list[ScoredDoc]], list[float]]:
        """Danh sách xếp hạng của từng retriever cho một câu hỏi, kèm trọng số RRF."""
        if isinstance(self.retriever, HybridRetriever):
            return self.retriever.rankings(query), self.retriever.weights
        return [self.retriever.search(query, self.config.retrieval.depth)], [1.0]

    def should_rewrite(self, query: str) -> bool:
        if self.rewriter is None:
            return False
        cfg = self.config.rewrite
        return not cfg.only_unaccented or accent_ratio(query) < cfg.accent_ratio

    def retrieve(self, query: str, timings: dict[str, float] | None = None) -> list[ScoredDoc]:
        """Truy xuất doc; truyền `timings` để nhận thời gian từng bước (chatbot hiển thị)."""
        cfg = self.config
        with timed(timings, "rewrite"):
            rewrites = self.rewriter.rewrite(query) if self.should_rewrite(query) else []
        with timed(timings, "search"):
            rankings, weights = [], []
            for q, qw in [(query, cfg.rewrite.original_weight)] + [(q, 1.0) for q in rewrites]:
                lists, ws = self._rankings(q)
                rankings += lists
                weights += [qw * w for w in ws]
            hits = rrf_fuse(rankings, weights, cfg.retrieval.rrf_k)

        if self.reranker is not None:
            use_rewrite = cfg.reranker.query == "rewrite" and rewrites
            with timed(timings, "rerank"):
                if cfg.reranker.multi_query != "none" and len(rewrites) > 1:
                    hits = self.reranker.rerank_multi(
                        rewrites, hits, cfg.reranker.multi_query, cfg.retrieval.rrf_k
                    )
                else:
                    hits = self.reranker.rerank(rewrites[0] if use_rewrite else query, hits)
        if cfg.club_boost.enabled:
            hits = boost_clubs(hits, detect_clubs(query, *rewrites), cfg.club_boost.boost)
        return hits[: cfg.retrieval.n_retrieved]

    def run(self, query: str, timings: dict[str, float] | None = None) -> RAGOutput:
        docs = [h.doc for h in self.retrieve(query, timings)]
        if self.config.generation.enabled:
            with timed(timings, "generate"):
                answer = self.generator.generate(query, docs)
        else:
            answer = RETRIEVAL_ONLY_ANSWER
        return RAGOutput(answer=answer, retrieved_ids=[d.id for d in docs])

    def run_stream(
        self, query: str, timings: dict[str, float] | None = None
    ) -> tuple[list[int], Iterator[str]]:
        """Cho chatbot: truy xuất xong mới trả về, còn câu trả lời là iterator sinh dần từng đoạn.

        `timings["generate"]` chỉ có sau khi đọc hết iterator.
        """
        docs = [h.doc for h in self.retrieve(query, timings)]
        ids = [d.id for d in docs]
        if not self.config.generation.enabled:
            return ids, iter([RETRIEVAL_ONLY_ANSWER])

        def pieces() -> Iterator[str]:
            with timed(timings, "generate"):
                yield from self.generator.generate_stream(query, docs)

        return ids, pieces()
