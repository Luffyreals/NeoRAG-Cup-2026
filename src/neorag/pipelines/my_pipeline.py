"""Pipeline dự thi: retrieval và generation tách module, tham số đọc từ configs/pipeline.toml.

Chạy: neorag run train --no-judge --pipeline neorag.pipelines.my_pipeline:MyPipeline
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from neorag.config import CONFIG_DIR, load_config
from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.pipelines.baseline import GatewayEmbedder
from neorag.pipelines.generation import Generator
from neorag.pipelines.retrieval import build_retriever

DEFAULT_CONFIG = CONFIG_DIR / "pipeline.toml"


@dataclass
class LLMConfig:
    model: str = "claude-haiku-5-5"
    cache_path: str = ".cache/pipeline.sqlite"


@dataclass
class RetrievalConfig:
    method: str = "dense"
    embedding_backend: str = "gateway"  # gateway | local
    embedding_model: str = "bge-m3"
    n_retrieved: int = 10


@dataclass
class GenerationConfig:
    prompt: str = "baseline"
    top_k: int = 5


@dataclass
class PipelineConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_dict(cls, raw: dict) -> PipelineConfig:
        return cls(
            llm=LLMConfig(**raw.get("llm", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
        )

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


class MyPipeline(RAGPipeline):
    def __init__(
        self,
        config: PipelineConfig | str | Path | None = None,
        llm: LLMClient | None = None,
        embedder=None,
    ):
        if not isinstance(config, PipelineConfig):
            config = PipelineConfig.load(config)
        self.config = config
        self.llm_model = config.llm.model
        self.llm = llm or LLMClient(cache_path=config.llm.cache_path)
        self.embedder = embedder or build_embedder(config.retrieval, self.llm)
        self.retriever = build_retriever(config.retrieval.method, self.embedder)
        self.generator = Generator(
            self.llm, self.llm_model, config.generation.prompt, config.generation.top_k
        )

    def prepare(self) -> None:
        if hasattr(self.embedder, "load"):
            self.embedder.load()

    def index(self, corpus: list[Document]) -> None:
        self.retriever.index(corpus)

    def run(self, query: str) -> RAGOutput:
        hits = self.retriever.search(query, self.config.retrieval.n_retrieved)
        docs = [h.doc for h in hits]
        answer = self.generator.generate(query, docs)
        return RAGOutput(answer=answer, retrieved_ids=[d.id for d in docs])
