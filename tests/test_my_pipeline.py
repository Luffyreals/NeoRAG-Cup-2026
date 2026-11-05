import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.data import Query
from neorag.pipelines import load_pipeline
from neorag.pipelines.baseline import BaselinePipeline
from neorag.pipelines.generation import Generator
from neorag.pipelines.my_pipeline import DEFAULT_CONFIG, MyPipeline, PipelineConfig
from neorag.pipelines.retrieval import DenseRetriever, build_retriever
from neorag.predict import predict


def make_pipeline(config=None, answer="Ngày 09/10/2011."):
    return MyPipeline(
        config or PipelineConfig(), llm=FakeLLM(lambda m: answer), embedder=FakeEmbedder()
    )


def test_dataclass_defaults_match_baseline():
    cfg = PipelineConfig()
    assert cfg.llm.model == "claude-haiku-5-5"
    assert cfg.retrieval.method == "dense"
    assert cfg.retrieval.embedding_model == "bge-m3"
    assert cfg.retrieval.n_retrieved == 10
    assert not (cfg.rewrite.enabled or cfg.reranker.enabled or cfg.club_boost.enabled)
    assert cfg.generation.prompt == "baseline"
    assert cfg.generation.top_k == 5


@pytest.mark.parametrize("name", ["pipeline.toml", "demo.toml"])
def test_shipped_config_files_load(name):
    cfg = PipelineConfig.load(DEFAULT_CONFIG.parent / name)
    assert cfg.llm.model == "claude-haiku-5-5"  # luật cuộc thi: chỉ dùng model này
    assert cfg.retrieval.method == "hybrid"
    assert cfg.reranker.enabled and cfg.club_boost.enabled and cfg.generation.enabled


def test_config_from_toml(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text("[generation]\ntop_k = 2\n", encoding="utf-8")
    cfg = PipelineConfig.load(path)
    assert cfg.generation.top_k == 2
    assert cfg.retrieval.n_retrieved == 10  # mục không ghi thì lấy mặc định


def test_config_env_var(tmp_path, monkeypatch):
    path = tmp_path / "p.toml"
    path.write_text("[retrieval]\nn_retrieved = 3\n", encoding="utf-8")
    monkeypatch.setenv("NEORAG_PIPELINE_CONFIG", str(path))
    assert PipelineConfig.load().retrieval.n_retrieved == 3


def test_invalid_options_raise():
    with pytest.raises(ValueError):
        build_retriever("nope", FakeEmbedder())
    with pytest.raises(ValueError):
        Generator(FakeLLM(), "m", prompt="nope")
    with pytest.raises(TypeError):
        PipelineConfig.from_dict({"generation": {"unknown": 1}})


def test_dense_retriever_ranks_closest_first(corpus):
    retriever = DenseRetriever(FakeEmbedder())
    retriever.index(corpus)
    hits = retriever.search("ProPTIT thành lập ngày nào?", k=2)
    assert [h.doc.id for h in hits][0] == 1
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_my_pipeline_runs_end_to_end(corpus):
    pipeline = make_pipeline()
    pipeline.prepare()
    pipeline.index(corpus)
    preds = predict(pipeline, [Query("q1", "ProPTIT thành lập ngày nào?")], max_retrieved=10)
    assert preds[0].answer == "Ngày 09/10/2011."
    assert preds[0].retrieved_ids[0] == 1
    assert len(preds[0].retrieved_ids) == 3


def test_my_pipeline_sends_same_prompt_as_baseline(corpus):
    """Cấu hình mặc định phải gửi đúng message như baseline để dùng lại cache LLM."""
    mine = make_pipeline(PipelineConfig.from_dict({"generation": {"top_k": 2}}))
    base = BaselinePipeline(top_k=2, llm=FakeLLM(), embedder=FakeEmbedder())
    for p in (mine, base):
        p.index(corpus)
    query = "Multimedia thành lập khi nào?"
    assert mine.run(query).retrieved_ids == base.run(query).retrieved_ids
    assert mine.llm.calls == base.llm.calls


def test_load_pipeline_by_spec():
    pipeline = load_pipeline(
        "neorag.pipelines.my_pipeline:MyPipeline", llm=FakeLLM(), embedder=FakeEmbedder()
    )
    assert isinstance(pipeline, MyPipeline)
    assert pipeline.llm_model == "claude-haiku-5-5"
