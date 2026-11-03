import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.data import Document
from neorag.pipelines.clubs import GCC, MULTIMEDIA, PROPTIT, S2B, boost_clubs, detect_clubs
from neorag.pipelines.my_pipeline import RETRIEVAL_ONLY_ANSWER, MyPipeline, PipelineConfig
from neorag.pipelines.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ScoredDoc,
    build_retriever,
    rrf_fuse,
)
from neorag.pipelines.rewrite import QueryRewriter, parse_queries
from neorag.pipelines.vietnamese import bm25_tokens, fold_accents, words


@pytest.fixture
def club_corpus() -> list[Document]:
    return [
        Document(1, PROPTIT, "Tuyển thành viên", "CLB Lập Trình PTIT tuyển thành viên khóa mới."),
        Document(
            2, MULTIMEDIA, "Tuyển thành viên", "CLB Multimedia PTIT tuyển thành viên khóa mới."
        ),
        Document(3, GCC, "Tuyển thành viên", "GCC tuyển thành viên khóa mới."),
        Document(4, S2B, "Giới thiệu chung", "S2B gồm ba câu lạc bộ học thuật."),
    ]


class FakeReranker:
    """Reranker giả: điểm = số âm tiết chung giữa câu hỏi và doc."""

    device = "cpu"

    def __init__(self):
        self.queries = []

    def load(self):
        return None

    def rerank(self, query, hits):
        self.queries.append(query)
        q = set(query.lower().split())
        scored = [ScoredDoc(h.doc, len(q & set(h.doc.text.lower().split()))) for h in hits]
        return sorted(scored, key=lambda h: -h.score)


# ---------------------------------------------------------------- tiếng Việt


def test_fold_accents():
    assert fold_accents("Lập Trình đào tạo Đội") == "Lap Trinh dao tao Doi"


def test_words_joins_compound_words():
    assert "thành_lập" in words("CLB được thành lập ngày 09/10/2011")
    assert words("Thành lập!", tokenizer="syllable") == ["thành", "lập"]
    with pytest.raises(ValueError):
        words("x", tokenizer="nope")


def test_bm25_tokens_fold_adds_unaccented_syllables():
    accented = bm25_tokens("Lập trình", tokenizer="syllable", fold=True)
    plain = bm25_tokens("lap trinh", tokenizer="syllable", fold=True)
    assert {"~lap", "~trinh"} <= set(accented) & set(plain)
    assert not set(bm25_tokens("lap", "syllable", fold=False)) & set(
        bm25_tokens("lập", "syllable", fold=False)
    )


# ---------------------------------------------------------------- BM25, RRF, hybrid


def test_bm25_matches_unaccented_query(corpus):
    retriever = BM25Retriever(tokenizer="syllable", fold=True)
    retriever.index(corpus)
    assert retriever.search("multimedia thanh lap khi nao", k=3)[0].doc.id == 2
    no_fold = BM25Retriever(tokenizer="syllable", fold=False)
    no_fold.index(corpus)
    assert all(h.doc.id != 1 for h in no_fold.search("lap trinh", k=3))


def test_bm25_underthesea_tokenizer(corpus):
    retriever = BM25Retriever(tokenizer="underthesea")
    retriever.index(corpus)
    assert retriever.search("GCC ra đời năm nào?", k=1)[0].doc.id == 3


def test_rrf_fuse_weights_and_ranks(corpus):
    a = [ScoredDoc(corpus[0], 9), ScoredDoc(corpus[1], 8)]
    b = [ScoredDoc(corpus[1], 5), ScoredDoc(corpus[2], 4)]
    fused = rrf_fuse([a, b], k=60)
    assert [h.doc.id for h in fused] == [2, 1, 3]  # doc 2 có mặt ở cả hai danh sách
    assert [h.doc.id for h in rrf_fuse([a, b], [1.0, 0.0], k=60)][0] == 1


def test_build_retriever_methods(corpus):
    assert isinstance(build_retriever("dense", FakeEmbedder()), DenseRetriever)
    assert isinstance(build_retriever("bm25", FakeEmbedder()), BM25Retriever)
    hybrid = build_retriever("hybrid", FakeEmbedder(), weights={"bm25": 0.5}, depth=2)
    assert isinstance(hybrid, HybridRetriever)
    assert hybrid.weights == [1.0, 0.5]
    hybrid.index(corpus)
    assert len(hybrid.search("ProPTIT thành lập ngày nào", k=10)) <= 3


# ---------------------------------------------------------------- CLB


def test_detect_clubs_aliases_and_unaccented():
    assert detect_clubs("clb lt ptit tuyen tv khi nao") == {PROPTIT}
    assert detect_clubs("Câu lạc bộ Lập Trình có gì?") == {PROPTIT}
    assert detect_clubs("mpc với gcc khác gì nhau") == {MULTIMEDIA, GCC}
    assert detect_clubs("Nhà sáng tạo Game", "S2B là gì") == {GCC, S2B}
    assert detect_clubs("học lập trình game ở đâu") == set()  # "lập trình" chung chung


def test_boost_clubs_is_soft(club_corpus):
    hits = [ScoredDoc(d, s) for d, s in zip(club_corpus, [0.9, 0.8, 0.5, 0.4], strict=True)]
    boosted = boost_clubs(hits, {GCC}, boost=1.0)  # +0.5 (= 1.0 × khoảng điểm)
    assert [h.doc.id for h in boosted] == [3, 1, 2, 4]
    assert len(boosted) == len(hits)  # không lọc doc CLB khác
    assert boost_clubs(hits, set(), boost=0.5) == hits


# ---------------------------------------------------------------- rewrite


def test_parse_queries_strips_bullets_and_duplicates():
    text = '1. Câu A?\n- Câu B?\n\n"Câu A?"\n* Câu C?'
    assert parse_queries(text, max_queries=4) == ["Câu A?", "Câu B?", "Câu C?"]
    assert parse_queries(text, max_queries=1) == ["Câu A?"]


def test_rewriter_calls_llm_and_survives_errors():
    llm = FakeLLM(lambda m: "Câu lạc bộ Lập Trình PTIT tuyển thành viên khi nào?")
    rewriter = QueryRewriter(llm, "claude-haiku-5-5")
    assert rewriter.rewrite("clb lt tuyen tv khi nao") == [
        "Câu lạc bộ Lập Trình PTIT tuyển thành viên khi nào?"
    ]
    assert "clb lt tuyen tv khi nao" in llm.calls[0][0]["content"]

    def boom(messages):
        raise RuntimeError("gateway down")

    assert QueryRewriter(FakeLLM(boom), "m").rewrite("x") == []
    with pytest.raises(ValueError):
        QueryRewriter(llm, "m", prompt="nope")


# ---------------------------------------------------------------- pipeline


def make(raw: dict, responder=None, reranker=None):
    return MyPipeline(
        PipelineConfig.from_dict(raw),
        llm=FakeLLM(responder or (lambda m: "Trả lời.")),
        embedder=FakeEmbedder(),
        reranker=reranker,
    )


def test_retrieval_only_mode_skips_generation(corpus):
    pipeline = make({"generation": {"enabled": False}})
    pipeline.index(corpus)
    out = pipeline.run("ProPTIT thành lập ngày nào?")
    assert out.answer == RETRIEVAL_ONLY_ANSWER
    assert out.retrieved_ids[0] == 1
    assert pipeline.llm.calls == []


def test_rewrite_used_for_retrieval_but_original_for_generation(club_corpus):
    rewritten = "CLB Nhà Sáng Tạo Game PTIT (GCC) tuyển thành viên"

    def responder(messages):
        content = messages[0]["content"]
        return rewritten if "Câu hỏi: " in content and "TÀI LIỆU" not in content else "Ok."

    reranker = FakeReranker()
    pipeline = make(
        {
            "retrieval": {"method": "hybrid"},
            "bm25": {"tokenizer": "syllable"},
            "rewrite": {"enabled": True},
            "reranker": {"enabled": True},
            "club_boost": {"enabled": True, "boost": 1.0},
        },
        responder,
        reranker,
    )
    pipeline.prepare()
    pipeline.index(club_corpus)
    query = "clb game tuyen tv ko"
    out = pipeline.run(query)
    assert out.retrieved_ids[0] == 3
    assert reranker.queries == [rewritten]
    assert len(pipeline.llm.calls) == 2  # rewrite + sinh câu trả lời
    assert f"CÂU HỎI: {query}" in pipeline.llm.calls[1][0]["content"]
    assert pipeline.device == "cpu"


def test_invalid_reranker_query_raises():
    with pytest.raises(ValueError):
        make({"reranker": {"query": "nope"}})
    with pytest.raises(TypeError):
        PipelineConfig.from_dict({"unknown_section": {}})
