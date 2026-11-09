import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.data import Document
from neorag.pipelines.clubs import (
    GCC,
    MULTIMEDIA,
    PROPTIT,
    S2B,
    boost_clubs,
    detect_clubs,
    filter_rewrites,
)
from neorag.pipelines.generation import clean_answer
from neorag.pipelines.my_pipeline import RETRIEVAL_ONLY_ANSWER, MyPipeline, PipelineConfig
from neorag.pipelines.rerank import CrossEncoderReranker, fuse_scores
from neorag.pipelines.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ScoredDoc,
    build_retriever,
    rrf_fuse,
)
from neorag.pipelines.rewrite import QueryRewriter, parse_queries
from neorag.pipelines.vietnamese import accent_ratio, bm25_tokens, fold_accents, words


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

    def rerank_multi(self, queries, hits, fusion="max", rrf_k=60):
        self.queries.append(tuple(queries))
        return hits

    def rerank_groups(self, groups, hits):
        self.queries.append([(q, [h.doc.id for h in group]) for q, group in groups])
        return hits


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
    assert detect_clubs("clb mm va clb dpt") == {MULTIMEDIA}
    assert detect_clubs("học lập trình game ở đâu") == set()  # "lập trình" chung chung


def test_filter_rewrites_fixes_mm_and_drops_hallucinated_clubs():
    assert filter_rewrites("mm gia han nhan don", ["Mình gia hạn nhận đơn?"]) == [
        "CLB Multimedia PTIT (MPC) gia hạn nhận đơn?"
    ]
    assert filter_rewrites(
        "clb mm hop ban nao",
        ["CLB Multimedia PTIT hợp ban nào?", "CLB Lập Trình PTIT có ban nào?"],
    ) == ["CLB Multimedia PTIT hợp ban nào?"]


def test_clean_answer_flips_refusal_and_strips_hedging_tail():
    raw = "ProPTIT có 3 bài hát, nhưng tài liệu hiện có không có thông tin về tên bài hát."
    assert clean_answer(raw).startswith("Tài liệu hiện có không có thông tin")
    tail = "Khẩu hiệu là X. Slogan X là khẩu hiệu chung, không phải của đợt tuyển này."
    assert clean_answer(tail) == "Khẩu hiệu là X."
    date_ans = "Mở đến 10/09/2026. Tổng kết ngày 21/09/2026."
    assert "11 ngày" in clean_answer(date_ans)


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


def test_accent_ratio():
    assert accent_ratio("clb lap trinh ptit thanh lap ngay nao") == 0.0
    assert accent_ratio("CLB Lập Trình PTIT thành lập ngày nào?") > 0.5
    assert accent_ratio("") == 0.0


def test_only_unaccented_skips_rewrite_for_accented_query(corpus):
    pipeline = make(
        {
            "rewrite": {"enabled": True, "only_unaccented": True},
            "generation": {"enabled": False},
        },
        lambda m: "Câu viết lại",
    )
    pipeline.index(corpus)
    pipeline.run("ProPTIT được thành lập ngày nào?")
    assert pipeline.llm.calls == []  # câu có dấu: không gọi LLM
    pipeline.run("proptit thanh lap ngay nao")
    assert len(pipeline.llm.calls) == 1  # câu không dấu: viết lại


def test_fuse_scores_max_and_rrf():
    # doc 0 khớp ý 1, doc 1 khớp ý 2, doc 2 khớp vừa cả hai
    matrix = [[0.9, 0.1, 0.5], [0.1, 0.8, 0.6]]
    assert fuse_scores(matrix, "max") == [0.9, 0.8, 0.6]
    rrf = fuse_scores(matrix, "rrf", rrf_k=0)
    assert rrf[2] == pytest.approx(1 / 2 + 1 / 2)  # hạng 2 ở cả hai câu
    assert rrf[0] == pytest.approx(1 / 1 + 1 / 3)
    assert fuse_scores(matrix, "mix") == pytest.approx([0.5, 0.45, 0.55])
    for fusion in ("max", "mix"):
        assert fuse_scores([[0.3, 0.7]], fusion) == [0.3, 0.7]  # một câu: giữ nguyên
    with pytest.raises(ValueError):
        fuse_scores(matrix, "nope")


def test_multi_query_rerank_uses_all_rewrites(corpus):
    reranker = FakeReranker()
    pipeline = make(
        {
            "rewrite": {"enabled": True},
            "reranker": {"enabled": True, "multi_query": "max"},
            "generation": {"enabled": False},
        },
        lambda m: "Câu đầy đủ?\nÝ một?\nÝ hai?",
        reranker,
    )
    pipeline.index(corpus)
    pipeline.run("cau hoi nhieu y")
    assert reranker.queries == [("Câu đầy đủ?", "Ý một?", "Ý hai?")]
    pipeline.llm.responder = lambda m: "Câu một ý?"
    pipeline.run("cau hoi mot y")
    assert reranker.queries[-1] == "Câu một ý?"  # chỉ một câu: rerank thường


def test_invalid_reranker_query_raises():
    with pytest.raises(ValueError):
        make({"reranker": {"query": "nope"}})
    with pytest.raises(ValueError):
        make({"reranker": {"multi_query": "nope"}})
    with pytest.raises(TypeError):
        PipelineConfig.from_dict({"unknown_section": {}})


def test_run_reports_step_timings(club_corpus):
    reranker = FakeReranker()
    pipeline = make(
        {"rewrite": {"enabled": True}, "reranker": {"enabled": True}},
        lambda m: "Câu viết lại",
        reranker,
    )
    pipeline.index(club_corpus)
    timings = {}
    out = pipeline.run("gcc tuyen tv", timings=timings)
    assert set(timings) == {"rewrite", "search", "rerank", "generate"}
    assert all(t >= 0 for t in timings.values())
    assert out.retrieved_ids == pipeline.run("gcc tuyen tv").retrieved_ids  # không đo vẫn chạy


def test_run_stream_matches_run(club_corpus):
    pipeline = make({"retrieval": {"method": "hybrid"}}, lambda m: "GCC tuyển thành viên khóa mới.")
    pipeline.index(club_corpus)
    out = pipeline.run("gcc tuyen tv")
    timings = {}
    ids, pieces = pipeline.run_stream("gcc tuyen tv", timings=timings)
    assert ids == out.retrieved_ids
    assert "generate" not in timings  # chưa đọc iterator thì chưa sinh
    chunks = list(pieces)
    assert len(chunks) > 1 and "".join(chunks) == out.answer
    assert "generate" in timings
    disabled = make({"generation": {"enabled": False}})
    disabled.index(club_corpus)
    assert list(disabled.run_stream("gcc")[1]) == [RETRIEVAL_ONLY_ANSWER]


def test_sub_query_k_reranks_each_sub_query_with_own_candidates(club_corpus):
    reranker = FakeReranker()
    pipeline = make(
        {
            "retrieval": {"method": "hybrid"},
            "bm25": {"tokenizer": "syllable"},
            "rewrite": {"enabled": True},
            "reranker": {"enabled": True, "multi_query": "max", "top_n": 3, "sub_query_k": 1},
            "generation": {"enabled": False},
        },
        lambda m: "Cả ba CLB tuyển thành viên?\nGCC tuyển thành viên?\nS2B gồm những CLB nào?",
        reranker,
    )
    pipeline.index(club_corpus)
    pipeline.run("ca 3 clb tuyen tv")
    groups = reranker.queries[-1]
    assert [q for q, _ in groups] == [
        "Cả ba CLB tuyển thành viên?",
        "GCC tuyển thành viên?",
        "S2B gồm những CLB nào?",
    ]
    assert len(groups[0][1]) == 3  # câu đầy đủ: top_n ứng viên sau khi gộp
    assert groups[1][1] == [3] and groups[2][1] == [4]  # câu con: ứng viên tốt nhất của riêng nó
    with pytest.raises(ValueError):
        make({"reranker": {"multi_query": "rrf", "sub_query_k": 2}})


def test_rerank_groups_takes_max_and_keeps_unscored_tail(club_corpus):
    class FakeModel:
        def __init__(self):
            self.pairs = []

        def predict(self, pairs, batch_size, show_progress_bar):
            self.pairs += pairs
            return [float(len(set(q.split()) & set(t.split()))) for q, t in pairs]

    reranker = CrossEncoderReranker()
    model = FakeModel()
    reranker.load = lambda: model
    hits = [ScoredDoc(d, 0.0) for d in club_corpus]
    groups = [("GCC tuyển", hits[:2]), ("GCC tuyển thành viên", [hits[2], hits[0]])]
    out = reranker.rerank_groups(groups, hits)
    assert len(model.pairs) == 4  # mỗi câu chỉ chấm nhóm của nó
    assert [h.doc.id for h in out] == [3, 1, 2, 4]  # doc 4 không thuộc nhóm nào: xếp cuối
    assert out[1].score == 3.0  # doc 1: max(1 với câu đầu, 3 với câu sau)
