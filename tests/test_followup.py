from conftest import FakeLLM
from neorag.pipelines.followup import MAX_ANSWER_CHARS, condense_question, format_history

HISTORY = [
    {"role": "user", "content": "GCC thành lập năm nào?"},
    {"role": "assistant", "content": "CLB Nhà Sáng Tạo Game PTIT (GCC) ra đời năm 2023."},
]


def test_condense_uses_history():
    llm = FakeLLM(lambda m: '"Chủ nhiệm CLB Nhà Sáng Tạo Game PTIT (GCC) là ai?"\n')
    out = condense_question(llm, "m", HISTORY, "thế chủ nhiệm là ai?")
    assert out == "Chủ nhiệm CLB Nhà Sáng Tạo Game PTIT (GCC) là ai?"
    prompt = llm.calls[0][0]["content"]
    assert "Người dùng: GCC thành lập năm nào?" in prompt
    assert "CÂU HỎI MỚI: thế chủ nhiệm là ai?" in prompt


def test_condense_skips_without_history_and_survives_errors():
    llm = FakeLLM(lambda m: "khác")
    assert condense_question(llm, "m", [], "GCC là gì?") == "GCC là gì?"
    assert llm.calls == []  # câu đầu tiên: không gọi LLM

    def boom(messages):
        raise RuntimeError("gateway down")

    assert condense_question(FakeLLM(boom), "m", HISTORY, "thế còn MPC?") == "thế còn MPC?"
    assert condense_question(FakeLLM(lambda m: "  \n"), "m", HISTORY, "x?") == "x?"


def test_format_history_keeps_recent_turns_and_trims_answers():
    long_answer = "từ " * 200
    messages = [{"role": "user", "content": f"câu {i}"} for i in range(5)]
    messages.append({"role": "assistant", "content": long_answer})
    text = format_history(messages)
    assert "câu 0" not in text and "câu 4" in text
    assert len(text.splitlines()[-1]) < MAX_ANSWER_CHARS + 20
