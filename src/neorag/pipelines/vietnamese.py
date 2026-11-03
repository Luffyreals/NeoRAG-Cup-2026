"""Tách từ và bỏ dấu tiếng Việt cho BM25 và nhận diện CLB."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from neorag.text import normalize

_WORD = re.compile(r"\w+", flags=re.UNICODE)


def fold_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt: "Lập Trình" -> "Lap Trinh", "đ" -> "d"."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace(
        "Đ", "D"
    )  # NFD không tách "đ" (nét gạch là một phần của chữ, không phải dấu phụ) nên phải đổi tay


def syllables(text: str) -> list[str]:
    """Các âm tiết viết thường, đã bỏ dấu câu."""
    return _WORD.findall(normalize(text))


@lru_cache(maxsize=1)
def _underthesea_tokenize():
    from underthesea import word_tokenize

    return word_tokenize


def words(text: str, tokenizer: str = "underthesea") -> list[str]:
    """Tách từ ghép tiếng Việt ("thành lập" -> "thành_lập"); `syllable` thì tách theo âm tiết."""
    if tokenizer == "syllable":
        return syllables(text)
    if tokenizer != "underthesea":
        raise ValueError(f"bm25.tokenizer không hợp lệ: {tokenizer!r}")
    tokens = []
    for word in _underthesea_tokenize()(normalize(text)):
        parts = _WORD.findall(word)
        if parts:
            tokens.append("_".join(parts))
    return tokens


def bm25_tokens(text: str, tokenizer: str = "underthesea", fold: bool = True) -> list[str]:
    """Token cho BM25: từ có dấu, cộng thêm âm tiết bỏ dấu (tiền tố `~`) nếu `fold`.

    Nhờ phần bỏ dấu, câu hỏi gõ không dấu ("lap trinh") vẫn khớp được doc có dấu.
    """
    tokens = words(text, tokenizer)
    if fold:
        tokens += ["~" + fold_accents(s) for s in syllables(text)]
    return tokens
