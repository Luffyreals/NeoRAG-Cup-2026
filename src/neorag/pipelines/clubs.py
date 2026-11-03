"""Nhận diện CLB được nhắc trong câu hỏi và ưu tiên mềm các doc của CLB đó."""

from __future__ import annotations

import re

from neorag.pipelines.retrieval import ScoredDoc
from neorag.pipelines.vietnamese import fold_accents
from neorag.text import normalize

PROPTIT = "CLB Lập Trình PTIT"
MULTIMEDIA = "CLB Multimedia PTIT"
GCC = "CLB Nhà Sáng Tạo Game PTIT (GCC)"
S2B = "Cộng đồng S2B"

# Tên gọi, viết tắt của từng CLB, viết thường và không dấu (so trên câu hỏi đã bỏ dấu).
ALIASES: dict[str, list[str]] = {
    PROPTIT: [
        "proptit",
        "pro ptit",
        "programming ptit",
        "clb lap trinh",
        "cau lac bo lap trinh",
        "lap trinh ptit",
        "clb lt",
        "lt ptit",
    ],
    MULTIMEDIA: ["multimedia", "multi media", "mpc", "clb media"],
    GCC: ["gcc", "nha sang tao game", "game creators club", "game creator club", "clb game"],
    S2B: ["s2b"],
}
_PATTERNS = {
    club: re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b")
    for club, aliases in ALIASES.items()
}


def detect_clubs(*texts: str) -> set[str]:
    """Các CLB (tên đúng như trường `club` của corpus) được nhắc trong bất kỳ đoạn nào."""
    found = set()
    for text in texts:
        folded = fold_accents(normalize(text))
        found |= {club for club, pattern in _PATTERNS.items() if pattern.search(folded)}
    return found


def boost_clubs(hits: list[ScoredDoc], clubs: set[str], boost: float) -> list[ScoredDoc]:
    """Cộng `boost × (max − min điểm)` cho doc thuộc CLB được nhắc rồi xếp lại.

    Không lọc: doc CLB khác vẫn còn trong danh sách (câu so sánh, doc S2B chung).
    """
    if not clubs or not hits or boost <= 0:
        return hits
    scores = [h.score for h in hits]
    bonus = boost * ((max(scores) - min(scores)) or 1.0)
    boosted = [ScoredDoc(h.doc, h.score + (bonus if h.doc.club in clubs else 0.0)) for h in hits]
    return sorted(boosted, key=lambda h: -h.score)
