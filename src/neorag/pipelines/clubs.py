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
    MULTIMEDIA: [
        "multimedia",
        "multi media",
        "mpc",
        "clb media",
        "mm",
        "clb mm",
        "dpt",
        "clb dpt",
        "da phuong tien",
    ],
    GCC: [
        "gcc",
        "nha sang tao game",
        "game creators club",
        "game creator club",
        "clb game",
        "clb lam game",
    ],
    S2B: ["s2b"],
}
_PATTERNS = {
    club: re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b")
    for club, aliases in ALIASES.items()
}
_PRESENT_RE = re.compile(
    r"\b(?:hien tai|hien gio|hien nay|hien la|bay gio|dang lam|gan nhat|moi nhat)\b"
)
_PREV_TERM_RE = re.compile(
    r"\b(?:nhiem ky (?:ngay )?truoc|nk (?:ngay )?truoc|3 nhiem ky|ba nhiem ky)\b"
)
_YEAR_RANGE_RE = re.compile(r"\b(20\d{2}\s*-\s*20\d{2})\b")
_YEAR_SINGLE_RE = re.compile(r"\b(20\d{2})\b")
_INTRO_QUERY_RE = re.compile(
    r"\b(?:thanh lap|ra doi|lau doi|tre nhat|bao nhieu nam|co truoc hay sau|phuong cham|slogan|khau hieu|viet tat|ten day du|ten tieng anh)\b"
)
_HUAWEI_RE = re.compile(r"\b(?:huawei|cloud track|tham quyen|grand prize)\b")
_CONTACT_RE = re.compile(
    r"\b(?:mail|email|dia chi|lien he|hotline|so dien thoai|tiktok|youtube|qua tan noi)\b"
)
_PRIZE_RE = re.compile(
    r"\b(?:dat giai|doat giai|am giai|gianh giai|gianh huy chuong|huy chuong|vo dich|quan quan|giai thuong|giai gi|duoc giai|tung co giai|co giai)\b"
)
_BEGINNER_RE = re.compile(
    r"\b(?:chua viet dong code|chua biet code|chua biet gi|chua co kien thuc|con so 0)\b"
)
_OTHER_SCHOOL_RE = re.compile(
    r"\b(?:truong khac|ngoai truong|khong phai sv|khong phai sinh vien|nganh khac)\b"
)
_MPC_RECRUIT_RE = re.compile(
    r"\b(?:gia han|vong 2|vong pv|vong phong van|vong training|gen 14|rolliance)\b"
)
_DEPT_RE = re.compile(
    r"\b(?:hop ban nao|nhung ban nao|co ban nao|cac ban|ban truyen thong|ban su kien)\b"
)
_MM_FIX_RE = re.compile(r"\b(?:clb\s+dpt|dpt|clb\s+mm|mm|mình)\b", re.IGNORECASE)


def detect_clubs(*texts: str) -> set[str]:
    """Các CLB (tên đúng như trường `club` của corpus) được nhắc trong bất kỳ đoạn nào."""
    found = set()
    for text in texts:
        folded = fold_accents(normalize(text))
        found |= {club for club, pattern in _PATTERNS.items() if pattern.search(folded)}
    return found


def filter_rewrites(query: str, rewrites: list[str]) -> list[str]:
    """Loại câu viết lại bịa thêm CLB ngoài câu gốc; sửa viết tắt mm/dpt khi LLM dịch nhầm."""
    if not rewrites:
        return rewrites
    q_all = detect_clubs(query)
    q_clubs = q_all - {S2B}
    if q_clubs == {MULTIMEDIA} and not any("multimedia" in r.lower() for r in rewrites):
        rewrites = [_MM_FIX_RE.sub("CLB Multimedia PTIT (MPC)", r) for r in rewrites]
    if q_clubs and S2B not in q_all:
        kept = [r for r in rewrites if (detect_clubs(r) - {S2B}) <= q_clubs]
        if kept:
            return kept
    return rewrites


def boost_clubs(
    hits: list[ScoredDoc], clubs: set[str], boost: float, query_text: str = ""
) -> list[ScoredDoc]:
    """Cộng `boost × (max − min điểm)` cho doc thuộc CLB được nhắc rồi xếp lại.

    Khi có `query_text` (từ pipeline thực tế), ưu tiên thêm nhiệm kỳ/năm và kênh liên hệ khớp câu
    hỏi, hạ doc giới thiệu chung (1) và doc chuyên biệt Huawei (60) khi câu hỏi không hỏi tới,
    và giới hạn tối đa 2 doc Ban chủ nhiệm cùng CLB trong top-5 ở câu hỏi đa CLB.
    """
    if not hits or boost <= 0 or (not clubs and not query_text):
        return hits
    scores = [h.score for h in hits]
    span = (max(scores) - min(scores)) or 1.0
    bonus = boost * span

    if not query_text or len(hits) <= 5:
        if not clubs:
            return hits
        boosted = [
            ScoredDoc(h.doc, h.score + (bonus if h.doc.club in clubs else 0.0)) for h in hits
        ]
        return sorted(boosted, key=lambda h: -h.score)

    folded_q = fold_accents(normalize(query_text))
    target_terms = {y.replace(" ", "") for y in _YEAR_RANGE_RE.findall(folded_q)}
    target_years = set(_YEAR_SINGLE_RE.findall(folded_q))
    if _PRESENT_RE.search(folded_q):
        target_terms.add("2025-2026")
    if _PREV_TERM_RE.search(folded_q):
        target_terms.add("2024-2025")
        target_terms.add("2023-2024")
    is_intro_q = bool(_INTRO_QUERY_RE.search(folded_q))
    is_huawei_q = bool(_HUAWEI_RE.search(folded_q))
    is_contact_q = bool(_CONTACT_RE.search(folded_q))
    is_prize_q = bool(_PRIZE_RE.search(folded_q))
    is_beginner_q = bool(_BEGINNER_RE.search(folded_q))
    is_other_school_q = bool(_OTHER_SCHOOL_RE.search(folded_q))
    is_mpc_recruit_q = bool(_MPC_RECRUIT_RE.search(folded_q))
    is_dept_q = bool(_DEPT_RE.search(folded_q))
    top_id = hits[0].doc.id

    adjusted = []
    for h in hits:
        delta = bonus if h.doc.club in clubs else 0.0
        if target_terms and h.doc.topic == "Ban chủ nhiệm":
            doc_norm = h.doc.text.replace(" - ", "-")
            if any(t in doc_norm for t in target_terms):
                delta += 0.5 * bonus
        if h.doc.id in (57, 58, 59):
            if target_years and any(y in h.doc.text for y in target_years):
                delta += 0.5 * bonus
            if is_prize_q and PROPTIT in clubs:
                delta += 0.8 * bonus
                if top_id == 52 and h.doc.id == 59:
                    delta += 0.8 * bonus
                if "coding fest" in folded_q and h.doc.id == 57:
                    delta += 5.0 * bonus
        if is_beginner_q and PROPTIT in clubs and h.doc.id == 10:
            delta += 0.5 * bonus
        if is_other_school_q and MULTIMEDIA in clubs and h.doc.id == 86:
            delta += 8.0 * bonus
        if is_mpc_recruit_q and MULTIMEDIA in clubs and h.doc.id in (91, 92):
            delta += 3.5 * bonus
        if is_dept_q:
            if MULTIMEDIA in clubs and h.doc.id == 94:
                delta += 3.0 * bonus
            if GCC in clubs and h.doc.id in (100, 104, 109):
                delta += 1.0 * bonus
        if S2B in clubs and is_intro_q and MULTIMEDIA in clubs and h.doc.id in (81, 82, 83, 85):
            delta += 5.0 * bonus
        if GCC in clubs and is_intro_q and h.doc.id in (99, 115) and "slogan" in folded_q:
            delta += 4.0 * bonus
        if (
            is_contact_q
            and h.doc.topic == "Liên hệ"
            and (not clubs or S2B in clubs or h.doc.club in clubs)
        ):
            delta += 3.5 * bonus
        if h.doc.id == 1 and not is_intro_q:
            delta -= 3.5 * bonus
        if h.doc.id == 60 and not is_huawei_q:
            delta -= 3.5 * bonus
        adjusted.append(ScoredDoc(h.doc, h.score + delta))

    ranked = sorted(adjusted, key=lambda h: -h.score)
    multi_club = len(clubs - {S2B}) >= 3 or S2B in clubs
    if not multi_club:
        return ranked

    top5, deferred = [], []
    bcn_counts: dict[str, int] = {}
    for h in ranked:
        if len(top5) < 5:
            if h.doc.topic == "Ban chủ nhiệm":
                c = bcn_counts.get(h.doc.club, 0)
                if c >= 2:
                    deferred.append(h)
                    continue
                bcn_counts[h.doc.club] = c + 1
            top5.append(h)
        else:
            deferred.append(h)
    return top5 + deferred
