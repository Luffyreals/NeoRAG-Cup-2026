"""So sánh hai lần chấm: chênh lệch metric tổng và các câu hỏi thắng/thua.

Đọc kết quả `neorag evaluate` đã có sẵn, không gọi API:

    python scripts/compare_runs.py exp05_rewrite exp05b_dense15
    python scripts/compare_runs.py outputs/eval_exp05_rewrite_pred outputs/eval_exp05b_dense15_pred

Mỗi tên được hiểu là `outputs/eval_<tên>_pred` nếu không phải đường dẫn thư mục có sẵn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neorag.data import DATA_DIR, load_queries, read_jsonl

# Metric tính điểm (configs/eval.toml, rank_k = 5) và vài metric tham khảo
REPORT_METRICS = ["recall@5", "mrr@5", "ndcg@5", "hit@5", "recall@7", "string_presence", "rouge_l"]
RETRIEVAL_KEYS = ["recall@5", "mrr@5", "ndcg@5"]
EPS = 1e-9


def eval_dir(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path
    path = Path("outputs") / f"eval_{name}_pred"
    if not path.is_dir():
        raise SystemExit(f"Không tìm thấy {path}. Chạy neorag evaluate trước.")
    return path


def per_query_score(row: dict, keys: list[str]) -> float:
    return sum(row.get(k, 0.0) for k in keys) / len(keys)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # tiếng Việt trên console Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", help="lần chạy gốc")
    parser.add_argument("new", help="lần chạy cần so sánh")
    parser.add_argument("--split", default="train", help="bộ câu hỏi để in nội dung câu")
    parser.add_argument("--show", type=int, default=5, help="số ví dụ thắng/thua in ra")
    parser.add_argument(
        "--metrics",
        default=",".join(RETRIEVAL_KEYS),
        help="metric (trong per_query.jsonl) dùng để xếp thắng/thua, mặc định retrieval@5",
    )
    args = parser.parse_args(argv)

    base_dir, new_dir = eval_dir(args.base), eval_dir(args.new)
    base_rep = json.loads((base_dir / "report.json").read_text(encoding="utf-8"))
    new_rep = json.loads((new_dir / "report.json").read_text(encoding="utf-8"))

    print(f"{'metric':<18}{'base':>9}{'new':>9}{'delta':>9}")
    rows = [(k, base_rep["metrics"].get(k), new_rep["metrics"].get(k)) for k in REPORT_METRICS]
    rows.append(("retrieval_score", base_rep["retrieval_score"], new_rep["retrieval_score"]))
    for key, b, n in rows:
        if b is None or n is None:
            continue
        print(f"{key:<18}{b:>9.4f}{n:>9.4f}{n - b:>+9.4f}")

    keys = args.metrics.split(",")
    base_q = {r["id"]: r for r in read_jsonl(base_dir / "per_query.jsonl")}
    new_q = {r["id"]: r for r in read_jsonl(new_dir / "per_query.jsonl")}
    texts = {q.id: q.query for q in load_queries(DATA_DIR / args.split / "questions.jsonl")}

    diffs = []
    for qid in base_q.keys() & new_q.keys():
        if not all(k in base_q[qid] for k in keys):
            continue  # ví dụ câu không có đáp án không có metric lexical
        d = per_query_score(new_q[qid], keys) - per_query_score(base_q[qid], keys)
        diffs.append((d, qid))
    wins = sorted((x for x in diffs if x[0] > EPS), reverse=True)
    losses = sorted(x for x in diffs if x[0] < -EPS)
    print(
        f"\nTheo mean({', '.join(keys)}): thắng {len(wins)}, thua {len(losses)}, "
        f"hòa {len(diffs) - len(wins) - len(losses)} / {len(diffs)} câu"
    )
    for title, items in (("Thắng nhiều nhất", wins), ("Thua nhiều nhất", losses)):
        if items:
            print(f"\n{title}:")
        for d, qid in items[: args.show]:
            print(f"  {d:+.3f}  {qid}  {texts.get(qid, '')[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
