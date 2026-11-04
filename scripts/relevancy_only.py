"""Chỉ chấm Response Relevancy (đúng hàm của bộ chấm) cho một file predictions.

Rẻ hơn `neorag evaluate` đầy đủ: mỗi câu một lần gọi judge sinh câu hỏi + một lần embedding,
không chấm Groundedness / Noise / Abstention. Dùng chung cache `.cache/judge.sqlite`.

    python scripts/relevancy_only.py outputs/exp08_prompt_v2_pred.jsonl
    python scripts/relevancy_only.py outputs/exp11_prompt_v3_pred.jsonl --split train
"""

from __future__ import annotations

import argparse
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

from neorag.config import load_config
from neorag.data import DATA_DIR, load_predictions, load_queries
from neorag.llm import LLMClient
from neorag.metrics.judge import Judge
from neorag.pipelines.baseline import GatewayEmbedder
from neorag.pipelines.vietnamese import accent_ratio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions")
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)

    jcfg = load_config()["judge"]
    llm = LLMClient.for_judge(cache_path=jcfg["cache_path"])
    judge = Judge(
        llm,
        chat_model=jcfg["chat_model"],
        embedder=GatewayEmbedder(llm, jcfg["embedding_model"]),
        temperature=jcfg.get("temperature"),
        relevancy_questions=jcfg["relevancy_questions"],
    )
    gold = {q.id: q for q in load_queries(DATA_DIR / args.split / "questions.jsonl")}
    preds = {p.id: p for p in load_predictions(args.predictions)}
    # như bộ chấm: chỉ chấm câu có đáp án
    ids = [i for i, q in gold.items() if q.answerable and i in preds]

    with ThreadPoolExecutor(max_workers=jcfg.get("max_workers", 4)) as pool:
        scores = dict(
            zip(
                ids,
                pool.map(lambda i: judge.response_relevancy(gold[i].query, preds[i].answer), ids),
                strict=True,
            )
        )

    groups = {
        "có dấu": [s for i, s in scores.items() if accent_ratio(gold[i].query) >= 0.2],
        "không dấu": [s for i, s in scores.items() if accent_ratio(gold[i].query) < 0.2],
    }
    print(f"response_relevancy  {statistics.mean(scores.values()):.4f}  ({len(scores)} câu)")
    for name, vals in groups.items():
        if vals:
            print(f"  {name:10s}  {statistics.mean(vals):.4f}  ({len(vals)} câu)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
