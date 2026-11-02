# Bảng thí nghiệm (ablation)

Mọi số đo trên tập **train** (126 câu, 11 câu không có đáp án), chấm bằng bộ chấm chính thức (`configs/eval.toml`).
Mỗi thí nghiệm chạy:

```bash
neorag run train --no-judge --pipeline neorag.pipelines.my_pipeline:MyPipeline --out outputs/expNN_<tên>_pred.jsonl
```

Chỉ chạy judge đầy đủ (bỏ `--no-judge`) cho thí nghiệm đáng giữ, để tiết kiệm ngân sách API.

Công thức: `retrieval = mean(Recall@5, MRR@5, NDCG@5)`, `generation = mean(SP, ROUGE-L, Groundedness, Relevancy, 1 − Noise, Abstention)`, `benchmark = 0.5 · retrieval + 0.5 · generation`.

| Mã | Ngày | Thay đổi | Recall@5 | MRR@5 | NDCG@5 | String Presence | ROUGE-L | Groundedness | Relevancy | Noise ↓ | Abstention | Retrieval | Generation | Benchmark | Latency median (ms) | Ghi chú |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 02/11 | `BaselinePipeline`: bge-m3 cosine top-10, top-5 vào prompt đơn giản | 0.7821 | 0.6925 | 0.6909 | 0.7391 | 0.4197 | 0.9341 | 0.6823 | 0.2690 | 1.0000 | 0.7218 | 0.7510 | 0.7364 | 2584 | Có judge. Latency là lần gọi API thật. |
| exp00 | 02/11 | `MyPipeline` cấu hình mặc định (`configs/pipeline.toml`), tách module retrieval/generation | 0.7821 | 0.6925 | 0.6909 | 0.7391 | 0.4197 | 0.9341† | 0.6823† | 0.2690† | 1.0000† | 0.7218 | 0.7510† | 0.7364† | 0.3‡ | Chạy `--no-judge`. `retrieved_ids` và câu trả lời trùng khớp 126/126 câu với baseline. |

† Không chạy judge; vì câu trả lời và `retrieved_ids` giống hệt baseline nên các metric judge bằng baseline.
‡ Toàn bộ lời gọi trúng cache `.cache/pipeline.sqlite`, latency không phản ánh tốc độ thật. Khi so latency, so với lần chạy không cache.

## Ghi chú từng thí nghiệm

- **exp00**: kiểm chứng refactor không đổi hành vi. Cache LLM và embedding trúng 100%, không tốn ngân sách.
