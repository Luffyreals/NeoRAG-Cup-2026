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

## Ablation retrieval (Ngày 2)

Chạy ở chế độ chỉ retrieval (`[generation] enabled = false`), config trong `configs/experiments/`. Mỗi dòng chỉ đổi một thứ so với dòng được ghi ở cột "So với". So sánh từng câu bằng `python scripts/compare_runs.py <gốc> <mới>`.

| Mã | Ngày | Thay đổi | So với | Recall@5 | MRR@5 | NDCG@5 | Hit@5 | Retrieval | Thắng / thua |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| exp01 | 03/11 | Chỉ dense bge-m3 (= baseline) | — | 0.7821 | 0.6925 | 0.6909 | 0.8443 | 0.7218 | |
| exp02 | 03/11 | Chỉ BM25 (underthesea, chưa bỏ dấu) | exp01 | 0.6783 | 0.5928 | 0.5755 | 0.7869 | 0.6155 | |
| exp03 | 03/11 | Hybrid dense + BM25, RRF k=60, trọng số 1:1 | exp01 | 0.7643 | 0.7176 | 0.6906 | 0.8525 | 0.7242 | 30 / 27 |
| exp04 | 03/11 | + kênh âm tiết bỏ dấu cho BM25 | exp03 | 0.8122 | 0.7643 | 0.7393 | 0.8852 | 0.7719 | |
| exp05 | 03/11 | + query rewrite (Haiku), gộp RRF câu gốc + câu viết lại | exp04 | 0.8572 | 0.7855 | 0.7631 | 0.9344 | 0.8019 | |
| exp05b | 03/11 | `dense_weight` = 2.0 | exp05 | 0.8422 | 0.7772 | 0.7559 | 0.9098 | 0.7918 | 11 / 15 |
| exp05b | 03/11 | `dense_weight` = 1.5 | exp05 | 0.8627 | 0.7791 | 0.7639 | 0.9262 | 0.8019 | 9 / 9 |
| **exp05b** | 03/11 | **`dense_weight` = 0.75** (chọn) | exp05 | 0.8654 | 0.8068 | 0.7808 | 0.9344 | **0.8177** | 15 / 4 |
| exp05b | 03/11 | `dense_weight` = 0.5 | exp05 | 0.8709 | 0.7948 | 0.7739 | 0.9426 | 0.8132 | 17 / 11 |

## Ghi chú từng thí nghiệm

- **exp00**: kiểm chứng refactor không đổi hành vi. Cache LLM và embedding trúng 100%, không tốn ngân sách.
- **exp05b (trọng số RRF)**: giả thuyết ban đầu là BM25 yếu hơn dense nên cần tăng trọng số dense; kết quả ngược lại. Tăng dense (1.5, 2.0) không lợi hoặc tụt, các câu thua chủ yếu là câu gõ không dấu (`di lien hoan voi clb...`, `clb multimedia co phong rieng...`) — nơi kênh bỏ dấu của BM25 là tín hiệu mạnh nhất. Giảm dense xuống 0.75 tăng 0.0158 với 15 câu thắng / 4 câu thua; 0.5 bắt đầu tụt lại (so với 0.75: 6 thắng / 10 thua), nên chọn 0.75 và không dò mịn hơn để tránh overfit tập train. Cần kiểm tra lại sau khi thêm reranker (exp06), vì reranker chấm lại thứ tự top-20.
