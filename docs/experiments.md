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
| exp05c | 03/11 | Chỉ rewrite câu không dấu (`only_unaccented`, accent_ratio < 0.2), `dense_weight` = 1.0 | exp05 | | | | | 0.8028 | 21 / 18 |
| exp05c | 03/11 | Chỉ rewrite câu không dấu, `dense_weight` = 0.75 (không chọn) | exp05b 0.75 | 0.8572 | 0.7892 | 0.7655 | 0.9426 | 0.8040 | 14 / 19 |
| **exp06** | 03/11 | **+ reranker local `bge-reranker-v2-m3`** (top-20, max_length 512, CPU), `dense_weight` = 1.0 | exp05 | 0.9208 | 0.8407 | 0.8374 | 0.9754 | **0.8663** | 41 / 20 |
| exp06b | 03/11 | + reranker, `dense_weight` = 0.75 | exp06 | 0.9126 | 0.8352 | 0.8313 | 0.9672 | 0.8597 | 2 / 1 |
| exp06c | 04/11 | Reranker `top_n` = 10, `num_threads` = 4 (latency median 4.2 s, p95 7.8 s) | exp06 | 0.9112 | 0.8370 | 0.8321 | 0.9590 | 0.8601 | 11 / 6 |
| **exp07** | 04/11 | **+ ưu tiên CLB** (boost mềm 0.1 × khoảng điểm sau rerank), top-20, 4 luồng (latency median 7.6 s) | exp06 | 0.9290 | 0.8643 | 0.8572 | 0.9836 | **0.8835** | 5 / 0 |

## Ghi chú từng thí nghiệm

- **exp00**: kiểm chứng refactor không đổi hành vi. Cache LLM và embedding trúng 100%, không tốn ngân sách.
- **exp05b (trọng số RRF)**: giả thuyết ban đầu là BM25 yếu hơn dense nên cần tăng trọng số dense; kết quả ngược lại. Tăng dense (1.5, 2.0) không lợi hoặc tụt, các câu thua chủ yếu là câu gõ không dấu (`di lien hoan voi clb...`, `clb multimedia co phong rieng...`) — nơi kênh bỏ dấu của BM25 là tín hiệu mạnh nhất. Giảm dense xuống 0.75 tăng 0.0158 với 15 câu thắng / 4 câu thua; 0.5 bắt đầu tụt lại (so với 0.75: 6 thắng / 10 thua), nên chọn 0.75 và không dò mịn hơn để tránh overfit tập train. Cần kiểm tra lại sau khi thêm reranker (exp06), vì reranker chấm lại thứ tự top-20.
- **exp05c (rewrite có điều kiện)**: phân tích exp04 → exp05 cho thấy rewrite chỉ có lợi với 26 câu không dấu (+0.145/câu) còn 96 câu có dấu thì ≈ 0 (18 thắng / 21 thua), nên thử chỉ viết lại câu không dấu để giảm latency. Với `dense_weight` = 1.0 đúng như dự đoán (+0.0009). Nhưng trên cấu hình tốt nhất (`dense_weight` = 0.75) lại tụt 0.0137: khi BM25 có trọng số lớn hơn, các câu con do rewrite tách ra (câu nhiều ý, so sánh nhiều CLB) giúp cả câu có dấu, ví dụ `mình muốn gửi mail hỏi cả 3 clb trong s2b...` (−0.68). Giữ rewrite cho mọi câu; tính năng vẫn còn trong config (mặc định tắt). Bài học: hai thay đổi tương tác với nhau, phải đo lại trên cấu hình mới nhất. Lưu ý: Public Test có 20/42 câu không dấu (train chỉ 26/126), nên rewrite càng quan trọng.
- **exp06 / exp06b (reranker)**: bước tăng lớn nhất của cả ablation (+0.064 so với exp05). Latency median 10.8–11.0 s/câu trên CPU (máy không có GPU; rewrite và embedding đều trúng cache nên gần như toàn bộ là thời gian rerank 20 cặp × 512 token). Với reranker, trọng số RRF hầu như không còn tác dụng: 0.75 so với 1.0 chỉ khác 3/122 câu, chênh lệch đến từ một câu (train-0102) rơi khỏi top-20 ứng viên. Đúng dự đoán là reranker xếp lại thứ tự nên trọng số RRF chỉ còn ảnh hưởng tới tập ứng viên. Giữ `dense_weight` = 1.0 (mặc định, ít tham số tinh chỉnh hơn).
- **exp06c (giảm latency reranker)**: đo riêng bước rerank: gần như toàn bộ latency (rewrite, dense, BM25 trúng cache chỉ 0.01–0.03 s). Máy i7-12800H (6 P-core + 8 E-core): PyTorch mặc định 14 luồng chậm hơn 4 luồng (top-20: ~6.4–11 s so với 6.2 s; top-10: 3.9 s), vì phần việc chia cho E-core làm P-core phải chờ. bf16 chậm gấp 10 lần (CPU không hỗ trợ bf16), `max_length` 256 vô ích (doc dài nhất 179 token). top-10 + 4 luồng: latency median 11 s → 4.2 s, retrieval −0.0062 (mất các câu có doc đúng ở hạng 11–20 sau hybrid, ví dụ train-0102, train-0120). Quyết định: hai cấu hình — benchmark dùng top-20 (điểm cao nhất, latency không tính vào điểm benchmark), chatbot demo dùng top-10 + 4 luồng. Số luồng tối ưu phụ thuộc máy, phải đo lại trên máy demo.
- **exp07 (ưu tiên CLB)**: phân tích exp06 trước khi chạy: 29 câu sai doc hạng 1, chỉ 4 câu sai vì doc hạng 1 thuộc nhầm CLB (đều là câu hỏi về GCC/Multimedia nhưng doc ProPTIT lên đầu, do ProPTIT chiếm 78/118 doc); 6 câu có doc đúng ngoài CLB nhận diện được (có thể bị hại). Kết quả khớp dự đoán: 5 câu thắng (gồm cả 4 câu trên), 0 câu thua, +0.0172. Boost mềm không làm hại câu nào vì doc đúng ngoài CLB vẫn còn trong top-5. Cấu hình tốt nhất hiện tại.

## Generation (Ngày 3)

Retrieval giữ nguyên exp07 (0.8835). Chấm `--no-judge`: chỉ String Presence và ROUGE-L.

| ID | Ngày | Thay đổi | So với | String Presence | ROUGE-L | BLEU-4 | Độ dài trả lời (trung vị / max) | Markdown |
|---|---|---|---|---:|---:|---:|---:|---:|
| exp07_gen | 04/11 | exp07 + prompt baseline | — | 0.8428 | 0.4796 | 0.3040 | 392 / 1245 ký tự | 113/126 câu |
| **exp08** | 04/11 | **prompt v2**: 1-3 câu, không markdown, nguyên văn số liệu, từ chối một câu | exp07_gen | 0.8671 | **0.6316** | 0.4813 | 226 / 645 ký tự | 0 |

- **exp08 (prompt v2)**: ROUGE-L +0.152, String Presence +0.024, câu trả lời ngắn gần bằng đáp án chuẩn (trung vị ~200 ký tự). 11/11 câu không có đáp án đều mở đầu "Tài liệu hiện có không có thông tin về ...". Có 6 câu String Presence giảm vì trả lời quá gọn nên bỏ mất ý phụ có trong key facts (train-0023 bỏ `mpc.proptit.com`, train-0028 bỏ `AICOVIDVN-115M`, train-0102 bỏ "bất lợi"). Latency median 9.7 s → 13.9 s dù câu trả lời ngắn hơn: cả hai lần đều gọi LLM thật (không trúng cache), nên nhiều khả năng do tải gateway lúc chạy chứ chưa kết luận được.

## Kết quả nộp Public Test

| Lần | Ngày | Cấu hình | Score | Điểm xếp hạng (norm) | Ghi chú |
|---|---|---|---:|---:|---|
| baseline | — | `BaselinePipeline` (số BTC công bố) | 52.40 | 0 | Ngưỡng = 53.40 |
| #1 (33066) | 04/11 | `configs/pipeline.toml` = exp07 + generation prompt baseline | 77.99 | 64.56 | Retrieval đã tối ưu, generation chưa. Từ norm suy ra điểm cao nhất bảng xếp hạng lúc nộp ≈ 91.5 |
