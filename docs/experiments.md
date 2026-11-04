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
| exp09 | 04/11 | Rerank với từng câu viết lại (câu đầy đủ + câu hỏi con), gộp bằng **max** (latency median 16.2 s) | exp07 | 0.9344 | 0.8914 | 0.8718 | 0.9754 | **0.8992** | 16 / 8 |
| exp09 | 04/11 | Như trên, gộp bằng **RRF** (latency median 17.2 s) | exp07 | 0.9536 | 0.8684 | 0.8661 | 0.9836 | 0.8960 | 13 / 14 |
| exp09 | 04/11 | Như trên, gộp bằng **mix** = ½ điểm câu đầy đủ + ½ max câu hỏi con (latency median 16.6 s) | exp07 | 0.9399 | 0.8880 | 0.8704 | 0.9836 | 0.8994 | 11 / 6 |

## Ghi chú từng thí nghiệm

- **exp00**: kiểm chứng refactor không đổi hành vi. Cache LLM và embedding trúng 100%, không tốn ngân sách.
- **exp05b (trọng số RRF)**: giả thuyết ban đầu là BM25 yếu hơn dense nên cần tăng trọng số dense; kết quả ngược lại. Tăng dense (1.5, 2.0) không lợi hoặc tụt, các câu thua chủ yếu là câu gõ không dấu (`di lien hoan voi clb...`, `clb multimedia co phong rieng...`) — nơi kênh bỏ dấu của BM25 là tín hiệu mạnh nhất. Giảm dense xuống 0.75 tăng 0.0158 với 15 câu thắng / 4 câu thua; 0.5 bắt đầu tụt lại (so với 0.75: 6 thắng / 10 thua), nên chọn 0.75 và không dò mịn hơn để tránh overfit tập train. Cần kiểm tra lại sau khi thêm reranker (exp06), vì reranker chấm lại thứ tự top-20.
- **exp05c (rewrite có điều kiện)**: phân tích exp04 → exp05 cho thấy rewrite chỉ có lợi với 26 câu không dấu (+0.145/câu) còn 96 câu có dấu thì ≈ 0 (18 thắng / 21 thua), nên thử chỉ viết lại câu không dấu để giảm latency. Với `dense_weight` = 1.0 đúng như dự đoán (+0.0009). Nhưng trên cấu hình tốt nhất (`dense_weight` = 0.75) lại tụt 0.0137: khi BM25 có trọng số lớn hơn, các câu con do rewrite tách ra (câu nhiều ý, so sánh nhiều CLB) giúp cả câu có dấu, ví dụ `mình muốn gửi mail hỏi cả 3 clb trong s2b...` (−0.68). Giữ rewrite cho mọi câu; tính năng vẫn còn trong config (mặc định tắt). Bài học: hai thay đổi tương tác với nhau, phải đo lại trên cấu hình mới nhất. Lưu ý: Public Test có 20/42 câu không dấu (train chỉ 26/126), nên rewrite càng quan trọng.
- **exp06 / exp06b (reranker)**: bước tăng lớn nhất của cả ablation (+0.064 so với exp05). Latency median 10.8–11.0 s/câu trên CPU (máy không có GPU; rewrite và embedding đều trúng cache nên gần như toàn bộ là thời gian rerank 20 cặp × 512 token). Với reranker, trọng số RRF hầu như không còn tác dụng: 0.75 so với 1.0 chỉ khác 3/122 câu, chênh lệch đến từ một câu (train-0102) rơi khỏi top-20 ứng viên. Đúng dự đoán là reranker xếp lại thứ tự nên trọng số RRF chỉ còn ảnh hưởng tới tập ứng viên. Giữ `dense_weight` = 1.0 (mặc định, ít tham số tinh chỉnh hơn).
- **exp06c (giảm latency reranker)**: đo riêng bước rerank: gần như toàn bộ latency (rewrite, dense, BM25 trúng cache chỉ 0.01–0.03 s). Máy i7-12800H (6 P-core + 8 E-core): PyTorch mặc định 14 luồng chậm hơn 4 luồng (top-20: ~6.4–11 s so với 6.2 s; top-10: 3.9 s), vì phần việc chia cho E-core làm P-core phải chờ. bf16 chậm gấp 10 lần (CPU không hỗ trợ bf16), `max_length` 256 vô ích (doc dài nhất 179 token). top-10 + 4 luồng: latency median 11 s → 4.2 s, retrieval −0.0062 (mất các câu có doc đúng ở hạng 11–20 sau hybrid, ví dụ train-0102, train-0120). Quyết định: hai cấu hình — benchmark dùng top-20 (điểm cao nhất, latency không tính vào điểm benchmark), chatbot demo dùng top-10 + 4 luồng. Số luồng tối ưu phụ thuộc máy, phải đo lại trên máy demo.
- **exp07 (ưu tiên CLB)**: phân tích exp06 trước khi chạy: 29 câu sai doc hạng 1, chỉ 4 câu sai vì doc hạng 1 thuộc nhầm CLB (đều là câu hỏi về GCC/Multimedia nhưng doc ProPTIT lên đầu, do ProPTIT chiếm 78/118 doc); 6 câu có doc đúng ngoài CLB nhận diện được (có thể bị hại). Kết quả khớp dự đoán: 5 câu thắng (gồm cả 4 câu trên), 0 câu thua, +0.0172. Boost mềm không làm hại câu nào vì doc đúng ngoài CLB vẫn còn trong top-5. Cấu hình tốt nhất hiện tại.
- **exp09 (rerank theo câu hỏi con)**: phân tích Public #2 cho thấy retrieval Public (0.80) thấp hơn train (0.88), Recall@5 tụt nhiều nhất (−0.11), và 34/42 câu Public được rewrite tách thành ≥ 3 câu (train trung bình 2.2). Trên train, câu có 3+ doc đúng chỉ đạt 0.640. Giả thuyết: reranker chỉ chấm bằng câu viết lại đầy đủ nên doc trả lời ý phụ bị đẩy khỏi top-5. Kết quả đúng hướng dự đoán: so sánh 0.802 → 0.891 (max) / 0.863 (RRF), nhiều đoạn 0.847 → 0.867 / 0.881, câu 3+ doc đúng 0.640 → 0.768 / 0.740, câu 2 doc 0.863 → 0.906 / 0.894. Nhưng câu chỉ có 1 doc đúng tụt 0.942 → 0.917 / 0.925 (diễn đạt khác 0.912 → 0.879 / 0.884): rewrite vẫn tách câu một ý thành câu con, câu con khớp mạnh một doc khác và đẩy doc đúng xuống, ví dụ train-0102 (−1.0 với max). RRF tăng Recall@5 nhiều hơn (+0.025 so với +0.005) nhưng thắng/thua chỉ 13/14; max tăng MRR (+0.027) và thắng/thua 16/8. Latency gấp đôi exp07 vì chấm 20 ứng viên × số câu.
- **exp09 mix**: nhằm giữ câu một ý mà vẫn lợi cho câu nhiều ý. Tổng điểm bằng max (0.8994 so với 0.8992; so với max 8 thắng / 13 thua). Đúng là giữ được câu 1 doc đúng (0.935, max 0.917, exp07 0.942) và diễn đạt khác (0.913 so với 0.879), train-0102 hồi lại +0.56. Nhưng mất bớt phần lợi ở câu nhiều ý: so sánh 0.858 (max 0.891), câu 3+ doc đúng 0.705 (max 0.768). Hai cách đánh đổi ngược nhau, train không phân định được; cấu hình nào tốt hơn phụ thuộc tỉ lệ câu nhiều ý trong tập test. Retrieval trên log chấm Public là xác định (không qua judge), nên có thể so trực tiếp hai cấu hình trên Public.

## Generation (Ngày 3)

Retrieval giữ nguyên exp07 (0.8835). Chấm `--no-judge`: chỉ String Presence và ROUGE-L.

| ID | Ngày | Thay đổi | So với | String Presence | ROUGE-L | BLEU-4 | Độ dài trả lời (trung vị / max) | Markdown |
|---|---|---|---|---:|---:|---:|---:|---:|
| exp07_gen | 04/11 | exp07 + prompt baseline | — | 0.8428 | 0.4796 | 0.3040 | 392 / 1245 ký tự | 113/126 câu |
| **exp08** | 04/11 | **prompt v2**: 1-3 câu, không markdown, nguyên văn số liệu, từ chối một câu | exp07_gen | 0.8671 | **0.6316** | 0.4813 | 226 / 645 ký tự | 0 |

- **exp08 (prompt v2)**: ROUGE-L +0.152, String Presence +0.024, câu trả lời ngắn gần bằng đáp án chuẩn (trung vị ~200 ký tự). 11/11 câu không có đáp án đều mở đầu "Tài liệu hiện có không có thông tin về ...". Có 6 câu String Presence giảm vì trả lời quá gọn nên bỏ mất ý phụ có trong key facts (train-0023 bỏ `mpc.proptit.com`, train-0028 bỏ `AICOVIDVN-115M`, train-0102 bỏ "bất lợi"). Latency median 9.7 s → 13.9 s dù câu trả lời ngắn hơn: cả hai lần đều gọi LLM thật (không trúng cache), nên nhiều khả năng do tải gateway lúc chạy chứ chưa kết luận được.
- **exp08 chấm đủ judge** (chỉ chấm, không sinh lại): Groundedness 0.9652, Response Relevancy 0.6821, Noise Sensitivity 0.0841, Abstention 1.0 → generation **0.8437**, benchmark **0.8636** (baseline Ngày 1: generation 0.7510, benchmark 0.7364; khi đó Groundedness 0.9341, Relevancy 0.6823, Noise 0.2690). Noise giảm mạnh nhờ câu trả lời ít câu, không lan sang doc nhiễu. Relevancy gần như không đổi và là metric thấp nhất: câu gõ không dấu chỉ đạt 0.488 (26 câu) so với 0.739 ở câu có dấu (89 câu), vì judge so embedding của *câu hỏi gốc không dấu* với các câu hỏi có dấu sinh ngược từ câu trả lời; phần này câu trả lời không thay đổi được. Câu có dấu thì lỗi chính là mở đầu bằng giới thiệu chung chung thay vì trả lời ngay (ví dụ câu "vào clb mm hợp ban nào" mở bằng năm thành lập MPC).

## Kết quả nộp Public Test

| Lần | Ngày | Cấu hình | Score | Điểm xếp hạng (norm) | Ghi chú |
|---|---|---|---:|---:|---|
| baseline | — | `BaselinePipeline` (số BTC công bố) | 52.40 | 0 | Ngưỡng = 53.40 |
| #1 (33066) | 04/11 | `configs/pipeline.toml` = exp07 + generation prompt baseline | 77.99 | 64.56 | Retrieval đã tối ưu, generation chưa. Từ norm suy ra điểm cao nhất bảng xếp hạng lúc nộp ≈ 91.5 |
| #2 (33079) | 04/11 | `configs/pipeline.toml` = exp08 (exp07 + prompt v2) | 80.10 | 67.05 | +2.11 so với #1, chỉ đổi prompt (retrieval giống #1 ở 42/42 câu). Câu trả lời trung vị 452 → 258 ký tự, 0 markdown, 5 câu từ chối. Điểm cao nhất bảng lúc nộp ≈ 93.2 |
| #3 (33096) | 04/11 | exp10 max = exp08 + rerank theo câu hỏi con, gộp **max** | 86.16 | **82.27** | Retrieval 80.09 → **88.55** (Recall@5 81.75 → 89.52, MRR@5 81.63 → 90.28). Generation 83.77. Cấu hình chọn |
| #4 (33097) | 04/11 | exp10 mix = exp08 + rerank theo câu hỏi con, gộp **mix** | 84.69 | 78.58 | Retrieval 86.15, generation 83.23. Đúng dự đoán: Public nhiều câu nhiều ý nên max hơn mix |

Chi tiết điểm (log chấm của trang thi) so với train (exp08, chấm judge):

| Tập | Recall@5 | MRR@5 | NDCG@5 | Retrieval | String Presence | ROUGE-L | Groundedness | Relevancy | 1 − Noise | Abstention | Generation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train (exp08) | 0.9290 | 0.8643 | 0.8572 | 0.8835 | 0.8671 | 0.6316 | 0.9652 | 0.6821 | 0.9159 | 1.0000 | 0.8437 |
| Public #1 | 0.8175 | 0.8163 | 0.7689 | 0.8009 | 0.8120 | 0.4410 | 0.9240 | 0.5896 | 0.7876 | 1.0000 | 0.7590 |
| Public #2 | 0.8175 | 0.8163 | 0.7689 | **0.8009** | 0.8291 | 0.5807 | 0.9316 | **0.6153** | 0.8504 | 1.0000 | 0.8012 |
| Public #3 (max) | 0.8952 | 0.9028 | 0.8585 | **0.8855** | 0.8761 | 0.5978 | 1.0000 | 0.5996 | 0.9530 | 1.0000 | 0.8377 |
| Public #4 (mix) | 0.8635 | 0.8897 | 0.8313 | 0.8615 | 0.8889 | 0.6123 | 0.9701 | 0.5996 | 0.9231 | 1.0000 | 0.8323 |

#1 → #2 (chỉ đổi prompt): ROUGE-L +0.140, 1 − Noise +0.063, Relevancy +0.026, String Presence +0.017, Groundedness +0.008; generation +0.042, khớp với hướng tăng trên train. Retrieval trên Public thấp hơn train 0.083 (Recall@5 −0.11), lớn hơn mức chênh của generation (−0.043). Không phải do câu không dấu: trên train, nhóm câu không dấu có retrieval 0.896 (26 câu), còn cao hơn nhóm có dấu 0.880 (96 câu). Nhiều khả năng do cấu hình được chọn trên train (overfit) và câu hỏi Public khác phân bố. Mỗi 0.01 retrieval = 0.5 điểm tổng, còn mỗi 0.01 của một metric generation chỉ = 0.083 điểm, nên retrieval là đòn bẩy lớn hơn.


#2 → #3 (chỉ đổi rerank): retrieval +0.085 trên Public, gấp 5 lần mức tăng trên train (+0.016), khớp phân tích là Public có tỉ lệ câu nhiều ý cao hơn train. Max hơn mix trên Public (retrieval 0.8855 so với 0.8615) dù hai cách hòa trên train. Retrieval Public giờ ngang train (0.886 so với 0.899). Response Relevancy (0.60) thành metric thấp nhất.