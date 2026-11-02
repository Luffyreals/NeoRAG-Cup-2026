# Phân tích lỗi baseline (train)

Tập train có 126 câu: 115 câu có đáp án, 11 câu không có đáp án. Nguồn số liệu: `outputs/eval_train_pred/per_query.jsonl`, `outputs/eval_train_pred/report.json` (chạy có judge), `outputs/train_pred.jsonl`, `data/train/questions.jsonl`, `data/corpus.jsonl`.

Baseline: bge-m3 cosine top-10, đưa top-5 vào prompt đơn giản, gọi `claude-haiku-5-5` một lần. Điểm tổng 0.7364 (retrieval 0.7218, generation 0.7510).

| Metric | Giá trị |
|---|---:|
| Recall@5 | 0.7821 |
| MRR@5 | 0.6925 |
| NDCG@5 | 0.6909 |
| String Presence | 0.7391 |
| ROUGE-L | 0.4197 |
| Groundedness | 0.9341 |
| Response Relevancy | 0.6823 |
| Noise Sensitivity ↓ | 0.2690 |
| Abstention | 1.0000 |

## 1. Điểm mất theo thành phần

Mỗi metric retrieval có trọng số 1/6, mỗi metric generation có trọng số 1/12. Điểm mất = trọng số × (1 − giá trị).

| Thành phần | Giá trị | Điểm mất | % tổng mất |
|---|---:|---:|---:|
| NDCG@5 | 0.691 | **0.0518** | 20% |
| MRR@5 | 0.693 | **0.0513** | 19% |
| ROUGE-L | 0.420 | **0.0484** | 18% |
| Recall@5 | 0.782 | 0.0363 | 14% |
| Response Relevancy | 0.682 | 0.0265 | 10% |
| 1 − Noise | 0.731 | 0.0224 | 8% |
| String Presence | 0.739 | 0.0217 | 8% |
| Groundedness | 0.934 | 0.0055 | 2% |
| Abstention | 1.000 | 0.0000 | 0% |
| **Tổng** | | **0.2639** | |

Retrieval mất 0.139 (53%), riêng MRR + NDCG (gold không ở hạng 1) mất 0.103. Generation mất 0.125 (47%), chủ yếu ROUGE-L. Groundedness và Abstention đã gần trần.

## 2. Nhóm lỗi

Tính trên 115 câu có đáp án; một câu có thể thuộc nhiều nhóm.

### a. Gold doc không có trong top-5: 33 câu (28.7%)

- **Mất hoàn toàn (Recall@5 = 0): 19 câu.** 14/19 câu thuộc type `không dấu/viết tắt`, 2 `so sánh`, 2 `diễn đạt khác`, 1 `nhiều đoạn`.
- **Thiếu một phần (0 < Recall@5 < 1): 14 câu**, chủ yếu câu nhiều gold doc (6 `nhiều đoạn`, 4 `so sánh`).

Ví dụ:
- `train-0011` "hay thay chu gcc tren may bai dang o hv, gcc la viet tat...": gold 97, 99 (Giới thiệu chung GCC); top-5 là 20, 115, 51, 118, 60. Query không dấu và viết tắt, embedding lệch hẳn. Baseline trả lời "không ghi cụm tiếng Anh đầy đủ".
- `train-0066` "minh la nguoi dung lop o clb lt ptit... bao truoc bao lau": gold 72 (Nội quy); top-5 là 118, 96, 73, 98, 62. "lt" = lập trình; doc 73 cùng chủ đề nhưng không phải gold.
- `train-0017` "3 CLB thuộc S2B, CLB nào lâu đời nhất, trẻ nhất?": gold 1, 85, 97; top-5 toàn doc "S2B chung" (79, 81, 82, 80, 84). Câu so sánh cần doc "Giới thiệu chung" của từng CLB.

### b. Gold có trong top-5 nhưng không ở hạng 1: 30 câu (26.1%)

Hạng của gold đầu tiên: hạng 2 có 21 câu, hạng 3 có 6, hạng 4 có 1, hạng 5 có 2. Chỉ 66/115 câu có MRR = 1.

- `train-0027` "Lịch tuyển thành viên năm 2026 của GCC": gold 116 ở hạng 2, hạng 1 là doc 108 (tuyển thành viên năm khác).
- `train-0022` "gcc dao tao thanh vien theo may vi tri": gold 100 ở hạng 2, bị doc 115 (Sự kiện nội bộ) chiếm hạng 1.
- `train-0023` "nộp đơn Multimedia ngày 5/9/2026, hạn 31/8": gold 91 ở hạng 2, hạng 1 là doc 6 (tuyển thành viên của CLB khác).

Nguyên nhân: nhiều doc cùng topic (Tuyển thành viên, Nội quy, Ban chủ nhiệm) của các CLB và các năm khác nhau; bi-encoder không phân biệt được CLB, năm, khóa.

### c. Lấy đúng doc nhưng thiếu key fact: 23 câu (20%)

String Presence < 1 ở 41 câu; 18 câu do không lấy được gold, 23 câu còn lại gold đã nằm trong top-5. Trong 28 key fact bị mất ở nhóm này, 18 có nguyên văn trong gold doc.

- **Diễn đạt lại thay vì chép nguyên văn**: `train-0041` viết "tuyệt đối không được gian lận" trong khi key fact là "tuyệt đối không gian lận" (SP = 0); `train-0126` thiếu "theo đuổi đam mê"; `train-0003` thiếu "hai tháng".
- **Câu nhiều ý trả lời thiếu ý**: `train-0098` (SP = 0), `train-0095` chỉ nêu 25 của ProPTIT mà thiếu "12" (CLB tuyển ít nhất), `train-0059` trả lời "C#" thay vì "Java"/"ProJApp" vì nhánh đúng nằm ngoài top-5.
- **Từ chối dù doc có thông tin**: `train-0053` trả lời "Tài liệu không cho biết...".

Markdown `**...**` không phải nguyên nhân chính làm mất key fact.

### d. Câu trả lời dài, lạc trọng tâm

| | Dự đoán | Gold |
|---|---:|---:|
| Số ký tự trung bình | 472 | 214 |
| Số từ trung bình | 97.3 | 43.6 |

- Câu dự đoán dài gấp 2.2 lần gold; 102/115 câu (89%) dùng markdown (bold, bullet).
- ROUGE-L < 0.4 ở 61 câu (53%); Relevancy < 0.6 ở 35 câu.
- Tương quan giữa tỉ lệ độ dài (dự đoán/gold) và ROUGE-L là **−0.66**: độ dài là nguyên nhân chính.
- Ví dụ: `train-0124` 784 vs 160 ký tự (ROUGE-L 0.10); `train-0080` 867 vs 235 (0.12); `train-0073` 712 vs 172 (0.10).
- Relevancy thấp nhất là các câu từ chối sai rồi liệt kê lan man: `train-0109` (0.37), `train-0011` (0.38), `train-0013` (0.39).

### e. Groundedness thấp, Noise Sensitivity cao

- Groundedness < 1 ở 29 câu, trung bình 0.934 (gần trần).
- Noise > 0 ở 58 câu (50%), noise ≥ 0.5 ở 30 câu. Noise phụ thuộc retrieval: trung bình 0.903 khi Recall@5 = 0, chỉ 0.144 khi Recall@5 > 0.
- Ví dụ: `train-0013`, `train-0015`, `train-0045`, `train-0057` có noise = 1.0: không lấy được gold, model nói "tài liệu không nói..." rồi liệt kê "một số thông tin có thể hữu ích" từ doc không liên quan. `train-0059` (groundedness 0.5) bịa "C#" từ doc sai.

### f. Abstention

- 11/11 câu không có đáp án đều đạt Abstention = 1.0 (ví dụ `train-0019`, `train-0020`, `train-0033`, `train-0049`), nhưng vẫn kèm thông tin liên quan, gây noise.
- Khoảng 14 câu có đáp án (12%) bị từ chối sai: một phần do không lấy được gold (`train-0034`, `0045`, `0057`, `0066`), một phần lấy đúng doc mà vẫn từ chối (`train-0053`, `0064`, `0087`, `0100`, `0125`). Abstention không phạt trường hợp này nhưng mất điểm ở SP, ROUGE-L, Relevancy, Noise.

## 3. Thống kê theo nhóm (115 câu có đáp án)

**Theo difficulty**

| difficulty | n | R@5 | MRR@5 | NDCG@5 | SP | ROUGE-L | Ground | Relev | Noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 1.000 | 0.950 | 0.963 | 0.808 | 0.687 | 1.000 | 0.811 | 0.000 |
| 2 | 62 | 0.745 | 0.642 | 0.656 | 0.707 | 0.398 | 0.933 | 0.657 | 0.314 |
| 3 | 43 | 0.762 | 0.693 | 0.664 | 0.769 | 0.389 | 0.920 | 0.689 | 0.267 |

**Theo club**

| club | n | R@5 | MRR@5 | NDCG@5 | SP | ROUGE-L | Ground | Relev | Noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLB Lập Trình PTIT | 52 | 0.716 | 0.600 | 0.615 | 0.644 | 0.407 | 0.923 | 0.684 | 0.325 |
| CLB Multimedia PTIT | 24 | 0.875 | 0.790 | 0.793 | 0.896 | 0.446 | 0.961 | 0.660 | 0.223 |
| GCC | 26 | 0.827 | 0.712 | 0.730 | 0.782 | 0.414 | 0.965 | 0.697 | 0.217 |
| Cộng đồng S2B | 2 | 1.000 | 1.000 | 1.000 | 0.833 | 0.659 | 0.900 | 0.819 | 0.100 |
| Nhiều CLB | 11 | 0.652 | 0.773 | 0.623 | 0.727 | 0.390 | 0.861 | 0.664 | 0.258 |

**Theo type**

| type | n | R@5 | MRR@5 | NDCG@5 | SP | ROUGE-L | Ground | Relev | Noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| không dấu/viết tắt | 26 | 0.436 | 0.359 | 0.360 | 0.462 | 0.369 | 0.904 | 0.477 | 0.498 |
| diễn đạt khác | 22 | 0.864 | 0.735 | 0.754 | 0.814 | 0.381 | 0.923 | 0.702 | 0.272 |
| nhiều đoạn | 20 | 0.812 | 0.712 | 0.697 | 0.800 | 0.400 | 0.935 | 0.710 | 0.226 |
| so sánh | 20 | 0.792 | 0.779 | 0.727 | 0.825 | 0.402 | 0.927 | 0.758 | 0.224 |
| thời gian | 12 | 1.000 | 0.861 | 0.897 | 0.917 | 0.468 | 0.963 | 0.732 | 0.138 |
| trực tiếp | 8 | 1.000 | 1.000 | 1.000 | 0.760 | 0.681 | 1.000 | 0.867 | 0.000 |
| thiếu dữ liệu | 5 | 0.900 | 0.700 | 0.730 | 0.700 | 0.437 | 1.000 | 0.786 | 0.270 |
| mâu thuẫn nguồn | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.490 | 0.900 | 0.808 | 0.000 |

Điểm yếu có hệ thống:
- **`không dấu/viết tắt`** (26 câu, 23%) thấp nhất ở mọi metric: Recall@5 0.436, Relevancy 0.477, Noise 0.498. Đây là nhóm đáng sửa nhất.
- **CLB Lập Trình PTIT** (52 câu) kém hai CLB còn lại ở Recall (0.716) và SP (0.644): nhiều doc na ná nhau và query hay viết tắt ("lt", "ptit", "proptit").
- `nhiều đoạn`, `so sánh`, `Nhiều CLB`: Recall khoảng 0.65–0.8, thiếu doc thứ hai.
- ROUGE-L chỉ 0.37–0.47 ở hầu hết các nhóm: vấn đề độ dài nằm ở tầng generation, không phụ thuộc loại câu.

## 4. Đề xuất cải tiến (xếp theo mức tăng kỳ vọng)

Mức tăng là ước lượng thô trên điểm benchmark (thang 0–1), các mục chồng lấn nên không cộng thẳng.

| # | Cải tiến | Nhóm lỗi | Tầng | Kỳ vọng |
|---:|---|---|---|---:|
| 1 | Query rewriting bằng `claude-haiku-5-5`: khôi phục dấu, mở rộng viết tắt (lt, gcc, mpc, hv, nk...), tách câu nhiều ý thành sub-query, nhận diện CLB | a, b | Retrieval | +0.02 – 0.04 |
| 2 | Reranker (cross-encoder `bge-reranker-v2-m3` local hoặc LLM rerank) trên top-10 + ưu tiên doc đúng CLB | b | Retrieval | +0.02 – 0.04 |
| 3 | Prompt ngắn: 1–2 câu, nhắc lại ý câu hỏi, không bullet, không kèm "thông tin có thể hữu ích" | d, e | Generation | +0.015 – 0.03 |
| 4 | Hybrid BM25 (tách từ tiếng Việt + bản bỏ dấu) + bge-m3, gộp bằng RRF | a, b | Retrieval | +0.015 – 0.03 |
| 5 | Prompt: chép nguyên văn số, ngày, tên, cụm từ khóa; trả lời đủ mọi ý được hỏi | c | Generation | +0.005 – 0.012 |
| 6 | Chỉ từ chối khi thật sự không có thông tin; từ chối ngắn, không liệt kê lan man | e, f | Generation | +0.005 – 0.01 |
| 7 | Câu so sánh / nhiều CLB: lấy doc riêng cho từng sub-query, đảm bảo đủ doc "Giới thiệu chung" mỗi CLB | a (một phần) | Retrieval | +0.005 – 0.01 |

Thứ tự gợi ý:
- **Ngày 2 (retrieval)**: hybrid BM25 + dense (RRF) → query rewriting/chuẩn hóa không dấu → reranker + ưu tiên CLB. Mỗi bước chạy `--no-judge` (chỉ tốn tiền ở bước rewrite/LLM rerank, được cache).
- **Ngày 3 (generation)**: mục 3, 5, 6 trong một prompt mới; kiểm tra lại 11 câu không có đáp án để giữ Abstention = 1.0.

## 5. Lưu ý chống overfit

- 126 câu train là mẫu nhỏ; các nhóm `mâu thuẫn nguồn` (2), `thiếu dữ liệu` (5), `Cộng đồng S2B` (2) không đủ để kết luận.
- Bảng viết tắt chỉ được là từ điển chuẩn hóa chung, không viết riêng cho từng câu (luật cấm hard-code).
- Key fact dạng cụm dài (ví dụ "chỉ áp dụng cho năm 2026") phụ thuộc cách gold viết; không tối ưu prompt chỉ để khớp vài key fact đó.
