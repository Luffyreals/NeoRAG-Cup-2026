# Kịch bản demo chatbot

Tài liệu dùng cho phần Demo Product (UI/UX, độ trễ, trải nghiệm) và pitching.

## 1. Chạy chatbot

```bash
.venv\Scripts\activate
python app/server.py            # mở http://127.0.0.1:8000
```

- Dùng cùng pipeline và cấu hình `configs/demo.toml` với bản Streamlit (reranker top-10, int8).
- Đổi pipeline: `NEORAG_PIPELINE="module:Class"`; đổi cấu hình: `NEORAG_CHAT_CONFIG=...`.
- **Phương án dự phòng:** `python -m streamlit run app/chat.py` (giao diện cũ, cùng pipeline).

Lần đầu khởi động mất một lúc vì phải nạp model và xây index. Chỉ khi terminal in `Uvicorn running` thì mới mở trang.

## 2. Checklist 30 phút trước giờ pitching

- [ ] `git status` sạch, đúng nhánh đã nộp. Không sửa code trong ngày Private Test.
- [ ] `.env` có key, và key còn hạn mức.
- [ ] Bật server, đợi dòng `Uvicorn running`.
- [ ] Hỏi 1 câu bất kỳ để làm nóng (lần chạy đầu của reranker chậm hơn).
- [ ] Trình duyệt toàn màn hình, rộng ≥ 1280px để thấy đủ 3 cột. Zoom 100%, hoặc 110–125% nếu chiếu máy chiếu.
- [ ] Xóa các cuộc trò chuyện thử bằng nút 🗑, để demo bắt đầu từ màn chào.
- [ ] Có mạng ổn định (gateway LLM ở ngoài). Có sẵn hotspot điện thoại để dự phòng.
- [ ] Mở sẵn một tab bản Streamlit để chuyển sang ngay nếu cần.

## 3. Kịch bản demo (~4 phút)

Mỗi câu thể hiện một khả năng. Câu hỏi do mình tự đặt, không lấy từ bộ test.

| # | Câu hỏi | Khả năng muốn khoe | Chỉ vào đâu trên màn hình |
|---|---|---|---|
| 1 | *(bấm câu mẫu)* CLB Lập Trình PTIT thành lập ngày nào? | Số liệu chính xác | Câu trả lời "09/10/2011"; panel phải hiện **nguồn trước khi chữ chạy xong**; thanh hiệu năng |
| 2 | Ai là người sáng lập? | Hỏi nối tiếp | Dòng **↺ Hiểu là: "Ai là người sáng lập CLB Lập Trình PTIT?"**; bước "Hiểu ngữ cảnh" trong panel hiệu năng |
| 3 | lam sao de dang ky vao proptit | Gõ không dấu | Link `ttv.proptit.com` và mốc 20/08–14/09/2026 được chép đúng từ tài liệu |
| 4 | Tham gia ProPTIT có mất phí không? | Câu hỏi đời thường | Bấm chip **① Phí & quỹ**: panel làm sáng nguồn đó → bấm **Xem toàn văn** |
| 5 | MPC và GCC khác nhau thế nào? | So sánh 2 CLB | Chip nguồn có màu của **cả 2 CLB**. Chatbot nói rõ phần nào tài liệu **không có** |
| 6 | GCC đào tạo theo những nhánh nào? | Nối tiếp sang CLB khác | Nguồn chuyển hết sang GCC (màu xanh mint) |
| 7 | Ba CLB trong S2B liên quan với nhau thế nào? | Câu về cả cộng đồng | Nguồn lấy từ "Cộng đồng S2B" và nhiều CLB |
| 8 | Học phí của PTIT là bao nhiêu? | **Từ chối khi không có thông tin** | Bong bóng vàng **"Ngoài phạm vi tài liệu"**, panel không đưa ra nguồn nào |
| 9 | *(bấm chip nguồn của câu 1)* | Xem lại câu cũ | Panel quay về đúng nguồn và thời gian của câu 1 |
| 10 | *(F5 tải lại trang)* | Lịch sử được lưu | Cuộc trò chuyện còn nguyên, và có tên ở cột trái |

Nếu còn thời gian:
- Thu nhỏ cửa sổ (hoặc mở DevTools ở chế độ điện thoại) để khoe ngăn kéo nguồn và lịch sử.
- Bấm **Dừng** giữa lúc bot đang trả lời.

**Mời BGK tự đặt một câu.** Câu chưa có trong cache cho thấy độ trễ thật, và chứng minh câu trả lời không được chuẩn bị trước.

### Lời dẫn gợi ý

- **Mở đầu:** "Chatbot trả lời câu hỏi về 3 CLB của cộng đồng S2B, *chỉ* dựa trên 118 tài liệu. Mỗi câu trả lời đều có nguồn bấm vào xem được."
- **Ở câu 1:** "Panel bên phải hiện tài liệu ngay khi tìm xong, trước cả khi model viết xong. Người dùng thấy hệ thống đang dựa vào đâu."
- **Ở câu 8:** "Khi tài liệu không có thông tin, chatbot nói thẳng là không có, chứ không đoán."

## 4. Khi có sự cố

| Sự cố | Giao diện làm gì | Mình làm gì |
|---|---|---|
| Server bị tắt hoặc sập giữa phiên | Bong bóng lỗi có nút **Thử lại**, banner "Đang thử kết nối lại" | Chạy lại `python app/server.py`. Banner tự tắt, rồi bấm **Thử lại** |
| Mất mạng | Banner "Mất kết nối mạng" | Chuyển sang hotspot rồi bấm **Thử lại** |
| Gateway chậm hoặc treo | Sau 60s không có phản hồi thì tự dừng và báo lỗi | Bấm **Thử lại**, hoặc chuyển sang câu đã chạy trước (có cache) |
| Hết hạn mức key | Thông báo "Khóa API đã hết hạn mức…" | Đổi key trong `.env`, khởi động lại server |
| Giao diện web lỗi | | Chuyển sang tab Streamlit |

## 5. Câu hỏi BGK có thể hỏi

- **Sao không dùng WebSocket?** Luồng chỉ cần một chiều, server gửi xuống. SSE chạy trên HTTP thường, đơn giản hơn và dễ debug.
- **Sao không dùng React?** Giao diện chỉ có một trang, không cần bước build, server Python phục vụ trực tiếp. Ít thứ phải cài nên ít rủi ro lúc demo.
- **Bấm Dừng thì server có dừng gọi LLM không?** Có. Trình duyệt ngắt kết nối, server đóng generator và nhả khóa pipeline, nên câu sau vẫn chạy bình thường.
- **Lịch sử lưu ở đâu?** Trong trình duyệt (localStorage). Không cần đăng nhập, server không giữ dữ liệu người dùng. Nhược điểm: đổi máy là mất.
- **Làm sao biết không bịa?** Mỗi câu có chip nguồn, mở ra đúng đoạn tài liệu đã đưa vào prompt. Câu ngoài phạm vi thì từ chối và không hiện nguồn.
- **Sao câu này nhanh, câu kia chậm?** Các lần gọi LLM được cache theo nội dung. Câu đã hỏi rồi chỉ còn bước reranker (khoảng 1–4s). Câu mới phải gọi LLM để viết lại câu hỏi và sinh câu trả lời (khoảng 8–13s). Panel hiệu năng hiện rõ từng bước.
- **Nhiều người dùng cùng lúc?** Hiện các câu hỏi chạy lần lượt vì model reranker dùng chung. Muốn mở rộng thì chạy nhiều worker, hoặc tách reranker thành service riêng.

## 6. Số đo (chạy thử 05/11/2026, máy cá nhân, CPU)

| Loại câu | Tổng | Chữ đầu tiên | Bước tốn thời gian nhất |
|---|---|---|---|
| Đã có trong cache LLM | 1.3–4.3s | ≈ tổng | Xếp hạng lại 1.3–4.2s |
| Câu mới | 8.2–13.2s | ≈ tổng − 0.4s | Xếp hạng lại 3–4s, viết lại câu hỏi 2–3s, sinh 2–4s |
| Câu nối tiếp mới | 11–13s | | Có thêm bước hiểu ngữ cảnh (gọi LLM, ~2.5s) |
| Câu dài về cả cộng đồng | 19s | | Reranker 8.4s (CPU đang bận) |

Nhận xét:
- **Chữ đầu tiên đến gần như cùng lúc với cả câu.** Gateway trả câu trả lời gần như một lần, nên hiệu ứng chữ chạy dần chỉ kéo dài khoảng 0.3–0.5s. Thời gian chờ chủ yếu nằm ở các bước trước khi sinh.
- **Reranker trên CPU dao động 1.3–9.6s** tùy máy đang bận gì. Lúc demo nên tắt các chương trình nặng.
- **Câu nối tiếp gọi LLM 2 lần liên tiếp** (hiểu ngữ cảnh, rồi viết lại câu hỏi). Đây là chỗ có thể rút ngắn khoảng 2–3s.
