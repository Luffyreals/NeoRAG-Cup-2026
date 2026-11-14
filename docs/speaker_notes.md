# Kịch Bản Thuyết Trình Báo Cáo Giữa Khóa · Team AI ProPTIT

Tài liệu hướng dẫn thuyết trình 15–20 phút dành cho buổi bảo vệ bài tập giữa khóa của **Nguyễn Hoàng Tùng**. Nội dung tập trung vào sản phẩm thực tế, phương pháp giải quyết bài toán và số liệu kiểm chứng.

- **File slide tương tác:** [docs/presentation.html](presentation.html)
- **Cấu hình trình chiếu:** 16:9 widescreen, phím mũi tên `←` / `→` hoặc `Space` chuyển slide, phím `F` bật toàn màn hình, `Ctrl + P` xuất PDF.

---

## Phân Bổ Thời Gian 15–20 Phút (15 Slide)

| Giai đoạn | Slide | Nội dung trọng tâm | Thời lượng gợi ý |
|---|---|---|---|
| **1. Khởi động & Mục lục** | 1 – 2 | Giới thiệu tên đề tài, người thực hiện và 5 phần nội dung báo cáo | 2.5 phút |
| **2. Khó khăn thực tế & Lỗi bản mẫu** | 3 – 4 | Hành vi gõ của sinh viên & 2 nguyên nhân khiến hệ thống mẫu điểm thấp | 3.5 phút |
| **3. Giải pháp & 4 Cải tiến đột phá** | 5 – 9 | Quy trình 4 bước, hiểu câu hỏi tiếng Việt, tìm kiếm 2 làn, lọc lại theo ý nhỏ & kỷ luật trả lời | 6.5 phút |
| **4. Đo lường khoa học & Tối ưu tốc độ** | 10 – 11 | Hành trình tiến bộ có số liệu, giảm từ 11s xuống dưới 2s trên laptop thường | 3.0 phút |
| **5. Trực tiếp trải nghiệm & Kết luận** | 12 – 15 | 4 tình huống chứng minh năng lực chatbot, thành tích vòng kín 88.03, Q&A & Lời cảm ơn | 4.0 phút |

---

## Lời Dẫn Gợi Ý Từng Slide

### Slide 1: Bìa Báo Cáo
* **Thông tin hiển thị:** Tiêu đề dự án, mô tả khái quát giải pháp và người thực hiện: **Nguyễn Hoàng Tùng (Team AI · ProPTIT)**.
* **Lời dẫn mở màn:** *"Chào các bạn và anh chị trong ban chuyên môn Team AI, hôm nay em xin được trình bày sản phẩm bài tập giữa khóa của mình: 'Hệ thống hỏi đáp thông minh cho cộng đồng sinh viên S2B' dựa trên bài toán của NeoRAG Cup."*

### Slide 2: Mục Lục Thuyết Trình
* **Thông điệp cốt lõi:** Mạch trình bày gồm 5 phần rõ ràng: Từ thực tế người dùng hỏi đến giải mã lỗi bản mẫu, đề xuất 4 cải tiến, đo đạc số liệu và trực tiếp xem chatbot hoạt động.

### Slide 3: Người Dùng Thực Tế Hỏi Như Thế Nào?
* **Điểm nhấn:** 
  1. Gõ nhanh không dấu (`"lam sao vao clb lt"`) và dùng từ viết tắt (`mm, gcc, dpt`).
  2. Một câu hỏi nhưng đòi hỏi so sánh cả 3 CLB.
  3. Hỏi cả những chuyện ngoài lề không có trong tài liệu.

### Slide 4: Vì Sao Hệ Thống Mẫu Chỉ Đạt 52.4 Điểm?
* **Điểm nhấn:** 
  * 53% điểm mất ở khâu tìm kiếm (bị lẫn lộn giữa các năm vì văn bản quá giống nhau).
  * 47% điểm mất ở khâu viết câu trả lời (viết dài dòng gấp 2.2 lần chuẩn, gạch đầu dòng lan man làm loãng đáp án chính).
  * Máy cố tình bịa chuyện khi tìm sai tài liệu.

### Slide 5: Quy Trình Xử Lý 4 Bước Tự Động
* **Điểm nhấn:** 
  * Bước 1: Làm rõ câu hỏi (thêm dấu, dịch từ viết tắt, tách câu).
  * Bước 2: Tìm kiếm 2 làn (làn từ khóa bắt số liệu, làn AI bắt nghĩa rộng).
  * Bước 3: Chấm điểm và lọc lại (đúng dù chỉ 1 chi tiết nhỏ cũng được chọn).
  * Bước 4: Viết câu trả lời súc tích 1–3 câu và nói rõ nếu không có dữ liệu.

### Slide 6: Cải Tiến #1 – Hiểu Đúng Tiếng Việt
* **Điểm nhấn:** 
  * Tự động tạo một làn tìm kiếm chữ không dấu để xử lý triệt để các câu gõ tắt.
  * Tách câu hỏi so sánh 3 CLB thành 3 câu hỏi con độc lập, giúp mang về đủ văn bản của cả 3 CLB cùng lúc và đạt tỷ lệ tìm đúng 93.4%.

### Slide 7: Cải Tiến #2 – Tìm Kiếm 2 Làn Bù Trừ Nhau
* **Điểm nhấn:** 
  * Làn AI giỏi hiểu từ đồng nghĩa nhưng hay quên năm và số liệu.
  * Làn từ khóa bắt trúng ngày tháng "09/10/2011" và link website.
  * Ghép cả hai làn giúp không bỏ sót tài liệu nào.

### Slide 8: Cải Tiến #3 – Chấm Điểm Lại Theo Từng Ý Nhỏ
* **Điểm nhấn:** 
  * Thay vì chấm điểm cả câu dài, mang từng câu hỏi con đi so với tài liệu.
  * Văn bản trả lời đúng cho một ý nhỏ sẽ lập tức được kéo lên Top-5.
  * Điểm tìm kiếm nhảy vọt từ 80.1 lên 88.55 ở vòng thi mở.

### Slide 9: Cải Tiến #4 – Kỷ Luật Trả Lời Gãy Gọn
* **Điểm nhấn:** 
  * Bắt buộc gói gọn trong 1–3 câu, cấm gạch đầu dòng, cấm in đậm làm rối văn bản.
  * Chép đúng từng con số, tên riêng.
  * Độ khớp đáp án chuẩn tăng 50%, thông tin rác giảm 70%.

### Slide 10: Đo Lường Khoa Học Qua Từng Bước
* **Điểm nhấn:** Bảng theo dõi từ bản mẫu (73.6 điểm) lên bản hoàn thiện (89.96 điểm trên tập luyện tập). Chứng minh mỗi bước làm đều có số đo kiểm chứng.

### Slide 11: Tối Ưu Tốc Độ Xuống Dưới 2 Giây
* **Điểm nhấn:** 
  * Nén mô hình để chạy nhẹ hơn trên laptop (độ trễ từ 3.9s xuống 1.8s).
  * Chỉ dùng 4 nhân hiệu năng để tránh nghẽn chip.
  * Truyền chữ từng từ thời gian thực, người dùng thấy nguồn trích dẫn sau chỉ 1 giây.

### Slide 12: 4 Tình Huống Chứng Minh Năng Lực
* **Kịch bản thực tế:**
  1. Hỏi ngày thành lập: Trả lời đúng `09/10/2011`, cột bên phải sáng đèn văn bản gốc.
  2. Hỏi tiếp: *"Ai là người sáng lập?"* → tự động hiểu ngữ cảnh trước.
  3. Gõ không dấu: Trả về link đăng ký chuẩn.
  4. Hỏi học phí: Báo vàng *"Ngoài phạm vi tài liệu"*, tuyệt đối không đoán mò.

### Slide 13: Kết Quả Chung Cuộc
* **Điểm số đối sánh:** Bản mẫu 52.4 → Vòng mở (Public) 86.17 → Vòng kín (Private) 88.03. Vòng kín điểm cao hơn vòng mở chứng minh hệ thống hiểu bản chất dữ liệu, không học vẹt.

### Slide 14: Demo & Hỏi Đáp
* Trình diễn trực tiếp chatbot đang chạy tại cổng 8000 (FastAPI).
* Mời anh chị và các bạn đặt câu hỏi hoặc thử nghiệm trực tiếp các câu hỏi phức tạp.

### Slide 15: Lời Cảm Ơn (Thank You)
* Lời cảm ơn anh chị và các bạn đã lắng nghe bài thuyết trình.
