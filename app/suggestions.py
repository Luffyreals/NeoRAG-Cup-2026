"""Câu hỏi gợi ý "hỏi tiếp" cho chatbot, theo CLB và chủ đề của tài liệu.

Sau mỗi câu trả lời, giao diện xem các nguồn vừa dùng thuộc CLB nào, chủ đề nào, rồi lấy
câu gợi ý từ bảng này: không cần gọi LLM, hiện ngay lập tức.

Khóa ngoài là nhãn CLB (giống `CLB_LABELS` trong server.py), khóa trong là `topic` của tài liệu
trong data/corpus.jsonl. Câu hỏi do mình tự soạn từ nội dung tài liệu (không lấy từ bộ test),
và đã chạy thử qua pipeline để chắc chatbot trả lời được.
"""

SUGGESTIONS: dict[str, dict[str, list[str]]] = {
    "ProPTIT": {
        "Giới thiệu chung": [
            "CLB Lập Trình PTIT hoạt động về lĩnh vực gì?",
            "Tên tiếng Anh của CLB Lập Trình PTIT là gì?",
        ],
        "Lĩnh vực & hoạt động": [
            "Các hoạt động của ProPTIT diễn ra thường xuyên như thế nào?",
            "ProPTIT khác gì một lớp học code thông thường?",
        ],
        "Cơ cấu & team dự án": ["ProPTIT có những team dự án nào?"],
        "Tuyển thành viên": [
            "Vòng tuyển thành viên ProPTIT gồm mấy vòng?",
            "Mỗi năm ProPTIT tuyển bao nhiêu thành viên?",
            "Biết code trước có được ưu tiên khi vào ProPTIT không?",
        ],
        "Điều kiện tham gia": [
            "Sinh viên năm hai có vào được ProPTIT không?",
            "Chưa biết lập trình có tham gia ProPTIT được không?",
            "Không học ngành CNTT có vào ProPTIT được không?",
        ],
        "Quyền lợi": ["Tham gia ProPTIT được những quyền lợi gì?"],
        "Đào tạo": [
            "Lộ trình đào tạo của ProPTIT gồm những giai đoạn nào?",
            "Vì sao ProPTIT dạy ngôn ngữ C trước?",
            "ProPTIT sinh hoạt mấy buổi một tuần?",
        ],
        "So sánh & tham gia nhiều CLB": [
            "ProPTIT khác CLB IT PTIT ở điểm nào?",
            "Có nên training cả ProPTIT và IT PTIT cùng lúc không?",
        ],
        "Trực thuộc": ["ProPTIT trực thuộc đơn vị nào?"],
        "Phí & quỹ": [
            "Quỹ CLB ProPTIT được dùng vào việc gì?",
            "Nội quy ProPTIT quy định giờ giấc thế nào?",
        ],
        "Sản phẩm": ["Thành viên ProPTIT làm ra những sản phẩm gì trong lúc học?"],
        "Sự kiện nội bộ": [
            "Big Game Training là sự kiện gì?",
            "ProPTIT có những cuộc thi nội bộ nào?",
        ],
        "Sự kiện đối ngoại": [
            "Code Battle là cuộc thi gì?",
            "ProPTIT tham gia những sân chơi bên ngoài nào?",
        ],
        "Giải thưởng": [
            "ProPTIT đạt thành tích gì tại Huawei ICT Competition 2025-2026?",
            "ProPTIT có những giải thưởng nổi bật nào gần đây?",
        ],
        "Lịch sử & truyền thống": [
            "Ai là người sáng lập ProPTIT?",
            "ProPTIT có truyền thống như thế nào?",
        ],
        "Ban chủ nhiệm": [
            "Ban chủ nhiệm ProPTIT nhiệm kỳ 2025-2026 gồm những ai?",
            "Ai là chủ nhiệm đầu tiên của ProPTIT?",
        ],
        "Nội quy & nghĩa vụ": [
            "Nội quy ProPTIT quy định trang phục thế nào?",
            "Điểm rèn luyện ở ProPTIT được tính ra sao?",
        ],
        "Liên hệ": ["Liên hệ ProPTIT qua những kênh nào?"],
    },
    "MPC": {
        "Giới thiệu chung": [
            "MPC là viết tắt của tên gì?",
            "Slogan của MPC là gì?",
        ],
        "Lĩnh vực & hoạt động": ["MPC phù hợp với những bạn thích gì?"],
        "Đào tạo": ["MPC đào tạo những kỹ năng gì?"],
        "Quyền lợi": ["Tham gia MPC được những quyền lợi gì?"],
        "Tuyển thành viên": [
            "Đợt tuyển Gen 14 của MPC có chủ đề gì?",
            "Quy trình tuyển thành viên MPC gồm mấy vòng?",
        ],
        "Cơ cấu": ["MPC có những ban nào?"],
        "Sản phẩm": ["MPC từng làm sản phẩm nội dung nào?"],
        "Liên hệ": ["Liên hệ CLB Multimedia PTIT bằng cách nào?"],
    },
    "GCC": {
        "Giới thiệu chung": ["GCC là CLB gì?", "Slogan của GCC là gì?"],
        "Lịch sử & thành lập": ["GCC được thành lập chính thức khi nào?"],
        "Cơ cấu & đào tạo": ["GCC đào tạo theo những nhánh nào?"],
        "Đào tạo": [
            "Lộ trình nhánh Dev của GCC gồm những gì?",
            "Nhánh Art của GCC học những gì?",
        ],
        "Quyền lợi": ["Trở thành thành viên GCC có lợi ích gì?"],
        "Đối tác & bảo trợ": ["Công ty nào hậu thuẫn GCC?"],
        "Lĩnh vực & hoạt động": ["GCC có những hoạt động chính nào?"],
        "Tuyển thành viên": [
            "GCC đang mở đơn tuyển thành viên chưa?",
            "Gen 1 của GCC có bao nhiêu thành viên?",
        ],
        "Sự kiện nội bộ": [
            "BigGame Training 2023 của GCC diễn ra ở đâu?",
            "EndContest Training 2023 của GCC tổ chức ngày nào?",
        ],
        "Giải thưởng & sản phẩm": [
            "Sweet Battle là game gì?",
            "Trang itch.io của GCC có trò chơi gì?",
        ],
        "Ban chủ nhiệm": ["Chủ nhiệm GCC là ai?"],
        "Liên hệ": ["Liên hệ GCC qua những kênh nào?"],
    },
    "S2B": {
        "Giới thiệu chung": ["Cộng đồng S2B là gì?"],
        "Sự kiện chung": [
            "Big Game S2B là hoạt động gì?",
            "Sinh nhật 13 năm của S2B có chủ đề gì?",
        ],
        "Quan hệ giữa các CLB": ["GCC liên quan thế nào với cộng đồng S2B?"],
    },
}
