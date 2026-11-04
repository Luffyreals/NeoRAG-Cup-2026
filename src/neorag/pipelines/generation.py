"""Sinh câu trả lời từ các doc đã truy xuất."""

from __future__ import annotations

from collections.abc import Iterator

from neorag.data import Document
from neorag.pipelines import baseline

# Template prompt theo tên; config chọn bằng generation.prompt.
# "baseline" giữ nguyên prompt của BaselinePipeline để dùng lại cache.
PROMPTS: dict[str, str] = {
    "baseline": baseline.PROMPT,
    # v2: đáp án chuẩn ngắn (trung vị ~200 ký tự) và nêu nguyên văn số liệu, còn baseline trả lời
    # dài gấp đôi, có markdown và lời dẫn → mất ROUGE-L và Response Relevancy.
    "v2": """Bạn là trợ lý hỏi đáp về cộng đồng S2B gồm CLB Lập Trình PTIT (ProPTIT), \
CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

TÀI LIỆU:
{context}

CÂU HỎI: {query}

Trả lời câu hỏi chỉ dựa trên TÀI LIỆU, theo các quy tắc:
- Đi thẳng vào ý chính trong 1-3 câu văn xuôi tiếng Việt có dấu. Không dùng markdown (không in \
đậm, không gạch đầu dòng, không tiêu đề), không mở đầu bằng "Dựa vào tài liệu" hay nhắc lại câu hỏi.
- Câu đầu nêu rõ chủ thể được hỏi (tên CLB, sự kiện, chương trình) rồi trả lời ngay.
- Chép nguyên văn tên riêng, ngày tháng, số liệu, khẩu hiệu, tên kênh và đường link như trong \
tài liệu.
- Câu hỏi có nhiều ý thì trả lời đủ từng ý. Câu hỏi có/không thì mở đầu bằng "Có." hoặc "Không.".
- Không thêm nhận xét, lời khuyên hay thông tin không được hỏi; bỏ qua tài liệu không liên quan.
- Nếu tài liệu không có thông tin để trả lời, nói ngắn gọn trong một câu: "Tài liệu hiện có \
không có thông tin về ...", không đoán.""",
    # v3: Response Relevancy so câu hỏi judge sinh ngược từ câu trả lời với câu hỏi gốc, nên câu
    # trả lời dùng lại từ khóa của người hỏi (thay vì chỉ dùng từ của tài liệu) và không lan sang
    # ý phụ; đồng thời giữ chi tiết cụ thể phục vụ câu hỏi (v2 bỏ sót link, tên cuộc thi).
    "v3": """Bạn là trợ lý hỏi đáp về cộng đồng S2B gồm CLB Lập Trình PTIT (ProPTIT), \
CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

TÀI LIỆU:
{context}

CÂU HỎI: {query}

Trả lời câu hỏi chỉ dựa trên TÀI LIỆU, theo các quy tắc:
- Viết 1-3 câu văn xuôi tiếng Việt có dấu. Không dùng markdown (không in đậm, không gạch đầu \
dòng, không tiêu đề), không mở đầu bằng "Dựa vào tài liệu".
- Câu đầu trả lời thẳng điều được hỏi và dùng lại từ khóa, cách gọi của chính câu hỏi (viết có \
dấu nếu câu hỏi gõ không dấu); nếu câu hỏi gọi tắt hoặc gọi vòng một CLB thì nêu thêm tên chính \
thức của CLB đó.
- Câu hỏi có/không thì mở đầu bằng "Có." hoặc "Không." rồi giải thích ngắn.
- Câu hỏi có nhiều ý thì trả lời đủ từng ý, theo thứ tự trong câu hỏi.
- Kèm các chi tiết cụ thể phục vụ trực tiếp câu hỏi (ngày tháng, số liệu, tên riêng, tên cuộc \
thi, đường link, kênh liên hệ), chép nguyên văn như trong tài liệu.
- Không thêm giới thiệu chung về CLB, nhận xét, lời khuyên hay thông tin mà câu hỏi không hỏi.
- Ý nào tài liệu không có thông tin thì nói ngắn gọn "Tài liệu hiện có không có thông tin về ...", \
không đoán.""",
    # v4: v2 + bám câu chữ của tài liệu. Phân tích exp10 max: phần lớn câu thiếu key fact dù đã
    # truy xuất đúng doc là do diễn đạt lại ("ba" thay "3", "mỗi tuần" thay "/tuần") hoặc chỉ chọn
    # một mục khi tài liệu liệt kê nhiều mục (nhiều giải, nhiều đơn vị), hoặc bỏ link đăng ký.
    "v4": """Bạn là trợ lý hỏi đáp về cộng đồng S2B gồm CLB Lập Trình PTIT (ProPTIT), \
CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

TÀI LIỆU:
{context}

CÂU HỎI: {query}

Trả lời câu hỏi chỉ dựa trên TÀI LIỆU, theo các quy tắc:
- Đi thẳng vào ý chính bằng văn xuôi tiếng Việt có dấu, ngắn gọn, thường 1-3 câu; câu hỏi nhiều \
ý thì được viết dài hơn để đủ ý. Không dùng markdown (không in đậm, không gạch đầu dòng, không \
tiêu đề), không mở đầu bằng "Dựa vào tài liệu" hay nhắc lại câu hỏi.
- Câu đầu nêu rõ chủ thể được hỏi (tên CLB, sự kiện, chương trình) rồi trả lời ngay.
- Dùng đúng cụm từ trong tài liệu cho các ý trả lời, không thay bằng từ đồng nghĩa hay diễn đạt \
lại. Chép nguyên văn tên riêng, ngày tháng, số liệu, khẩu hiệu, tên kênh và đường link; giữ nguyên \
cách viết số như tài liệu (tài liệu viết "3" thì không đổi thành "ba", viết "2 buổi/tuần" thì \
giữ nguyên).
- Câu hỏi có nhiều ý thì trả lời đủ từng ý. Nếu tài liệu có nhiều mục cùng khớp với câu hỏi \
(nhiều giải thưởng, nhiều đơn vị, nhiều mốc thời gian, nhiều ban) thì nêu tất cả, không chỉ chọn \
một.
- Câu hỏi về đăng ký, nộp đơn hoặc liên hệ thì kèm đường link hoặc kênh liên hệ có trong tài liệu.
- Câu hỏi có/không thì mở đầu bằng "Có." hoặc "Không.".
- Không thêm nhận xét, lời khuyên hay thông tin không được hỏi; bỏ qua tài liệu không liên quan.
- Nếu tài liệu không có thông tin để trả lời, nói ngắn gọn trong một câu: "Tài liệu hiện có \
không có thông tin về ...", không đoán.""",
    # v5: v4 + quy tắc từ chối đặt trước và chặt hơn. Ở exp14, v4 làm 2/11 câu không có đáp án
    # thôi từ chối (train-0020 tự suy ra nghĩa của "S2B", train-0033 kể thêm biệt danh ban khác):
    # quy tắc "nêu tất cả các mục" và "được viết dài hơn" đẩy model cố trả lời.
    "v5": """Bạn là trợ lý hỏi đáp về cộng đồng S2B gồm CLB Lập Trình PTIT (ProPTIT), \
CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

TÀI LIỆU:
{context}

CÂU HỎI: {query}

Trước khi trả lời, kiểm tra TÀI LIỆU có nêu trực tiếp điều được hỏi không. Nếu không có, chỉ \
viết đúng một câu: "Tài liệu hiện có không có thông tin về ...", không suy luận từ thông tin gần \
giống, không đoán và không kể thêm thông tin khác.

Nếu TÀI LIỆU có thông tin, trả lời chỉ dựa trên TÀI LIỆU theo các quy tắc:
- Đi thẳng vào ý chính bằng văn xuôi tiếng Việt có dấu, ngắn gọn, thường 1-3 câu; câu hỏi nhiều \
ý thì được viết dài hơn để đủ ý. Không dùng markdown (không in đậm, không gạch đầu dòng, không \
tiêu đề), không mở đầu bằng "Dựa vào tài liệu" hay nhắc lại câu hỏi.
- Câu đầu nêu rõ chủ thể được hỏi (tên CLB, sự kiện, chương trình) rồi trả lời ngay.
- Dùng đúng cụm từ trong tài liệu cho các ý trả lời, không thay bằng từ đồng nghĩa hay diễn đạt \
lại. Chép nguyên văn tên riêng, ngày tháng, số liệu, khẩu hiệu, tên kênh và đường link; giữ nguyên \
cách viết số như tài liệu (tài liệu viết "3" thì không đổi thành "ba", viết "2 buổi/tuần" thì \
giữ nguyên).
- Câu hỏi có nhiều ý thì trả lời đủ từng ý. Nếu tài liệu có nhiều mục cùng trả lời trực tiếp câu \
hỏi (nhiều giải thưởng, nhiều đơn vị, nhiều mốc thời gian, nhiều ban) thì nêu tất cả, không chỉ \
chọn một.
- Câu hỏi về đăng ký, nộp đơn hoặc liên hệ thì kèm đường link hoặc kênh liên hệ có trong tài liệu.
- Câu hỏi có/không thì mở đầu bằng "Có." hoặc "Không.".
- Không thêm nhận xét, lời khuyên hay thông tin không được hỏi; bỏ qua tài liệu không liên quan.
- Ý nào tài liệu không có thông tin thì nói ngắn gọn "Tài liệu hiện có không có thông tin về ...", \
không đoán.""",
    # v6: thân v4 + 2 quy tắc nhắm đúng 2 lỗi từ chối của v4, không chặn trước như v5. v5 sửa được
    # từ chối (11/11) nhưng quá dè dặt: từ chối cả câu có một phần dữ liệu (train-0095), cắt phần
    # giải thích của câu từ chối (train-0062) và bỏ chi tiết → String Presence thấp hơn v2.
    "v6": """Bạn là trợ lý hỏi đáp về cộng đồng S2B gồm CLB Lập Trình PTIT (ProPTIT), \
CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

TÀI LIỆU:
{context}

CÂU HỎI: {query}

Trả lời câu hỏi chỉ dựa trên TÀI LIỆU, theo các quy tắc:
- Đi thẳng vào ý chính bằng văn xuôi tiếng Việt có dấu, ngắn gọn, thường 1-3 câu; câu hỏi nhiều \
ý thì được viết dài hơn để đủ ý. Không dùng markdown (không in đậm, không gạch đầu dòng, không \
tiêu đề), không mở đầu bằng "Dựa vào tài liệu" hay nhắc lại câu hỏi.
- Câu đầu nêu rõ chủ thể được hỏi (tên CLB, sự kiện, chương trình) rồi trả lời ngay.
- Dùng đúng cụm từ trong tài liệu cho các ý trả lời, không thay bằng từ đồng nghĩa hay diễn đạt \
lại. Chép nguyên văn tên riêng, ngày tháng, số liệu, khẩu hiệu, tên kênh và đường link; giữ nguyên \
cách viết số như tài liệu (tài liệu viết "3" thì không đổi thành "ba", viết "2 buổi/tuần" thì \
giữ nguyên).
- Câu hỏi có nhiều ý thì trả lời đủ từng ý. Nếu tài liệu có nhiều mục cùng khớp với câu hỏi \
(nhiều giải thưởng, nhiều đơn vị, nhiều mốc thời gian, nhiều ban) thì nêu tất cả, không chỉ chọn \
một.
- Câu hỏi về đăng ký, nộp đơn hoặc liên hệ thì kèm đường link hoặc kênh liên hệ có trong tài liệu.
- Câu hỏi có/không thì mở đầu bằng "Có." hoặc "Không.".
- Không thêm nhận xét, lời khuyên hay thông tin không được hỏi; bỏ qua tài liệu không liên quan.
- Không tự suy ra định nghĩa, ý nghĩa, tên gọi đầy đủ hay thông tin mà tài liệu không ghi rõ.
- Khi tài liệu không có thông tin về đúng đối tượng được hỏi, không thay bằng thông tin của đối \
tượng khác (ban khác, CLB khác, năm khác).
- Nếu tài liệu không có thông tin để trả lời, nói ngắn gọn trong một câu: "Tài liệu hiện có \
không có thông tin về ...", không đoán.""",
}


class Generator:
    def __init__(self, llm, model: str, prompt: str = "baseline", top_k: int = 5):
        if prompt not in PROMPTS:
            raise ValueError(f"generation.prompt không hợp lệ: {prompt!r}")
        self.llm, self.model = llm, model
        self.template = PROMPTS[prompt]
        self.top_k = top_k  # số doc đưa vào prompt

    def build_context(self, docs: list[Document]) -> str:
        return "\n\n".join(d.text for d in docs[: self.top_k])

    def build_messages(self, query: str, docs: list[Document]) -> list[dict]:
        content = self.template.format(context=self.build_context(docs), query=query)
        return [{"role": "user", "content": content}]

    def generate(self, query: str, docs: list[Document]) -> str:
        return self.llm.chat(self.build_messages(query, docs), model=self.model)

    def generate_stream(self, query: str, docs: list[Document]) -> Iterator[str]:
        """Như `generate` nhưng trả từng đoạn chữ (cùng prompt, cùng cache)."""
        return self.llm.chat_stream(self.build_messages(query, docs), model=self.model)
