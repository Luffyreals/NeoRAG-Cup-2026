# Quy trình Git

Repo dùng GitFlow rút gọn cho một người làm, kèm quy ước commit Conventional Commits.

## Nhánh

| Nhánh | Vai trò | Quy tắc |
|---|---|---|
| `main` | Bản ổn định, dùng để nộp Public/Private Test | Chỉ nhận merge từ `develop` (hoặc `hotfix/*`). Mỗi lần nộp bài gắn tag, ví dụ `v0.1.0-public`. |
| `develop` | Nhánh tích hợp, luôn chạy được `pytest` và `neorag run train --no-judge` | Chỉ nhận merge từ `feature/*`. Không commit thẳng. |
| `feature/<tên>` | Mỗi hướng cải tiến một nhánh, ví dụ `feature/my-pipeline`, `feature/hybrid-retrieval`, `feature/reranker`, `feature/prompt-v2`, `feature/chatbot-ui` | Tách từ `develop`, merge lại `develop` bằng `--no-ff` khi test pass và điểm train không giảm. |
| `hotfix/<tên>` | Sửa lỗi gấp trên `main` | Tách từ `main`, merge vào cả `main` và `develop`. |

Luồng làm việc:

```bash
git switch develop
git switch -c feature/hybrid-retrieval
# ... code, commit nhỏ theo từng ý ...
pytest && ruff check .
neorag run train --no-judge --pipeline neorag.pipelines.my_pipeline:MyPipeline --out outputs/exp01_pred.jsonl
git switch develop && git merge --no-ff feature/hybrid-retrieval
```

Mỗi thí nghiệm có điểm được ghi vào [bảng thí nghiệm](experiments.md) trước khi merge.

## Commit

Định dạng [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <mô tả ngắn>`.

| Type | Dùng khi |
|---|---|
| `feat` | Thêm tính năng (retriever mới, reranker, prompt mới, UI) |
| `fix` | Sửa lỗi |
| `refactor` | Đổi cấu trúc code, không đổi hành vi |
| `perf` | Tăng tốc, giảm latency hoặc chi phí API |
| `test` | Thêm/sửa test |
| `docs` | Tài liệu, bảng thí nghiệm, phân tích lỗi |
| `chore` | Cấu hình, dependency, `.gitignore` |
| `exp` | Thay đổi tham số thí nghiệm trong `configs/pipeline.toml` |

Scope gợi ý: `retrieval`, `generation`, `config`, `app`, `eval`. Ví dụ:

```
feat(retrieval): add BM25 with Vietnamese tokenization
exp(config): raise top_k in prompt from 5 to 3
docs: add baseline error analysis
```

Quy tắc chung: mỗi commit một ý, mô tả ở thể mệnh lệnh, không commit `.env`, `outputs/`, `.cache/`.
