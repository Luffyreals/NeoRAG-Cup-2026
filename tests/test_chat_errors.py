import httpx2 as httpx  # openai 3.x dùng httpx2
import openai

from neorag.chat_errors import friendly_error

REQUEST = httpx.Request("POST", "http://localhost/v1/chat/completions")


def _status_error(cls, code):
    return cls("x", response=httpx.Response(code, request=REQUEST), body=None)


def test_friendly_error_by_type():
    assert "OPENAI_API_KEY" in friendly_error(_status_error(openai.AuthenticationError, 401))
    assert "hạn mức" in friendly_error(_status_error(openai.RateLimitError, 429))
    assert "kết nối" in friendly_error(openai.APIConnectionError(request=REQUEST))
    assert "hạn mức" in friendly_error(_status_error(openai.APIStatusError, 402))
    assert "sự cố" in friendly_error(_status_error(openai.InternalServerError, 503))
    assert friendly_error(ValueError("x")).startswith("Đã có lỗi")
