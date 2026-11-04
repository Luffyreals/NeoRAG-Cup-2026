from types import SimpleNamespace

from neorag.llm import LLMClient


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _FakeCompletions:
    def __init__(self, pieces):
        self.pieces, self.calls = pieces, 0

    def create(self, model, messages, stream=False, **kwargs):
        self.calls += 1
        if stream:
            return iter([_chunk(p) for p in self.pieces] + [SimpleNamespace(choices=[])])
        text = "".join(p or "" for p in self.pieces)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_chat_stream_shares_cache_with_chat(tmp_path):
    client = LLMClient(api_key="x", base_url="http://localhost", cache_path=tmp_path / "c.sqlite")
    fake = _FakeCompletions(["\n ", "GCC ra ", None, "đời năm 2023. "])
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    messages = [{"role": "user", "content": "GCC ra đời năm nào?"}]
    assert list(client.chat_stream(messages, "m")) == ["GCC ra ", "đời năm 2023. "]
    assert client.chat(messages, "m") == "GCC ra đời năm 2023."  # lấy từ cache
    assert list(client.chat_stream(messages, "m")) == ["GCC ra đời năm 2023."]
    assert fake.calls == 1
