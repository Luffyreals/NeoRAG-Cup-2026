from app.server import app, state
from fastapi.testclient import TestClient


def test_server_static_and_info():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    state["pipeline"] = None
    state["docs"] = {1: "doc1"}
    res_info = client.get("/api/info")
    assert res_info.status_code == 200
    data = res_info.json()
    assert "examples" in data
    assert data["doc_count"] == 1
    assert "suggestions" in data
    state.clear()
