import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(tiny_artifacts, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_artifacts / "model.onnx"))
    monkeypatch.setenv("CARD_PATH", str(tiny_artifacts / "model_card.json"))
    from service.app import app

    with TestClient(app) as c:
        yield c


def _jpeg(size=(256, 256), color=(30, 120, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model-info").json()["model_version"] == "test"


def test_predict_happy_path(client):
    r = client.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    j = r.json()
    assert j["label"] in client.get("/model-info").json()["classes"]
    assert 0 <= j["confidence"] <= 1 and len(j["top_k"]) == 3
    assert isinstance(j["is_weed"], bool) and "X-Request-ID" in r.headers


def test_rejects_bad_inputs(client):
    assert client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/predict", files={"file": ("a.jpg", b"not an image", "image/jpeg")}).status_code == 400
    assert client.post("/predict", files={"file": ("a.jpg", b"", "image/jpeg")}).status_code == 400
    big = b"\xff\xd8" + b"0" * (5 * 1024 * 1024 + 10)
    assert client.post("/predict", files={"file": ("a.jpg", big, "image/jpeg")}).status_code == 413
    r = client.post("/predict", files={"file": ("a.jpg", _jpeg((8, 8)), "image/jpeg")})
    assert r.status_code == 400 and "request_id" in r.json()


def test_batch_limit(client):
    files = [("files", (f"{i}.jpg", _jpeg(), "image/jpeg")) for i in range(17)]
    assert client.post("/predict/batch", files=files).status_code == 400
    ok = client.post("/predict/batch", files=files[:2])
    assert ok.status_code == 200 and len(ok.json()["results"]) == 2
