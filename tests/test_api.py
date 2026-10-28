import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


class FakeModel:
    """Small deterministic test double; API contract tests need no trained weights."""

    def predict(self, batch, verbose=0):
        probabilities = np.full((1, 10), 0.01, dtype=np.float32)
        probabilities[0, 3] = 0.91
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


def png_bytes() -> bytes:
    image = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_root_and_model_info():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["class_count"] == 10


def test_health_and_readiness_without_model():
    with TestClient(app) as client:
        client.app.state.model = None
        health = client.get("/health")
        ready = client.get("/ready")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert ready.status_code == 503


def test_health_and_readiness_with_model():
    with TestClient(app) as client:
        client.app.state.model = FakeModel()
        health = client.get("/health")
        ready = client.get("/ready")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert ready.status_code == 200


def test_prediction_endpoint_contract():
    with TestClient(app) as client:
        client.app.state.model = FakeModel()
        response = client.post(
            "/predict",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "Dress"
    assert len(payload["top_predictions"]) == 3
    assert len(payload["probabilities"]) == 10
    assert "X-Request-ID" in response.headers


def test_rejects_unsupported_content_type():
    with TestClient(app) as client:
        client.app.state.model = FakeModel()
        response = client.post(
            "/predict",
            files={"file": ("sample.txt", b"not an image", "text/plain")},
        )
    assert response.status_code == 415


def test_rejects_invalid_image_bytes():
    with TestClient(app) as client:
        client.app.state.model = FakeModel()
        response = client.post(
            "/predict",
            files={"file": ("sample.png", b"not really a PNG", "image/png")},
        )
    assert response.status_code == 400


def test_batch_prediction():
    with TestClient(app) as client:
        client.app.state.model = FakeModel()
        response = client.post(
            "/predict/batch",
            files=[
                ("files", ("sample1.png", png_bytes(), "image/png")),
                ("files", ("sample2.png", png_bytes(), "image/png")),
            ],
        )
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["predicted_class"] == "Dress"


def test_real_model_inference_and_explainability():
    from app.model import load_keras_model

    model_path = Path("artifacts/best_model.keras")
    if not model_path.is_file():
        pytest.skip("best_model.keras artifact not found")

    with TestClient(app) as client:
        client.app.state.model = load_keras_model(model_path)
        explain_res = client.post(
            "/explain",
            files={"file": ("sneaker.png", png_bytes(), "image/png")},
        )
        assert explain_res.status_code == 200
        data = explain_res.json()
        assert "gradcam_base64" in data
        assert data["gradcam_base64"].startswith("data:image/png;base64,")

