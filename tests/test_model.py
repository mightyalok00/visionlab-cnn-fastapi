import io
import numpy as np
import pytest
from PIL import Image

from app.model import (
    CLASS_NAMES,
    build_model,
    predict_array,
    preprocess_array,
    process_raw_image,
)


def test_model_output_shape_and_probabilities():
    pytest.importorskip("tensorflow")
    model = build_model("baseline")
    output = model(np.zeros((2, 28, 28, 1), dtype=np.float32)).numpy()
    assert output.shape == (2, 10)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)


def test_preprocessing_shape_and_range():
    result = preprocess_array(np.full((28, 28), 255, dtype=np.uint8))
    assert result.shape == (28, 28, 1)
    assert result.dtype == np.float32
    assert result.min() == pytest.approx(1.0)


@pytest.mark.parametrize("invalid", [
    np.zeros((27, 28)),
    np.full((28, 28), np.nan),
    np.full((28, 28), 256),
])
def test_preprocessing_rejects_invalid_inputs(invalid):
    with pytest.raises(ValueError):
        preprocess_array(invalid)


def test_prediction_schema():
    class DummyModel:
        def predict(self, batch, verbose=0):
            probs = np.full((1, 10), 0.1, dtype=np.float32)
            probs[0, 0] = 0.5
            probs /= probs.sum(axis=1, keepdims=True)
            return probs

    result = predict_array(DummyModel(), np.zeros((28, 28), dtype=np.uint8))
    assert result["predicted_class"] in CLASS_NAMES
    assert len(result["top_predictions"]) == 3
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_process_raw_image_auto_inversion():
    # Create white image with dark center (light background)
    img = Image.new("RGB", (60, 60), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    pixels, was_inverted = process_raw_image(buf.getvalue(), auto_invert=True)
    assert was_inverted is True
    assert pixels.shape == (28, 28)
    # The inverted border should now be black (~0)
    assert pixels[0, 0] == pytest.approx(0.0, abs=1.0)


def test_process_raw_image_dark_background_no_inversion():
    # Create black image (already dark background)
    img = Image.new("RGB", (60, 60), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    pixels, was_inverted = process_raw_image(buf.getvalue(), auto_invert=True)
    assert was_inverted is False
    assert pixels[0, 0] == pytest.approx(0.0, abs=1.0)


def test_process_raw_image_rgba_transparency():
    # Create RGBA image with transparent background
    img = Image.new("RGBA", (40, 40), color=(0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    pixels, _ = process_raw_image(buf.getvalue(), auto_invert=False)
    assert pixels.shape == (28, 28)
