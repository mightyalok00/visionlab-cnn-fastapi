from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def build_model(variant: str = "balanced", seed: int = 42) -> Any:
    """Build a CNN from random initialization; no pretrained weights are used."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(seed)
    if variant not in {"baseline", "balanced", "regularized"}:
        raise ValueError(f"Unknown model variant: {variant}")

    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = inputs
    filters = (16, 32) if variant == "baseline" else (32, 64)

    x = layers.Conv2D(filters[0], 3, padding="same", activation="relu",
                      name="conv_1")(x)
    if variant == "regularized":
        x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.MaxPooling2D(name="pool_1")(x)
    x = layers.Conv2D(filters[1], 3, padding="same", activation="relu",
                      name="conv_2")(x)
    if variant == "regularized":
        x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling2D(name="pool_2")(x)
    x = layers.Dropout(0.20 if variant == "baseline" else 0.30,
                       name="conv_dropout")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    if variant != "baseline":
        x = layers.Dropout(0.30, name="dense_dropout")(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax",
                           name="class_probabilities")(x)

    model = keras.Model(inputs, outputs, name=f"visionlab_{variant}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def process_raw_image(payload: bytes, auto_invert: bool = True) -> tuple[np.ndarray, bool]:
    """
    Read image bytes, handle alpha transparency safely, resize to 28x28 grayscale,
    and optionally auto-invert light backgrounds to match Fashion-MNIST distribution (black bg).
    """
    with Image.open(io.BytesIO(payload)) as img:
        # Handle transparency safely by compositing over white
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img_rgba = img.convert("RGBA")
            background = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
            alpha_mask = img_rgba.split()[-1]
            background.paste(img_rgba, mask=alpha_mask)
            img = background.convert("L")
        else:
            img = img.convert("L")

        resized = img.resize((28, 28), Image.Resampling.LANCZOS)
        pixels = np.asarray(resized, dtype=np.float32)

    was_inverted = False
    if auto_invert:
        # Fashion-MNIST has black backgrounds (pixel ~ 0) and light foreground items.
        # Real-world photos almost always have light backgrounds (pixel ~ 255).
        # Check average perimeter/border brightness (outer 2 pixels).
        border_pixels = np.concatenate([
            pixels[:2, :].flatten(),
            pixels[-2:, :].flatten(),
            pixels[2:-2, :2].flatten(),
            pixels[2:-2, -2:].flatten(),
        ])
        if border_pixels.mean() > 127.5:
            pixels = 255.0 - pixels
            was_inverted = True

    return pixels, was_inverted


def preprocess_array(image: np.ndarray) -> np.ndarray:
    """Validate a 28x28 grayscale image and scale pixels to [0, 1]."""
    array = np.asarray(image)
    if array.shape != (28, 28):
        raise ValueError("Image must be exactly 28x28 pixels after resizing.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Image pixels must be numeric.")
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite values.")
    array = array.astype(np.float32)
    if array.min() < 0 or array.max() > 255:
        raise ValueError("Pixel values must be within [0, 255].")
    return (array / 255.0)[..., np.newaxis]


def predict_array(model: Any, image: np.ndarray, was_inverted: bool = False) -> dict[str, Any]:
    batch = preprocess_array(image)[np.newaxis, ...]
    probabilities = model.predict(batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    return {
        "predicted_class": CLASS_NAMES[int(top_indices[0])],
        "confidence": float(probabilities[top_indices[0]]),
        "top_predictions": [
            {"class_name": CLASS_NAMES[int(i)], "probability": float(probabilities[i])}
            for i in top_indices
        ],
        "probabilities": {
            name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "inverted_background": was_inverted,
    }


def load_keras_model(path: str | Path) -> Any:
    from tensorflow import keras

    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    return keras.models.load_model(model_path)


def generate_gradcam_overlay(
    model: Any,
    image_28x28: np.ndarray,
    last_conv_layer_name: str = "conv_2",
) -> tuple[str, str]:
    """
    Computes a Grad-CAM heatmap for the top prediction and generates
    base64-encoded PNG strings for the preprocessed image and colored overlay.
    """
    import base64

    import matplotlib
    import tensorflow as tf
    from tensorflow import keras

    batch = preprocess_array(image_28x28)[np.newaxis, ...]

    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    # Preprocessed input image as base64 (upscaled for crisp display)
    input_img = Image.fromarray(np.uint8(np.clip(image_28x28, 0, 255))).convert("RGB")
    input_buf = io.BytesIO()
    input_img.resize((140, 140), Image.Resampling.NEAREST).save(input_buf, format="PNG")
    preprocessed_base64 = f"data:image/png;base64,{base64.b64encode(input_buf.getvalue()).decode('utf-8')}"

    # Upscale heatmap and apply colormap
    colormap = matplotlib.colormaps["jet"]
    colored_heatmap = colormap(heatmap_np)[:, :, :3]
    heatmap_img = Image.fromarray(np.uint8(255 * colored_heatmap)).resize((140, 140), Image.Resampling.BILINEAR)

    # Superimpose heatmap on input image with 0.5 alpha
    base_img = input_img.resize((140, 140), Image.Resampling.NEAREST)
    overlay = Image.blend(base_img, heatmap_img, alpha=0.5)

    overlay_buf = io.BytesIO()
    overlay.save(overlay_buf, format="PNG")
    gradcam_base64 = f"data:image/png;base64,{base64.b64encode(overlay_buf.getvalue()).decode('utf-8')}"

    return gradcam_base64, preprocessed_base64

