from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import UnidentifiedImageError

from app.model import (
    CLASS_NAMES,
    generate_gradcam_overlay,
    load_keras_model,
    predict_array,
    preprocess_array,
    process_raw_image,
)
from app.schemas import (
    BatchItemPrediction,
    BatchPredictionResponse,
    ExplainResponse,
    PredictionResponse,
    TopPrediction,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("visionlab")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("VISIONLAB_MODEL_PATH", str(BASE_DIR / "artifacts" / "best_model.keras")))
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_FILES = 16
MAX_BATCH_TOTAL_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = None
    application.state.model_error = None
    application.state.model_name = None
    application.state.param_count = None
    if MODEL_PATH.is_file():
        try:
            loaded = load_keras_model(MODEL_PATH)
            application.state.model = loaded
            application.state.model_name = getattr(loaded, "name", "custom_cnn")
            try:
                application.state.param_count = int(loaded.count_params())
            except Exception:
                application.state.param_count = None
            logger.info("Loaded model artifact: %s (%s params)", MODEL_PATH, application.state.param_count)
        except Exception as exc:  # Keep health/info endpoints available for diagnosis.
            application.state.model_error = type(exc).__name__
            logger.exception("Model loading failed")
    yield


app = FastAPI(
    title="VisionLab 2.0 API",
    description="Fashion-MNIST classification using a CNN trained from scratch.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/")
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept and TEMPLATE_PATH.is_file():
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return {
        "name": "VisionLab 2.0",
        "demo": "/demo",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
    }


@app.get("/demo", response_class=HTMLResponse)
def demo_playground():
    if TEMPLATE_PATH.is_file():
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Demo template not found.")


@app.get("/health")
def health(request: Request) -> dict:
    model_loaded = request.app.state.model is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_error": request.app.state.model_error,
    }


@app.get("/ready")
def readiness(request: Request) -> dict:
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    return {"ready": True}


@app.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    model_loaded = request.app.state.model is not None
    return {
        "model_type": "Custom CNN trained from random initialization",
        "dataset": "Fashion-MNIST",
        "model_loaded": model_loaded,
        "model_name": getattr(request.app.state, "model_name", None),
        "parameter_count": getattr(request.app.state, "param_count", None),
        "artifact_path": str(MODEL_PATH) if MODEL_PATH.is_file() else None,
        "input_shape": [28, 28, 1],
        "class_count": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    auto_invert: bool = Query(True, description="Automatically invert light backgrounds to match Fashion-MNIST distribution"),
):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable. Run `python -m scripts.train` first.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WebP image.")

    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 5 MB limit.")

        pixels, was_inverted = process_raw_image(payload, auto_invert=auto_invert)
        prediction = predict_array(request.app.state.model, pixels, was_inverted=was_inverted)
        return PredictionResponse(**prediction)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image.") from exc
    finally:
        await file.close()


@app.post("/explain", response_model=ExplainResponse)
async def explain(
    request: Request,
    file: UploadFile = File(...),
    auto_invert: bool = Query(True, description="Automatically invert light backgrounds"),
):
    """Generates Grad-CAM visual saliency heatmap overlay showing model attention."""
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable. Train a model first.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WebP image.")

    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 5 MB limit.")

        pixels, was_inverted = process_raw_image(payload, auto_invert=auto_invert)
        prediction = predict_array(request.app.state.model, pixels, was_inverted=was_inverted)
        gradcam_b64, preprocessed_b64 = generate_gradcam_overlay(request.app.state.model, pixels)

        return ExplainResponse(
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            gradcam_base64=gradcam_b64,
            preprocessed_base64=preprocessed_b64,
            inverted_background=was_inverted,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Failed to compute explanation for image.") from exc
    finally:
        await file.close()


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    auto_invert: bool = Query(True, description="Automatically invert light backgrounds"),
):
    """Vectorized batch prediction endpoint with per-file validation and batch payload limits."""
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable. Train a model first.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {MAX_BATCH_FILES} images; received {len(files)}.",
        )

    total_batch_bytes = 0
    results: list[BatchItemPrediction] = []
    valid_items: list[tuple[int, np.ndarray, bool]] = []  # (result_index, pixels, was_inverted)

    for idx, f in enumerate(files):
        filename = f.filename or f"image_{idx + 1}.png"
        try:
            if f.content_type not in ALLOWED_CONTENT_TYPES:
                results.append(
                    BatchItemPrediction(
                        filename=filename,
                        status="error",
                        error=f"Unsupported content type '{f.content_type}'. Expected image/png, image/jpeg, or image/webp.",
                    )
                )
                continue

            payload = await f.read(MAX_UPLOAD_BYTES + 1)
            total_batch_bytes += len(payload)
            if total_batch_bytes > MAX_BATCH_TOTAL_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Total batch payload exceeds the {MAX_BATCH_TOTAL_BYTES // (1024 * 1024)} MB limit.",
                )

            if not payload:
                results.append(
                    BatchItemPrediction(
                        filename=filename,
                        status="error",
                        error="Uploaded file is empty.",
                    )
                )
                continue

            if len(payload) > MAX_UPLOAD_BYTES:
                results.append(
                    BatchItemPrediction(
                        filename=filename,
                        status="error",
                        error=f"File exceeds the individual {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )
                )
                continue

            pixels, was_inverted = process_raw_image(payload, auto_invert=auto_invert)
            valid_items.append((idx, pixels, was_inverted))
            results.append(
                BatchItemPrediction(
                    filename=filename,
                    status="success",
                    inverted_background=was_inverted,
                )
            )
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            results.append(
                BatchItemPrediction(
                    filename=filename,
                    status="error",
                    error=f"Invalid or unreadable image: {type(exc).__name__}",
                )
            )
        finally:
            await f.close()

    # Vectorized inference across all valid images in the batch
    if valid_items:
        batch_arrays = np.stack([preprocess_array(item[1]) for item in valid_items], axis=0)
        all_probabilities = request.app.state.model.predict(batch_arrays, verbose=0)

        for (item_idx, _, was_inverted), probs in zip(valid_items, all_probabilities):
            top_indices = np.argsort(probs)[::-1][:3]
            top_preds = [
                TopPrediction(class_name=CLASS_NAMES[int(i)], probability=float(probs[i]))
                for i in top_indices
            ]
            results[item_idx].predicted_class = CLASS_NAMES[int(top_indices[0])]
            results[item_idx].confidence = float(probs[top_indices[0]])
            results[item_idx].top_predictions = top_preds

    successful = sum(1 for r in results if r.status == "success")
    failed = len(results) - successful

    return BatchPredictionResponse(
        total_images=len(files),
        successful=successful,
        failed=failed,
        predictions=results,
    )

