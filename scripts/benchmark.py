from __future__ import annotations

import io
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.model import load_keras_model, predict_array, preprocess_array

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "best_model.keras"
REPORTS_DIR = ROOT / "reports"


def get_hardware_info() -> dict[str, str]:
    import tensorflow as tf

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "processor": platform.processor() or "CPU",
    }


def create_synthetic_image() -> bytes:
    arr = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def run_benchmark(num_warmup: int = 20, num_runs: int = 200) -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")

    model = load_keras_model(MODEL_PATH)
    sample_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

    # 1. Warm-up
    for _ in range(num_warmup):
        _ = predict_array(model, sample_array)

    # 2. Single-image raw inference latency (model + preprocessing)
    latencies_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = predict_array(model, sample_array)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    single_metrics = {
        "runs": num_runs,
        "mean_ms": round(float(np.mean(latencies_ms)), 3),
        "std_ms": round(float(np.std(latencies_ms)), 3),
        "median_ms": round(float(np.median(latencies_ms)), 3),
        "p90_ms": round(float(np.percentile(latencies_ms, 90)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "min_ms": round(float(np.min(latencies_ms)), 3),
        "max_ms": round(float(np.max(latencies_ms)), 3),
        "throughput_fps": round(1000.0 / float(np.mean(latencies_ms)), 2),
    }

    # 3. Batch throughput benchmarks
    batch_sizes = [1, 4, 8, 16]
    batch_results = {}
    for bs in batch_sizes:
        batch_input = np.stack([preprocess_array(sample_array) for _ in range(bs)], axis=0)
        # Warm-up
        for _ in range(10):
            _ = model.predict(batch_input, verbose=0)

        batch_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model.predict(batch_input, verbose=0)
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000.0)

        mean_batch_ms = float(np.mean(batch_times))
        batch_results[f"batch_{bs}"] = {
            "batch_size": bs,
            "mean_batch_latency_ms": round(mean_batch_ms, 3),
            "median_batch_latency_ms": round(float(np.median(batch_times)), 3),
            "p95_batch_latency_ms": round(float(np.percentile(batch_times, 95)), 3),
            "per_image_latency_ms": round(mean_batch_ms / bs, 3),
            "throughput_img_per_sec": round((bs / mean_batch_ms) * 1000.0, 2),
        }

    # 4. End-to-end FastAPI client benchmark
    from fastapi.testclient import TestClient

    from app.main import app

    e2e_latencies = []
    png_data = create_synthetic_image()
    with TestClient(app) as client:
        # Warm-up
        for _ in range(10):
            client.post("/predict", files={"file": ("sample.png", png_data, "image/png")})

        for _ in range(100):
            t0 = time.perf_counter()
            res = client.post("/predict", files={"file": ("sample.png", png_data, "image/png")})
            t1 = time.perf_counter()
            assert res.status_code == 200
            e2e_latencies.append((t1 - t0) * 1000.0)

    e2e_metrics = {
        "requests": 100,
        "mean_ms": round(float(np.mean(e2e_latencies)), 3),
        "median_ms": round(float(np.median(e2e_latencies)), 3),
        "p95_ms": round(float(np.percentile(e2e_latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(e2e_latencies, 99)), 3),
        "throughput_req_per_sec": round(1000.0 / float(np.mean(e2e_latencies)), 2),
    }

    report = {
        "environment": get_hardware_info(),
        "single_image_inference": single_metrics,
        "batch_inference": batch_results,
        "api_end_to_end": e2e_metrics,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "benchmark_summary.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Benchmark report saved to {out_path}")
    print("\n--- Benchmark Summary ---")
    print(f"Hardware: {report['environment']['processor']} | {report['environment']['os']}")
    print(f"Single image latency: Median={single_metrics['median_ms']}ms, P95={single_metrics['p95_ms']}ms, Throughput={single_metrics['throughput_fps']} FPS")
    print(f"API End-to-End: Median={e2e_metrics['median_ms']}ms, P95={e2e_metrics['p95_ms']}ms, Throughput={e2e_metrics['throughput_req_per_sec']} req/s")
    for k, v in batch_results.items():
        print(f"Batch {v['batch_size']}: Latency={v['median_batch_latency_ms']}ms, Throughput={v['throughput_img_per_sec']} img/s")

    return report


if __name__ == "__main__":
    run_benchmark()
