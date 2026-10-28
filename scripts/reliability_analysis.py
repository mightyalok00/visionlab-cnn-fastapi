from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from app.model import CLASS_NAMES, load_keras_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "best_model.keras"
REPORTS_DIR = ROOT / "reports"


def compute_calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Computes Expected Calibration Error (ECE), MCE, and Brier Score."""
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = np.mean(in_bin)
        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)
            bin_accs.append(float(accuracy_in_bin))
            bin_confs.append(float(avg_confidence_in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((lower + upper) / 2))

    # Brier score: mean squared difference between predicted probability and one-hot true label
    n_classes = probabilities.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    brier_score = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    metrics = {
        "expected_calibration_error": round(float(ece), 4),
        "maximum_calibration_error": round(float(mce), 4),
        "brier_score": round(brier_score, 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "overall_accuracy": round(float(np.mean(accuracies)), 4),
        "bin_counts": bin_counts,
        "bin_accuracies": [round(x, 4) for x in bin_accs],
        "bin_confidences": [round(x, 4) for x in bin_confs],
    }

    return metrics, np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)


def evaluate_inversion_risks(
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: keras.Model,
) -> dict:
    """
    Simulates real-world background variations to identify when auto-inversion
    helps or harms predictions.
    """
    # 1. Standard black background (benchmark condition)
    probs_orig = model.predict(x_test[..., None], batch_size=128, verbose=0)
    acc_orig = float(np.mean(probs_orig.argmax(axis=1) == y_test))

    # 2. Inverted test set (simulating white backgrounds) with auto_invert
    x_white_bg = 1.0 - x_test
    # Measure border brightness
    borders = np.concatenate([
        x_white_bg[:, :2, :].reshape(len(x_test), -1),
        x_white_bg[:, -2:, :].reshape(len(x_test), -1),
        x_white_bg[:, 2:-2, :2].reshape(len(x_test), -1),
        x_white_bg[:, 2:-2, -2:].reshape(len(x_test), -1),
    ], axis=1)
    border_means = borders.mean(axis=1)
    auto_inverted_count = int(np.sum(border_means > 0.5))

    # Apply auto-inversion rule
    x_recovered = np.where(border_means[:, None, None] > 0.5, 1.0 - x_white_bg, x_white_bg)
    probs_recovered = model.predict(x_recovered[..., None], batch_size=128, verbose=0)
    acc_recovered = float(np.mean(probs_recovered.argmax(axis=1) == y_test))

    # 3. Pathological failure simulation: Light garment filling entire frame
    # (where the clothing borders touch all four edges, making the border brighter than 0.5 even on dark bg)
    orig_borders = np.concatenate([
        x_test[:, :2, :].reshape(len(x_test), -1),
        x_test[:, -2:, :].reshape(len(x_test), -1),
        x_test[:, 2:-2, :2].reshape(len(x_test), -1),
        x_test[:, 2:-2, -2:].reshape(len(x_test), -1),
    ], axis=1)
    false_inversions = (orig_borders.mean(axis=1) > 0.5)
    false_inversion_rate = float(np.mean(false_inversions))

    # Identify classes most vulnerable to edge-touching false inversion
    vulnerable_classes = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        mask = (y_test == c_idx)
        vulnerable_classes[c_name] = {
            "false_inversion_rate_on_dark_bg": round(float(np.mean(false_inversions[mask])), 4),
            "original_accuracy": round(float(np.mean(probs_orig.argmax(axis=1)[mask] == c_idx)), 4),
        }

    return {
        "benchmark_dark_bg_accuracy": round(acc_orig, 4),
        "white_bg_with_auto_invert_accuracy": round(acc_recovered, 4),
        "white_bg_auto_inversion_detection_rate": round(auto_inverted_count / len(x_test), 4),
        "false_inversion_risk_on_dark_bg": round(false_inversion_rate, 4),
        "class_vulnerability_profile": vulnerable_classes,
        "critical_guidelines": [
            "Auto-inversion relies on the assumption that background dominates the outer 2-pixel perimeter.",
            "Items that touch all image borders (e.g. wide oversized coats or pullovers cropped closely) can falsely trigger inversion.",
            "Users should be provided with an explicit toggle to disable auto-inversion when uploading pre-segmented dark garments.",
        ],
    }


def main():
    print("Loading test dataset and model...")
    (_, _), (x_test_raw, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test = x_test_raw.astype("float32") / 255.0

    model = load_keras_model(MODEL_PATH)
    probs = model.predict(x_test[..., None], batch_size=128, verbose=0)

    # 1. Calibration
    cal_metrics, bin_confs, bin_accs, bin_counts = compute_calibration_metrics(y_test, probs)

    # Plot Calibration Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax1.plot(bin_confs, bin_accs, "s-", color="#6366f1", label=f"CNN (ECE = {cal_metrics['expected_calibration_error']})")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Positives (Accuracy)")
    ax1.set_title("Reliability Diagram (Calibration Curve)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(np.linspace(0.05, 0.95, 10), bin_counts, width=0.08, color="#06b6d4", alpha=0.8, edgecolor="none")
    ax2.set_xlabel("Confidence Interval")
    ax2.set_ylabel("Count of Predictions")
    ax2.set_title("Confidence Distribution Histogram")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    cal_plot_path = REPORTS_DIR / "calibration_curve.png"
    fig.savefig(cal_plot_path, dpi=150)
    plt.close(fig)

    # 2. Inversion Risk Analysis
    inversion_analysis = evaluate_inversion_risks(x_test, y_test, model)

    # Full Reliability Report
    full_report = {
        "calibration": cal_metrics,
        "background_inversion_analysis": inversion_analysis,
    }

    report_json_path = REPORTS_DIR / "model_reliability.json"
    report_json_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    # Write Markdown Document
    md_content = f"""# Model Reliability, Calibration & Distribution Shift Analysis

This report documents the statistical reliability of VisionLab 2.0's winning scratch CNN (`regularized`), measuring confidence calibration on test data and boundary risks of automatic background inversion.

---

## 1. Confidence Calibration & Uncertainty

Modern neural networks are frequently overconfident. A well-calibrated classifier's predicted probability reflects its actual probability of being correct (e.g., of all predictions made with 90% confidence, 90% should be accurate).

### Calibration Metrics (Test Set, N = 10,000)

| Metric | Measured Score | Interpretation |
| :--- | :---: | :--- |
| **Expected Calibration Error (ECE)** | **{cal_metrics['expected_calibration_error']}** | Average gap between model confidence and true accuracy across 10 confidence bins. |
| **Maximum Calibration Error (MCE)** | **{cal_metrics['maximum_calibration_error']}** | Maximum error gap in any single confidence bin. |
| **Brier Score** | **{cal_metrics['brier_score']}** | Mean squared error of probability forecasts (lower is better; 0.0 is perfect). |
| **Mean Predicted Confidence** | **{cal_metrics['mean_confidence'] * 100:.1f}%** | Overall average certainty across all test samples. |
| **True Overall Accuracy** | **{cal_metrics['overall_accuracy'] * 100:.1f}%** | Actual empirical accuracy on the test set. |

<p align="center">
  <img src="calibration_curve.png" alt="Reliability Diagram" width="700"/>
</p>

---

## 2. Background Auto-Inversion Analysis & Risk Profile

Fashion-MNIST clothing items are bright pixels against dark background ($0$). Real photos typically have white or light backgrounds ($255$). The project uses automated perimeter luminance detection to bridge this distribution gap.

### Empirical Recovery Performance

- **Standard Dark Background Accuracy:** **{inversion_analysis['benchmark_dark_bg_accuracy'] * 100:.2f}%**
- **White Background with Auto-Inversion Accuracy:** **{inversion_analysis['white_bg_with_auto_invert_accuracy'] * 100:.2f}%**
- **Detection Rate on Light Backgrounds:** **{inversion_analysis['white_bg_auto_inversion_detection_rate'] * 100:.2f}%**
- **False Inversion Rate on Benchmark Dark Backgrounds:** **{inversion_analysis['false_inversion_risk_on_dark_bg'] * 100:.2f}%**

### Failure Modes: When Auto-Inversion May Harm Predictions

1. **Edge-Bleeding Garments**:
   - When a garment is tightly cropped such that wide sections of the fabric touch all 4 outer borders (perimeter brightness $> 127.5$), auto-inversion can mistake the garment itself for a light background.
2. **Transparent / Checkerboard PNGs**:
   - When uploaded without compositing, transparent areas may evaluate to erratic alpha values. (VisionLab handles this by compositing over white prior to border sampling).
3. **Low-Contrast Gray Backgrounds**:
   - Backgrounds with mean brightness hovering around the $127.5$ decision threshold ($110-140$) can produce threshold flipping.

### Mitigation in Production

VisionLab provides:
- An explicit query parameter `auto_invert=false` on `POST /predict`, `POST /explain`, and `POST /predict/batch`.
- An interactive UI toggle on the Web Playground allowing users to immediately switch off inversion when uploading garments with naturally light borders.
- Response metadata `inverted_background: bool` so downstream clients know if an inversion took place.
"""

    report_md_path = REPORTS_DIR / "model_reliability.md"
    report_md_path.write_text(md_content, encoding="utf-8")
    print(f"Reliability analysis completed! Saved to {report_json_path} and {report_md_path}")


if __name__ == "__main__":
    main()
