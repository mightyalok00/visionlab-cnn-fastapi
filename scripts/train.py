from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras

from app.model import CLASS_NAMES, build_model

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VisionLab CNN experiments.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--variants", nargs="+",
        choices=["baseline", "balanced", "regularized"],
        default=["baseline", "balanced", "regularized"],
    )
    return parser.parse_args()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Fashion-MNIST Confusion Matrix (Test Set)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive integers.")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    # Fixed, disjoint slices keep the benchmark split reproducible.
    x_train, y_train = x_train_full[:-6000], y_train_full[:-6000]
    x_val, y_val = x_train_full[-6000:], y_train_full[-6000:]

    x_train = x_train.astype("float32")[..., None] / 255.0
    x_val = x_val.astype("float32")[..., None] / 255.0
    x_test = x_test.astype("float32")[..., None] / 255.0

    results = []
    for variant in args.variants:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = build_model(variant=variant, seed=SEED)
        model_path = ARTIFACTS / f"{variant}.keras"
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(model_path), monitor="val_loss", save_best_only=True
            ),
        ]

        started = time.perf_counter()
        history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val),
            epochs=args.epochs, batch_size=args.batch_size,
            callbacks=callbacks, verbose=2,
        )
        training_seconds = time.perf_counter() - started
        model = keras.models.load_model(model_path)
        val_probabilities = model.predict(
            x_val, batch_size=args.batch_size, verbose=0
        )
        val_predictions = val_probabilities.argmax(axis=1)
        val_loss, val_accuracy = model.evaluate(
            x_val, y_val, batch_size=args.batch_size, verbose=0
        )
        val_macro_f1 = float(f1_score(y_val, val_predictions, average="macro"))

        variant_dir = REPORTS / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history.history["loss"], label="train")
        axes[0].plot(history.history["val_loss"], label="validation")
        axes[0].set(title=f"{variant}: loss", xlabel="Epoch", ylabel="Loss")
        axes[0].legend()
        axes[1].plot(history.history["accuracy"], label="train")
        axes[1].plot(history.history["val_accuracy"], label="validation")
        axes[1].set(title=f"{variant}: accuracy", xlabel="Epoch", ylabel="Accuracy")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(variant_dir / "learning_curves.png", dpi=150)
        plt.close(fig)

        result = {
            "variant": variant,
            "epochs_completed": len(history.history["loss"]),
            "training_seconds": round(training_seconds, 3),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_accuracy),
            "validation_macro_f1": val_macro_f1,
            "parameter_count": int(model.count_params()),
            "model_path": str(model_path.relative_to(ROOT)),
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    # Select only from validation results; do not use the test set to tune variants.
    selected = max(results, key=lambda item: item["validation_macro_f1"])
    selected_model_path = ROOT / selected["model_path"]
    selected_model = keras.models.load_model(selected_model_path)
    selected_model.save(ARTIFACTS / "best_model.keras")

    test_probabilities = selected_model.predict(
        x_test, batch_size=args.batch_size, verbose=0
    )
    test_predictions = test_probabilities.argmax(axis=1)
    test_loss, test_accuracy = selected_model.evaluate(
        x_test, y_test, batch_size=args.batch_size, verbose=0
    )
    test_report = classification_report(
        y_test, test_predictions, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )

    cm = confusion_matrix(y_test, test_predictions)
    np.save(REPORTS / "confusion_matrix.npy", cm)
    plot_confusion_matrix(cm, CLASS_NAMES, REPORTS / "confusion_matrix.png")
    (REPORTS / "classification_report.json").write_text(
        json.dumps(test_report, indent=2), encoding="utf-8"
    )


    summary = {
        "seed": SEED,
        "dataset": "Fashion-MNIST",
        "train_examples": int(len(y_train)),
        "validation_examples": int(len(y_val)),
        "test_examples": int(len(y_test)),
        "selection_metric": "validation_macro_f1",
        "selected_variant": selected["variant"],
        "experiments": results,
        "final_test_evaluation": {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "macro_f1": float(f1_score(y_test, test_predictions, average="macro")),
            "weighted_f1": float(f1_score(y_test, test_predictions, average="weighted")),
            "classification_report": "classification_report.json",
            "confusion_matrix": "confusion_matrix.npy",
        },
        "note": "Variants are selected using validation macro-F1. The test set is evaluated only once for the selected model.",
    }
    (REPORTS / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["final_test_evaluation"], indent=2))
    print(f"Selected variant: {selected['variant']}")
    print(f"Saved serving model: {ARTIFACTS / 'best_model.keras'}")


if __name__ == "__main__":
    main()
