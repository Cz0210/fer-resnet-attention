"""Generate course-report figures from training and evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

_cache_dir = Path(tempfile.gettempdir()) / "fer_cache"
_mpl_cache = _cache_dir / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize FER experiment metrics and reports.")
    parser.add_argument("--run_dir", required=True, help="Directory containing metrics.csv and evaluation CSV files.")
    parser.add_argument("--assets_dir", default="assets/figures", help="Shared figure output directory.")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory scanned for model comparison.")
    return parser


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_line_plot(rows, x_key, y_keys, labels, title, ylabel, output_paths):
    import matplotlib.pyplot as plt

    xs = [int(float(row[x_key])) for row in rows]
    plt.figure(figsize=(8, 5))
    for key, label in zip(y_keys, labels):
        ys = [float(row[key]) for row in rows]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=3, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200)
    plt.close()


def _load_confusion_matrix(path: Path):
    import numpy as np

    rows = _read_rows(path)
    class_names = []
    matrix = []
    for row in rows:
        class_names.append(row["label"])
        matrix.append([int(row[name]) for name in row.keys() if name != "label"])
    return class_names, np.array(matrix, dtype=float)


def _save_confusion_plot(confusion_path: Path, output_paths):
    import matplotlib.pyplot as plt
    import numpy as np

    class_names, matrix = _load_confusion_matrix(confusion_path)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_f1_bar(report_path: Path, output_paths):
    import matplotlib.pyplot as plt

    rows = [row for row in _read_rows(report_path) if row["class"] not in {"macro_avg", "weighted_avg", "accuracy"}]
    names = [row["class"] for row in rows]
    f1s = [float(row["f1"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, f1s, color="#3f7cac")
    ax.set_title("Per-Class F1 Score")
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
    plt.close(fig)


def _best_metric(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return max(values) if values else ""


def _last_metric(rows, key):
    return float(rows[-1][key]) if rows and rows[-1].get(key) not in {None, ""} else ""


def _write_comparison(outputs_dir: Path, output_paths):
    fieldnames = ["run_name", "best_val_acc", "best_macro_f1", "best_balanced_acc", "final_train_acc", "final_val_loss"]
    rows = []
    for metrics_path in sorted(outputs_dir.glob("*/metrics.csv")):
        metrics = _read_rows(metrics_path)
        rows.append(
            {
                "run_name": metrics_path.parent.name,
                "best_val_acc": _best_metric(metrics, "val_acc"),
                "best_macro_f1": _best_metric(metrics, "macro_f1"),
                "best_balanced_acc": _best_metric(metrics, "balanced_acc"),
                "final_train_acc": _last_metric(metrics, "train_acc"),
                "final_val_loss": _last_metric(metrics, "val_loss"),
            }
        )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    run_figures = run_dir / "figures"
    assets_dir = Path(args.assets_dir)

    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        rows = _read_rows(metrics_path)
        _save_line_plot(
            rows,
            "epoch",
            ["train_loss", "val_loss"],
            ["Train Loss", "Validation Loss"],
            "Training and Validation Loss",
            "Loss",
            [run_figures / "train_val_loss.png", assets_dir / f"{run_dir.name}_train_val_loss.png"],
        )
        _save_line_plot(
            rows,
            "epoch",
            ["train_acc", "val_acc"],
            ["Train Accuracy", "Validation Accuracy"],
            "Training and Validation Accuracy",
            "Accuracy",
            [run_figures / "train_val_acc.png", assets_dir / f"{run_dir.name}_train_val_acc.png"],
        )
        _save_line_plot(
            rows,
            "epoch",
            ["macro_f1"],
            ["Validation Macro F1"],
            "Validation Macro F1 Curve",
            "Macro F1",
            [run_figures / "macro_f1_curve.png", assets_dir / f"{run_dir.name}_macro_f1_curve.png"],
        )
        _save_line_plot(
            rows,
            "epoch",
            ["lr"],
            ["Learning Rate"],
            "Learning Rate Curve",
            "Learning Rate",
            [run_figures / "learning_rate_curve.png", assets_dir / f"{run_dir.name}_learning_rate_curve.png"],
        )

    confusion_path = run_dir / "confusion_matrix.csv"
    if confusion_path.exists():
        _save_confusion_plot(
            confusion_path,
            [run_figures / "normalized_confusion_matrix.png", assets_dir / f"{run_dir.name}_normalized_confusion_matrix.png"],
        )

    report_path = run_dir / "classification_report.csv"
    if report_path.exists():
        _save_f1_bar(report_path, [run_figures / "per_class_f1_bar.png", assets_dir / f"{run_dir.name}_per_class_f1_bar.png"])

    _write_comparison(
        Path(args.outputs_dir),
        [run_dir / "model_comparison_table.csv", assets_dir / "model_comparison_table.csv"],
    )
    print(f"Figures saved to: {run_figures} and {assets_dir}")


if __name__ == "__main__":
    main()
