"""Plot ablation experiment comparison for PPT slides."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_cache_dir = Path(tempfile.gettempdir()) / "fer_cache"
_mpl_cache = _cache_dir / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

ABLATION_RUNS = [
    ("No Aug", "ablation_resnet18_no_aug"),
    ("Aug", "ablation_resnet18_aug"),
    ("Aug + CBAM", "ablation_resnet18_cbam_aug"),
    ("Aug + CBAM + Focal", "ablation_resnet18_cbam_aug_focal"),
    ("Aug + CBAM + Focal + Sampler", "ablation_resnet18_cbam_aug_focal_sampler"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot FER ablation comparison.")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory containing ablation output folders.")
    parser.add_argument("--assets_dir", default="assets/figures", help="Directory for PPT tables and figures.")
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _from_eval_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "eval_metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
        "source": "test",
    }


def _from_report(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "classification_report.csv"
    if not path.exists():
        return None
    rows = _read_csv(path)
    by_name = {row["class"]: row for row in rows}
    macro = by_name.get("macro_avg", {})
    weighted = by_name.get("weighted_avg", {})
    accuracy = by_name.get("accuracy", {})
    return {
        "accuracy": float(accuracy.get("f1") or 0.0),
        "balanced_accuracy": float(macro.get("recall") or 0.0),
        "macro_f1": float(macro.get("f1") or 0.0),
        "weighted_f1": float(weighted.get("f1") or 0.0),
        "source": "test_report",
    }


def _from_training_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    rows = _read_csv(path)
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("macro_f1") or 0.0))
    return {
        "accuracy": float(best.get("val_acc") or 0.0),
        "balanced_accuracy": float(best.get("balanced_acc") or 0.0),
        "macro_f1": float(best.get("macro_f1") or 0.0),
        "weighted_f1": "",
        "source": "validation",
    }


def collect(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, run_name in ABLATION_RUNS:
        run_dir = outputs_dir / run_name
        metrics = _from_eval_metrics(run_dir) or _from_report(run_dir) or _from_training_metrics(run_dir)
        if metrics is None:
            metrics = {
                "accuracy": "",
                "balanced_accuracy": "",
                "macro_f1": "",
                "weighted_f1": "",
                "source": "missing",
            }
        rows.append({"ablation": label, "run_name": run_name, **metrics})
    return rows


def save_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ablation", "run_name", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "source"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def save_bar(rows: list[dict[str, Any]], metric: str, title: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    names = [row["ablation"] for row in rows]
    values = [_numeric(row, metric) for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(names, values, color="#3f7cac")
    ax.set_title(title)
    ax.set_xlabel("Ablation Setting")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(values):
        ax.text(idx, min(value + 0.02, 0.98), f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs_dir = Path(args.outputs_dir)
    assets_dir = Path(args.assets_dir)
    rows = collect(outputs_dir)
    save_table(rows, assets_dir / "ablation_table.csv")
    save_bar(rows, "macro_f1", "Ablation Study: Macro F1", "Macro F1", assets_dir / "ablation_macro_f1_bar.png")
    save_bar(
        rows,
        "balanced_accuracy",
        "Ablation Study: Balanced Accuracy",
        "Balanced Accuracy",
        assets_dir / "ablation_balanced_acc_bar.png",
    )
    print(f"Saved ablation comparison to: {assets_dir}")


if __name__ == "__main__":
    main()
