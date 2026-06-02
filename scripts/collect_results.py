"""Collect FER experiment metrics for PPT model comparison."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect FER evaluation results across runs.")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory containing outputs/<run_name>/ folders.")
    parser.add_argument("--assets_dir", default="assets/figures", help="Directory for comparison tables and figures.")
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_from_report(report_path: Path) -> dict[str, float]:
    rows = _read_csv(report_path)
    by_name = {row["class"]: row for row in rows}
    macro = by_name.get("macro_avg", {})
    weighted = by_name.get("weighted_avg", {})
    accuracy = by_name.get("accuracy", {})
    return {
        "accuracy": float(accuracy.get("f1") or 0.0),
        "balanced_accuracy": float(macro.get("recall") or 0.0),
        "macro_f1": float(macro.get("f1") or 0.0),
        "weighted_f1": float(weighted.get("f1") or 0.0),
    }


def _per_class_f1(report_path: Path) -> dict[str, float]:
    rows = _read_csv(report_path)
    result: dict[str, float] = {}
    for row in rows:
        name = row.get("class", "")
        if name in {"macro_avg", "weighted_avg", "accuracy"}:
            continue
        if row.get("f1") not in {None, ""}:
            result[f"f1_{name}"] = float(row["f1"])
    return result


def collect(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "eval_metrics.json"
        report_path = run_dir / "classification_report.csv"
        if not report_path.exists() and not metrics_path.exists():
            continue

        metrics = _read_json(metrics_path) if metrics_path.exists() else _metrics_from_report(report_path)
        if report_path.exists():
            report_metrics = _metrics_from_report(report_path)
            for key, value in report_metrics.items():
                metrics.setdefault(key, value)
            per_class = _per_class_f1(report_path)
        else:
            per_class = {}

        row = {
            "run_name": run_dir.name,
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
            "num_samples": int(metrics.get("num_samples", 0) or 0),
        }
        row.update(per_class)
        rows.append(row)
    return rows


def save_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["run_name", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "num_samples"]
    extra_fields = sorted({key for row in rows for key in row.keys() if key not in base_fields})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        writer.writerows(rows)


def save_bar(rows: list[dict[str, Any]], metric: str, title: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    names = [row["run_name"] for row in rows]
    values = [float(row.get(metric, 0.0)) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, values, color="#3f7cac")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
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
    rows.sort(key=lambda row: float(row.get("macro_f1", 0.0)), reverse=True)

    save_table(rows, assets_dir / "model_comparison_table.csv")
    save_bar(rows, "macro_f1", "Model Comparison: Macro F1", "Macro F1", assets_dir / "model_comparison_macro_f1.png")
    save_bar(
        rows,
        "balanced_accuracy",
        "Model Comparison: Balanced Accuracy",
        "Balanced Accuracy",
        assets_dir / "model_comparison_balanced_acc.png",
    )
    save_bar(rows, "accuracy", "Model Comparison: Accuracy", "Accuracy", assets_dir / "model_comparison_accuracy.png")
    print(f"Collected {len(rows)} runs.")
    print(f"Saved comparison outputs to: {assets_dir}")


if __name__ == "__main__":
    main()
