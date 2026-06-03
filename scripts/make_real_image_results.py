"""Create PPT-ready real-image prediction examples from predictions.csv."""

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

DEFAULT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make PPT real-image FER prediction examples.")
    parser.add_argument("--predictions", default="outputs/resnet18/predictions.csv", help="Path to predictions.csv.")
    parser.add_argument("--data_dir", default="face_images", help="ImageFolder data root.")
    parser.add_argument("--save_dir", default="assets/figures/ppt", help="Output directory for PPT figures.")
    parser.add_argument("--n", type=int, default=12, help="Number of images per figure.")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def class_names_from_data(data_dir: Path) -> list[str]:
    test_dir = data_dir / "test"
    if test_dir.exists():
        names = sorted(path.name for path in test_dir.iterdir() if path.is_dir())
        if names:
            return names
    return DEFAULT_CLASSES


def normalize_label(value: str | None, class_names: list[str]) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip()
    if value == "":
        return "unknown"
    if value.isdigit():
        idx = int(value)
        if 0 <= idx < len(class_names):
            return class_names[idx]
    return value


def parse_probabilities(row: dict[str, str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in row.items():
        if key.startswith("prob_") and value not in {None, ""}:
            try:
                result[key.removeprefix("prob_")] = float(value)
            except ValueError:
                pass
    if result:
        return result

    raw = row.get("probabilities")
    if raw:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return {str(key): float(value) for key, value in payload.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return result


def get_confidence(row: dict[str, str]) -> float:
    raw = row.get("confidence")
    if raw not in {None, ""}:
        try:
            return float(raw)
        except ValueError:
            pass
    probs = parse_probabilities(row)
    return max(probs.values()) if probs else 0.0


def resolve_image_path(raw_path: str, data_dir: Path) -> Path | None:
    if not raw_path:
        return None
    candidates: list[Path] = []
    path = Path(raw_path)
    candidates.append(path)
    candidates.append(data_dir / raw_path)
    candidates.append(data_dir / "test" / raw_path)

    parts = path.parts
    if "face_images" in parts:
        idx = parts.index("face_images")
        candidates.append(data_dir / Path(*parts[idx + 1 :]))
    if "test" in parts:
        idx = parts.index("test")
        candidates.append(data_dir / "test" / Path(*parts[idx + 1 :]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def prepare_rows(rows: list[dict[str, str]], data_dir: Path, class_names: list[str]) -> list[dict[str, Any]]:
    if not rows:
        return []
    fieldnames = set(rows[0].keys())
    if "image_path" not in fieldnames:
        raise ValueError("predictions.csv is missing image_path. Re-run src.evaluate after the latest update.")

    prepared = []
    for row in rows:
        image_path = resolve_image_path(row.get("image_path", ""), data_dir)
        if image_path is None:
            continue
        label = normalize_label(row.get("label"), class_names)
        pred = normalize_label(row.get("pred"), class_names)
        confidence = get_confidence(row)
        prepared.append(
            {
                "image_path": image_path,
                "label": label,
                "pred": pred,
                "confidence": confidence,
                "correct": label == pred,
            }
        )
    return prepared


def pick_examples(rows: list[dict[str, Any]], n: int, mode: str) -> list[dict[str, Any]]:
    if mode == "correct":
        pool = [row for row in rows if row["correct"]]
        pool.sort(key=lambda row: row["confidence"], reverse=True)
        return pool[:n]
    if mode == "wrong":
        pool = [row for row in rows if not row["correct"]]
        pool.sort(key=lambda row: row["confidence"], reverse=True)
        return pool[:n]
    correct = pick_examples(rows, max(n // 2, 1), "correct")
    wrong = pick_examples(rows, n - len(correct), "wrong")
    mixed = []
    for idx in range(max(len(correct), len(wrong))):
        if idx < len(correct):
            mixed.append(correct[idx])
        if idx < len(wrong):
            mixed.append(wrong[idx])
    return mixed[:n]


def save_grid(examples: list[dict[str, Any]], output: Path, title: str, n: int) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    output.parent.mkdir(parents=True, exist_ok=True)
    if not examples:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No examples available", ha="center", va="center", fontsize=16)
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output, dpi=200)
        plt.close(fig)
        return

    cols = 4
    rows = max((min(len(examples), n) + cols - 1) // cols, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    axes = list(getattr(axes, "flat", [axes]))
    fig.suptitle(title, fontsize=16)
    for ax, example in zip(axes, examples[:n]):
        image = Image.open(example["image_path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(
            f"True: {example['label']}\nPred: {example['pred']}\nConf: {example['confidence']:.2f}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(examples[:n]) :]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predictions = Path(args.predictions)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    if not predictions.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions}")
    class_names = class_names_from_data(data_dir)
    rows = prepare_rows(read_csv(predictions), data_dir, class_names)
    if not rows:
        raise RuntimeError("No prediction rows could be matched to local test images.")

    save_grid(
        pick_examples(rows, args.n, "correct"),
        save_dir / "ppt_resnet18_correct_examples.png",
        "ResNet18 Correct Predictions",
        args.n,
    )
    save_grid(
        pick_examples(rows, args.n, "wrong"),
        save_dir / "ppt_resnet18_wrong_examples.png",
        "ResNet18 Wrong Predictions",
        args.n,
    )
    save_grid(
        pick_examples(rows, args.n, "mixed"),
        save_dir / "ppt_resnet18_mixed_prediction_examples.png",
        "ResNet18 Mixed Prediction Examples",
        args.n,
    )
    print(f"Matched images: {len(rows)}")
    print(f"Figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
