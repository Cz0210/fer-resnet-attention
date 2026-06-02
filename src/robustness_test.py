"""Robustness evaluation under common image perturbations."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Callable

_cache_dir = Path(tempfile.gettempdir()) / "fer_cache"
_mpl_cache = _cache_dir / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FER robustness under image perturbations.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--test_dir", "--data_dir", dest="test_dir", required=True, help="ImageFolder test split directory.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to checkpoint config.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to checkpoint directory.")
    parser.add_argument("--assets_dir", default="assets/figures", help="Shared PPT figure directory.")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Evaluation device.")
    return parser


class BrightnessPerturbation:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, image):
        from PIL import ImageEnhance

        return ImageEnhance.Brightness(image).enhance(self.factor)


class ContrastPerturbation:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, image):
        from PIL import ImageEnhance

        return ImageEnhance.Contrast(image).enhance(self.factor)


class GaussianNoisePerturbation:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, image):
        import numpy as np
        from PIL import Image

        arr = np.asarray(image).astype("float32") / 255.0
        rng = np.random.default_rng(12345)
        noisy = np.clip(arr + rng.normal(0.0, self.sigma, size=arr.shape), 0.0, 1.0)
        return Image.fromarray((noisy * 255).astype("uint8"), mode=image.mode)


class BlurPerturbation:
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def __call__(self, image):
        from PIL import ImageFilter

        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


class RotationPerturbation:
    def __init__(self, degrees: float) -> None:
        self.degrees = degrees

    def __call__(self, image):
        fill = 0 if image.mode == "L" else (0, 0, 0)
        return image.rotate(self.degrees, resample=2, expand=False, fillcolor=fill)


PERTURBATIONS: dict[str, list[tuple[int, float, Callable[[], object]]]] = {
    "brightness": [
        (1, 0.85, lambda: BrightnessPerturbation(0.85)),
        (2, 0.70, lambda: BrightnessPerturbation(0.70)),
        (3, 0.55, lambda: BrightnessPerturbation(0.55)),
        (4, 0.40, lambda: BrightnessPerturbation(0.40)),
    ],
    "contrast": [
        (1, 0.85, lambda: ContrastPerturbation(0.85)),
        (2, 0.70, lambda: ContrastPerturbation(0.70)),
        (3, 0.55, lambda: ContrastPerturbation(0.55)),
        (4, 0.40, lambda: ContrastPerturbation(0.40)),
    ],
    "gaussian_noise": [
        (1, 0.03, lambda: GaussianNoisePerturbation(0.03)),
        (2, 0.06, lambda: GaussianNoisePerturbation(0.06)),
        (3, 0.10, lambda: GaussianNoisePerturbation(0.10)),
        (4, 0.15, lambda: GaussianNoisePerturbation(0.15)),
    ],
    "blur": [
        (1, 0.5, lambda: BlurPerturbation(0.5)),
        (2, 1.0, lambda: BlurPerturbation(1.0)),
        (3, 1.5, lambda: BlurPerturbation(1.5)),
        (4, 2.0, lambda: BlurPerturbation(2.0)),
    ],
    "rotation": [
        (1, 5.0, lambda: RotationPerturbation(5.0)),
        (2, 10.0, lambda: RotationPerturbation(10.0)),
        (3, 15.0, lambda: RotationPerturbation(15.0)),
        (4, 20.0, lambda: RotationPerturbation(20.0)),
    ],
}

FIGURE_NAMES = {
    "brightness": "robustness_brightness.png",
    "contrast": "robustness_contrast.png",
    "gaussian_noise": "robustness_noise.png",
    "blur": "robustness_blur.png",
    "rotation": "robustness_rotation.png",
}


def _load_checkpoint(path: str | Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_config(checkpoint: dict, checkpoint_path: Path, config_path: str | None) -> dict:
    from src.utils.config import load_config

    if config_path:
        return load_config(config_path)
    config = checkpoint.get("config")
    if config is not None:
        return config
    return load_config(checkpoint_path.with_name("config.yaml"))


def _transform(config: dict, perturbation):
    from torchvision import transforms

    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    grayscale_to_rgb = bool(data_cfg.get("grayscale_to_rgb", True))
    channels = 3 if grayscale_to_rgb else 1
    if channels == 3:
        mean = data_cfg.get("mean", [0.485, 0.456, 0.406])
        std = data_cfg.get("std", [0.229, 0.224, 0.225])
    else:
        mean = data_cfg.get("mean", [0.5])
        std = data_cfg.get("std", [0.5])

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=channels),
            transforms.Resize((image_size, image_size)),
            perturbation,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _make_loader(test_dir: Path, config: dict, perturbation, batch_size: int, num_workers: int):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(root=str(test_dir), transform=_transform(config, perturbation))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _evaluate(model, loader, device, num_classes: int) -> dict[str, float]:
    import torch

    from src.utils.metrics import summarize_predictions

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
    return summarize_predictions(y_true, y_pred, num_classes)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["perturbation", "level", "strength", "accuracy", "macro_f1", "balanced_accuracy"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig, paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)


def _plot_single(rows: list[dict[str, object]], perturbation: str, output_dir: Path, assets_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sub = [row for row in rows if row["perturbation"] == perturbation]
    levels = [int(row["level"]) for row in sub]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(levels, [float(row["accuracy"]) for row in sub], marker="o", linewidth=2, label="Accuracy")
    ax.plot(levels, [float(row["macro_f1"]) for row in sub], marker="o", linewidth=2, label="Macro F1")
    ax.plot(levels, [float(row["balanced_accuracy"]) for row in sub], marker="o", linewidth=2, label="Balanced Accuracy")
    ax.set_title(f"Robustness to {perturbation.replace('_', ' ').title()}")
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Score")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filename = FIGURE_NAMES[perturbation]
    _save_figure(fig, [output_dir / "figures" / filename, assets_dir / filename])
    plt.close(fig)


def _plot_summary(rows: list[dict[str, object]], output_dir: Path, assets_dir: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(PERTURBATIONS.keys())
    avg_macro = []
    avg_balanced = []
    strongest_acc = []
    for name in names:
        sub = [row for row in rows if row["perturbation"] == name]
        avg_macro.append(sum(float(row["macro_f1"]) for row in sub) / max(len(sub), 1))
        avg_balanced.append(sum(float(row["balanced_accuracy"]) for row in sub) / max(len(sub), 1))
        strongest = next((row for row in sub if int(row["level"]) == 4), sub[-1])
        strongest_acc.append(float(strongest["accuracy"]))

    x = range(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - width for i in x], avg_macro, width=width, label="Avg Macro F1", color="#3f7cac")
    ax.bar(list(x), avg_balanced, width=width, label="Avg Balanced Accuracy", color="#6aa84f")
    ax.bar([i + width for i in x], strongest_acc, width=width, label="Level-4 Accuracy", color="#c27c2c")
    ax.set_title("Robustness Summary")
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels([name.title() for name in names], rotation=20)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, [output_dir / "figures" / "robustness_summary.png", assets_dir / "robustness_summary.png"])
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from src.models.build_model import build_model
    from src.utils.io import ensure_dir

    checkpoint_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)
    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent)
    assets_dir = ensure_dir(args.assets_dir)
    device = _device(args.device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    config = _load_config(checkpoint, checkpoint_path, args.config)

    clean_loader = _make_loader(test_dir, config, lambda image: image, args.batch_size, args.num_workers)
    class_names = checkpoint.get("class_names") or clean_loader.dataset.classes
    model = build_model(config, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    rows: list[dict[str, object]] = []
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test dir: {test_dir}")
    print(f"Classes: {class_names}")
    for perturbation, specs in PERTURBATIONS.items():
        for level, strength, factory in specs:
            loader = _make_loader(test_dir, config, factory(), args.batch_size, args.num_workers)
            metrics = _evaluate(model, loader, device, len(class_names))
            row = {
                "perturbation": perturbation,
                "level": level,
                "strength": strength,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
            rows.append(row)
            print(
                f"{perturbation:10s} level={level} strength={strength} "
                f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                f"balanced_acc={metrics['balanced_accuracy']:.4f}"
            )

    _write_csv(output_dir / "robustness_results.csv", rows)
    for perturbation in PERTURBATIONS:
        _plot_single(rows, perturbation, output_dir, assets_dir)
    _plot_summary(rows, output_dir, assets_dir)
    print(f"Robustness CSV saved to: {output_dir / 'robustness_results.csv'}")
    print(f"Robustness figures saved to: {output_dir / 'figures'} and {assets_dir}")


if __name__ == "__main__":
    main()
