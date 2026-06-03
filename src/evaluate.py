"""Evaluate a trained checkpoint and export reports."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

PPT_CLASS_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate facial expression recognition checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to checkpoint config.")
    parser.add_argument("--data_dir", default=None, help="Override config data.data_dir.")
    parser.add_argument("--split", default="test", choices=["test", "valid"], help="Split to evaluate.")
    parser.add_argument("--output_dir", default=None, help="Directory for reports. Defaults to checkpoint directory.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    return parser


def _disable_pretrained_download(config: dict) -> dict:
    config = copy.deepcopy(config)
    if "pretrained" in config:
        config["pretrained"] = False
    if "weights" in config:
        config["weights"] = None
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        if "pretrained" in model_cfg:
            model_cfg["pretrained"] = False
        if "weights" in model_cfg:
            model_cfg["weights"] = None
    return config


def _load_checkpoint(path: str | Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _predict(model, loader, device):
    import torch

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[list[float]] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            probs.extend(prob.cpu().tolist())
    return y_true, y_pred, probs


def _write_predictions(path: Path, image_paths, y_true, y_pred, probs, class_names) -> None:
    prob_order = [name for name in PPT_CLASS_ORDER if name in class_names]
    prob_order += [name for name in class_names if name not in prob_order]
    prob_fields = [f"prob_{name}" for name in prob_order]
    fieldnames = ["image_path", "label", "pred", "confidence"] + prob_fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image_path, label, pred, probability in zip(image_paths, y_true, y_pred, probs):
            confidence = float(max(probability)) if probability else 0.0
            prob_by_name = {class_names[idx]: float(probability[idx]) for idx in range(len(class_names))}
            row = {
                "image_path": image_path,
                "label": class_names[int(label)],
                "pred": class_names[int(pred)],
                "confidence": confidence,
            }
            for class_name in prob_order:
                row[f"prob_{class_name}"] = prob_by_name.get(class_name, 0.0)
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    import torch

    from src.data.dataset import build_dataloaders
    from src.models.build_model import build_model
    from src.utils.config import load_config, set_nested
    from src.utils.io import ensure_dir
    from src.utils.metrics import (
        classification_stats,
        confusion_matrix_np,
        save_classification_report,
        save_confusion_matrix,
        summarize_predictions,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(args.checkpoint, device)
    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint.get("config")
        if config is None:
            config_path = Path(args.checkpoint).with_name("config.yaml")
            config = load_config(config_path)
    config = _disable_pretrained_download(config)
    if args.data_dir is not None:
        set_nested(config, ("data", "data_dir"), args.data_dir)
    if args.batch_size is not None:
        set_nested(config, ("train", "batch_size"), args.batch_size)

    splits = (args.split,)
    if args.split == "valid":
        bundle = build_dataloaders(config, splits=("valid",))
        loader = bundle.val_loader
    else:
        bundle = build_dataloaders(config, splits=splits)
        loader = bundle.test_loader
    if loader is None:
        raise RuntimeError(f"Could not build loader for split: {args.split}")

    class_names = checkpoint.get("class_names") or bundle.class_names
    model = build_model(config, num_classes=len(class_names)).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    y_true, y_pred, probs = _predict(model, loader, device)
    matrix = confusion_matrix_np(y_true, y_pred, len(class_names))
    report_rows = classification_stats(matrix, class_names)
    summary = summarize_predictions(y_true, y_pred, len(class_names))

    out_dir = ensure_dir(args.output_dir or Path(args.checkpoint).parent)
    save_classification_report(out_dir / "classification_report.csv", report_rows)
    save_confusion_matrix(out_dir / "confusion_matrix.csv", matrix, class_names)
    image_paths = [sample[0] for sample in loader.dataset.samples]
    _write_predictions(out_dir / "predictions.csv", image_paths, y_true, y_pred, probs, class_names)
    eval_metrics = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "num_samples": len(y_true),
        "class_names": list(class_names),
        **summary,
    }
    with (out_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    print(f"accuracy: {summary['accuracy']:.4f}")
    print(f"balanced_accuracy: {summary['balanced_accuracy']:.4f}")
    print(f"macro_f1: {summary['macro_f1']:.4f}")
    print(f"weighted_f1: {summary['weighted_f1']:.4f}")
    print(f"Reports saved to: {out_dir}")


if __name__ == "__main__":
    main()
