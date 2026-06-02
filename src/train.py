"""Unified training entry point."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train facial expression recognition models.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--data_dir", default=None, help="Override config data.data_dir.")
    parser.add_argument("--output_dir", default=None, help="Base output directory. Defaults to config output_dir or outputs.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    from src.utils.config import set_nested

    if args.data_dir is not None:
        set_nested(config, ("data", "data_dir"), args.data_dir)
    if args.epochs is not None:
        set_nested(config, ("train", "epochs"), args.epochs)
    if args.batch_size is not None:
        set_nested(config, ("train", "batch_size"), args.batch_size)
    if args.lr is not None:
        set_nested(config, ("train", "lr"), args.lr)
    if args.seed is not None:
        set_nested(config, ("train", "seed"), args.seed)
    return config


def _make_optimizer(config: dict, model):
    import torch

    train_cfg = config.get("train", {})
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _make_scheduler(config: dict, optimizer, epochs: int):
    import torch

    train_cfg = config.get("train", {})
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    if name in {"none", "off", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("step_size", 15)),
            gamma=float(train_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def _run_epoch(model, loader, criterion, device, optimizer=None):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            total_items += batch_size
            preds = logits.argmax(dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total_items, 1)
    return avg_loss, y_true, y_pred


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "macro_f1", "balanced_acc", "lr"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    import torch

    from src.data.dataset import build_dataloaders
    from src.models.build_model import build_model
    from src.utils.config import load_config, save_config
    from src.utils.io import ensure_dir
    from src.utils.losses import build_loss, make_class_weights
    from src.utils.metrics import summarize_predictions
    from src.utils.seed import set_seed

    config = _apply_cli_overrides(load_config(args.config), args)
    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    output_base = Path(args.output_dir or config.get("output_dir", "outputs"))
    run_name = str(config.get("experiment", {}).get("run_name") or Path(args.config).stem)
    run_dir = ensure_dir(output_base / run_name)
    ensure_dir(run_dir / "figures")
    ensure_dir("logs")
    save_config(config, run_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataloaders(config, splits=("train", "valid"))
    if bundle.train_loader is None or bundle.val_loader is None:
        raise RuntimeError("Both train and valid splits are required for training.")

    model = build_model(config, num_classes=len(bundle.class_names)).to(device)
    class_weights = None
    if bool(config.get("loss", {}).get("use_class_weight", False)):
        class_weights = make_class_weights(bundle.class_counts, device)
    criterion = build_loss(config, class_weights=class_weights)
    optimizer = _make_optimizer(config, model)
    epochs = int(train_cfg.get("epochs", 50))
    scheduler = _make_scheduler(config, optimizer, epochs)

    best_macro_f1 = -1.0
    rows: list[dict[str, object]] = []
    start = time.time()
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Classes: {bundle.class_names}")
    print(f"Train split: {bundle.split_dirs.get('train')}")
    print(f"Valid split: {bundle.split_dirs.get('valid')}")

    for epoch in range(1, epochs + 1):
        train_loss, train_true, train_pred = _run_epoch(model, bundle.train_loader, criterion, device, optimizer)
        val_loss, val_true, val_pred = _run_epoch(model, bundle.val_loader, criterion, device)

        train_metrics = summarize_predictions(train_true, train_pred, len(bundle.class_names))
        val_metrics = summarize_predictions(val_true, val_pred, len(bundle.class_names))
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "balanced_acc": val_metrics["balanced_accuracy"],
            "lr": lr,
        }
        rows.append(row)
        _write_metrics(run_dir / "metrics.csv", rows)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_macro_f1": max(best_macro_f1, val_metrics["macro_f1"]),
            "config": config,
            "class_names": bundle.class_names,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            checkpoint["best_macro_f1"] = best_macro_f1
            torch.save(checkpoint, run_dir / "best.pt")

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

    elapsed = time.time() - start
    print(f"Finished in {elapsed / 60:.1f} min. Best val macro F1: {best_macro_f1:.4f}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()

