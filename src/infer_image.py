"""Single-image inference script."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference on a single facial expression image.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to checkpoint config.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of predictions to print.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    return parser


def disable_pretrained_download(config: dict) -> dict:
    """Disable pretrained weight resolution for offline inference/evaluation."""
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


def _load_checkpoint_file(checkpoint_path: str | Path, device):
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _state_dict_from_checkpoint(checkpoint):
    if hasattr(checkpoint, "get"):
        return checkpoint.get("model_state_dict", checkpoint)
    return checkpoint


def _print_incompatible_keys(load_result) -> None:
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing:
        print(f"Missing checkpoint keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"Unexpected checkpoint keys ({len(unexpected)}): {unexpected}")


def load_model_for_inference(checkpoint_path, config_path=None, device_name="auto"):
    import torch

    from src.models.build_model import build_model
    from src.utils.config import load_config

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    checkpoint = _load_checkpoint_file(checkpoint_path, device)
    config = load_config(config_path) if config_path else checkpoint.get("config")
    if config is None:
        config = load_config(Path(checkpoint_path).with_name("config.yaml"))
    config = disable_pretrained_download(config)
    class_names = checkpoint.get("class_names") or config.get("data", {}).get("class_names")
    if not class_names:
        raise ValueError("Could not infer class names from checkpoint or config.")
    model = build_model(config, num_classes=len(class_names)).to(device)
    load_result = model.load_state_dict(_state_dict_from_checkpoint(checkpoint), strict=False)
    _print_incompatible_keys(load_result)
    model.eval()
    return model, config, class_names, device


def predict_pil(model, config, class_names, image, device):
    import torch

    from src.data.dataset import build_transforms

    transform = build_transforms(config, train=False)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    pred_idx = int(max(range(len(probabilities)), key=lambda idx: probabilities[idx]))
    return class_names[pred_idx], probabilities


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from PIL import Image

    model, config, class_names, device = load_model_for_inference(args.checkpoint, args.config, args.device)
    image = Image.open(args.image).convert("RGB")
    pred, probabilities = predict_pil(model, config, class_names, image, device)
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)[: args.top_k]
    print(f"Prediction: {pred}")
    for idx, score in ranked:
        print(f"{class_names[idx]}: {score:.4f}")


if __name__ == "__main__":
    main()
