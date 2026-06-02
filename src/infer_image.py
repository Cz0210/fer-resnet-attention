"""Single-image inference script."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference on a single facial expression image.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to checkpoint config.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of predictions to print.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    return parser


def load_model_for_inference(checkpoint_path, config_path=None, device_name="auto"):
    import torch

    from src.models.build_model import build_model
    from src.utils.config import load_config

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = load_config(config_path) if config_path else checkpoint.get("config")
    if config is None:
        config = load_config(Path(checkpoint_path).with_name("config.yaml"))
    class_names = checkpoint.get("class_names") or config.get("data", {}).get("class_names")
    model = build_model(config, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
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
