"""Grad-CAM visualization for FER checkpoints."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_cache_dir = Path(tempfile.gettempdir()) / "fer_cache"
_mpl_cache = _cache_dir / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples for a trained FER model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--images", nargs="+", required=True, help="One or more input image paths.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to checkpoint config.")
    parser.add_argument("--output", default="gradcam_examples.png", help="Output image path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--max_images", type=int, default=8, help="Maximum number of images to show.")
    return parser


def get_default_target_layer(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4[-1]
    if hasattr(model, "features"):
        conv_layers = [module for module in model.features.modules() if module.__class__.__name__.lower().startswith("conv")]
        if conv_layers:
            return conv_layers[-1]
    raise ValueError("Could not infer a Grad-CAM target layer for this model.")


def generate_gradcam(model, input_tensor, target_layer, class_idx=None):
    import torch
    import torch.nn.functional as F

    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()
        grad = gradients[-1]
        act = activations[-1]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), int(class_idx)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def overlay_cam(image, cam, alpha=0.42):
    import numpy as np
    from PIL import Image
    import matplotlib

    image = image.convert("RGB")
    cam_image = Image.fromarray((cam * 255).astype("uint8")).resize(image.size)
    heatmap = matplotlib.colormaps.get_cmap("jet")(np.asarray(cam_image) / 255.0)[..., :3]
    base = np.asarray(image).astype("float32") / 255.0
    overlay = (1 - alpha) * base + alpha * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype("uint8")
    return Image.fromarray(overlay)


def gradcam_for_pil(model, config, image, device, class_idx=None):
    from src.data.dataset import build_transforms

    target_layer = get_default_target_layer(model)
    transform = build_transforms(config, train=False)
    tensor = transform(image).unsqueeze(0).to(device)
    cam, pred_idx = generate_gradcam(model, tensor, target_layer, class_idx=class_idx)
    return overlay_cam(image, cam), pred_idx


def _save_grid(images, labels, output):
    import matplotlib.pyplot as plt

    cols = min(len(images), 4)
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = list(getattr(axes, "flat", axes))
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from PIL import Image

    from src.infer_image import load_model_for_inference

    model, config, class_names, device = load_model_for_inference(args.checkpoint, args.config, args.device)
    output_images = []
    labels = []
    for path in args.images[: args.max_images]:
        image = Image.open(path).convert("RGB")
        overlay, pred_idx = gradcam_for_pil(model, config, image, device)
        output_images.append(overlay)
        labels.append(class_names[pred_idx])
    _save_grid(output_images, labels, args.output)
    print(f"Grad-CAM examples saved to: {args.output}")


if __name__ == "__main__":
    main()
