"""Gradio demo for facial expression recognition."""

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
    parser = argparse.ArgumentParser(description="Launch the Facial Expression Recognition Demo.")
    parser.add_argument("--checkpoint", default="outputs/resnet18_cbam/best.pt", help="Path to a trained checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to checkpoint config.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--server_name", default="127.0.0.1", help="Gradio server host.")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    return parser


def _confidence_plot(class_names, probabilities):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(class_names, probabilities, color="#3f7cac")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Class Confidence")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def create_demo(checkpoint: str, config_path: str | None, device_name: str):
    import gradio as gr

    from src.gradcam_vis import gradcam_for_pil
    from src.infer_image import load_model_for_inference, predict_pil

    model, config, class_names, device = load_model_for_inference(checkpoint, config_path, device_name)

    def predict(image):
        if image is None:
            return "No image", None, None
        image = image.convert("RGB")
        pred, probabilities = predict_pil(model, config, class_names, image, device)
        cam_image, _ = gradcam_for_pil(model, config, image, device)
        confidence = max(probabilities)
        return f"{pred} ({confidence:.2%})", _confidence_plot(class_names, probabilities), cam_image

    with gr.Blocks(title="Facial Expression Recognition Demo") as demo:
        gr.Markdown("# Facial Expression Recognition Demo")
        with gr.Row():
            try:
                input_image = gr.Image(label="Upload or Camera", type="pil", sources=["upload", "webcam"])
            except TypeError:
                input_image = gr.Image(label="Upload or Camera", type="pil", source="upload")
            with gr.Column():
                prediction = gr.Textbox(label="Predicted Expression")
                confidence_plot = gr.Plot(label="Confidence Scores")
                gradcam_image = gr.Image(label="Grad-CAM", type="pil")
        run_button = gr.Button("Predict", variant="primary")
        run_button.click(predict, inputs=input_image, outputs=[prediction, confidence_plot, gradcam_image])
    return demo


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    demo = create_demo(args.checkpoint, args.config, args.device)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
