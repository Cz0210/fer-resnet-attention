"""Webcam inference script for live FER demos."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run webcam facial expression recognition.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to checkpoint config.")
    parser.add_argument("--camera_index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    import cv2
    from PIL import Image

    from src.infer_image import load_model_for_inference, predict_pil

    model, config, class_names, device = load_model_for_inference(args.checkpoint, args.config, args.device)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    print("Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        pred, probabilities = predict_pil(model, config, class_names, image, device)
        confidence = max(probabilities)
        cv2.putText(
            frame,
            f"{pred}: {confidence:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Facial Expression Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

