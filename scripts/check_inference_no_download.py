"""Check that inference model construction does not download pretrained weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check offline inference model loading.")
    parser.add_argument("--config", required=True, help="Path to model config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt.")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"], help="Device for loading.")
    return parser


def _install_download_guard() -> None:
    def fail_download(*args, **kwargs):
        raise RuntimeError("Attempted to download pretrained weights during inference.")

    try:
        import torch.hub

        torch.hub.load_state_dict_from_url = fail_download
    except Exception:
        pass

    try:
        import torchvision.models._api as tv_api

        tv_api.load_state_dict_from_url = fail_download
    except Exception:
        pass


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    _install_download_guard()
    from src.infer_image import load_model_for_inference

    load_model_for_inference(args.checkpoint, args.config, args.device)
    print("Inference model loaded without downloading pretrained weights.")


if __name__ == "__main__":
    main()
