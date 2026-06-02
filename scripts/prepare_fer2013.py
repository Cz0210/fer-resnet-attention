"""Convert FER2013 CSV files to ImageFolder format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FER_CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "valid",
    "PrivateTest": "test",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert FER2013 CSV to ImageFolder folders.")
    parser.add_argument("--csv", required=True, help="Path to fer2013.csv.")
    parser.add_argument("--output_dir", required=True, help="Output directory for train/valid/test folders.")
    parser.add_argument("--image_size", type=int, default=48, help="FER image size. Default: 48.")
    return parser


def _save_image(pixels: str, path: Path, image_size: int) -> None:
    import numpy as np
    from PIL import Image

    values = np.fromstring(pixels, sep=" ", dtype=np.uint8)
    expected = image_size * image_size
    if values.size != expected:
        raise ValueError(f"Expected {expected} pixels, got {values.size}")
    image = values.reshape((image_size, image_size))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(path)


def convert(csv_path: str | Path, output_dir: str | Path, image_size: int = 48) -> None:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    counters = {split: 0 for split in USAGE_TO_SPLIT.values()}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"emotion", "pixels", "Usage"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"FER2013 CSV must contain columns: {sorted(required)}")
        for row in reader:
            label = int(row["emotion"])
            usage = row.get("Usage", "Training")
            split = USAGE_TO_SPLIT.get(usage, "train")
            class_name = FER_CLASS_NAMES[label]
            index = counters[split]
            counters[split] += 1
            image_path = output_dir / split / class_name / f"{split}_{index:06d}.png"
            _save_image(row["pixels"], image_path, image_size=image_size)

    for split, count in counters.items():
        print(f"{split}: {count} images")
    print(f"ImageFolder data saved to: {output_dir}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    convert(args.csv, args.output_dir, image_size=args.image_size)


if __name__ == "__main__":
    main()

