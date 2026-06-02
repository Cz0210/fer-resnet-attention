"""Unified ImageFolder data pipeline for FER and legacy project folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FER_CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TRAIN_CANDIDATES = ("train", "train_set", "training", "resnet_train_set", "vgg_train_set")
VAL_CANDIDATES = ("valid", "val", "verify_set", "validation", "vaild_set", "resnet_vaild_set", "vgg_vaild_set")
TEST_CANDIDATES = ("test", "test_set", "private_test", "private", "PrivateTest")


@dataclass
class DataBundle:
    train_loader: object | None
    val_loader: object | None
    test_loader: object | None
    class_names: list[str]
    class_counts: list[int]
    split_dirs: dict[str, Path]


def _find_split_dir(data_dir: Path, explicit: str | None, candidates: Sequence[str], required: bool) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = data_dir / path
        if path.exists():
            return path
        if required:
            raise FileNotFoundError(f"Split directory does not exist: {path}")
        return None

    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    if required:
        joined = ", ".join(candidates)
        raise FileNotFoundError(f"Could not find split under {data_dir}. Tried: {joined}")
    return None


def resolve_split_dirs(config: dict, require_train: bool = True, require_val: bool = True, require_test: bool = False) -> dict[str, Path]:
    data_cfg = config.get("data", {})
    data_dir = Path(data_cfg.get("data_dir", "face_images"))
    split_dirs: dict[str, Path] = {}
    train_dir = _find_split_dir(data_dir, data_cfg.get("train_dir"), TRAIN_CANDIDATES, require_train)
    val_dir = _find_split_dir(data_dir, data_cfg.get("val_dir"), VAL_CANDIDATES, require_val)
    test_dir = _find_split_dir(data_dir, data_cfg.get("test_dir"), TEST_CANDIDATES, require_test)
    if train_dir is not None:
        split_dirs["train"] = train_dir
    if val_dir is not None:
        split_dirs["valid"] = val_dir
    if test_dir is not None:
        split_dirs["test"] = test_dir
    return split_dirs


def build_transforms(config: dict, train: bool):
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

    common = [
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize((image_size, image_size)),
    ]

    if train:
        aug_cfg = data_cfg.get("augmentation", {})
        transform_list = common + [
            transforms.RandomHorizontalFlip(p=float(aug_cfg.get("horizontal_flip_p", 0.5))),
            transforms.RandomRotation(degrees=float(aug_cfg.get("rotation_degrees", 10))),
            transforms.RandomAffine(
                degrees=0,
                translate=tuple(aug_cfg.get("affine_translate", [0.05, 0.05])),
                scale=tuple(aug_cfg.get("affine_scale", [0.95, 1.05])),
                shear=float(aug_cfg.get("affine_shear", 5.0)),
            ),
            transforms.ColorJitter(
                brightness=float(aug_cfg.get("brightness", 0.2)),
                contrast=float(aug_cfg.get("contrast", 0.2)),
                saturation=float(aug_cfg.get("saturation", 0.05 if channels == 3 else 0.0)),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=float(aug_cfg.get("random_erasing_p", 0.25)),
                scale=tuple(aug_cfg.get("random_erasing_scale", [0.02, 0.12])),
                ratio=tuple(aug_cfg.get("random_erasing_ratio", [0.3, 3.3])),
                value="random",
            ),
        ]
        return transforms.Compose(transform_list)

    return transforms.Compose(common + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def _class_counts(targets: Sequence[int], num_classes: int) -> list[int]:
    counts = [0] * num_classes
    for target in targets:
        counts[int(target)] += 1
    return counts


def make_weighted_sampler(dataset):
    import torch
    from torch.utils.data import WeightedRandomSampler

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("WeightedRandomSampler requires a dataset with a `targets` attribute.")
    counts = _class_counts(targets, len(dataset.classes))
    class_weights = [1.0 / max(count, 1) for count in counts]
    sample_weights = torch.DoubleTensor([class_weights[int(target)] for target in targets])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_dataloaders(config: dict, splits: Sequence[str] = ("train", "valid")) -> DataBundle:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    split_dirs = resolve_split_dirs(
        config,
        require_train="train" in splits,
        require_val="valid" in splits,
        require_test="test" in splits,
    )

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", torch.cuda.is_available()))
    datasets = {}
    loaders = {}

    for split in splits:
        if split not in split_dirs:
            continue
        datasets[split] = ImageFolder(
            root=str(split_dirs[split]),
            transform=build_transforms(config, train=(split == "train")),
        )

    class_names = []
    class_counts = []
    if datasets:
        reference = datasets.get("train") or next(iter(datasets.values()))
        class_names = list(reference.classes)
        if "train" in datasets:
            class_counts = _class_counts(datasets["train"].targets, len(class_names))
        else:
            class_counts = _class_counts(reference.targets, len(class_names))

    for split, dataset in datasets.items():
        sampler = None
        shuffle = split == "train"
        if split == "train" and bool(loss_cfg.get("use_weighted_sampler", False)):
            sampler = make_weighted_sampler(dataset)
            shuffle = False
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    return DataBundle(
        train_loader=loaders.get("train"),
        val_loader=loaders.get("valid"),
        test_loader=loaders.get("test"),
        class_names=class_names,
        class_counts=class_counts,
        split_dirs=split_dirs,
    )

