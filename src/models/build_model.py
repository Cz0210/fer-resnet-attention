"""Factory for model construction from YAML configs."""

from __future__ import annotations


def build_model(config: dict, num_classes: int | None = None):
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "resnet18")).lower()
    channels = 3 if bool(data_cfg.get("grayscale_to_rgb", True)) else 1
    classes = int(num_classes or model_cfg.get("num_classes", 7))

    if name in {"cnn", "cnn_baseline", "legacy_cnn"}:
        from src.models.simple_cnn import CNNBaseline

        return CNNBaseline(
            num_classes=classes,
            in_channels=channels,
            dropout=float(model_cfg.get("dropout", 0.4)),
        )

    if name in {"resnet18", "resnet34", "resnet18_cbam", "resnet34_cbam"}:
        from src.models.resnet_cbam import build_resnet_classifier

        attention_type = str(model_cfg.get("attention_type", "none")).lower()
        if name.endswith("_cbam") and attention_type == "none":
            attention_type = "cbam"
        arch = "resnet34" if "34" in name else "resnet18"
        return build_resnet_classifier(
            arch=arch,
            num_classes=classes,
            attention_type=attention_type,
            pretrained=bool(model_cfg.get("pretrained", True)),
            in_channels=channels,
        )

    raise ValueError(f"Unsupported model name: {name}")

