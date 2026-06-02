"""Config loading and lightweight dictionary helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read YAML configs. Install it with `pip install pyyaml`."
        ) from exc
    return yaml


def load_config(path: str | Path) -> dict[str, Any]:
    yaml = _require_yaml()
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    yaml = _require_yaml()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False, allow_unicode=True)


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def get_nested(config: Mapping[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current

