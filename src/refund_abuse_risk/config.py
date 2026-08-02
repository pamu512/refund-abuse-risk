from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_DIR = _ROOT / "config"


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def default_config_dir() -> Path:
    return _DEFAULT_CONFIG_DIR


def load_policy(config_dir: Path | None = None) -> dict[str, Any]:
    return load_yaml((config_dir or default_config_dir()) / "policy.default.yaml")


def load_label_weights(config_dir: Path | None = None) -> dict[str, Any]:
    return load_yaml((config_dir or default_config_dir()) / "label_weights.default.yaml")


def load_operating_point(config_dir: Path | None = None) -> dict[str, Any]:
    return load_yaml((config_dir or default_config_dir()) / "operating_point.default.yaml")


def resolve_policy(
    policy: dict[str, Any],
    *,
    market: str,
    vertical: str,
    entity_type: str = "user",
) -> dict[str, Any]:
    """Merge defaults with matching market × vertical × entity_type overlays."""
    resolved = deep_merge({}, policy.get("defaults") or {})
    for overlay in policy.get("overlays") or []:
        if overlay.get("market") != market:
            continue
        if overlay.get("vertical") != vertical:
            continue
        ov_entity = overlay.get("entity_type")
        if ov_entity is not None and ov_entity != entity_type:
            continue
        resolved = deep_merge(resolved, overlay.get("patch") or {})
    return resolved
