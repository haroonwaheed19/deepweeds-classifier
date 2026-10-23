"""YAML config loading with dotted CLI overrides, e.g. --override train.lr=1e-3."""
from pathlib import Path
from typing import Any

import yaml


def _coerce(value: str) -> Any:
    # yaml.safe_load handles int/bool/null/list, but reads "3e-4" as a string (YAML 1.1
    # wants "3.0e-4"), which would silently break learning-rate overrides.
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        try:
            return float(parsed)
        except ValueError:
            pass
    return parsed


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"override path '{key}' does not exist in config (stopped at '{p}')")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"override key '{key}' does not exist in config")
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), _coerce(raw))
    return cfg
