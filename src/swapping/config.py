from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def apply_override(config: dict[str, Any], override: str) -> None:
    key, value = override.split("=", 1)
    parts = key.split(".")
    cur = config
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = yaml.safe_load(value)


def resolved_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = deepcopy(load_config(path))
    for override in overrides or []:
        apply_override(cfg, override)
    return cfg

