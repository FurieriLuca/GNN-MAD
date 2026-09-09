from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .reproducibility import sha256_file


SCHEMA_VERSION = 1


def rng_state(generator: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_generator": generator.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any], generator: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    generator.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    update: int,
    best_metric: float,
    resolved_config: dict[str, Any],
    hard_bank,
    generator: np.random.Generator,
    history_reference: str,
    parent_checkpoint: str | None,
    parent_sha256: str | None,
    git_commit: str | None,
    source_tree_hash: str | None,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_update": update,
        "best_validation_metric": best_metric,
        "resolved_config": resolved_config,
        "hard_cases": hard_bank.state_dict() if hard_bank is not None else [],
        "rng_state": rng_state(generator),
        "history_reference": history_reference,
        "parent_checkpoint": parent_checkpoint,
        "parent_sha256": parent_sha256,
        "git_commit": git_commit,
        "source_tree_hash": source_tree_hash,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location="cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def verify_parent(path: str | None, expected_sha256: str | None) -> None:
    if not path:
        return
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError(f"parent checkpoint hash mismatch: expected {expected_sha256}, got {actual}")

