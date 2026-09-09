"""Supervised warm-start from the existing local teacher; deployment is unchanged."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .checkpointing import load_checkpoint, save_checkpoint
from .config import resolved_config
from .dynamics import DynamicsConfig, step_positions_velocities
from .policy import PolicyConfig, PolicyState, StableMADPolicy
from .reproducibility import runtime_manifest, set_global_seed, sha256_file, stable_json_hash
from .scenarios import sample_batch
from .teacher import avoidance_teacher
from .validation import validate_random


@torch.no_grad()
def demonstrations(starts, targets, velocities, horizon, cfg, sampling_policy=None):
    positions, speeds, labels = [], [], []
    p, v = starts, velocities
    state = sampling_policy.initial_state(p, v, targets) if sampling_policy is not None else None
    dynamics = DynamicsConfig(**cfg["environment"])
    for step in range(horizon):
        correction = avoidance_teacher(p, v, targets, cfg["model"]["communication_radius"])
        positions.append(p)
        speeds.append(v)
        labels.append(correction)
        action = cfg["model"]["kp"] * (targets-p) + correction
        if sampling_policy is not None:
            action, state, _ = sampling_policy(p, v, targets, state, first_step=(step == 0))
        p, v = step_positions_velocities(p, v, action, dynamics)
    return torch.stack(positions), torch.stack(speeds), torch.stack(labels)


def predict_at_times(policy, starts, targets, velocities, positions, speeds, times):
    """Evaluate the unchanged policy on demonstration states, with its own LRU clock."""
    state = policy.initial_state(starts, velocities, targets)
    magnitudes = [state.magnitude]
    lru = state.lru
    for _ in range(int(times.max())):
        magnitude, lru = policy.lru.step(lru)
        magnitudes.append(magnitude)
    k, batch, agents = len(times), starts.shape[0], starts.shape[1]
    magnitude = torch.stack(magnitudes)[times].reshape(k*batch, agents, 1)
    # first_step selects the explicitly computed magnitude for each sampled time.
    # The returned LRU state is unused; its recurrence above is fully differentiated.
    batched_state = PolicyState(state.lru.repeat(k, 1, 1), magnitude)
    _, _, info = policy(positions[times].reshape(k*batch, agents, 2),
                       speeds[times].reshape(k*batch, agents, 2), targets.repeat(k, 1, 1),
                       batched_state, first_step=True)
    return info["residual"].reshape(k, batch, agents, 2)


def pretrain(config_path, overrides, parent_checkpoint=None):
    cfg = resolved_config(config_path, overrides)
    cfg["training"]["method"] = "teacher_supervised_warm_start"
    torch.set_num_threads(int(cfg["run"].get("threads", 1)))
    set_global_seed(int(cfg["run"]["seed"]), cfg["run"].get("deterministic", True))
    device = torch.device(cfg["run"]["device"])
    output = Path(cfg["run"]["output_dir"])
    if (output / "metrics.jsonl").exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = StableMADPolicy(PolicyConfig(**cfg["model"])).to(device)
    if parent_checkpoint:
        parent = load_checkpoint(Path(parent_checkpoint), map_location=device)
        if parent["resolved_config"]["model"] != cfg["model"] or parent["resolved_config"]["environment"] != cfg["environment"]:
            raise ValueError("Parent architecture and dynamics must match")
        policy.load_state_dict(parent["model"])
        cfg["run"].update(parent_checkpoint=parent_checkpoint, parent_sha256=sha256_file(parent_checkpoint))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["optimizer"]["learning_rate"],
                                 weight_decay=cfg["optimizer"]["weight_decay"])
    repo = Path(__file__).resolve().parents[2]
    files = {str(p.relative_to(repo)): sha256_file(p) for p in sorted((repo / "src").rglob("*.py"))}
    source_hash = stable_json_hash(files)
    manifest = runtime_manifest(repo, cfg, sys.argv)
    manifest.update(source_files=files, source_tree_hash=source_hash, torch_threads=torch.get_num_threads())
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output / "resolved_config.yaml").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    generator = np.random.default_rng(cfg["run"]["seed"])
    history = output / "metrics.jsonl"
    updates, horizon = cfg["training"]["updates"], cfg["training"]["horizon"]
    source = cfg["training"].get("demonstration_source", "teacher")
    if source not in {"teacher", "mixed", "policy"}:
        raise ValueError(f"Unknown demonstration_source: {source}")
    best = -float("inf")
    for update in range(updates):
        agents = int(generator.choice((4, 6, 8, 10, 12)))
        p, g, v, _ = sample_batch(cfg["training"]["batch_size"], agents,
            ("random_targets", "random_targets", "mixed_clutter_targets", "mixed_clutter_targets", "paired"), generator, device)
        sampling_policy = policy if source == "policy" or (source == "mixed" and update % 2 == 1) else None
        positions, speeds, labels = demonstrations(p, g, v, horizon, cfg, sampling_policy)
        times = torch.as_tensor(np.sort(generator.choice(horizon, min(16, horizon), replace=False)), device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_at_times(policy, p, g, v, positions, speeds, times)
        loss = (prediction-labels[times]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite supervised loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["optimizer"]["max_grad_norm"])
        optimizer.step()
        row = {"update": update, "loss": float(loss.detach()), "agents": agents}
        validation = update % cfg["training"]["validation_interval"] == 0 or update == updates-1
        if validation:
            row.update(validate_random(policy, device, cfg["training"]["validation_horizon"], cfg))
        with history.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        save_best = validation and row["validation_reward"] > best
        if save_best:
            best = row["validation_reward"]
        names = (["best_validation.pt"] if save_best else []) + (["latest.pt"] if validation else [])
        for name in names:
            save_checkpoint(output / name, model=policy, optimizer=optimizer, scheduler=None,
                            epoch=update, update=update, best_metric=best, resolved_config=cfg,
                            hard_bank=None, generator=generator, history_reference=str(history),
                            parent_checkpoint=parent_checkpoint, parent_sha256=cfg["run"].get("parent_sha256"),
                            git_commit=manifest["git"]["commit"], source_tree_hash=source_hash)
        if update % cfg["training"]["log_interval"] == 0 or validation:
            print(json.dumps(row), flush=True)
    return output / "best_validation.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_random.yaml")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    print(pretrain(args.config, args.override, args.parent_checkpoint))


if __name__ == "__main__":
    main()
