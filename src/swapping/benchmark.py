"""Unfiltered, disjoint random-target benchmarks; no rejection by policy outcome."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .checkpointing import load_checkpoint
from .dynamics import DynamicsConfig
from .policy import PolicyConfig, StableMADPolicy
from .reproducibility import sha256_file
from .rollout import rollout
from .scenarios import ScenarioKey, make_initial_velocity, make_scenario

SPLIT_BASE = {"validation": 3_000_000_000, "test": 4_000_000_000, "gallery": 5_000_000_000}
KINDS = ("random_targets", "mixed_clutter_targets")
AGENTS = (6, 8, 10, 12)


def benchmark_keys(split="validation", count=16):
    if not 1 <= count <= 10000:
        raise ValueError("count must be in [1, 10000]")
    return [ScenarioKey(n, kind, SPLIT_BASE[split] + (ni * len(KINDS) + ki) * 10000 + i)
            for ni, n in enumerate(AGENTS) for ki, kind in enumerate(KINDS) for i in range(count)]


def trajectory_metrics(trajectory, targets):
    positions = trajectory["positions"]
    final = torch.linalg.vector_norm(positions[-1] - targets, dim=-1)
    n = targets.shape[1]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=targets.device), diagonal=1)
    distances = torch.linalg.vector_norm(positions[:, :, :, None] - positions[:, :, None, :], dim=-1)[..., upper]
    minimum = distances.amin(dim=(0, 2))
    reached = (final < 0.30).all(dim=-1)
    safe = minimum >= 0.55
    return [{"all_targets_reached": bool(reached[i]), "collision_free": bool(safe[i]),
             "joint_success": bool(reached[i] & safe[i]),
             "agent_success_rate": float((final[i] < 0.30).float().mean()),
             "final_mean_error": float(final[i].mean()), "min_separation": float(minimum[i])}
            for i in range(targets.shape[0])]


def summarize(rows):
    result = {"count": len(rows)}
    for name in ("all_targets_reached", "collision_free", "joint_success"):
        k, n = sum(r[name] for r in rows), len(rows)
        p, z = k / n, 1.96
        centre = (p + z*z/(2*n)) / (1 + z*z/n)
        radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        result[name] = {"successes": k, "rate": p, "wilson_95": [centre-radius, centre+radius]}
    result["agent_success_rate"] = float(np.mean([r["agent_success_rate"] for r in rows]))
    result["final_mean_error"] = float(np.mean([r["final_mean_error"] for r in rows]))
    result["min_separation"] = min(r["min_separation"] for r in rows)
    return result


@torch.no_grad()
def evaluate_policy(policy, keys, horizon=260, batch_size=16, dynamics=None):
    was_training = policy.training
    policy.eval()
    device = next(policy.parameters()).device
    rows = []
    try:
        for n in sorted({key.agents for key in keys}):
            group = [key for key in keys if key.agents == n]
            for offset in range(0, len(group), batch_size):
                batch = group[offset:offset+batch_size]
                scenarios = [make_scenario(key, jitter=0.0) for key in batch]
                starts = torch.as_tensor(np.stack([s for s, t in scenarios]), device=device)
                targets = torch.as_tensor(np.stack([t for s, t in scenarios]), device=device)
                velocities = torch.as_tensor(np.stack([make_initial_velocity(key, s, t)
                    for key, (s, t) in zip(batch, scenarios)]), device=device)
                trajectory = rollout(policy, starts, targets, velocities, horizon=horizon,
                                     dynamics=dynamics or DynamicsConfig())
                for key, metrics in zip(batch, trajectory_metrics(trajectory, targets)):
                    rows.append({"agents": key.agents, "kind": key.kind, "seed": key.seed, **metrics})
    finally:
        policy.train(was_training)
    groups = {f"{n}_{kind}": summarize([r for r in rows if r["agents"] == n and r["kind"] == kind])
              for n, kind in sorted({(r["agents"], r["kind"]) for r in rows})}
    return {"summary": summarize(rows), "groups": groups, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=SPLIT_BASE, default="validation")
    parser.add_argument("--count", type=int, default=16, help="cases per agent-count/family group")
    parser.add_argument("--horizon", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    saved = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    cfg = saved["resolved_config"]
    policy = StableMADPolicy(PolicyConfig(**cfg["model"]))
    policy.load_state_dict(saved["model"])
    result = evaluate_policy(policy, benchmark_keys(args.split, args.count), args.horizon,
                             args.batch_size, DynamicsConfig(**cfg["environment"]))
    result["protocol"] = {"split": args.split, "count_per_group": args.count, "horizon": args.horizon,
                          "jitter": 0.0, "target_tolerance": 0.30, "collision_distance": 0.55,
                          "checkpoint_sha256": sha256_file(args.checkpoint), "checkpoint": args.checkpoint}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": result["summary"], "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()
