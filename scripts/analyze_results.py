"""Replay the fixed validation and gallery suites and measure sampled separation."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from swapping.benchmark import benchmark_keys, summarize, trajectory_metrics
from swapping.checkpointing import load_checkpoint
from swapping.dynamics import DynamicsConfig
from swapping.policy import PolicyConfig, StableMADPolicy
from swapping.refine import structured_keys
from swapping.reproducibility import sha256_file
from swapping.rollout import rollout
from swapping.scenarios import make_initial_velocity, make_scenario

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.55
HORIZON = 260


def separation_metrics(positions, targets, dt=0.05):
    """Count unordered pairs at every state, including both endpoints."""
    if positions.ndim != 4 or positions.shape[2] < 2:
        raise ValueError("Expected positions shaped [states, batch, agents >= 2, 2]")
    if not torch.isfinite(positions).all() or not torch.isfinite(targets).all():
        raise ValueError("Non-finite trajectory or target")
    n = positions.shape[2]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=positions.device), diagonal=1)
    distances = torch.linalg.vector_norm(
        positions[:, :, :, None] - positions[:, :, None, :], dim=-1)[..., upper]
    violations = distances < THRESHOLD
    rows = trajectory_metrics({"positions": positions}, targets)
    for i, row in enumerate(rows):
        any_pair = violations[:, i].any(dim=-1)
        indices = any_pair.nonzero().flatten().tolist()
        states, pairs = distances.shape[0], distances.shape[2]
        pair_count = int(violations[:, i].sum())
        final = torch.linalg.vector_norm(positions[-1, i] - targets[i], dim=-1)
        row.update(
            sampled_states=states, unordered_pairs=pairs,
            violating_states=len(indices), pair_state_observations=states*pairs,
            violating_pair_states=pair_count,
            violating_state_percent=100*len(indices)/states,
            violating_pair_state_percent=100*pair_count/(states*pairs),
            max_threshold_violation=max(0.0, THRESHOLD-row["min_separation"]),
            first_violation_time_s=indices[0]*dt if indices else None,
            violating_state_indices=indices,
            agents_reached=int((final < 0.30).sum()),
            final_max_error=float(final.max()),
        )
    return rows


def aggregate(rows):
    result = summarize(rows)
    for key in ("sampled_states", "violating_states", "pair_state_observations",
                "violating_pair_states", "agents_reached", "agents"):
        result[key] = sum(row[key] for row in rows)
    result["violating_state_percent"] = 100*result["violating_states"]/result["sampled_states"]
    result["violating_pair_state_percent"] = 100*result["violating_pair_states"]/result["pair_state_observations"]
    result["pooled_agent_goal_percent"] = 100*result["agents_reached"]/result["agents"]
    result["max_threshold_violation"] = max(row["max_threshold_violation"] for row in rows)
    onsets = [row["first_violation_time_s"] for row in rows if row["first_violation_time_s"] is not None]
    result["first_violation_time_s_among_violating_cases"] = {
        "count": len(onsets), "min": min(onsets) if onsets else None,
        "median": float(np.median(onsets)) if onsets else None,
        "max": max(onsets) if onsets else None,
    }
    return result


@torch.no_grad()
def evaluate(policy, keys, dynamics, batch_size=16):
    rows, traces = [], []
    for n in sorted({key.agents for key in keys}):
        group = [key for key in keys if key.agents == n]
        for offset in range(0, len(group), batch_size):
            batch = group[offset:offset+batch_size]
            scenes = [make_scenario(key, jitter=0.0) for key in batch]
            starts = torch.as_tensor(np.stack([p for p, g in scenes]))
            targets = torch.as_tensor(np.stack([g for p, g in scenes]))
            velocities = torch.as_tensor(np.stack([
                make_initial_velocity(key, p, g) for key, (p, g) in zip(batch, scenes)]))
            trajectory = rollout(policy, starts, targets, velocities, horizon=HORIZON, dynamics=dynamics)
            metrics = separation_metrics(trajectory["positions"], targets, dynamics.dt)
            for i, (key, row) in enumerate(zip(batch, metrics)):
                rows.append({"agents": n, "kind": key.kind, "seed": key.seed, **row})
                if batch_size == 1:
                    p = trajectory["positions"][:, i]
                    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
                    d = torch.linalg.vector_norm(p[:, :, None]-p[:, None, :], dim=-1)[:, upper]
                    traces.append(d.min(dim=-1).values.tolist())
    return {"summary": aggregate(rows), "groups": {
        f"{n}_{kind}": aggregate([r for r in rows if r["agents"] == n and r["kind"] == kind])
        for n, kind in sorted({(r["agents"], r["kind"]) for r in rows})}, "cases": rows}, traces


def compare(reference, current):
    old = {(r["agents"], r["kind"], r["seed"]): r for r in reference["cases"]}
    new = {(r["agents"], r["kind"], r["seed"]): r for r in current["cases"]}
    if old.keys() != new.keys():
        raise ValueError("Scenario keys differ from the recorded suite")
    flags = ("joint_success", "collision_free", "all_targets_reached")
    mismatches = [list(k) for k in old if any(old[k][f] != new[k][f] for f in flags)]
    return {"outcome_mismatches": mismatches, "max_absolute_metric_difference": {
        f: max(abs(old[k][f]-new[k][f]) for k in old)
        for f in ("min_separation", "final_mean_error", "agent_success_rate")}}


def plot_gallery(rows, traces, dt, path):
    fig, axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True, sharey=True, layout="constrained")
    for ax, row, trace in zip(axes.flat, rows, traces):
        t = np.arange(len(trace))*dt
        ax.plot(t, trace, color="#245778", linewidth=1.4)
        ax.axhline(THRESHOLD, color="#b44835", linestyle="--", linewidth=1)
        ax.fill_between(t, trace, THRESHOLD, where=np.asarray(trace)<THRESHOLD, color="#b44835", alpha=.3)
        label = "random targets" if row["kind"] == "random_targets" else "mixed clutter"
        ax.set_title(f"{row['agents']} agents · {label}", fontsize=10, loc="left")
        ax.grid(alpha=.18)
        ax.set_xlim(0, HORIZON*dt)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Minimum separation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"artifacts/model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"runs/analysis")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    saved = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = saved["resolved_config"]
    policy = StableMADPolicy(PolicyConfig(**cfg["model"]))
    policy.load_state_dict(saved["model"])
    policy.eval()
    dynamics = DynamicsConfig(**cfg["environment"])
    result = {"checkpoint_sha256": sha256_file(args.checkpoint),
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "numpy": np.__version__, "platform": platform.platform(), "device": "cpu", "threads": 1},
              "protocol": {"horizon_steps": HORIZON, "dt_s": dynamics.dt, "sampled_states_per_case": HORIZON+1,
                           "initial_and_final_state_included": True, "separation_threshold": THRESHOLD,
                           "violation_rule": "pair distance < 0.55; unordered distinct pairs",
                           "goal_rule": "final distance < 0.30 for each agent",
                           "state_denominator": "sum of sampled states across cases",
                           "pair_state_denominator": "sum of (sampled states * N*(N-1)/2) across cases",
                           "first_violation": "earliest sampled violation time; null if none",
                           "continuous_time_check": False, "final_test_evaluated": False,
                           "validation_batch_size": 16, "gallery_batch_size": 1, "jitter": 0.0}}
    refs = json.loads((ROOT/"results/validation.json").read_text())
    comparisons = {}
    for name, keys in (("random_validation", benchmark_keys("validation", 32)),
                       ("structured_validation", structured_keys(8)), ("gallery", benchmark_keys("gallery", 1))):
        suite, traces = evaluate(policy, keys, dynamics, 1 if name == "gallery" else 16)
        reference = json.loads((ROOT/"artifacts/gifs/manifest.json").read_text()) if name == "gallery" else refs[name]
        comparisons[name] = compare(reference, suite)
        result[name] = suite
        if name == "gallery":
            result["gallery_min_separation_by_state"] = traces
        print(f"{name}: {suite['summary']['joint_success']['successes']}/{len(keys)} strict successes", flush=True)
    result["reference_comparison"] = comparisons
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"separation.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    plot_gallery(result["gallery"]["cases"], result["gallery_min_separation_by_state"], dynamics.dt,
                 args.output_dir/"gallery_separation.png")
    if any(c["outcome_mismatches"] for c in comparisons.values()):
        raise RuntimeError("Outcome differences from reference; inspect separation.json")


if __name__ == "__main__":
    main()
