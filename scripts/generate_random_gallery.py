"""Render a predeclared random seed per group, including any failures."""
import argparse
import json
from pathlib import Path

import torch

from swapping.benchmark import benchmark_keys, evaluate_policy, summarize, trajectory_metrics
from swapping.reproducibility import sha256_file
from rendering import load_policy, render_gif, rollout_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    policy = load_policy(args.checkpoint, device)
    keys = benchmark_keys("gallery", 1)
    result = evaluate_policy(policy, keys, horizon=260)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["protocol"] = {"selection": "first predeclared seed in every group; no outcome filtering",
                          "horizon": 260, "jitter": 0.0,
                          "stride": 1, "frame_duration_ms": 50,
                          "checkpoint_sha256": sha256_file(args.checkpoint)}
    for key, row in zip(keys, result["cases"]):
        starts, targets, trajectory = rollout_case(policy, key, 260, 0.0, device)
        row.update(trajectory_metrics(trajectory, torch.as_tensor(targets[None], device=device))[0])
        name = f"N{key.agents:02d}_{key.kind}_seed{key.seed}.gif"
        outcome = "success" if row["joint_success"] else "failure"
        render_gif(trajectory["positions"][:, 0].cpu().numpy(), targets,
                   args.output_dir / name, f"{key.agents} agents | seed {key.seed} | {outcome}",
                   policy.cfg.communication_radius, stride=1, frame_duration=50)
        row.update(gif=name, gif_sha256=sha256_file(args.output_dir / name))
        print(f"{name}: {outcome}", flush=True)
    result["summary"] = summarize(result["cases"])
    result["groups"] = {f"{row['agents']}_{row['kind']}": summarize([row]) for row in result["cases"]}
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
