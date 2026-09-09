"""Evaluate random and structured target-swapping scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from swapping.benchmark import benchmark_keys, evaluate_policy
from swapping.checkpointing import load_checkpoint
from swapping.dynamics import DynamicsConfig
from swapping.policy import PolicyConfig, StableMADPolicy
from swapping.refine import structured_keys
from swapping.reproducibility import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    saved = load_checkpoint(args.checkpoint)
    config = saved["resolved_config"]
    policy = StableMADPolicy(PolicyConfig(**config["model"]))
    policy.load_state_dict(saved["model"])
    dynamics = DynamicsConfig(**config["environment"])
    suites = {
        "random_validation": benchmark_keys("validation", 32),
        "structured_validation": structured_keys(8),
    }
    result = {"checkpoint": str(args.checkpoint),
              "checkpoint_sha256": sha256_file(args.checkpoint),
              "global_update": saved["global_update"],
              "protocol": {"horizon": 260, "target_tolerance": 0.30,
                           "collision_distance": 0.55, "jitter": 0.0,
                           "final_test_evaluated": False}}
    for name, keys in suites.items():
        result[name] = evaluate_policy(policy, keys, horizon=260, dynamics=dynamics)
    random = result["random_validation"]
    mean = random["summary"]["joint_success"]["rate"]
    worst = min(g["joint_success"]["rate"] for g in random["groups"].values())
    structured = result["structured_validation"]["summary"]["joint_success"]["rate"]
    result["comparison"] = {
        "random_success": mean, "worst_random_group": worst,
        "structured_success": structured,
        "validation_score": 100*(0.50*mean + 0.30*worst + 0.20*structured)
                            - 0.001*random["summary"]["final_mean_error"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint": str(args.checkpoint), **result["comparison"]}), flush=True)


if __name__ == "__main__":
    main()
