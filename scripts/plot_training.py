"""Plot original stage logs after verifying their recorded history hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read_histories(log_dir, reference):
    histories = []
    for stage in reference["stages"]:
        path = log_dir/stage["name"]/"metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != stage["training_history_sha256"]:
            raise ValueError(f"Training history does not match the reference: {path}")
        histories.append(rows)
    return histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path,
                        help="Directory containing <stage name>/metrics.jsonl for all four stages")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/figures/training.png")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads((ROOT/"results/reproduction.json").read_text())
    histories = read_histories(args.log_dir, reference)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), layout="constrained")
    names = ("Teacher imitation", "Mixed imitation", "On-policy imitation", "Dense replay")
    for i, (stage, rows, name) in enumerate(zip(reference["stages"], histories, names)):
        loss = [r for r in rows if "loss" in r]
        axes[0, i].plot([r["update"]+1 for r in loss], [r["loss"] for r in loss], lw=.7, color="#245778")
        axes[0, i].set_title(f"Stage {i+1}: {name}", fontsize=10)
        metrics = [("validation_joint_success", "Random")] if i < 3 else [
            ("random_success", "Random"), ("structured_success", "Structured"),
            ("worst_random_group", "Worst random group")]
        for field, label in metrics:
            points = [r for r in rows if field in r]
            axes[1, i].plot([r["update"]+1 for r in points], [100*r[field] for r in points],
                            ".-", lw=1, ms=4, label=label)
        axes[1, i].set_ylim(-2, 102)
        axes[1, i].set_xlabel("Stage updates completed")
        axes[1, i].legend(fontsize=7, loc="best")
        for ax in axes[:, i]:
            ax.axvline(stage["selected_update"]+1, color="#b44835", linestyle="--", lw=1)
            ax.set_xlim(0, stage["updates"])
            ax.grid(alpha=.18)
    axes[0, 0].set_ylabel("Training loss (unsmoothed)")
    axes[1, 0].set_ylabel("Validation strict success (%)")
    fig.suptitle("Stage boundaries separate panels; dashed lines mark selected checkpoints", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
