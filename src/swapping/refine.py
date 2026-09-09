"""On-policy training with dense-scene sampling and episode-level replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .benchmark import benchmark_keys, evaluate_policy
from .checkpointing import load_checkpoint, save_checkpoint
from .config import resolved_config
from .policy import PolicyConfig, StableMADPolicy
from .pretrain import demonstrations, predict_at_times
from .reproducibility import runtime_manifest, set_global_seed, sha256_file, stable_json_hash
from .scenarios import ScenarioKey, sample_batch


@dataclass
class Episode:
    key: ScenarioKey
    starts: torch.Tensor
    targets: torch.Tensor
    velocities: torch.Tensor
    positions: torch.Tensor
    speeds: torch.Tensor
    labels: torch.Tensor
    priority: float


class EpisodeReplay:
    """Stratified replay; complete initial conditions and absolute time are retained."""
    def __init__(self, capacity_per_group=64):
        self.capacity = capacity_per_group
        self.groups = {}

    def add(self, episode, rng):
        group = self.groups.setdefault((episode.key.agents, episode.key.kind), [])
        if len(group) < self.capacity:
            group.append(episode)
        else:
            # Preserve difficult examples and a rotating sample of ordinary examples.
            candidates = np.argsort([e.priority for e in group])[:max(1, len(group)//2)]
            index = int(rng.choice(candidates)) if rng.random() < 0.75 else int(rng.integers(len(group)))
            group[index] = episode

    def sample(self, key, rng):
        group = self.groups.get((key.agents, key.kind), [])
        if not group:
            return None
        weights = np.asarray([0.10 + e.priority for e in group])
        return group[int(rng.choice(len(group), p=weights/weights.sum()))]

    def __len__(self):
        return sum(map(len, self.groups.values()))


def sample_training_batch(variant, batch_size, rng):
    if variant == "control":
        n = int(rng.choice((4, 6, 8, 10, 12)))
        kinds = ("random_targets", "random_targets", "mixed_clutter_targets", "mixed_clutter_targets", "paired")
    else:
        choice = rng.random()
        if choice < 0.60:
            n, kinds = int(rng.choice((10, 12))), ("random_targets", "mixed_clutter_targets")
        elif choice < 0.80:
            n, kinds = int(rng.choice((4, 6, 8))), ("random_targets", "mixed_clutter_targets")
        else:
            n = int(rng.choice((4, 6, 8, 10, 12)))
            kinds = ("paired", "wide_paired", "ring", "crossing")
    return sample_batch(batch_size, n, kinds, rng, torch.device("cpu"))


@torch.no_grad()
def danger_scores(positions, speeds, radius=1.6):
    delta = positions[:, :, :, None] - positions[:, :, None, :]
    relative = speeds[:, :, :, None] - speeds[:, :, None, :]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    n = positions.shape[2]
    eye = torch.eye(n, dtype=torch.bool, device=positions.device)
    minimum = distance.masked_fill(eye, float("inf")).amin(dim=(-1, -2))
    closest_time = (-(delta*relative).sum(-1)/relative.square().sum(-1).clamp_min(1e-6)).clamp(0, 1.0)
    closest = torch.linalg.vector_norm(delta + closest_time[..., None]*relative, dim=-1)
    risk = ((0.85-closest)/0.85).clamp(0, 1).square()
    risk = risk.masked_fill(eye | (distance > radius), 0)
    return risk.amax(-1), minimum


def select_times(risk, count, rng, focus):
    horizon = risk.shape[0]
    count = min(count, horizon)
    if not focus:
        return torch.as_tensor(np.sort(rng.choice(horizon, count, replace=False)))
    weights = risk.mean(dim=(1, 2)).cpu().numpy() + 1e-8
    probability = 0.5/horizon + 0.5*weights/weights.sum()
    return torch.as_tensor(np.sort(rng.choice(horizon, count, replace=False, p=probability)))


def structured_keys(count=4):
    return [ScenarioKey(n, kind, 3_100_000_000 + ni*10000 + ki*100 + i)
            for ni, n in enumerate((4, 6, 8, 10, 12))
            for ki, kind in enumerate(("paired", "wide_paired", "ring", "crossing"))
            for i in range(count)]


def new_test_keys(count=128):
    return [ScenarioKey(k.agents, k.kind, k.seed + 2_000_000_000)
            for k in benchmark_keys("test", count)]


@torch.no_grad()
def validation(policy, random_count=8, structured_count=4, horizon=260):
    random = evaluate_policy(policy, benchmark_keys("validation", random_count), horizon=horizon)
    structured = evaluate_policy(policy, structured_keys(structured_count), horizon=horizon)
    mean = random["summary"]["joint_success"]["rate"]
    worst = min(group["joint_success"]["rate"] for group in random["groups"].values())
    retained = structured["summary"]["joint_success"]["rate"]
    score = 100*(0.50*mean + 0.30*worst + 0.20*retained) - 0.001*random["summary"]["final_mean_error"]
    return {"validation_score": score, "random_success": mean, "worst_random_group": worst,
            "structured_success": retained, "final_error": random["summary"]["final_mean_error"]}


def refine(args):
    if (args.output_dir / "metrics.jsonl").exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    set_global_seed(args.seed, True)
    parent = load_checkpoint(args.parent, map_location="cpu")
    cfg = resolved_config("configs/train_random.yaml")
    for field in ("model", "environment"):
        if cfg[field] != parent["resolved_config"][field]:
            raise ValueError(f"Parent {field} differs")
    cfg["run"].update(output_dir=str(args.output_dir), seed=args.seed,
                      parent_checkpoint=str(args.parent), parent_sha256=sha256_file(args.parent))
    cfg["training"].update(method="episode_replay_refinement", variant=args.variant,
                          updates=args.updates, horizon=args.horizon, batch_size=args.batch_size,
                          replay_fraction=args.replay_fraction, capacity_per_group=args.capacity,
                          sampled_times=args.sampled_times, danger_weight=args.danger_weight,
                          validation_interval=args.validation_interval, validation_count=args.validation_count,
                          structured_validation_count=args.structured_count,
                          learning_rate_schedule="cosine, final factor 0.25")
    cfg["optimizer"]["learning_rate"] = args.learning_rate
    policy = StableMADPolicy(PolicyConfig(**cfg["model"]))
    policy.load_state_dict(parent["model"])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    data_rng, replay_rng, time_rng = [np.random.default_rng(args.seed + offset) for offset in (0, 10000, 20000)]
    replay = EpisodeReplay(args.capacity)
    repo = Path(__file__).resolve().parents[2]
    files = {str(p.relative_to(repo)): sha256_file(p) for p in sorted((repo / "src").rglob("*.py"))}
    source_hash = stable_json_hash(files)
    manifest = runtime_manifest(repo, cfg, sys.argv)
    manifest.update(source_files=files, source_tree_hash=source_hash, torch_threads=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolved_config.yaml").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    history = args.output_dir / "metrics.jsonl"

    def save(name, update, score):
        save_checkpoint(args.output_dir / name, model=policy, optimizer=optimizer, scheduler=None,
                        epoch=update, update=update, best_metric=score, resolved_config=cfg, hard_bank=None,
                        generator=data_rng, history_reference=str(history), parent_checkpoint=str(args.parent),
                        parent_sha256=cfg["run"]["parent_sha256"], git_commit=manifest["git"]["commit"],
                        source_tree_hash=source_hash)

    row = {"update": -1, **validation(policy, args.validation_count, args.structured_count, args.validation_horizon)}
    best = row["validation_score"]
    save("best_validation.pt", -1, best)
    history.write_text(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    for update in range(args.updates):
        policy.train()
        p, g, v, keys = sample_training_batch(args.variant, args.batch_size, data_rng)
        positions, speeds, labels = demonstrations(p, g, v, args.horizon, cfg, policy)
        _, separation = danger_scores(positions, speeds)
        fresh = [Episode(key, p[i].clone(), g[i].clone(), v[i].clone(), positions[:, i].clone(),
                         speeds[:, i].clone(), labels[:, i].clone(),
                         float(((0.80-separation[:, i].min())/0.80).clamp(0, 1))) for i, key in enumerate(keys)]
        episodes, replayed = list(fresh), 0
        if args.variant in ("replay", "risk"):
            for i, key in enumerate(keys):
                if replay_rng.random() < args.replay_fraction:
                    old = replay.sample(key, replay_rng)
                    if old is not None:
                        episodes[i], replayed = old, replayed+1
        p, g, v = [torch.stack([getattr(e, field) for e in episodes]) for field in ("starts", "targets", "velocities")]
        positions, speeds, labels = [torch.stack([getattr(e, field) for e in episodes], dim=1)
                                    for field in ("positions", "speeds", "labels")]
        risk, _ = danger_scores(positions, speeds)
        times = select_times(risk, args.sampled_times, time_rng, args.variant == "risk")
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_at_times(policy, p, g, v, positions, speeds, times)
        error = (prediction-labels[times]).square().mean(-1)
        weight = 1 + (args.danger_weight*risk[times] if args.variant == "risk" else torch.zeros_like(error))
        loss = (weight*error).sum()/weight.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        factor = 0.25 + 0.75*0.5*(1+np.cos(np.pi*update/max(1, args.updates-1)))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate*factor
        optimizer.step()
        if args.variant in ("replay", "risk"):
            for episode in fresh:
                replay.add(episode, replay_rng)
        row = {"update": update, "loss": float(loss.detach()), "agents": len(p[0]),
               "replayed": replayed, "buffer_size": len(replay), "learning_rate": optimizer.param_groups[0]["lr"]}
        do_validate = (update+1) % args.validation_interval == 0 or update == args.updates-1
        if do_validate:
            row.update(validation(policy, args.validation_count, args.structured_count, args.validation_horizon))
            if row["validation_score"] > best:
                best = row["validation_score"]
                save("best_validation.pt", update, best)
            save("latest.pt", update, best)
        with history.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if update % 50 == 0 or do_validate:
            print(json.dumps(row), flush=True)
    torch.save({"episodes": {str(k): [asdict(e) for e in es] for k, es in replay.groups.items()},
                "data_rng": data_rng.bit_generator.state, "replay_rng": replay_rng.bit_generator.state,
                "time_rng": time_rng.bit_generator.state}, args.output_dir / "replay_state.pt")
    return args.output_dir / "best_validation.pt"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("control", "coverage", "replay", "risk"), required=True)
    p.add_argument("--parent", type=Path, default=Path("artifacts/model.pt"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--horizon", type=int, default=180)
    p.add_argument("--sampled-times", type=int, default=16)
    p.add_argument("--capacity", type=int, default=64)
    p.add_argument("--replay-fraction", type=float, default=0.5)
    p.add_argument("--danger-weight", type=float, default=4.0)
    p.add_argument("--validation-interval", type=int, default=100)
    p.add_argument("--validation-count", type=int, default=8)
    p.add_argument("--structured-count", type=int, default=4)
    p.add_argument("--validation-horizon", type=int, default=260)
    return p


if __name__ == "__main__":
    print(refine(parser().parse_args()))
