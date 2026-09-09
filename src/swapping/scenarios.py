from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


TRAIN_KINDS = ("free", "paired", "ring", "crossing", "permutation")


@dataclass(frozen=True, order=True)
class ScenarioKey:
    agents: int
    kind: str
    seed: int


def _rotation(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _separated_points(rng: np.random.Generator, count: int, extent: float = 3.2) -> np.ndarray:
    points: list[np.ndarray] = []
    for _ in range(5000):
        candidate = rng.uniform(-extent, extent, size=2)
        if all(np.linalg.norm(candidate - point) >= 0.65 for point in points):
            points.append(candidate)
            if len(points) == count:
                return np.stack(points)
    raise RuntimeError("Could not sample separated points")


def _project_minimum_separation(points: np.ndarray, minimum: float, rng: np.random.Generator) -> np.ndarray:
    points = points.copy()
    for _ in range(80):
        changed = False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                delta = points[i] - points[j]
                distance = np.linalg.norm(delta)
                if distance < minimum:
                    if distance < 1e-8:
                        angle = rng.uniform(-np.pi, np.pi)
                        direction = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        direction = delta / distance
                    shift = 0.5 * (minimum - distance + 1e-4) * direction
                    points[i] += shift
                    points[j] -= shift
                    changed = True
        if not changed:
            break
    return points


def make_scenario(key: ScenarioKey, jitter: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    n, kind = key.agents, key.kind
    rng = np.random.default_rng(key.seed)
    if n < 2:
        raise ValueError("Swapping requires at least two agents")

    if kind == "vanilla":
        if n != 12:
            raise ValueError("The original neurSLS layout has 12 agents")
        y = np.array([5, 3, 1, -1, -3, -5], dtype=float)
        starts = np.vstack([np.column_stack([-3 * np.ones(6), y]), np.column_stack([3 * np.ones(6), y])])
        targets = np.vstack([np.column_stack([3 * np.ones(6), y[::-1]]), np.column_stack([-3 * np.ones(6), y[::-1]])])
        return starts.astype(np.float32), targets.astype(np.float32)

    if kind in ("paired", "wide_paired"):
        if n % 2:
            kind = "ring"
        else:
            half = n // 2
            spacing = rng.uniform(1.45, 1.85) if kind == "wide_paired" else min(0.9, 5.0 / max(half - 1, 1))
            lanes = (np.arange(half) - (half - 1) / 2) * spacing
            separation = rng.uniform(2.7, 3.3) if kind == "wide_paired" else rng.uniform(2.4, 3.2)
            left = np.column_stack([-separation * np.ones(half), lanes])
            right = np.column_stack([separation * np.ones(half), lanes])
            starts = np.vstack([left, right])
            targets = np.vstack([right[::-1], left[::-1]])

    if kind in ("ring", "crossing"):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        angles += rng.uniform(-0.12, 0.12, size=n)
        radius = rng.uniform(2.4, 3.2)
        starts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        targets = np.roll(starts, max(1, n // 2), axis=0) if kind == "ring" else -starts
    elif kind == "permutation":
        starts = _separated_points(rng, n)
        targets = starts[np.roll(rng.permutation(n), 1)].copy()
    elif kind == "random_targets":
        starts = _separated_points(rng, n, extent=3.1)
        targets = _separated_points(rng, n, extent=3.1)
    elif kind == "clustered_targets":
        targets = _separated_points(rng, n, extent=2.35)
        starts = 1.85 * targets + rng.normal(0.0, 0.65, size=targets.shape)
        starts = _project_minimum_separation(starts, 0.75, rng)
    elif kind == "clustered_far_targets":
        targets = _separated_points(rng, n, extent=3.0)
        angles = rng.uniform(-np.pi, np.pi, size=n)
        radii = np.linspace(2.8, 7.0, n)
        rng.shuffle(radii)
        displacement = radii[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
        starts = targets + displacement
        starts = _project_minimum_separation(starts, 0.80, rng)
    elif kind == "clustered_swirl_targets":
        targets = _separated_points(rng, n, extent=2.75)
        center = targets.mean(axis=0, keepdims=True)
        centered = targets - center
        norm = np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 0.25)
        radial = centered / norm
        angle = rng.uniform(0.30, 0.55) * rng.choice([-1.0, 1.0])
        rotation = _rotation(angle)
        starts = center + (2.15 * centered + 2.2 * radial) @ rotation.T + rng.normal(0.0, 0.28, size=targets.shape)
        starts = _project_minimum_separation(starts, 0.80, rng)
    elif kind == "mixed_clutter_targets":
        targets = _separated_points(rng, n, extent=2.45)
        depth_classes = np.arange(n) % 3
        rng.shuffle(depth_classes)
        points = []
        for i, depth in enumerate(depth_classes):
            for _ in range(2000):
                if depth == 0:
                    angle = rng.uniform(-np.pi, np.pi)
                    radius = np.sqrt(rng.uniform(0.0, 1.0)) * 1.55
                    candidate = radius * np.array([np.cos(angle), np.sin(angle)])
                elif depth == 1:
                    angle = rng.uniform(-np.pi, np.pi)
                    radius = rng.uniform(2.0, 3.0)
                    candidate = targets[i] + radius * np.array([np.cos(angle), np.sin(angle)])
                else:
                    angle = rng.uniform(-np.pi, np.pi)
                    radius = rng.uniform(3.7, 4.8)
                    candidate = targets[i] + radius * np.array([np.cos(angle), np.sin(angle)])
                if all(np.linalg.norm(candidate - point) >= 0.80 for point in points):
                    points.append(candidate)
                    break
            else:
                raise RuntimeError("Could not sample mixed clutter starts")
        starts = np.stack(points)
    elif kind == "free":
        starts = _separated_points(rng, n, extent=2.7)
        radial = starts / np.maximum(np.linalg.norm(starts, axis=1, keepdims=True), 0.4)
        targets = starts + radial * rng.uniform(0.8, 1.8, size=(n, 1))
    elif kind.startswith("grid_movers_") or kind in (
        "grid_pair_swap", "grid_sparse_swap", "grid_medium_shuffle", "grid_heavy_shuffle",
        "grid_partial_shuffle", "grid_shuffle", "grid_shuffle_spaced",
    ):
        side = int(round(np.sqrt(n)))
        if side * side != n:
            raise ValueError("grid scenarios require a perfect-square agent count")
        spacing = 2.10 if kind == "grid_shuffle_spaced" else 1.05
        coords = (np.arange(side) - (side - 1) / 2.0) * spacing
        xx, yy = np.meshgrid(coords, coords)
        starts = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        identity = np.arange(n)
        permutation = identity.copy()
        if kind in ("grid_pair_swap", "grid_sparse_swap"):
            edges: list[tuple[int, int]] = []
            for row in range(side):
                for col in range(side):
                    i = row * side + col
                    if col + 1 < side:
                        edges.append((i, i + 1))
                    if row + 1 < side:
                        edges.append((i, i + side))
            rng.shuffle(edges)
            pair_count = 1 if kind == "grid_pair_swap" else max(2, side - 1)
            used: set[int] = set()
            for i, j in edges:
                if i in used or j in used:
                    continue
                permutation[i], permutation[j] = j, i
                used.update((i, j))
                if len(used) == 2 * pair_count:
                    break
        elif kind.startswith("grid_movers_") or kind in ("grid_medium_shuffle", "grid_heavy_shuffle", "grid_partial_shuffle"):
            if kind.startswith("grid_movers_"):
                movers = min(n - 1, max(2, int(kind.rsplit("_", 1)[1])))
            elif kind == "grid_medium_shuffle":
                movers = max(4, n // 2)
            elif kind == "grid_heavy_shuffle":
                movers = min(n - 1, max(6, int(round(0.75 * n))))
            else:
                movers = int(rng.integers(max(4, n // 2), max(5, n - 1)))
            selected = rng.choice(n, size=movers, replace=False)
            for _ in range(1000):
                shuffled = rng.permutation(selected)
                if np.all(shuffled != selected):
                    permutation[selected] = shuffled
                    break
            else:
                permutation[selected] = np.roll(selected, 1)
        else:
            for _ in range(1000):
                permutation = rng.permutation(n)
                if np.all(permutation != identity):
                    break
            else:
                permutation = np.roll(identity, 1)
        targets = starts[permutation].copy()
    elif kind not in ("paired", "wide_paired"):
        raise ValueError(f"Unknown scenario kind: {kind}")

    if kind != "free":
        rotation = _rotation(rng.uniform(-np.pi, np.pi))
        translation = rng.uniform(-0.45, 0.45, size=2)
        starts = starts @ rotation.T + translation
        targets = targets @ rotation.T + translation
    starts = starts + rng.normal(0.0, jitter, size=starts.shape)
    targets = targets + rng.normal(0.0, jitter * 0.35, size=targets.shape)
    starts = _project_minimum_separation(starts, 0.70, rng)
    targets = _project_minimum_separation(targets, 0.60, rng)
    return starts.astype(np.float32), targets.astype(np.float32)


def make_initial_velocity(key: ScenarioKey, starts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if key.kind == "vanilla" and key.agents == 12:
        vertical = np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5] * 2, dtype=np.float32)
        return np.column_stack([np.zeros(12, dtype=np.float32), vertical])
    rng = np.random.default_rng(key.seed + 7919)
    direction = targets - starts
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    speed = rng.uniform(0.0, 0.30, size=(key.agents, 1))
    return (speed * direction + rng.normal(0.0, 0.04, size=starts.shape)).astype(np.float32)


def sample_batch(
    batch_size: int,
    agents: int,
    kinds: Sequence[str],
    rng: np.random.Generator,
    device: torch.device,
    hard_bank=None,
    replay_probability: float = 0.25,
) -> tuple[Tensor, Tensor, Tensor, list[ScenarioKey]]:
    starts, targets, velocities, keys = [], [], [], []
    for _ in range(batch_size):
        key = None
        if hard_bank is not None and rng.random() < replay_probability:
            key = hard_bank.sample(agents, rng)
        if key is None:
            key = ScenarioKey(agents, str(rng.choice(kinds)), int(rng.integers(0, 2**31 - 1)))
        start, target = make_scenario(key)
        starts.append(start)
        targets.append(target)
        velocities.append(make_initial_velocity(key, start, target))
        keys.append(key)
    return (
        torch.as_tensor(np.stack(starts), device=device),
        torch.as_tensor(np.stack(targets), device=device),
        torch.as_tensor(np.stack(velocities), device=device),
        keys,
    )


class HardCaseBank:
    def __init__(self, capacity: int = 96):
        self.capacity = capacity
        self._scores: dict[ScenarioKey, float] = {}

    def update(self, keys: Sequence[ScenarioKey], costs: Sequence[float]) -> None:
        for key, cost in zip(keys, costs):
            self._scores[key] = max(float(cost), self._scores.get(key, -np.inf))
        if len(self._scores) > self.capacity:
            self._scores = dict(sorted(self._scores.items(), key=lambda item: item[1], reverse=True)[: self.capacity])

    def sample(self, agents: int, rng: np.random.Generator) -> ScenarioKey | None:
        candidates = [(key, score) for key, score in self._scores.items() if key.agents == agents]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1], reverse=True)
        ranks = np.arange(len(candidates), 0, -1, dtype=float)
        probabilities = ranks / ranks.sum()
        return candidates[int(rng.choice(len(candidates), p=probabilities))][0]

    def state_dict(self) -> list[dict]:
        return [{"agents": k.agents, "kind": k.kind, "seed": k.seed, "score": v} for k, v in self._scores.items()]

    def load_state_dict(self, entries: Sequence[dict]) -> None:
        self._scores = {ScenarioKey(int(e["agents"]), str(e["kind"]), int(e["seed"])): float(e["score"]) for e in entries}


def validation_keys() -> list[ScenarioKey]:
    return [
        ScenarioKey(4, "paired", 101),
        ScenarioKey(6, "crossing", 102),
        ScenarioKey(8, "ring", 103),
        ScenarioKey(8, "paired", 10800),
        ScenarioKey(10, "paired", 104),
        ScenarioKey(10, "paired", 11010),
        ScenarioKey(12, "vanilla", 0),
        ScenarioKey(12, "paired", 11220),
    ]
