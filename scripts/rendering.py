from __future__ import annotations

from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from PIL import Image

from swapping.checkpointing import load_checkpoint
from swapping.policy import PolicyConfig, StableMADPolicy
from swapping.rollout import rollout
from swapping.scenarios import ScenarioKey, make_initial_velocity, make_scenario


def load_policy(checkpoint: Path, device: torch.device) -> StableMADPolicy:
    saved = load_checkpoint(checkpoint, map_location=device)
    model_cfg = saved["resolved_config"]["model"]
    policy = StableMADPolicy(PolicyConfig(**model_cfg)).to(device)
    policy.load_state_dict(saved["model"])
    policy.eval()
    return policy


@torch.no_grad()
def rollout_case(policy: StableMADPolicy, key: ScenarioKey, horizon: int, jitter: float, device: torch.device):
    starts, targets = make_scenario(key, jitter=jitter)
    velocities = make_initial_velocity(key, starts, targets)
    trajectory = rollout(
        policy,
        torch.as_tensor(starts[None], device=device),
        torch.as_tensor(targets[None], device=device),
        torch.as_tensor(velocities[None], device=device),
        horizon=horizon,
    )
    return starts, targets, trajectory


def render_gif(
    positions: np.ndarray,
    targets: np.ndarray,
    output: Path,
    title: str,
    communication_radius: float,
    stride: int = 4,
    tail_steps: int = 22,
    frame_duration: float = 0.075,
) -> None:
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, positions.shape[1]))
    points = np.concatenate([positions.reshape(-1, 2), targets], axis=0)
    lo, hi = points.min(0) - 0.7, points.max(0) + 0.7
    span = max(float((hi - lo).max()), 2.0)
    center = (hi + lo) / 2
    xlim = (center[0] - span / 2, center[0] + span / 2)
    ylim = (center[1] - span / 2, center[1] + span / 2)
    frame_ids = list(range(0, len(positions), stride))
    if frame_ids[-1] != len(positions) - 1:
        frame_ids.append(len(positions) - 1)
    frames = []
    for t in frame_ids:
        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=95)
        pairwise = np.linalg.norm(positions[t, :, None, :] - positions[t, None, :, :], axis=-1)
        collision_matrix = (pairwise < 0.55) & (pairwise > 0)
        colliding_agents = collision_matrix.any(axis=1)
        ax.set(xlim=xlim, ylim=ylim, aspect="equal", title=f"{title} — t={0.05 * t:.1f}s")
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=7)
        for i in range(positions.shape[1]):
            for j in range(i + 1, positions.shape[1]):
                distance = pairwise[i, j]
                if distance < communication_radius:
                    strength = np.clip((communication_radius - distance) / (0.35 * communication_radius), 0.0, 1.0)
                    ax.plot(
                        positions[t, [i, j], 0],
                        positions[t, [i, j], 1],
                        color="0.20",
                        alpha=0.75 * strength,
                        linewidth=0.4 + 1.4 * strength,
                        zorder=1,
                    )
        for i in range(positions.shape[1]):
            tail_start = max(0, t - tail_steps)
            tail = positions[tail_start : t + 1, i]
            if len(tail) >= 2:
                segments = np.stack([tail[:-1], tail[1:]], axis=1)
                segment_colors = np.repeat(colors[i][None], len(segments), axis=0)
                segment_colors[:, 3] = np.linspace(0.03, 0.68, len(segments))
                ax.add_collection(LineCollection(segments, colors=segment_colors, linewidths=1.2, zorder=2))
            ax.plot(targets[i, 0], targets[i, 1], marker="*", ms=9, color=colors[i])
            agent = plt.Circle(
                positions[t, i],
                0.275,
                facecolor=colors[i],
                alpha=0.85,
                edgecolor="red" if colliding_agents[i] else colors[i],
                linewidth=2.5 if colliding_agents[i] else 0.8,
                zorder=3,
            )
            ax.add_patch(agent)
        ax.text(0.01, 0.01, "circles: agents | fading lines: communication | stars: targets", transform=ax.transAxes, fontsize=7)
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        frames.append(np.asarray(Image.open(buffer).convert("RGB")))
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=frame_duration, loop=0)
