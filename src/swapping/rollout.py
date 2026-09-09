from __future__ import annotations

import torch
from torch import Tensor

from .dynamics import DynamicsConfig, step_positions_velocities
from .policy import StableMADPolicy
from .teacher import avoidance_teacher


def rollout(policy: StableMADPolicy, positions: Tensor, targets: Tensor, velocities: Tensor | None = None, horizon: int = 120, dynamics: DynamicsConfig = DynamicsConfig()):
    if velocities is None:
        velocities = torch.zeros_like(positions)
    pos, vel = positions, velocities
    policy_state = policy.initial_state(pos, vel, targets)
    logs = {k: [] for k in ("positions", "velocities", "actions", "residuals", "magnitudes", "neighbors", "teachers", "directions")}
    logs["positions"].append(pos)
    logs["velocities"].append(vel)
    for step in range(horizon):
        action, policy_state, info = policy(pos, vel, targets, policy_state, first_step=(step == 0))
        teacher = avoidance_teacher(pos, vel, targets, policy.cfg.communication_radius)
        pos, vel = step_positions_velocities(pos, vel, action, dynamics)
        logs["positions"].append(pos)
        logs["velocities"].append(vel)
        logs["actions"].append(action)
        logs["residuals"].append(info["residual"])
        logs["magnitudes"].append(info["magnitude"])
        logs["neighbors"].append(info["adjacency"].sum(-1))
        logs["teachers"].append(teacher)
        logs["directions"].append(info["direction"])
    return {key: torch.stack(values, dim=0) for key, values in logs.items()}

