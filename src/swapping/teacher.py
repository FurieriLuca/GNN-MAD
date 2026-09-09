from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def avoidance_teacher(positions: Tensor, velocities: Tensor, targets: Tensor, radius: float) -> Tensor:
    agents = positions.shape[1]
    away = positions[:, :, None, :] - positions[:, None, :, :]
    relative_velocity = velocities[:, :, None, :] - velocities[:, None, :, :]
    distance_sq = away.square().sum(-1).clamp_min(1e-4)
    distance = torch.sqrt(distance_sq)
    eye = torch.eye(agents, device=positions.device, dtype=torch.bool)[None]
    mask = (distance < radius) & (~eye)
    base_acceleration = -velocities + (targets - positions)
    relative_base_acceleration = base_acceleration[:, :, None, :] - base_acceleration[:, None, :, :]
    h = distance_sq - 1.0
    h_dot = 2.0 * (away * relative_velocity).sum(-1)
    h_ddot_base = 2.0 * relative_velocity.square().sum(-1) + 2.0 * (away * relative_base_acceleration).sum(-1)
    required = F.relu(-(h_ddot_base + 12.0 * h_dot + 20.0 * h)) * mask
    repulsion = (required.unsqueeze(-1) * away / (4.0 * distance_sq.unsqueeze(-1))).sum(2)
    goal_direction = F.normalize(targets - positions, dim=-1, eps=1e-6)
    tangent = torch.stack([-goal_direction[..., 1], goal_direction[..., 0]], dim=-1)
    proximity = ((radius - distance) / radius).clamp_min(0.0) * mask
    crowd = proximity.sum(2, keepdim=True).clamp_max(2.0)
    return (repulsion + tangent * crowd).clamp(-15.0, 15.0)

