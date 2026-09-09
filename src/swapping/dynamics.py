from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DynamicsConfig:
    dt: float = 0.05
    damping: float = 1.0


def step_positions_velocities(
    positions: Tensor,
    velocities: Tensor,
    actions: Tensor,
    cfg: DynamicsConfig,
) -> tuple[Tensor, Tensor]:
    """Legacy-faithful damped point-mass update.

    v_{t+1} = v_t + dt (-d v_t + u_t)
    p_{t+1} = p_t + dt v_{t+1}
    """

    next_velocities = velocities + cfg.dt * (-cfg.damping * velocities + actions)
    next_positions = positions + cfg.dt * next_velocities
    return next_positions, next_velocities

