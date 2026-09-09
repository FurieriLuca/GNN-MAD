from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .graph import RadiusGraphEncoder, linear
from .magnitude import SignedStableLRU


@dataclass
class PolicyState:
    lru: Tensor
    magnitude: Tensor


@dataclass(frozen=True)
class PolicyConfig:
    hidden_dim: int = 48
    lru_dim: int = 32
    communication_radius: float = 1.6
    kp: float = 1.0
    residual_scale: float = 5.0
    r_min: float = 0.97
    r_max: float = 0.995
    aggregation: str = "sqrt_degree_sum"
    no_neighbor_gate: bool = True
    magnitude_arch: str = "local_lru"


class StableMADPolicy(nn.Module):
    feature_dim = 9

    def __init__(self, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        # For magnitude_arch="local_lru", keep module names:
        # direction_gnn, magnitude_encoder, direction_head, lru.  The final
        # reproducible config uses magnitude_arch="gnn_lru", where the magnitude
        # LRU input is v_i(0)=GNN_i(x(0), G_0), followed by zero input for t>0.
        if cfg.magnitude_arch not in {"local_lru", "gnn_lru"}:
            raise ValueError(f"unknown magnitude_arch {cfg.magnitude_arch!r}")
        self.direction_gnn = RadiusGraphEncoder(self.feature_dim, cfg.hidden_dim, zero_preserving=False, aggregation=cfg.aggregation)
        if cfg.magnitude_arch == "local_lru":
            self.magnitude_encoder = nn.Sequential(
                linear(self.feature_dim, cfg.hidden_dim, bias=False),
                nn.Tanh(),
                linear(cfg.hidden_dim, cfg.hidden_dim, bias=False),
                nn.Tanh(),
            )
        else:
            self.magnitude_gnn = RadiusGraphEncoder(self.feature_dim, cfg.hidden_dim, zero_preserving=True, aggregation=cfg.aggregation)
        self.direction_head = nn.Sequential(linear(cfg.hidden_dim, cfg.hidden_dim, True), nn.Tanh(), linear(cfg.hidden_dim, 2, True, gain=1.20), nn.Tanh())
        self.lru = SignedStableLRU(cfg.hidden_dim, cfg.lru_dim, r_min=cfg.r_min, r_max=cfg.r_max)

    @staticmethod
    def features(positions: Tensor, velocities: Tensor, targets: Tensor) -> Tensor:
        relative_goal = targets - positions
        goal_distance = torch.linalg.vector_norm(relative_goal, dim=-1, keepdim=True)
        return torch.cat([velocities / 2.0, positions / 4.0, targets / 4.0, relative_goal / 4.0, goal_distance / 4.0], dim=-1)

    def initial_state(self, positions: Tensor, velocities: Tensor, targets: Tensor) -> PolicyState:
        features = self.features(positions, velocities, targets)
        if self.cfg.magnitude_arch == "local_lru":
            embedding = self.magnitude_encoder(features)
        else:
            embedding, _ = self.magnitude_gnn(features, positions, velocities, targets, self.cfg.communication_radius)
        magnitude, lru = self.lru.initialize(embedding)
        return PolicyState(lru=lru, magnitude=magnitude)

    def forward(self, positions: Tensor, velocities: Tensor, targets: Tensor, state: PolicyState, first_step: bool = False):
        features = self.features(positions, velocities, targets)
        embedding, adjacency = self.direction_gnn(features, positions, velocities, targets, self.cfg.communication_radius)
        direction = self.direction_head(embedding)
        if self.cfg.no_neighbor_gate:
            direction = direction * adjacency.any(dim=-1, keepdim=True).to(direction.dtype)
        if first_step:
            magnitude, lru = state.magnitude, state.lru
        else:
            magnitude, lru = self.lru.step(state.lru)
        residual = self.cfg.residual_scale * magnitude * direction
        base = self.cfg.kp * (targets - positions)
        action = base + residual
        return action, PolicyState(lru=lru, magnitude=magnitude), {
            "base": base,
            "residual": residual,
            "direction": direction,
            "magnitude": magnitude,
            "adjacency": adjacency,
        }
