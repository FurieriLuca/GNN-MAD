from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def linear(in_dim: int, out_dim: int, bias: bool, gain: float = 1.0) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


@dataclass(frozen=True)
class GraphConfig:
    communication_radius: float = 1.6
    aggregation: str = "sqrt_degree_sum"


class RadiusGraphEncoder(nn.Module):
    """Legacy-faithful dense one-hop radius message passing.

    For receiver i and sender j:
    rel_pos = p_j - p_i, rel_vel = v_j - v_i, rel_target = g_j - g_i.
    Messages are summed over neighbours and divided by sqrt(max(degree, 1)).
    """

    def __init__(self, node_dim: int, hidden_dim: int, zero_preserving: bool, aggregation: str = "sqrt_degree_sum"):
        super().__init__()
        if aggregation != "sqrt_degree_sum":
            raise ValueError("faithful baseline only supports sqrt_degree_sum aggregation")
        bias = not zero_preserving
        self.aggregation = aggregation
        self.node = nn.Sequential(linear(node_dim, hidden_dim, bias), nn.Tanh(), linear(hidden_dim, hidden_dim, bias), nn.Tanh())
        self.message = nn.Sequential(linear(hidden_dim + 6, hidden_dim, bias), nn.Tanh(), linear(hidden_dim, hidden_dim, bias), nn.Tanh())
        self.update = nn.Sequential(linear(2 * hidden_dim, hidden_dim, bias), nn.Tanh(), linear(hidden_dim, hidden_dim, bias), nn.Tanh())

    def forward(self, features: Tensor, positions: Tensor, velocities: Tensor, targets: Tensor, radius: float) -> tuple[Tensor, Tensor]:
        batch, agents, _ = positions.shape
        h = self.node(features)
        rel_pos = positions[:, None, :, :] - positions[:, :, None, :]
        rel_vel = velocities[:, None, :, :] - velocities[:, :, None, :]
        rel_target = targets[:, None, :, :] - targets[:, :, None, :]
        distances = torch.linalg.vector_norm(rel_pos, dim=-1)
        eye = torch.eye(agents, device=positions.device, dtype=torch.bool)[None]
        adjacency = (distances <= radius) & (~eye)
        sender_h = h[:, None, :, :].expand(batch, agents, agents, -1)
        geometry = torch.cat([rel_pos / radius, rel_vel / 2.0, rel_target / 4.0], dim=-1)
        messages = self.message(torch.cat([sender_h, geometry], dim=-1))
        mask = adjacency.unsqueeze(-1).to(messages.dtype)
        degree = mask.sum(dim=2).clamp_min(1.0)
        aggregate = (messages * mask).sum(dim=2) / torch.sqrt(degree)
        return self.update(torch.cat([h, aggregate], dim=-1)), adjacency

