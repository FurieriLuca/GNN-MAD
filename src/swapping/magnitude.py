from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .graph import linear


@dataclass(frozen=True)
class MagnitudeConfig:
    input_dim: int
    hidden_dim: int = 48
    lru_dim: int = 32
    r_min: float = 0.97
    r_max: float = 0.995


class SignedStableLRU(nn.Module):
    """Bias-free complex diagonal LRU with radii constrained at every forward pass."""

    def __init__(self, input_dim: int, hidden_dim: int, r_min: float = 0.97, r_max: float = 0.995):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.radius_logits = nn.Parameter(torch.zeros(hidden_dim))
        self.phase = nn.Parameter(torch.linspace(0.0, 2.0, hidden_dim))
        self.b_real = linear(input_dim, hidden_dim, bias=False, gain=0.7)
        self.b_imag = linear(input_dim, hidden_dim, bias=False, gain=0.7)
        self.c_real = linear(hidden_dim, 1, bias=False, gain=0.40)
        self.c_imag = linear(hidden_dim, 1, bias=False, gain=0.40)
        self.feedthrough = linear(input_dim, 1, bias=False, gain=0.80)

    def eigenvalues(self) -> Tensor:
        radius = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(self.radius_logits)
        return torch.polar(radius, self.phase)

    def _readout(self, state: Tensor) -> Tensor:
        return self.c_real(state.real) - self.c_imag(state.imag)

    def initialize(self, impulse: Tensor) -> tuple[Tensor, Tensor]:
        radius = self.eigenvalues().abs()
        gamma = torch.sqrt((1.0 - radius.square()).clamp_min(1e-6))
        state = torch.complex(self.b_real(impulse), self.b_imag(impulse)) * gamma
        magnitude = self._readout(state) + self.feedthrough(impulse)
        return magnitude, state

    def step(self, state: Tensor) -> tuple[Tensor, Tensor]:
        state = state * self.eigenvalues()
        return self._readout(state), state


class InitialDataMagnitude(nn.Module):
    def __init__(self, feature_dim: int = 9, hidden_dim: int = 48, lru_dim: int = 32, r_min: float = 0.97, r_max: float = 0.995):
        super().__init__()
        self.encoder = nn.Sequential(linear(feature_dim, hidden_dim, bias=False), nn.Tanh(), linear(hidden_dim, hidden_dim, bias=False), nn.Tanh())
        self.lru = SignedStableLRU(hidden_dim, lru_dim, r_min=r_min, r_max=r_max)

    def initialize(self, initial_features: Tensor) -> tuple[Tensor, Tensor]:
        return self.lru.initialize(self.encoder(initial_features))

    def step(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self.lru.step(state)

