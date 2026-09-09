import math

import pytest

torch = pytest.importorskip("torch")

from swapping.dynamics import DynamicsConfig, step_positions_velocities
from swapping.policy import PolicyConfig, StableMADPolicy
from swapping.rollout import rollout


def test_dynamics_one_step_identity():
    p = torch.tensor([[[1.0, -1.0]]])
    v = torch.tensor([[[0.2, 0.4]]])
    u = torch.tensor([[[3.0, -2.0]]])
    cfg = DynamicsConfig(dt=0.05, damping=1.0)
    pn, vn = step_positions_velocities(p, v, u, cfg)
    expected_v = v + 0.05 * (-v + u)
    expected_p = p + 0.05 * expected_v
    assert torch.allclose(vn, expected_v)
    assert torch.allclose(pn, expected_p)


def test_action_identity_and_direction_bounds():
    policy = StableMADPolicy(PolicyConfig(hidden_dim=12, lru_dim=6))
    p = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    v = torch.zeros_like(p)
    g = torch.tensor([[[2.0, 0.0], [-1.0, 0.0]]])
    state = policy.initial_state(p, v, g)
    action, _, info = policy(p, v, g, state, first_step=True)
    assert torch.allclose(action, info["base"] + info["residual"])
    assert torch.all(info["direction"].abs() <= 1.0 + 1e-6)
    assert torch.linalg.vector_norm(info["direction"], dim=-1).max() <= math.sqrt(2) + 1e-6


def test_no_neighbor_gate_zeroes_residual_direction():
    policy = StableMADPolicy(PolicyConfig(hidden_dim=12, lru_dim=6, communication_radius=0.2))
    p = torch.tensor([[[0.0, 0.0], [5.0, 0.0]]])
    v = torch.zeros_like(p)
    g = torch.tensor([[[2.0, 0.0], [-1.0, 0.0]]])
    state = policy.initial_state(p, v, g)
    _, _, info = policy(p, v, g, state, first_step=True)
    assert info["adjacency"].sum() == 0
    assert torch.allclose(info["direction"], torch.zeros_like(info["direction"]))
    assert torch.allclose(info["residual"], torch.zeros_like(info["residual"]))


def test_lru_eigenvalues_inside_configured_bounds():
    cfg = PolicyConfig(hidden_dim=12, lru_dim=6, r_min=0.90, r_max=0.95)
    policy = StableMADPolicy(cfg)
    radius = policy.lru.eigenvalues().abs()
    assert torch.all(radius > 0.90)
    assert torch.all(radius < 0.95)


def test_magnitude_autonomous_after_initialization():
    policy = StableMADPolicy(PolicyConfig(hidden_dim=12, lru_dim=6))
    p = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    v = torch.zeros_like(p)
    g = torch.tensor([[[2.0, 0.0], [-1.0, 0.0]]])
    state = policy.initial_state(p, v, g)
    changed_p = p + 100.0
    m1, _ = policy.lru.step(state.lru)
    m2, _ = policy.lru.step(state.lru)
    assert torch.allclose(m1, m2)
    _, _, info = policy(changed_p, v, g, state, first_step=False)
    assert torch.allclose(info["magnitude"], m1)


def test_gnn_lru_magnitude_autonomous_after_initialization():
    policy = StableMADPolicy(PolicyConfig(hidden_dim=12, lru_dim=6, magnitude_arch="gnn_lru"))
    p = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]])
    v = torch.zeros_like(p)
    g = torch.tensor([[[2.0, 0.0], [-1.0, 0.0], [4.0, 0.0]]])
    state = policy.initial_state(p, v, g)
    changed_p = p + 100.0
    m1, _ = policy.lru.step(state.lru)
    _, _, info = policy(changed_p, v, g, state, first_step=False)
    assert torch.allclose(info["magnitude"], m1)


def test_gnn_lru_magnitude_permutation_equivariant_initialization():
    policy = StableMADPolicy(PolicyConfig(hidden_dim=12, lru_dim=6, magnitude_arch="gnn_lru"))
    p = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]])
    v = torch.zeros_like(p)
    g = torch.tensor([[[2.0, 0.0], [-1.0, 0.0], [4.0, 0.0]]])
    state = policy.initial_state(p, v, g)
    perm = torch.tensor([2, 0, 1])
    inv = torch.argsort(perm)
    state_p = policy.initial_state(p[:, perm], v[:, perm], g[:, perm])
    assert torch.allclose(state.magnitude, state_p.magnitude[:, inv], atol=1e-5)


def test_rollout_finite():
    policy = StableMADPolicy(PolicyConfig(hidden_dim=12, lru_dim=6))
    p = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    v = torch.zeros_like(p)
    g = torch.tensor([[[2.0, 0.0], [-1.0, 0.0]]])
    traj = rollout(policy, p, g, v, horizon=4)
    for value in traj.values():
        assert torch.isfinite(value).all()
