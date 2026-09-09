import pytest

torch = pytest.importorskip("torch")

from swapping.policy import PolicyConfig, StableMADPolicy


def _case():
    p = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]])
    v = torch.zeros_like(p)
    g = torch.tensor([[[2.0, 0.0], [-1.0, 0.0], [5.0, 0.0]]])
    return p, v, g


def test_permutation_equivariance_forward():
    policy = StableMADPolicy(PolicyConfig(hidden_dim=12, lru_dim=6, communication_radius=1.6))
    p, v, g = _case()
    state = policy.initial_state(p, v, g)
    action, _, _ = policy(p, v, g, state, first_step=True)
    perm = torch.tensor([2, 0, 1])
    inv = torch.argsort(perm)
    pp, vv, gg = p[:, perm], v[:, perm], g[:, perm]
    state_p = policy.initial_state(pp, vv, gg)
    action_p, _, _ = policy(pp, vv, gg, state_p, first_step=True)
    assert torch.allclose(action, action_p[:, inv], atol=1e-5)


def test_locality_current_non_neighbor_perturbation():
    policy = StableMADPolicy(PolicyConfig(hidden_dim=12, lru_dim=6, communication_radius=1.6))
    p, v, g = _case()
    state = policy.initial_state(p, v, g)
    _, _, info = policy(p, v, g, state, first_step=True)
    p2 = p.clone()
    p2[:, 2] += torch.tensor([100.0, 100.0])
    _, _, info2 = policy(p2, v, g, state, first_step=True)
    assert torch.allclose(info["direction"][:, 0], info2["direction"][:, 0], atol=1e-5)


def test_sqrt_degree_aggregation_scale_diagnostic_shape():
    policy = StableMADPolicy(PolicyConfig(hidden_dim=12, lru_dim=6, communication_radius=10.0))
    p = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
    v = torch.zeros_like(p)
    g = -p
    state = policy.initial_state(p, v, g)
    _, _, info = policy(p, v, g, state, first_step=True)
    assert info["adjacency"].shape == (1, 4, 4)
    assert torch.equal(info["adjacency"].sum(-1), torch.full((1, 4), 3))

