import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).resolve().parents[1]/"scripts/analyze_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def scene(distances):
    p = torch.zeros(len(distances), 1, 2, 2, dtype=torch.float64)
    p[:, 0, 1, 0] = torch.tensor(distances, dtype=torch.float64)
    return p


def test_threshold_endpoints_and_late_onset():
    p = scene([0.55, 0.8, 0.4])
    row = analysis.separation_metrics(p, p[-1])[0]
    assert row["violating_state_indices"] == [2]
    assert row["first_violation_time_s"] == pytest.approx(0.1)
    assert row["violating_state_percent"] == pytest.approx(100/3)
    assert row["violating_pair_state_percent"] == pytest.approx(100/3)
    assert row["max_threshold_violation"] == pytest.approx(0.15)
    assert row["all_targets_reached"] and not row["joint_success"]
    initial = analysis.separation_metrics(scene([0.1, 0.8]), p[-1])[0]
    assert initial["first_violation_time_s"] == 0


def test_unordered_pair_denominator_and_safe_null():
    p = torch.tensor([[[[0., 0.], [0.2, 0.], [2., 0.]]]])
    row = analysis.separation_metrics(p, p[-1])[0]
    assert row["unordered_pairs"] == 3
    assert row["violating_states"] == 1
    assert row["violating_pair_states"] == 1
    assert row["violating_pair_state_percent"] == pytest.approx(100/3)
    safe = analysis.separation_metrics(scene([0.55, 0.9]), scene([0.9])[-1])[0]
    assert safe["first_violation_time_s"] is None
    assert safe["max_threshold_violation"] == 0
    assert safe["joint_success"]
    pooled = analysis.aggregate([dict(row, agents=3), dict(safe, agents=2)])
    assert pooled["violating_state_percent"] == pytest.approx(100/3)
    assert pooled["violating_pair_state_percent"] == 20


def test_nonfinite_rollout_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        analysis.separation_metrics(scene([float("nan")]), scene([1.0])[-1])


def test_released_model_supports_backward_without_weight_updates():
    from swapping.checkpointing import load_checkpoint
    from swapping.policy import PolicyConfig, StableMADPolicy
    from swapping.pretrain import predict_at_times
    root = Path(__file__).resolve().parents[1]
    saved = load_checkpoint(root/"artifacts/model.pt")
    policy = StableMADPolicy(PolicyConfig(**saved["resolved_config"]["model"]))
    policy.load_state_dict(saved["model"])
    original = {k: v.clone() for k, v in policy.state_dict().items()}
    p = torch.tensor([[[0., 0.], [1., 0.]]])
    v = torch.zeros_like(p)
    g = torch.flip(p, [1])
    prediction = predict_at_times(policy, p, g, v, p.repeat(3, 1, 1, 1),
                                  v.repeat(3, 1, 1, 1), torch.tensor([0, 1, 2]))
    prediction.square().mean().backward()
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in policy.parameters())
    assert any(param.grad.abs().sum() > 0 for param in policy.parameters())
    assert all(torch.equal(value, original[name]) for name, value in policy.state_dict().items())
