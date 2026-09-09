"""Validation score used by the supervised training stages."""
from .benchmark import benchmark_keys, evaluate_policy
from .dynamics import DynamicsConfig


def validate_random(policy, device, horizon, cfg):
    result = evaluate_policy(policy, benchmark_keys("validation", int(cfg["training"].get("validation_count", 8))),
                             horizon, dynamics=DynamicsConfig(**cfg["environment"]))
    summary = result["summary"]
    return {"validation_reward": 100.0 * summary["joint_success"]["rate"] - summary["final_mean_error"],
            "validation_joint_success": summary["joint_success"]["rate"],
            "validation_collision_free": summary["collision_free"]["rate"],
            "validation_final_error": summary["final_mean_error"],
            "validation_min_separation": summary["min_separation"]}
