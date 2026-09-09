import json
from pathlib import Path

import numpy as np
import torch

from swapping.benchmark import benchmark_keys
from swapping.checkpointing import load_checkpoint
from swapping.refine import Episode, EpisodeReplay, danger_scores, new_test_keys, parser, refine, sample_training_batch, select_times, structured_keys
from swapping.scenarios import ScenarioKey


def test_coverage_variants_share_fresh_scenario_stream():
    streams = []
    for variant in ("coverage", "replay", "risk"):
        rng = np.random.default_rng(61)
        streams.append([sample_training_batch(variant, 2, rng)[-1] for _ in range(12)])
    assert streams[0] == streams[1] == streams[2]


def test_replay_preserves_initial_state_and_complete_time_axis():
    p = torch.randn(4, 2)
    trajectory = torch.randn(7, 4, 2)
    episode = Episode(ScenarioKey(4, "ring", 123), p, p+1, p*0, trajectory,
                      trajectory*0, trajectory+2, 0.8)
    replay = EpisodeReplay(2)
    rng = np.random.default_rng(3)
    replay.add(episode, rng)
    sampled = replay.sample(ScenarioKey(4, "ring", 999), rng)
    assert sampled is episode
    assert torch.equal(sampled.starts, p)
    assert torch.equal(sampled.positions, trajectory)
    assert replay.sample(ScenarioKey(6, "ring", 999), rng) is None
    assert replay.sample(ScenarioKey(4, "paired", 999), rng) is None
    for _ in range(5):
        replay.add(episode, rng)
    assert len(replay) == 2


def test_danger_is_predictive_and_local():
    positions = torch.tensor([[[[-0.6, 0.0], [0.6, 0.0]]]])
    approaching = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]])
    risk, _ = danger_scores(positions, approaching)
    retreat, _ = danger_scores(positions, -approaching)
    distant, _ = danger_scores(positions*10, approaching)
    assert risk.min() > retreat.max()
    assert distant.max() == 0
    times = select_times(risk.expand(12, 1, 2), 6, np.random.default_rng(3), True)
    assert len(times.unique()) == 6
    assert times.min() >= 0 and times.max() < 12


def test_new_holdout_is_disjoint():
    fresh = {key.seed for key in new_test_keys()}
    used = {key.seed for split in ("validation", "test", "gallery") for key in benchmark_keys(split, 128)}
    used |= {key.seed for key in structured_keys()}
    assert len(fresh) == 1024
    assert not fresh & used
    assert min(fresh) == 6_000_000_000


def test_refinement_repeatability(tmp_path):
    outputs = []
    for name in ("first", "second"):
        args = parser().parse_args(["--variant", "risk", "--output-dir", str(tmp_path/name),
                                   "--updates", "2", "--batch-size", "2", "--horizon", "6",
                                   "--sampled-times", "4", "--validation-count", "1",
                                   "--structured-count", "1", "--validation-horizon", "6",
                                   "--validation-interval", "1"])
        refine(args)
        outputs.append(load_checkpoint(args.output_dir / "latest.pt")["model"])
    assert all(torch.equal(outputs[0][name], outputs[1][name]) for name in outputs[0])
