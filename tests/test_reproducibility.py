import pytest

torch = pytest.importorskip("torch")

import numpy as np

from swapping.scenarios import ScenarioKey, make_scenario


def test_scenario_same_seed_repeatability():
    a = make_scenario(ScenarioKey(6, "crossing", 123))
    b = make_scenario(ScenarioKey(6, "crossing", 123))
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_scenario_different_seed_changes():
    a = make_scenario(ScenarioKey(6, "crossing", 123))
    b = make_scenario(ScenarioKey(6, "crossing", 124))
    assert not np.array_equal(a[0], b[0])

