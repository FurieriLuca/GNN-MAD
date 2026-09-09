import importlib.util
import json
from pathlib import Path

from swapping.checkpointing import load_checkpoint
from swapping.config import resolved_config
from swapping.reproducibility import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_architecture_and_final_model_match_reference():
    reference = json.loads((ROOT/"results/reproduction.json").read_text())
    for path, expected in reference["architecture_hashes"].items():
        assert sha256_file(ROOT/path) == expected, path
    spec = importlib.util.spec_from_file_location("release_recipe", ROOT/"scripts/train_from_scratch.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    saved = load_checkpoint(ROOT/"artifacts/model.pt")
    assert recipe.model_hash(saved["model"]) == reference["final_model_tensor_sha256"]
    config = resolved_config(ROOT/"configs/train_random.yaml")
    for field in ("model", "environment"):
        assert saved["resolved_config"][field] == config[field]


def test_gallery_has_every_predeclared_case_and_matching_hashes():
    from swapping.benchmark import benchmark_keys
    manifest = json.loads((ROOT/"artifacts/gifs/manifest.json").read_text())
    expected = {(k.agents, k.kind, k.seed) for k in benchmark_keys("gallery", 1)}
    actual = {(row["agents"], row["kind"], row["seed"]) for row in manifest["cases"]}
    assert actual == expected and len(manifest["cases"]) == 8
    for row in manifest["cases"]:
        assert sha256_file(ROOT/"artifacts/gifs"/row["gif"]) == row["gif_sha256"]
    assert manifest["protocol"]["stride"] == 1
    assert manifest["protocol"]["frame_duration_ms"] == 50
