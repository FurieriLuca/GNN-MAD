import importlib.util
from pathlib import Path

import torch

from swapping import pretrain, refine
from swapping.checkpointing import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_from_scratch", ROOT/"scripts/train_from_scratch.py")
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def test_entire_chain_is_repeatable_and_has_no_external_checkpoint_input(tmp_path, monkeypatch):
    def guarded_load(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve()), path
        return load_checkpoint(path, *args, **kwargs)
    monkeypatch.setattr(pretrain, "load_checkpoint", guarded_load)
    monkeypatch.setattr(refine, "load_checkpoint", guarded_load)
    monkeypatch.setattr(recipe, "load_checkpoint", guarded_load)
    first = recipe.train_chain(tmp_path/"first", smoke=True)
    second = recipe.train_chain(tmp_path/"second", smoke=True)
    assert first["external_checkpoint_inputs"] == second["external_checkpoint_inputs"] == []
    assert first["stages"][0]["parent"] is None
    assert len(first["stages"]) == 4
    for a, b in zip(first["stages"], second["stages"]):
        assert a["model_tensor_sha256"] == b["model_tensor_sha256"]
        assert a["selected_update"] == b["selected_update"]
    state = load_checkpoint(tmp_path/"first/selected.pt")["model"]
    assert all(torch.isfinite(value).all() for value in state.values())


def test_chain_refuses_to_overwrite_existing_directory(tmp_path):
    import pytest
    with pytest.raises(FileExistsError):
        recipe.train_chain(tmp_path, smoke=True)
