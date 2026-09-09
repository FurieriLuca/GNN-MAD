# GNN-MAD multi-agent target swapping

Stable GNN-MAD control for planar damped point-mass agents with fixed targets.
A local GNN supplies a bounded correction direction; a second GNN initializes
an autonomous stable LRU that supplies its magnitude. The communication radius
is 1.6, the GNN width is 48, and the LRU dimension is 32. Stability does not
guarantee collision avoidance.

## Install

Reference environment: Python 3.10.18, CPU, and the pinned dependencies below.

```bash
conda create -n swapping-gnn-mad python=3.10.18
conda activate swapping-gnn-mad
python -m pip install -r requirements-reference.txt
python -m pip install -e . --no-deps
make test
```

## Reproduce from scratch

```bash
make from-scratch
```

This trains from random weights, checks the reference tensor and training-log
hashes, evaluates the final model, and generates all eight GIFs. No pretrained
checkpoint is used as a training input.

| Stage | Training trajectories | Updates | Seed |
|---|---|---:|---:|
| 1 | Local avoidance teacher | 1,000 | 17 |
| 2 | Alternating teacher and learned policy | 600 | 27 |
| 3 | Learned policy | 1,000 | 47 |
| 4 | Dense random scenes and episode replay | 400 | 61 |

Each stage starts from the preceding stage's automatically selected validation
checkpoint. The [recipe](configs/from_scratch_recipe.json) and
[configuration](configs/train_random.yaml) specify the settings.
Stages 1–3 sample 4–12 agents with 80% random scenes and 20% paired exchanges.
Stage 4 uses 60% dense 10–12-agent random scenes, 20% smaller random scenes,
and 20% structured exchanges. Training uses 16-episode batches and 9-second
trajectories; replay preserves each episode's initial conditions and recurrence time.
Outputs are written to `runs/from_scratch/`: stage logs and checkpoints,
`selected.pt`, `reproduction.json`, `validation.json`, and `gifs/`.

`make from-scratch-smoke` runs two updates per stage. Existing output directories
are not overwritten. To run the full recipe elsewhere:

```bash
PYTHONPATH=src python scripts/train_from_scratch.py \
  --output-dir runs/another_run --verify-reference
```

The [reproduction record](results/reproduction.json) contains the reference
hashes and environment. Exact reproduction was verified with that CPU setup;
bitwise identity across platforms or dependency versions is not claimed.

## Results

Validation uses random initial positions, targets, and velocities with 6, 8,
10, or 12 agents. There are 32 cases per table entry. Success requires every
agent to finish within 0.30 of its target and every pair to remain at least
0.55 apart at all 261 sampled states over 13 simulated seconds.

| Agents | Random targets | Mixed clutter |
|---:|---:|---:|
| 6 | 28/32 | 29/32 |
| 8 | 28/32 | 24/32 |
| 10 | 20/32 | 17/32 |
| 12 | 18/32 | 14/32 |

Overall random-scene success is **178/256 (69.5%)**. Structured-scene success is
**91/160 (56.9%)**. Complete case metrics are in
[results/validation.json](results/validation.json).
These are validation results for one fixed training recipe; an independent
held-out test has not yet been run.

## Animations

One predefined seed is shown per agent-count/scenario pair, with no selection
by outcome. All five successes and three collision failures are retained.
Stars mark targets; circles mark agents; lines show communication links;
red outlines mark sampled collisions. Playback is real-time at 20 fps.
The [manifest](artifacts/gifs/manifest.json) records seeds, outcomes, and hashes.

<table>
  <tr>
    <td><img src="artifacts/gifs/N06_random_targets_seed5000000000.gif" width="360" alt="6 agents · random targets · success"><br>6 agents · random targets · success</td>
    <td><img src="artifacts/gifs/N06_mixed_clutter_targets_seed5000010000.gif" width="360" alt="6 agents · mixed clutter · success"><br>6 agents · mixed clutter · success</td>
  </tr>
  <tr>
    <td><img src="artifacts/gifs/N08_random_targets_seed5000020000.gif" width="360" alt="8 agents · random targets · success"><br>8 agents · random targets · success</td>
    <td><img src="artifacts/gifs/N08_mixed_clutter_targets_seed5000030000.gif" width="360" alt="8 agents · mixed clutter · success"><br>8 agents · mixed clutter · success</td>
  </tr>
  <tr>
    <td><img src="artifacts/gifs/N10_random_targets_seed5000040000.gif" width="360" alt="10 agents · random targets · collision failure"><br>10 agents · random targets · collision failure</td>
    <td><img src="artifacts/gifs/N10_mixed_clutter_targets_seed5000050000.gif" width="360" alt="10 agents · mixed clutter · success"><br>10 agents · mixed clutter · success</td>
  </tr>
  <tr>
    <td><img src="artifacts/gifs/N12_random_targets_seed5000060000.gif" width="360" alt="12 agents · random targets · collision failure"><br>12 agents · random targets · collision failure</td>
    <td><img src="artifacts/gifs/N12_mixed_clutter_targets_seed5000070000.gif" width="360" alt="12 agents · mixed clutter · collision failure"><br>12 agents · mixed clutter · collision failure</td>
  </tr>
</table>

To evaluate the included [model](artifacts/model.pt) or regenerate its GIFs
without retraining:

```bash
make evaluate
make gifs
```

These write `runs/validation.json` and `runs/gifs/`. Evaluation and rendering use
the learned policy alone, without teacher actions, safety filters, or trajectory
corrections.
