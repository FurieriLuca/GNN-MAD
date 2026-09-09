PYTHON ?= python
MPLCONFIGDIR ?= /tmp/matplotlib-stable-swapping
CHECKPOINT ?= artifacts/model.pt

.PHONY: test check-release from-scratch from-scratch-smoke evaluate gifs analyze training-curves

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

from-scratch:
	PYTHONPATH=src MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/train_from_scratch.py --verify-reference

from-scratch-smoke:
	PYTHONPATH=src $(PYTHON) scripts/train_from_scratch.py --smoke --output-dir runs/smoke

evaluate:
	PYTHONPATH=src $(PYTHON) scripts/evaluate.py --checkpoint $(CHECKPOINT) --output runs/validation.json

gifs:
	PYTHONPATH=src MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/generate_random_gallery.py --checkpoint $(CHECKPOINT) --output-dir runs/gifs

check-release:
	PYTHONPATH=src MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) -m pytest -q -k 'not test_entire_chain_is_repeatable_and_has_no_external_checkpoint_input and not test_refinement_repeatability'

analyze:
	PYTHONPATH=src MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/analyze_results.py --checkpoint $(CHECKPOINT) --output-dir runs/analysis

training-curves:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/plot_training.py --log-dir $(LOG_DIR) --output artifacts/figures/training.png
