PYTHON ?= python
MPLCONFIGDIR ?= /tmp/matplotlib-stable-swapping
CHECKPOINT ?= artifacts/model.pt

.PHONY: test from-scratch from-scratch-smoke evaluate gifs

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
