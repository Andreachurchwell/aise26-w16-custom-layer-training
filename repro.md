# Reproducibility

This document describes exactly how to reproduce the training run and results for this project.

All experiments were run with a fixed random seed so that results are deterministic and repeatable.

---

## Random Seed

Seed used: `42`

In `train.py` the following are set:

- `random.seed(42)`
- `torch.manual_seed(42)`

This ensures:
- Weight initialization is the same across runs
- Data shuffling is consistent
- Reported metrics and plots are reproducible

---

## Environment Setup

Create and activate a virtual environment:

```
python -m venv .venv
source .venv/Scripts/activate  
```
Install dependencies:
```
python -m pip install -r requirements.txt
```
Run Training (3 epochs)

Train the model on FashionMNIST:
```
python train.py
```

This will:

- Download FashionMNIST (if not already present)

- Train for 3 epochs using AdamW and StepLR

- Print training and validation metrics per epoch

Run Tests

Run the sanity test for the custom layer:
```
pytest -q

```
Expected output:

1 passed

Notes:

- Training is CPU-only and should complete within a few minutes.

- Results may differ slightly if the seed is changed or if different hardware is used.
