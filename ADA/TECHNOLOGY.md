# Technology and Concepts (Adagrad)

## Stack
- Python 3.9–3.12
- PyTorch + Torchvision
- Standard library for logging (CSV) and serving (http.server)
- Optional: matplotlib for plotting in report.py

## Optimizer: Adagrad
- Accumulates per-parameter sum of squared gradients and scales updates by the inverse square root of this accumulator.
- Parameters:
  - lr (typical 1e-2 on vision tasks; can require tuning)
  - lr_decay (default 0.0): learning rate decays over time based on steps
  - weight_decay (L2)
  - initial_accumulator_value (default 0.0)
  - eps (default 1e-10) for numerical stability
- Pros: adaptive per-parameter steps; Cons: accumulator grows monotonically which can shrink lr too much over long training.

## Datasets
- MNIST: quick smoke tests
- CIFAR-10: crop+flip augmentation

## Models
- Small CNN for MNIST
- ResNet18 for CIFAR-10

## Reproducibility
- Global seeding for Python/NumPy/PyTorch; cuDNN deterministic flags where feasible.

## Outputs
- CSV at `runs/adagrad/<dataset>/seed_<N>/metrics.csv` with per-epoch metrics.

## Next
- Compare against SGD/Adam/AdamW/RMSProp with shared reporting and ranking.
