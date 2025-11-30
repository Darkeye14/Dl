# Technology and Concepts (RMSProp)

## Stack
- Python 3.9–3.12
- PyTorch + Torchvision
- Standard library for logging (CSV) and serving (http.server)
- Optional: matplotlib for plotting in report.py

## Optimizer: RMSProp
- Maintains an exponential moving average (EMA) of squared gradients, scaling updates by the root of this average.
- Parameters:
  - lr (default 1e-3)
  - alpha (EMA decay) default 0.99
  - eps default 1e-8
  - momentum default 0.0 (optional acceleration)
  - centered: if True, use variance (E[g^2] - (E[g])^2) yielding a more adaptive denominator
  - weight_decay (L2)
- Pros: stable on non-stationary objectives; Cons: sensitive to lr/alpha and sometimes slower convergence than Adam.

## Datasets
- MNIST (fast smoke test)
- CIFAR-10 (with random crop + flip)

## Models
- Small CNN for MNIST
- ResNet18 for CIFAR-10 (`num_classes=10`)

## Reproducibility
- Global seeding across Python/NumPy/PyTorch; cuDNN deterministic where feasible.

## Outputs
- CSV at `runs/rmsprop/<dataset>/seed_<N>/metrics.csv` with per-epoch metrics.

## Next
- Compare against SGD/Adam/AdamW/Adagrad using shared harness and ranking.
