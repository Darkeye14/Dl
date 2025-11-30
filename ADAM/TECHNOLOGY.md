# Technology and Concepts (Adam)

## Stack
- Python 3.9–3.12
- PyTorch + Torchvision
- Standard library for logging (CSV) and serving (http.server)
- Optional: matplotlib for plotting in report.py

## Optimizer: Adam
- Update rule combines momentum and adaptive learning rates per-parameter.
- Parameters:
  - lr (default 1e-3)
  - betas=(beta1, beta2) defaults (0.9, 0.999)
  - eps (default 1e-8)
  - weight_decay (L2) coupled to gradients (unlike decoupled AdamW)
- Pros: robust defaults, fast convergence; Cons: may generalize differently vs SGD.

## Datasets
- MNIST: 28x28 grayscale digits (10 classes) — fast smoke tests.
- CIFAR-10: 32x32 color images (10 classes) — standard crop/flip augmentation.

## Models
- Small CNN for MNIST.
- ResNet18 for CIFAR-10 (`num_classes=10`).

## Reproducibility
- Global seeding for Python/NumPy/PyTorch; cuDNN deterministic where feasible.

## Outputs
- CSV at `runs/adam/<dataset>/seed_<N>/metrics.csv` with per-epoch metrics.

## Next
- Extend to more optimizers and add small LR/WD sweeps for fair ranking.
