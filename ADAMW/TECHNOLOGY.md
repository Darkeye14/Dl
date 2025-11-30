# Technology and Concepts (AdamW)

## Stack
- Python 3.9–3.12
- PyTorch + Torchvision
- Standard library for logging (CSV) and serving (http.server)
- Optional: matplotlib for plotting in report.py

## Optimizer: AdamW
- Adam with decoupled weight decay: regularization is applied to parameters directly, not via gradient L2.
- Parameters:
  - lr (default 1e-3)
  - betas=(beta1, beta2) defaults (0.9, 0.999)
  - eps default 1e-8
  - weight_decay (decoupled) commonly 1e-4–1e-2 depending on task
- Pros: often better behaved than Adam w/ L2 weight decay; strong baseline.

## Datasets
- MNIST: fast smoke test
- CIFAR-10: standard crop+flip

## Models
- Small CNN (MNIST)
- ResNet18 (CIFAR-10, num_classes=10)

## Reproducibility
- Global seeds; cuDNN deterministic flags where feasible.

## Outputs
- CSV at `runs/adamw/<dataset>/seed_<N>/metrics.csv` with per-epoch metrics.

## Next
- Compare against SGD/Adam/RMSProp/Adagrad with shared reporting and ranking.
