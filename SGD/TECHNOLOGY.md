# Technology and Concepts

## Stack
- Python 3.9–3.12
- PyTorch + Torchvision
- Standard library only for logging (CSV) and simple serving (http.server)
- Optional helper: tqdm for progress bars

## Why these choices
- PyTorch is a widely used DL framework with reliable datasets/models/utilities.
- Torchvision provides MNIST/CIFAR-10 and common models (e.g., ResNet18).
- CSV logging keeps results portable and tool-agnostic.
- Minimal dependencies to simplify install on Windows.

## Optimizer: SGD with Momentum
- Update rule (concept):
  - v_t = m * v_{t-1} + g_t
  - w_{t+1} = w_t - lr * v_t
  - where m is momentum (e.g., 0.9), g_t is gradient, lr is learning rate.
- Benefits: smooths noisy gradients, accelerates along consistent descent directions.
- Weight decay (L2) is applied as `weight_decay` within optimizer.
- Nesterov momentum optionally available; computes gradient at the lookahead point.

## Datasets
- MNIST: 28x28 grayscale digits (10 classes). Used for quick smoke tests.
- CIFAR-10: 32x32 color images (10 classes). Uses standard augmentation: random crop + flip.

## Models
- MNIST: small CNN for speed.
- CIFAR-10: ResNet18 from torchvision with `num_classes=10`.

## Reproducibility
- Global seeds set for Python, NumPy, and PyTorch.
- Deterministic flags for cuDNN where feasible (`cudnn.deterministic=True`, `benchmark=False`).

## Outputs
- Each run writes a CSV at `runs/sgd_momentum/<dataset>/seed_<N>/metrics.csv` with per-epoch metrics.
- Fields: epoch, train_loss, train_acc, val_loss, val_acc, epoch_time_sec, total_time_sec.

## Next directions
- Add more optimizers (Adam, AdamW, RMSprop, Adagrad) via a simple optimizer factory.
- Add seed sweeps and summary tables/plots.
- Optional: cosine LR schedule and weight decay sweeps for fair comparisons.
