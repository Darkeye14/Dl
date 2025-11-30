# DL Optimizer Benchmark (MNIST + CIFAR-10)

This root README summarizes the whole benchmarking project across multiple optimizers. Each optimizer lives in its own subfolder with an isolated training+reporting setup, but they all follow the same design so results are comparable.

Subprojects:
- `SGD/` — SGD with momentum
- `ADAM/` — Adam
- `RMS/` — RMSProp
- `ADAMW/` — AdamW (decoupled weight decay)
- `ADA/` — Adagrad
- `dl-optim-bench/` — original unified harness prototype (not required for the per-optimizer folders).

---

## Datasets

All optimizers are evaluated on the same standard vision datasets:

- **MNIST**
  - 60k train / 10k test, 28×28 grayscale digits (10 classes).
  - Transforms: `ToTensor` + `Normalize(mean=0.1307, std=0.3081)`.
  - Purpose: fast, low-cost smoke test and basic comparison.

- **CIFAR-10** (supported in all projects, even if you only ran MNIST so far)
  - 50k train / 10k test, 32×32 RGB images (10 classes).
  - Train transforms: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` + normalization with CIFAR stats.
  - Test transforms: `ToTensor` + same normalization.

Both datasets are downloaded automatically via `torchvision.datasets` into each project’s `data/` directory.

---

## Common Design Across Optimizer Folders

Each optimizer subfolder (`SGD`, `ADAM`, `RMS`, `ADAMW`, `ADA`) has:

- **`train.py`**
  - CLI arguments (names vary slightly by optimizer):
    - `--dataset {mnist,cifar10}`
    - `--epochs`, `--batch-size`
    - `--device {auto,cpu,cuda}`
    - Optimizer-specific hyperparameters (e.g., `--momentum`, `--beta1`, `--alpha`, etc.).
  - Device selection: tries CUDA if available, otherwise CPU. If CUDA is requested but unavailable, it clearly prints a message and falls back to CPU.
  - Reproducibility:
    - Seeds Python, NumPy, and PyTorch (CPU + CUDA).
    - Enables deterministic cuDNN where feasible.
  - Data loading: uses a local `bench.data.get_dataloaders` that applies consistent transforms and deterministic `DataLoader` workers.
  - Models:
    - MNIST: a small custom CNN (`MNISTSmallCNN`).
    - CIFAR-10: `torchvision.models.resnet18(num_classes=10)`.
  - Loss: `nn.CrossEntropyLoss`.
  - Training loop:
    - Standard: forward → loss → backward → `optimizer.step()`.
    - Computes train and validation accuracy each epoch.
    - Measures `epoch_time_sec` and `total_time_sec`.
  - Logging:
    - CSV per run at `runs/<optimizer_name>/<dataset>/seed_<N>/metrics.csv`.
    - Columns: `epoch, train_loss, train_acc, val_loss, val_acc, epoch_time_sec, total_time_sec`.

- **`report.py`**
  - Usage: `python report.py --csv path/to/metrics.csv --plot`.
  - Reads the metrics CSV and prints:
    - Best validation accuracy and which epoch it occurs at.
    - Final validation accuracy.
    - Average epoch time and total training time.
    - Time-to-95%-of-best accuracy.
    - AUC of `val_acc` vs epoch (a single "quality of learning curve" number).
  - With `--plot` and `matplotlib` installed, saves `metrics.png` showing train+val accuracy curves.

This shared structure makes it easier to compare optimizers and present the project.

---

## Optimizers: Concepts and Settings

All optimizers run on the same datasets, models, and seeds; only the update rule and associated hyperparameters differ.

### 1. SGD with Momentum (`SGD/`)

- **Concept**
  - Basic gradient descent with a velocity term that accumulates gradients.
  - Update (conceptually):
    - `v_t = m * v_{t-1} + g_t`
    - `w_{t+1} = w_t - lr * v_t`
  - Momentum smooths noisy gradients and accelerates along consistent descent directions.
- **Key hyperparameters**
  - `lr` (learning rate), e.g. `0.1` for MNIST.
  - `momentum` (usually `0.9`).
  - `weight_decay` (L2 regularization).
  - Optional: `--nesterov` for Nesterov momentum.
- **Project behavior**
  - File: `SGD/train.py` with `torch.optim.SGD` (momentum + optional Nesterov).
  - Logs to `SGD/runs/sgd_momentum/...`.

### 2. Adam (`ADAM/`)

- **Concept**
  - Adaptive method combining momentum (first moment) and RMS-type scaling (second moment).
  - Maintains:
    - `m_t`: moving average of gradients.
    - `v_t`: moving average of squared gradients.
  - Uses bias-corrected `m̂_t`, `v̂_t` to scale the update per parameter.
- **Key hyperparameters**
  - `lr` (default `1e-3`).
  - `betas=(beta1, beta2)` (defaults `(0.9, 0.999)`).
  - `eps` (default `1e-8`).
  - `weight_decay` (L2, **coupled** with the gradient; not decoupled).
- **Project behavior**
  - File: `ADAM/train.py` with `torch.optim.Adam`.
  - Logs to `ADAM/runs/adam/...`.

### 3. RMSProp (`RMS/`)

- **Concept**
  - Maintains an exponential moving average of squared gradients and divides the gradient by its root, stabilizing learning.
  - Good for non-stationary problems and was historically used in many RL setups.
- **Key hyperparameters**
  - `lr` (default `1e-3`).
  - `alpha` (EMA decay, e.g. `0.99`).
  - `eps` (~`1e-8`).
  - `momentum` (optional extra acceleration).
  - `centered` (if True, uses variance rather than raw second moment).
  - `weight_decay`.
- **Project behavior**
  - File: `RMS/train.py` with `torch.optim.RMSprop`.
  - Logs to `RMS/runs/rmsprop/...`.

### 4. AdamW (`ADAMW/`)

- **Concept**
  - Adam variant with **decoupled weight decay**.
  - Regularization is applied directly to parameters, not mixed into the gradient.
  - Often more stable and better-behaved than standard Adam with L2 for deep nets.
- **Key hyperparameters**
  - `lr` (default `1e-3`).
  - `betas=(0.9, 0.999)`, `eps=1e-8`.
  - `weight_decay` (decoupled), typically in `[1e-4, 1e-2]`.
- **Project behavior**
  - File: `ADAMW/train.py` with `torch.optim.AdamW`.
  - Logs to `ADAMW/runs/adamw/...`.

### 5. Adagrad (`ADA/`)

- **Concept**
  - Accumulates the sum of squared gradients per parameter and scales updates by the inverse square root of this accumulator.
  - Gives larger effective learning rates to infrequent features and smaller to frequent ones.
- **Key hyperparameters**
  - `lr` (often higher than Adam, e.g. `1e-2`).
  - `lr_decay` (optional learning-rate decay over time).
  - `initial_accumulator_value` (starting value for the squared-grad accumulator).
  - `eps` (~`1e-10`).
  - `weight_decay` (L2).
- **Project behavior**
  - File: `ADA/train.py` with `torch.optim.Adagrad`.
  - Logs to `ADA/runs/adagrad/...`.

---

## Example MNIST Results (Seed 42, 3 Epochs, CPU)

Below is a simple comparison table based on your actual MNIST runs with seed `42` (each for 3 epochs). All are CPU runs, so epoch times reflect CPU, not GPU.

**Per-optimizer MNIST metrics (seed_42)**

| Optimizer | Best Val Acc (%) | Final Val Acc (%) | Epochs | Avg Epoch Time (s) | Total Time (s) |
|-----------|------------------|-------------------|--------|--------------------|----------------|
| SGD (momentum) | 98.28 (epoch 2) | 98.19 | 3 | ~146.3 | ~438.9 |
| Adam          | 98.65 (epoch 2) | 98.49 | 3 | ~60.6  | ~181.9 |
| RMSProp       | 98.73 (epoch 3) | 98.73 | 3 | ~50.6  | ~151.8 |
| AdamW         | 98.65 (epoch 2) | 98.49 | 3 | ~147.5 | ~442.5 |
| Adagrad       | 98.66 (epoch 3) | 98.66 | 3 | ~47.5  | ~141.4 |

Notes:
- All optimizers reach **~98–99% validation accuracy on MNIST** after only 3 epochs.
- **Speed (CPU)**:
  - Adam, RMSProp, and Adagrad are much faster per epoch than your SGD/AdamW runs (which currently have higher epoch times, likely due to CPU and DataLoader settings).
- **Stability**:
  - Final and best accuracies are almost identical for RMSProp and Adagrad, indicating stable convergence by epoch 3.

If you re-run these on GPU with a CUDA-enabled PyTorch build, epoch times will drop significantly.

---

## How to Run and Compare Optimizers

For each optimizer folder (example: `ADAM`):

1. **Create venv and install deps**
   ```powershell
   cd ADAM
   py -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Check CUDA availability** (optional)
   ```powershell
   python -c "import torch;print('cuda_available:',torch.cuda.is_available()); \
       print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
   ```

3. **Train a MNIST model**
   ```powershell
   # Example Adam
   python train.py --dataset mnist --epochs 3 --batch-size 128 --lr 1e-3 --weight-decay 0.0 --device cuda
   ```

4. **Generate a human-readable report and plot**
   ```powershell
   python report.py --csv runs/adam/mnist/seed_42/metrics.csv --plot
   ```

5. **Repeat** with other optimizers (`SGD`, `RMS`, `ADAMW`, `ADA`) using the analogous commands in their READMEs.

---

## Overall Understanding of the Project

- You built a **mini benchmarking suite for deep learning optimizers** using PyTorch.
- Each optimizer has its **own self-contained project** (code, README, metrics, reports) so you can:
  - Show each optimizer individually to your teacher.
  - Demonstrate that the core pipeline (datasets, models, logging) is consistent.
- The design emphasizes:
  - **Reproducibility**: fixed seeds, deterministic settings, logged metrics.
  - **Transparency**: raw CSV logs plus small `report.py` scripts that summarize results.
  - **Extensibility**: easy to add more optimizers or modify hyperparameters.

When explaining this to your teacher, you can highlight:
- The **common experimental setup** (same datasets, architecture, seeds).
- The **differences between optimizers** (SGD vs Adam vs RMSProp vs AdamW vs Adagrad).
- The **metrics you use** to judge them (accuracy, speed, time-to-target, AUC).
- How the code structure makes it easy to **reproduce and extend** the experiments.
