# DL Optimizer Benchmark (SGD with Momentum)

A minimal, reproducible starting point for a DL benchmarking project. This seed includes a runnable training harness that uses SGD with momentum on MNIST/CIFAR-10 and logs metrics to CSV. Simple, no external UI required.

## Quickstart (Windows)

- Prereqs: Python 3.9–3.12
- Open a terminal in the project folder `dl-optim-bench`.

### 1) Create and activate venv

```
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```
pip install -r requirements.txt
```

If PyTorch installation fails, use the official selector for your system: https://pytorch.org/get-started/locally/

### 3) Run a quick MNIST smoke test

```
python train.py --dataset mnist --epochs 3 --batch-size 128 --lr 0.1 --momentum 0.9 --weight-decay 5e-4 --device cuda
```
















### 4) Run CIFAR-10 (longer)

```
python train.py --dataset cifar10 --epochs 20 --batch-size 128 --lr 0.1 --momentum 0.9 --weight-decay 5e-4
```

### Device control

- Auto-selects CUDA if available. To force CUDA explicitly (after installing CUDA-enabled PyTorch):
```
python train.py --device cuda
```
- Force CPU:
```
python train.py --device cpu
```
- Verify PyTorch CUDA availability:
```
python -c "import torch;print('cuda_available:',torch.cuda.is_available());\
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```
- If CUDA shows False, install a CUDA build of PyTorch using the official selector: https://pytorch.org/get-started/locally/
  Example (adjust for your CUDA version):
```
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

### Outputs

- CSV logs per run at:
```
runs/sgd_momentum/<dataset>/seed_<N>/metrics.csv
```
Columns: epoch, train_loss, train_acc, val_loss, val_acc, epoch_time_sec, total_time_sec.

## Reporting

Turn the CSV into a human-readable summary (and optional plot):

```
python report.py --csv runs/sgd_momentum/<dataset>/seed_<N>/metrics.csv --plot
```







Outputs:
- Best and final validation accuracy
- Average epoch time and total time
- Time-to-95%-of-best accuracy
- AUC(val_acc vs epoch)

## Optional: NPM commands (not required)

This project is Python-only. If you want to serve the `runs/` folder in a browser with a simple static server:

```
# Initialize a Node project (optional)
npm init -y

# Install a lightweight static server
yarn add --dev serve  # or: npm install --save-dev serve

# Serve the runs directory on localhost:3000
npx serve runs -l 3000
```

Alternatively, with Python standard library only:
```
python -m http.server 8000
# then open http://localhost:8000/runs in your browser
```

## Next steps

- Add more optimizers (Adam, AdamW, RMSprop, Adagrad) into the same harness.
- Add seed sweeps and a simple reporting script.
- Keep runs short initially to validate logs, then scale epochs.
