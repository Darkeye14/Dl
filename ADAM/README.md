# ADAM Optimizer Benchmark

A minimal, reproducible project to train and evaluate Adam on MNIST/CIFAR-10. Includes a runnable training harness, CSV logs, and a simple reporting script.

## Quickstart (Windows)

- Prereqs: Python 3.9–3.12
- Open a terminal in the project folder `ADAM`.

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
If PyTorch install fails, use the selector: https://pytorch.org/get-started/locally/

### 3) Run a quick MNIST smoke test (GPU)
```
python train.py --dataset mnist --epochs 3 --batch-size 128 --lr 1e-3 --weight-decay 0.0 --device cuda
```

### 4) Run CIFAR-10 (longer, GPU)
```
python train.py --dataset cifar10 --epochs 20 --batch-size 128 --lr 1e-3 --weight-decay 1e-4 --device cuda
```



python report.py --csv runs/adam/mnist/seed_42/metrics.csv --plot






### Device control
- Force CUDA (after installing CUDA-enabled PyTorch):
```
python train.py --device cuda
```
- Force CPU:
```
python train.py --device cpu
```
- Verify CUDA availability:
```
python -c "import torch;print('cuda_available:',torch.cuda.is_available()); \
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```
- If CUDA is False, install a CUDA build (adjust cu version):
```
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

### Outputs
- CSV logs per run at:
```
runs/adam/<dataset>/seed_<N>/metrics.csv
```
Columns: epoch, train_loss, train_acc, val_loss, val_acc, epoch_time_sec, total_time_sec.

## Reporting
Summarize results and optionally save a plot:
```
python report.py --csv runs/adam/<dataset>/seed_<N>/metrics.csv --plot
```
Outputs:
- Best and final validation accuracy
- Average epoch time and total time
- Time-to-95%-of-best accuracy
- AUC(val_acc vs epoch)

## Notes
- Defaults: Adam with betas=(0.9, 0.999), eps=1e-8.
- For fair comparisons later, keep augmentations and seeds consistent.
