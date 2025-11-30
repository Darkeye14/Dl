import argparse
import csv
import os
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Make local 'bench' package importable when run from any CWD
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench.data import get_dataloaders
from bench.models import build_model
from bench.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train with Adagrad")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"], help="Dataset to use")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    p.add_argument("--lr-decay", type=float, default=0.0, help="Adagrad lr_decay")
    p.add_argument("--initial-accumulator-value", type=float, default=0.0, help="Adagrad initial accumulator value")
    p.add_argument("--eps", type=float, default=1e-10, help="Adagrad epsilon")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    return p.parse_args()


def select_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA requested but not available; falling back to CPU")
            return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    device = select_device(args.device)

    print("Args:", vars(args))
    print("Device:", device)
    if device.type == "cuda":
        try:
            print("CUDA device:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    set_seed(args.seed, deterministic=True)

    train_loader, val_loader = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(args.dataset)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adagrad(
        model.parameters(),
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        initial_accumulator_value=args.initial_accumulator_value,
        eps=args.eps,
    )

    run_dir = os.path.join("runs", "adagrad", args.dataset, f"seed_{args.seed}")
    ensure_dir(run_dir)
    csv_path = os.path.join(run_dir, "metrics.csv")

    total_start = time.perf_counter()

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "epoch_time_sec",
            "total_time_sec",
        ])

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()

            # Train
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * images.size(0)
                with torch.no_grad():
                    train_correct += (outputs.argmax(dim=1) == targets).sum().item()
                    train_total += targets.size(0)

            train_loss = train_loss_sum / max(1, train_total)
            train_acc = 100.0 * train_correct / max(1, train_total)

            # Eval
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss_sum += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(dim=1) == targets).sum().item()
                    val_total += targets.size(0)

            val_loss = val_loss_sum / max(1, val_total)
            val_acc = 100.0 * val_correct / max(1, val_total)

            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - epoch_start
            total_time = time.perf_counter() - total_start

            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.2f}",
                f"{val_loss:.6f}",
                f"{val_acc:.2f}",
                f"{epoch_time:.3f}",
                f"{total_time:.3f}",
            ])
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
                f"epoch_time={epoch_time:.2f}s"
            )


if __name__ == "__main__":
    main()
