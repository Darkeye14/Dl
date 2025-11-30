import argparse
import csv
import os
from typing import List, Dict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Summarize a training metrics CSV (AdamW)")
    p.add_argument("--csv", required=True, help="Path to metrics.csv")
    p.add_argument("--plot", action="store_true", help="If set, attempts to save a PNG plot next to CSV")
    return p.parse_args()


def read_metrics(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "epoch": int(r["epoch"]),
                "train_loss": float(r["train_loss"]),
                "train_acc": float(r["train_acc"]),
                "val_loss": float(r["val_loss"]),
                "val_acc": float(r["val_acc"]),
                "epoch_time_sec": float(r["epoch_time_sec"]),
                "total_time_sec": float(r["total_time_sec"]),
            })
    return rows


def summarize(rows: List[Dict[str, float]]):
    if not rows:
        print("No data found in CSV.")
        return

    epochs = np.array([r["epoch"] for r in rows], dtype=float)
    val_acc = np.array([r["val_acc"] for r in rows], dtype=float)
    train_acc = np.array([r["train_acc"] for r in rows], dtype=float)
    epoch_time = np.array([r["epoch_time_sec"] for r in rows], dtype=float)
    total_time_last = float(rows[-1]["total_time_sec"])

    best_idx = int(np.argmax(val_acc))
    best_epoch = int(epochs[best_idx])
    best_val = float(val_acc[best_idx])
    final_val = float(val_acc[-1])

    # Time to reach 95% of best
    threshold = 0.95 * best_val
    tt95_epoch = None
    for i, a in enumerate(val_acc):
        if a >= threshold:
            tt95_epoch = int(epochs[i])
            break

    avg_epoch_time = float(np.mean(epoch_time))
    auc_val_vs_epoch = float(np.trapezoid(val_acc, epochs) / (epochs[-1] - epochs[0] + 1e-8))

    print("Summary")
    print("- Best val accuracy: {:.2f}% (epoch {})".format(best_val, best_epoch))
    print("- Final val accuracy: {:.2f}%".format(final_val))
    print("- Avg epoch time: {:.2f}s".format(avg_epoch_time))
    print("- Total training time: {:.2f}s".format(total_time_last))
    if tt95_epoch is not None:
        print("- Time-to-95%-of-best: epoch {} (~{:.2f}s)".format(tt95_epoch, avg_epoch_time * tt95_epoch))
    else:
        print("- Time-to-95%-of-best: not reached")
    print("- AUC(val_acc vs epoch): {:.2f}".format(auc_val_vs_epoch))

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(epochs, val_acc, label="val_acc (%)", color="tab:blue")
            ax1.plot(epochs, train_acc, label="train_acc (%)", color="tab:green", alpha=0.6)
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("accuracy (%)")
            ax1.grid(True, linestyle=":", alpha=0.5)
            ax1.legend(loc="lower right")

            out_png = os.path.join(os.path.dirname(args.csv), "metrics.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            print(f"- Saved plot: {out_png}")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    args = parse_args()
    rows = read_metrics(args.csv)
    summarize(rows)
