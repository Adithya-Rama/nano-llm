"""
plot_training.py — Plot learning curves from JSONL training logs
=================================================================
Reads the train_log.jsonl file(s) produced by train.py and generates
clean learning curve plots suitable for a 2-page report.

Usage:
    # Single run:
    python plot_training.py --log out-rocstories/train_log.jsonl

    # Compare multiple runs (ablation overlay):
    python plot_training.py \
        --log out-rocstories-baseline/train_log.jsonl \
        --log out-rocstories-rope/train_log.jsonl \
        --log out-rocstories-ffn/train_log.jsonl \
        --log out-rocstories-qknorm/train_log.jsonl \
        --log out-rocstories/train_log.jsonl \
        --labels "Baseline,+RoPE,+RMSNorm+SwiGLU,+QK-Norm,All Modern" \
        --output ablation_curves.png

    # Single run with val loss overlay:
    python plot_training.py --log out-rocstories/train_log.jsonl --output train_curves.png

Outputs:
    PNG file with training / validation loss curves.
"""

import argparse
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (works on Colab/headless)
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)


def load_log(path):
    """Load a JSONL training log file."""
    steps, train_losses, val_steps, val_losses, lrs = [], [], [], [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            step = entry.get('step', entry.get('iter', 0))
            steps.append(step)
            train_losses.append(entry.get('train_loss', entry.get('loss', None)))
            lrs.append(entry.get('lr', None))
            vl = entry.get('val_loss', None)
            if vl is not None:
                val_steps.append(step)
                val_losses.append(vl)
    return {
        'steps': steps,
        'train_loss': train_losses,
        'val_steps': val_steps,
        'val_loss': val_losses,
        'lr': lrs,
    }


def plot_single(log_data, output_path, title="Training Curves"):
    """Plot train + val loss for a single run."""
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # Train loss
    ax1.plot(log_data['steps'], log_data['train_loss'],
             alpha=0.4, color='#2196F3', linewidth=0.5, label='Train loss (raw)')

    # Smoothed train loss (exponential moving average)
    smoothed = _ema(log_data['train_loss'], alpha=0.95)
    ax1.plot(log_data['steps'], smoothed,
             color='#1565C0', linewidth=2, label='Train loss (smoothed)')

    # Val loss
    if log_data['val_steps']:
        ax1.plot(log_data['val_steps'], log_data['val_loss'],
                 'o-', color='#E53935', markersize=3, linewidth=1.5, label='Val loss')

    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved: {output_path}")


def plot_comparison(log_paths, labels, output_path, title="Ablation Comparison"):
    """Plot val loss curves from multiple runs on the same axes."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1']

    for i, (path, label) in enumerate(zip(log_paths, labels)):
        data = load_log(path)
        color = colors[i % len(colors)]

        # Plot smoothed train loss
        smoothed = _ema(data['train_loss'], alpha=0.95)
        ax.plot(data['steps'], smoothed, color=color, linewidth=1.5,
                alpha=0.5, linestyle='--')

        # Plot val loss (thicker, with markers)
        if data['val_steps']:
            ax.plot(data['val_steps'], data['val_loss'],
                    'o-', color=color, markersize=3, linewidth=2, label=label)
        else:
            ax.plot(data['steps'], smoothed, color=color, linewidth=2, label=label)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved: {output_path}")


def _ema(values, alpha=0.95):
    """Exponential moving average for smoothing."""
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        if v is None:
            smoothed.append(last)
            continue
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return smoothed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot nanoGPT training curves')
    parser.add_argument('--log', type=str, action='append', required=True,
                        help='Path to train_log.jsonl (can specify multiple for comparison)')
    parser.add_argument('--labels', type=str, default=None,
                        help='Comma-separated labels for each log (for comparison plots)')
    parser.add_argument('--output', type=str, default='training_curves.png',
                        help='Output image path (default: training_curves.png)')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title')
    args = parser.parse_args()

    if len(args.log) == 1:
        # Single run plot
        data = load_log(args.log[0])
        title = args.title or f"Training Curves ({os.path.dirname(args.log[0])})"
        plot_single(data, args.output, title=title)
    else:
        # Multi-run comparison plot
        if args.labels:
            labels = [l.strip() for l in args.labels.split(',')]
        else:
            labels = [os.path.basename(os.path.dirname(p)) for p in args.log]
        title = args.title or "Ablation Comparison — Validation Loss"
        plot_comparison(args.log, labels, args.output, title=title)
