"""Generate the class-distribution and confusion-matrix PNGs used in the README.

Regenerate after changing the dataset or the baseline model:
    python visualize.py --data ../reddit/Crypto_com.csv
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from baseline import build_baseline
from berttest import LABELS, load_dataset

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order: blue, orange, aqua
SEQUENTIAL = ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95"]  # light -> dark, one hue

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def plot_class_distribution(labels, out_path):
    counts = [int((labels == i).sum()) for i in range(len(LABELS))]

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    bars = ax.bar(LABELS, counts, color=CATEGORICAL, width=0.6)
    ax.set_title("Class distribution (full dataset, n={})".format(len(labels)), color=INK, fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(left=False)
    ax.set_yticks([])
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{count} ({count / len(labels):.0%})",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    row_pct = cm / cm.sum(axis=1, keepdims=True)

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)

    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
    im = ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
    ax.set_ylabel("Actual", color=MUTED, fontsize=9)
    ax.set_title("Confusion matrix (baseline, row-normalized)", color=INK, fontsize=10)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            text_color = "#ffffff" if row_pct[i, j] > 0.5 else INK
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({row_pct[i, j]:.0%})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="../reddit/Crypto_com.csv")
    parser.add_argument("--text-column", default="DATA_COLUMN")
    parser.add_argument("--label-column", default="LABEL_COLUMN")
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--out-dir", default="assets")
    return parser.parse_args()


def main():
    import os

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_df, test_df = load_dataset(args.data, args.text_column, args.label_column, args.test_size)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[args.label_column] = train_df[args.label_column].astype(int)
    test_df[args.label_column] = test_df[args.label_column].astype(int)

    full_labels = np.concatenate([train_df[args.label_column], test_df[args.label_column]])
    plot_class_distribution(full_labels, f"{args.out_dir}/class_distribution.png")

    model = build_baseline()
    model.fit(train_df[args.text_column], train_df[args.label_column])
    predictions = model.predict(test_df[args.text_column])
    plot_confusion_matrix(test_df[args.label_column], predictions, f"{args.out_dir}/confusion_matrix.png")

    print(f"Wrote {args.out_dir}/class_distribution.png and {args.out_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()
