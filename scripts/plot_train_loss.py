"""Plot the training loss against step from a train_wm.py Slurm log.

The number this reads is the progress bar's `loss=` postfix, printed to one decimal, so
the series is a staircase at 0.1 resolution rather than the exact per-step loss. It is
enough to read the trend; it is not enough to compare two runs that sit within 0.1 of
each other. The offline W&B run alongside it holds the precise scalars, but recent
wandb versions do not expose the history records through the datastore scanner.

    python scripts/plot_train_loss.py --log <slurm.out> --out outputs/<tag>/figures/loss.png
"""
import re
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP_LOSS = re.compile(r"(\d+)/(\d+) .*?loss=([0-9.]+)")

# One series, so no legend: the title names it. Ink stays in greys; the line alone
# carries identity.
LINE = "#3b6fb6"
INK = "#1f2328"
MUTED = "#6a737d"


def read(path):
    # tqdm separates its redraws with carriage returns, but copying the log off the
    # cluster can normalise those to newlines, so scan the whole text rather than
    # assuming either separator survived.
    text = open(path, errors="ignore").read()
    seen = {}
    for m in STEP_LOSS.finditer(text):
        seen[int(m.group(1))] = float(m.group(3))   # last value wins for a repeated step
    steps = sorted(seen)
    return np.array(steps), np.array([seen[s] for s in steps])


def main():
    p = ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--total", type=int, default=30000)
    p.add_argument("--window", type=int, default=201, help="rolling mean window, in steps")
    args = p.parse_args()

    steps, loss = read(args.log)
    if len(steps) == 0:
        raise SystemExit(f"no 'loss=' entries found in {args.log}")

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    ax.plot(steps, loss, lw=1, color=LINE, alpha=0.28, solid_capstyle="round")
    if len(steps) >= args.window:
        k = np.ones(args.window) / args.window
        smooth = np.convolve(loss, k, mode="valid")
        off = args.window // 2
        ax.plot(steps[off:off + len(smooth)], smooth, lw=2, color=LINE,
                solid_capstyle="round")

    ax.set_xlim(0, max(args.total, steps[-1]))
    ax.set_xlabel("step", color=MUTED, fontsize=9)
    ax.set_ylabel("training loss", color=MUTED, fontsize=9)
    ax.set_title(f"abc_mcap training loss  -  step {steps[-1]:,} of {args.total:,}",
                 color=INK, fontsize=11, loc="left", pad=12)
    # Recessive axes: no box, a faint horizontal grid only.
    ax.grid(axis="y", color="#d8dee4", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8dee4")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    # Direct label on the last point rather than a number on every point.
    ax.annotate(f"{loss[-1]:.1f}", (steps[-1], loss[-1]), color=INK, fontsize=9,
                xytext=(6, 0), textcoords="offset points", va="center")
    fig.text(0.125, -0.02, "progress-bar loss, 0.1 resolution; heavy line is a "
             f"{args.window}-step rolling mean", color=MUTED, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"{args.out}: {len(steps)} points, step {steps[0]}-{steps[-1]}, "
          f"loss {loss[0]:.1f} -> {loss[-1]:.1f}")


if __name__ == "__main__":
    main()
