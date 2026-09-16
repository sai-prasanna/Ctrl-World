"""Plot evaluation metrics across Ctrl-World checkpoints.

One figure serves every run, so it lives beside the per-run directories rather than
inside one of them.

Usage:
    python3 experiments/plot_checkpoint_metrics.py \
        --inputs experiments/0003_abc_mcap/eval/metrics_step*.json \
        --out experiments/0003_abc_mcap/checkpoint_metrics.svg
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ("psnr", "ssim", "lpips", "fid", "fvd")
GROUPS = {"third_view": "Third view", "wrist_view": "Wrist views"}
HIGHER_IS_BETTER = {"psnr", "ssim"}


def checkpoint_step(record, path):
    """Return the checkpoint step from the record or its filename."""
    match = re.search(r"checkpoint-(\d+)\.pt", record.get("checkpoint", ""))
    if match:
        return int(match.group(1))
    match = re.search(r"step(\d+)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not determine checkpoint step from {path}")


def load_records(paths):
    records = []
    for path in paths:
        with path.open() as file:
            record = json.load(file)
        records.append((checkpoint_step(record, path), record))
    if not records:
        raise ValueError("At least one metric JSON file is required")
    return sorted(records)


def value_and_interval(metric_data):
    """Return a metric value and optional confidence interval."""
    return metric_data.get("mean", metric_data.get("value")), metric_data.get("ci95")


def step_label(step):
    """Return a compact axis label, so adjacent checkpoints stay legible."""
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def plot(records, output):
    steps = [step for step, _ in records]
    colors = {"third_view": "#1769aa", "wrist_view": "#d55e00"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()

    for axis, metric in zip(axes, METRICS):
        for group, label in GROUPS.items():
            values, intervals = [], []
            for _, record in records:
                value, interval = value_and_interval(record["per_group"][group][metric])
                values.append(value)
                intervals.append(interval)
            axis.plot(steps, values, marker="o", linewidth=2, label=label,
                      color=colors[group])
            if all(interval is not None for interval in intervals):
                axis.fill_between(steps, [i[0] for i in intervals],
                                  [i[1] for i in intervals], color=colors[group],
                                  alpha=0.14, linewidth=0)
        direction = "higher is better" if metric in HIGHER_IS_BETTER else "lower is better"
        axis.set_title(f"{metric.upper()} — {direction}")
        axis.set_xticks(steps)
        # Checkpoints 10000 apart collide at this figure width, so shorten and tilt.
        axis.set_xticklabels([step_label(step) for step in steps], rotation=45,
                             ha="right", fontsize=9)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Checkpoint step")

    axes[0].set_ylabel("Metric value")
    axes[3].set_ylabel("Metric value")

    # The sixth cell is free, so the key goes there instead of over the PSNR curves.
    legend_axis = axes[-1]
    legend_axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, frameon=False, loc="center", fontsize=12,
                       title="View group", alignment="left")

    fig.suptitle("Ctrl-World checkpoint evaluation metrics", fontsize=15)
    fig.text(0.5, 0.01,
             "Shaded regions show 95% bootstrap confidence intervals when reported.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format=output.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="Metric JSON files, one per checkpoint.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output image path, such as .svg or .png.")
    args = parser.parse_args()
    plot(load_records(args.inputs), args.out)


if __name__ == "__main__":
    main()
