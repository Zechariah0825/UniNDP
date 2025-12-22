import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PIM decode utilization versus decode step from CSV output."
    )
    parser.add_argument("--input-csv", required=True, help="CSV file produced by pim_attention_utilization.py")
    parser.add_argument("--model", default=None, help="Filter rows to a specific model name.")
    parser.add_argument("--op", default=None, help="Filter rows to a specific operator (e.g., qk, kv).")
    parser.add_argument(
        "--output",
        default="plots/pim_utilization.png",
        help="Path to save the rendered plot (PNG).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title. Defaults to '<model> <op>'.",
    )
    parser.add_argument(
        "--active-axis",
        action="store_true",
        help="Overlay active batch count on a secondary y-axis for context.",
    )
    return parser.parse_args()


def load_rows(csv_path: str, model: str = None, op: str = None) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if model and row.get("model") != model:
                continue
            if op and row.get("op") != op:
                continue
            try:
                decode_step = int(row["decode_step"])
                utilization = float(row["pim_utilization"])
                active_batches = int(row.get("active_batches", 0))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Malformed row in {csv_path}: {row}") from exc
            rows.append(
                {
                    "decode_step": decode_step,
                    "pim_utilization": utilization,
                    "active_batches": active_batches,
                }
            )
    if not rows:
        raise ValueError("No rows matched the provided filters. Check model/op arguments.")
    rows.sort(key=lambda item: item["decode_step"])
    return rows


def plot_utilization(rows: List[Dict[str, float]], title: str, output_path: str, show: bool, plot_active: bool) -> None:
    steps = [item["decode_step"] for item in rows]
    util_pct = [item["pim_utilization"] * 100 for item in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, util_pct, marker="o", linewidth=1.6, label="PIM Utilization (%)")
    ax1.set_xlabel("Decode Step")
    ax1.set_ylabel("Utilization (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if plot_active:
        ax2 = ax1.twinx()
        active_batches = [item["active_batches"] for item in rows]
        ax2.step(steps, active_batches, where="post", color="tab:orange", label="Active Batches")
        ax2.set_ylabel("Active Batches")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    if title:
        ax1.set_title(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv, model=args.model, op=args.op)
    inferred_title = args.title
    if inferred_title is None:
        target = []
        if args.model:
            target.append(args.model)
        if args.op:
            target.append(args.op)
        inferred_title = " ".join(target) if target else "PIM Decode Utilization"
    plot_utilization(rows, inferred_title, args.output, args.show, args.active_axis)


if __name__ == "__main__":
    main()
