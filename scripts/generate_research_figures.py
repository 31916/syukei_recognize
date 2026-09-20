"""Redraw historical reported results; this script does not train or evaluate models."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "docs" / "tables"
DEFAULT_OUTPUT = ROOT / "docs" / "figures"
COLORS = {"side_distinguished": "#315a83", "side_ignored": "#008477"}


def read_table(filename):
    with (TABLES / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def format_axis(ax):
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Accuracy (%)", labelpad=10)
    ax.xaxis.grid(True, color="#dfe4e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig, output, name):
    # No timestamps or random SVG element IDs: stable figures on repeated builds.
    svg_path = output / f"{name}.svg"
    fig.savefig(svg_path, metadata={"Date": None}, facecolor="white")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8", newline="\n",
    )
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelcolor": "#243444", "text.color": "#243444",
        "xtick.color": "#536272", "ytick.color": "#243444",
        "svg.fonttype": "path", "svg.hashsalt": "syukei-reported-results",
    })
    rows = read_table("reported-overall-accuracy.csv")
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    values = [float(row["accuracy_percent"]) for row in rows]
    ax.barh([1, 0], values, color=[COLORS[row["condition"]] for row in rows], height=0.48)
    ax.set_yticks([1, 0], ["Left / right\ndistinguished", "Left / right\nignored"])
    for y, value in zip([1, 0], values):
        ax.text(value + 1.5, y, f"{value:.2f}%", va="center", fontsize=14, fontweight="bold")
    format_axis(ax)
    fig.suptitle("Reported handshape recognition accuracy", x=0.04, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.04, 0.055, "Source: graduation thesis, Results and Conclusion; final presentation, slides 9 and 11.", fontsize=8.5)
    fig.text(0.04, 0.015, "Historical reported values. Different label criteria; not a rerun of the current code.", fontsize=8.5)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.81, bottom=0.26)
    save_figure(fig, output, "reported-overall-accuracy")

    rows = read_table("reported-label-accuracy.csv")
    for condition, table, subtitle in (
        ("side_distinguished", 2, "Left / right distinguished"),
        ("side_ignored", 5, "Left / right ignored"),
    ):
        selected = [row for row in rows if row["condition"] == condition]
        selected.sort(key=lambda row: -float(row["accuracy_percent"]))
        values = [float(row["accuracy_percent"]) for row in selected]
        fig, ax = plt.subplots(figsize=(8.8, 6.2))
        ax.barh(range(len(selected)), values, color=COLORS[condition], height=0.62)
        ax.set_yticks(range(len(selected)), [row["label_id"] for row in selected])
        ax.invert_yaxis()
        ax.set_ylabel("Label ID", labelpad=12)
        for y, value in enumerate(values):
            inside = value > 87
            ax.text(value - 1.5 if inside else value + 1.5, y, f"{value:.2f}%",
                    ha="right" if inside else "left", va="center",
                    color="white" if inside else "#243444", fontsize=10)
        format_axis(ax)
        fig.suptitle(subtitle, x=0.04, ha="left", fontsize=16, fontweight="bold")
        fig.text(0.04, 0.90, "Selected labels with high and low reported accuracy", fontsize=11)
        fig.text(0.04, 0.04, f"Source: graduation thesis, Table {table}. These 10 labels are a reported subset, not all classes.", fontsize=8.5)
        fig.text(0.04, 0.012, "No per-sample predictions or uncertainty estimates are available in the supplied documents.", fontsize=8.5)
        fig.subplots_adjust(left=0.14, right=0.96, top=0.85, bottom=0.17)
        save_figure(fig, output, f"reported-label-accuracy-{condition.replace('_', '-')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    generate(parser.parse_args().output_dir)
