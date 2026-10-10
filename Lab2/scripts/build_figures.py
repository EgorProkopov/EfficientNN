"""Rebuild report figures from saved results without loading models."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
(ROOT / "cache/matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp").mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache")
os.environ["TMPDIR"] = str(ROOT / "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator
import numpy as np


FIGURES = ROOT / "figures"
SIZES = (1000, 2500, 5000)
DPI = 180
COLORS = {"softmax": "#2367a1", "epanechnikov": "#e77827"}
STYLE = {
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "savefig.facecolor": "white",
}


def require(path, strict):
    if path.exists():
        return True
    if strict:
        raise FileNotFoundError(path)
    print(f"SKIP missing {path.relative_to(ROOT)}")
    return False


def save(fig, stem):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    for extension in ("png", "svg"):
        path = FIGURES / f"{stem}.{extension}"
        fig.savefig(path, dpi=DPI)
        print(path.relative_to(ROOT))
    plt.close(fig)


def kernel(strict):
    path = ROOT / "results/kernel_plot_data.npz"
    if not require(path, strict):
        return
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"]
        rank = data["rank"]
        temperatures = data["temperatures"]
        weights = data["weights"]
        support = data["support_fraction"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, weight_curve, support_curve in zip(labels, weights, support):
        axes[0].plot(rank, weight_curve, label=label)
        axes[1].plot(temperatures, support_curve, marker="o", label=label)
    axes[0].set(xlabel="Key rank by similarity", ylabel="Attention weight",
                title="One query, 128 normalized keys, temperature 0.1")
    axes[1].set(xlabel="Temperature", ylabel="Fraction of keys with nonzero weight",
                ylim=(-0.03, 1.03), title="Support size for the same query")
    axes[0].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    save(fig, "kernel_equivalence")


def training_curves(run):
    path = ROOT / "runs/regbench" / run / "metrics/version_0/metrics.csv"
    by_epoch = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["epoch"]:
                continue
            values = by_epoch.setdefault(int(row["epoch"]), {})
            for key in ("test/dfa_accuracy", "val/loss"):
                if row[key]:
                    values[key] = float(row[key])
    return by_epoch


def curves(size, tag, strict):
    suffix = f"_{tag}" if tag else ""
    path = ROOT / f"results/regbench_{size}_seed0{suffix}.json"
    if not require(path, strict):
        return
    rows = json.loads(path.read_text())
    csv_paths = [ROOT / "runs/regbench" / row["run"] / "metrics/version_0/metrics.csv" for row in rows]
    if not all(require(csv_path, strict) for csv_path in csv_paths):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row in rows:
        run = row["run"]
        method = "softmax" if run.startswith("softmax") else "epanechnikov"
        label = "Gaussian / softmax" if method == "softmax" else "Epanechnikov / sparsemax"
        data = training_curves(run)
        for ax, key in zip(axes, ("test/dfa_accuracy", "val/loss")):
            points = [(epoch, values[key]) for epoch, values in sorted(data.items()) if key in values]
            ax.plot([point[0] for point in points], [point[1] for point in points],
                    label=label, color=COLORS[method])
    axes[0].set(ylabel="Accuracy", xlabel="Эпоха")
    axes[1].set(ylabel="Validation loss", xlabel="Эпоха")
    for ax in axes:
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.2)
    late_accuracy = [
        value
        for line in axes[0].lines
        for epoch, value in zip(line.get_xdata(), line.get_ydata())
        if epoch >= 50
    ]
    low, high = min(late_accuracy), max(late_accuracy)
    margin = 0.1 * (high - low)
    axes[0].set(xlim=(50, 200), ylim=(low - margin, high + margin))
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[0].yaxis.set_minor_locator(NullLocator())
    save(fig, f"regbench_{size}_seed0{suffix}")


def methods(tag, strict):
    suffix = f"_{tag}" if tag else ""
    path = ROOT / f"results/regbench_1000_eight_methods{suffix}.json"
    if not require(path, strict):
        return
    rows = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.7), sharey=True)
    y = np.arange(len(rows))
    for ax, paper_key, observed_key, title in (
        (axes[0], "paper_accuracy_percent", "observed_accuracy_percent", "Accuracy"),
        (axes[1], "paper_tvd_percent", "observed_tvd_percent", "TVD"),
    ):
        for index, row in enumerate(rows):
            ax.plot([row[paper_key], row[observed_key]], [index, index], color="0.7", lw=2)
        ax.scatter([row[paper_key] for row in rows], y, label="Статья", s=55)
        ax.scatter([row[observed_key] for row in rows], y, label="Эксперимент", marker="x", s=60)
        ax.set(xlabel="Проценты", title=title, yticks=y)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticklabels([row["label"] for row in rows])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    save(fig, f"regbench_1000_eight_methods{suffix}")


def effects(tag, strict):
    path = ROOT / f"results/regbench_paired_bootstrap_{tag}.json"
    if not require(path, strict):
        return
    rows = json.loads(path.read_text())
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ci_key, title in (
        (axes[0], "delta_accuracy_pp", "accuracy_bootstrap_95_pp", "Accuracy"),
        (axes[1], "delta_tvd_pp", "tvd_bootstrap_95_pp", "TVD"),
    ):
        values = np.array([row[key] for row in rows])
        intervals = np.array([row[ci_key] for row in rows])
        errors = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))
        ax.errorbar(x, values, yerr=np.maximum(errors, 0), fmt="o", capsize=4)
        ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set(xticks=x, xticklabels=[str(row["training_examples"]) for row in rows],
               ylabel="Epanechnikov − Gaussian, п.п.", title=title)
        ax.grid(axis="y", alpha=0.2)
    save(fig, f"regbench_paired_effects_{tag}")


def paired_scores(tag, strict):
    paths = [ROOT / f"results/regbench_{size}_seed0_{tag}.json" for size in SIZES]
    if not all(require(path, strict) for path in paths):
        return
    grouped = {size: json.loads(path.read_text()) for size, path in zip(SIZES, paths)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, label in (("softmax", "Gaussian"), ("epanechnikov", "Epanechnikov")):
        results = [next(row for row in grouped[size] if row["run"].startswith(method)) for size in SIZES]
        axes[0].plot(SIZES, [100 * row["dfa_accuracy"] for row in results], marker="o",
                     label=label, color=COLORS[method])
        axes[1].plot(SIZES, [100 * row["tvd"] for row in results], marker="o",
                     label=label, color=COLORS[method])
    axes[0].set(ylabel="Accuracy, %", title="Accuracy")
    axes[1].set(ylabel="TVD, %", title="TVD")
    for ax in axes:
        ax.set(xticks=SIZES)
        ax.grid(alpha=0.2)
        ax.legend()
    save(fig, f"regbench_paired_scores_{tag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "kernel", "curves", "methods", "effects", "overview"), default="all")
    parser.add_argument("--tag", default="paired_hash0")
    parser.add_argument("--strict", action="store_true", help="Fail when an expected saved result is missing")
    args = parser.parse_args()
    plt.rcParams.update(STYLE)
    if args.only in ("all", "kernel"):
        kernel(args.strict)
    if args.only in ("all", "curves"):
        for size in SIZES:
            for tag in ("", args.tag):
                curves(size, tag, args.strict)
    if args.only in ("all", "methods"):
        methods(args.tag, args.strict)
    if args.only in ("all", "effects"):
        effects(args.tag, args.strict)
    if args.only in ("all", "overview"):
        paired_scores(args.tag, args.strict)


if __name__ == "__main__":
    main()
