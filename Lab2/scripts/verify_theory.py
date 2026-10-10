"""Check the kernel identities in Proposition 1 on normalized random vectors."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(260122766)


def unit_vectors(shape):
    x = RNG.normal(size=shape)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def entmax(scores, alpha):
    if alpha == 1:
        weights = np.exp(scores - scores.max())
        return weights / weights.sum(), None

    r = 1 / (alpha - 1)
    scaled = (alpha - 1) * scores
    high = scaled.max()
    low = scaled.min() - 1
    while np.maximum(scaled - low, 0).__pow__(r).sum() < 1:
        low -= 1
    for _ in range(80):
        threshold = (low + high) / 2
        if np.maximum(scaled - threshold, 0).__pow__(r).sum() > 1:
            low = threshold
        else:
            high = threshold
    threshold = (low + high) / 2
    return np.maximum(scaled - threshold, 0) ** r, threshold


def check(alpha, gamma, query, keys):
    scores = keys @ query / gamma
    weights, threshold = entmax(scores, alpha)
    distances_squared = np.sum((keys - query) ** 2, axis=-1)
    if alpha == 1:
        kernel = np.exp(-distances_squared / (2 * gamma))
        bandwidth_squared = gamma
    else:
        r = 1 / (alpha - 1)
        bandwidth_squared = 2 - 2 * r * gamma * threshold
        kernel = np.maximum(1 - distances_squared / bandwidth_squared, 0) ** r
    kernel /= kernel.sum()
    return float(np.max(np.abs(weights - kernel))), float(bandwidth_squared), weights


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    errors = {}
    bandwidths = {}
    for alpha in (1, 4 / 3, 1.5, 2):
        for gamma in (0.05, 0.1, 0.2, 0.4):
            observed = []
            widths = []
            for _ in range(100):
                query = unit_vectors(16)
                keys = unit_vectors((128, 16))
                error, width, _ = check(alpha, gamma, query, keys)
                observed.append(error)
                widths.append(width)
            label = f"alpha={alpha:g}, gamma={gamma:g}"
            errors[label] = max(observed)
            bandwidths[label] = [min(widths), max(widths)]

    query = unit_vectors(16)
    keys = unit_vectors((128, 16))
    raw_scores = keys @ query
    order = np.argsort(raw_scores)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = []
    weight_curves = []
    support_curves = []
    for alpha, name in ((1, "softmax / Gaussian"), (4 / 3, "4/3-entmax / triweight"),
                        (1.5, "1.5-entmax / biweight"), (2, "sparsemax / Epanechnikov")):
        _, _, weights = check(alpha, 0.1, query, keys)
        axes[0].plot(np.arange(len(order)), weights[order], label=name)
        labels.append(name)
        weight_curves.append(weights[order])
        fractions = []
        for gamma in (0.05, 0.1, 0.2, 0.4):
            _, _, w = check(alpha, gamma, query, keys)
            fractions.append(np.count_nonzero(w) / len(w))
        axes[1].plot((0.05, 0.1, 0.2, 0.4), fractions, marker="o", label=name)
        support_curves.append(fractions)
    axes[0].set(xlabel="Key rank by similarity", ylabel="Attention weight",
                title="One query, 128 normalized keys, temperature 0.1")
    axes[1].set(xlabel="Temperature", ylabel="Fraction of keys with nonzero weight",
                ylim=(-0.03, 1.03), title="Support size for the same query")
    axes[0].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.savefig(ROOT / "figures" / "kernel_equivalence.png", dpi=180)
    fig.savefig(ROOT / "figures" / "kernel_equivalence.svg")
    (ROOT / "results").mkdir(exist_ok=True)
    np.savez_compressed(
        ROOT / "results/kernel_plot_data.npz",
        labels=np.asarray(labels),
        rank=np.arange(len(order)),
        temperatures=np.asarray((0.05, 0.1, 0.2, 0.4)),
        weights=np.asarray(weight_curves),
        support_fraction=np.asarray(support_curves),
    )
    result = {"vectors_per_case": 100, "keys_per_query": 128, "dimension": 16,
              "max_absolute_weight_error": errors, "bandwidth_squared_range": bandwidths}
    (ROOT / "theory_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
