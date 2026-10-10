import json
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Lab2.scripts.analyze_regbench import final_metrics


ROOT = Path(__file__).resolve().parent
METHODS = (
    ("softmax", "Gaussian", 93.9, 34.1),
    ("epanechnikov", "Epanechnikov / sparsemax", 94.2, 32.5),
    ("biweight", "Biweight / 1.5-entmax", 93.8, 33.0),
    ("triweight", "Triweight / 4/3-entmax", 93.9, 32.9),
    ("relumax", "Epanechnikov / 1-ReLUmax", 93.8, 36.8),
    ("norm_relu", "Epanechnikov / normReLU", 84.3, 43.6),
    ("top16_softmax", "Gaussian kNN, k=16", 93.4, 33.3),
    ("top16_uniform", "Uniform kNN, k=16", 84.6, 41.1),
)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    suffix = f"_{tag}" if tag else ""
    initial = json.loads((ROOT / f"results/regbench_1000_seed0{suffix}.json").read_text())
    by_method = {item["run"].split("_1000_")[0]: item for item in initial}
    for method, _, _, _ in METHODS[2:]:
        run = f"{method}_1000_seed0_e200{suffix}"
        exit_file = ROOT / "runs/regbench" / run / "EVAL_EXIT_CODE"
        if not exit_file.exists() or exit_file.read_text().strip() != "0":
            raise RuntimeError(f"Evaluation incomplete: {run}")
        by_method[method] = final_metrics(run)

    comparison = []
    sample_hashes = None
    for method, label, paper_accuracy, paper_tvd in METHODS:
        result = by_method[method]
        run_dir = ROOT / "runs/regbench" / result["run"]
        if tag:
            train_env = (run_dir / "env.txt").read_text().strip()
            eval_env = (run_dir / "eval_env.txt").read_text().strip()
            if train_env != eval_env or train_env == "PYTHONHASHSEED=unset":
                raise RuntimeError(f"Hash seed differs for {method}")
        samples_dir = run_dir / "samples"
        hashes = {
            split: hashlib.sha256((samples_dir / f"{split}.txt").read_bytes()).hexdigest()
            for split in ("val", "test")
        }
        if tag and sample_hashes is not None and hashes != sample_hashes:
            raise RuntimeError(f"Sample mismatch for {method}: {hashes}")
        sample_hashes = hashes
        comparison.append({
            "method": method,
            "label": label,
            "paper_accuracy_percent": paper_accuracy,
            "observed_accuracy_percent": result["dfa_accuracy"] * 100,
            "paper_tvd_percent": paper_tvd,
            "observed_tvd_percent": result["tvd"] * 100,
            "test_examples": result["examples"],
            "test_tokens": result["tokens"],
            "sample_sha256": hashes,
        })
    (ROOT / f"results/regbench_1000_eight_methods{suffix}.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.7), sharey=True)
    y = np.arange(len(comparison))
    for ax, paper_key, observed_key, title in (
        (axes[0], "paper_accuracy_percent", "observed_accuracy_percent", "Accuracy ↑"),
        (axes[1], "paper_tvd_percent", "observed_tvd_percent", "TVD ↓"),
    ):
        for index, row in enumerate(comparison):
            ax.plot([row[paper_key], row[observed_key]], [index, index], color="0.7", lw=2)
        ax.scatter([row[paper_key] for row in comparison], y, label="Paper", s=55)
        ax.scatter([row[observed_key] for row in comparison], y, label="Reproduction", marker="x", s=60)
        ax.set(xlabel="Percent", title=title, yticks=y)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticklabels([row["label"] for row in comparison])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    fig.suptitle(f"RegBench, 1k training examples, seed 0{suffix}")
    fig.tight_layout()
    fig.savefig(ROOT / f"figures/regbench_1000_eight_methods{suffix}.png", dpi=180)
    fig.savefig(ROOT / f"figures/regbench_1000_eight_methods{suffix}.svg")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
