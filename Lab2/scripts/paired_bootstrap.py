import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Lab2.scripts.analyze_regbench import ROOT, Vocab, get_dfa_probs


def contributions(run):
    run_dir = ROOT / "runs/regbench" / run
    exit_file = run_dir / "EVAL_EXIT_CODE"
    if not exit_file.exists() or exit_file.read_text().strip() != "0":
        raise RuntimeError(f"Evaluation incomplete: {run}")
    train_env = (run_dir / "env.txt").read_text().strip()
    eval_env = (run_dir / "eval_env.txt").read_text().strip()
    if train_env != eval_env or train_env == "PYTHONHASHSEED=unset":
        raise RuntimeError(f"Hash seed differs between training and evaluation: {run}")
    folder = sorted((run_dir / "generations").glob("*_test_batch"))[-1]
    rows = folder.with_name(folder.name.replace("_batch", ".txt")).read_text().splitlines()
    values = []
    target_hashes = []
    for path in sorted(folder.glob("*.pkl.gz"), key=lambda p: int(p.name.split(".")[0])):
        with gzip.open(path, "rb") as handle:
            batch = pickle.load(handle)
        vocab = Vocab(batch["vocab"])
        for logits, dfa in zip(batch["probs"], batch["dfas"]):
            target = get_dfa_probs(rows[len(values)].split("\t")[0], dfa, vocab)
            target_hashes.append(hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest())
            logits = np.asarray(logits[: len(target)], dtype=np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)
            assert probs.shape == target.shape
            correct = (target[np.arange(len(target)), probs.argmax(axis=-1)] > 0).sum()
            distance = np.abs(probs - target).sum() / 2
            values.append((correct, distance, len(target)))
    assert len(values) == len(rows), (run, len(values), len(rows))
    sample_hash = hashlib.sha256((run_dir / "samples/test.txt").read_bytes()).hexdigest()
    return np.asarray(values, dtype=np.float64), sample_hash, target_hashes


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "paired_hash0"
    rng = np.random.default_rng(20261010)
    output = []
    for size in (1000, 2500, 5000):
        runs = [f"{method}_{size}_seed0_e200_{tag}" for method in ("softmax", "epanechnikov")]
        gaussian, gaussian_hash, gaussian_targets = contributions(runs[0])
        compact, compact_hash, compact_targets = contributions(runs[1])
        if (gaussian_hash != compact_hash or gaussian_targets != compact_targets
                or not np.array_equal(gaussian[:, 2], compact[:, 2])):
            raise RuntimeError(f"Test examples differ for {size}")
        indices = rng.integers(len(gaussian), size=(10000, len(gaussian)))
        counts = gaussian[indices, 2].sum(axis=1)
        accuracy_deltas = 100 * (compact[indices, 0].sum(axis=1) - gaussian[indices, 0].sum(axis=1)) / counts
        tvd_deltas = 100 * (compact[indices, 1].sum(axis=1) - gaussian[indices, 1].sum(axis=1)) / counts
        total_tokens = gaussian[:, 2].sum()
        output.append({
            "training_examples": size,
            "test_examples": len(gaussian),
            "test_tokens": int(total_tokens),
            "test_sample_sha256": gaussian_hash,
            "delta_accuracy_pp": float(100 * (compact[:, 0].sum() - gaussian[:, 0].sum()) / total_tokens),
            "accuracy_bootstrap_95_pp": np.quantile(accuracy_deltas, [0.025, 0.975]).tolist(),
            "delta_tvd_pp": float(100 * (compact[:, 1].sum() - gaussian[:, 1].sum()) / total_tokens),
            "tvd_bootstrap_95_pp": np.quantile(tvd_deltas, [0.025, 0.975]).tolist(),
        })
    (ROOT / f"results/regbench_paired_bootstrap_{tag}.json").write_text(json.dumps(output, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(output))
    for ax, key, ci_key, title in (
        (axes[0], "delta_accuracy_pp", "accuracy_bootstrap_95_pp", "DFA accuracy"),
        (axes[1], "delta_tvd_pp", "tvd_bootstrap_95_pp", "TVD"),
    ):
        values = np.array([row[key] for row in output])
        intervals = np.array([row[ci_key] for row in output])
        ax.errorbar(x, values, yerr=np.vstack((values - intervals[:, 0], intervals[:, 1] - values)), fmt="o", capsize=4)
        ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set(xticks=x, xticklabels=[str(row["training_examples"]) for row in output],
               xlabel="Training DFA examples", ylabel="Epanechnikov − Gaussian, pp", title=title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Paired RegBench test examples, 95% DFA bootstrap intervals")
    fig.tight_layout()
    fig.savefig(ROOT / f"figures/regbench_paired_effects_{tag}.png", dpi=180)
    fig.savefig(ROOT / f"figures/regbench_paired_effects_{tag}.svg")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
