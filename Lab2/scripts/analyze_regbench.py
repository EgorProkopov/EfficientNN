import csv
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


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "seq_icl"))
from analyze import get_dfa_probs  # noqa: E402


class Vocab:
    def __init__(self, tokens):
        self.tokens = tokens
        self.inv_vocab = {token: index for index, token in enumerate(tokens)}

    def get_id(self, token):
        return self.inv_vocab[token]

    def __len__(self):
        return len(self.tokens)


def curves(run):
    path = ROOT / "runs/regbench" / run / "metrics/version_0/metrics.csv"
    by_epoch = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["epoch"]:
                continue
            epoch = int(row["epoch"])
            values = by_epoch.setdefault(epoch, {})
            for key in ("test/dfa_accuracy", "val/dfa_accuracy", "val/loss"):
                if row[key]:
                    values[key] = float(row[key])
    return by_epoch


def final_metrics(run):
    run_dir = ROOT / "runs/regbench" / run
    folders = sorted((run_dir / "generations").glob("*_test_batch"))
    if not folders:
        raise FileNotFoundError(f"No final generation files for {run}")
    folder = folders[-1]
    rows_path = folder.with_name(folder.name.replace("_batch", ".txt"))
    with rows_path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    correct = 0
    distance = 0.0
    tokens = 0
    examples = 0
    for file in sorted(folder.glob("*.pkl.gz"), key=lambda p: int(p.name.split(".")[0])):
        with gzip.open(file, "rb") as handle:
            batch = pickle.load(handle)
        vocab = Vocab(batch["vocab"])
        for logits, dfa in zip(batch["probs"], batch["dfas"]):
            index = examples
            if index >= len(rows):
                break
            input_string = rows[index][0]
            target = get_dfa_probs(input_string, dfa, vocab)
            logits = np.asarray(logits[: len(target)], dtype=np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)
            assert probs.shape == target.shape
            distance += np.abs(probs - target).sum()
            correct += (target[np.arange(len(target)), probs.argmax(axis=-1)] > 0).sum()
            tokens += len(target)
            examples += 1
    assert examples == len(rows), (examples, len(rows))
    return {
        "run": run,
        "examples": examples,
        "tokens": tokens,
        "dfa_accuracy": float(correct / tokens),
        "tvd": float(distance / (2 * tokens)),
        "generation_folder": str(folder.relative_to(ROOT)),
    }


def main():
    training_examples = int(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    suffix = f"_{tag}" if tag else ""
    runs = [
        f"softmax_{training_examples}_seed0_e200{suffix}",
        f"epanechnikov_{training_examples}_seed0_e200{suffix}",
    ]
    for run in runs:
        exit_file = ROOT / "runs/regbench" / run / "EXIT_CODE"
        eval_file = exit_file.with_name("EVAL_EXIT_CODE")
        if not eval_file.exists() or eval_file.read_text().strip() != "0":
            raise RuntimeError(f"Run is incomplete: {run}")
    if tag:
        sample_hashes = []
        for run in runs:
            run_dir = ROOT / "runs/regbench" / run
            train_env = (run_dir / "env.txt").read_text().strip()
            eval_env = (run_dir / "eval_env.txt").read_text().strip()
            if train_env != eval_env or train_env == "PYTHONHASHSEED=unset":
                raise RuntimeError(f"Hash seed differs between training and evaluation: {run}")
            sample_hashes.append(tuple(
                hashlib.sha256((run_dir / "samples" / f"{split}.txt").read_bytes()).hexdigest()
                for split in ("val", "test")
            ))
        if sample_hashes[0] != sample_hashes[1]:
            raise RuntimeError(f"Validation or test examples differ: {sample_hashes}")
    results = [final_metrics(run) for run in runs]
    output = ROOT / f"results/regbench_{training_examples}_seed0{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for run, result in zip(runs, results):
        data = curves(run)
        label = "Gaussian / softmax" if run.startswith("softmax") else "Epanechnikov / sparsemax"
        for ax, key in zip(axes, ("test/dfa_accuracy", "val/loss")):
            points = [(epoch, row[key]) for epoch, row in sorted(data.items()) if key in row]
            ax.plot([p[0] for p in points], [p[1] for p in points], label=label)
        print(result)
    axes[0].set(ylabel="DFA accuracy", xlabel="Epoch", title=f"RegBench, {training_examples} training examples")
    axes[1].set(ylabel="Validation loss", xlabel="Epoch", title=f"RegBench, {training_examples} training examples")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / f"figures/regbench_{training_examples}_seed0{suffix}.png", dpi=180)
    fig.savefig(ROOT / f"figures/regbench_{training_examples}_seed0{suffix}.svg")


if __name__ == "__main__":
    main()
