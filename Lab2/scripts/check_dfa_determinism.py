import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "seq_icl"))
from src.dataloaders.dfa import ICLDFADataModule  # noqa: E402


def main(label):
    dataset = ICLDFADataModule(
        num_examples=1000,
        num_test_examples=1000,
        vocab_size=20,
        max_num_nodes=12,
        max_num_in_context_examples=20,
        min_num_in_context_examples=10,
        max_outgoing_edges=4,
        max_len_per_example=50,
        input_seq_len=4096,
        seed=0,
        batch_size=32,
    )
    dataset.setup()
    result = {"label": label, "python_hash_seed": os.environ.get("PYTHONHASHSEED")}
    for split in ("train", "val", "test"):
        data = dataset.dataset[split]
        digest = hashlib.sha256()
        digest.update(data.inputs.numpy().tobytes())
        digest.update(data.targets.numpy().tobytes())
        result[split] = {"examples": len(data), "sha256": digest.hexdigest()}
    path = ROOT / "results" / f"dfa_determinism_{label}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
