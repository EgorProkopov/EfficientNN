import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "seq_icl"))
from train import SequenceLightningModule, create_trainer  # noqa: E402
from src.utils.train import process_config  # noqa: E402


def main(run_name):
    run_dir = ROOT / "runs/regbench" / run_name
    checkpoints = list((ROOT / "checkpoints/regbench" / run_name).glob("**/*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one best checkpoint, got {checkpoints}")
    config = process_config(OmegaConf.load(run_dir / ".hydra/config.yaml"))
    config.callbacks = {}
    os.chdir(run_dir)
    model = SequenceLightningModule(config)
    trainer = create_trainer(config)
    trainer.test(model, ckpt_path=str(checkpoints[0]))


if __name__ == "__main__":
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    main(sys.argv[1])
