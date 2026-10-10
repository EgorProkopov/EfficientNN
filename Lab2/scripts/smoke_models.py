"""Verify that the RegBench adapter selects distinct attention methods."""

import torch

from src.models.sequence.mm import StackAssoMem


def main():
    torch.manual_seed(2601)
    softmax = StackAssoMem(vocab_size=20, method="softmax", n_layer=1,
                           n_head=4, n_embd=32, block_size=64).eval()
    sparsemax = StackAssoMem(vocab_size=20, method="epanechnikov", n_layer=1,
                             n_head=4, n_embd=32, block_size=64).eval()
    sparsemax.load_state_dict(softmax.state_dict())
    assert softmax.blocks[0].attn.method == "softmax"
    assert sparsemax.blocks[0].attn.method == "epanechnikov"
    tokens = torch.randint(0, 20, (2, 32), device="cuda")
    with torch.no_grad():
        dense = softmax(tokens)[0].logits
        sparse = sparsemax(tokens)[0].logits
    assert torch.isfinite(dense).all() and torch.isfinite(sparse).all()
    difference = (dense - sparse).abs().max().item()
    assert difference > 0
    print(f"methods=softmax,epanechnikov max_abs_logit_difference={difference:.8g}")


if __name__ == "__main__":
    main()
