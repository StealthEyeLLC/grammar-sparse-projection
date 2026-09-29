import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from gsp.triton_runtime import (
    adaptive_grammar_logits,
    dense_masked_logits,
    full_word_fraction_cpu,
)


def check_case(batch: int, pattern: str, expected_mode: str):
    torch.manual_seed(1000 + batch)
    vocab = 8192
    hidden_dim = 512
    h = torch.randn((batch, hidden_dim), device="cuda", dtype=torch.float16)
    w = torch.randn((vocab, hidden_dim), device="cuda", dtype=torch.float16)
    words = (vocab + 31) // 32
    mask = torch.zeros((batch, words), dtype=torch.int32, pin_memory=True)

    for b in range(batch):
        if pattern == "all_dense":
            mask[b].fill_(-1)
        elif pattern == "all_sparse":
            mask[b, 0] = 1
        elif pattern == "mixed":
            if b % 2:
                mask[b].fill_(-1)
            else:
                mask[b, 0] = 1
        else:
            raise ValueError(pattern)

    mg = mask.to("cuda", non_blocking=True)
    frac = full_word_fraction_cpu(mask)
    got, mode = adaptive_grammar_logits(h, w, mg, full_word_fraction=frac)
    ref = dense_masked_logits(h, w, mg)

    if got.ndim == 1:
        got = got.unsqueeze(0)
    if ref.ndim == 1:
        ref = ref.unsqueeze(0)

    torch.cuda.synchronize()
    finite = bool(torch.equal(torch.isfinite(got), torch.isfinite(ref)))
    top = bool(torch.equal(got.argmax(-1), ref.argmax(-1)))
    row = {
        "batch": batch,
        "pattern": pattern,
        "mode": mode,
        "expected_mode": expected_mode,
        "finite": finite,
        "top1": top,
    }
    print(row)
    if mode != expected_mode or not finite or not top:
        raise AssertionError(row)


def main():
    check_case(1, "all_sparse", "row-sparse")
    check_case(1, "all_dense", "row-sparse")
    for batch in (2, 4, 8, 16):
        check_case(batch, "all_sparse", "block-sparse")
        check_case(batch, "all_dense", "dense-mask")
        check_case(batch, "mixed", "block-sparse")
    check_case(32, "all_sparse", "block-sparse")
    check_case(32, "all_dense", "dense-mask")
    check_case(32, "mixed", "dense-mask")


if __name__ == "__main__":
    main()
