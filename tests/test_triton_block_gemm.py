import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import xgrammar as xgr
from gsp.triton_runtime import bitmask_block_gemm_logits


def make_mask(batch, vocab, ks):
    words = np.zeros((batch, (vocab + 31) // 32), dtype=np.uint32)
    g = np.random.default_rng(123)
    for b, k in enumerate(ks):
        ids = g.choice(vocab, size=k, replace=False)
        words[b, ids // 32] |= np.left_shift(np.uint32(1), (ids % 32).astype(np.uint32))
    return torch.from_numpy(words.view(np.int32))


def main():
    torch.manual_seed(99)
    vocab = 248320
    hidden_dim = 2048
    for batch in (2, 4, 8, 16, 32):
        h = torch.randn((batch, hidden_dim), device="cuda", dtype=torch.float16)
        w = torch.randn((vocab, hidden_dim), device="cuda", dtype=torch.float16)
        ks = [7 if i % 2 == 0 else 4096 for i in range(batch)]
        mask = make_mask(batch, vocab, ks).to("cuda")
        dense = F.linear(h, w)
        xgr.apply_token_bitmask_inplace(dense, mask, vocab_size=vocab, backend="triton")
        got = bitmask_block_gemm_logits(h, w, mask)
        torch.cuda.synchronize()
        finite_match = bool(torch.equal(torch.isfinite(dense), torch.isfinite(got)))
        top1_match = bool(torch.equal(dense.argmax(-1), got.argmax(-1)))
        finite = torch.isfinite(dense)
        max_abs = float((dense[finite].float() - got[finite].float()).abs().max().item())
        row = {"batch": batch, "finite_match": finite_match, "top1_match": top1_match, "max_abs": max_abs}
        print(row)
        if not finite_match or not top1_match:
            raise AssertionError(row)
        del h, w, mask, dense, got
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
