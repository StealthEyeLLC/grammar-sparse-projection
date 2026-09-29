import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import xgrammar as xgr
from gsp.triton_runtime import bitmask_sparse_logits


def ids_to_bitmask(ids, vocab):
    words = np.zeros((vocab + 31) // 32, dtype=np.uint32)
    for token_id in ids:
        words[token_id // 32] |= np.uint32(1) << np.uint32(token_id % 32)
    return torch.from_numpy(words.view(np.int32)).reshape(1, -1)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(17)
    vocab = 151936
    hidden_dim = 1024
    h = torch.randn((1, hidden_dim), device="cuda", dtype=torch.float16)
    w = torch.randn((vocab, hidden_dim), device="cuda", dtype=torch.float16)

    cases = [1, 13, 428, 4096, 32768, 120000]
    report = []
    for k in cases:
        ids = torch.randperm(vocab, device="cpu")[:k].numpy()
        mask_cpu = ids_to_bitmask(ids, vocab)
        mask_gpu = mask_cpu.to("cuda")

        dense = F.linear(h, w)
        xgr.apply_token_bitmask_inplace(
            dense, mask_gpu, vocab_size=vocab, backend="triton"
        )
        sparse = bitmask_sparse_logits(h, w, mask_gpu)

        torch.cuda.synchronize()
        dense_1d = dense[0]
        finite = torch.isfinite(dense_1d)
        max_err = (
            (dense_1d[finite].float() - sparse[finite].float()).abs().max().item()
            if finite.any()
            else 0.0
        )
        row = {
            "k": k,
            "top1_match": int(dense_1d.argmax()) == int(sparse.argmax()),
            "finite_match": bool(torch.equal(finite, torch.isfinite(sparse))),
            "max_abs_logit_error": max_err,
        }
        report.append(row)
        if not row["top1_match"] or not row["finite_match"]:
            raise AssertionError(row)
    print(report)


if __name__ == "__main__":
    main()
