import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsp.runtime import constrained_projection


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(20260928)
    device = "cuda"
    vocab = 151_936
    hidden_dim = 2_560
    ks = [1, 4, 16, 64, 256, 1024, 4096, 16384, 32768]
    report = []

    for dtype in (torch.float16, torch.bfloat16):
        h = torch.randn((1, hidden_dim), device=device, dtype=dtype)
        w = torch.randn((vocab, hidden_dim), device=device, dtype=dtype)
        dense = F.linear(h, w)

        for k in ks:
            idx = torch.randperm(vocab, device=device)[:k]
            result = constrained_projection(h, w, idx, dense_threshold=1.0)
            reference = dense.index_select(-1, result.token_ids)
            delta = (reference.float() - result.logits.float()).abs()
            ref_top = result.token_ids[reference.argmax(-1)]
            sparse_top = result.token_ids[result.logits.argmax(-1)]
            ref_prob = torch.softmax(reference.float(), dim=-1)
            sparse_prob = torch.softmax(result.logits.float(), dim=-1)
            l1 = (ref_prob - sparse_prob).abs().sum().item()

            row = {
                "dtype": str(dtype).replace("torch.", ""),
                "k": k,
                "max_abs_logit_error": delta.max().item(),
                "mean_abs_logit_error": delta.mean().item(),
                "probability_l1": l1,
                "top1_token_match": bool(torch.equal(ref_top, sparse_top)),
            }
            report.append(row)
            if not row["top1_token_match"]:
                raise AssertionError(row)
        del h, w, dense
        torch.cuda.empty_cache()

    out = ROOT / "results" / "exactness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
