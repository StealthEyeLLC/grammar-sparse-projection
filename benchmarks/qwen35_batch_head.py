import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import xgrammar as xgr
from benchmarks.grammar_density_trace import CASES
from gsp.triton_runtime import bitmask_sparse_logits

MODEL = "Qwen/Qwen3.5-2B"
BATCHES = [1, 2, 4, 8, 16, 32]


def cuda_median_ms(fn, warmup=15, repeats=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        vals.append(a.elapsed_time(b))
    return statistics.median(vals)


def allowed_k(mask, vocab):
    raw = mask.numpy().view(np.uint32).view(np.uint8)
    return int(np.unpackbits(raw, bitorder="little")[:vocab].sum())


def snapshot_mask(compiler, tokenizer, case, step_index, vocab):
    matcher = xgr.GrammarMatcher(
        compiler.compile_json_schema(json.dumps(case["schema"]))
    )
    mask = xgr.allocate_token_bitmask(1, vocab)
    ids = tokenizer.encode(case["sample"], add_special_tokens=False)
    for step, token in enumerate(ids):
        matcher.fill_next_token_bitmask(mask)
        if step == step_index:
            return mask[0].clone()
        if not matcher.accept_token(int(token)):
            raise RuntimeError((case["name"], step, token))
    raise ValueError((case["name"], step_index))


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    weight = model.lm_head.weight
    vocab, hidden_dim = weight.shape
    ti = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab)
    compiler = xgr.GrammarCompiler(ti)

    enum_case = next(c for c in CASES if c["name"] == "enum_record")
    free_case = next(c for c in CASES if c["name"] == "free_string")
    sparse_mask = snapshot_mask(compiler, tokenizer, enum_case, 0, vocab)
    free_mask = snapshot_mask(compiler, tokenizer, free_case, 4, vocab)
    masks = {
        "sparse": sparse_mask,
        "free": free_mask,
    }
    mask_meta = {k: {"allowed_k": allowed_k(v, vocab)} for k, v in masks.items()}

    rows = []
    for batch in BATCHES:
        hidden = torch.randn((batch, hidden_dim), device="cuda", dtype=weight.dtype)
        scenarios = {
            "all_sparse": torch.stack([sparse_mask] * batch, dim=0),
            "all_free": torch.stack([free_mask] * batch, dim=0),
            "mixed": torch.stack(
                [sparse_mask if i % 2 == 0 else free_mask for i in range(batch)],
                dim=0,
            ),
        }

        for scenario, mask_cpu in scenarios.items():
            mask_gpu = mask_cpu.to("cuda")

            def dense_masked():
                logits = F.linear(hidden, weight)
                xgr.apply_token_bitmask_inplace(
                    logits, mask_gpu, vocab_size=vocab, backend="triton"
                )
                return logits

            def gsp():
                return bitmask_sparse_logits(hidden, weight, mask_gpu)

            dense_ms = cuda_median_ms(dense_masked)
            gsp_ms = cuda_median_ms(gsp)

            if batch <= 4:
                dense = dense_masked()
                sparse = gsp()
                if sparse.ndim == 1:
                    sparse = sparse.unsqueeze(0)
                torch.cuda.synchronize()
                finite_match = bool(torch.equal(torch.isfinite(dense), torch.isfinite(sparse)))
                top1_match = bool(torch.equal(dense.argmax(-1), sparse.argmax(-1)))
                if not finite_match or not top1_match:
                    raise AssertionError((batch, scenario, finite_match, top1_match))
            else:
                finite_match = None
                top1_match = None

            rows.append({
                "batch": batch,
                "scenario": scenario,
                "dense_masked_ms": dense_ms,
                "gsp_ms": gsp_ms,
                "speedup": dense_ms / gsp_ms,
                "dense_tokens_per_s": batch * 1000.0 / dense_ms,
                "gsp_tokens_per_s": batch * 1000.0 / gsp_ms,
                "finite_match": finite_match,
                "top1_match": top1_match,
            })

    payload = {
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "vocab": vocab,
        "hidden": hidden_dim,
        "mask_meta": mask_meta,
        "rows": rows,
    }
    out = ROOT / "results" / "qwen35-batch-head" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
