import json
import statistics
import sys
import time
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
from gsp.triton_runtime import allocate_pinned_bitmask, bitmask_sparse_logits, copy_bitmask_to_gpu

MODEL = "Qwen/Qwen3-0.6B"


def cuda_median_ms(fn, warmup=20, repeats=100):
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


def allowed_k(mask_cpu, vocab):
    raw = mask_cpu[0].numpy().view(np.uint32).view(np.uint8)
    return int(np.unpackbits(raw, bitorder="little")[:vocab].sum())


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    weight = model.lm_head.weight
    vocab, hidden_dim = weight.shape
    hidden = torch.randn((1, hidden_dim), device="cuda", dtype=weight.dtype)

    ti = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab)
    compiler = xgr.GrammarCompiler(ti)

    report = []
    for case in CASES:
        matcher = xgr.GrammarMatcher(
            compiler.compile_json_schema(json.dumps(case["schema"]))
        )
        mask_cpu = allocate_pinned_bitmask(1, vocab)
        mask_gpu = torch.empty_like(mask_cpu, device="cuda")
        rows = []
        token_ids = tokenizer.encode(case["sample"], add_special_tokens=False)

        for step, token_id in enumerate(token_ids):
            t0 = time.perf_counter_ns()
            matcher.fill_next_token_bitmask(mask_cpu)
            fill_us = (time.perf_counter_ns() - t0) / 1000.0
            k = allowed_k(mask_cpu, vocab)
            full_word_fraction = float((mask_cpu == -1).sum().item()) / mask_cpu.numel()

            copy_bitmask_to_gpu(mask_cpu, mask_gpu)
            torch.cuda.synchronize()

            def dense_masked():
                logits = F.linear(hidden, weight)
                xgr.apply_token_bitmask_inplace(
                    logits, mask_gpu, vocab_size=vocab, backend="triton"
                )
                return logits

            def sparse():
                return bitmask_sparse_logits(hidden, weight, mask_gpu)

            dense_ms = cuda_median_ms(dense_masked, repeats=60)
            sparse_ms = cuda_median_ms(sparse, repeats=60)

            dense_logits = dense_masked()[0]
            sparse_logits = sparse()
            torch.cuda.synchronize()
            top1_match = int(dense_logits.argmax()) == int(sparse_logits.argmax())
            finite_match = bool(torch.equal(torch.isfinite(dense_logits), torch.isfinite(sparse_logits)))
            if not top1_match or not finite_match:
                raise AssertionError((case["name"], step, k, top1_match, finite_match))

            t1 = time.perf_counter_ns()
            mask_gpu.copy_(mask_cpu, non_blocking=True)
            sparse()
            torch.cuda.synchronize()
            sparse_with_copy_ms = (time.perf_counter_ns() - t1) / 1e6

            rows.append({
                "step": step,
                "k": k,
                "density": k / vocab,
                "full_word_fraction": full_word_fraction,
                "fill_us": fill_us,
                "dense_masked_ms": dense_ms,
                "sparse_kernel_ms": sparse_ms,
                "sparse_speedup": dense_ms / sparse_ms,
                "sparse_with_copy_ms": sparse_with_copy_ms,
                "top1_match": top1_match,
            })
            if not matcher.accept_token(int(token_id)):
                raise RuntimeError((case["name"], step, token_id))

        summary = {
            "case": case["name"],
            "tokens": len(rows),
            "sparse_kernel_faster_fraction": sum(r["sparse_speedup"] > 1 for r in rows) / len(rows),
            "median_dense_masked_ms": statistics.median(r["dense_masked_ms"] for r in rows),
            "median_sparse_kernel_ms": statistics.median(r["sparse_kernel_ms"] for r in rows),
            "median_sparse_speedup": statistics.median(r["sparse_speedup"] for r in rows),
            "median_sparse_with_copy_ms": statistics.median(r["sparse_with_copy_ms"] for r in rows),
            "median_fill_us": statistics.median(r["fill_us"] for r in rows),
            "max_k_where_sparse_faster": max((r["k"] for r in rows if r["sparse_speedup"] > 1), default=None),
        }
        report.append({"summary": summary, "rows": rows})

    out = ROOT / "results" / "triton-direct-bitmask" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(weight.dtype).replace("torch.", ""),
        "vocab": vocab,
        "hidden": hidden_dim,
        "results": report,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps([x["summary"] for x in report], separators=(",", ":")))


if __name__ == "__main__":
    main()
