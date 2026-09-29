import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
import xgrammar as xgr

from grammar_density_trace import CASES, MODEL, allowed_ids_from_bitmask


def cuda_median_ms(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end))
    return statistics.median(vals)


def sync_sparse_ms(bitmask_cpu, vocab_size, hidden, weight, repeats=10):
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        ids_np = allowed_ids_from_bitmask(bitmask_cpu, vocab_size)
        ids_gpu = torch.from_numpy(ids_np).to(device=weight.device, dtype=torch.long)
        out = F.linear(hidden, weight.index_select(0, ids_gpu))
        torch.cuda.synchronize()
        vals.append((time.perf_counter_ns() - t0) / 1e6)
        del ids_gpu, out
    return statistics.median(vals)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    vocab_size = int(config.vocab_size)
    hidden_dim = 2560
    threshold_k = 45056
    dtype = torch.float16
    device = torch.device("cuda")

    ti = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(ti)

    torch.manual_seed(20260928)
    hidden = torch.randn((1, hidden_dim), device=device, dtype=dtype)
    weight = torch.randn((vocab_size, hidden_dim), device=device, dtype=dtype)
    dense_reference = F.linear(hidden, weight)
    torch.cuda.synchronize()

    case_summaries = []
    traces = {}

    for case in CASES:
        compiled = compiler.compile_json_schema(json.dumps(case["schema"]))
        matcher = xgr.GrammarMatcher(compiled)
        bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        token_ids = tokenizer.encode(case["sample"], add_special_tokens=False)
        rows = []

        for step, token_id in enumerate(token_ids):
            matcher.fill_next_token_bitmask(bitmask)
            ids_np = allowed_ids_from_bitmask(bitmask, vocab_size)
            k = int(ids_np.size)
            ids_gpu = torch.from_numpy(ids_np).to(device=device, dtype=torch.long)

            def dense_projection():
                logits = F.linear(hidden, weight)
                return logits

            dense_ms = cuda_median_ms(dense_projection)

            if k <= threshold_k:
                def sparse_prepared():
                    return F.linear(hidden, weight.index_select(0, ids_gpu))

                sparse_ms = cuda_median_ms(sparse_prepared)
                sync_ms = sync_sparse_ms(bitmask, vocab_size, hidden, weight)

                sparse_logits = F.linear(hidden, weight.index_select(0, ids_gpu))
                ref_selected = dense_reference.index_select(-1, ids_gpu)
                top_ref = ids_gpu[ref_selected.argmax(-1)]
                top_sparse = ids_gpu[sparse_logits.argmax(-1)]
                max_abs = (ref_selected.float() - sparse_logits.float()).abs().max().item()
                top1_match = bool(torch.equal(top_ref, top_sparse))
                if not top1_match:
                    raise AssertionError(
                        {"case": case["name"], "step": step, "k": k, "max_abs": max_abs}
                    )
                mode = "sparse"
                dispatcher_prepared_ms = sparse_ms
                dispatcher_sync_ms = sync_ms
            else:
                sparse_ms = None
                sync_ms = None
                max_abs = None
                top1_match = True
                mode = "dense"
                dispatcher_prepared_ms = dense_ms
                dispatcher_sync_ms = dense_ms

            accepted = matcher.accept_token(int(token_id))
            if not accepted:
                raise RuntimeError(
                    {"case": case["name"], "step": step, "token_id": int(token_id)}
                )

            rows.append({
                "step": step,
                "token_id": int(token_id),
                "token_text": tokenizer.decode([token_id]),
                "k": k,
                "density": k / vocab_size,
                "mode": mode,
                "dense_projection_ms": dense_ms,
                "sparse_prepared_ms": sparse_ms,
                "sparse_sync_ms": sync_ms,
                "dispatcher_prepared_ms": dispatcher_prepared_ms,
                "dispatcher_sync_ms": dispatcher_sync_ms,
                "max_abs_logit_error": max_abs,
                "top1_match": top1_match,
            })
            del ids_gpu

        dense_vals = [r["dense_projection_ms"] for r in rows]
        prep_vals = [r["dispatcher_prepared_ms"] for r in rows]
        sync_vals = [r["dispatcher_sync_ms"] for r in rows]
        sparse_steps = sum(r["mode"] == "sparse" for r in rows)
        summary = {
            "case": case["name"],
            "tokens": len(rows),
            "sparse_steps": sparse_steps,
            "sparse_fraction": sparse_steps / len(rows),
            "dense_projection_mean_ms": statistics.fmean(dense_vals),
            "dispatcher_prepared_mean_ms": statistics.fmean(prep_vals),
            "dispatcher_sync_mean_ms": statistics.fmean(sync_vals),
            "prepared_speedup": statistics.fmean(dense_vals) / statistics.fmean(prep_vals),
            "sync_speedup": statistics.fmean(dense_vals) / statistics.fmean(sync_vals),
            "all_top1_match": all(r["top1_match"] for r in rows),
        }
        case_summaries.append(summary)
        traces[case["name"]] = rows

    out = ROOT / "results" / "grammar-end-to-end"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_tokenizer": MODEL,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "dtype": "float16",
        "threshold_k": threshold_k,
        "case_summaries": case_summaries,
        "traces": traces,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "model_tokenizer": MODEL,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "threshold_k": threshold_k,
        "case_summaries": case_summaries,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
