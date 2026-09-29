import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import xgrammar as xgr
from benchmarks.grammar_density_trace import CASES
from gsp.triton_runtime import allocate_pinned_bitmask, bitmask_sparse_logits

MODEL = "Qwen/Qwen3.5-2B"
TRIALS = 5


def make_prompt(schema):
    compact = json.dumps(schema, separators=(",", ":"))
    return (
        "Return only compact JSON matching this JSON Schema. "
        "Do not add explanation.\nSchema:" + compact + "\nJSON:"
    )


def new_matcher(compiler, schema):
    return xgr.GrammarMatcher(compiler.compile_json_schema(json.dumps(schema)))


@torch.inference_mode()
def run_case(model, tokenizer, compiler, case, mode):
    matcher = new_matcher(compiler, case["schema"])
    vocab = model.config.vocab_size
    bitmask_cpu = allocate_pinned_bitmask(1, vocab)
    bitmask_gpu = torch.empty_like(bitmask_cpu, device=model.device)
    targets = tokenizer.encode(case["sample"], add_special_tokens=False)
    prompt = tokenizer.encode(
        make_prompt(case["schema"]),
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    past = None
    prev = None
    step_ms = []
    fill_us = []

    torch.cuda.synchronize()
    total_start = time.perf_counter_ns()

    for step, target in enumerate(targets):
        fill_start = time.perf_counter_ns()
        matcher.fill_next_token_bitmask(bitmask_cpu)
        fill_us.append((time.perf_counter_ns() - fill_start) / 1000.0)
        bitmask_gpu.copy_(bitmask_cpu, non_blocking=True)

        current = (
            prompt
            if step == 0
            else torch.tensor([[prev]], device=model.device, dtype=torch.long)
        )

        torch.cuda.synchronize()
        step_start = time.perf_counter_ns()

        out = model.model(
            input_ids=current,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]

        if mode == "dense_mask":
            logits = F.linear(hidden, model.lm_head.weight)
            xgr.apply_token_bitmask_inplace(
                logits, bitmask_gpu, vocab_size=vocab, backend="triton"
            )
            _ = logits.argmax(-1)
        elif mode == "gsp":
            logits = bitmask_sparse_logits(hidden, model.lm_head.weight, bitmask_gpu)
            _ = logits.argmax(-1)
        else:
            raise ValueError(mode)

        torch.cuda.synchronize()
        step_ms.append((time.perf_counter_ns() - step_start) / 1e6)

        if not matcher.accept_token(int(target)):
            raise RuntimeError((case["name"], step, int(target)))
        prev = int(target)

    total_ms = (time.perf_counter_ns() - total_start) / 1e6
    decode_only = step_ms[1:] if len(step_ms) > 1 else step_ms
    return {
        "tokens": len(targets),
        "total_ms": total_ms,
        "total_ms_per_token": total_ms / len(targets),
        "step_ms": step_ms,
        "decode_only_ms_per_token": statistics.fmean(decode_only),
        "decode_only_median_ms": statistics.median(decode_only),
        "fill_us_median": statistics.median(fill_us),
    }


@torch.inference_mode()
def correctness_case(model, tokenizer, compiler, case):
    matcher = new_matcher(compiler, case["schema"])
    vocab = model.config.vocab_size
    mask_cpu = allocate_pinned_bitmask(1, vocab)
    mask_gpu = torch.empty_like(mask_cpu, device=model.device)
    targets = tokenizer.encode(case["sample"], add_special_tokens=False)
    prompt = tokenizer.encode(
        make_prompt(case["schema"]),
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    past = None
    prev = None
    rows = []
    for step, target in enumerate(targets):
        matcher.fill_next_token_bitmask(mask_cpu)
        mask_gpu.copy_(mask_cpu, non_blocking=True)

        current = (
            prompt
            if step == 0
            else torch.tensor([[prev]], device=model.device, dtype=torch.long)
        )
        out = model.model(
            input_ids=current,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]

        dense = F.linear(hidden, model.lm_head.weight)
        xgr.apply_token_bitmask_inplace(
            dense, mask_gpu, vocab_size=vocab, backend="triton"
        )
        sparse = bitmask_sparse_logits(hidden, model.lm_head.weight, mask_gpu)
        torch.cuda.synchronize()

        dense_1d = dense[0]
        finite = torch.isfinite(dense_1d)
        finite_match = bool(torch.equal(finite, torch.isfinite(sparse)))
        top1_match = int(dense_1d.argmax()) == int(sparse.argmax())
        max_abs = (
            float((dense_1d[finite].float() - sparse[finite].float()).abs().max().item())
            if finite.any()
            else 0.0
        )
        rows.append({
            "step": step,
            "finite_match": finite_match,
            "top1_match": top1_match,
            "max_abs_logit_error": max_abs,
        })
        if not finite_match or not top1_match:
            raise AssertionError((case["name"], rows[-1]))

        if not matcher.accept_token(int(target)):
            raise RuntimeError("teacher-forced target rejected")
        prev = int(target)
    return rows


def summarize(trials, key):
    xs = [x[key] for x in trials]
    return {
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "min": min(xs),
        "max": max(xs),
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()

    vocab = model.config.vocab_size
    ti = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab)
    compiler = xgr.GrammarCompiler(ti)

    # Compile/warm Triton paths before timed trials.
    warm_case = CASES[0]
    run_case(model, tokenizer, compiler, warm_case, "dense_mask")
    run_case(model, tokenizer, compiler, warm_case, "gsp")

    cases = []
    for case in CASES:
        correctness = correctness_case(model, tokenizer, compiler, case)
        dense_trials = [run_case(model, tokenizer, compiler, case, "dense_mask") for _ in range(TRIALS)]
        gsp_trials = [run_case(model, tokenizer, compiler, case, "gsp") for _ in range(TRIALS)]

        dense_decode = summarize(dense_trials, "decode_only_ms_per_token")
        gsp_decode = summarize(gsp_trials, "decode_only_ms_per_token")
        dense_total = summarize(dense_trials, "total_ms_per_token")
        gsp_total = summarize(gsp_trials, "total_ms_per_token")

        cases.append({
            "case": case["name"],
            "tokens": dense_trials[0]["tokens"],
            "decode_dense": dense_decode,
            "decode_gsp": gsp_decode,
            "decode_speedup": dense_decode["median"] / gsp_decode["median"],
            "total_dense": dense_total,
            "total_gsp": gsp_total,
            "total_speedup": dense_total["median"] / gsp_total["median"],
            "correctness": correctness,
        })

    payload = {
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "trials": TRIALS,
        "baseline": "transformer + full LM head + XGrammar Triton mask + argmax",
        "gsp": "transformer + direct grammar-bitmask sparse LM head + argmax",
        "cases": cases,
    }
    out = ROOT / "results" / "real-qwen3-triton-gsp" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps([
        {
            "case": x["case"],
            "decode_speedup": x["decode_speedup"],
            "total_speedup": x["total_speedup"],
            "dense_decode_ms": x["decode_dense"]["median"],
            "gsp_decode_ms": x["decode_gsp"]["median"],
        }
        for x in cases
    ], separators=(",", ":")))


if __name__ == "__main__":
    main()
