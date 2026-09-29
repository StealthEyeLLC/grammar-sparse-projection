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
PAIRS = 10


def make_prompt(schema):
    return (
        "Return only compact JSON matching this JSON Schema. Do not add explanation.\nSchema:"
        + json.dumps(schema, separators=(",", ":"))
        + "\nJSON:"
    )


def new_matcher(compiler, schema):
    return xgr.GrammarMatcher(compiler.compile_json_schema(json.dumps(schema)))


def percentile(xs, p):
    ys = sorted(xs)
    if not ys:
        return None
    return ys[min(len(ys) - 1, max(0, round((len(ys) - 1) * p)))]


@torch.inference_mode()
def run_sequence(model, tokenizer, compiler, case, mode):
    vocab = model.config.vocab_size
    matcher = new_matcher(compiler, case["schema"])
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
    wall_steps = []
    gpu_steps = []
    fill_us = []

    for step, target in enumerate(targets):
        wall0 = time.perf_counter_ns()
        fill0 = time.perf_counter_ns()
        matcher.fill_next_token_bitmask(mask_cpu)
        fill_us.append((time.perf_counter_ns() - fill0) / 1000.0)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        mask_gpu.copy_(mask_cpu, non_blocking=True)

        current = prompt if step == 0 else torch.tensor(
            [[prev]], device=model.device, dtype=torch.long
        )
        out = model.model(
            input_ids=current,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]

        if mode == "dense":
            logits = F.linear(hidden, model.lm_head.weight)
            xgr.apply_token_bitmask_inplace(
                logits, mask_gpu, vocab_size=vocab, backend="triton"
            )
            _ = logits.argmax(-1)
        elif mode == "gsp":
            logits = bitmask_sparse_logits(hidden, model.lm_head.weight, mask_gpu)
            _ = logits.argmax(-1)
        else:
            raise ValueError(mode)

        end.record()
        end.synchronize()
        gpu_steps.append(start.elapsed_time(end))
        wall_steps.append((time.perf_counter_ns() - wall0) / 1e6)

        if not matcher.accept_token(int(target)):
            raise RuntimeError((case["name"], step, target))
        prev = int(target)

    decode_gpu = gpu_steps[1:] if len(gpu_steps) > 1 else gpu_steps
    decode_wall = wall_steps[1:] if len(wall_steps) > 1 else wall_steps
    return {
        "gpu_decode_mean_ms": statistics.fmean(decode_gpu),
        "gpu_decode_median_ms": statistics.median(decode_gpu),
        "wall_decode_mean_ms": statistics.fmean(decode_wall),
        "wall_decode_median_ms": statistics.median(decode_wall),
        "fill_us_median": statistics.median(fill_us),
        "tokens": len(targets),
    }


@torch.inference_mode()
def correctness(model, tokenizer, compiler, case):
    vocab = model.config.vocab_size
    matcher = new_matcher(compiler, case["schema"])
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
        current = prompt if step == 0 else torch.tensor(
            [[prev]], device=model.device, dtype=torch.long
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
        gsp = bitmask_sparse_logits(hidden, model.lm_head.weight, mask_gpu)
        torch.cuda.synchronize()
        d = dense[0]
        finite = torch.isfinite(d)
        row = {
            "step": step,
            "top1_match": int(d.argmax()) == int(gsp.argmax()),
            "finite_match": bool(torch.equal(finite, torch.isfinite(gsp))),
            "max_abs_logit_error": float(
                (d[finite].float() - gsp[finite].float()).abs().max().item()
            ) if finite.any() else 0.0,
        }
        if not row["top1_match"] or not row["finite_match"]:
            raise AssertionError((case["name"], row))
        rows.append(row)
        if not matcher.accept_token(int(target)):
            raise RuntimeError("teacher target rejected")
        prev = int(target)
    return rows


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    ti = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=model.config.vocab_size
    )
    compiler = xgr.GrammarCompiler(ti)

    # Compile all kernels and warm model paths.
    run_sequence(model, tokenizer, compiler, CASES[0], "dense")
    run_sequence(model, tokenizer, compiler, CASES[0], "gsp")

    output = []
    for case in CASES:
        corr = correctness(model, tokenizer, compiler, case)
        pairs = []
        for i in range(PAIRS):
            order = ("dense", "gsp") if i % 2 == 0 else ("gsp", "dense")
            got = {}
            for mode in order:
                got[mode] = run_sequence(model, tokenizer, compiler, case, mode)
            pairs.append({
                "pair": i,
                "order": list(order),
                "dense": got["dense"],
                "gsp": got["gsp"],
                "gpu_ratio": got["dense"]["gpu_decode_mean_ms"] / got["gsp"]["gpu_decode_mean_ms"],
                "wall_ratio": got["dense"]["wall_decode_mean_ms"] / got["gsp"]["wall_decode_mean_ms"],
            })

        gpu_ratios = [x["gpu_ratio"] for x in pairs]
        wall_ratios = [x["wall_ratio"] for x in pairs]
        output.append({
            "case": case["name"],
            "tokens": pairs[0]["dense"]["tokens"],
            "gpu_speedup_median": statistics.median(gpu_ratios),
            "gpu_speedup_p10": percentile(gpu_ratios, 0.10),
            "gpu_speedup_p90": percentile(gpu_ratios, 0.90),
            "wall_speedup_median": statistics.median(wall_ratios),
            "wall_speedup_p10": percentile(wall_ratios, 0.10),
            "wall_speedup_p90": percentile(wall_ratios, 0.90),
            "correctness": corr,
            "pairs": pairs,
        })

    payload = {
        "model": MODEL,
        "pairs": PAIRS,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "vocab": model.config.vocab_size,
        "hidden": model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size,
        "method": "paired AB/BA; GPU event includes pinned H2D mask copy + transformer + head/mask + argmax; wall includes CPU grammar fill",
        "results": output,
    }
    out = ROOT / "results" / "qwen35-paired" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps([
        {
            "case": x["case"],
            "gpu_speedup_median": x["gpu_speedup_median"],
            "gpu_p10": x["gpu_speedup_p10"],
            "gpu_p90": x["gpu_speedup_p90"],
            "wall_speedup_median": x["wall_speedup_median"],
            "wall_p10": x["wall_speedup_p10"],
            "wall_p90": x["wall_speedup_p90"],
        }
        for x in output
    ], separators=(",", ":")))


if __name__ == "__main__":
    main()
