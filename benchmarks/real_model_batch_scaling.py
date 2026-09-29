import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-0.6B"
BATCHES = [1, 2, 4, 8, 16, 32]
KS = [13, 428, 4096, 32768]
HEAD_REPEATS = 80
DECODE_STEPS = 20


@torch.inference_mode()
def cuda_samples(fn, warmup=10, repeats=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        xs.append(float(a.elapsed_time(b)))
    return {
        "median_ms": statistics.median(xs),
        "mean_ms": statistics.fmean(xs),
        "p10_ms": sorted(xs)[max(0, int(0.10 * len(xs)) - 1)],
        "p90_ms": sorted(xs)[min(len(xs) - 1, int(0.90 * len(xs)))],
    }


@torch.inference_mode()
def transformer_decode_samples(model, tokenizer, batch):
    prompt = "Return only compact JSON. JSON:"
    ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    ids = ids.repeat(batch, 1)

    pre = model.model(input_ids=ids, use_cache=True, return_dict=True)
    past = pre.past_key_values
    token = tokenizer.encode("{", add_special_tokens=False)[0]
    current = torch.full((batch, 1), token, device=model.device, dtype=torch.long)

    # First few steps settle kernels/cache growth before recorded samples.
    for _ in range(3):
        out = model.model(
            input_ids=current,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values

    xs = []
    hidden = None
    for _ in range(DECODE_STEPS):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        out = model.model(
            input_ids=current,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        b.record()
        b.synchronize()
        xs.append(float(a.elapsed_time(b)))
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]

    return {
        "median_ms": statistics.median(xs),
        "mean_ms": statistics.fmean(xs),
        "p10_ms": sorted(xs)[max(0, int(0.10 * len(xs)) - 1)],
        "p90_ms": sorted(xs)[min(len(xs) - 1, int(0.90 * len(xs)))],
    }, hidden.detach()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
    model.eval()
    weight = model.lm_head.weight
    vocab = int(weight.shape[0])

    # Fixed token subsets let all batch sizes compare identical row locality.
    torch.manual_seed(20260928)
    ids_by_k = {}
    for k in KS:
        ids_by_k[k] = torch.sort(torch.randperm(vocab, device="cuda")[:k]).values

    rows = []
    for batch in BATCHES:
        decode, hidden = transformer_decode_samples(model, tokenizer, batch)

        dense = cuda_samples(
            lambda: F.linear(hidden, weight),
            repeats=HEAD_REPEATS,
        )
        for k in KS:
            ids = ids_by_k[k]
            packed = weight.index_select(0, ids).contiguous()

            prepacked = cuda_samples(
                lambda p=packed: F.linear(hidden, p),
                repeats=HEAD_REPEATS,
            )
            dynamic = cuda_samples(
                lambda i=ids: F.linear(hidden, weight.index_select(0, i)),
                repeats=HEAD_REPEATS,
            )

            baseline_total = decode["median_ms"] + dense["median_ms"]
            prepacked_total = decode["median_ms"] + prepacked["median_ms"]
            dynamic_total = decode["median_ms"] + dynamic["median_ms"]

            rows.append({
                "batch": batch,
                "k": k,
                "density": k / vocab,
                "transformer_decode_median_ms": decode["median_ms"],
                "dense_head_median_ms": dense["median_ms"],
                "dense_head_fraction_of_transformer_plus_head": (
                    dense["median_ms"] / baseline_total
                ),
                "prepacked_head_median_ms": prepacked["median_ms"],
                "dynamic_head_median_ms": dynamic["median_ms"],
                "head_speedup_prepacked": dense["median_ms"] / prepacked["median_ms"],
                "head_speedup_dynamic": dense["median_ms"] / dynamic["median_ms"],
                "projected_total_speedup_prepacked": baseline_total / prepacked_total,
                "projected_total_speedup_dynamic": baseline_total / dynamic_total,
                "decode": decode,
                "dense": dense,
                "prepacked": prepacked,
                "dynamic": dynamic,
            })
            del packed
        del hidden
        torch.cuda.empty_cache()

    payload = {
        "model": MODEL,
        "vocab_size": vocab,
        "hidden_size": int(model.config.hidden_size),
        "dtype": "float16",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "batches": BATCHES,
        "ks": KS,
        "head_repeats": HEAD_REPEATS,
        "decode_steps": DECODE_STEPS,
        "method_note": (
            "Transformer decode and LM-head kernels are timed separately on real "
            "Qwen3-0.6B hidden states. Projected totals sum their medians. "
            "Prepacked represents legal rows ready when the transformer finishes; "
            "dynamic additionally pays GPU row gathering."
        ),
        "rows": rows,
    }
    out = ROOT / "results" / "real-qwen3-batch-scaling"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    compact = []
    for r in rows:
        compact.append({
            "batch": r["batch"],
            "k": r["k"],
            "transformer_ms": round(r["transformer_decode_median_ms"], 4),
            "dense_head_ms": round(r["dense_head_median_ms"], 4),
            "head_fraction": round(r["dense_head_fraction_of_transformer_plus_head"], 4),
            "dynamic_head_speedup": round(r["head_speedup_dynamic"], 3),
            "projected_total_dynamic": round(r["projected_total_speedup_dynamic"], 4),
            "projected_total_prepacked": round(r["projected_total_speedup_prepacked"], 4),
        })
    print(json.dumps(compact, separators=(",", ":")))


if __name__ == "__main__":
    main()
