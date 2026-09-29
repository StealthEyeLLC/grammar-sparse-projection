import argparse
import csv
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch


def cuda_time_ms(fn, warmup=25, repeats=100):
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
    vals.sort()
    return {
        "median_ms": statistics.median(vals),
        "p10_ms": vals[max(0, int(len(vals) * 0.10) - 1)],
        "p90_ms": vals[min(len(vals) - 1, int(len(vals) * 0.90))],
        "mean_ms": statistics.fmean(vals),
    }


def bench_case(vocab, hidden, dtype, ks, repeats, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    h = torch.randn((1, hidden), device=device, dtype=dtype)
    w = torch.randn((vocab, hidden), device=device, dtype=dtype)

    dense = cuda_time_ms(lambda: torch.nn.functional.linear(h, w), repeats=repeats)
    rows = []

    for k in ks:
        if k > vocab:
            continue
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + k)
        idx = torch.randperm(vocab, generator=gen, device=device)[:k]
        idx_sorted = torch.sort(idx).values

        # Idealized upper bound: grammar compiler/runtime has already materialized
        # the exact legal rows into a compact contiguous buffer.
        w_prepacked = w.index_select(0, idx_sorted).contiguous()
        prepacked = cuda_time_ms(
            lambda: torch.nn.functional.linear(h, w_prepacked),
            repeats=repeats,
        )

        # Dynamic path: exact allowed IDs arrive this step and row gathering is
        # paid in the timed critical path.
        dynamic = cuda_time_ms(
            lambda: torch.nn.functional.linear(h, w.index_select(0, idx_sorted)),
            repeats=repeats,
        )

        rows.append({
            "vocab": vocab,
            "hidden": hidden,
            "dtype": str(dtype).replace("torch.", ""),
            "k": k,
            "density": k / vocab,
            "dense_median_ms": dense["median_ms"],
            "prepacked_median_ms": prepacked["median_ms"],
            "dynamic_median_ms": dynamic["median_ms"],
            "prepacked_speedup": dense["median_ms"] / prepacked["median_ms"],
            "dynamic_speedup": dense["median_ms"] / dynamic["median_ms"],
            "dense": dense,
            "prepacked": prepacked,
            "dynamic": dynamic,
        })
        del w_prepacked, idx, idx_sorted
        torch.cuda.empty_cache()

    del h, w
    torch.cuda.empty_cache()
    return dense, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--ks",
        default="1,4,16,64,256,1024,4096,16384,32768,65536,131072",
    )
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment.")

    dtype = getattr(torch, args.dtype)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    props = torch.cuda.get_device_properties(0)
    meta = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "total_vram_bytes": props.total_memory,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vocab": args.vocab,
        "hidden": args.hidden,
        "dtype": args.dtype,
        "repeats": args.repeats,
        "seed": args.seed,
        "ks": ks,
        "claim_under_test": (
            "Exact grammar-constrained decoding can avoid full-vocabulary output "
            "projection when the legal-token set is sufficiently sparse."
        ),
    }

    dense, rows = bench_case(
        args.vocab, args.hidden, dtype, ks, args.repeats, args.seed
    )

    # Empirical break-even: largest K where the timed method still beats dense.
    winning_prepacked = [r["k"] for r in rows if r["prepacked_speedup"] > 1.0]
    winning_dynamic = [r["k"] for r in rows if r["dynamic_speedup"] > 1.0]
    summary = {
        "metadata": meta,
        "dense": dense,
        "rows": rows,
        "break_even_observed": {
            "prepacked_max_k_faster_than_dense": max(winning_prepacked) if winning_prepacked else None,
            "dynamic_max_k_faster_than_dense": max(winning_dynamic) if winning_dynamic else None,
        },
    }

    (args.out / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "vocab", "hidden", "dtype", "k", "density",
            "dense_median_ms", "prepacked_median_ms", "dynamic_median_ms",
            "prepacked_speedup", "dynamic_speedup",
        ]
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in fields})

    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
