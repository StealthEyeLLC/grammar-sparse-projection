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
sys.path.insert(0, str(ROOT / "vendor"))
import xgrammar as xgr

from grammar_density_trace import CASES, allowed_ids_from_bitmask

MODEL = "Qwen/Qwen3-0.6B"
THRESHOLD_K = 45056
TRIALS = 5


def make_prompt(schema):
    compact = json.dumps(schema, separators=(",", ":"))
    return (
        "Return only compact JSON matching this JSON Schema. "
        "Do not add explanation.\nSchema:" + compact + "\nJSON:"
    )


def new_matcher(compiler, schema):
    compiled = compiler.compile_json_schema(json.dumps(schema))
    return xgr.GrammarMatcher(compiled)


@torch.inference_mode()
def run_case(model, tokenizer, compiler, case, mode):
    matcher = new_matcher(compiler, case["schema"])
    bitmask = xgr.allocate_token_bitmask(1, model.config.vocab_size)
    target_ids = tokenizer.encode(case["sample"], add_special_tokens=False)
    prompt_ids = tokenizer.encode(
        make_prompt(case["schema"]),
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    past = None
    prev = None
    sparse_steps = 0
    dense_steps = 0
    ks = []

    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for step, target in enumerate(target_ids):
        matcher.fill_next_token_bitmask(bitmask)

        if mode == "dense":
            ids_np = None
            k = None
        else:
            ids_np = allowed_ids_from_bitmask(bitmask, model.config.vocab_size)
            k = int(ids_np.size)
            ks.append(k)

        if step == 0:
            current = prompt_ids
        else:
            current = torch.tensor([[prev]], device=model.device, dtype=torch.long)

        out = model.model(
            input_ids=current,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]

        if mode == "dense":
            _ = F.linear(hidden, model.lm_head.weight)
            dense_steps += 1
        else:
            if k <= THRESHOLD_K:
                ids_gpu = torch.from_numpy(ids_np).to(
                    device=model.device, dtype=torch.long
                )
                _ = F.linear(hidden, model.lm_head.weight.index_select(0, ids_gpu))
                sparse_steps += 1
            else:
                _ = F.linear(hidden, model.lm_head.weight)
                dense_steps += 1

        accepted = matcher.accept_token(int(target))
        if not accepted:
            raise RuntimeError(
                {
                    "case": case["name"],
                    "step": step,
                    "target": int(target),
                    "text": tokenizer.decode([target]),
                }
            )
        prev = int(target)

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    return {
        "elapsed_ms": elapsed_ms,
        "tokens": len(target_ids),
        "ms_per_token": elapsed_ms / len(target_ids),
        "sparse_steps": sparse_steps,
        "dense_steps": dense_steps,
        "ks": ks,
    }


@torch.inference_mode()
def correctness_case(model, tokenizer, compiler, case):
    matcher = new_matcher(compiler, case["schema"])
    bitmask = xgr.allocate_token_bitmask(1, model.config.vocab_size)
    target_ids = tokenizer.encode(case["sample"], add_special_tokens=False)
    prompt_ids = tokenizer.encode(
        make_prompt(case["schema"]),
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    past = None
    prev = None
    rows = []
    for step, target in enumerate(target_ids):
        matcher.fill_next_token_bitmask(bitmask)
        ids_np = allowed_ids_from_bitmask(bitmask, model.config.vocab_size)
        k = int(ids_np.size)
        ids_gpu = torch.from_numpy(ids_np).to(model.device, dtype=torch.long)

        current = (
            prompt_ids
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

        dense_full = F.linear(hidden, model.lm_head.weight)
        dense_legal = dense_full.index_select(-1, ids_gpu)

        if k <= THRESHOLD_K:
            sparse = F.linear(
                hidden, model.lm_head.weight.index_select(0, ids_gpu)
            )
            delta = (dense_legal.float() - sparse.float()).abs()
            dense_top = ids_gpu[dense_legal.argmax(-1)]
            sparse_top = ids_gpu[sparse.argmax(-1)]
            top_match = bool(torch.equal(dense_top, sparse_top))
            max_abs = float(delta.max().item())
            if not top_match:
                raise AssertionError(
                    {
                        "case": case["name"],
                        "step": step,
                        "k": k,
                        "max_abs": max_abs,
                    }
                )
        else:
            top_match = True
            max_abs = None

        target_pos = np.searchsorted(ids_np, int(target))
        target_legal = bool(
            target_pos < len(ids_np) and ids_np[target_pos] == int(target)
        )
        if not target_legal:
            raise AssertionError(
                {"case": case["name"], "step": step, "target_not_legal": int(target)}
            )

        rows.append(
            {
                "step": step,
                "k": k,
                "mode": "sparse" if k <= THRESHOLD_K else "dense",
                "top1_match": top_match,
                "max_abs_logit_error": max_abs,
            }
        )
        if not matcher.accept_token(int(target)):
            raise RuntimeError("matcher rejected teacher-forced target")
        prev = int(target)

    return rows


def summarize_trials(trials):
    xs = [x["ms_per_token"] for x in trials]
    return {
        "median_ms_per_token": statistics.median(xs),
        "mean_ms_per_token": statistics.fmean(xs),
        "min_ms_per_token": min(xs),
        "max_ms_per_token": max(xs),
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
        low_cpu_mem_usage=False,
    ).to("cuda")
    model.eval()

    vocab_size = int(model.config.vocab_size)
    ti = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(ti)

    case_results = []
    correctness = {}

    # Warm the model kernels before recorded timing.
    _ = run_case(model, tokenizer, compiler, CASES[0], "dense")
    _ = run_case(model, tokenizer, compiler, CASES[0], "gsp")

    for case in CASES:
        correctness[case["name"]] = correctness_case(
            model, tokenizer, compiler, case
        )

        dense_trials = []
        gsp_trials = []
        for i in range(TRIALS):
            if i % 2 == 0:
                dense_trials.append(run_case(model, tokenizer, compiler, case, "dense"))
                gsp_trials.append(run_case(model, tokenizer, compiler, case, "gsp"))
            else:
                gsp_trials.append(run_case(model, tokenizer, compiler, case, "gsp"))
                dense_trials.append(run_case(model, tokenizer, compiler, case, "dense"))

        dense_summary = summarize_trials(dense_trials)
        gsp_summary = summarize_trials(gsp_trials)
        sparse_fraction = (
            gsp_trials[0]["sparse_steps"] / gsp_trials[0]["tokens"]
        )
        speedup = (
            dense_summary["median_ms_per_token"]
            / gsp_summary["median_ms_per_token"]
        )
        case_results.append(
            {
                "case": case["name"],
                "tokens": gsp_trials[0]["tokens"],
                "sparse_fraction": sparse_fraction,
                "dense": dense_summary,
                "gsp": gsp_summary,
                "end_to_end_speedup": speedup,
                "all_sparse_top1_match": all(
                    row["top1_match"] for row in correctness[case["name"]]
                ),
            }
        )

    props = torch.cuda.get_device_properties(0)
    payload = {
        "model": MODEL,
        "model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "vocab_size": vocab_size,
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_vram_bytes": int(props.total_memory),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "threshold_k": THRESHOLD_K,
        "trials": TRIALS,
        "baseline_note": (
            "Dense baseline times grammar state progression + transformer + full "
            "LM-head projection, but deliberately excludes the conventional GPU "
            "grammar-mask application and token-selection costs. This privileges "
            "the baseline."
        ),
        "gsp_note": (
            "GSP includes CPU bitmask-to-ID materialization, host-to-GPU ID copy, "
            "row gathering, and density-adaptive sparse/dense projection."
        ),
        "case_results": case_results,
        "correctness": correctness,
    }

    out = ROOT / "results" / "real-qwen3-0.6b"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    compact = dict(payload)
    compact["correctness"] = {
        k: {
            "steps": len(v),
            "sparse_steps": sum(r["mode"] == "sparse" for r in v),
            "all_top1_match": all(r["top1_match"] for r in v),
            "max_abs_logit_error": max(
                [r["max_abs_logit_error"] for r in v if r["max_abs_logit_error"] is not None]
                or [0.0]
            ),
        }
        for k, v in correctness.items()
    }
    print(json.dumps(compact, separators=(",", ":")))


if __name__ == "__main__":
    main()
