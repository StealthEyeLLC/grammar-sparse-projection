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
from gsp.triton_runtime import (
    adaptive_grammar_logits,
    allocate_pinned_bitmask,
    dense_masked_logits,
    full_word_fraction_cpu,
)

MODEL = "Qwen/Qwen3.5-2B"
PAIRS = 30


def make_prompt(schema):
    return (
        "Return only compact JSON matching this JSON Schema. Do not add explanation.\nSchema:"
        + json.dumps(schema, separators=(",", ":"))
        + "\nJSON:"
    )


def new_matcher(compiler, schema):
    return xgr.GrammarMatcher(compiler.compile_json_schema(json.dumps(schema)))


def pct(xs, p):
    ys = sorted(xs)
    return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]


@torch.inference_mode()
def run_sequence(model, tokenizer, compiler, case, mode):
    vocab = model.config.vocab_size
    matcher = new_matcher(compiler, case["schema"])
    mask_cpu = allocate_pinned_bitmask(1, vocab)
    mask_gpu = torch.empty_like(mask_cpu, device=model.device)
    targets = tokenizer.encode(case["sample"], add_special_tokens=False)
    prompt = tokenizer.encode(
        make_prompt(case["schema"]), add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    past = None
    prev = None
    gpu_steps = []
    wall_steps = []
    routes = []

    for step, target in enumerate(targets):
        wall0 = time.perf_counter_ns()
        matcher.fill_next_token_bitmask(mask_cpu)
        fractions = full_word_fraction_cpu(mask_cpu)
        mask_gpu.copy_(mask_cpu, non_blocking=True)

        current = prompt if step == 0 else torch.tensor(
            [[prev]], device=model.device, dtype=torch.long
        )
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        out = model.model(
            input_ids=current,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]

        if mode == "dense":
            logits = dense_masked_logits(hidden, model.lm_head.weight, mask_gpu)
            route = "dense-mask"
        elif mode == "adaptive":
            logits, route = adaptive_grammar_logits(
                hidden,
                model.lm_head.weight,
                mask_gpu,
                full_word_fraction=fractions,
            )
        else:
            raise ValueError(mode)
        _ = logits.argmax(-1)
        b.record(); b.synchronize()
        gpu_steps.append(a.elapsed_time(b))
        wall_steps.append((time.perf_counter_ns()-wall0)/1e6)
        routes.append(route)

        if not matcher.accept_token(int(target)):
            raise RuntimeError((case["name"], step, target))
        prev = int(target)

    gd = gpu_steps[1:] if len(gpu_steps)>1 else gpu_steps
    wd = wall_steps[1:] if len(wall_steps)>1 else wall_steps
    return {
        "gpu_decode_mean_ms": statistics.fmean(gd),
        "wall_decode_mean_ms": statistics.fmean(wd),
        "routes": routes,
    }


@torch.inference_mode()
def correctness(model, tokenizer, compiler, case):
    vocab=model.config.vocab_size
    matcher=new_matcher(compiler,case["schema"])
    mc=allocate_pinned_bitmask(1,vocab)
    mg=torch.empty_like(mc,device=model.device)
    targets=tokenizer.encode(case["sample"],add_special_tokens=False)
    prompt=tokenizer.encode(make_prompt(case["schema"]),add_special_tokens=False,return_tensors="pt").to(model.device)
    past=None; prev=None; rows=[]
    for step,target in enumerate(targets):
        matcher.fill_next_token_bitmask(mc)
        frac=full_word_fraction_cpu(mc)
        mg.copy_(mc,non_blocking=True)
        current=prompt if step==0 else torch.tensor([[prev]],device=model.device,dtype=torch.long)
        out=model.model(input_ids=current,past_key_values=past,use_cache=True,return_dict=True)
        past=out.past_key_values
        h=out.last_hidden_state[:,-1,:]
        ref=dense_masked_logits(h,model.lm_head.weight,mg)
        got,route=adaptive_grammar_logits(h,model.lm_head.weight,mg,full_word_fraction=frac)
        if ref.ndim==2: ref=ref[0]
        if got.ndim==2: got=got[0]
        torch.cuda.synchronize()
        fm=bool(torch.equal(torch.isfinite(ref),torch.isfinite(got)))
        tm=int(ref.argmax())==int(got.argmax())
        rows.append({"step":step,"route":route,"full_word_fraction":frac[0],"finite_match":fm,"top1_match":tm})
        if not fm or not tm: raise AssertionError((case["name"],rows[-1]))
        if not matcher.accept_token(int(target)): raise RuntimeError("target rejected")
        prev=int(target)
    return rows


def main():
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,low_cpu_mem_usage=True).to("cuda").eval()
    ti=xgr.TokenizerInfo.from_huggingface(tok,vocab_size=model.config.vocab_size)
    compiler=xgr.GrammarCompiler(ti)

    run_sequence(model,tok,compiler,CASES[0],"dense")
    run_sequence(model,tok,compiler,CASES[0],"adaptive")

    results=[]
    for case in CASES:
        corr=correctness(model,tok,compiler,case)
        pairs=[]
        for i in range(PAIRS):
            order=("dense","adaptive") if i%2==0 else ("adaptive","dense")
            got={}
            for mode in order:
                got[mode]=run_sequence(model,tok,compiler,case,mode)
            pairs.append({
                "pair":i,"order":list(order),
                "gpu_speedup":got["dense"]["gpu_decode_mean_ms"]/got["adaptive"]["gpu_decode_mean_ms"],
                "wall_speedup":got["dense"]["wall_decode_mean_ms"]/got["adaptive"]["wall_decode_mean_ms"],
                "dense":got["dense"],"adaptive":got["adaptive"],
            })
        gs=[p["gpu_speedup"] for p in pairs]
        ws=[p["wall_speedup"] for p in pairs]
        route_counts={}
        for row in corr: route_counts[row["route"]]=route_counts.get(row["route"],0)+1
        results.append({
            "case":case["name"],
            "gpu_speedup_median":statistics.median(gs),
            "gpu_p10":pct(gs,.1),"gpu_p90":pct(gs,.9),
            "wall_speedup_median":statistics.median(ws),
            "wall_p10":pct(ws,.1),"wall_p90":pct(ws,.9),
            "route_counts":route_counts,
            "correctness":corr,
            "pairs":pairs,
        })

    payload={
        "model":MODEL,"gpu":torch.cuda.get_device_name(0),"pairs":PAIRS,
        "baseline":"full LM head + GPU grammar mask",
        "system":"adaptive GSP: dense fallback for full-word fraction >=0.80; row-sparse batch1; block-sparse batch>=2",
        "results":results,
    }
    out=ROOT/"results"/"qwen35-adaptive-paired"/"results.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps([{
        "case":r["case"],"gpu_median":r["gpu_speedup_median"],"gpu_p10":r["gpu_p10"],"gpu_p90":r["gpu_p90"],
        "wall_median":r["wall_speedup_median"],"routes":r["route_counts"]
    } for r in results],separators=(",",":")))

if __name__=="__main__":
    main()
