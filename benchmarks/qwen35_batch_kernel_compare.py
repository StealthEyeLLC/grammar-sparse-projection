import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import xgrammar as xgr
from benchmarks.qwen35_batch_head import snapshot_mask
from benchmarks.grammar_density_trace import CASES
from gsp.triton_runtime import bitmask_sparse_logits, bitmask_block_gemm_logits

MODEL = "Qwen/Qwen3.5-2B"
BATCHES = [2, 4, 8, 16, 32]


def tmed(fn, warmup=20, repeats=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        xs.append(a.elapsed_time(b))
    return statistics.median(xs)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    w = model.lm_head.weight
    vocab, hidden = w.shape
    ti = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab)
    compiler = xgr.GrammarCompiler(ti)
    enum_case = next(c for c in CASES if c["name"] == "enum_record")
    free_case = next(c for c in CASES if c["name"] == "free_string")
    sparse = snapshot_mask(compiler, tok, enum_case, 0, vocab)
    free = snapshot_mask(compiler, tok, free_case, 4, vocab)

    rows = []
    for batch in BATCHES:
        h = torch.randn((batch, hidden), device="cuda", dtype=w.dtype)
        for scenario, cpu in {
            "all_sparse": torch.stack([sparse]*batch),
            "all_free": torch.stack([free]*batch),
            "mixed": torch.stack([sparse if i%2==0 else free for i in range(batch)]),
        }.items():
            m = cpu.to("cuda")
            def dense():
                z = F.linear(h, w)
                xgr.apply_token_bitmask_inplace(z, m, vocab_size=vocab, backend="triton")
                return z
            def row():
                z = bitmask_sparse_logits(h, w, m)
                if z.ndim == 1: z = z.unsqueeze(0)
                return z
            def block():
                return bitmask_block_gemm_logits(h, w, m)
            d = tmed(dense)
            r = tmed(row)
            bg = tmed(block)
            ref = dense()
            got = block()
            torch.cuda.synchronize()
            if not torch.equal(torch.isfinite(ref), torch.isfinite(got)) or not torch.equal(ref.argmax(-1), got.argmax(-1)):
                raise AssertionError((batch, scenario))
            rows.append({
                "batch":batch,"scenario":scenario,
                "dense_ms":d,"row_ms":r,"block_ms":bg,
                "row_speedup":d/r,"block_speedup":d/bg,
                "best":"block" if bg < min(d,r) else ("row" if r<d else "dense"),
            })
    out=ROOT/"results"/"qwen35-batch-kernel-compare"/"results.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print(json.dumps(rows,separators=(",",":")))

if __name__=="__main__":
    main()
