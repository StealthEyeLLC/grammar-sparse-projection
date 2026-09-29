import json, statistics, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"vendor"))
import xgrammar as xgr
from benchmarks.qwen35_batch_head import snapshot_mask
from benchmarks.grammar_density_trace import CASES
from gsp.triton_runtime import adaptive_grammar_logits, full_word_fraction_cpu, dense_masked_logits

MODEL="Qwen/Qwen3.5-2B"
BATCHES=[1,2,4,8,16,32]

def med(fn,warm=20,reps=80):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); xs=[]
    for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); xs.append(a.elapsed_time(b))
    return statistics.median(xs)

def main():
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,low_cpu_mem_usage=True).to("cuda").eval()
    w=model.lm_head.weight; vocab,hidden=w.shape
    comp=xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok,vocab_size=vocab))
    enum=next(c for c in CASES if c["name"]=="enum_record")
    free=next(c for c in CASES if c["name"]=="free_string")
    sm=snapshot_mask(comp,tok,enum,0,vocab)
    fm=snapshot_mask(comp,tok,free,4,vocab)
    rows=[]
    for batch in BATCHES:
        h=torch.randn((batch,hidden),device="cuda",dtype=w.dtype)
        scenarios={
            "all_sparse":torch.stack([sm]*batch),
            "all_free":torch.stack([fm]*batch),
            "mixed":torch.stack([sm if i%2==0 else fm for i in range(batch)]),
        }
        for name,mc in scenarios.items():
            mg=mc.to("cuda"); fractions=full_word_fraction_cpu(mc)
            def base(): return dense_masked_logits(h,w,mg)
            route=[None]
            def adapt():
                z,r=adaptive_grammar_logits(h,w,mg,full_word_fraction=fractions); route[0]=r; return z
            bm=med(base); am=med(adapt)
            ref=base(); got=adapt()
            if ref.ndim==1: ref=ref.unsqueeze(0)
            if got.ndim==1: got=got.unsqueeze(0)
            torch.cuda.synchronize()
            if not torch.equal(torch.isfinite(ref),torch.isfinite(got)) or not torch.equal(ref.argmax(-1),got.argmax(-1)):
                raise AssertionError((batch,name,route[0]))
            rows.append({"batch":batch,"scenario":name,"route":route[0],"baseline_ms":bm,"adaptive_ms":am,"speedup":bm/am})
    out=ROOT/"results"/"qwen35-adaptive-batch-head"/"results.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,separators=(",",":")))
if __name__=="__main__": main()
