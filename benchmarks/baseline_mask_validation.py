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
from gsp.triton_runtime import dense_masked_logits

MODEL="Qwen/Qwen3.5-2B"

def med(fn,warm=25,reps=120):
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
    ti=xgr.TokenizerInfo.from_huggingface(tok,vocab_size=vocab); comp=xgr.GrammarCompiler(ti)
    enum=next(c for c in CASES if c["name"]=="enum_record")
    free=next(c for c in CASES if c["name"]=="free_string")
    masks={"sparse":snapshot_mask(comp,tok,enum,0,vocab),"free":snapshot_mask(comp,tok,free,4,vocab)}
    rows=[]
    for batch in (1,8):
        h=torch.randn((batch,hidden),device="cuda",dtype=w.dtype)
        for name,mc in masks.items():
            mg=torch.stack([mc]*batch).to("cuda")
            def stock():
                z=F.linear(h,w)
                xgr.apply_token_bitmask_inplace(z,mg,vocab_size=vocab,backend="triton")
                return z
            def ours():
                return dense_masked_logits(h,w,mg)
            sm=med(stock); om=med(ours)
            rows.append({"batch":batch,"mask":name,"xgrammar_ms":sm,"our_dense_ms":om,"our_over_stock":om/sm})
    out=ROOT/"results"/"baseline-mask-validation"/"results.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,separators=(",",":")))
if __name__=="__main__": main()
