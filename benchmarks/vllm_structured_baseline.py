import json
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

ROOT = Path("/mnt/x/Research/grammar-sparse-projection")
MODEL = "Qwen/Qwen3-0.6B"

CASES = [
    ("enum_record", {
        "type":"object","properties":{
            "kind":{"type":"string","enum":["create","update","delete"]},
            "status":{"type":"string","enum":["ready","blocked","done"]},
            "priority":{"type":"integer","minimum":0,"maximum":5}},
        "required":["kind","status","priority"],"additionalProperties":False}),
    ("nested_array", {
        "type":"object","properties":{"items":{
            "type":"array","minItems":1,"maxItems":3,
            "items":{"type":"object","properties":{
                "id":{"type":"integer","minimum":0,"maximum":9999},
                "label":{"type":"string","enum":["alpha","beta","gamma"]}},
                "required":["id","label"],"additionalProperties":False}}},
        "required":["items"],"additionalProperties":False}),
    ("free_string", {
        "type":"object","properties":{"message":{"type":"string","maxLength":120}},
        "required":["message"],"additionalProperties":False}),
]

def main():
    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=0.65,
        enforce_eager=False,
        trust_remote_code=False,
    )
    records = []
    for name, schema in CASES:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=64,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        prompt = "Return only one compact JSON value matching the required schema."
        llm.generate([prompt], params, use_tqdm=False)
        trials = []
        for i in range(6):
            t0 = time.perf_counter()
            out = llm.generate([prompt], params, use_tqdm=False)[0]
            wall = time.perf_counter() - t0
            choice = out.outputs[0]
            tids = list(choice.token_ids)
            m = getattr(out, "metrics", None)
            row = {"trial":i,"wall_s":wall,"tokens":len(tids),"text":choice.text}
            if m is not None:
                for key in ["arrival_time","first_token_time","finished_time",
                            "first_scheduled_time","last_token_time"]:
                    val = getattr(m, key, None)
                    if val is not None:
                        row[key] = float(val)
                ft = getattr(m, "first_token_time", None)
                fin = getattr(m, "finished_time", None)
                if ft is not None and fin is not None:
                    dt = float(fin-ft)
                    row["decode_s_after_first"] = dt
                    if len(tids) > 1 and dt > 0:
                        row["decode_tokens_per_s"] = (len(tids)-1)/dt
            trials.append(row)
        tps = [x["decode_tokens_per_s"] for x in trials if "decode_tokens_per_s" in x]
        records.append({
            "case":name,
            "wall_median_s":statistics.median(x["wall_s"] for x in trials),
            "decode_tps_median":statistics.median(tps) if tps else None,
            "tokens_median":statistics.median(x["tokens"] for x in trials),
            "trials":trials,
        })
    payload={"model":MODEL,"vllm_version":__import__("vllm").__version__,
             "mode":"baseline","cases":records}
    outdir=ROOT/"results"/"vllm-baseline"
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"model":MODEL,"vllm_version":payload["vllm_version"],
        "cases":[{"case":r["case"],"wall_median_s":r["wall_median_s"],
                  "decode_tps_median":r["decode_tps_median"],
                  "tokens_median":r["tokens_median"],
                  "example":r["trials"][0]["text"]} for r in records]},
        separators=(",",":")))

if __name__ == "__main__":
    main()
