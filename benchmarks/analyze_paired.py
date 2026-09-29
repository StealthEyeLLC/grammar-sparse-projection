import json, math, random, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
src=ROOT/"results"/"qwen35-adaptive-paired"/"results.json"
data=json.loads(src.read_text(encoding="utf-8"))
rng=random.Random(20260928)

def bootstrap_ci(xs, fn=statistics.median, n=10000, alpha=0.05):
    vals=[]
    L=len(xs)
    for _ in range(n):
        sample=[xs[rng.randrange(L)] for _ in range(L)]
        vals.append(fn(sample))
    vals.sort()
    lo=vals[int((alpha/2)*n)]
    hi=vals[min(n-1,int((1-alpha/2)*n))]
    return [lo,hi]

def sign_p_two_sided(wins, losses):
    n=wins+losses
    if n==0: return 1.0
    k=min(wins,losses)
    tail=sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,2*tail)

out=[]
for case in data["results"]:
    gpu=[p["gpu_speedup"] for p in case["pairs"]]
    wall=[p["wall_speedup"] for p in case["pairs"]]
    wins=sum(x>1 for x in gpu); losses=sum(x<1 for x in gpu); ties=len(gpu)-wins-losses
    out.append({
        "case":case["case"],
        "n":len(gpu),
        "gpu_median":statistics.median(gpu),
        "gpu_median_bootstrap_95ci":bootstrap_ci(gpu),
        "gpu_mean":statistics.fmean(gpu),
        "gpu_mean_bootstrap_95ci":bootstrap_ci(gpu,statistics.fmean),
        "wall_median":statistics.median(wall),
        "wall_median_bootstrap_95ci":bootstrap_ci(wall),
        "wins_vs_1":wins,
        "losses_vs_1":losses,
        "ties":ties,
        "two_sided_sign_test_p":sign_p_two_sided(wins,losses),
    })
dst=src.parent/"stats.json"
dst.write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,separators=(",",":")))
