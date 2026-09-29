import json
import importlib.metadata
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
import xgrammar as xgr


MODEL = "Qwen/Qwen3-4B"

CASES = [
    {
        "name": "enum_record",
        "schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["create", "update", "delete"]},
                "status": {"type": "string", "enum": ["ready", "blocked", "done"]},
                "priority": {"type": "integer", "minimum": 0, "maximum": 5},
            },
            "required": ["kind", "status", "priority"],
            "additionalProperties": False,
        },
        "sample": '{"kind":"create","status":"ready","priority":3}',
    },
    {
        "name": "tool_call",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"const": "search"},
                "arguments": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 80},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                        "safe": {"type": "boolean"},
                    },
                    "required": ["query", "top_k", "safe"],
                    "additionalProperties": False,
                },
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        },
        "sample": '{"name":"search","arguments":{"query":"grammar sparse projection","top_k":5,"safe":true}}',
    },
    {
        "name": "nested_array",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "minimum": 0, "maximum": 9999},
                            "label": {"type": "string", "enum": ["alpha", "beta", "gamma"]},
                        },
                        "required": ["id", "label"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "sample": '{"items":[{"id":7,"label":"alpha"},{"id":8,"label":"beta"}]}',
    },
    {
        "name": "free_string",
        "schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "maxLength": 120}
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        "sample": '{"message":"Structured decoding should avoid work that the grammar already proved impossible."}',
    },
]


def allowed_ids_from_bitmask(bitmask: torch.Tensor, vocab_size: int):
    raw = bitmask[0].numpy().view(np.uint32).view(np.uint8)
    bits = np.unpackbits(raw, bitorder="little")[:vocab_size]
    return np.flatnonzero(bits).astype(np.int64)


def pct(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, round((len(ys) - 1) * p)))
    return ys[i]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    vocab_size = int(config.vocab_size)
    ti = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(ti)

    all_summaries = []
    traces = {}

    for case in CASES:
        compiled = compiler.compile_json_schema(json.dumps(case["schema"]))
        matcher = xgr.GrammarMatcher(compiled)
        bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        token_ids = tokenizer.encode(case["sample"], add_special_tokens=False)
        rows = []

        for step, token_id in enumerate(token_ids):
            t0 = time.perf_counter_ns()
            need_apply = matcher.fill_next_token_bitmask(bitmask)
            mask_us = (time.perf_counter_ns() - t0) / 1000.0
            if need_apply:
                allowed = allowed_ids_from_bitmask(bitmask, vocab_size)
                k = int(allowed.size)
                token_legal = bool(np.searchsorted(allowed, token_id) < allowed.size and
                                   allowed[np.searchsorted(allowed, token_id)] == token_id)
            else:
                allowed = None
                k = vocab_size
                token_legal = True

            accepted = matcher.accept_token(int(token_id))
            if not accepted or not token_legal:
                raise RuntimeError({
                    "case": case["name"],
                    "step": step,
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id]),
                    "token_legal": token_legal,
                    "accepted": accepted,
                })

            rows.append({
                "step": step,
                "token_id": int(token_id),
                "token_text": tokenizer.decode([token_id]),
                "need_apply": bool(need_apply),
                "allowed_k": k,
                "density": k / vocab_size,
                "mask_us": mask_us,
            })

        ks = [r["allowed_k"] for r in rows]
        densities = [r["density"] for r in rows]
        mask_times = [r["mask_us"] for r in rows]
        summary = {
            "case": case["name"],
            "tokens": len(rows),
            "vocab_size": vocab_size,
            "allowed_k_min": min(ks),
            "allowed_k_p50": pct(ks, 0.50),
            "allowed_k_p90": pct(ks, 0.90),
            "allowed_k_max": max(ks),
            "density_p50": pct(densities, 0.50),
            "density_p90": pct(densities, 0.90),
            "fraction_k_le_32768": sum(k <= 32768 for k in ks) / len(ks),
            "fraction_k_le_45056": sum(k <= 45056 for k in ks) / len(ks),
            "fraction_k_le_49152": sum(k <= 49152 for k in ks) / len(ks),
            "mask_us_median": statistics.median(mask_times),
            "mask_us_p90": pct(mask_times, 0.90),
        }
        all_summaries.append(summary)
        traces[case["name"]] = rows

    out = ROOT / "results" / "grammar-density"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    (out / "traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(json.dumps({
        "model": MODEL,
        "xgrammar": importlib.metadata.version("xgrammar"),
        "vocab_size": vocab_size,
        "summaries": all_summaries,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
