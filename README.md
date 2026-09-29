# Grammar-Conditioned Sparse Projection (GSP)

GSP is a research prototype for exact structured decoding that uses a grammar's packed legal-token mask to control execution of the language-model output projection itself.

Conventional constrained decoding typically computes the full vocabulary logits and masks illegal tokens afterward. GSP instead skips language-model-head work for grammar-invalid vocabulary regions before or during the projection.

## Scope and non-claims

GSP optimizes the output projection. It does not prune the transformer KV cache.

GSP does not claim to be the first observation that grammar constraints can skip LM-head work. Murali's 2025 CUDA study implemented batch-1 block and thread filtering for constrained decoding and explicitly identified batching and hybrid routing as future work. GSP builds on that systems direction.

The contribution evaluated here is a complete exact execution path built around:

1. Direct packed-bitmask row-sparse projection for latency-oriented batch-1 decoding.
2. Grammar-aware block GEMM for batched decoding, using block matrix multiplication across requests while skipping vocabulary tiles with no legal outputs.
3. A batch/density dispatcher that falls back to the conventional dense LM head plus stock XGrammar mask when sparse execution is not profitable.
4. End-to-end evaluation on modern large-vocabulary target models, including Qwen3.5-2B with a 248,320-token output vocabulary.

## Exactness

For a hidden state h, output weight W, vocabulary V, and grammar-legal set A:

    dense constrained decoding: softmax(mask(W h))
    GSP:                       softmax(W_A h)

Tokens outside A have probability zero under the constrained distribution, so their logits do not need to be computed.

The prototype emits -inf for illegal vocabulary entries when a full vocabulary-shaped output is required. Tests verify identical legal support and identical greedy top-1 selection against dense projection plus grammar masking. Floating-point logits can differ slightly because the kernels use a different reduction order.

## Current measured result

Primary model:
- Qwen/Qwen3.5-2B
- 248,320-token LM output
- hidden dimension 2,048
- FP16 inference
- NVIDIA GeForce RTX 5060 Laptop GPU
- PyTorch 2.13.0 + CUDA 13.0
- Transformers 5.17.0
- XGrammar 0.2.8
- Triton 3.7.1 for Windows

Thirty paired AB/BA end-to-end decode runs per workload, compared against full LM-head projection plus stock XGrammar GPU masking:

| Workload | Median speedup | Bootstrap 95% CI for median | Wins / 30 |
| --- | ---: | ---: | ---: |
| Enum-heavy JSON | 1.0756x | [1.0665x, 1.0780x] | 27 |
| Tool-call JSON | 1.0642x | [1.0611x, 1.0698x] | 28 |
| Nested-array JSON | 1.0842x | [1.0810x, 1.0878x] | 28 |
| Free-string control | 1.0128x median | [1.0084x, 1.0153x] median | 26 |

The free-string control is not treated as a reliable throughput improvement: its mean bootstrap interval crosses 1.0.

At the LM-head level, the batched block-GEMM path preserves large gains for highly constrained requests:
- batch 2: about 4.38x
- batch 4: about 4.30x
- batch 8: about 4.46x
- batch 16: about 4.60x
- batch 32: about 2.48x

The dispatcher uses dense projection for batched near-unconstrained states and for large mixed-density batches where shared block activation makes sparse execution unattractive.

## Reproduction

The reference environment and portable setup are documented in REPRODUCIBILITY.md. A project-local vendor/ directory may be used on Windows but is intentionally not committed.

Use:

    tools\python_triton.cmd tests\test_triton_sparse.py
    tools\python_triton.cmd tests\test_triton_block_gemm.py
    tools\python_triton.cmd tests\test_adaptive_dispatch.py
    tools\python_triton.cmd benchmarks\qwen35_adaptive_paired.py
    python benchmarks\analyze_paired.py

Primary result files:
- results/qwen35-adaptive-paired/results.json
- results/qwen35-adaptive-paired/stats.json
- results/qwen35-batch-kernel-compare/results.json
- results/grammar-density/summary.json

## Publication target

The current target is MLSys 2027. The research submission deadline is October 30, 2026.

See paper/draft.md for the working paper and paper/references.bib for references.

## Replication wanted

**Independent tests are wanted, including negative results.** If you have a different GPU, model, serving stack, or batching regime, see REPLICATION.md and open a Replication report issue with the raw evidence.

High-value targets include H100/H200, B100/B200, A100, RTX 4090/5090, other large-vocabulary models, and integration experiments with XGrammar, vLLM, or SGLang.

## Citation

GitHub renders the repository's CITATION.cff into copyable citation formats. Until a DOI or archival paper identifier is assigned, cite version 0.1.0 and include the repository URL plus the exact commit SHA used.

## License

- GSP software: **Apache-2.0** (LICENSE)
- GSP paper/original figures: **CC BY 4.0** (paper/LICENSE.md)
- Third-party conference/LaTeX material: retains upstream terms (THIRD_PARTY.md)

See LICENSING.md for scope.

## Preservation

The public repository is independently archived by Software Heritage. The first successful full snapshot of the public GSP repository is:

swh:1:snp:f7ea97e93f115b79ec140930838c59bc8eed9f2a

Software Heritage origin: https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/StealthEyeLLC/grammar-sparse-projection
