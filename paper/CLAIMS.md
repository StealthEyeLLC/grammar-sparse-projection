# Claim ledger

This file separates claims currently supported by the committed experiments from future-work claims.

## Supported

- GSP operates on the target-model output projection, not the KV cache.
- Packed grammar masks can be consumed directly by a Triton output-projection kernel without CPU legal-ID expansion.
- The tested row-sparse kernel preserves the same legal vocabulary support and greedy top-1 result as dense projection plus XGrammar masking over the tested legal-set range.
- The tested block-GEMM kernel preserves the same support and greedy top-1 result at batch 2, 4, 8, 16, and 32.
- Qwen3.5-2B exposes a 248,320-token tied LM output and hidden dimension 2,048.
- On the tested RTX 5060 system, thirty paired AB/BA runs show median end-to-end GPU decode speedups of 1.0756x, 1.0642x, and 1.0842x for enum, tool-call, and nested-array traces.
- Bootstrap 95% confidence intervals for the median paired ratio are entirely above 1.0 for those three structured traces.
- Highly constrained Qwen3.5-2B head-only block-GEMM runs show about 4.3-4.6x speedup at batches 2-16 and about 2.48x at batch 32.
- Near-unconstrained states can erase the sparse advantage; dense fallback remains necessary.
- Batch-1 constrained LM-head filtering has prior art and is not claimed as novel.

## Not supported / do not claim

- KV-cache pruning from grammar legality.
- Zero-compute generation.
- Universal speedup across GPUs, models, schemas, or batch sizes.
- Bitwise equality of FP16 logits.
- Reliable mean speedup for the free-string control.
- First-ever grammar-aware sparse LM-head projection.
- Production-scale datacenter throughput improvement.
- A completed fused grammar-aware sampler.
- Peer-reviewed acceptance.
