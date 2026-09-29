# GSP Verified Results

Date: 2026-09-28

## Research question

Can an exact formal grammar control execution of the target model's LM head before full-vocabulary projection, rather than computing every logit and masking illegal tokens afterward?

## Answer so far

Yes, for the tested large-vocabulary models and structured states.

The effect is strongly dependent on where the time is spent:
- On Qwen3-0.6B, the LM head is a small fraction of decode time, so end-to-end gains are only a few percent.
- On Qwen3.5-2B, the 248,320 x 2,048 FP16 output head is large enough for GSP to produce repeatable 6-8% end-to-end structured-decoding gains on the RTX 5060.
- Near-unconstrained free-string states provide little useful sparsity and should not be advertised as a GSP win.

## Grammar density

Using XGrammar and the Qwen tokenizer:

| Trace | Median legal tokens | p90 legal tokens | Sparse-friendly observation |
| --- | ---: | ---: | --- |
| Enum record | 13 | 424 | All tested steps highly sparse |
| Tool call | 13 | about 148k on string steps | 18 structural steps sparse; 4 free-string-like steps dense |
| Nested array | 13 | 445 | All tested steps highly sparse |
| Free string | about 148k | about 148k | Mostly near-unconstrained |

## Qwen3.5-2B end-to-end paired benchmark

Thirty AB/BA pairs. Each pair runs the conventional baseline and GSP in alternating order. GPU-event timing includes model decode and output projection. Wall timing also includes grammar work.

Baseline: full LM-head projection plus stock XGrammar GPU bitmask application.

| Trace | Median GPU speedup | 95% bootstrap CI, median | Mean speedup | 95% bootstrap CI, mean | Pair wins | Sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Enum record | 1.075577x | [1.066488, 1.077953] | 1.051949x | [1.021476, 1.074850] | 27/30 | 8.43e-6 |
| Tool call | 1.064208x | [1.061069, 1.069800] | 1.063017x | [1.042463, 1.083158] | 28/30 | 8.68e-7 |
| Nested array | 1.084155x | [1.081001, 1.087846] | 1.093317x | [1.077439, 1.110048] | 28/30 | 8.68e-7 |
| Free string | 1.012795x | [1.008382, 1.015326] | 0.999840x | [0.980981, 1.014933] | 26/30 | 5.95e-5 |

Interpretation: the structured cases show repeatable benefit. The free-string mean interval crosses 1.0 and is a control, not evidence of a reliable mean speedup.

## Batched LM-head benchmark

Qwen3.5-2B head only, FP16.

Highly constrained requests, block-GEMM path versus dense head plus grammar mask:
- batch 2: 4.38x
- batch 4: 4.30x
- batch 8: 4.46x
- batch 16: 4.60x
- batch 32: 2.48x

Observed failure modes:
- Near-free batches can favor dense execution.
- A 50/50 constrained/free batch at batch 32 can activate enough shared vocabulary tiles that dense wins.
- The dispatcher therefore falls back to dense for near-unconstrained batched states and sufficiently large mixed-density batches.

## Exactness evidence

Tests verify:
- identical finite/logically legal vocabulary support;
- identical greedy top-1 token on all tested states;
- row-sparse correctness over legal-set sizes 1 through 120,000;
- block-GEMM correctness at batches 2, 4, 8, 16, and 32.

FP16 logits can differ by small amounts due to different reduction order. The method is exact in constrained support; numerical floating-point evaluation is not claimed to be bitwise identical.

## Prior-art boundary

Not novel by itself:
- grammar-mask generation;
- the observation that post-hoc masking wastes LM-head work;
- batch-1 grammar-aware row or block filtering;
- generic vocabulary pruning for speculative drafters;
- dense LM-head/sampling fusion.

Our current systems contribution:
- direct packed grammar-bitmask consumption;
- latency-oriented row-sparse target-head projection;
- batched block-GEMM target-head projection across heterogeneous grammar masks;
- routing among row-sparse, block-sparse, and conventional dense execution;
- controlled end-to-end target-model evidence on a 248K-vocabulary model.
