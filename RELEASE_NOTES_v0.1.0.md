# GSP v0.1.0

First public research release of Grammar-Conditioned Sparse Projection.

## Included

- direct packed-bitmask row-sparse projection;
- grammar-aware batched block-GEMM projection;
- density/batch adaptive dispatch with dense fallback;
- exactness tests;
- raw benchmark evidence;
- paired end-to-end Qwen3.5-2B evaluation;
- reproduction instructions;
- manuscript and bibliography;
- Apache-2.0 software license;
- CC BY 4.0 paper license;
- machine-readable citation metadata.

## Headline measured result

On the reference RTX 5060 Laptop GPU with Qwen3.5-2B, 30 paired AB/BA end-to-end runs showed median GPU speedups of approximately:

- 1.0756x for enum-heavy JSON;
- 1.0642x for tool-call JSON;
- 1.0842x for nested-array JSON.

The free-string control is not claimed as a reliable mean speedup because its mean bootstrap interval crosses 1.0.

## Replication request

Please test GSP on other GPUs/models/stacks and open a Replication report issue. Negative results are welcome.

## Reproducibility anchors

Original immutable experimental checkpoint:
f9a9976a9126cf380c99661c4ee7f53f37099abe

Publication-preparation checkpoint:
863bb004f47e534fcf843e9cd0b5ff4c6f20ed99
