# GSP: Grammar-Conditioned Sparse Output Projection for Exact Structured LLM Decoding

Anonymous submission draft for MLSys 2027.

## Abstract

Structured generation engines such as XGrammar make formal grammar constraints inexpensive by computing the set of legal next tokens before sampling. However, conventional inference still performs a dense language-model output projection over the full vocabulary and only then applies the grammar mask. This ordering is increasingly expensive for models with vocabularies in the hundreds of thousands of tokens.

We present Grammar-Conditioned Sparse Projection (GSP), an exact output-projection runtime that consumes the grammar's packed legal-token bitmask before full-vocabulary LM-head computation. GSP uses two GPU execution paths. A row-sparse kernel targets latency-oriented decoding by avoiding weight loads and multiply work for illegal vocabulary rows. A batched block-GEMM kernel preserves matrix-multiplication reuse across requests while skipping vocabulary tiles that contain no legal outputs. A lightweight dispatcher selects sparse or conventional dense projection based on batch size and grammar density. GSP changes neither the transformer nor its KV cache and requires no model training.

On Qwen3.5-2B, whose tied output vocabulary contains 248,320 tokens, thirty paired AB/BA runs on an NVIDIA RTX 5060 Laptop GPU show median end-to-end decode speedups of 1.0756x for enum-heavy JSON, 1.0642x for tool-call JSON, and 1.0842x for nested-array JSON relative to full LM-head projection followed by stock XGrammar GPU masking. Bootstrap 95% confidence intervals for the median remain above 1.0 for all three structured workloads. At the LM-head level, the batched block path is 4.30-4.60x faster than dense projection plus masking for highly constrained batches of 2-16 requests and 2.48x faster at batch 32. A free-string control shows no reliable mean end-to-end benefit, demonstrating that GSP's gain comes from grammar-induced sparsity rather than a generally faster output head.

The core idea of filtering constrained output projection has prior batch-1 implementations. Our contribution is an exact packed-bitmask execution design that extends this idea to batched block GEMM, combines latency and serving kernels under a hardware-adaptive dispatcher, and evaluates the resulting target-model system end to end on a modern large-vocabulary model.

## 1. Introduction

Formal constrained decoding has become a core inference primitive for tool calling, JSON output, code generation, and agentic systems. A grammar engine determines which tokens are legal from the current parser state, and the inference runtime prevents the model from sampling outside that set. Recent systems have made the grammar side of this loop extremely efficient. XGrammar separates context-independent and context-dependent token checks and overlaps grammar processing with model execution, while XGrammar 2 extends the design to dynamic agentic grammars with JIT compilation and cross-grammar caching.

The output projection remains ordered differently. Let the final hidden state at step t be h_t in R^d, the LM-head matrix be W in R^(V x d), and the grammar-legal token set be A_t, a subset of the V-token vocabulary. Conventional constrained decoding computes

    z_t = W h_t

for all V rows and subsequently sets z_i = -infinity for every i not in A_t. The grammar therefore proves that most outputs are unusable before the runtime pays to compute them.

This mismatch matters as vocabularies grow. Qwen3.5-2B, for example, has a hidden width of 2,048 and a padded tied vocabulary of 248,320 tokens. In FP16, the output matrix alone is approximately 0.95 GiB. At decode batch sizes where the LM head is memory sensitive, reading and multiplying that matrix can consume a meaningful fraction of token latency.

The obvious alternative is

    z_(A_t) = W_(A_t) h_t,

which is mathematically sufficient because tokens outside A_t receive zero probability under the constrained distribution. Unfortunately, a naive gather is not enough for a production runtime. Per-step token-ID materialization can require CPU expansion of a packed grammar mask, synchronization, host-to-device transfer, and irregular GPU gathers. Batching is harder: each request can occupy a different grammar state and therefore a different legal vocabulary, destroying the simple shared submatrix that makes dense GEMM efficient.

GSP addresses these two systems problems directly. It consumes the same packed int32 token bitmask already produced by the grammar engine and never expands that mask into a CPU list of token IDs. At batch 1, each output row checks its grammar bit before loading the corresponding LM-head weights. At batch sizes greater than one, GSP groups requests and vocabulary rows into tiles, computes a block GEMM only for active request-vocabulary tiles, and writes negative infinity for illegal elements. When the packed mask indicates little useful sparsity, GSP falls back to the standard dense head and stock XGrammar mask.

Our evaluation answers four questions:

1. Are real structured grammars sparse enough to make pre-projection filtering useful?
2. Can the packed grammar mask be consumed directly without changing constrained-decoding semantics?
3. Does a sparse target-model LM head improve end-to-end decode latency, rather than only an isolated microbenchmark?
4. Can the approach survive batched serving, where dense GEMM receives much better hardware utilization?

Our results are encouraging but bounded. Highly structured grammar states often expose only tens or hundreds of legal tokens from a vocabulary of hundreds of thousands. On Qwen3.5-2B, this produces repeatable 6-8% median end-to-end gains for three structured traces. The free-string control does not show a reliable mean improvement. For batched structured heads, the block-GEMM kernel preserves 2.48-4.60x kernel-level speedups through batch 32. These results suggest that grammar state can be treated not only as a sampling constraint but also as an execution schedule for the LM head.

### Contributions

This paper makes four contributions.

- We define an exact grammar-conditioned output-projection interface that consumes a packed legal-token bitmask directly, avoiding per-step CPU token-ID expansion.
- We implement a row-sparse Triton kernel for latency-oriented constrained decoding and a batched block-GEMM kernel that preserves cross-request matrix-multiplication reuse while skipping inactive vocabulary tiles.
- We introduce a simple hardware-adaptive dispatcher that selects row-sparse, block-sparse, or conventional dense projection based on batch size and grammar density.
- We provide controlled target-model measurements on Qwen3.5-2B, including thirty paired end-to-end AB/BA trials with bootstrap intervals and sign tests, plus batch-scaling and free-string controls.

## 2. Background and Motivation

### 2.1 Structured decoding

A structured decoder maintains a parser or finite-state representation of the accepted output language. At each generation step, it maps the current grammar state to a set of valid model tokens. XGrammar and related systems focus on making this mapping efficient enough that grammar processing adds nearly zero overhead to model serving.

For a vocabulary of size V, a convenient runtime representation is a packed bitmask with one bit per token. A set bit marks a legal token. XGrammar exposes such a bitmask and provides GPU kernels that apply it to an already-computed logits tensor.

This representation is important for GSP. A V-bit mask costs V/8 bytes, whereas a list of all legal token IDs can approach O(V) 32-bit integers in permissive states. More importantly, expanding the packed mask into an ID list introduces a variable-length CPU operation and a synchronization boundary. GSP keeps the packed representation intact.

### 2.2 The large-vocabulary output head

The LM head computes a matrix-vector or matrix-matrix product between the final hidden state and the vocabulary embedding/output matrix. Its cost scales linearly with vocabulary size. Large-vocabulary models make this component increasingly visible, particularly at small decode batches where weight movement is difficult to amortize.

Several speculative-decoding systems attack a related output-head bottleneck in a draft model. FR-Spec, VocabTrim, and SpecVocab reduce or predict the draft vocabulary to make speculative proposal generation cheaper. Their setting differs from ours: they optimize a speculative drafter and rely on verification by the target model. GSP instead uses a formal grammar to prove that omitted target-model tokens have zero probability under the constrained distribution.

FlashSampling attacks a complementary bottleneck. It fuses dense LM-head matrix multiplication with exact categorical sampling so full logits need not be materialized in HBM. GSP instead removes matrix-multiplication work when formal constraints make vocabulary regions impossible. The two ideas are conceptually composable.

### 2.3 Prior constrained LM-head filtering

The idea of exploiting grammar constraints before full output projection is not new. Murali's 2025 CUDA study explicitly observes that standard constrained decoding computes full-vocabulary logits before masking, then evaluates weight slicing, block-level filtering, and thread-level filtering in Triton. The study also identifies batching as a limitation: different requests occupy different grammar states and therefore expose different allowed-token sets, reducing shared parallelism. Its stated future work includes batching the CUDA kernel and developing hybrid approaches.

We treat that study as direct prior art. GSP is not a claim of first discovery of constrained LM-head sparsity. Our systems question is what is required to make the idea work as an exact target-model execution path under heterogeneous batching, and whether that path yields repeatable end-to-end benefit on a modern large-vocabulary model.

## 3. Design

### 3.1 Exact constrained projection

Let A_t be the legal token set returned by the grammar at step t. Conventional masking defines

    p(i | h_t, A_t) =
        exp(w_i^T h_t) / sum_(j in A_t) exp(w_j^T h_t), if i in A_t
        0,                                             otherwise.

Therefore logits for i not in A_t are irrelevant to the constrained distribution. GSP changes execution but not this definition.

When a full vocabulary-shaped result is required by downstream code, GSP writes the computed legal logits into their vocabulary positions and emits negative infinity for illegal positions. This makes greedy selection and ordinary masked softmax semantics equivalent to dense projection plus masking.

Floating-point reductions are not required to be bitwise identical. Different Triton and dense-library reduction orders can produce small FP16 differences. Our correctness criterion is identical legal support plus identical greedy top-1 on all evaluated states; Section 5 reports the observed numerical differences.

### 3.2 Direct packed-bitmask row kernel

The latency kernel uses one-dimensional vocabulary tiles. For each candidate output row it:

1. loads the packed grammar word containing that token's legality bit;
2. tests the bit before reading the LM-head row;
3. loads h_t and W_i only for legal rows;
4. accumulates the dot product; and
5. writes the result for legal rows or negative infinity otherwise.

The critical design choice is step 1: the kernel consumes the grammar bitmask directly. An earlier prototype expanded the CPU mask with NumPy, materialized legal token IDs, copied those IDs to the GPU, gathered LM-head rows, and called a smaller dense projection. That design produced large isolated speedups but could lose the gain end to end because mask expansion and row gathering became part of the critical path. The direct kernel removes that conversion.

### 3.3 Batched block-GEMM kernel

Applying the row kernel independently to each request works well at batch 1 but loses to dense GEMM as batch size increases. Dense matrix multiplication reuses each LM-head tile across multiple hidden states, while independent sparse row kernels reread weights for each request.

The block-GEMM kernel restores this reuse. Its two-dimensional launch grid partitions the batch dimension and vocabulary dimension. For each B_m x B_v output tile, the kernel loads the corresponding packed grammar bits for every request in the request block. If no request-token pair in the tile is legal, the kernel skips the LM-head computation. Otherwise it performs a block matrix multiplication

    H_block W_block^T

using Triton dot operations and then masks illegal elements inside the active tile.

This design makes sparsity coarser than the row kernel: one permissive request can activate a vocabulary tile for other requests in the same block. The tradeoff is intentional. It preserves tensor-core-style matrix-multiplication reuse while still skipping fully inactive vocabulary tiles.

### 3.4 Dispatcher

No single kernel wins across all batch sizes and grammar densities.

GSP currently uses the following policy:

- batch 1: row-sparse kernel;
- batch greater than 1: block-GEMM kernel when grammar sparsity is useful;
- batched states in which all requests are nearly unconstrained: conventional dense LM head plus stock XGrammar GPU masking;
- sufficiently large mixed-density batches: dense fallback when permissive requests activate too many shared vocabulary tiles.

The prototype estimates density directly from the packed CPU bitmask already produced by XGrammar. A 32-bit word equal to all ones means all 32 token IDs in that word are legal. Counting these words provides a cheap estimate without unpacking the mask.

The current thresholds are empirical, not universal. They are exposed as calibration parameters. A production runtime should calibrate them per GPU, dtype, hidden width, vocabulary size, and serving batch regime.

### 3.5 What GSP does not change

GSP does not modify model weights, attention, transformer layers, or KV-cache semantics. It requires no training. Grammar state affects only the output projection and downstream constrained sampling. This boundary is important: next-token grammar legality does not imply that past sequence positions in the attention KV cache are invalid.


## 4. Experimental Methodology

### 4.1 Hardware and software

Unless otherwise noted, experiments run on a Windows 11 laptop with an NVIDIA GeForce RTX 5060 Laptop GPU with approximately 8 GiB of VRAM. The primary software stack is PyTorch 2.13.0 with CUDA 13.0, Transformers 5.17.0, XGrammar 0.2.8, and Triton 3.7.1 for Windows. Triton is installed project-locally to avoid modifying the model environment.

The primary target model is Qwen/Qwen3.5-2B. Its language model has hidden width 2,048 and a padded tied vocabulary/output size of 248,320 tokens. We use FP16 weights for the experiments. We also use Qwen3-0.6B and synthetic head shapes during mechanism falsification and crossover testing; the primary paper claims use Qwen3.5-2B.

### 4.2 Workloads

We use four JSON-schema traces designed to expose different grammar-density regimes.

**Enum record.** A compact object containing enumerated string fields and a bounded integer. This represents a highly constrained structured response.

**Tool call.** A function-style object with a fixed tool name and an arguments object containing a free-text query, bounded integer, and boolean. This workload deliberately alternates between strongly constrained syntax and a permissive string field.

**Nested array.** An object containing an array of small objects with bounded integers and enum labels. This creates repeated structured transitions.

**Free-string control.** An object containing one bounded free-text string. Most content positions admit nearly the full vocabulary. This is our control for the case where a grammar provides syntactic validity but little projection sparsity.

For grammar-density analysis, we teacher-force a valid sample through XGrammar and record the legal-token mask at every output token. The median legal set is 13 tokens for the enum, tool-call, and nested-array traces. The free-string content positions expose roughly 148 thousand legal tokens under the earlier 151,936-token tokenizer analysis, making them deliberately unfavorable to sparse projection.

### 4.3 Baseline

The primary baseline is the conventional constrained-decoding ordering:

1. compute the full target-model LM head with PyTorch linear projection;
2. apply the packed grammar bitmask with XGrammar's Triton GPU masking kernel; and
3. perform greedy selection.

We explicitly validated the fallback implementation against stock XGrammar masking. An earlier custom dense-mask kernel was slower for batch-1 permissive masks and was removed from the primary comparison. All final end-to-end results therefore use the stock XGrammar masking path as the dense reference and fallback.

### 4.4 End-to-end paired protocol

For each workload we run thirty paired trials. Trial order alternates AB and BA so that GSP and the dense baseline take turns running first. Each sequence is teacher-forced through the same valid structured output so both systems execute identical transformer tokens and grammar states. This isolates output-projection execution from generation divergence.

GPU-event timing includes the decode forward path, packed mask transfer, output projection/masking, and greedy selection. Wall-clock timing additionally captures host grammar work. The first prompt-processing step is excluded from the per-token decode statistic because its sequence length differs from steady-state one-token decoding.

We report the median ratio of baseline time to GSP time over the thirty pairs. We also compute deterministic 10,000-resample bootstrap confidence intervals for the median and mean paired ratios, plus an exact two-sided sign test for the number of pairs faster than 1.0.

### 4.5 Kernel-level batch protocol

To separate output-head scaling from transformer cost, we benchmark the real Qwen3.5-2B LM-head weights on hidden states at batch sizes 2, 4, 8, 16, and 32. We evaluate:

- all-sparse batches, where every request uses a highly constrained grammar mask;
- all-free batches, where every request uses a near-unconstrained string mask; and
- mixed batches, alternating constrained and permissive masks.

We compare dense projection plus XGrammar masking, the request-independent row kernel, and the batched block-GEMM kernel. Correctness checks compare legal support and greedy top-1 against the dense baseline.

## 5. Results

### 5.1 Real grammars expose substantial output sparsity

The first prerequisite for GSP is empirical: structured grammar states must remove enough vocabulary to matter.

For the enum record, the median legal set contains only 13 tokens and the 90th percentile contains 424. For the nested-array trace the median is again 13 and the 90th percentile is 445. The tool-call trace has a median of 13 but includes four free-string-like steps where approximately the full tokenizer vocabulary is legal. This creates a natural within-sequence control. The free-string workload is dominated by these permissive states.

These measurements show that grammar density is highly bimodal in realistic structured output: syntax, keys, enums, delimiters, booleans, and bounded numeric transitions can be extremely sparse, while arbitrary string content can be almost dense. This motivates a dispatcher rather than a single sparse kernel.

### 5.2 Direct bitmask projection eliminates the CPU-ID bottleneck

Our first sparse prototype expanded the packed CPU grammar mask into a legal-token ID list, transferred those IDs to the GPU, gathered legal LM-head rows, and applied a smaller dense projection. On synthetic Qwen-like heads this approach showed a large raw crossover region, but end-to-end gains were fragile because ID materialization and irregular gathers sat on the critical path.

The final row kernel reads the packed bitmask directly. On the real Qwen3.5-2B head, strongly constrained states reduce output-head latency from approximately 2.8-3.0 ms for dense projection plus masking to approximately 0.5-0.7 ms, depending on state and run. Across representative structured states this is roughly a 4-6x head-level reduction. Near-unconstrained string states approach dense performance, as expected.

### 5.3 End-to-end structured decoding improves by 6-8 percent

Table 1 reports the primary thirty-pair Qwen3.5-2B result.

| Workload | Median speedup | Bootstrap 95% CI, median | Mean speedup | Bootstrap 95% CI, mean | Wins / 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Enum record | 1.0756x | [1.0665, 1.0780] | 1.0519x | [1.0215, 1.0748] | 27 |
| Tool call | 1.0642x | [1.0611, 1.0698] | 1.0630x | [1.0425, 1.0832] | 28 |
| Nested array | 1.0842x | [1.0810, 1.0878] | 1.0933x | [1.0774, 1.1100] | 28 |
| Free string | 1.0128x | [1.0084, 1.0153] | 0.9998x | [0.9810, 1.0149] | 26 |

The three structured workloads show consistent positive effects. Exact two-sided sign-test p-values are 8.43e-6 for the enum trace and 8.68e-7 for both tool-call and nested-array traces.

The free-string result is intentionally different. Its median ratio is slightly above 1.0, but its mean bootstrap interval crosses 1.0. We therefore do not interpret it as evidence of a reliable average GSP speedup. Instead it confirms that the material benefit tracks grammar-induced sparsity.

The absolute gain is bounded by Amdahl's law. GSP only changes the output projection. On smaller-vocabulary or transformer-dominated models, even a large head-level speedup can translate into a small end-to-end improvement. This effect was visible on Qwen3-0.6B, where the dense LM head represented only a few percent of measured decode time and end-to-end gains were correspondingly limited.

### 5.4 Batched block GEMM preserves sparse benefit

The independent row kernel stops scaling once dense GEMM can amortize LM-head weights across a larger batch. The block-GEMM kernel restores that reuse.

For highly constrained Qwen3.5-2B batches, the block kernel achieves the following head-level speedups over dense projection plus grammar masking:

| Batch | Block-GEMM speedup |
| ---: | ---: |
| 2 | 4.38x |
| 4 | 4.30x |
| 8 | 4.46x |
| 16 | 4.60x |
| 32 | 2.48x |

The decline at batch 32 reflects growing efficiency of the dense GEMM and the fixed overhead of sparse tile logic. Nevertheless, the sparse kernel remains substantially faster when all requests are strongly constrained.

Heterogeneous batches expose a more subtle effect. A permissive request can activate a vocabulary tile that would otherwise be skipped for neighboring constrained requests. At moderate batches, the block kernel is near parity or positive for our mixed workload. At batch 32, one measured 50/50 mixed configuration favored dense execution. The dispatcher therefore includes a conservative large-mixed-batch fallback based on the fraction of fully legal packed mask words.

This result is central to the design: batching does not invalidate grammar-aware projection, but it changes the required sparsity granularity. Per-request row sparsity is appropriate for latency; shared block sparsity is appropriate for serving; dense execution remains necessary when heterogeneity destroys skip opportunities.

### 5.5 Correctness

Across all tested row-sparse legal-set sizes from 1 through 120,000 tokens, GSP produced identical legal support and the same greedy top-1 token as dense projection plus grammar masking.

The block-GEMM kernel passed the same checks at batches 2, 4, 8, 16, and 32, including heterogeneous masks. The adaptive dispatcher also passes legal-support and top-1 checks across row, block, and dense routes.

We do observe small FP16 logit differences in some tests because GSP and the dense library kernel use different reduction orders. The largest observed differences in the synthetic correctness sweeps are on the scale expected for FP16 reductions. We therefore make an exactness claim about the constrained support and mathematical computation, not bitwise floating-point identity.

## 6. Analysis

### 6.1 Why packed masks matter

A grammar engine already paid to determine token legality. The packed bitmask is therefore a useful execution artifact, not merely a postprocessing mask. Expanding it into a legal-ID list adds work whose cost is worst in permissive states and introduces a variable-length representation exactly where a fixed-size GPU control signal is preferable.

GSP instead transfers or consumes approximately V bits of legality metadata and lets each GPU tile decide whether expensive LM-head data must be loaded. For Qwen3.5-2B, the 248,320-token mask contains only 7,760 int32 words, roughly 30 KiB per request, compared with an FP16 LM-head matrix near 1 GiB.

### 6.2 The bottleneck is memory as well as arithmetic

Skipping invalid rows reduces multiply-accumulate work, but on consumer GPUs the stronger mechanism is often avoided weight traffic. A sparse grammar state that permits tens of tokens does not need the runtime to stream nearly a gigabyte of output weights merely to reject almost every result afterward.

This also explains the batch behavior. Dense GEMM becomes more attractive when a single weight tile serves many requests. GSP's block kernel must retain that reuse or it gives back the memory advantage.

### 6.3 Grammar structure suggests natural scheduling classes

Our traces reveal at least three execution regimes:

1. **Structural sparse:** delimiters, property names, enums, booleans, bounded numeric transitions. Legal sets are tiny. Row or block sparse projection is strongly favorable.
2. **Permissive content:** arbitrary strings. Most vocabulary rows are legal. Dense projection is competitive.
3. **Heterogeneous batch:** requests simultaneously occupy both regimes. Whether block sparsity helps depends on how request masks overlap inside shared vocabulary tiles.

A production system can exploit this classification more aggressively than our simple threshold dispatcher. Possible extensions include grouping requests by grammar-density class, reordering batch lanes, learning hardware-specific crossover models, or using finer vocabulary tiling for heterogeneous batches.

### 6.4 Relationship to fused sampling

GSP currently materializes a full vocabulary-shaped output tensor with negative infinity in illegal positions so it can drop into conventional downstream selection code. FlashSampling shows that an exact sampler can instead be fused into the LM-head computation and avoid materializing logits in HBM.

The two designs attack different axes. FlashSampling computes every vocabulary tile but avoids logit materialization and sampling passes. GSP avoids impossible vocabulary computation but still materializes the masked output. A natural next step is a grammar-conditioned fused sampler that skips grammar-empty tiles and reduces legal logits to the required sampling statistics on chip. We leave this combination to future work because the current paper isolates the effect of grammar-controlled projection itself.

## 7. Related Work

### Structured generation engines

XGrammar accelerates context-free grammar execution through prechecking, efficient persistent parser state, and overlap with model inference. XGrammar 2 extends the engine to dynamic agentic grammars using TagDispatch, JIT compilation, cross-grammar caching, and generalized parsing. These systems make legal-token mask production cheap enough that the dense LM head becomes a more visible remaining target.

Other structured generation systems, including Outlines and grammar support in modern serving engines, similarly enforce output validity by constraining token selection. GSP is orthogonal to the grammar formalism: it requires only a packed or otherwise efficiently queryable legal-token representation.

### Constrained LM-head filtering

Murali's 2025 CUDA study is the closest prior work we found. It evaluates grammar-aware submatrix projection and Triton kernels that skip output blocks or threads based on a token mask. It also explicitly identifies batching and hybrid execution as open problems. GSP adopts the same underlying observation but develops a direct packed-mask target-model runtime with separate row and block-GEMM kernels, heterogeneous batch handling, dense fallback, and paired end-to-end evaluation.

### Vocabulary reduction and output-head acceleration

FR-Spec reduces draft LM-head computation by restricting a speculative drafter to frequency-ranked vocabulary subsets while preserving the final target distribution through verification. VocabTrim prunes a speculative drafter vocabulary to reduce memory-bound output cost. SpecVocab predicts a step-specific speculative vocabulary. These methods demonstrate that large vocabularies can dominate lightweight draft heads, but their subset selection is not a formal target-model constraint.

FlashHead reframes dense classification as retrieval to reduce output-head cost approximately. Its objective differs from GSP, which has a formal zero-probability certificate for omitted tokens under structured decoding.

FlashSampling fuses exact categorical sampling into dense LM-head tiles and avoids full-logit HBM materialization. It is complementary to GSP's grammar-driven compute sparsity.

## 8. Limitations

This study has several important limitations.

**Hardware scope.** The primary measurements use one NVIDIA RTX 5060 Laptop GPU. Kernel crossover points depend on memory bandwidth, tensor-core throughput, cache hierarchy, compiler version, and batch size. Datacenter GPU results are required before making general claims about H100/B200-class serving.

**Model scope.** The strongest end-to-end result uses Qwen3.5-2B. The mechanism should apply to any decoder with a conventional output matrix, but the end-to-end gain depends on vocabulary size and the fraction of decode time spent in the LM head. We observed smaller gains on Qwen3-0.6B.

**Workload scope.** The evaluation uses four controlled JSON-schema traces rather than a large production tool-calling corpus. They intentionally span sparse and permissive grammar states, but broader schema distributions are needed.

**Batch scheduler scope.** Our dispatcher uses simple thresholds. It does not reorder requests, group similar grammar states, or predict tile occupancy beyond a packed-word density statistic.

**Floating-point identity.** GSP is mathematically equivalent under the constrained distribution, but our FP16 kernels are not guaranteed to reproduce dense-kernel logits bit for bit. We verify legal support and greedy top-1. Exact stochastic sampling equivalence under floating-point implementation would require a carefully specified sampler and numerical contract.

**Full-logit materialization.** The current kernels still write a vocabulary-shaped output containing negative infinity for illegal tokens. Fusing grammar-aware projection with exact sampling could remove this remaining memory traffic.

**Prior-art boundary.** Batch-1 grammar-aware LM-head filtering predates this work. The novelty claim is limited to the combined packed-mask, batched block-GEMM, dispatch, and evaluated target-model system described here.

## 9. Reproducibility and Artifact

The artifact contains:
- Triton row-sparse and block-GEMM kernels;
- stock-XGrammar dense fallback;
- adaptive routing logic;
- correctness tests for support and top-1;
- grammar-density traces;
- Qwen3.5-2B paired end-to-end benchmark scripts and raw per-pair JSON;
- bootstrap and sign-test analysis;
- batch-scaling microbenchmarks.

All timing claims in the paper are derived from committed result files rather than manually transcribed logs. Project-local third-party packages are excluded from the repository and can be installed independently.

The artifact is designed so a reviewer can rerun correctness independently from the more expensive model benchmarks.

## 10. Conclusion

Structured decoding already knows which next tokens are impossible before the model samples. GSP treats that formal knowledge as an execution schedule for the target model's output projection rather than merely a mask applied after dense computation.

A direct packed-bitmask row kernel provides latency-oriented sparse projection, while a batched block-GEMM kernel restores weight reuse under serving batches. A simple dispatcher handles the cases where grammar sparsity disappears. On Qwen3.5-2B, this design yields repeatable median end-to-end decode improvements of approximately 6-8% on three structured JSON workloads and multi-fold LM-head speedups for highly constrained batches through batch 32, while the free-string control shows no reliable mean improvement.

The broader systems lesson is that formal output constraints can inform not only what an LLM may emit, but also which parts of the model's output computation need to execute. As vocabularies continue to grow and structured generation becomes standard in agentic inference, this interface between grammar engines and GPU projection kernels is a useful optimization boundary.
