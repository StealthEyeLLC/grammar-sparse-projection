# Reproducibility

## Reference environment

Primary measurements were collected on:

- Windows 11 x64
- NVIDIA GeForce RTX 5060 Laptop GPU (~8 GiB VRAM)
- Python 3.12
- PyTorch 2.13.0 + CUDA 13.0
- Transformers 5.17.0
- XGrammar 0.2.8
- Triton 3.7.1 for Windows
- Qwen/Qwen3.5-2B, FP16

The code does not require the original machine paths used during development.

## Environment

Create and activate a Python 3.12 virtual environment. Install a CUDA-capable PyTorch build compatible with your NVIDIA driver, then install:

    pip install -r requirements.txt

On Windows, `triton-windows` is used. A Visual Studio C/C++ build environment may be required for first-run Triton driver compilation.

The helper `tools/python_triton.cmd` uses the active `python` by default. Override it with `GSP_PYTHON` if needed.

## Correctness gates

    tools\python_triton.cmd tests\test_triton_sparse.py
    tools\python_triton.cmd tests\test_triton_block_gemm.py
    tools\python_triton.cmd tests\test_adaptive_dispatch.py

Acceptance requires identical legal-token support and identical greedy top-1 against dense LM-head projection followed by grammar masking. FP16 logits are not required to be bitwise identical because reduction order differs.

## Primary paired benchmark

    tools\python_triton.cmd benchmarks\qwen35_adaptive_paired.py
    python benchmarks\analyze_paired.py

The paired benchmark alternates AB/BA order and teacher-forces the same valid output through both systems. The baseline is full target-model LM-head projection followed by stock XGrammar Triton masking.

Primary outputs:

- `results/qwen35-adaptive-paired/results.json`
- `results/qwen35-adaptive-paired/stats.json`

## Batch kernel comparison

See:

- `benchmarks/qwen35_batch_kernel_compare.py`
- `benchmarks/qwen35_adaptive_batch_head.py`
- `results/qwen35-batch-kernel-compare/results.json`
- `results/qwen35-adaptive-batch-head/results.json`

## Grammar-density traces

See:

- `benchmarks/grammar_density_trace.py`
- `results/grammar-density/summary.json`

## Determinism and statistics

`benchmarks/analyze_paired.py` uses a fixed bootstrap RNG seed and 10,000 resamples. It reports median and mean paired speedup, 95% bootstrap intervals, pair wins/losses, and an exact two-sided sign test.

## Scope

The artifact establishes results for the tested models, grammars, GPU, dtypes, and batch regimes only. It does not claim universal crossover thresholds or datacenter-GPU speedups.
