from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError(
        "GSP Triton runtime requires project-local triton-windows. "
        "Launch through tools/python_triton.cmd."
    ) from exc


@triton.jit
def _bitmask_sparse_lm_head_kernel(
    hidden_ptr,
    weight_ptr,
    bitmask_ptr,
    out_ptr,
    VOCAB: tl.constexpr,
    HIDDEN: tl.constexpr,
    MASK_WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(0)
    pid = tl.program_id(1)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    in_vocab = rows < VOCAB

    mask_base = bitmask_ptr + batch_id * MASK_WORDS
    words = tl.load(mask_base + rows // 32, mask=in_vocab, other=0)
    bit = rows % 32
    allowed = in_vocab & (((words >> bit) & 1) != 0)

    any_allowed = tl.sum(allowed.to(tl.int32), axis=0) != 0
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    hidden_base = hidden_ptr + batch_id * HIDDEN

    for k0 in tl.static_range(0, HIDDEN, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        col_ok = cols < HIDDEN
        h = tl.load(
            hidden_base + cols,
            mask=col_ok & any_allowed,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            weight_ptr + rows[:, None] * HIDDEN + cols[None, :],
            mask=allowed[:, None] & col_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * h[None, :], axis=1)

    values = tl.where(allowed, acc, -float("inf"))
    out_base = out_ptr + batch_id * VOCAB
    tl.store(out_base + rows, values, mask=in_vocab)


def bitmask_sparse_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bitmask_gpu: torch.Tensor,
    *,
    block_m: int = 32,
    block_k: int = 256,
) -> torch.Tensor:
    """Compute exact grammar-constrained logits without invalid-row matmul work.

    Supports one or more hidden states sharing the same LM-head weight matrix,
    with one grammar bitmask per batch element. Invalid token rows are never
    loaded from the LM-head weight and are emitted as -inf.
    """
    if hidden.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("hidden and weight must be CUDA tensors")
    if bitmask_gpu.device.type != "cuda" or bitmask_gpu.dtype != torch.int32:
        raise ValueError("bitmask_gpu must be CUDA int32")
    if hidden.dtype != weight.dtype:
        raise ValueError("hidden and weight dtypes must match")
    if hidden.ndim == 1:
        hidden_2d = hidden.reshape(1, -1)
        squeeze = True
    elif hidden.ndim == 2:
        hidden_2d = hidden
        squeeze = hidden.shape[0] == 1
    else:
        raise ValueError("hidden must have shape [D] or [B, D]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [V, D]")

    vocab, hidden_dim = weight.shape
    batch = hidden_2d.shape[0]
    if hidden_2d.shape[1] != hidden_dim:
        raise ValueError("hidden dimension mismatch")
    if not hidden_2d.is_contiguous() or not weight.is_contiguous():
        raise ValueError("hidden and weight must be contiguous")

    mask_words = (vocab + 31) // 32
    if bitmask_gpu.ndim == 1:
        mask_2d = bitmask_gpu.reshape(1, -1)
    elif bitmask_gpu.ndim == 2:
        mask_2d = bitmask_gpu
    else:
        raise ValueError("bitmask must have shape [words] or [B, words]")
    if mask_2d.shape[0] != batch or mask_2d.shape[1] < mask_words:
        raise ValueError("bitmask shape does not match hidden batch/vocabulary")
    if not mask_2d.is_contiguous():
        raise ValueError("bitmask must be contiguous")

    out = torch.empty((batch, vocab), device=weight.device, dtype=hidden.dtype)
    grid = (batch, triton.cdiv(vocab, block_m))
    _bitmask_sparse_lm_head_kernel[grid](
        hidden_2d,
        weight,
        mask_2d,
        out,
        VOCAB=vocab,
        HIDDEN=hidden_dim,
        MASK_WORDS=mask_words,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out[0] if squeeze else out


def allocate_pinned_bitmask(batch_size: int, vocab_size: int) -> torch.Tensor:
    shape = (batch_size, (vocab_size + 31) // 32)
    return torch.full(shape, -1, dtype=torch.int32, pin_memory=True)


def copy_bitmask_to_gpu(
    bitmask_cpu: torch.Tensor,
    bitmask_gpu: torch.Tensor | None = None,
) -> torch.Tensor:
    if bitmask_cpu.device.type != "cpu" or bitmask_cpu.dtype != torch.int32:
        raise ValueError("bitmask_cpu must be CPU int32")
    if not bitmask_cpu.is_pinned():
        raise ValueError("bitmask_cpu must be pinned for asynchronous transfer")
    if bitmask_gpu is None:
        bitmask_gpu = torch.empty_like(bitmask_cpu, device="cuda")
    bitmask_gpu.copy_(bitmask_cpu, non_blocking=True)
    return bitmask_gpu


@triton.jit
def _bitmask_block_gemm_kernel(
    hidden_ptr,
    weight_ptr,
    bitmask_ptr,
    out_ptr,
    BATCH: tl.constexpr,
    VOCAB: tl.constexpr,
    HIDDEN: tl.constexpr,
    MASK_WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_m = offs_m < BATCH
    valid_n = offs_n < VOCAB

    mask_words = tl.load(
        bitmask_ptr
        + offs_m[:, None] * MASK_WORDS
        + (offs_n[None, :] // 32),
        mask=valid_m[:, None] & valid_n[None, :],
        other=0,
    )
    bits = offs_n % 32
    allowed = (
        valid_m[:, None]
        & valid_n[None, :]
        & (((mask_words >> bits[None, :]) & 1) != 0)
    )

    tile_active = tl.sum(allowed.to(tl.int32), axis=1)
    tile_active = tl.sum(tile_active, axis=0) != 0

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)

    for k0 in tl.static_range(0, HIDDEN, BLOCK_K):
        ks = k0 + offs_k
        valid_k = ks < HIDDEN
        a = tl.load(
            hidden_ptr + offs_m[:, None] * HIDDEN + ks[None, :],
            mask=valid_m[:, None] & valid_k[None, :] & tile_active,
            other=0.0,
        )
        b = tl.load(
            weight_ptr + offs_n[None, :] * HIDDEN + ks[:, None],
            mask=valid_k[:, None] & valid_n[None, :] & tile_active,
            other=0.0,
        )
        acc += tl.dot(a, b)

    values = tl.where(allowed, acc, -float("inf"))
    tl.store(
        out_ptr + offs_m[:, None] * VOCAB + offs_n[None, :],
        values,
        mask=valid_m[:, None] & valid_n[None, :],
    )


def bitmask_block_gemm_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bitmask_gpu: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """Batched grammar-aware LM-head using block GEMM and tile skipping."""
    if hidden.ndim != 2 or hidden.shape[0] < 2:
        raise ValueError("block GEMM path expects hidden shape [B,D] with B >= 2")
    if hidden.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("hidden and weight must be CUDA tensors")
    if bitmask_gpu.device.type != "cuda" or bitmask_gpu.dtype != torch.int32:
        raise ValueError("bitmask_gpu must be CUDA int32")
    if hidden.dtype != weight.dtype:
        raise ValueError("hidden and weight dtypes must match")
    if not hidden.is_contiguous() or not weight.is_contiguous() or not bitmask_gpu.is_contiguous():
        raise ValueError("inputs must be contiguous")

    batch, hidden_dim = hidden.shape
    vocab, weight_hidden = weight.shape
    if hidden_dim != weight_hidden:
        raise ValueError("hidden dimension mismatch")
    mask_words = (vocab + 31) // 32
    if bitmask_gpu.ndim != 2 or bitmask_gpu.shape[0] != batch or bitmask_gpu.shape[1] < mask_words:
        raise ValueError("bitmask shape mismatch")

    out = torch.empty((batch, vocab), device=hidden.device, dtype=hidden.dtype)
    grid = (triton.cdiv(batch, block_m), triton.cdiv(vocab, block_n))
    _bitmask_block_gemm_kernel[grid](
        hidden,
        weight,
        bitmask_gpu,
        out,
        BATCH=batch,
        VOCAB=vocab,
        HIDDEN=hidden_dim,
        MASK_WORDS=mask_words,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def _apply_bitmask_kernel(
    logits_ptr,
    bitmask_ptr,
    BATCH: tl.constexpr,
    VOCAB: tl.constexpr,
    MASK_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = BATCH * VOCAB
    valid = idx < total
    batch = idx // VOCAB
    token = idx % VOCAB
    word = tl.load(
        bitmask_ptr + batch * MASK_WORDS + token // 32,
        mask=valid,
        other=0,
    )
    allowed = ((word >> (token % 32)) & 1) != 0
    x = tl.load(logits_ptr + idx, mask=valid, other=0.0)
    tl.store(logits_ptr + idx, tl.where(allowed, x, -float("inf")), mask=valid)


def dense_masked_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bitmask_gpu: torch.Tensor,
) -> torch.Tensor:
    """Best-available conventional baseline/fallback: dense head + stock XGrammar mask."""
    import torch.nn.functional as F
    import xgrammar as xgr

    if hidden.ndim == 1:
        hidden_2d = hidden.reshape(1, -1)
        squeeze = True
    elif hidden.ndim == 2:
        hidden_2d = hidden
        squeeze = hidden.shape[0] == 1
    else:
        raise ValueError("hidden must have shape [D] or [B,D]")
    logits = F.linear(hidden_2d, weight)
    batch, vocab = logits.shape
    mask_2d = bitmask_gpu.reshape(batch, -1)
    xgr.apply_token_bitmask_inplace(
        logits, mask_2d, vocab_size=vocab, backend="triton"
    )
    return logits[0] if squeeze else logits

def full_word_fraction_cpu(bitmask_cpu: torch.Tensor) -> list[float]:
    """Cheap CPU-side sparsity signal from XGrammar's already-resident bitmask.

    A full int32 word (-1) means all 32 vocabulary IDs in that word are legal.
    This avoids expanding the packed grammar mask merely to choose a kernel.
    """
    if bitmask_cpu.device.type != "cpu" or bitmask_cpu.dtype != torch.int32:
        raise ValueError("bitmask_cpu must be CPU int32")
    mask_2d = bitmask_cpu.reshape(bitmask_cpu.shape[0], -1)
    counts = (mask_2d == -1).sum(dim=1)
    denom = mask_2d.shape[1]
    return [float(x) / denom for x in counts.tolist()]


def adaptive_grammar_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bitmask_gpu: torch.Tensor,
    *,
    full_word_fraction: list[float] | None = None,
    dense_full_word_threshold: float = 0.80,
) -> tuple[torch.Tensor, str]:
    """Hardware-adaptive exact grammar-conditioned output projection.

    Routing policy:
      * batched near-unconstrained states use dense projection plus masking;
      * large mixed-density batches can fall back when shared tiles are overactivated;
      * batch 1 uses row-sparse direct projection;
      * other batches use grammar-aware block GEMM.
    The density threshold is intentionally exposed for calibration/ablation.
    """
    batch = 1 if hidden.ndim == 1 else hidden.shape[0]
    if full_word_fraction is not None:
        if len(full_word_fraction) != batch:
            raise ValueError("full_word_fraction length must equal batch")
        mean_full = sum(full_word_fraction) / batch
        all_near_dense = min(full_word_fraction) >= dense_full_word_threshold
        large_mixed_dense = batch >= 32 and mean_full >= 0.25
        if batch > 1 and (all_near_dense or large_mixed_dense):
            return dense_masked_logits(hidden, weight, bitmask_gpu), "dense-mask"

    if batch == 1:
        return bitmask_sparse_logits(hidden, weight, bitmask_gpu), "row-sparse"
    return bitmask_block_gemm_logits(hidden, weight, bitmask_gpu), "block-sparse"
