from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ProjectionResult:
    token_ids: torch.Tensor
    logits: torch.Tensor
    mode: str


def constrained_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    allowed_ids: Optional[torch.Tensor],
    *,
    dense_threshold: float = 0.30,
) -> ProjectionResult:
    """Project only legal vocabulary rows when profitable.

    Returns logits paired with the token IDs those logits correspond to.
    The mathematical distribution is identical to dense projection followed by
    masking all tokens outside allowed_ids.
    """
    vocab = weight.shape[0]
    if allowed_ids is None:
        ids = torch.arange(vocab, device=weight.device, dtype=torch.long)
        return ProjectionResult(ids, F.linear(hidden, weight), "dense-unconstrained")

    ids = allowed_ids.to(device=weight.device, dtype=torch.long)
    if ids.ndim != 1:
        raise ValueError("allowed_ids must be one-dimensional")
    if ids.numel() == 0:
        raise ValueError("grammar produced an empty legal-token set")
    if torch.any(ids < 0) or torch.any(ids >= vocab):
        raise ValueError("allowed_ids contains an out-of-range token ID")

    # Duplicate legal IDs are semantically redundant and waste projection work.
    ids = torch.unique(ids, sorted=True)
    density = ids.numel() / vocab

    if density >= dense_threshold:
        dense = F.linear(hidden, weight)
        return ProjectionResult(ids, dense.index_select(-1, ids), "dense-selected")

    rows = weight.index_select(0, ids)
    return ProjectionResult(ids, F.linear(hidden, rows), "sparse-gather")
