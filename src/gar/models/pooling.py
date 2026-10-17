"""Pool player representations into a scene representation (paper eq. 10-13).

Players are in slots ordered left to right on the target frame, valid slots first, padding last.
With d groups over K slots, group m covers slots [m*K/d, (m+1)*K/d): for d=2 and K=12, slots 0-5
(left team) and 6-11 (right team). As in the paper, missing players make the last group smaller.
"""

import torch


def group_pool(x, mask, groups=1, mode="max"):
    """x: (..., K, D), mask: (..., K) bool -> (..., groups * D)."""
    k = x.shape[-2]
    if k % groups:
        raise ValueError(f"{k} slots cannot be split into {groups} equal groups")
    size = k // groups
    m = mask.unsqueeze(-1)
    out = []
    for g in range(groups):
        xs, ms = x[..., g * size : (g + 1) * size, :], m[..., g * size : (g + 1) * size, :]
        if mode == "max":
            pooled = xs.masked_fill(~ms, float("-inf")).amax(dim=-2)
            pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)  # empty group
        elif mode == "avg":
            pooled = (xs * ms).sum(-2) / ms.sum(-2).clamp(min=1)
        else:
            raise ValueError(mode)
        out.append(pooled)
    return torch.cat(out, dim=-1)
