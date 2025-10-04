from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """
    return_indices: torch.jit.Final[bool]

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(self, x) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        assert self.prob > 0
        # if not self.training or self.prob == 0.:
        #     if self.return_indices:
        #         return x, None
        #     return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1. - self.prob)))
        keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[:, :num_keep]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x

class MaskedPatchDropout(nn.Module):
    """
    Modified version of PatchDropout that uses a binary mask to select which patches to keep.
    The mask should be 1 for patches to keep, 0 for patches to drop.
    
    Based on:
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """
    return_indices: torch.jit.Final[bool]

    def __init__(
            self,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape [B, L, D] where B is batch size, L is sequence length, D is dimension
            mask: Binary mask of shape [L-num_prefix_tokens] where 1 indicates patches to keep
        """
        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        
        # Move mask to same device as input
        mask = mask.to(x.device)
        
        # Verify mask shape matches sequence length
        assert mask.shape[-1] == L, f"Mask length {mask.shape[-1]} must match sequence length {L}"
        
        # Get indices where mask is 1 (on same device as x)
        keep_indices = mask.nonzero().squeeze(-1)  # [K] where K is number of 1s
        if self.ordered and keep_indices.numel():
            keep_indices = keep_indices.sort(dim=-1)[0]
        
        # Expand indices for batch dimension and gather
        keep_indices = keep_indices.unsqueeze(0).expand(B, -1)  # [B, K]
        x = torch.gather(x, dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x
