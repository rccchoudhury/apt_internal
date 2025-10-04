"""
Storing the old patch embeddings that we tried.
"""
import ipdb
import itertools
from typing import Union, Tuple, List, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import copy
import einops

from timm.layers import trunc_normal_, resample_abs_pos_embed, PatchEmbed
from timm.layers.helpers import to_2tuple
from timm.layers.format import Format
from timm.layers.trace_utils import _assert

from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from .entropy_utils import select_patches_by_threshold
from .graph_utils import generate_edge_dict, create_edge_embeds, group_patches, calculate_merge_ratio
from .transformer import Block
from .vit_components import Block, Attention

def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    if verbose:
        _logger.info(f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.")

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed

class MixedPatchEmbed(nn.Module):
    # TODO: refactor this, make it more generic / cleaner impl
    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            embed_dim: Optional[int] = None,
            window_size: Optional[int] = None,
            merge_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.img_size = to_2tuple(image_size)[0]  # Assume square image
        self.min_patch_size = 8  # Hardcoded for now

    def _pos_embed(
        self,
        x: torch.Tensor,
        x32: torch.Tensor,
        mask16: Optional[torch.Tensor] = None,
        mask32: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert self.pos_embed16 is not None
        assert self.pos_embed32 is not None
        assert self.pos_drop is not None

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))

        x = x + self.pos_embed16
        x32 = x32 + self.pos_embed32
        x = torch.cat([x, x32], dim=1)
        mask = torch.cat([mask16, mask32], dim=1)
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
            #mask = torch.cat([torch.ones(x.shape[0], len(to_cat), dtype=torch.bool, device=mask.device), mask], dim=1)
        x = x[mask]
        assert len(x.shape) == 2
        x = x.unsqueeze(0)

        return self.pos_drop(x)

    def construct_masks(self, x, entropy_maps):
        mask32, mask16 = select_patches_by_threshold(entropy_maps, threshold=7.0)

        B = x.shape[0]
        num_tokens = []
        for idx in range(B):
            base = mask16[idx].sum().item() + mask32[idx].sum().item()
            #if self.cls_token is not None:
            base += 1 # handle cls token
            #if self.reg_token is not None:
            #    base += self.num_reg_tokens
            num_tokens.append(int(base))

        #block_mask = BlockDiagonalMask.from_seqlens(num_tokens)
        cls_token_indices = [0] + list(itertools.accumulate(num_tokens[:-1]))

        return mask32, mask16, num_tokens, cls_token_indices

    def forward(self, x, entropy_maps):
        # Get patches and add position embeddings
        assert self.patch_embed16 is not None
        assert self.patch_embed32 is not None

        with torch.no_grad():
            mask32, mask16, seqlens, cls_token_indices = self.construct_masks(x, entropy_maps)
        
        # 32 needs to come first!
        x32 = self.patch_embed32(x)
        x = self.patch_embed16(x)
        
        # handle the class token - always retain it.
        mask16 = mask16.flatten(1)
        #TODO: FIX THIS!  HARD CODED !
        mask32 = mask32.flatten(1)
        mask16 = torch.cat([torch.ones(mask16.shape[0], 1, device=mask16.device), mask16], dim=1)

        x = self._pos_embed(x, x32, mask16=mask16.bool(), mask32=mask32.bool())
        
        return x, seqlens, cls_token_indices
    
class MixedSquarePatchEmbed(nn.Module):
    """Mixed Square Patch Embedding module.
    
    This module implements a multi-scale patch embedding that can handle different patch sizes
    based on entropy maps. It supports adaptive patch selection and attention mechanisms.
    
    Args:
        image_size (Union[int, Tuple[int, int]]): Input image size. Default: 224
        patch_size (Union[int, Tuple[int, int]]): Base patch size. Default: 16
        embed_dim (Optional[int]): Embedding dimension
        window_size (Optional[int]): Size of attention window
        merge_ratio (Optional[float]): Ratio for merging patches
        num_scales (int): Number of different patch scales. Default: 3
        thresholds (Optional[List[float]]): Thresholds for patch selection
        mode (Optional[str]): Mode of operation for position embedding
    """
    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            embed_dim: Optional[int] = None,
            window_size: Optional[int] = None,
            merge_ratio: Optional[float] = None,
            num_scales: int = 2,
            thresholds: Optional[List[float]] = None,
            mode: Optional[str] = None,
            alpha_schedule: Optional[bool] = None,
    ):
        super().__init__()
        
        # Initialize size parameters
        self.img_size = to_2tuple(image_size)[0]  # Assume square image
        self.base_patch_size = patch_size
        self.num_scales = num_scales
        self.patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
        self.thresholds = thresholds if thresholds else [6.0] * (num_scales - 1)
        self.alpha_schedule = alpha_schedule
        self.alpha = 1.0  # Training progress (current_epoch / max_epochs)
        
        # Initialize embedding tokens
        self.grp_token = nn.Parameter(torch.zeros(embed_dim))
        self.pos_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.grp_token, std=0.01)
        nn.init.normal_(self.pos_token, std=0.01)
        
        # Initialize patch embedding
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            bias=True,#not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=False,
        )
        
        # Initialize attention layers
        attn_config = dict(
            dim=embed_dim,
            num_heads=4,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        self.patch_attn = Attention(**attn_config)
        self.pos_attn = Attention(**attn_config)

        # Initialize position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * .02)

        # Other parameters
        self.cls_token = None  # Will be set externally if needed
        self.mode = mode
        self.ln = nn.LayerNorm(embed_dim)

    def construct_masks(self, x: torch.Tensor, entropy_maps: torch.Tensor) -> Tuple[List[torch.Tensor], List[int], List[int], int]:
        """Construct masks for patch selection based on entropy maps.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            entropy_maps (torch.Tensor): Entropy maps for patch selection
            
        Returns:
            Tuple containing:
                - List[torch.Tensor]: Masks for each scale
                - List[int]: Number of tokens per batch item
                - List[int]: Indices for class tokens
                - int: Total number of tokens
        """
        # Get masks for each scale using the corresponding thresholds
        alpha = self.alpha
        if not self.training or not self.alpha_schedule:
            alpha = 1.0
        masks = select_patches_by_threshold(entropy_maps, thresholds=self.thresholds, alpha=alpha)
        
        batch_size = x.shape[0]
        num_tokens = []
        
        # Calculate number of tokens per batch item
        for idx in range(batch_size):
            tokens = sum(mask[idx].sum().item() for patch_size, mask in masks.items())
            tokens += 1  # Add class token
            num_tokens.append(int(tokens))
        
        # Calculate indices for class tokens and total tokens
        cls_token_indices = [0] + list(itertools.accumulate(num_tokens[:-1]))
        total_tokens = sum(num_tokens)
        
        return masks, num_tokens, cls_token_indices, total_tokens

    def _pos_embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply positional embedding to the input tensor.
        
        Handles class token concatenation if present.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Tensor with position embedding applied
                - Position embedding tensor
        """
        to_cat = []
        
        # Add class token if present
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + self.pos_embed
            
        return x

    def forward(self, x, entropy_maps):
        batch_size = x.shape[0]
        # Set this explicitly. Will make formatting a bit easier.
        with torch.no_grad():
            masks, seq_lengths, cls_tok_loc, total_tokens = self.construct_masks(x, entropy_maps)
        masks = {k: v.view(batch_size, -1) for k, v in masks.items()}

        # Apply patch embedding and get embeddings
        patch_embed = self.patch_embed(x)
        patch_embed = self._pos_embed(patch_embed)
        embed_dim = patch_embed.shape[-1]
        
        # IF WE HAVE A CLASS TOKEN ALREADY EMBEDDED< SEPARATE IT OUT. 
        if self.cls_token is not None:
            cls_token = patch_embed[:, 0]
            patch_embed = patch_embed[:, 1:]
            
        h = w = self.img_size // self.base_patch_size
        
        patch_embeds = {}
        for i, patch_size in enumerate(self.patch_sizes):
            scale = 2**i
            patch_embed_rearrange = einops.rearrange(patch_embed, 'b (h p1 w p2) c -> b (h w) (p1 p2) c', h=h//scale, w=w//scale, p1=scale, p2=scale)
            patch_embeds[patch_size] = patch_embed_rearrange
        
        num_elements = {}
        for patch_size in self.patch_sizes:
            mask = masks[patch_size]
            num_elements[patch_size] = mask.sum().int()
            
        assert sum([count * 4**i for i, count in enumerate(num_elements.values())]) == batch_size * h * w
        
        #### Masking rule ####
        # grp_tok = -2
        # cls_tok = -1
        # Not selected patches = 0
        # Size (Base), selected = 1
        # Size (Base*2), selected = 2
        # Size (Base*4), selected = 3 
        # Size (Base*8), selected = 4
        # And so on.. 
        #### Masking rule ####

        masks_to_combine = []
            
        cls_to_combine = torch.full((batch_size, 1), -1, device=x.device)
        masks_to_combine.append(cls_to_combine)
        
        for i, patch_size in enumerate(self.patch_sizes):
            mask = masks[patch_size]
            masks_to_combine.append(mask * (i+1))
        
        combined_mask = torch.cat(masks_to_combine, dim=1)

        if self.mode != 'drop':
            valid_patches = combined_mask[combined_mask != 0]
        else:
            valid_patches = combined_mask[(combined_mask == 1) | (combined_mask == -1)]

        mask_locs = {}
        for i, patch_size in enumerate(self.patch_sizes):
            mask_locs[patch_size] = torch.where(valid_patches == i+1)

        patch_embeds_postprocessed = {}
        for patch_size in self.patch_sizes:
            patch_embed = patch_embeds[patch_size]
            
            if patch_size == self.base_patch_size:
                patch_embeds_postprocessed[patch_size] = patch_embed[masks[self.base_patch_size].bool()].squeeze(1)
                
            elif mask_locs[patch_size][0].numel() > 0:
                scale = patch_size // self.base_patch_size
                grp_token_expanded = self.grp_token.view(1, 1, 1, -1).expand(
                    batch_size,
                    patch_embed.shape[1], # number of patches
                    1,
                    embed_dim
                )
                
                if self.mode == 'grp_token':
                    # Expand the "group token" to match the total nbumber of 32x32 patches 
                    # available in input.
                    patch32_seqlens = [(scale**2+1) for _ in range(num_elements[patch_size].item())]
                    patch_embed = torch.cat([grp_token_expanded, patch_embed], dim=2)
                    patch_embed = patch_embed[masks[patch_size].bool()].unsqueeze(0)
                    
                    patch_embed_attn = patch_embed.reshape(1, -1, embed_dim)
                    patch_embed_attn = self.patch_attn(patch_embed_attn, patch32_seqlens)
                    patch_embed_attn = patch_embed_attn.reshape(-1, (scale**2+1), embed_dim)
                    
                    patch_embed_final = patch_embed_attn[:, 0]
                    
                elif self.mode == 'grp_token_residual':
                    patch32_seqlens = [(scale**2+1) for _ in range(num_elements[patch_size].item())]
                    patch_embed = torch.cat([grp_token_expanded, patch_embed], dim=2)
                    patch_embed = patch_embed[masks[patch_size].bool()].unsqueeze(0)
                    
                    patch_embed_attn = patch_embed.reshape(1, -1, embed_dim)
                    patch_embed_attn = self.patch_attn(patch_embed_attn, patch32_seqlens)
                    patch_embed_attn = patch_embed_attn.reshape(-1, (scale**2+1), embed_dim)
                    
                    patch_embed_final = patch_embed_attn[:, 0] + patch_embed.mean(dim=1)
                    
                elif self.mode == 'grp_token_ln':
                    patch32_seqlens = [(scale**2+1) for _ in range(num_elements[patch_size].item())]
                    patch_embed = torch.cat([grp_token_expanded, patch_embed], dim=2)
                    patch_embed = patch_embed[masks[patch_size].bool()].unsqueeze(0)
                    
                    patch_embed_attn = patch_embed.reshape(1, -1, embed_dim)
                    patch_embed_attn = self.patch_attn(patch_embed_attn, patch32_seqlens)
                    patch_embed_attn = patch_embed_attn.reshape(-1, (scale**2+1), embed_dim)
                    
                    patch_embed_final = patch_embed_attn[:, 0]
                    
                elif self.mode == 'mean':
                    patch_embed = patch_embed[masks[patch_size].bool()].unsqueeze(0)
                    patch_embed = patch_embed.reshape(-1, (scale**2), embed_dim)
                    patch_embed_final = patch_embed.mean(dim=1)
                    
                elif self.mode == 'attn_mean':
                    patch32_seqlens = [(scale**2) for _ in range(num_elements[patch_size].item())]
                    patch_embed = patch_embed[masks[patch_size].bool()].unsqueeze(0)
                    
                    patch_embed_attn = patch_embed.reshape(1, -1, embed_dim)
                    patch_embed_attn = self.patch_attn(patch_embed_attn, patch32_seqlens)
                    patch_embed_attn = patch_embed_attn.reshape(-1, (scale**2), embed_dim)
                    
                    patch_embed_final = patch_embed_attn.mean(dim=1)
                    
                elif self.mode == 'first_token':
                    patch_embed = patch_embed[masks[patch_size].bool()].unsqueeze(0)
                    patch_embed_final = patch_embed[:, :, 0]
                    
                else:
                    raise NotImplementedError
                patch_embeds_postprocessed[patch_size] = patch_embed_final
                
            else:
                patch_embed = torch.zeros((0, embed_dim), device=x.device)
                patch_embeds_postprocessed[patch_size] = patch_embed
                
        # Convert back to full precision here, since patches16 is in full. Could do half later though?>
        expanded_output = torch.zeros((sum(num_elements.values()) + batch_size, embed_dim), device=x.device)
        expanded_output[cls_tok_loc] = cls_token
        for patch_size in self.patch_sizes:
            expanded_output[mask_locs[patch_size]] = patch_embeds_postprocessed[patch_size].float()
        expanded_output = expanded_output.unsqueeze(0)
        
        return expanded_output, seq_lengths, cls_tok_loc
