"""
Customizable patch embedding code pasted from timm.
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

_logger = logging.getLogger(__name__)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
        self.embed_dim = embed_dim

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_2tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv2d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(resample_patch_embed(self.proj.weight, new_patch_size, verbose=True))
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape

        # Split the image into patches
        patches = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
        patches = patches.contiguous().view(B, C, -1, self.patch_size[0] * self.patch_size[1])
        patches = patches.transpose(1, 2)
        patches = patches.reshape(-1, C, self.patch_size[0], self.patch_size[1])
        # Apply convolution to each patch
        x = self.proj(patches)
        
        if self.flatten:
            x = x.reshape(B, -1, self.proj.out_channels)
        elif self.output_fmt != Format.NCHW:
            x = nlc_to(x, self.output_fmt)
        x = self.norm(x)
        return x

    def forward_patch(self, patches):
        x = self.proj(patches)
        
        if self.flatten:
            x = x.reshape(-1, self.proj.out_channels)
        elif self.output_fmt != Format.NCHW:
            x = nlc_to(x, self.output_fmt)
        x = self.norm(x)
        return x

class OnlySmallPatch(nn.Module):
    # TODO: refactor this, make it more generic / cleaner impl
    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = to_2tuple(image_size)[0]  # Assume square image
        self.min_patch_size = 8  # Hardcoded for now
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            bias=True,#not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=False,
        )
        self.reg_token = False
        
        self.num_patches = self.patch_embed.num_patches
        embed_len = self.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

    def construct_masks(self, x, entropy_maps):
        mask32, mask16 = select_patches_by_threshold(entropy_maps, threshold=5.5)

        B = x.shape[0]
        num_tokens = []
        for idx in range(B):
            # DO NOT INCLUDE 32!
            base = mask16[idx].sum().item()
            base += 1 # handle cls token
            num_tokens.append(int(base))

        #block_mask = BlockDiagonalMask.from_seqlens(num_tokens)
        cls_token_indices = [0] + list(itertools.accumulate(num_tokens[:-1]))

        total_tokens = sum(num_tokens)

        return mask16, num_tokens, cls_token_indices, total_tokens

    def forward(self, x, entropy_maps):
        B = x.shape[0]
        # Set this explicitly. Will make formatting a bit easier.
        #ipdb.set_trace()
        with torch.no_grad():
            mask16, seqlens, cls_token_indices, total_tokens = self.construct_masks(x, entropy_maps)

        patchembed = self.patch_embed(x)
        
        mask16 = mask16.flatten(1)
        # Prepend class token mask
        cls_token_mask = torch.ones_like(mask16[:, :1], device=mask16.device)
        mask16 = torch.cat([cls_token_mask, mask16], dim=1).bool()
        # Prepend class oken to input patches
        patchembed = torch.cat([self.cls_token.expand(B, -1, -1), patchembed], dim=1)
        # Add in the pos embed (if it includes cls token; need to make this flexible)
        patchembed = patchembed + self.pos_embed

        patchembed = patchembed[mask16].unsqueeze(0)
        
        return patchembed, seqlens, cls_token_indices

class QuadformerDownsizePatchEmbed(nn.Module):
    # TODO: refactor this, make it more generic / cleaner impl
    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            embed_dim: Optional[int] = None,
            num_scales: int = 2,
            thresholds: Optional[List[float]] = None,
            mode: Optional[str] = None,
            alpha_schedule: Optional[bool] = None,
    ):
        super().__init__()
        self.img_size = to_2tuple(image_size)[0]  # Assume square image
        self.min_patch_size = 8  # Hardcoded for now
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            bias=True,#not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=False,
        )
        self.num_scales = num_scales
        self.thresholds = thresholds
        self.alpha_schedule = False
        
        self.num_patches = self.patch_embed.num_patches
        embed_len = self.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        # NEed a pos embedding for 32 (resized pos-embed.)
        self.pos_embed32 = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

    def construct_masks(self, x, entropy_maps):
        masks = select_patches_by_threshold(entropy_maps, thresholds=[5.5], alpha=1)
        mask32, mask16 = masks[32], masks[16]

        B = x.shape[0]
        num_tokens = []
        for idx in range(B):
            base = mask16[idx].sum().item() + mask32[idx].sum().item()
            base += 1 # handle cls token
            num_tokens.append(int(base))

        #block_mask = BlockDiagonalMask.from_seqlens(num_tokens)
        cls_token_indices = [0] + list(itertools.accumulate(num_tokens[:-1]))

        total_tokens = sum(num_tokens)

        return mask32, mask16, num_tokens, cls_token_indices, total_tokens

    def forward(self, x, entropy_maps):
        B = x.shape[0]
        # Set this explicitly. Will make formatting a bit easier.
        # Only return the patches we will actually use.
        with torch.no_grad():
            downsized_img = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            mask32, mask16, seqlens, cls_token_indices, total_tokens = self.construct_masks(x, entropy_maps)

            patches32 = einops.rearrange(
                downsized_img, 
                'b c (h p1) (w p2)-> b (h w) c p1 p2', 
                p1=16, p2=16
            )
            patches16 = einops.rearrange(
                x, 
                'b c (h p1) (w p2)-> b (h w) c p1 p2', 
                p1=16, p2=16
            )
            all_patches = torch.cat([patches16, patches32], dim=1)
            mask16 = mask16.flatten(1)
            mask32 = mask32.flatten(1)
            combined_mask = torch.cat([mask16, mask32], dim=1).bool()
            selected_patches = all_patches[combined_mask]

            # pos_embed32 = resample_abs_pos_embed(
            #    self.pos_embed,
            #    new_size=(7, 7),
            #    old_size=self.patch_embed16.grid_size,
            #    num_prefix_tokens=1,
            # )

        patch_embeds = self.patch_embed(selected_patches).squeeze(1)
        embed_dim = patch_embeds.shape[-1]
        
        # Add pos embed, skip cls token. 
        pos_embeds_to_add = torch.cat([self.pos_embed[:, 1:], self.pos_embed32[:, 1:]], dim=1)
        # repeat along batch dimension.
        pos_embeds_to_add = pos_embeds_to_add.repeat(B, 1, 1)
        selected_pos_embeds = pos_embeds_to_add[combined_mask]

        patch_embeds = patch_embeds + selected_pos_embeds
        cls_token = self.cls_token + self.pos_embed[:, :1]
        # Add the cls tokens in the right spots, and the embeddings everywhere else.
        # Total tokens already handles the class token, no need to add an extra 1.
        expanded_output = torch.zeros((total_tokens, embed_dim), device=cls_token.device)
        output_mask = torch.ones((total_tokens))
        expanded_output[cls_token_indices] = cls_token
        output_mask[cls_token_indices] = 0
        expanded_output[output_mask.bool()] = patch_embeds
        # Add a batch dim = 1.
        expanded_output = expanded_output.unsqueeze(0)
        
        return expanded_output, seqlens, cls_token_indices

class ZeroConvPatchAttn(nn.Module):
    # TODO: refactor this, make it more generic / cleaner impl
    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            embed_dim: Optional[int] = None,
            num_scales: int = 2,
            thresholds: Optional[List[float]] = None,
            mode: Optional[str] = None,
            alpha_schedule: Optional[bool] = None,
    ):
        super().__init__()
        self.img_size = to_2tuple(image_size)[0]  # Assume square image
        self.base_patch_size = patch_size
        self.num_scales = num_scales
        self.patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
        self.thresholds = thresholds
        self.alpha_schedule = False
        self.embed_dim = embed_dim
        
        self.num_patches = (self.img_size // self.base_patch_size) ** 2 + 1 # Add 1 for the cls token

        attn_config = dict(
            dim=embed_dim,
            num_heads=1,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.1,
            proj_drop=0.1,
        )
        self.patch_attn = Attention(**attn_config)
        # Base pos embed for the patch. Will dynamically resample for other sizes.
        self.base_mini_pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * .02)

        # Zero conv for adding in attention.
        self.zero_conv = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

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
        masks = select_patches_by_threshold(entropy_maps, thresholds=self.thresholds, alpha=1.0) # alpha 1.0 means we don't use alpha scheduling
        
        batch_size = x.shape[0]
        num_tokens = []
        
        # Calculate number of tokens per batch item
        for idx in range(batch_size):
            tokens = sum(mask[idx].sum().item() for patch_size, mask in masks.items())
            tokens += 1  # Add class token
            num_tokens.append(int(tokens))
        
        # Calculate indices for class tokens and total tokens
        cls_tok_loc = [0] + list(itertools.accumulate(num_tokens[:-1]))
        total_tokens = sum(num_tokens)
        
        return masks, num_tokens, cls_tok_loc, total_tokens

    def _prepare_input(self, x, entropy_maps):
        batch_size = x.shape[0]
        base_patch_size = self.base_patch_size
        h = w = self.img_size // base_patch_size
        
        # Downsize the input image (for resized embeds)
        with torch.no_grad():
            masks, seq_lengths, cls_tok_loc, total_tokens = self.construct_masks(x, entropy_maps)
            masks = {k: v.view(batch_size, -1) for k, v in masks.items()}
            
        # Make sure that the mask is correct
        assert_count = 0
        for i, patch_size in enumerate(self.patch_sizes):
            mask = masks[patch_size]
            assert_count += mask.sum().int() * 4**i
        assert assert_count == batch_size * h * w
            
        patch_embed_base = self.patch_embed(x)
        
        return batch_size, h, w, masks, cls_tok_loc, total_tokens, patch_embed_base, seq_lengths

    def _compute_multi_scale_embeddings(self, x, masks, patch_embed_base):
        batch_size = x.shape[0]
        patch_embeds_masked = {}
        for i, patch_size in enumerate(self.patch_sizes):
            mask = masks[patch_size]
            if mask.sum() == 0:
                patch_embed = torch.zeros((0, self.embed_dim), device=x.device)
                patch_embeds_masked[patch_size] = patch_embed
                continue
            
            if patch_size == self.base_patch_size:
                patch_embed = patch_embed_base[mask.bool()]
                patch_embeds_masked[patch_size] = patch_embed
                continue
            
            scale = 2**i
            x_downsize = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
        
            patches = x_downsize.unfold(2, self.base_patch_size, self.base_patch_size).unfold(3, self.base_patch_size, self.base_patch_size)
            patches = patches.contiguous().view(batch_size, 3, -1, self.base_patch_size * self.base_patch_size)
            patches = patches.transpose(1, 2)
            patches = patches.reshape(-1, 3, self.base_patch_size, self.base_patch_size)
            
            patches_selected = patches[mask.flatten().bool()]
            patch_embed = self.patch_embed.forward_patch(patches_selected)
            patch_embeds_masked[patch_size] = patch_embed
        return patch_embeds_masked

    def _prepare_attention_output(self, patch_embeds_masked, masks, batch_size, h, w, patch_embed_base):
        patch_embeds_postprocessed = {}
        for i, patch_size in enumerate(self.patch_sizes):
            patch_embed = patch_embeds_masked[patch_size]
            mask = masks[patch_size]
            scale = 2**i
            
            if mask.sum() == 0:
                patch_embed_postprocessed = patch_embed
            elif patch_size == self.base_patch_size:
                pos_embed = self.base_mini_pos_embed.repeat(batch_size, 1, 1)
                patch_embed_postprocessed = patch_embed + pos_embed[:, 1:][mask.bool()]
            else:
                patch_embed_base_rearranged = einops.rearrange(
                    patch_embed_base,
                    'b (h p1 w p2) c -> b (h w) (p1 p2) c',
                    h = h//scale, w = w//scale, p1 = scale, p2 = scale
                )
                # Select patches for attention based on mask
                patch_embed_for_attn = patch_embed_base_rearranged[mask.bool()]
                # Apply local positional encoding
                scale_factor = patch_size // self.base_patch_size
                mini_grid_size = int(np.sqrt(patch_embed_for_attn.shape[1]))
                resampled_pos_embed = resample_abs_pos_embed(
                    self.base_mini_pos_embed,
                    new_size=(mini_grid_size, mini_grid_size),
                    old_size=(2, 2),  # Base size is 2x2 grid (4 patches)
                    num_prefix_tokens=0,
                )
                patch_embed_for_attn = patch_embed_for_attn + resampled_pos_embed
                # Process through attention layer
                patch_embed = self.patch_attn(patch_embed_for_attn)
                # Pool attention output
                patch_embed = patch_embed.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
                # Control net style zero conv with the resized ones.
                patch_embed = self.zero_conv(patch_embed).flatten(1)
                # Combine with resized embeddings
                resized_patch_embed = patch_embeds_masked[patch_size].unsqueeze(1)
                patch_embed = resized_patch_embed.squeeze(1) + patch_embed
                # Add positional embedding
                pos_embed = self.base_mini_pos_embed.repeat(batch_size, 1, 1)[:, 1:]
                patch_embed_postprocessed = patch_embed + pos_embed[mask.bool()]
            
            patch_embeds_postprocessed[patch_size] = patch_embed_postprocessed
                
        # Set up cls token.
        cls_token = self.cls_token.repeat(batch_size, 1, 1) + self.base_mini_pos_embed[:, 0:1]
        cls_token = cls_token.squeeze(1)
        return patch_embeds_postprocessed, cls_token

    def _assemble_output(self, masks, batch_size, x, patch_embeds_postprocessed, cls_token, cls_tok_loc, total_tokens):
        #### Masking rule ####
        # grp_tok = -2
        # cls_tok = -1
        # Not selected patches = 0
        # Size (Base) = 1
        # Size (Base*2) = 2
        # Size (Base*4) = 3
        # Size (Base*8) = 4
        # And so on.. 
        #### Masking rule ####

        masks_to_combine = []
            
        cls_to_combine = torch.full((batch_size, 1), -1, device=x.device)
        masks_to_combine.append(cls_to_combine)
        
        for i, patch_size in enumerate(self.patch_sizes):
            mask = masks[patch_size]
            masks_to_combine.append(mask * (i+1))
        
        combined_mask = torch.cat(masks_to_combine, dim=1)
        valid_patches = combined_mask[combined_mask != 0]
        
        mask_locs = {}
        for i, patch_size in enumerate(self.patch_sizes):
            mask_locs[patch_size] = torch.where(valid_patches == i+1)
        
        # Add the cls tokens in the right spots, and the embeddings everywhere else.
        # Total tokens already handles the class token, no need to add an extra 1.
        expanded_output = torch.zeros((total_tokens, self.embed_dim), device=x.device)
        expanded_output[cls_tok_loc] = cls_token
        
        for patch_size, locs in mask_locs.items():
            expanded_output[locs] = patch_embeds_postprocessed[patch_size]
        
        # Add a batch dim = 1.
        expanded_output = expanded_output.unsqueeze(0)
        
        return expanded_output

    def forward(self, x, entropy_maps):
        batch_size, h, w, masks, cls_tok_loc, total_tokens, patch_embed_base, seq_lengths = self._prepare_input(x, entropy_maps)
        patch_embeds_masked = self._compute_multi_scale_embeddings(x, masks, patch_embed_base)
        patch_embeds_postprocessed, cls_token = self._prepare_attention_output(patch_embeds_masked, masks, batch_size, h, w, patch_embed_base)
        expanded_output = self._assemble_output(masks, batch_size, x, patch_embeds_postprocessed, cls_token, cls_tok_loc, total_tokens)
        return expanded_output, seq_lengths, cls_tok_loc

class TokenizedZeroConvOldPatchAttn(nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        embed_dim: Optional[int] = None,
        num_scales: int = 2,
        thresholds: Optional[List[float]] = None,
        mode: Optional[str] = None,
        alpha_schedule: Optional[bool] = None):

        super().__init__()
        self.img_size = to_2tuple(image_size)[0]  # Assume square image
        self.base_patch_size = patch_size
        self.num_scales = num_scales
        self.patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
        self.thresholds = thresholds
        self.alpha_schedule = False
        self.embed_dim = embed_dim
        
        self.num_patches = (self.img_size // self.base_patch_size) ** 2 + 1 # Add 1 for the cls token

        attn_config = dict(
            dim=embed_dim,
            num_heads=4,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.1,
            proj_drop=0.1,
        )
        self.patch_attn = Attention(**attn_config)
        # Base pos embed for the patch. Will dynamically resample for other sizes.
        self.base_mini_pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * .02)

        # Zero conv for adding in attention.
        self.zero_conv = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        

    def forward(self, x, base_pos_embed, input_dict, rope=None):
        """
        base_pos_embed: base position embedding from main vit
        """
        batch_size = x.shape[0]
        output_mask = input_dict["output_mask"]
        base16 = input_dict["resized_patches_{}".format(self.base_patch_size)]
        posmask_16 = input_dict["pos_embed_mask_{}".format(self.base_patch_size)]

        # Get position embeddings for the base patch size
        pos_embed16 = base_pos_embed[:, 1:].repeat(batch_size, 1, 1)
        pos_embed16 = pos_embed16[posmask_16]

        # Get cls token position embedding
        cls_token_pos_embed = base_pos_embed[:, :1]

        # Process base scale (16)
        embed16 = self.patch_embed.forward_patch(base16) + pos_embed16
        cls_tok_loc = (output_mask == -1).nonzero().squeeze(1)
        expanded_outputs = torch.zeros((output_mask.shape[0], self.embed_dim), device=embed16.device, dtype=embed16.dtype)
        expanded_outputs[output_mask == -1] = self.cls_token + cls_token_pos_embed
        expanded_outputs[output_mask == 1] = embed16

        # Process larger scales (32 and 64) in a loop
        for scale_idx, cur_patch_size in enumerate(self.patch_sizes[1:]):
            base_patches = input_dict[f"resized_patches_{cur_patch_size}"]
            full_patches = input_dict[f"full_patches_{cur_patch_size}"]
            pos_embed_masks = input_dict[f"pos_embed_mask_{cur_patch_size}"]
            
            # Dynamically resample position embeddings for the current scale
            num_prefix_tokens = 1  # For cls token
            new_grid_size = self.img_size // cur_patch_size
            
            # Resample position embeddings for the current patch size
            resampled_pos_embed = resample_abs_pos_embed(
                base_pos_embed,
                new_size=(new_grid_size, new_grid_size),
                num_prefix_tokens=num_prefix_tokens,
            )
            
            # Extract and apply position embeddings
            pos_embed = resampled_pos_embed[:, 1:].repeat(batch_size, 1, 1)
            pos_embed = pos_embed[pos_embed_masks]
            
            # Dynamically re sample the mini pos for the patch attn.
            scale_factor = cur_patch_size // self.base_patch_size
            resampled_mini_pos_embed = resample_abs_pos_embed(
                    self.base_mini_pos_embed,
                    new_size=(scale_factor, scale_factor),
                    num_prefix_tokens=0,
                )
            
            # Check if there are any elements to pick (if pos_embed_masks has any 1s)
            if pos_embed_masks.sum() > 0:
                # Process base patches
                embed_scale = self.patch_embed.forward_patch(base_patches)
                
                # Process full patches
                num_mini_patches = (cur_patch_size // self.base_patch_size) ** 2
                full_patches = full_patches.view(-1, 3, self.base_patch_size, self.base_patch_size)
                full_patches = self.patch_embed.forward_patch(full_patches).view(-1, num_mini_patches, self.embed_dim)
                
                full_patches = full_patches + resampled_mini_pos_embed
                attn_scale = self.patch_attn(full_patches).mean(dim=1)
                #Add the pos embed at the END! not before
                embed_scale = embed_scale + pos_embed + self.zero_conv(attn_scale)
            else:
                # If no elements to pick, create dummy tensors and run all operations on zeros
                # Create a dummy tensor for patch_embed with fixed shape
                # Using shape [1, 3, patch_size, patch_size] for a single dummy patch
                dummy_base_patches = torch.zeros((1, 3, self.base_patch_size, self.base_patch_size), 
                                               device=x.device, dtype=embed16.dtype)
                dummy_embed = self.patch_embed.forward_patch(dummy_base_patches)
                # Create a dummy pos_embed of appropriate size
                dummy_pos_embed = torch.zeros_like(dummy_embed)
                embed_scale = dummy_embed + dummy_pos_embed
                
                # Create a dummy tensor for patch_attn
                num_mini_patches = (cur_patch_size // self.base_patch_size) ** 2
                dummy_patches = torch.zeros((1, num_mini_patches, self.embed_dim), 
                                          device=x.device, dtype=embed16.dtype)
                
                dummy_patches = dummy_patches + resampled_mini_pos_embed
                attn_scale = self.patch_attn(dummy_patches).mean(dim=1)
                embed_scale = self.zero_conv(attn_scale) + embed_scale
                
            # place in output based on size.
            expanded_outputs[output_mask == (scale_idx+2)] = embed_scale.float()

        expanded_outputs = expanded_outputs.unsqueeze(0).contiguous()

        return expanded_outputs, input_dict["attn_mask"], cls_tok_loc

class TokenizedZeroConvPatchAttn(nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        embed_dim: Optional[int] = None,
        num_scales: int = 2,
        thresholds: Optional[List[float]] = None,
        mode: Optional[str] = None,
        alpha_schedule: Optional[bool] = None):

        super().__init__()
        self.img_size = to_2tuple(image_size)[0]  # Assume square image
        self.base_patch_size = patch_size
        self.num_scales = num_scales
        self.patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
        self.thresholds = thresholds
        self.alpha_schedule = False
        self.embed_dim = embed_dim
        
        self.num_patches = (self.img_size // self.base_patch_size) ** 2 + 1 # Add 1 for the cls token

        self.patch_attn = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=2,
            stride=2
        )
        # Base pos embed for the patch. Will dynamically resample for other sizes.
        self.base_mini_pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * .02)

        # Zero conv for adding in attention.
        self.zero_conv = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        

    def forward(self, x, base_pos_embed, input_dict, ropes=None):
        """
        base_pos_embed: base position embedding from main vit
        """
        batch_size = x.shape[0]
        # Initial rope
       

        output_mask = input_dict["output_mask"]
        base16 = input_dict["resized_patches_{}".format(self.base_patch_size)]
        posmask_16 = input_dict["pos_embed_mask_{}".format(self.base_patch_size)]

        # Get position embeddings for the base patch size
        pos_embed16 = base_pos_embed[:, 1:].repeat(batch_size, 1, 1)
        pos_embed16 = pos_embed16[posmask_16]

        # Have to handle the ropes with a new expanded_outputs mask. 
        # We keep all the ole stuff and then remove the cls tok codes, 
        # leaving a rope of the correct size.
        expanded_rope_mask = None
        if ropes is not None:
            assert len(ropes) == len(self.patch_sizes)
            rope_base = ropes[0][posmask_16]
            expanded_rope_mask = torch.zeros(
                (output_mask.shape[0], rope_base.shape[-1]), 
                device=base_pos_embed.device, 
                dtype=base_pos_embed.dtype)
                
            expanded_rope_mask[output_mask == 1] = rope_base

        # Get cls token position embedding
        cls_token_pos_embed = base_pos_embed[:, :1]

        # Process base scale (16)
        embed16 = self.patch_embed.forward_patch(base16) + pos_embed16
        cls_tok_loc = (output_mask == -1).nonzero().squeeze(1)
        expanded_outputs = torch.zeros((output_mask.shape[0], self.embed_dim), device=embed16.device, dtype=embed16.dtype)
        expanded_outputs[output_mask == -1] = self.cls_token + cls_token_pos_embed
        expanded_outputs[output_mask == 1] = embed16

        # Process larger scales (32 and 64) in a loop
        for scale_idx, cur_patch_size in enumerate(self.patch_sizes[1:]):
            base_patches = input_dict[f"resized_patches_{cur_patch_size}"]
            full_patches = input_dict[f"full_patches_{cur_patch_size}"]
            pos_embed_masks = input_dict[f"pos_embed_mask_{cur_patch_size}"]
            
            # Dynamically resample position embeddings for the current scale
            num_prefix_tokens = 1  # For cls token
            base_grid_size = int(np.sqrt(len(base_pos_embed[0]) - num_prefix_tokens))
            new_grid_size = self.img_size // cur_patch_size
            
            # Resample position embeddings for the current patch size
            resampled_pos_embed = resample_abs_pos_embed(
                base_pos_embed,
                new_size=(new_grid_size, new_grid_size),
                old_size=(base_grid_size, base_grid_size),
                num_prefix_tokens=num_prefix_tokens,
            )
            
            # Extract and apply position embeddings
            pos_embed = resampled_pos_embed[:, 1:].repeat(batch_size, 1, 1)
            pos_embed = pos_embed[pos_embed_masks]

            if ropes is not None:
                rope = ropes[scale_idx+1][pos_embed_masks]
                # Code starts at 1, so have to add 2 to scale idx.
                expanded_rope_mask[output_mask == (scale_idx + 2)] = rope
                        
            # Check if there are any elements to pick (if pos_embed_masks has any 1s)
            if pos_embed_masks.sum() > 0:
                # Process base patches
                embed_scale = self.patch_embed.forward_patch(base_patches)
                
                # Process full patches
                n_patches = cur_patch_size // self.base_patch_size
                full_patches = full_patches.view(-1, 3, self.base_patch_size, self.base_patch_size)
                full_patches = self.patch_embed.forward_patch(full_patches).view(-1, n_patches, n_patches, self.embed_dim)
                full_patches = full_patches.permute(0, 3, 1, 2)
                
                # Resample position embeddings based on the scale factor
                # Apply twice for 64s
                for _ in range(scale_idx + 1):
                    full_patches = self.patch_attn(full_patches)
                attn_scale = full_patches.squeeze(-1).squeeze(-1)
                # Add the pos embed at the END! not before
                embed_scale = self.zero_conv(attn_scale) + embed_scale + pos_embed
            else:
                # If no elements to pick, create dummy tensors and run all operations on zeros
                # Process base patches
                dummy_base_patches = torch.zeros((1, 3, self.base_patch_size, self.base_patch_size), 
                                                 device=x.device, dtype=embed16.dtype)
                embed_scale = self.patch_embed.forward_patch(dummy_base_patches)
                
                # Process full patches
                dummy_full_patches = torch.zeros((1, 3, self.base_patch_size * 2, self.base_patch_size * 2), 
                                                 device=x.device, dtype=embed16.dtype)
                dummy_full_patches = self.patch_embed.forward_patch(dummy_full_patches.reshape(-1, 3, self.base_patch_size, self.base_patch_size))
                dummy_full_patches = dummy_full_patches.reshape(1, 2, 2, self.embed_dim)
                dummy_full_patches = dummy_full_patches.permute(0, 3, 1, 2)
                
                attn_scale = self.patch_attn(dummy_full_patches).squeeze(-1).squeeze(-1)
                dummy_pos_embed = torch.zeros_like(embed_scale)
                embed_scale = self.zero_conv(attn_scale) + embed_scale + dummy_pos_embed
                
            # place in output based on size.
            expanded_outputs[output_mask == (scale_idx+2)] = embed_scale.float()

        expanded_outputs = expanded_outputs.unsqueeze(0)
        output_rope = None
        if expanded_rope_mask is not None:
            output_rope = expanded_rope_mask[output_mask > 0].contiguous()

        return expanded_outputs, input_dict["attn_mask"], cls_tok_loc, output_rope
