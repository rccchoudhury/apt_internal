""" EVA

EVA from https://github.com/baaivision/EVA , paper: https://arxiv.org/abs/2211.07636

@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang,
  Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}

EVA-02: A Visual Representation for Neon Genesis - https://arxiv.org/abs/2303.11331
@article{EVA02,
  title={EVA-02: A Visual Representation for Neon Genesis},
  author={Fang, Yuxin and Sun, Quan and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2303.11331},
  year={2023}
}

This file contains EVA & EVA02 model implementations evolved from BEiT, additional models in vision_transformer.py.

Modifications by / Copyright 2023 Ross Wightman, original copyrights below
"""
# EVA models Copyright (c) 2022 BAAI-Vision
# EVA02 models Copyright (c) 2023 BAAI-Vision
import math
from typing import Callable, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, RotaryEmbeddingCat, \
    apply_keep_indices_nlc, trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    to_2tuple, use_fused_attn

from src.models.patch_embed import PatchEmbed

__all__ = ['Eva']

def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed_cat(x: torch.Tensor, emb):
    # Now, assumes x has been flattened first. 
    # x shape is (B, n_heads, seqlen, emb_dim / n_heads)
    n_heads, B, emb_dim = x.shape
    x = x.permute(1, 0, 2)
    # Shape should be (B, n_heads, seqlen, emb_dim / n_heads)
    # Emb shape should be (seqlen, emb_dim / n_heads)
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    if sin_emb.ndim == 3:
        unsq_cos = cos_emb.unsqueeze(1).expand_as(x)
        unsq_sin = sin_emb.unsqueeze(1).expand_as(x)
        return x * unsq_cos + rot(x) * unsq_sin

    x =  x * cos_emb + rot(x) * sin_emb
    x = x.permute(1, 0, 2)
    return x

class EvaAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            qkv_bias_separate: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()
        self.qkv_bias_separate = qkv_bias_separate

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            class_mask: Optional[torch.Tensor] = None,
    ):

        B, batch_times_seqlen, C = x.shape
        assert B == 1
        assert self.qkv is None

        q = self.q_proj(x).reshape(1, batch_times_seqlen, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
        k = self.k_proj(x).reshape(1, batch_times_seqlen, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).reshape(1, batch_times_seqlen, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            # q shape is (B, n_heads, seqlen, -1). 
            # reshape to (1, B*n_heads, seqlen, -1)
            q = q.permute(1, 0, 2, 3).reshape(self.num_heads, batch_times_seqlen, -1).permute(1, 0, 2).unsqueeze(0)
            k = k.permute(1, 0, 2, 3).reshape(self.num_heads, batch_times_seqlen, -1).permute(1, 0, 2).unsqueeze(0)

            # Select the "non-class tokens", embed them and put them back.
            class_mask = class_mask.bool()
            q_to_embed = q[~class_mask]
            q_to_embed = apply_rot_embed_cat(q_to_embed, rope)
            q[~class_mask] = q_to_embed.type_as(v)

            k_to_embed = k[~class_mask]
            k_to_embed = apply_rot_embed_cat(k_to_embed, rope)
            k[~class_mask] = k_to_embed.type_as(v)

        if self.fused_attn:
            # Flatten the Q after rot has been applie.
            #q = q.transpose(1, 2).reshape(1, B * N, self.num_heads, -1)
            #k = k.transpose(1, 2).reshape(1, B * N, self.num_heads, -1)
            v = v.transpose(1, 2).reshape(1, batch_times_seqlen, self.num_heads, -1)

            x = fmha.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_mask,
                p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )

        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1)
            
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2)
        
        x = x.reshape(1, batch_times_seqlen, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, 
        x, 
        rope: Optional[torch.Tensor] = None, 
        attn_mask: Optional[torch.Tensor] = None, 
        class_mask: Optional[torch.Tensor] = None):

        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, class_mask=class_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, class_mask=class_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class EvaBlockPostNorm(nn.Module):
    """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,  # ignore for post-norm
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            num_reg_tokens: int = 0,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
            num_scales: Optional[int] = None,
            thresholds: Optional[float] = None,
            mixed_patch_embed: Optional[Callable] = None,
           
    ):
        """

        Args:
            img_size:
            patch_size:
            in_chans:
            num_classes:
            global_pool:
            embed_dim:
            depth:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            drop_rate:
            pos_drop_rate:
            proj_drop_rate:
            attn_drop_rate:
            drop_path_rate:
            norm_layer:
            init_values:
            class_token:
            use_abs_pos_emb:
            use_rot_pos_emb:
            use_post_norm:
            ref_feat_shape:
            head_init_scale:
        """
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None

        if mixed_patch_embed is not None:
            self.mixed_patch = mixed_patch_embed(
                embed_dim=embed_dim,
            )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                dim=embed_dim // num_heads,
                in_pixels=False,
                feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                ref_feat_shape=ref_feat_shape,
            )
            # make the patch embeds, downscaling for each scale.
            self.rope_28 = RotaryEmbeddingCat(
                dim=embed_dim // num_heads,
                in_pixels=False,
                feat_shape=(16, 16),
                ref_feat_shape=ref_feat_shape,
            )

            self.rope_56 = RotaryEmbeddingCat(
                dim=embed_dim // num_heads,
                in_pixels=False,
                feat_shape=(8, 8),
                ref_feat_shape=ref_feat_shape,
            )

            # add these to a paramterdict
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                num_prefix_tokens=self.num_prefix_tokens,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def init_multiscale_patch_embed(self):
        """
        Set up multiscale rope embeds and set the patch to mixed_patch.
        """
        print("Initiaizling EVA multiscale patch embed!")
        self.mixed_patch.patch_embed = self.patch_embed
        self.mixed_patch.cls_token = self.cls_token

        # Default scale (32×32)
        self.default_embed = self.rope.get_embed() if self.rope is not None else None
        
        # For 28×28 scale
        scale_28_embed = None
        if self.rope_28 is not None:
            # Get the bands and recompute embeddings for 28×28
            self.scale_28_embed = self.rope_28.get_embed() 
        
        # For 56×56 scale
        scale_56_embed = None
        if self.rope_56 is not None:
            # Get the bands and recompute embeddings for 56×56
            self.scale_56_embed = self.rope_56.get_embed()


    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x, attn_mask, class_mask, rope):
        # Run the forward features. Assume the Rope is fully taken care of.
        
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rope, attn_mask=attn_mask, class_mask=class_mask)
        x = self.norm(x)
        
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, input_dict):
        #ipdb.set_trace()
        #print("Pos embed 16 shape: ", input_dict['pos_embed_mask_16'].shape)
        # Hnalde this smoother
        if self.default_embed.device != x.device:
            self.default_embed = self.default_embed.to(x.device).expand(x.shape[0], -1, -1)
            self.scale_28_embed = self.scale_28_embed.to(x.device).expand(x.shape[0], -1, -1)
            self.scale_56_embed = self.scale_56_embed.to(x.device).expand(x.shape[0], -1, -1)

        if self.mixed_patch is not None:
            B = x.shape[0]
            # Should be able to hanlde ropes in here too - they don't change between patches!
            x, attn_mask, cls_token_indices, rope = self.mixed_patch(x, 
                self.pos_embed, 
                input_dict, 
                ropes=[self.default_embed, self.scale_28_embed, self.scale_56_embed])
            class_mask = torch.zeros(x.shape[:2], device=x.device)
            class_mask[:, cls_token_indices] = 1
            # prefix tokens aren't in here yet.
            x = self.pos_drop(x)

            x = self.forward_features(x, attn_mask=attn_mask, class_mask=class_mask, rope=rope)
            if self.global_pool == 'token':
                x = x[0, cls_token_indices]
            else:
                assert x.shape[0] == 1
                seqlens = input_dict['seqlens']
                splits = [seq[:, self.num_prefix_tokens:].mean(1) for seq in torch.split(x, seqlens, dim=1)]
                x = torch.vstack(splits)
        else:
            x = self.patch_embed(x)
            B, N = x.shape[:2]
            # prefix tokens aren't in here yet.
            block_mask = BlockDiagonalMask.from_seqlens([N + self.num_prefix_tokens for _ in range(B)])
           
            x = self.forward_features(x, attn_mask=block_mask)
            if self.global_pool:
                x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        
        x = self.forward_head(x)
        return x
