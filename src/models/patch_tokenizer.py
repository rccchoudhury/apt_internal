"""
Stand-alone module that computes the necessary inputs for the 
patch embedding. The goal is to compute only the parts 
that are needed, nothing more, and place them in a dictionary 
for the patch embedding to use. 
"""
from typing import Dict, List, Tuple, Union
import ipdb
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import math
from src.models.entropy_utils import (
    select_patches_by_threshold,
    visualize_selected_patches_cv2,
    compute_patch_entropy_vectorized,
    compute_patch_entropy_batched,
    compute_patch_laplacian_vectorized,
    compute_patch_laplacian_batched,
    compute_patch_mse_batched
)


class PatchTokenizer(nn.Module):
    """Tokenizer for mixed-resolution patches.
    
    Args:
        num_scales (int): Number of scales to use
        base_patch_size (int): Base patch size
        image_size (int): Image size
        thresholds (List[float]): Entropy thresholds for patch selection at each scale
        mean (List[float]): Mean values for normalization
        std (List[float]): Standard deviation values for normalization
        method (str): Method to use for computing patch importance maps ('entropy' or 'laplacian')
        laplacian_aggregate (str): Method to aggregate Laplacian values ('mean', 'max', or 'std')
    """
    def __init__(
        self,
        num_scales: int,
        base_patch_size: int,
        image_size: int,
        thresholds: List[float],
        mean: List[float],
        std: List[float],
        method: str = 'entropy',
        laplacian_aggregate: str = 'mean',
    ):
        super().__init__()
        self.num_scales = num_scales
        self.base_patch_size = base_patch_size
        self.image_size = image_size
        self.thresholds = thresholds
        self.method = method
        self.laplacian_aggregate = laplacian_aggregate

        self.pos_embed16: Union[torch.Tensor, None] = None
        self.pos_embed32: Union[torch.Tensor, None] = None

        self.norm = transforms.Normalize(mean=mean, std=std)
        self.unnorm = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )


    def construct_masks(
        self,
        importance_maps: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[int]]:
        """Constructs selection masks for patches based on importance maps.
        
        Args:
            importance_maps (Dict[int, torch.Tensor]): Dictionary mapping patch sizes to their
                importance maps. Shape of each map: (batch_size, height, width)
        
        Returns:
            Tuple containing:
                - masks (Dict[int, torch.Tensor]): Dictionary of selection masks for each
                  patch size
                - output_mask (torch.Tensor): Flattened mask with scale indicators
                  (-1 for class token)
                - seqlens (List[int]): Sequence lengths for each batch item
        """
        masks = select_patches_by_threshold(importance_maps, thresholds=self.thresholds)
        batch_size = masks[self.base_patch_size].shape[0]
        all_masks = []
        output_dict = {}
        
        # Set up output mask with class token (-1)
        device = importance_maps[self.base_patch_size].device
        temp_masks = [torch.ones((batch_size, 1), device=device) * -1]
        seqlens = torch.ones((batch_size), device=device)
        
        for idx in range(0, self.num_scales):
            cur_patch_size = self.base_patch_size * 2 ** idx
            temp_mask = masks[cur_patch_size].flatten(1)
            seqlens += temp_mask.sum(1)
            temp_masks.append(temp_mask * (idx + 1))

        output_mask = torch.cat(temp_masks, dim=1)
        output_mask = output_mask[output_mask != 0]
        seqlens = seqlens.int().tolist()

        return masks, output_mask, seqlens

    def construct_patch_groups(
        self,
        images: torch.Tensor,
        masks: Dict[int, torch.Tensor],
        pos_embeds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Constructs groups of patches at different scales with their position embeddings.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, channels, height, width)
            masks (Dict[int, torch.Tensor]): Selection masks for each patch size
            pos_embeds (Dict[str, torch.Tensor]): Position embeddings for each patch size
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - full_patches_{size}: Original resolution patches for each size
                - resized_patches_{size}: Downsampled patches for each size
                - pos_embed_{size}: Position embeddings for selected patches
                - pos_embed_cls_token: Position embedding for class token
        """
        output_dict = {}
        B = images.shape[0]

        for idx in range(0, self.num_scales):
            cur_patch_size = self.base_patch_size * 2 ** idx
            cur_mask = masks[cur_patch_size].bool()
            # cur_pos_embed = pos_embeds[str(cur_patch_size)]
            # cur_pos_embed = cur_pos_embed[:, 1:].repeat(B, 1, 1)
            
            scale_img = images
            if idx > 0:
                scale_img = F.interpolate(
                    scale_img,
                    scale_factor=0.5 ** idx,
                    mode="bilinear"
                )

                constituent_patches = einops.rearrange(
                    images,
                    "b c (h n1 p3) (w n2 p4) -> b h w (n1 n2) c p3 p4",
                    h=self.image_size // cur_patch_size,
                    w=self.image_size // cur_patch_size,
                    n1=cur_patch_size // self.base_patch_size,
                    n2=cur_patch_size // self.base_patch_size,
                    p3=self.base_patch_size,
                    p4=self.base_patch_size
                )
                selected_constituent_patches = constituent_patches[cur_mask]
                output_dict[f"full_patches_{cur_patch_size}"] = selected_constituent_patches

            scaled_patches = einops.rearrange(
                scale_img, 
                "b c (h p1) (w p2) -> b h w c p1 p2",
                p1=self.base_patch_size,
                p2=self.base_patch_size
            )
        
            selected_patches = scaled_patches[masks[cur_patch_size].bool()]
            output_dict[f"resized_patches_{cur_patch_size}"] = selected_patches
            flat_mask = masks[cur_patch_size].flatten(1).bool()
            output_dict[f"pos_embed_mask_{cur_patch_size}"] = flat_mask

        #output_dict["pos_embed_cls_token"] = pos_embeds[str(self.base_patch_size)][:, 0]
        return output_dict

    def compute_importance_maps(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Compute importance maps for images after unnormalizing them.
        Uses either entropy or Laplacian method based on the tokenizer configuration.
        
        Args:
            images: Normalized images tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary mapping patch sizes to importance maps with shape (B, H_p, W_p)
        """
        # Unnormalize the images using vectorized operations
        with torch.no_grad():
            # Apply unnormalization directly to the batch
            unnormalized_images = self.unnorm(images)
            
            # Scale to [0, 255] range for computation
            unnormalized_images = torch.clamp(unnormalized_images * 255.0, 0, 255)
            
            # Compute maps based on selected method
            if self.method == 'entropy':
                # Compute entropy maps for the entire batch
                batch_maps = compute_patch_entropy_batched(
                    unnormalized_images, 
                    patch_size=self.base_patch_size, 
                    num_scales=self.num_scales
                )
            elif self.method == 'laplacian':
                # Compute Laplacian maps for the entire batch
                batch_maps = compute_patch_laplacian_batched(
                    unnormalized_images, 
                    patch_size=self.base_patch_size, 
                    num_scales=self.num_scales,
                    aggregate=self.laplacian_aggregate
                )
            elif self.method == 'upsample_mse':
                batch_maps = compute_patch_mse_batched(
                    unnormalized_images, 
                    patch_size=self.base_patch_size, 
                    num_scales=self.num_scales,
                    scale_factors=[1.0, 0.5, 0.25],
                    aggregate='mean'
                )
            else:
                raise ValueError(f"Unknown method: {self.method}. Choose 'entropy' or 'laplacian'")
            
        return batch_maps

    def forward(
        self,
        images: torch.Tensor,
        importance_maps: Dict[int, torch.Tensor] = None,
        pos_embeds: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, List[int]]]:
        """Forward pass of the patch tokenizer.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, channels, height, width)
            importance_maps (Dict[int, torch.Tensor]): Pre-computed importance maps for different patch sizes
            pos_embeds (Dict[str, torch.Tensor]): Position embeddings for different patch sizes
            
        Returns:
            Dictionary containing:
                - All outputs from construct_patch_groups
                - output_mask: Mask indicating patch scales
                - seqlens: Sequence lengths for each batch item
        """
        B, C, H, W = images.shape
        max_patches = B * H * W / (self.base_patch_size ** 2)
        
        # If importance maps are not provided, compute them

        # PRECOMPTUTE
        if self.method != "entropy":
            importance_maps = self.compute_importance_maps(images)

        masks, output_mask, seqlens = self.construct_masks(importance_maps)
        output_dict = self.construct_patch_groups(images, masks, pos_embeds)
        output_dict["output_mask"] = output_mask
        output_dict["seqlens"] = seqlens
        
        # Compute cu_seqlens and max_seqlen for flash attention varlen
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=images.device),
                                torch.tensor(seqlens, dtype=torch.int32, device=images.device).cumsum(0)])
        output_dict["cu_seqlens"] = cu_seqlens
        output_dict["max_seqlen"] = max(seqlens)

        retained_patches = sum(seqlens)
        output_dict["retained_frac"] = retained_patches / max_patches

        return output_dict