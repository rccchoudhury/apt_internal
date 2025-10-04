"""Utilities for computing and visualizing patch-wise MSE in images using multi-scale reconstruction.

This module provides functions for:
1. Computing MSE maps for different patch sizes in images using multi-scale downsampling/upsampling
2. Selecting patches based on MSE thresholds for mixed-resolution processing
3. Visualizing the selected patches with different sizes

The main components are:
- compute_patch_mse_vectorized: Efficiently computes MSE for multiple patch sizes using reconstruction
- compute_patch_mse_batched: Batched version for multiple images
- select_patches_by_threshold: Selects patches of different sizes based on MSE threshold
- visualize_selected_patches_cv2: Visualizes the selected patches using OpenCV

The approach works by:
1. For each scale, downsample the image by factor 2^scale
2. Upsample back to original resolution
3. Compare patches of the reconstructed image with original patches
4. Compute MSE between original and reconstructed patches

These utilities are particularly useful for mixed-resolution image processing where
different regions of an image can be processed at different scales based on their
reconstruction quality (MSE).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict
from torchvision.transforms import functional as TF
import math
import cv2
from PIL import Image


def compute_patch_mse_vectorized(image, patch_size=16, num_scales=2, pad_value=1e6):
    """
    Compute MSE maps for multiple patch sizes in the input image using multi-scale reconstruction.
    
    For each scale:
    1. Downsample image by factor 2^scale
    2. Upsample back to original resolution
    3. Compare patches between original and reconstructed image
    4. Compute MSE for each patch
    
    Args:
        image: torch.Tensor of shape (C, H, W) or (H, W) with values in range [0, 255]
        patch_size: base patch size (default: 16)
        num_scales: number of scales to compute (default: 2)
        pad_value: high MSE value to pad incomplete patches with (default: 1e6)
    
    Returns:
        mse_maps: dict containing torch.Tensor MSE maps for each patch size
    """
    if len(image.shape) == 3:
        # Convert to grayscale if image is RGB
        if image.shape[0] == 3:
            image = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        else:
            image = image[0]
    
    mse_maps = {}
    H, W = image.shape
    
    # Generate patch sizes for different scales
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    for scale_idx, current_patch_size in enumerate(patch_sizes):
        # Downsample factor for this scale
        downsample_factor = 2 ** scale_idx
        
        # Downsample the image
        if downsample_factor > 1:
            # Add batch and channel dimensions for F.interpolate
            image_4d = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            downsampled = F.interpolate(
                image_4d, 
                size=(H // downsample_factor, W // downsample_factor),
                mode='bilinear',
                align_corners=False
            )
            
            # Upsample back to original resolution
            reconstructed = F.interpolate(
                downsampled,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            reconstructed = reconstructed.squeeze(0).squeeze(0)  # Back to (H, W)
        else:
            # Scale 0: no downsampling, perfect reconstruction
            reconstructed = image.clone()
        
        # Compute patch-wise MSE
        num_patches_h = (H + current_patch_size - 1) // current_patch_size
        num_patches_w = (W + current_patch_size - 1) // current_patch_size
        
        # Pad images to ensure they fit into patches cleanly
        pad_h = num_patches_h * current_patch_size - H
        pad_w = num_patches_w * current_patch_size - W
        
        padded_original = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_reconstructed = F.pad(reconstructed, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Unfold both images into patches
        original_patches = padded_original.unfold(0, current_patch_size, current_patch_size).unfold(1, current_patch_size, current_patch_size)
        reconstructed_patches = padded_reconstructed.unfold(0, current_patch_size, current_patch_size).unfold(1, current_patch_size, current_patch_size)
        
        # Reshape to (num_patches, patch_size, patch_size)
        original_patches = original_patches.reshape(num_patches_h * num_patches_w, current_patch_size, current_patch_size)
        reconstructed_patches = reconstructed_patches.reshape(num_patches_h * num_patches_w, current_patch_size, current_patch_size)
        
        # Compute MSE for each patch
        mse_per_patch = torch.mean((original_patches - reconstructed_patches) ** 2, dim=(1, 2))
        
        # Reshape back to spatial dimensions
        mse_map = mse_per_patch.reshape(num_patches_h, num_patches_w)
        
        # Assign high MSE value to padded regions
        if pad_h > 0:
            mse_map[-1, :] = pad_value
        if pad_w > 0:
            mse_map[:, -1] = pad_value
            
        mse_maps[current_patch_size] = mse_map
    
    return mse_maps


def compute_patch_mse_batched(images, patch_size=16, num_scales=2, pad_value=1e6):
    """
    Compute MSE maps for multiple patch sizes in a batch of images using multi-scale reconstruction.
    
    Args:
        images: torch.Tensor of shape (B, C, H, W) with values in range [0, 255]
        patch_size: base patch size (default: 16)
        num_scales: number of scales to compute (default: 2)
        pad_value: high MSE value to pad incomplete patches with (default: 1e6)
    
    Returns:
        batch_mse_maps: dict mapping patch sizes to torch.Tensor MSE maps
                        with shape (B, H_p, W_p) where H_p and W_p depend on the patch size
    """
    B, C, H, W = images.shape
    
    # Convert to grayscale if needed
    if C == 3:
        # RGB to grayscale conversion
        grayscale_images = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]
    elif C == 1:
        grayscale_images = images[:, 0]
    else:
        grayscale_images = images[:, 0]  # Take first channel
    
    batch_mse_maps = {}
    
    # Generate patch sizes for different scales
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    for scale_idx, current_patch_size in enumerate(patch_sizes):
        # Downsample factor for this scale
        downsample_factor = 2 ** scale_idx
        
        # Downsample the batch of images
        if downsample_factor > 1:
            # Add channel dimension for F.interpolate
            images_4d = grayscale_images.unsqueeze(1)  # (B, 1, H, W)
            downsampled = F.interpolate(
                images_4d,
                size=(H // downsample_factor, W // downsample_factor),
                mode='bilinear',
                align_corners=False
            )
            
            # Upsample back to original resolution
            reconstructed = F.interpolate(
                downsampled,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            reconstructed = reconstructed.squeeze(1)  # Back to (B, H, W)
        else:
            # Scale 0: no downsampling, perfect reconstruction
            reconstructed = grayscale_images.clone()
        
        # Compute patch-wise MSE for the batch
        num_patches_h = (H + current_patch_size - 1) // current_patch_size
        num_patches_w = (W + current_patch_size - 1) // current_patch_size
        
        # Pad images to ensure they fit into patches cleanly
        pad_h = num_patches_h * current_patch_size - H
        pad_w = num_patches_w * current_patch_size - W
        
        padded_original = F.pad(grayscale_images, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_reconstructed = F.pad(reconstructed, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Unfold both batches into patches
        # Shape after unfold: (B, num_patches_h, num_patches_w, patch_size, patch_size)
        original_patches = padded_original.unfold(1, current_patch_size, current_patch_size).unfold(2, current_patch_size, current_patch_size)
        reconstructed_patches = padded_reconstructed.unfold(1, current_patch_size, current_patch_size).unfold(2, current_patch_size, current_patch_size)
        
        # Compute MSE for each patch across the batch
        # Shape: (B, num_patches_h, num_patches_w)
        mse_per_patch = torch.mean((original_patches - reconstructed_patches) ** 2, dim=(3, 4))
        
        # Assign high MSE value to padded regions
        if pad_h > 0:
            mse_per_patch[:, -1, :] = pad_value
        if pad_w > 0:
            mse_per_patch[:, :, -1] = pad_value
            
        batch_mse_maps[current_patch_size] = mse_per_patch
    
    return batch_mse_maps


def select_patches_by_threshold(mse_maps: Dict[int, torch.Tensor], thresholds: Dict[int, float], alpha: float = 1.0):
    """
    Vectorized version of patch selection based on MSE thresholds.
    
    Args:
        mse_maps (dict): Contains patch sizes as keys mapping to
            torch.Tensor MSE maps of shape (B, H_p, W_p) where
            H_p and W_p depend on the patch size
        thresholds (dict): Contains patch sizes as keys mapping to
            threshold values for MSE-based selection
        alpha (float): Scaling factor for the MSE threshold
        
    Returns:
        masks (dict): Dictionary mapping patch sizes to their 0/1 masks
    """
    masks = {}
    
    for patch_size, mse_map in mse_maps.items():
        if patch_size in thresholds:
            threshold = thresholds[patch_size] * alpha
            # Select patches where MSE is above threshold (high reconstruction error)
            # This indicates regions that benefit from higher resolution processing
            mask = (mse_map > threshold).float()
            masks[patch_size] = mask
        else:
            # If no threshold specified, select all patches
            masks[patch_size] = torch.ones_like(mse_map)
    
    return masks


def visualize_selected_patches_cv2(
    image_tensor, 
    masks, 
    patch_sizes,
    color=(255, 255, 255),  # BGR in OpenCV, but white is the same in BGR or RGB
    thickness=1
):
    """
    Draw rectangles (using cv2) where masks are 1 for patches of different sizes,
    then return a PIL Image.

    Args:
        image_tensor   (torch.Tensor): Grayscale or RGB image of shape (H, W) or (C, H, W).
        masks          (Dict[int, torch.Tensor]): Dictionary mapping patch sizes to 0/1 masks.
        patch_sizes    (List[int]): List of patch sizes corresponding to each mask.
        color          (tuple): BGR color for rectangle outlines (default white).
        thickness      (int): Thickness of the rectangle outline.

    Returns:
        annotated_image_pil (PIL.Image): The original image with drawn rectangles (in white).
    """

    # 1. Convert the image tensor to a NumPy array for OpenCV
    if image_tensor.ndim == 3:
        # If image_tensor is (C, H, W) with channels first
        if image_tensor.shape[0] in [1, 3]:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            # Already (H, W, C)
            image_np = image_tensor.cpu().numpy()
    else:
        # (H, W) -> expand dimension for single-channel (H, W, 1)
        image_np = image_tensor.cpu().numpy()
        # Expand to 3 channels to draw colored rectangles
        image_np = np.stack([image_np]*3, axis=-1)

    # Convert to uint8 if needed
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    # Get full image dimensions
    img_h, img_w = image_np.shape[:2]
    
    # 2. Create a copy of the image to draw on
    annotated_np = image_np.copy()
    
    # 3. Draw rectangles for each patch size
    for patch_size in patch_sizes:
        if patch_size in masks:
            mask = masks[patch_size]
            
            # Handle batch dimension if present
            if mask.ndim == 3:
                mask = mask[0]  # Take first image in batch
            
            H, W = mask.shape
            for i in range(H):
                for j in range(W):
                    if mask[i, j] == 1:
                        # Calculate patch coordinates
                        y1 = i * patch_size
                        x1 = j * patch_size
                        y2 = min(y1 + patch_size, img_h)  # Ensure we don't go beyond image bounds
                        x2 = min(x1 + patch_size, img_w)
                        
                        # Draw rectangle
                        cv2.rectangle(annotated_np, (x1, y1), (x2, y2), color, thickness)
    
    # 4. Convert back to PIL image
    annotated_image_pil = Image.fromarray(annotated_np)
    
    return annotated_image_pil


def visualize_selected_patches_cv2_non_overlapping(
    image_tensor, 
    masks, 
    patch_sizes,
    color=(255, 255, 255),  # BGR in OpenCV, but white is the same in BGR or RGB
    thickness=1
):
    """
    Draw rectangles (using cv2) where masks are 1 for patches of different sizes,
    avoiding overlapping boundaries to prevent thick lines.

    Args:
        image_tensor   (torch.Tensor): Grayscale or RGB image of shape (H, W) or (C, H, W).
        masks          (Dict[int, torch.Tensor]): Dictionary mapping patch sizes to 0/1 masks.
        patch_sizes    (List[int]): List of patch sizes corresponding to each mask.
        color          (tuple): BGR color for rectangle outlines (default white).
        thickness      (int): Thickness of the rectangle outline.

    Returns:
        annotated_image_pil (PIL.Image): The original image with drawn rectangles (in white).
    """

    # 1. Convert the image tensor to a NumPy array for OpenCV
    if image_tensor.ndim == 3:
        # If image_tensor is (C, H, W) with channels first
        if image_tensor.shape[0] in [1, 3]:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            # Already (H, W, C)
            image_np = image_tensor.cpu().numpy()
    else:
        # (H, W) -> expand dimension for single-channel (H, W, 1)
        image_np = image_tensor.cpu().numpy()
        # Expand to 3 channels to draw colored rectangles
        image_np = np.stack([image_np]*3, axis=-1)

    # Convert to uint8 if needed
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    # Get full image dimensions
    img_h, img_w = image_np.shape[:2]
    
    # 2. Create a set to track which edges have been drawn
    # We'll use (y1, x1, y2, x2) tuples to represent line segments
    drawn_edges = set()
    
    # 3. Create a copy of the image to draw on
    annotated_np = image_np.copy()
    
    # Process masks from largest to smallest patch size to handle hierarchy
    for patch_size in sorted(patch_sizes, reverse=True):
        if patch_size in masks:
            mask = masks[patch_size]
            
            # Handle batch dimension if present
            if mask.ndim == 3:
                mask = mask[0]  # Take first image in batch
            
            H, W = mask.shape
            for i in range(H):
                for j in range(W):
                    if mask[i, j] == 1:
                        # Calculate patch coordinates
                        y1 = i * patch_size
                        x1 = j * patch_size
                        y2 = min(y1 + patch_size, img_h)  # Ensure we don't go beyond image bounds
                        x2 = min(x1 + patch_size, img_w)
                        
                        # Draw top edge if not already drawn
                        if (y1, x1, y1, x2) not in drawn_edges:
                            cv2.line(annotated_np, (x1, y1), (x2, y1), color, thickness)
                            drawn_edges.add((y1, x1, y1, x2))
                        
                        # Draw bottom edge if not already drawn
                        if (y2, x1, y2, x2) not in drawn_edges:
                            cv2.line(annotated_np, (x1, y2), (x2, y2), color, thickness)
                            drawn_edges.add((y2, x1, y2, x2))
                        
                        # Draw left edge if not already drawn
                        if (y1, x1, y2, x1) not in drawn_edges:
                            cv2.line(annotated_np, (x1, y1), (x1, y2), color, thickness)
                            drawn_edges.add((y1, x1, y2, x1))
                        
                        # Draw right edge if not already drawn
                        if (y1, x2, y2, x2) not in drawn_edges:
                            cv2.line(annotated_np, (x2, y1), (x2, y2), color, thickness)
                            drawn_edges.add((y1, x2, y2, x2))
    
    # 4. Convert back to PIL image
    annotated_image_pil = Image.fromarray(annotated_np)
    
    return annotated_image_pil


if __name__ == '__main__':
    # Example usage and testing
    pass
