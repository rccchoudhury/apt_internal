#!/usr/bin/env python
"""
Test script to demonstrate and compare different importance map computation methods
(entropy vs. Laplacian) in the PatchTokenizer.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.patch_tokenizer import PatchTokenizer
from src.models.entropy_utils import visualize_selected_patches_cv2

def normalize_map_for_display(map_tensor):
    """Normalize a map tensor to [0, 1] for visualization."""
    map_np = map_tensor.cpu().numpy()
    map_min, map_max = map_np.min(), map_np.max()
    return (map_np - map_min) / (map_max - map_min)

def main():
    # Define parameters
    img_size = 224
    patch_size = 16
    num_scales = 2
    thresholds = [0.5, 0.5]  # Thresholds for each scale
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    # Load a sample image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Replace with your image path
    image_path = "path/to/your/image.jpg"  
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}. Using a random image instead.")
        # Create a random image if no image is provided
        random_img = torch.randn(3, img_size, img_size)
        # Normalize to [0, 1] range for better visualization
        random_img = (random_img - random_img.min()) / (random_img.max() - random_img.min())
        # Apply normalization
        random_img = transforms.Normalize(mean=mean, std=std)(random_img)
        img_tensor = random_img.unsqueeze(0)  # Add batch dimension
    else:
        # Load and process the image
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Create tokenizers with different methods
    tokenizer_entropy = PatchTokenizer(
        num_scales=num_scales,
        base_patch_size=patch_size,
        image_size=img_size,
        thresholds=thresholds,
        mean=mean,
        std=std,
        method="entropy"
    )
    
    tokenizer_laplacian_mean = PatchTokenizer(
        num_scales=num_scales,
        base_patch_size=patch_size,
        image_size=img_size,
        thresholds=thresholds,
        mean=mean,
        std=std,
        method="laplacian",
        laplacian_aggregate="mean"
    )
    
    tokenizer_laplacian_max = PatchTokenizer(
        num_scales=num_scales,
        base_patch_size=patch_size,
        image_size=img_size,
        thresholds=thresholds,
        mean=mean,
        std=std,
        method="laplacian",
        laplacian_aggregate="max"
    )
    
    # Compute importance maps
    with torch.no_grad():
        maps_entropy = tokenizer_entropy.compute_importance_maps(img_tensor)
        maps_laplacian_mean = tokenizer_laplacian_mean.compute_importance_maps(img_tensor)
        maps_laplacian_max = tokenizer_laplacian_max.compute_importance_maps(img_tensor)
    
    # Create masks for visualization
    masks_entropy, _, _ = tokenizer_entropy.construct_masks(maps_entropy)
    masks_laplacian_mean, _, _ = tokenizer_laplacian_mean.construct_masks(maps_laplacian_mean)
    masks_laplacian_max, _, _ = tokenizer_laplacian_max.construct_masks(maps_laplacian_max)
    
    # Unnormalize the image for visualization
    unnorm = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    vis_img = unnorm(img_tensor[0])
    
    # Create visualizations
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    # Visualize the original image
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, 1)
    plt.imshow(vis_img.permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")
    plt.axis('off')
    
    # Visualize entropy maps
    for i, ps in enumerate(patch_sizes):
        plt.subplot(3, 3, i+2)
        plt.imshow(normalize_map_for_display(maps_entropy[ps][0]), cmap='viridis')
        plt.title(f"Entropy Map (Patch Size {ps})")
        plt.axis('off')
    
    # Visualize Laplacian mean maps
    for i, ps in enumerate(patch_sizes):
        plt.subplot(3, 3, i+4)
        plt.imshow(normalize_map_for_display(maps_laplacian_mean[ps][0]), cmap='viridis')
        plt.title(f"Laplacian Mean Map (Patch Size {ps})")
        plt.axis('off')
    
    # Visualize Laplacian max maps
    for i, ps in enumerate(patch_sizes):
        plt.subplot(3, 3, i+6)
        plt.imshow(normalize_map_for_display(maps_laplacian_max[ps][0]), cmap='viridis')
        plt.title(f"Laplacian Max Map (Patch Size {ps})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("importance_maps_comparison.png")
    print(f"Saved visualization to importance_maps_comparison.png")
    
    # Visualize selected patches
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Visualize patches selected by entropy
    vis_entropy = visualize_selected_patches_cv2(
        vis_img * 255.0, 
        masks_entropy, 
        patch_sizes
    )
    axes[0].imshow(np.array(vis_entropy))
    axes[0].set_title("Patches Selected by Entropy")
    axes[0].axis('off')
    
    # Visualize patches selected by Laplacian mean
    vis_laplacian_mean = visualize_selected_patches_cv2(
        vis_img * 255.0, 
        masks_laplacian_mean, 
        patch_sizes
    )
    axes[1].imshow(np.array(vis_laplacian_mean))
    axes[1].set_title("Patches Selected by Laplacian Mean")
    axes[1].axis('off')
    
    # Visualize patches selected by Laplacian max
    vis_laplacian_max = visualize_selected_patches_cv2(
        vis_img * 255.0, 
        masks_laplacian_max, 
        patch_sizes
    )
    axes[2].imshow(np.array(vis_laplacian_max))
    axes[2].set_title("Patches Selected by Laplacian Max")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("selected_patches_comparison.png")
    print(f"Saved patch visualization to selected_patches_comparison.png")

if __name__ == "__main__":
    main()
