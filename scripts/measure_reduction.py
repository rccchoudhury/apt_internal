"""
Measures patch reduction using entropy-based adaptive patching.
"""
import sys
sys.path.append("..")
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from src.data.transforms import transforms_imagenet_eval, ImageFolderWithEntropy
from src.models.entropy_utils import select_patches_by_threshold
import torch.distributed as dist

def count_patches(mask32, mask16):
    """
    Count patches based on entropy masks.
    For each position where mask32=1, we use a 32x32 patch (counts as 1)
    For each position where mask16=1, we use a 16x16 patch (counts as 1)
    """
    num_32_patches = torch.sum(mask32).item()
    num_16_patches = torch.sum(mask16).item()
    return num_32_patches + num_16_patches

def main(args):
    # Set up data loading
    pre_transform, post_transform = transforms_imagenet_eval(img_size=args.image_size, pre_post_divide=True)
    data_val = ImageFolderWithEntropy(
        args.data_dir, 
        transform=(pre_transform, post_transform),
        patch_sizes=[16, 32]
    )
    
    # Generate random indices
    total_images = len(data_val)
    indices = torch.randperm(total_images)[:args.num_images].tolist()
    
    total_new_patches = 0
    total_orig_patches = 0
    
    # Calculate original number of patches for a single image (using all 16x16 patches)
    patches_per_side = args.image_size // 16  # Using 16x16 as base patch size
    orig_patches_per_image = patches_per_side * patches_per_side
    
    # Process images
    with tqdm(indices, desc="Processing images") as pbar:
        for idx in pbar:
            img, _, entropy_maps = data_val[idx]
            
            # Convert single image to batch
            entropy_maps = {k: v.unsqueeze(0) for k, v in entropy_maps.items()}
            
            # Get patch masks using entropy threshold
            mask32, mask16 = select_patches_by_threshold(entropy_maps, args.threshold)
            
            # Count patches after adaptive splitting
            num_patches = count_patches(mask32, mask16)
            
            total_new_patches += num_patches
            total_orig_patches += orig_patches_per_image
            
            # Update progress bar with current reduction
            reduction = 1.0 - (total_new_patches / total_orig_patches)
            pbar.set_postfix({"reduction": f"{reduction:.2%}"})
    
    # Print final statistics
    print(f"\nProcessed {args.num_images} images")
    print(f"Original patches: {total_orig_patches}")
    print(f"New patches: {total_new_patches}")
    print(f"Reduction: {1.0 - (total_new_patches / total_orig_patches):.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure patch reduction using entropy-based adaptive patching")
    parser.add_argument("--data_dir", type=str, default="/edrive1/rchoudhu/ILSVRC2012/val",
                      help="Path to ImageNet validation directory")
    parser.add_argument("--image_size", type=int, default=384,
                      help="Size of input images")
    parser.add_argument("--num_images", type=int, default=1000,
                      help="Number of images to process")
    parser.add_argument("--threshold", type=float, default=6.0,
                      help="Entropy threshold for patch selection")
    
    args = parser.parse_args()
    main(args)