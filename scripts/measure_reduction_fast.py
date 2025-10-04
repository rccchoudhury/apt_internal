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

def main(args):
    # Set up data loading
    pre_transform, post_transform = transforms_imagenet_eval(img_size=args.image_size, pre_post_divide=True)
    data_val = ImageFolderWithEntropy(
        args.data_dir, 
        transform=(pre_transform, post_transform),
        patch_size=16,
        num_scales=args.num_scales
    )
    
    # Generate random indices
    total_images = len(data_val)
    indices = torch.randperm(total_images)[:args.num_images].tolist()
    
    total_new_patches = 0
    total_orig_patches = 0
    
    # Calculate original number of patches for a single image (using all 16x16 patches)
    patches_per_side = args.image_size // 16  # Using 16x16 as base patch size
    orig_patches_per_image = patches_per_side * patches_per_side
    
    print("====================")
    print("Image size:", args.image_size)
    
    # Process images
    with tqdm(indices, desc="Processing images") as pbar:
        total_new_patches = {5.0: 0, 5.25: 0, 5.5: 0, 5.75: 0, 6.0: 0, 6.25: 0, 6.5: 0, 6.75: 0, 7.0: 0, 7.25: 0, 7.5: 0, 7.75: 0, 8.0: 0}
        total_orig_patches = 0
        
        for idx in pbar:
            img, _, entropy_maps = data_val[idx]
            total_orig_patches += orig_patches_per_image
            
            # Convert single image to batch
            entropy_maps = {k: v.unsqueeze(0) for k, v in entropy_maps.items()}
            
            # for threshold in [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0]:
            #     mask_32 = (entropy_maps[64] < threshold).float().sum()
            #     total_new_patch = orig_patches_per_image - 15 * mask_32
            #     total_new_patches[threshold] += total_new_patch
                
                
            for threshold in [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0]:
                mask64 = (entropy_maps[64] < 5.5).float()
                mask32 = (entropy_maps[32] < threshold).float()
                
                mask64_upscaled = mask64.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
                mask32 = mask32 * (1-mask64_upscaled)
                
                total_new_patch = orig_patches_per_image - 15 * mask64.sum() - 3 * mask32.sum()
                total_new_patches[threshold] += total_new_patch
                
        for threshold in [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0]:
            reduction = 1.0 - (total_new_patches[threshold] / total_orig_patches)
            
            print(f"Threshold: {threshold}, Reduction: {reduction:.2%}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure patch reduction using entropy-based adaptive patching")
    parser.add_argument("--data_dir", type=str, default="/edrive1/rchoudhu/ILSVRC2012/val",
                      help="Path to ImageNet validation directory")
    parser.add_argument("--image_size", type=int, default=224,
                      help="Size of input images")
    parser.add_argument("--num_images", type=int, default=1000,
                      help="Number of images to process")
    parser.add_argument("--num_scales", type=int, default=2,
                      help="Number of scales")
    
    
    args = parser.parse_args()
    args.data_dir = '/home/jungeun4/Mixed-Resolution-Patch/dataset/ILSVRC2012/val'
    
    
    # # check 16/32
    # for image_size in [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]:
    #     args.image_size = image_size
        
    #     main(args)
    
    # # check 16/64
    # for image_size in [256, 320, 384, 448, 512]:
    #     args.image_size = image_size
    #     args.num_scales = 3
        
    #     main(args)
            
    # check 16/32/64
    for image_size in [256, 320, 384, 448, 512]:
        args.image_size = image_size
        args.num_scales = 3
        
        main(args)