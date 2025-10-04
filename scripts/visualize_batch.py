import sys
sys.path.append("../")
import cv2
import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from timm.data import create_transform
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

from src.data.transforms import transforms_imagenet_eval, ImageFolderWithEntropy
from src.models.entropy_utils import compute_patch_entropy_vectorized, visualize_selected_patches_cv2

# Use this because of the hard coding. Should load from entropy_utils when solved
def select_patches_by_threshold(entropy_maps, thresholds, alpha=1.):
    """
    Vectorized version of patch selection based on entropy thresholds.
    
    Args:
        entropy_maps (dict): Contains patch sizes as keys mapping to
            torch.Tensor entropy maps of shape (B, H_p, W_p) where
            H_p and W_p depend on the patch size
        thresholds (list): List of thresholds for selecting patches at each scale,
            should have length = len(entropy_maps) - 1
        alpha (float): Ratio of the entropy based. 
    Returns:
        masks (dict): Dictionary mapping patch sizes to their 0/1 masks
    """
    patch_sizes = sorted(list(entropy_maps.keys()))
    if len(thresholds) != len(patch_sizes) - 1:
        raise ValueError(f'Number of thresholds ({len(thresholds)}) must be one less than number of patch sizes ({len(patch_sizes)})')

    masks_init = {}
    # Initialize mask for smallest patch size
    masks_init[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
    
    # Process each scale from largest to smallest
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        threshold = thresholds[i-1]
        
        # Create mask for current patch size
        masks_init[current_size] = (entropy_maps[current_size] < threshold).float()
    masks = masks_init
        
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        for j in range(i):
            # Upscale mask to match smaller patch size
            smaller_size = patch_sizes[j]
            scale_factor = current_size // smaller_size 
            mask_upscaled = masks[current_size].repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
            
            # Ensure upscaled mask matches the dimensions of smaller patches
            H_small, W_small = entropy_maps[smaller_size].shape[1:]  # Assuming batch dimension
            mask_upscaled = mask_upscaled[:, :H_small, :W_small]
            
            # Update mask for smaller patches
            masks[smaller_size] = masks[smaller_size] * (1 - mask_upscaled)
    
    return masks

def hstack_images(images: list[Image.Image], gap: int = 20) -> Image.Image:
    prev_concat = images[0]
    for image in images[1:]:
        concat = Image.new("RGB", (prev_concat.width + gap +
                        image.width, image.height), (255, 255, 255))
        concat.paste(prev_concat, (0, 0))
        concat.paste(image, (prev_concat.width + gap, 0))
        prev_concat = concat
    return concat

def main():
    # Setup parameters
    data_dir = "/home/jungeun4/Mixed-Resolution-Patch/dataset/ILSVRC2012/val"
    output_dir = "vis_results"
    image_size = 224
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    val_pre_transform, val_post_transform = transforms_imagenet_eval(
        img_size=image_size, 
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        pre_post_divide=True
    )

    data_val = ImageFolderWithEntropy(
        root=f"{data_dir}",
        transform=(val_pre_transform, val_post_transform),
        patch_size=16,
        num_scales=num_scales
    )

    unnorm = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5, 1/0.5, 1/0.5]
    )
    
    patches = [16 * 2**i for i in range(num_scales)]

    # Process all images in the dataset
    for idx in range(len(data_val)):
        # Only process every 10th image
        if idx % 25 != 0:
            continue
            
        to_visualize = []
        images, labels, entropy_maps = data_val[idx]
        images = unnorm(images) * 255
        images_init = images.permute(1, 2, 0).numpy().astype(np.uint8)
        images_init = Image.fromarray(images_init)
        to_visualize.append(images_init)
        
        entropy_maps = {patch: entropy_maps[patch].unsqueeze(0) for patch in patches}
        
        masks = select_patches_by_threshold(entropy_maps, [-100] * len(thresholds))
        masks = {patch: masks[patch].squeeze(0) for patch in patches}
        image_finest = visualize_selected_patches_cv2(
            images,
            masks,
            patches,
            color = (255, 255, 255),
            thickness = 1
        )
        to_visualize.append(image_finest)
        
        masks = select_patches_by_threshold(entropy_maps, thresholds)
        masks = {patch: masks[patch].squeeze(0) for patch in patches}
        image_ours = visualize_selected_patches_cv2(
            images,
            masks,
            patches,
            color = (255, 255, 255),
            thickness = 1
        )
        to_visualize.append(image_ours)

        vis = hstack_images(to_visualize)
        output_path = os.path.join(output_dir, f"visualization_{idx:04d}.png")
        vis.save(output_path)
        print(f"Saved visualization for image {idx} to {output_path}")

num_scales = 3
thresholds = [5.5, 6.5]
assert num_scales == len(thresholds) + 1
    
if __name__ == "__main__":
    main()
