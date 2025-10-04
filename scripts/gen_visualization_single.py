import sys
sys.path.append("..")
import torch
import cv2
import argparse
import einops
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF

import math
import PIL.Image as Image
from PIL import ImageDraw
import os
import matplotlib.pyplot as plt
from matplotlib import cm

from src.data.transforms import transforms_imagenet_train, transforms_imagenet_eval, ImageFolderWithEntropy
from src.models.entropy_utils import compute_patch_entropy_vectorized, select_patches_by_threshold, visualize_selected_patches_cv2_non_overlapping

# Default parameters
IMAGE_SIZE = 336
BASE_PATCH_SIZE = 14
NUM_SCALES = 3
THRESHOLDS = [6.0, 6.0]
# [0.035-45] fo rboth scales for laplacian
LINE_COLOR = (255, 255, 255)  # White color
LINE_THICKNESS = 2  # Integer thickness - fractional effect achieved through upsampling/downsampling
UPSCALE_FACTOR = 1.5  # Factor to upscale the image before drawing lines
OVERLAY_ALPHA = 0.7  # Alpha value for non-selected patches overlay
OUTPUT_PATH = "../../assets/vis_single.jpg"
OUTPUT_DIR = "../../assets"

def parse_args():
    parser = argparse.ArgumentParser(description='Generate image visualizations with different options')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--vis_type', type=str, default='entropy', choices=['entropy', 'grid', 'none'], 
                        help='Visualization type: entropy-based, standard grid, or no lines')
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, help='Target image size for the shorter side')
    parser.add_argument('--patch_size', type=int, default=BASE_PATCH_SIZE, help='Base patch size')
    parser.add_argument('--num_scales', type=int, default=NUM_SCALES, help='Number of scales for entropy calculation')
    parser.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS, 
                        help='Thresholds for entropy-based patch selection')
    parser.add_argument('--line_color', type=int, nargs=3, default=list(LINE_COLOR), 
                        help='Line color in RGB format (3 values from 0-255)')
    parser.add_argument('--line_thickness', type=int, default=LINE_THICKNESS, 
                        help='Line thickness (integer value)')
    parser.add_argument('--upscale_factor', type=float, default=UPSCALE_FACTOR,
                        help='Factor to upscale the image before drawing lines (for fractional thickness)')
    parser.add_argument('--output_prefix', type=str, default='entropy_map',
                        help='Prefix for entropy map output files (will be saved as prefix_l1.jpg, prefix_l2.jpg, etc.)')
    parser.add_argument('--colormap', type=str, default='inferno',
                        help='Colormap for entropy heatmaps (e.g., inferno, plasma, magma, jet, hot, etc.)')
    parser.add_argument('--overlay_alpha', type=float, default=OVERLAY_ALPHA,
                        help='Alpha value for non-selected patches overlay (0.0 is transparent, 1.0 is opaque)')
    
    return parser.parse_args()

def save_entropy_map_as_heatmap(entropy_map, output_path, original_size, colormap='inferno'):
    """Save entropy map as a matplotlib heatmap resized to the original image size."""
    # Convert entropy map to numpy array
    entropy_np = entropy_map.cpu().numpy()
    
    # Create figure without axes
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    # Create heatmap with specified colormap
    plt.imshow(entropy_np, cmap=colormap)
    plt.tight_layout(pad=0)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Resize the saved image to match the original image size
    saved_img = Image.open(output_path)
    saved_img = saved_img.resize(original_size, Image.LANCZOS)
    saved_img.save(output_path)

def visualize_selected_patches(mask, patch_size, image_size, output_path):
    """Create a binary visualization of selected patches (white) and non-selected patches (black)."""
    # Get mask dimensions
    h, w = mask.shape
    
    # Create a blank image
    vis_img = Image.new('RGB', image_size, color=(0, 0, 0))
    draw = ImageDraw.Draw(vis_img)
    
    # Calculate scaling factors
    scale_h = image_size[1] / h
    scale_w = image_size[0] / w
    
    # Draw white rectangles for selected patches (where mask is 1)
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1:
                x1 = j * scale_w
                y1 = i * scale_h
                x2 = (j + 1) * scale_w
                y2 = (i + 1) * scale_h
                draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
    
    # Save the visualization
    vis_img.save(output_path)
    return vis_img

def create_overlay_visualization(original_img, mask, patch_size, alpha, output_path):
    """Create a visualization with the original image and high alpha overlay on non-selected patches."""
    # Convert original image to PIL if it's a tensor
    if isinstance(original_img, torch.Tensor):
        if original_img.dim() == 3 and original_img.shape[0] == 3:
            # Convert CHW to HWC for PIL
            np_img = original_img.permute(1, 2, 0).cpu().numpy()
            if np_img.max() <= 1.0:
                np_img = np_img * 255.0
            np_img = np_img.astype(np.uint8)
            original_img = Image.fromarray(np_img)
        else:
            raise ValueError("Unsupported tensor format for original_img")
    
    # Make a copy of the original image
    overlay_img = original_img.copy()
    draw = ImageDraw.Draw(overlay_img, 'RGBA')
    
    # Get mask dimensions and image size
    h, w = mask.shape
    img_w, img_h = original_img.size
    
    # Calculate scaling factors
    scale_h = img_h / h
    scale_w = img_w / w
    
    # Draw semi-transparent overlay for non-selected patches (where mask is 0)
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0:
                x1 = j * scale_w
                y1 = i * scale_h
                x2 = (j + 1) * scale_w
                y2 = (i + 1) * scale_h
                # Convert alpha to 0-255 range for PIL
                alpha_int = int(alpha * 255)
                draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, alpha_int))
    
    # Save the visualization
    overlay_img.save(output_path)
    return overlay_img

def process_image(image_path, vis_type, image_size, patch_size, num_scales, thresholds, line_color, line_thickness, upscale_factor, output_prefix, colormap, overlay_alpha):
    """Process a single image and return the visualization based on the specified type."""
    print(f"Processing image: {image_path}")
    
    # Open the image
    image = Image.open(image_path)
    print(f"Original image size: {image.size}")
    
    # Find the image size
    width, height = image.size

    # Calculate the target size for the shorter side
    # Make the shorter side always image_size
    if width < height:
        new_width = image_size
        new_height = int(height * (image_size / width))
    else:
        new_height = image_size
        new_width = int(width * (image_size / height))

    # Adjust the longer side to be a multiple of patch_size * 4
    if width > height:
        new_width = (new_width // (patch_size * 4)) * (patch_size * 4)
    else:
        new_height = (new_height // (patch_size * 4)) * (patch_size * 4)

    # Resize the image
    img = image.resize((new_width, new_height))
    print(f"Resized image size: {img.size}")

    # Load the image as numpy array
    np_img = np.array(img)
    
    # If visualization type is 'none', return the resized image without any lines
    if vis_type == 'none':
        return img
    
    # Convert to 3-channel if the input is grayscale
    if len(np_img.shape) == 2:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    
    # Get image dimensions
    height, width = np_img.shape[:2]
    
    if vis_type == 'grid':
        # Create a copy for drawing grid
        image_with_grid = np_img.copy()
        
        # Upsample the image before drawing lines - this is how we achieve fractional line thickness
        upsampled_height, upsampled_width = int(height * upscale_factor), int(width * upscale_factor)
        upsampled_image = cv2.resize(image_with_grid, (upsampled_width, upsampled_height), interpolation=cv2.INTER_LINEAR)

        # Draw vertical lines on upsampled image
        for x in range(0, upsampled_width, int(patch_size * upscale_factor)):
            cv2.line(upsampled_image, (x, 0), (x, upsampled_height), tuple(line_color), line_thickness)

        # Draw horizontal lines on upsampled image
        for y in range(0, upsampled_height, int(patch_size * upscale_factor)):
            cv2.line(upsampled_image, (0, y), (upsampled_width, y), tuple(line_color), line_thickness)

        # Downsample back to original size - this creates the effect of fractional line thickness
        image_with_grid = cv2.resize(upsampled_image, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to PIL image
        return Image.fromarray(image_with_grid)
    
    elif vis_type == 'entropy':
        # Convert image to tensor for entropy calculation
        img_tensor = TF.to_tensor(img) * 255.0
        img_int_tensor = img_tensor.to(torch.uint8)
        
        # Compute entropy maps
        entropy_maps = compute_patch_entropy_vectorized(img_tensor, patch_size, num_scales)
        
        # Save entropy maps as heatmaps
        for scale, entropy_map in entropy_maps.items():
            # Create output path for this entropy map
            scale_idx = scale
            output_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{scale_idx}.jpg")
            
            # Save the entropy map as a heatmap
            save_entropy_map_as_heatmap(entropy_map, output_path, (width, height), colormap)
            print(f"Saved entropy map for scale {scale} to {output_path}")
        
        # Prepare entropy maps for visualization
        for k, v in entropy_maps.items():
            entropy_maps[k] = v.unsqueeze(0)

        # Select patches based on threshold
        all_masks = select_patches_by_threshold(
            entropy_maps, thresholds
        )

        # Create a copy of the masks for our custom visualizations
        mask_copies = {}
        for scale, mask in all_masks.items():
            mask_copies[scale] = mask.clone()

        # Create visualizations for each mask
        for scale, mask in mask_copies.items():
            # Get the mask as a 2D tensor
            mask_2d = mask.squeeze(0)
            
            # Create a binary visualization of selected patches
            binary_output_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{scale}_selected.jpg")
            visualize_selected_patches(mask_2d, patch_size, (width, height), binary_output_path)
            print(f"Saved selected patches visualization for {scale} to {binary_output_path}")
            
            # Create an overlay visualization
            overlay_output_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{scale}_overlay.jpg")
            create_overlay_visualization(img, mask_2d, patch_size, overlay_alpha, overlay_output_path)
            print(f"Saved overlay visualization for {scale} to {overlay_output_path}")

        # Visualize selected patches
        patch_sizes = [patch_size]
        for i in range(1, num_scales):
            patch_sizes.append(patch_size * (2**i))
        
        # Ensure masks are in the right format for the visualization function
        # The function expects masks to be a dictionary with keys like 'scale_1', 'scale_2', etc.
        # and values that are 2D tensors (not 3D with a batch dimension)
        visualization_masks = {}
        for scale, mask in all_masks.items():
            visualization_masks[scale] = mask.squeeze(0)
            
        # The visualize_selected_patches_cv2_non_overlapping function has its own upscaling logic
        vis_img = visualize_selected_patches_cv2_non_overlapping(
            image_tensor=img_tensor,
            masks=visualization_masks,
            patch_sizes=patch_sizes,
            color=tuple(line_color),
            thickness=line_thickness
        )
        
        return vis_img

def main():
    args = parse_args()
    
    # Validate thresholds length
    if len(args.thresholds) != args.num_scales - 1:
        print(f"Warning: Number of thresholds ({len(args.thresholds)}) does not match required number ({args.num_scales - 1})")
        print(f"Using default thresholds: {THRESHOLDS}")
        args.thresholds = THRESHOLDS
    
    # Process the image
    result_image = process_image(
        args.input, 
        args.vis_type, 
        args.image_size, 
        args.patch_size, 
        args.num_scales, 
        args.thresholds, 
        args.line_color, 
        args.line_thickness,
        args.upscale_factor,
        args.output_prefix,
        args.colormap,
        args.overlay_alpha
    )
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Save the result
    result_image.save(OUTPUT_PATH)
    print(f"Saved visualization to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
