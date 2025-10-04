import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
sys.path.append("..")
import torch
import einops
from dataclasses import dataclass
import time
from tqdm import tqdm
import gc
from timm.data import Mixup

import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

import PIL.Image as Image

from src.data.transforms import transforms_imagenet_train, transforms_imagenet_eval, ImageFolderWithEntropy
from src.models.entropy_utils import *
from src.models.patch_tokenizer import PatchTokenizer


IMAGE_SIZE = 336
BASE_PATCH_SIZE = 14
NUM_SCALES = 3
THRESHOLDS = [5.75, 4.5]
#THRESHOLDS = [0.0475, 0.03]
BATCH_SIZE = 32

unnorm = transforms.Normalize(
    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
    std=[1/0.5, 1/0.5, 1/0.5]
)

@dataclass
class AugmentConfig:
    color_jitter: float = 0.0
    auto_augment: str = 'rand-m9-mstd0.5-inc1'  # RandAugment with magnitude 9
    interpolation: str = 'bicubic'
    re_prob: float = 0.25  # Random erasing probability
    re_mode: str = 'const'  # Random erasing fill mode ('pixel', 'const', etc.)
    re_count: int = 1  # Number of random erasing regions

# Create an instance to pass to transforms_imagenet_train
augment_config = AugmentConfig()

transform = transforms_imagenet_train(
    img_size=IMAGE_SIZE, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5],
    pre_post_divide=False,
    augment=augment_config,
)

mixup_fn = Mixup(mixup_alpha=0.8, 
                cutmix_alpha=1.0, 
                cutmix_minmax=None,
                prob=1.0, 
                switch_prob=0.5, 
                mode='batch',
                label_smoothing=0.1, 
                num_classes=1000)

# dataset = ImageFolderWithEntropy(
#     root="/edrive1/rchoudhu/ILSVRC2012/train",
#     transform=(pre_transform, post_transform),
#     patch_size=BASE_PATCH_SIZE,
#     num_scales=NUM_SCALES)

dataset = ImageFolder(
    root="/edrive1/rchoudhu/ILSVRC2012/train",
    transform=transform)


dataloader = DataLoader(dataset, 
    batch_size=BATCH_SIZE, 
    num_workers=12, 
    pin_memory=True, 
    shuffle=True)

# Set up GPU memory optimization
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
torch.cuda.empty_cache()  # Clear GPU cache before starting

tokenizer = PatchTokenizer(
    base_patch_size=BASE_PATCH_SIZE,
    num_scales=NUM_SCALES,
    thresholds=THRESHOLDS,
    image_size=IMAGE_SIZE,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    method="entropy"
)

tokenizer = tokenizer.to('cuda')

# Initialize metrics tracking
total_batches = 0
total_retain_frac = 0.0
total_time = 0.0
batch_times = []

# Warmup run to initialize CUDA kernels and cache
print("Performing warmup iteration...")
with torch.no_grad():
    warmup_images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device='cuda')
    _ = tokenizer.compute_importance_maps(warmup_images)
    torch.cuda.synchronize()  # Wait for all operations to complete
    
print("Starting benchmark...")

# Create a tqdm progress bar
pbar = tqdm(dataloader, desc="Processing batches", unit="batch")

# Process all batches
for batch_idx, batch in enumerate(pbar):
    if batch_idx > 1000: break
    
    # Clear cache between iterations to prevent memory buildup
    if batch_idx % 10 == 0:
        torch.cuda.empty_cache()
        gc.collect()
    
    start_time = time.time()
    
    images, labels = batch
    
    # Process the batch
    images = images.to('cuda', non_blocking=True)  # Use non_blocking for potential speedup
    images, labels = mixup_fn(images, labels)
    
    with torch.no_grad():  # Ensure we don't track gradients for inference
        entropy_maps = tokenizer.compute_importance_maps(images)
        masks = select_patches_by_threshold(entropy_maps, thresholds=THRESHOLDS)
    
    torch.cuda.synchronize()  # Ensure all GPU operations are complete before timing
    
    # Calculate token statistics
    batch_vis_masks = {k: v for k, v in masks.items()}
    actual_token_count = sum(v.sum().item() for v in batch_vis_masks.values())
    expected_token_count = (IMAGE_SIZE // BASE_PATCH_SIZE) ** 2 * BATCH_SIZE
    
    # Calculate retain fraction
    retain_frac = actual_token_count / expected_token_count
    
    # Update metrics
    total_batches += 1
    total_retain_frac += retain_frac
    
    # Calculate iteration time
    end_time = time.time()
    iter_time = end_time - start_time
    total_time += iter_time
    batch_times.append(iter_time)
    
    # Update progress bar with current metrics
    pbar.set_postfix({
        'iter_time': f"{iter_time:.4f}s", 
        'retain_frac': f"{retain_frac:.4f}",
        'avg_retain': f"{total_retain_frac/total_batches:.4f}"
    })

# Calculate and print final statistics
avg_retain_frac = total_retain_frac / total_batches
avg_iter_time = total_time / total_batches
std_iter_time = np.std(batch_times) if batch_times else 0

print("\nBenchmark Results:")
print(f"Total batches processed: {total_batches}")
print(f"Average retain fraction: {avg_retain_frac:.4f}")
print(f"Average iteration time: {avg_iter_time:.4f}s")
print(f"Std deviation of iteration time: {std_iter_time:.4f}s")
print(f"Total processing time: {total_time:.2f}s")

# Final cleanup
torch.cuda.empty_cache()
