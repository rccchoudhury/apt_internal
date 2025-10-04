import argparse
import sys
sys.path.append("..")
import ipdb
import numpy as np
import PIL.Image as Image
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import einops
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.transforms import transforms_imagenet_train, transforms_imagenet_eval, ImageFolderWithEntropy
from src.models.patch_tokenizer import PatchTokenizer

def setup_dataloaders(args):
    pre_transform, post_transform = transforms_imagenet_eval(
        img_size=args.image_size, 
        mean=[0.5, 0.5, 0.5], 
        std=[0.5, 0.5, 0.5],
        pre_post_divide=True,
        eval_crop_ratio=None
    )
    dataset = ImageFolderWithEntropy(
        root="/edrive1/rchoudhu/ILSVRC2012/val",
        transform=(pre_transform, post_transform),
        patch_size=args.base_patch_size,
        num_scales=args.num_scales)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=12, 
        pin_memory=True
    )

    return dataloader

def benchmark_tokenizer(tokenizer, dataloader, num_batches=1000, device='cuda'):
    total_time = 0
    num_processed = 0
    total_token_ratio = 0
    
    print(f"\nBenchmarking tokenizer over {num_batches} batches...")
    pbar = tqdm(enumerate(dataloader), total=num_batches, colour='green')
    
    for idx, batch in pbar:
        if idx >= num_batches:
            break
            
        images, labels, entropy_maps = batch
        images = images.to(device)
        entropy_maps = {k: v.to(device) for k, v in entropy_maps.items()}
        
        # Time the tokenizer
        start_time = time.time()
        with torch.no_grad():
            output_dict = tokenizer(images, entropy_maps)
        torch.cuda.synchronize()  # Make sure GPU operations are done
        batch_time = time.time() - start_time
        
        # Compute token statistics
        expected_tokens = args.batch_size * (args.image_size // args.base_patch_size) ** 2 + args.batch_size  # +batch_size for cls tokens
        actual_tokens = sum(output_dict['seqlens'])
        token_ratio = actual_tokens / expected_tokens
        total_token_ratio += token_ratio
        
        total_time += batch_time
        num_processed += 1
        
        avg_time = total_time / num_processed
        avg_token_ratio = total_token_ratio / num_processed
        pbar.set_description(f"Avg time: {avg_time:.4f}s, Avg token ratio: {avg_token_ratio:.3f}")
    
    print(f"\nBenchmark Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per batch: {total_time/num_processed:.4f}s")
    print(f"Throughput: {args.batch_size * num_processed / total_time:.2f} images/second")
    print(f"\nToken Statistics:")
    print(f"Expected tokens per batch: {expected_tokens}")
    print(f"Average actual tokens per batch: {actual_tokens/args.batch_size:.1f}")
    print(f"Average token reduction: {1 - (actual_tokens/expected_tokens):.3%}")
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Setting up dataset and dataloader...")
    dataloader = setup_dataloaders(args)
    
    tokenizer = PatchTokenizer(
        num_scales=args.num_scales,
        base_patch_size=args.base_patch_size,
        image_size=args.image_size,
        thresholds=args.thresholds,
        batch_size=args.batch_size
    ).to(device)
    
    output_dict = benchmark_tokenizer(tokenizer, dataloader, device=device)

    # print("\nSize of elements in output dict: ")
    # print("-" * 20)
    # for key, value in output_dict.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"{key}: {value.shape}")

    # print(f"Combined mask shape: {output_dict['output_mask'].shape}")
    # print(f"seqlens: {output_dict['seqlens']}")
    # print(f"Sum of seqlens: {sum(output_dict['seqlens'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image processing parameters')
    parser.add_argument('--image_size', type=int, default=336, help='Size of the image')
    parser.add_argument('--base_patch_size', type=int, default=14, help='Base size of the patch')
    parser.add_argument('--num_scales', type=int, default=3, help='Number of scales')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[6.5, 5.5], help='List of threshold values')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()
    main(args)