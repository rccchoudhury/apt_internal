"""
Script to test running through the imagenet dataloader with patch embedding.
"""
import sys
sys.path.append("..")

import os
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import argparse

from src.data.transforms import transforms_imagenet_eval, ImageFolderWithEntropy
from src.models.vision_transformer import VisionTransformer
from src.models.patch_embed import MixedSquarePatchEmbed, MixedPatchEmbed,

def create_dataloader(data_dir, image_size, batch_size, num_workers, pin_memory) -> DataLoader:
    """Create dataloader with the given configuration."""
    pre_transform, post_transform = transforms_imagenet_eval(
        img_size=image_size, 
        pre_post_divide=True
    )
    
    dataset = ImageFolderWithEntropy(
        root=data_dir,
        transform=(pre_transform, post_transform),
        patch_sizes=[16, 32]  # Match the patch sizes used in training
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

def create_model(patch_embed_cls, image_size: int = 384, embed_dim: int = 1024) -> VisionTransformer:
    """Create and initialize the model for benchmarking."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the full vision transformer
    model = VisionTransformer(
        img_size=image_size,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        group_token=True,
        no_embed_class=True,
        mixed_patch_embed=patch_embed_cls,
        window_size=4,
        merge_ratio=0.5,
        weight_init='skip'
    ).to(device)
    
    # Initialize the mixed patch embedding
    print(f"\nInitializing {patch_embed_cls.__name__}...")
    model.init_multiscale_patch_embed()
    model.eval()
    
    return model

def run_benchmark(split, data_dir, image_size, batch_size, num_workers, pin_memory):
    """Run the benchmark with the given configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = create_dataloader(data_dir, image_size, batch_size, num_workers, pin_memory)
    
    # Create both types of models
    model_square = create_model(MixedSquarePatchEmbed, image_size=image_size)
    model_regular = create_model(MixedPatchEmbed, image_size=image_size)
    model_regular.mixed_patch.to(device)
    model_square.mixed_patch.to(device)
    
    total_time_square = 0
    embed_time_square = 0
    total_time_regular = 0
    embed_time_regular = 0
    n_batches = 0
    
    print(f"\nBenchmarking {split} dataloader with both patch embeddings..")
    start_time = time.time()
    
    for images, _, entropy_maps in tqdm(loader, colour="blue"):
        batch_start = time.time()
        
        images = images.to(device)
        # Move entropy maps to device
        expected_count = 0
        for k in entropy_maps.keys():
            entropy_maps[k] = entropy_maps[k].to(device)

        print("Expected number patches = ",expected_count )
        # Time square patch embedding
        torch.cuda.synchronize()
        embed_start = time.time()
        with torch.no_grad():
            x_square, mask_square, idx_square = model_square.mixed_patch(images, entropy_maps)
        torch.cuda.synchronize()
        embed_time_square += time.time() - embed_start
        total_time_square += time.time() - batch_start
        
        # Time regular patch embedding
        batch_start = time.time()
        torch.cuda.synchronize()
        embed_start = time.time()
        with torch.no_grad():
            x_regular, mask_regular, idx_regular = model_regular.mixed_patch(images, entropy_maps)
        torch.cuda.synchronize()
        embed_time_regular += time.time() - embed_start
        total_time_regular += time.time() - batch_start
        
        # Compare outputs
        print(f"\nBatch {n_batches} shapes:")
        print("Input shape:", images.shape)
        print(f"Square output shape: {x_square.shape}")
        print(f"Regular output shape: {x_regular.shape}")
        if x_square.shape != x_regular.shape:
            print("WARNING: Output shapes do not match!")
        
        n_batches += 1
        if n_batches >= 5:  # Only test first 5 batches
            break
    
    # Calculate metrics
    total_images = n_batches * batch_size
    total_time_taken = time.time() - start_time
    
    print("\nBenchmark Results:")
    print(f"Total images processed: {total_images}")
    print(f"Total time: {total_time_taken:.2f}s")
    print(f"Images loaded per second: {total_images / total_time_taken:.2f}")
    
    print(f"\nMixedSquarePatchEmbed:")
    print(f"Mean time per batch: {(embed_time_square / n_batches * 1000):.2f}ms")
    print(f"Throughput: {total_images / embed_time_square:.2f} images/sec")
    
    print(f"\nMixedPatchEmbed:")
    print(f"Mean time per batch: {(embed_time_regular / n_batches * 1000):.2f}ms")
    print(f"Throughput: {total_images / embed_time_regular:.2f} images/sec")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--data-dir', type=str, default='/edrive1/rchoudhu/ILSVRC2012')
    parser.add_argument('--image-size', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--pin-memory', type=bool, default=True)
    args = parser.parse_args()
    
    # Run benchmark
    run_benchmark(
        split=args.split,
        data_dir=os.path.join(args.data_dir, args.split),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

if __name__ == "__main__":
    main()
