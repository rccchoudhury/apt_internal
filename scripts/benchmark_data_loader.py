"""
Script to test running through the imagenet dataloader.
"""
import sys
sys.path.append("../")

import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms, datasets
from timm.data import create_transform
import time
from tqdm import tqdm

from src.data.transforms import transforms_imagenet_eval


def create_dataloader(data_dir, image_size, batch_size, num_workers=0, pin_memory=False, start_idx=0):
    val_transform = transforms_imagenet_eval(img_size=image_size)
    dataset = datasets.ImageFolder(
        root=f"{data_dir}",
        transform=val_transform
    )
    
    # Create subset starting from start_idx
    total_samples = len(dataset)
    indices = list(range(start_idx, total_samples))
    subset_dataset = Subset(dataset, indices)
    
    val_loader = DataLoader(
        dataset=subset_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return val_loader

def main():
    split = "train"
    data_dir = os.path.join("/edrive1/rchoudhu/ILSVRC2012", split)
    image_size = 384
    batch_size = 256
    num_workers = 84
    pin_memory = True
    
    # Calculate start index based on desired batch number (3500)
    start_batch = 3500
    start_idx = start_batch * batch_size
    
    print(f"Starting from sample index {start_idx}")
    val_loader = create_dataloader(
        data_dir, 
        image_size, 
        batch_size, 
        num_workers, 
        pin_memory,
        start_idx=start_idx
    )
    
    max_iters = 10000
    start_time = time.time()
    
    for i, (images, labels) in tqdm(enumerate(val_loader), 
                                  total=max_iters-start_batch,
                                  desc="Benchmarking {} dataloader..".format(split)):
        if i >= (max_iters - start_batch):
            break
            
    total_time = time.time() - start_time
    actual_iters = max_iters - start_batch
    
    print("Benchmark complete.")
    print(f"Mean iter time: {total_time / actual_iters}")
    print(f"Images loaded per second: {batch_size * actual_iters / total_time}")
    print(f"Total time: {total_time}")

if __name__ == "__main__": 
    main()