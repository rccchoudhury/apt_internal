import sys
sys.path.append("..")
import torch
import torchvision.transforms as transforms
from timm.data import create_transform
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF
import numpy as np
import ipdb

from src.models.entropy_utils import compute_patch_entropy_vectorized, compute_patch_laplacian_vectorized

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

IMAGENET_FLEXI_MEAN = (0.5, 0.5, 0.5)
IMAGENET_FLEXI_STD = (0.5, 0.5, 0.5)

def transforms_imagenet_train(
    img_size=256,
    interpolation=transforms.InterpolationMode.BILINEAR,
    mean=IMAGENET_FLEXI_MEAN,
    std=IMAGENET_FLEXI_STD,
    pre_post_divide=False,
    augment=None
    ):  
    
    if augment is not None:
        pre_transform = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=augment.color_jitter,
            auto_augment=augment.auto_augment,
            interpolation=augment.interpolation,
            re_prob=augment.re_prob,
            re_mode=augment.re_mode,
            re_count=augment.re_count
        ).transforms[:-3] # Removing totensor and normalize
    else:
        pre_transform = [
            transforms.RandomResizedCrop(img_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    
    post_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    
    if pre_post_divide:
        return transforms.Compose(pre_transform), transforms.Compose(post_transform)
    else:
        return transforms.Compose(pre_transform + post_transform)

def transforms_imagenet_eval(
        img_size=256,
        interpolation=transforms.InterpolationMode.BILINEAR,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        pre_post_divide=False,
        eval_crop_ratio=0.875):
    
    if eval_crop_ratio is not None:
        size = int(img_size / eval_crop_ratio)
        pre_transform = [
            transforms.Resize(size, interpolation=3),
            transforms.CenterCrop(img_size),
        ]
    else:
        pre_transform = [
            transforms.Resize(img_size, interpolation=interpolation),
            transforms.CenterCrop(img_size),
        ]
    
    post_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    
    if pre_post_divide:
        return transforms.Compose(pre_transform), transforms.Compose(post_transform)
    else:
        return transforms.Compose(pre_transform + post_transform)

class ImageFolderWithEntropy(ImageFolder):
    """ImageFolder dataset that computes entropy before applying transforms."""
    
    def __init__(self, root, transform=None, patch_size=16, num_scales=2, **kwargs):
        # transform here should be a tuple of (pre_transform, post_transform)
        super().__init__(root, transform=None, **kwargs)  # Set transform to None since we'll handle it manually
        self.pre_transform, self.post_transform = transform
        self.patch_size = patch_size
        self.num_scales = num_scales
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        # Load image using PIL
        sample = self.loader(path)
        
        # Apply pre-entropy transforms (resize, crop)
        if self.pre_transform is not None:
            sample = self.pre_transform(sample)
        
        # Convert to tensor for entropy computation (values in [0, 255])
        img_tensor = TF.to_tensor(sample) * 255.0
        
        # Compute entropy maps
        entropy_maps = compute_patch_entropy_vectorized(img_tensor, self.patch_size, self.num_scales)
        #entropy_maps = compute_patch_laplacian_vectorized(img_tensor, self.patch_size, self.num_scales)
        
        # Apply post-entropy transforms (normalization)
        if self.post_transform is not None:
            sample = self.post_transform(sample)
            
        return sample, target, entropy_maps