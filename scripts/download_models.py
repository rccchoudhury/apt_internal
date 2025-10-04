import sys
sys.path.append("..")
import os
import torch
import torch.nn as nn
import timm

from src.models.vision_transformer import VisionTransformer

###########################################################
# UNCOMMENT BELOW TO MOVE WEIGHTS FROM FLAX -> PYTORCH
###########################################################

checkpoints = [
    #'vit_base_patch8_224.augreg2_in21k_ft_in1k', 
    # 'vit_base_patch16_224.augreg2_in21k_ft_in1k', 
    # 'vit_base_patch16_384.augreg_in21k_ft_in1k', 
    # 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', 
    # 'eva_large_patch14_196.in22k_ft_in22k_in1k', 
    # 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k', 
    # 'vit_large_patch16_224.augreg_in21k_ft_in1k', 
     'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k',
     'vit_large_patch16_224.mae.pth',
    # 'eva_large_patch14_336.in22k_ft_in22k_in1k',
    # 'vit_large_patch16_384.augreg_in21k_ft_in1k',
    # 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
    # 'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k',
    # 'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k',
    # 'eva_giant_patch14_224.clip_ft_in1k',
    # 'eva_giant_patch14_336.m30m_ft_in22k_in1k',
    # 'eva_giant_patch14_560.m30m_ft_in22k_in1k',
]

for checkpoint in checkpoints:
    print(checkpoint)
    model = timm.create_model(checkpoint, pretrained=True)
    # Save checkpoint
    ckpt_folder_path = "../checkpoints"
    name = checkpoint.split(".")[0]
    torch.save(model.state_dict(), os.path.join(ckpt_folder_path, f"{name}.pth")) 