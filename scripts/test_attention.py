"""Test script for comparing standard and packed attention implementations.

This script contains tests to verify that the packed attention implementation
(which uses sequence packing and block diagonal masking) produces identical
outputs to the standard attention implementation when processing the same input
in different layouts.
"""

import torch
from typing import Tuple, Optional
import sys
import random
import numpy as np
sys.path.append("..")
from src.models.packed_vit import Attention as PackedAttention
from src.models.vision_transformer import Attention as StandardAttention
from xformers.ops.fmha.attn_bias import BlockDiagonalMask


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Integer seed for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_attention_modules(
    embed_dim: int,
    num_heads: int,
    device: torch.device
) -> Tuple[PackedAttention, StandardAttention]:
    """Create and initialize attention modules with identical weights.
    
    Args:
        embed_dim: Dimension of the input embeddings.
        num_heads: Number of attention heads.
        device: Device to place the modules on.
    
    Returns:
        Tuple containing:
            - packed_attention: Attention module supporting sequence packing
            - standard_attention: Standard attention module
    """
    # Initialize packed attention
    packed_attention = PackedAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ).to(device)

    # Initialize standard attention with same parameters
    standard_attention = StandardAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ).to(device)

    # Copy weights from packed to standard attention
    standard_attention.load_state_dict(packed_attention.state_dict())

    # Convert to float16 if using GPU for memory efficient attention
    if device.type == 'cuda':
        packed_attention = packed_attention.half()
        standard_attention = standard_attention.half()

    # Enable fused attention for standard attention
    standard_attention.fused_attn = True

    return packed_attention, standard_attention


def generate_test_input(
    batch_size: int,
    num_patches: int,
    embed_dim: int,
    device: torch.device
) -> torch.Tensor:
    """Generate test input tensor with reproducible values.
    
    Args:
        batch_size: Number of sequences in the batch.
        num_patches: Number of patches per sequence.
        embed_dim: Dimension of the embeddings.
        device: Device to place the tensor on.
    
    Returns:
        Tensor of shape (batch_size, num_patches, embed_dim).
    """
    set_seed()  # Set random seed for reproducible input
    return torch.rand(batch_size, num_patches, embed_dim, dtype=torch.float16, device=device)


def test_equivalence_without_mask() -> None:
    """Test equivalence of attention implementations without masking.
    
    This test verifies that both attention implementations produce identical
    outputs when processing batched input without any attention masking.
    """
    set_seed()  # Set random seed
    
    # Model parameters
    embed_dim = 768
    num_heads = 12
    batch_size = 1
    num_patches = 196  # For 224x224 image with 16x16 patches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create attention modules with identical weights
    packed_attention, standard_attention = create_attention_modules(embed_dim, num_heads, device)

    # Generate input tensor
    x = generate_test_input(batch_size, num_patches, embed_dim, device)

    # Run through both attention modules
    packed_output = packed_attention(x)
    print("Packed output shape: ", packed_output.shape)
    standard_output = standard_attention(x)
    print("Standard output shape: ", standard_output.shape)

    # Compare outputs
    if not torch.allclose(packed_output, standard_output, atol=1e-5):
        print("Outputs are not the same!")
        print("Difference:", packed_output - standard_output)
    else:
        print("Test passed: Attention outputs are equivalent.")


def test_equivalence_with_mask() -> None:
    """Test equivalence of attention with sequence packing and masking.
    
    This test verifies that the packed attention implementation with block
    diagonal masking produces identical outputs to the standard implementation
    when processing the same input in different layouts (packed vs. batched).
    """
    set_seed()  # Set random seed
    
    # Model parameters
    embed_dim = 768
    num_heads = 12
    batch_size = 4
    num_patches = 196  # For 224x224 image with 16x16 patches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create attention module
    packed_attention, _ = create_attention_modules(embed_dim, num_heads, device)

    # Generate input tensor
    x = generate_test_input(batch_size, num_patches, embed_dim, device)

    # Get output without masking (standard batched layout)
    standard_output = packed_attention(x)
    print("Packed output shape without mask: ", standard_output.shape)

    # Create block diagonal mask for packed layout
    seq_lens = [num_patches for _ in range(batch_size)]
    block_mask = BlockDiagonalMask.from_seqlens(seq_lens)

    # Run with packed layout and masking
    x_packed = x.reshape(1, batch_size * num_patches, embed_dim)
    packed_output = packed_attention(x_packed, attn_mask=block_mask)
    packed_output = packed_output.reshape(batch_size, num_patches, embed_dim)
    print("Packed output shape with mask: ", packed_output.shape)

    # Compare outputs
    if not torch.allclose(packed_output, standard_output, atol=1e-5):
        print("Outputs with mask are not the same!")
        print("Difference:", packed_output - standard_output)
        print("Sequence lengths: ", seq_lens)
        print("Materialized attention Mask: ")
        print(block_mask.materialize(shape=(num_patches * batch_size, num_patches * batch_size), device=device))
    else:
        print("Test passed: Attention outputs with mask are equivalent.")


if __name__ == "__main__":
    print("Testing attention equivalence without mask...")
    test_equivalence_without_mask()
    print("\nTesting attention equivalence with mask...")
    test_equivalence_with_mask()
