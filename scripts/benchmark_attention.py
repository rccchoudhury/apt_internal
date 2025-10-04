import torch
import torch.nn.functional as F
import xformers.ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
import time
from typing import Optional, Tuple
import argparse
import ipdb
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from functools import lru_cache


class BaseAttention(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        
        # Initialize q, k norms like in vision transformer
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.block_mask = None

    def forward(self, x: torch.Tensor, attn_mask: Optional[BlockDiagonalMask] = None) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses")

class PackedAttention(BaseAttention):
    @torch._dynamo.disable
    def forward(self, x: torch.Tensor, attn_mask: Optional[BlockDiagonalMask] = None) -> torch.Tensor:
        B, N, C = x.shape
        x = x.reshape(1, B*N, C).contiguous()
        qkv = self.qkv(x).reshape(1, B*N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, 1, num_heads, B*N, head_dim)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        q = q.reshape(1, B*N, self.num_heads, self.head_dim)
        k = k.reshape(1, B*N, self.num_heads, self.head_dim)
        v = v.reshape(1, B*N, self.num_heads, self.head_dim)
        
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.device)
            
        x = xops.fmha.memory_efficient_attention(
            q, k, v,
            p=self.attn_drop.p if self.training else 0.,
            attn_bias=attn_mask
        )
        
        x = x.reshape(1, self.num_heads, B, N, self.head_dim)
        x = x.squeeze(0).permute(1, 0, 2, 3).contiguous()  # (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FlexAttention(BaseAttention):
    @lru_cache
    def create_mask(self, B, N):
        def seq_packing_mask(b, h, q_idx, kv_idx):
            # Here b is the batch index (only one batch: b == 0) and h is head index.
            # q_idx and kv_idx are token indices in the range [0, B*N).
            return (q_idx // N) == (kv_idx // N)

        # Create the block mask using the helper.
        block_mask = create_block_mask(seq_packing_mask,
                                       B=1,             # our packed batch size is 1
                                       H=self.num_heads,
                                       Q_LEN=B * N,         # query length = B*N
                                       KV_LEN=B * N,         # key length = B*N
                                       device='cuda')
        self.block_mask = block_mask

    def forward(self, x: torch.Tensor, attn_mask: Optional[BlockDiagonalMask] = None) -> torch.Tensor:
        B, N, C = x.shape
        # x: [B, N, C] -- here, B is the number of sequences
        # First, pack the sequences into a single batch dimension.
        # We want to treat the entire packed sequence as one long sequence.
        # In your original code, you did:
        #    x = x.reshape(1, B*N, C)
        # Here we follow that convention.
        packed_x = x.reshape(1, B * N, C).contiguous()

        # Compute QKV projections.
        # qkv: [1, B*N, 3, num_heads, head_dim]
        qkv = self.qkv(packed_x).reshape(1, B * N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # now: (3, 1, num_heads, B*N, head_dim)
        q, k, v = qkv.unbind(0)           # each is of shape: [1, num_heads, B*N, head_dim]

        # Apply optional normalization to queries and keys.
        q = self.q_norm(q)
        k = self.k_norm(k)
        # v remains unchanged.

        # ---
        # Create the block-diagonal mask to restrict attention to tokens within the same sequence.
        #
        # Since our tokens are packed as a single long sequence of length (B*N),
        # we want to allow attention only among tokens within the same block of N tokens.
        #
        # The mask function below returns True (i.e. keep the attention score)
        # if and only if the query token and key token belong to the same block.
        # (It does this by integer dividing the token indices by N.)
        #
        if self.block_mask is None:
            self.create_mask(B, N)

        # ---
        # Call FlexAttention.
        #
        # FlexAttention expects inputs of shape [B, H, S, D], so our q, k, v
        # are already in the proper shape: [1, num_heads, B*N, head_dim].
        #
        attn_out = flex_attention(q, k, v, block_mask=self.block_mask)
        # attn_out: [1, num_heads, B*N, head_dim]

        # ---
        # Reshape the output back to the original packed structure.
        # For example, you might want to end up with a tensor of shape [B, N, num_heads*head_dim]
        # or some other layout. Here we first reshape to [1, num_heads, B, N, head_dim],
        # then squeeze the batch dimension and permute if desired.
        #
        attn_out = attn_out.reshape(1, self.num_heads, B, N, self.head_dim)
        # For instance, to get [B, num_heads, N, head_dim]:
        attn_out = attn_out.squeeze(0)  # now [num_heads, B, N, head_dim]
        # Permute to [B, num_heads, N, head_dim]:
        attn_out = attn_out.permute(1, 0, 2, 3).contiguous()

        return attn_out

class StandardAttention(BaseAttention):
    def forward(self, x: torch.Tensor, attn_mask: Optional[BlockDiagonalMask] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def benchmark_attention(
    batch_size: int,
    seq_len: int,
    dim: int,
    num_heads: int,
    num_iterations: int = 100,
    device: str = "cuda",
    attention_class: BaseAttention = PackedAttention,
    use_mask: bool = True
) -> Tuple[float, float]:
    """Run attention benchmark and return mean and std of iteration times."""
    model = attention_class(dim=dim, num_heads=num_heads).to(device)
    model = torch.compile(model)
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    if use_mask:
        seq_lens = [seq_len for _ in range(batch_size)]
        mask = BlockDiagonalMask.from_seqlens(seq_lens).to(device)
    else:
        mask = None
    
    # Warmup
    for _ in range(10):
        model(x, mask)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        model(x, mask)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    times = torch.tensor(times)
    return float(times.mean()), float(times.std())

def main():
    parser = argparse.ArgumentParser(description="Benchmark attention operations")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=226)  # Default for 14x14 patches
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print(f"\nRunning attention benchmark with following parameters:")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Embedding dimension: {args.dim}")
    print(f"Number of heads: {args.num_heads}")
    print(f"Number of iterations: {args.iterations}")
    print(f"Device: {args.device}\n")
    
    attention_types = {
        "Packed Attention": PackedAttention,
        "Standard Attention": StandardAttention,
        "Flex Attention": FlexAttention
    }
    
    for name, model_class in attention_types.items():
        print(f"\nBenchmarking {name}...")
        mean_time, std_time = benchmark_attention(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            dim=args.dim,
            num_heads=args.num_heads,
            num_iterations=args.iterations,
            device=args.device,
            attention_class=model_class
        )
        print(f"{name} - Mean time: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")

if __name__ == "__main__":
    main()
