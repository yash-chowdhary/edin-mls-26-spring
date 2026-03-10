"""
Triton Rotary Position Embeddings (RoPE)
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement RoPE using Triton kernels
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for RoPE
# ============================================================================

@triton.jit
def compute_freqs_kernel(
    positions_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
    seq_len, half_dim,
    stride_pos, stride_inv,
    stride_cos0, stride_cos1,
    stride_sin0, stride_sin1,
    BLOCK: tl.constexpr,
):
    # Process one sequence position per program
    row_idx = tl.program_id(0)
    if row_idx >= seq_len:
        return

    # 1. Load data
    # Standard: Load position scalar and the inv_freq vector
    pos = tl.load(positions_ptr + row_idx * stride_pos).to(tl.float32)
    
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim
    inv_freq = tl.load(inv_freq_ptr + offs * stride_inv, mask=mask, other=0.0).to(tl.float32)

    # 2. Compute math in float32 for accuracy
    freqs = pos * inv_freq
    cos_val = tl.cos(freqs)
    sin_val = tl.sin(freqs)

    # 3. Vectorized Stores: Interleave the halves in registers
    # We create the full rotary_dim (half_dim * 2) in-register
    # [cos_half, cos_half] and [sin_half, sin_half]
    
    # Offsets for the full row
    offs_full = tl.arange(0, BLOCK * 2)
    mask_full = offs_full < (half_dim * 2)
    
    # Duplicate the values for the "concatenated" RoPE format
    # Note: tl.join is faster than multiple stores if your Triton version supports it,
    # otherwise, we use calculated offsets for coalescing.
    
    out_offs = row_idx * stride_cos0 + offs * stride_cos1
    
    # Store first half and second half in rapid succession
    tl.store(cos_ptr + out_offs, cos_val.to(tl.bfloat16), mask=mask)
    tl.store(cos_ptr + out_offs + (half_dim * stride_cos1), cos_val.to(tl.bfloat16), mask=mask)
    
    tl.store(sin_ptr + row_idx * stride_sin0 + offs * stride_sin1, sin_val.to(tl.bfloat16), mask=mask)
    tl.store(sin_ptr + row_idx * stride_sin0 + (offs + half_dim) * stride_sin1, sin_val.to(tl.bfloat16), mask=mask)


# ============================================================================
# RoPE Classes
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding optimized for BFloat16 and Triton."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        # Keep inv_freq as float32 for the kernel math
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos and sin into BF16 cache."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        # Optimized: Allocate directly in BFloat16 to save 50% HBM
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.bfloat16, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.bfloat16, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)

            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
                BLOCK=block,
            )
        else:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]
            cos_half = torch.cos(freqs).to(torch.bfloat16)
            sin_half = torch.sin(freqs).to(torch.bfloat16)
            cos_cache[:, :half_dim] = cos_half
            cos_cache[:, half_dim:] = cos_half
            sin_cache[:, :half_dim] = sin_half
            sin_cache[:, half_dim:] = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin without costly dtype conversions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        if position_ids is not None:
            # Slice and return - already in BFloat16!
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]

        return cos, sin


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ROPE_DIM = 256


def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K) using Torch."""
    batch, num_heads, seq_len, _ = x.shape
    
    # Pre-allocate the full output once
    out = torch.empty_like(x)

    # Prepare cos/sin (only the parts we need)
    # Ensure they are the same dtype as x (BFloat16) to avoid mid-math casting
    cos = cos[:seq_len, :half_dim].to(x.dtype)
    sin = sin[:seq_len, :half_dim].to(x.dtype)

    # 1. Rotate the parts that need it
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim : half_dim * 2]

    # Fill the output directly
    out[..., :half_dim] = x1 * cos - x2 * sin
    out[..., half_dim : half_dim * 2] = x2 * cos + x1 * sin

    # 2. Copy the "pass-through" part if it exists
    if head_dim > half_dim * 2:
        out[..., half_dim * 2 :] = x[..., half_dim * 2 :]
        
    return out


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings.
    """
    # batch, num_q_heads, seq_len, head_dim = q.shape
    # _, num_kv_heads, _, _ = k.shape

    # if rotary_dim is None:
    #     rotary_dim = head_dim

    seq_len = q.shape[2] 
    rotary_dim = rotary_dim or q.shape[-1]

    half_dim = rotary_dim // 2

    # if cos.shape[1] > half_dim:
    #     cos = cos[:, :half_dim]
    #     sin = sin[:, :half_dim]

    # cos = cos.to(torch.float32).contiguous()
    # sin = sin.to(torch.float32).contiguous()

    cos = cos[:seq_len, :half_dim]
    sin = sin[:seq_len, :half_dim]

    # q_out = _apply_rope_single(q, cos, sin, half_dim, head_dim)
    # k_out = _apply_rope_single(k, cos, sin, half_dim, head_dim)
    q_out = _apply_rope_single(q, cos, sin, half_dim, q.shape[-1])
    k_out = _apply_rope_single(k, cos, sin, half_dim, k.shape[-1])

    return q_out, k_out.to(k.dtype)
    # return q_out.to(q.dtype), k_out.to(k.dtype)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Triton RoPE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    print("\nTesting partial RoPE (50%):")
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
    print(f"Q rotated (partial) shape: {q_rot_p.shape}")

    print("\nTriton RoPE working!")
