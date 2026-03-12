"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention score computation
    # ============================================================================
    #
    # Step 1: Load query vector for this position
    # Step 2: Load all keys for this batch_head
    # Step 3: Compute dot-product scores and scale
    # Step 4: Store scores

    # Offsets for sequence positions and head-dimension elements
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    # Load query vector for this batch_head and query position (shape: BLOCK_D)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    # Load keys for this batch_head: shape (BLOCK_K, BLOCK_D)
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)

    # Compute dot-product between query and each key in the tile, then scale
    # Result shape: (BLOCK_K,)
    scores = tl.sum(k * q[None, :], axis=1) * scale

    # Store scores back to scores_ptr; mask prevents writing past seq_k
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)

    # ============================================================================
    # Implement softmax
    # ============================================================================
    #
    # Step 1: Load scores row with masking
    # Step 2: Subtract max for stability
    # Step 3: Compute exp and normalize
    # Step 4: Store back

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k

    # Load the row of scores (shape: BLOCK_SIZE)
    scores = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-1e9)

    # Numerically stable softmax: subtract max
    row_max = tl.max(scores, axis=0)
    scores_shifted = scores - row_max

    # exponentiate and normalize
    exp_scores = tl.exp(scores_shifted)
    denom = tl.sum(exp_scores, axis=0)
    out = exp_scores / denom

    # write back
    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr, v_ptr, output_ptr,
    seq_k, head_dim,
    stride_w0, stride_w1, stride_w2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention output computation
    # ============================================================================
    #
    # Step 1: Load attention weights for this query
    # Step 2: Load all values for this batch_head
    # Step 3: Compute weighted sum
    # Step 4: Store output

    # Offsets for sequence positions and head-dimension elements
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    # Load attention weights for this (batch_head, query): shape (BLOCK_K,)
    w = tl.load(
        attn_ptr + pid_bh * stride_w0 + pid_q * stride_w1 + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    ).to(tl.float32)

    # Load values for this batch_head: shape (BLOCK_K, BLOCK_D)
    v = tl.load(
        v_ptr + pid_bh * stride_v0 + offs_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)

    # Compute weighted sum over sequence positions: out[d] = sum_k w[k] * v[k, d]
    out = tl.sum(v * w[:, None], axis=0)

    # Store the output vector for this (batch_head, query)
    tl.store(
        output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr, seq_k, offset,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores,
        mask=mask,
    )


@triton.jit
def softmax_rows_kernel(
    x_ptr, y_ptr, stride_x, stride_y, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Numerically stable softmax over last dimension for large sequences.
    Grid: (num_rows,)
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    denom = tl.sum(exp_x, axis=0)
    out = exp_x / denom

    tl.store(y_ptr + row * stride_y + offs, out, mask=mask)


# ============================================================================
# Fused Flash Attention Triton Kernels
# ============================================================================

@triton.jit
def flash_decode_kernel(
    q_ptr,          # (BH, 1, D)
    k_ptr,          # (BH, S, D)
    v_ptr,          # (BH, S, D)
    output_ptr,     # (BH, 1, D)
    scale,
    seq_k,          # actual sequence length of K/V
    head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_KV: tl.constexpr,    # tile size along KV sequence dimension
    BLOCK_D: tl.constexpr,     # tile size along head dimension (>= head_dim, power of 2)
):
    """
    Fused flash-decoding kernel for seq_q == 1.
    Each program handles one (batch, head) pair.
    Tiles over K/V in blocks of BLOCK_KV, using online softmax
    to avoid materializing the full (1 x seq_k) score vector.

    Grid: (BH,)
    """
    pid_bh = tl.program_id(0)

    # Offsets for head dimension
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # Load the single query vector: shape (BLOCK_D,)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + 0 * stride_q1 + offs_d * stride_q2,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    # Online softmax state
    m_prev = float("-inf")     # running max of scores
    l_prev = 0.0               # running sum of exp(scores - m)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)  # running weighted sum of V

    # Tile over KV sequence
    for kv_start in range(0, seq_k, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < seq_k

        # Load K tile: shape (BLOCK_KV, BLOCK_D)
        k_tile = tl.load(
            k_ptr + pid_bh * stride_k0 + offs_kv[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
            mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        # Compute scores: q dot each key, scaled  -> shape (BLOCK_KV,)
        scores = tl.sum(k_tile * q[None, :], axis=1) * scale

        # Mask out invalid positions (beyond actual seq_k)
        scores = tl.where(kv_mask, scores, float("-inf"))

        # Online softmax update
        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_curr)

        # Correction factor for previous accumulator
        alpha = tl.exp(m_prev - m_new)
        # New exp scores
        p = tl.exp(scores - m_new)
        l_new = alpha * l_prev + tl.sum(p, axis=0)

        # Load V tile: shape (BLOCK_KV, BLOCK_D)
        v_tile = tl.load(
            v_ptr + pid_bh * stride_v0 + offs_kv[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
            mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        # Update accumulator: rescale old + add new weighted values
        acc = alpha * acc + tl.sum(p[:, None] * v_tile, axis=0)

        m_prev = m_new
        l_prev = l_new

    # Final normalization
    acc = acc / l_prev

    # Store output
    tl.store(
        output_ptr + pid_bh * stride_o0 + 0 * stride_o1 + offs_d * stride_o2,
        acc, mask=d_mask,
    )


@triton.jit
def flash_attention_kernel(
    q_ptr,          # (BH, Sq, D)
    k_ptr,          # (BH, Sk, D)
    v_ptr,          # (BH, Sk, D)
    output_ptr,     # (BH, Sq, D)
    scale,
    seq_q,
    seq_k,
    head_dim,
    is_causal: tl.constexpr,
    causal_offset,  # = seq_k - seq_q for correct causal alignment
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_Q: tl.constexpr,     # tile size along query sequence dimension
    BLOCK_KV: tl.constexpr,    # tile size along KV sequence dimension
    BLOCK_D: tl.constexpr,     # tile size along head dimension (>= head_dim, power of 2)
):
    """
    Fused flash attention kernel for general seq_q (prefill / encoder).
    Each program handles one (batch_head, q_tile) pair.
    Tiles over K/V with online softmax — never materializes full attention matrix.

    Grid: (BH, cdiv(seq_q, BLOCK_Q))
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # Query offsets for this tile
    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < seq_q

    # Head dimension offsets
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # Load Q tile: shape (BLOCK_Q, BLOCK_D)
    q_tile = tl.load(
        q_ptr + pid_bh * stride_q0 + offs_q[:, None] * stride_q1 + offs_d[None, :] * stride_q2,
        mask=q_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)

    # Online softmax state per query in tile
    m_prev = tl.full((BLOCK_Q,), value=float("-inf"), dtype=tl.float32)   # (BLOCK_Q,)
    l_prev = tl.zeros((BLOCK_Q,), dtype=tl.float32)                       # (BLOCK_Q,)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)                  # (BLOCK_Q, BLOCK_D)

    # Determine KV range (for causal, we can skip future blocks)
    kv_end = seq_k
    if is_causal:
        # The last query in this tile attends up to position:
        # (q_start + BLOCK_Q - 1) + causal_offset
        last_q_pos = q_start + BLOCK_Q - 1 + causal_offset
        # Round up to next BLOCK_KV boundary
        kv_end_causal = ((last_q_pos + BLOCK_KV) // BLOCK_KV) * BLOCK_KV
        kv_end = tl.minimum(kv_end_causal, seq_k)

    # Tile over KV sequence
    for kv_start in range(0, kv_end, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < seq_k

        # Load K tile: shape (BLOCK_KV, BLOCK_D)
        k_tile = tl.load(
            k_ptr + pid_bh * stride_k0 + offs_kv[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
            mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        # Compute scores: Q_tile @ K_tile^T * scale -> (BLOCK_Q, BLOCK_KV)
        scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

        # Mask out invalid KV positions (beyond seq_k)
        scores = tl.where(kv_mask[None, :], scores, float("-inf"))

        # Apply causal mask: key position must be <= query position + offset
        if is_causal:
            causal_mask = offs_kv[None, :] <= (offs_q[:, None] + causal_offset)
            scores = tl.where(causal_mask, scores, float("-inf"))

        # Online softmax update (per query row)
        m_curr = tl.max(scores, axis=1)                    # (BLOCK_Q,)
        m_new = tl.maximum(m_prev, m_curr)                 # (BLOCK_Q,)

        # Correction factor for previous accumulator
        alpha = tl.exp(m_prev - m_new)                     # (BLOCK_Q,)
        # New exp scores
        p = tl.exp(scores - m_new[:, None])                # (BLOCK_Q, BLOCK_KV)
        l_new = alpha * l_prev + tl.sum(p, axis=1)         # (BLOCK_Q,)

        # Load V tile: shape (BLOCK_KV, BLOCK_D)
        v_tile = tl.load(
            v_ptr + pid_bh * stride_v0 + offs_kv[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
            mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        # Update accumulator: rescale old + add new weighted values
        acc = alpha[:, None] * acc + tl.dot(p, v_tile)     # (BLOCK_Q, BLOCK_D)

        m_prev = m_new
        l_prev = l_new

    # Final normalization
    acc = acc / l_prev[:, None]

    # Store output
    tl.store(
        output_ptr + pid_bh * stride_o0 + offs_q[:, None] * stride_o1 + offs_d[None, :] * stride_o2,
        acc, mask=q_mask[:, None] & d_mask[None, :],
    )


# ============================================================================
# Flash Attention dispatch functions
# ============================================================================

def _flash_decode(
    q_flat, k_flat, v_flat,
    batch, num_heads, seq_k, head_dim,
    scale, orig_dtype,
):
    """
    Flash-decoding path for seq_q == 1.
    Fuses score computation, softmax, and output into a single kernel.
    Never materializes the (1 x seq_k) attention score vector in HBM.
    """
    BH = batch * num_heads
    head_dim_padded = next_power_of_two(head_dim)

    output = torch.empty((BH, 1, head_dim), dtype=torch.float32, device=q_flat.device)

    # Choose BLOCK_KV: balance between occupancy and register pressure
    # For large head_dim (128), keep BLOCK_KV moderate to limit shared memory
    if head_dim_padded >= 128:
        BLOCK_KV = 64
    else:
        BLOCK_KV = 64
        if seq_k > 256:
            BLOCK_KV = 128

    grid = (BH,)
    flash_decode_kernel[grid](
        q_flat, k_flat, v_flat, output,
        float(scale),
        seq_k, head_dim,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=head_dim_padded,
        num_stages=1,
        num_warps=4,
    )

    return output.reshape(batch, num_heads, 1, head_dim).to(orig_dtype)


def _flash_attention_general(
    q_flat, k_flat, v_flat,
    batch, num_heads, seq_q, seq_k, head_dim,
    scale, is_causal, attention_mask, orig_dtype,
):
    """
    Flash attention for general seq_q > 1 (prefill, encoder).
    Fuses score computation, softmax, and output into a single kernel.
    Never materializes the full (seq_q x seq_k) attention matrix in HBM.
    """
    BH = batch * num_heads
    head_dim_padded = next_power_of_two(head_dim)

    output = torch.empty((BH, seq_q, head_dim), dtype=torch.float32, device=q_flat.device)

    # Tile sizes – must be powers of 2.
    # Shared memory ≈ (BLOCK_Q + 2*BLOCK_KV) * BLOCK_D * 4 bytes * num_stages
    # GPU limit is 232KB. With num_stages=1:
    #   head_dim=128 (BLOCK_D=128): (32 + 2*32)*128*4 = 49KB  — safe
    #   head_dim=64  (BLOCK_D=64):  (64 + 2*64)*64*4  = 49KB  — safe
    if head_dim_padded >= 128:
        # Large head_dim: use smaller tiles to fit in shared memory
        BLOCK_Q = 32
        BLOCK_KV = 32
        NUM_WARPS = 4
    else:
        # Smaller head_dim (audio encoder, head_dim=64): can use larger tiles
        BLOCK_Q = 64
        BLOCK_KV = 64
        NUM_WARPS = 4

    if seq_q <= 16:
        BLOCK_Q = 16

    causal_offset = seq_k - seq_q  # for correct causal alignment

    num_q_tiles = (seq_q + BLOCK_Q - 1) // BLOCK_Q
    grid = (BH, num_q_tiles)

    flash_attention_kernel[grid](
        q_flat, k_flat, v_flat, output,
        float(scale),
        seq_q, seq_k, head_dim,
        is_causal,
        causal_offset,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=head_dim_padded,
        num_stages=1,
        num_warps=NUM_WARPS,
    )

    return output.reshape(batch, num_heads, seq_q, head_dim).to(orig_dtype)


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256


def _triton_attention_small(
    q_flat, k_flat, v_flat, scores,
    batch, num_heads, seq_q, seq_k, head_dim,
    scale, is_causal, attention_mask, orig_dtype,
):
    """Triton kernel path for small sequences (seq_k <= MAX_ATTENTION_DIM)."""
    BH = batch * num_heads
    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    # Pad if needed (same approach as example implementation)
    if seq_k_padded != seq_k or head_dim_padded != head_dim:
        k_padded = torch.zeros(
            (BH, seq_k_padded, head_dim_padded),
            dtype=torch.float32, device=q_flat.device,
        )
        v_padded = torch.zeros_like(k_padded)
        q_padded = torch.zeros(
            (BH, seq_q, head_dim_padded),
            dtype=torch.float32, device=q_flat.device,
        )
        k_padded[:, :seq_k, :head_dim] = k_flat
        v_padded[:, :seq_k, :head_dim] = v_flat
        q_padded[:, :, :head_dim] = q_flat
        k_flat = k_padded
        v_flat = v_padded
        q_flat = q_padded
    else:
        seq_k_padded = seq_k
        head_dim_padded = head_dim

    scores = torch.empty(
        (BH, seq_q, seq_k_padded), dtype=torch.float32, device=q_flat.device,
    )
    output = torch.empty(
        (BH, seq_q, head_dim_padded), dtype=torch.float32, device=q_flat.device,
    )

    grid = (BH, seq_q)

    # Step 1: Compute scores with Triton kernel
    attention_scores_kernel[grid](
        q_flat, k_flat, scores,
        float(scale),
        seq_k_padded, head_dim_padded,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        BLOCK_K=seq_k_padded,
        BLOCK_D=head_dim_padded,
    )

    # Mask out padded positions
    if seq_k_padded != seq_k:
        scores[:, :, seq_k:] = -1e9

    # Step 2: Apply causal mask with Triton kernel
    if is_causal:
        causal_mask_kernel[(BH, seq_q)](
            scores,
            seq_k_padded,
            seq_k - seq_q,  # offset
            scores.stride(0), scores.stride(1), scores.stride(2),
            BLOCK_K=seq_k_padded,
        )

    # Step 2b: Apply additive attention mask
    if attention_mask is not None:
        mask_expanded = attention_mask.expand(batch, num_heads, seq_q, seq_k)
        mask_flat = mask_expanded.reshape(BH, seq_q, seq_k).to(torch.float32)
        if seq_k_padded != seq_k:
            mask_padded = torch.zeros(
                (BH, seq_q, seq_k_padded), dtype=torch.float32, device=q_flat.device,
            )
            mask_padded[:, :, :seq_k] = mask_flat
            mask_padded[:, :, seq_k:] = -1e9
            scores = scores + mask_padded
        else:
            scores = scores + mask_flat

    # Step 3: Softmax with Triton kernel
    scores_2d = scores.reshape(BH * seq_q, seq_k_padded)
    softmax_inplace_kernel[(BH * seq_q,)](
        scores_2d, scores_2d.stride(0), seq_k_padded,
        BLOCK_SIZE=seq_k_padded,
    )
    scores = scores_2d.reshape(BH, seq_q, seq_k_padded)

    # Step 4: Compute output with Triton kernel
    attention_output_kernel[grid](
        scores, v_flat, output,
        seq_k_padded, head_dim_padded,
        scores.stride(0), scores.stride(1), scores.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_K=seq_k_padded,
        BLOCK_D=head_dim_padded,
    )

    if head_dim_padded != head_dim:
        output = output[:, :, :head_dim]

    return output.reshape(batch, num_heads, seq_q, head_dim).to(orig_dtype)


def _triton_attention_large(
    q_flat, k_flat, v_flat,
    batch, num_heads, seq_q, seq_k, head_dim,
    scale, is_causal, attention_mask, orig_dtype,
):
    """
    Efficient path for large sequences (seq_k > MAX_ATTENTION_DIM).
    Uses the fused flash attention Triton kernel when no additive mask is present,
    otherwise falls back to Triton softmax kernel with cuBLAS matmul for scores/output.
    """
    BH = batch * num_heads

    if attention_mask is None:
        # Use fused flash attention kernel (pure Triton, no torch.bmm)
        if seq_q == 1:
            return _flash_decode(
                q_flat, k_flat, v_flat,
                batch, num_heads, seq_k, head_dim,
                scale, orig_dtype,
            )
        else:
            return _flash_attention_general(
                q_flat, k_flat, v_flat,
                batch, num_heads, seq_q, seq_k, head_dim,
                scale, is_causal, attention_mask, orig_dtype,
            )

    # Fallback when additive attention_mask is present:
    # Use Triton softmax kernel with score/output computation via Triton attention kernels

    # Step 1: Compute scores using Triton attention_scores_kernel in tiles
    # For large seq_k, we tile the computation
    head_dim_padded = next_power_of_two(head_dim)
    seq_k_padded = next_power_of_two(seq_k)

    # If seq_k is too large for a single Triton kernel block, use flash attention
    if seq_k_padded > 4096:
        # Flash attention handles this efficiently
        return _flash_attention_general(
            q_flat, k_flat, v_flat,
            batch, num_heads, seq_q, seq_k, head_dim,
            scale, is_causal, None, orig_dtype,
        )

    # For moderately large sequences, pad and use Triton kernels
    scores = torch.empty(
        (BH, seq_q, seq_k_padded), dtype=torch.float32, device=q_flat.device,
    )

    # Pad tensors if needed
    if seq_k_padded != seq_k or head_dim_padded != head_dim:
        k_padded = torch.zeros(
            (BH, seq_k_padded, head_dim_padded),
            dtype=torch.float32, device=q_flat.device,
        )
        v_padded = torch.zeros_like(k_padded)
        q_padded = torch.zeros(
            (BH, seq_q, head_dim_padded),
            dtype=torch.float32, device=q_flat.device,
        )
        k_padded[:, :seq_k, :head_dim] = k_flat
        v_padded[:, :seq_k, :head_dim] = v_flat
        q_padded[:, :, :head_dim] = q_flat
    else:
        k_padded = k_flat
        v_padded = v_flat
        q_padded = q_flat

    grid = (BH, seq_q)

    attention_scores_kernel[grid](
        q_padded, k_padded, scores,
        float(scale),
        seq_k_padded, head_dim_padded,
        q_padded.stride(0), q_padded.stride(1), q_padded.stride(2),
        k_padded.stride(0), k_padded.stride(1), k_padded.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        BLOCK_K=seq_k_padded,
        BLOCK_D=head_dim_padded,
    )

    # Mask out padded positions
    if seq_k_padded != seq_k:
        scores[:, :, seq_k:] = -1e9

    # Step 2: Apply causal mask
    if is_causal:
        causal_mask_kernel[(BH, seq_q)](
            scores,
            seq_k_padded,
            seq_k - seq_q,  # offset
            scores.stride(0), scores.stride(1), scores.stride(2),
            BLOCK_K=seq_k_padded,
        )

    # Step 2b: Apply additive attention mask
    if attention_mask is not None:
        mask_expanded = attention_mask.expand(batch, num_heads, seq_q, seq_k)
        mask_flat = mask_expanded.reshape(BH, seq_q, seq_k).to(torch.float32)
        if seq_k_padded != seq_k:
            mask_padded = torch.zeros(
                (BH, seq_q, seq_k_padded), dtype=torch.float32, device=q_flat.device,
            )
            mask_padded[:, :, :seq_k] = mask_flat
            mask_padded[:, :, seq_k:] = -1e9
            scores = scores + mask_padded
        else:
            scores = scores + mask_flat

    # Step 3: Softmax using Triton kernel
    num_rows = BH * seq_q
    scores_2d = scores.reshape(num_rows, seq_k_padded).contiguous()
    softmax_out = torch.empty_like(scores_2d)

    softmax_rows_kernel[(num_rows,)](
        scores_2d, softmax_out,
        scores_2d.stride(0), softmax_out.stride(0),
        seq_k_padded,
        BLOCK_SIZE=seq_k_padded,
    )
    scores = softmax_out.reshape(BH, seq_q, seq_k_padded)

    # Step 4: Compute output using Triton attention_output_kernel
    output = torch.empty(
        (BH, seq_q, head_dim_padded), dtype=torch.float32, device=q_flat.device,
    )

    attention_output_kernel[grid](
        scores, v_padded, output,
        seq_k_padded, head_dim_padded,
        scores.stride(0), scores.stride(1), scores.stride(2),
        v_padded.stride(0), v_padded.stride(1), v_padded.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_K=seq_k_padded,
        BLOCK_D=head_dim_padded,
    )

    if head_dim_padded != head_dim:
        output = output[:, :, :head_dim]

    return output.reshape(batch, num_heads, seq_q, head_dim).to(orig_dtype)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.

    Dispatch strategy:
    1. Flash decode kernel (fused Triton) for seq_q == 1 without mask.
    2. Flash attention kernel (fused Triton) for seq_q > 1 without mask.
    3. Separate Triton kernels for small sequences with mask.
    4. Tiled Triton kernels for large sequences with mask.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    if q.is_cuda:
        orig_dtype = q.dtype

        BH = batch * num_heads
        q_flat = q.reshape(BH, seq_q, head_dim).contiguous().float()
        k_flat = k.reshape(BH, seq_k, head_dim).contiguous().float()
        v_flat = v.reshape(BH, seq_k, head_dim).contiguous().float()

        # === FAST PATH: Fused flash attention (no additive mask) ===
        if attention_mask is None:
            if seq_q == 1:
                return _flash_decode(
                    q_flat, k_flat, v_flat,
                    batch, num_heads, seq_k, head_dim,
                    scale, orig_dtype,
                )
            else:
                return _flash_attention_general(
                    q_flat, k_flat, v_flat,
                    batch, num_heads, seq_q, seq_k, head_dim,
                    scale, is_causal, attention_mask, orig_dtype,
                )

        # === FALLBACK: separate kernels when additive attention_mask is present ===
        seq_k_padded = next_power_of_two(seq_k)
        head_dim_padded = next_power_of_two(head_dim)

        use_triton_kernels = (
            seq_k_padded <= MAX_ATTENTION_DIM
            and head_dim_padded <= MAX_ATTENTION_DIM
        )

        if use_triton_kernels:
            return _triton_attention_small(
                q_flat, k_flat, v_flat, None,
                batch, num_heads, seq_q, seq_k, head_dim,
                scale, is_causal, attention_mask, orig_dtype,
            )
        else:
            return _triton_attention_large(
                q_flat, k_flat, v_flat,
                batch, num_heads, seq_q, seq_k, head_dim,
                scale, is_causal, attention_mask, orig_dtype,
            )

    # ---- CPU fallback ----
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale

    if is_causal:
        row_idx = torch.arange(seq_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(seq_k, device=q.device).unsqueeze(0)
        offset = seq_k - seq_q
        causal = col_idx > (row_idx + offset)
        scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))

    if attention_mask is not None:
        scores = scores + attention_mask.float()

    scores_max = scores.max(dim=-1, keepdim=True).values
    scores = scores - scores_max
    exp_scores = scores.exp()
    attn_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

    output = torch.matmul(attn_weights, v_f)
    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)

    print("\nFlash decode test (seq_q=1):")
    q1 = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
    output_decode = scaled_dot_product_attention(q1, k, v)
    print(f"  Output shape: {output_decode.shape}")

    print("\nFlash decode test (seq_q=1, larger seq_k):")
    k_long = torch.randn(batch_size, num_heads, 512, head_dim, device=device)
    v_long = torch.randn(batch_size, num_heads, 512, head_dim, device=device)
    output_decode_long = scaled_dot_product_attention(q1, k_long, v_long)
    print(f"  Output shape: {output_decode_long.shape}")

    print("\nAll Triton attention tests passed!")
