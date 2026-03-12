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
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """
    Compute cos and sin for rotary embeddings.

    *** TODO: Implement this kernel ***

    Grid: (seq_len,)
    """
    pid = tl.program_id(0)

    # Safety: if program index is out of range, exit
    if pid >= seq_len:
        return

    # Load the scalar position for this row
    pos = tl.load(positions_ptr + pid * stride_pos)

    # Vector indices within a block
    ids = tl.arange(0, BLOCK)

    # Only entries < half_dim are valid
    mask = ids < half_dim

    # Load inverse frequencies (masked)
    inv = tl.load(inv_freq_ptr + ids * stride_inv, mask=mask)

    # freqs = position * inv_freq
    freqs = pos * inv

    # Compute cos and sin for the half-dimension
    cos_half = tl.cos(freqs)
    sin_half = tl.sin(freqs)

    # Store cos: first half
    cos_addr = cos_ptr + pid * stride_cos0 + ids * stride_cos1
    tl.store(cos_addr, cos_half, mask=mask)

    # Store cos: second half (duplicate the half block)
    cos_addr2 = cos_ptr + pid * stride_cos0 + (ids + half_dim) * stride_cos1
    tl.store(cos_addr2, cos_half, mask=mask)

    # Store sin: first half
    sin_addr = sin_ptr + pid * stride_sin0 + ids * stride_sin1
    tl.store(sin_addr, sin_half, mask=mask)

    # Store sin: second half
    sin_addr2 = sin_ptr + pid * stride_sin0 + (ids + half_dim) * stride_sin1
    tl.store(sin_addr2, sin_half, mask=mask)


@triton.jit
def fused_rope_kernel(
    x_ptr,          # input tensor  (B*H, seq_len, head_dim)
    cos_ptr,        # cos values    (seq_len, half_dim)
    sin_ptr,        # sin values    (seq_len, half_dim)
    out_ptr,        # output tensor (B*H, seq_len, head_dim)
    half_dim,       # number of pairs to rotate
    head_dim,       # full head dimension
    seq_len,        # sequence length
    stride_xbh, stride_xs, stride_xd,   # x strides: (batch_head, seq, dim)
    stride_cs, stride_cd,                # cos/sin strides
    stride_obh, stride_os, stride_od,   # output strides
    BLOCK_HD: tl.constexpr,   # >= half_dim, power of 2
):
    """
    Fused RoPE kernel: applies rotary embeddings in a single pass.

    For each element in the rotary portion:
        out[..., i]            = x[..., i]            * cos[s, i] - x[..., i + half_dim] * sin[s, i]
        out[..., i + half_dim] = x[..., i + half_dim] * cos[s, i] + x[..., i]            * sin[s, i]

    Elements beyond 2*half_dim are copied through unchanged.

    Grid: (B * num_heads * seq_len,)
    Each program processes one (batch_head, seq_pos) vector of size head_dim.
    """
    pid = tl.program_id(0)

    # Decompose flat pid into (batch_head, seq_pos)
    s = pid % seq_len
    bh = pid // seq_len

    offs_half = tl.arange(0, BLOCK_HD)
    mask_half = offs_half < half_dim

    # Load x1 = x[..., :half_dim] and x2 = x[..., half_dim:2*half_dim]
    x_base = bh * stride_xbh + s * stride_xs
    x1 = tl.load(x_ptr + x_base + offs_half * stride_xd, mask=mask_half, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + x_base + (offs_half + half_dim) * stride_xd, mask=mask_half, other=0.0).to(tl.float32)

    # Load cos and sin for this sequence position
    cos_val = tl.load(cos_ptr + s * stride_cs + offs_half * stride_cd, mask=mask_half, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + s * stride_cs + offs_half * stride_cd, mask=mask_half, other=0.0).to(tl.float32)

    # Apply rotation
    out1 = x1 * cos_val - x2 * sin_val
    out2 = x2 * cos_val + x1 * sin_val

    # Store rotated halves
    o_base = bh * stride_obh + s * stride_os
    tl.store(out_ptr + o_base + offs_half * stride_od, out1, mask=mask_half)
    tl.store(out_ptr + o_base + (offs_half + half_dim) * stride_od, out2, mask=mask_half)

    # Copy through any remaining dimensions beyond 2*half_dim
    # For most configs this is a no-op (head_dim == 2*half_dim),
    # but for partial RoPE (audio encoder) we need to copy the passthrough dims.
    remaining = head_dim - 2 * half_dim
    if remaining > 0:
        offs_rest = tl.arange(0, BLOCK_HD)
        mask_rest = offs_rest < remaining
        rest_in = tl.load(
            x_ptr + x_base + (2 * half_dim + offs_rest) * stride_xd,
            mask=mask_rest, other=0.0,
        )
        tl.store(
            out_ptr + o_base + (2 * half_dim + offs_rest) * stride_od,
            rest_in, mask=mask_rest,
        )


@triton.jit
def fused_rope_pair_kernel(
    q_ptr,          # query  (B*Hq, S, D)
    k_ptr,          # key    (B*Hk, S, D)
    cos_ptr,        # cos    (S, half_dim)
    sin_ptr,        # sin    (S, half_dim)
    qo_ptr,         # q out  (B*Hq, S, D)
    ko_ptr,         # k out  (B*Hk, S, D)
    half_dim,
    head_dim,
    seq_len,
    total_qh,       # B * num_q_heads
    total_kh,       # B * num_kv_heads
    stride_qs, stride_qd,   # q strides (seq, dim) — batch_head stride = stride_qs * seq_len
    stride_ks, stride_kd,
    stride_cs, stride_cd,
    stride_qos, stride_qod,
    stride_kos, stride_kod,
    BLOCK_HD: tl.constexpr,
):
    """
    Fused RoPE kernel that processes BOTH Q and K in a single grid launch.
    Grid: ((total_qh + total_kh) * seq_len,)

    Programs 0..total_qh*seq_len-1 handle Q, the rest handle K.
    This halves kernel launch overhead (1 launch instead of 2).
    """
    pid = tl.program_id(0)
    total_q_programs = total_qh * seq_len

    is_q = pid < total_q_programs

    # Branch: Q or K
    if is_q:
        bh = pid // seq_len
        s = pid % seq_len
        x_ptr = q_ptr
        o_ptr = qo_ptr
        stride_s = stride_qs
        stride_d = stride_qd
        stride_os_val = stride_qos
        stride_od_val = stride_qod
        stride_bh = stride_qs * seq_len
        stride_obh = stride_qos * seq_len
    else:
        local_pid = pid - total_q_programs
        bh = local_pid // seq_len
        s = local_pid % seq_len
        x_ptr = k_ptr
        o_ptr = ko_ptr
        stride_s = stride_ks
        stride_d = stride_kd
        stride_os_val = stride_kos
        stride_od_val = stride_kod
        stride_bh = stride_ks * seq_len
        stride_obh = stride_kos * seq_len

    offs_half = tl.arange(0, BLOCK_HD)
    mask_half = offs_half < half_dim

    # Load x1 and x2
    base = bh * stride_bh + s * stride_s
    x1 = tl.load(x_ptr + base + offs_half * stride_d, mask=mask_half, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + (offs_half + half_dim) * stride_d, mask=mask_half, other=0.0).to(tl.float32)

    # Load cos/sin
    cos_val = tl.load(cos_ptr + s * stride_cs + offs_half * stride_cd, mask=mask_half, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + s * stride_cs + offs_half * stride_cd, mask=mask_half, other=0.0).to(tl.float32)

    # Rotate
    out1 = x1 * cos_val - x2 * sin_val
    out2 = x2 * cos_val + x1 * sin_val

    # Store
    obase = bh * stride_obh + s * stride_os_val
    tl.store(o_ptr + obase + offs_half * stride_od_val, out1, mask=mask_half)
    tl.store(o_ptr + obase + (offs_half + half_dim) * stride_od_val, out2, mask=mask_half)

    # Copy passthrough dimensions (for partial RoPE, e.g. audio encoder)
    remaining = head_dim - 2 * half_dim
    if remaining > 0:
        offs_rest = tl.arange(0, BLOCK_HD)
        mask_rest = offs_rest < remaining
        rest_in = tl.load(x_ptr + base + (2 * half_dim + offs_rest) * stride_d, mask=mask_rest, other=0.0)
        tl.store(o_ptr + obase + (2 * half_dim + offs_rest) * stride_od_val, rest_in, mask=mask_rest)


# ============================================================================
# RoPE Classes
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""

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

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

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
            cos_half = torch.cos(freqs)
            sin_half = torch.sin(freqs)
            cos_cache[:, :half_dim] = cos_half
            cos_cache[:, half_dim : half_dim * 2] = cos_half
            sin_cache[:, :half_dim] = sin_half
            sin_cache[:, half_dim : half_dim * 2] = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

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
    """Apply RoPE to a single tensor (Q or K) — fused Triton kernel on CUDA."""
    batch, num_heads, seq_len, _ = x.shape

    cos = cos[:seq_len]
    sin = sin[:seq_len]

    # --- CUDA fast path: fused Triton kernel ---
    if x.is_cuda:
        BH = batch * num_heads
        BLOCK_HD = next_power_of_two(max(half_dim, head_dim - 2 * half_dim, 1))

        # Flatten (batch, num_heads) -> (B*H) for the kernel
        x_flat = x.reshape(BH, seq_len, head_dim).contiguous()
        out_flat = torch.empty_like(x_flat)

        # cos/sin are (seq_len, rotary_dim) — we only need first half_dim columns
        cos_half = cos[:, :half_dim].contiguous()
        sin_half = sin[:, :half_dim].contiguous()

        grid = (BH * seq_len,)
        fused_rope_kernel[grid](
            x_flat, cos_half, sin_half, out_flat,
            half_dim, head_dim, seq_len,
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            cos_half.stride(0), cos_half.stride(1),
            out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
            BLOCK_HD=BLOCK_HD,
            num_stages=1,
            num_warps=4,
        )

        return out_flat.reshape(batch, num_heads, seq_len, head_dim)

    # --- CPU fallback: original PyTorch path ---
    output = torch.empty_like(x)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim : half_dim * 2]

    cos_expanded = cos[None, None, :, :]
    sin_expanded = sin[None, None, :, :]

    output[..., :half_dim] = x1 * cos_expanded - x2 * sin_expanded
    output[..., half_dim : half_dim * 2] = x2 * cos_expanded + x1 * sin_expanded

    if head_dim > half_dim * 2:
        output[..., half_dim * 2 :] = x[..., half_dim * 2 :]

    return output


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings.
    On CUDA, uses a single fused Triton kernel launch for both Q and K.
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    if cos.shape[1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

    # Only convert dtype/contiguity if needed
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if sin.dtype != torch.float32:
        sin = sin.to(torch.float32)
    if not sin.is_contiguous():
        sin = sin.contiguous()

    # --- CUDA fast path: single fused kernel launch for both Q and K ---
    if q.is_cuda:
        total_qh = batch * num_q_heads
        total_kh = batch * num_kv_heads
        BLOCK_HD = next_power_of_two(max(half_dim, head_dim - 2 * half_dim, 1))

        # Flatten (batch, num_heads) into first dim — only call .contiguous() if needed
        q_flat = q.reshape(total_qh, seq_len, head_dim)
        if not q_flat.is_contiguous():
            q_flat = q_flat.contiguous()
        k_flat = k.reshape(total_kh, seq_len, head_dim)
        if not k_flat.is_contiguous():
            k_flat = k_flat.contiguous()
        qo_flat = torch.empty_like(q_flat)
        ko_flat = torch.empty_like(k_flat)

        # Slice cos/sin to seq_len — avoid .contiguous() if already contiguous
        cos_half = cos[:seq_len]
        if not cos_half.is_contiguous():
            cos_half = cos_half.contiguous()
        sin_half = sin[:seq_len]
        if not sin_half.is_contiguous():
            sin_half = sin_half.contiguous()

        total_programs = (total_qh + total_kh) * seq_len
        grid = (total_programs,)

        fused_rope_pair_kernel[grid](
            q_flat, k_flat,
            cos_half, sin_half,
            qo_flat, ko_flat,
            half_dim, head_dim, seq_len,
            total_qh, total_kh,
            q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(1), k_flat.stride(2),
            cos_half.stride(0), cos_half.stride(1),
            qo_flat.stride(1), qo_flat.stride(2),
            ko_flat.stride(1), ko_flat.stride(2),
            BLOCK_HD=BLOCK_HD,
            num_stages=1,
            num_warps=4,
        )

        q_out = qo_flat.reshape(batch, num_q_heads, seq_len, head_dim)
        k_out = ko_flat.reshape(batch, num_kv_heads, seq_len, head_dim)
        return q_out.to(q.dtype), k_out.to(k.dtype)

    # --- CPU fallback ---
    q_out = _apply_rope_single(q, cos, sin, half_dim, head_dim)
    k_out = _apply_rope_single(k, cos, sin, half_dim, head_dim)

    return q_out.to(q.dtype), k_out.to(k.dtype)


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
