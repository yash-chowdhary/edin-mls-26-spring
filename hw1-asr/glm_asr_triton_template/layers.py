"""
Triton Neural Network Layers
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement core layers using Triton kernels
"""

import os
import math
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl
from benchmark_utils import profiled, Profiler, microbenchmark

# ============================================================================
# Tunable Parameters (read from environment variables, with sensible defaults)
# ============================================================================
GELU_BLOCK_SIZE = int(os.getenv("GELU_BLOCK_SIZE", "1024"))
SILU_BLOCK_SIZE = int(os.getenv("SILU_BLOCK_SIZE", "1024"))
SOFTMAX_BLOCK_SIZE = int(os.getenv("SOFTMAX_BLOCK_SIZE", "0"))  # 0 = auto (next_power_of_two)
LINEAR_TILE_M = int(os.getenv("LINEAR_TILE_M", "64"))
LINEAR_TILE_N = int(os.getenv("LINEAR_TILE_N", "64"))
LINEAR_TILE_K = int(os.getenv("LINEAR_TILE_K", "32"))
LINEAR_BACKEND = os.getenv("LINEAR_BACKEND", "")  # "" = use default from __init__.py / class attr


# ============================================================================
# Helper Functions
# ============================================================================

def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


def pad_to_multiple(size: int, multiple: int) -> int:
    """Pad size to be a multiple of the given value."""
    return ((size + multiple - 1) // multiple) * multiple


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def rmsnorm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        stride_x,
        stride_y,
        hidden_size,
        eps,
        BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm: x / RMS(x) * weight

    Grid: (batch_size,)
    """
    pid = tl.program_id(0)

    # ==========================================================================
    # TODO: Implement RMSNorm kernel
    # ==========================================================================
    #
    # Step 1: Load input row and weight
    # Step 2: Compute variance = mean(x^2)
    # Step 3: Normalize: x / sqrt(variance + eps)
    # Step 4: Apply weight and store

    # Compute offsets for this program id and load a tile of the input
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    # Load input (row) and cast to float32 for accumulation
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute variance = mean(x^2)
    var = tl.sum(x * x, axis=0) / hidden_size

    # Normalize using rsqrt for numerical stability
    x_norm = x * tl.rsqrt(var + eps)

    # Load weight and apply (weight may be shorter/padded; mask protects)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = x_norm * w

    # Store the result
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)


@triton.jit
def rmsnorm_fp16_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        stride_x,
        stride_y,
        hidden_size,
        eps,
        BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm that outputs float16 directly.
    Avoids the fp32→HBM→fp16 round-trip when feeding into cuBLAS Linear.

    Grid: (batch_size,)
    """
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    # Load input row in float32 for numerically-stable normalization
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute variance = mean(x^2) and normalize
    var = tl.sum(x * x, axis=0) / hidden_size
    x_norm = x * tl.rsqrt(var + eps)

    # Load weight and apply
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w

    # Store result directly as float16 — ready for cuBLAS matmul
    tl.store(y_ptr + pid * stride_y + offs, y.to(tl.float16), mask=mask)


@triton.jit
def layernorm_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        stride_x,
        stride_y,
        hidden_size,
        eps,
        BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias

    Grid: (batch_size,)
    """
    pid = tl.program_id(0)

    # ============================================================================
    # TODO: Implement LayerNorm kernel
    # ============================================================================
    #
    # Step 1: Load input, weight, and bias
    # Step 2: Compute mean
    # Step 3: Center the data
    # Step 4: Compute variance = mean((x - mean)^2)
    # Step 5: Normalize and apply affine transform

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    # Load input row and cast to float32 for numerically-stable ops
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute mean across hidden dimension
    mean = tl.sum(x, axis=0) / hidden_size

    # Center and compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_size

    # Normalize
    x_norm = x_centered * tl.rsqrt(var + eps)

    # Load parameters and apply affine transform
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    y = x_norm * w + b

    # Store result
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    GELU using tanh approximation.

    *** TODO: Implement this kernel ***
    """
    pid = tl.program_id(0)

    # ============================================================================
    # TODO: Implement GELU kernel
    # ============================================================================
    #
    # Step 1: Calculate offsets and load input tile
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Step 2: Compute tanh approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    tanh_inner = tl.extra.cuda.libdevice.tanh(inner)

    # Step 3: Final GELU calculation
    y = 0.5 * x * (1.0 + tanh_inner)

    # Step 4: Store output
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    SiLU/Swish: x * sigmoid(x)

    *** TODO: Implement this kernel ***
    """
    pid = tl.program_id(0)

    # ============================================================================
    # TODO: Implement SiLU kernel
    # ============================================================================
    #
    # Step 1: Load input tile
    # Step 2: Compute sigmoid
    # Step 3: Multiply and store

    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    sigmoid = 1.0 / (1.0 + tl.exp(-x))

    y = x * sigmoid

    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def linear_kernel_tf32(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    """
    TF32-style matmul: output = A @ B.
    A: (M, K), B: (K, N), C: (M, N)

    *** TODO: Implement this kernel ***

    Grid: (M // BLOCK_M, N // BLOCK_N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ============================================================================
    # TODO: Implement tiled matrix multiplication
    # ============================================================================
    #
    # Step 1: Initialize accumulator
    # Step 2: Loop over K tiles and accumulate tl.dot
    # Step 3: Store the result

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k in range(0, K, BLOCK_K):
        # Load tile from A: shape (BLOCK_M, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        # Load tile from B: shape (BLOCK_K, BLOCK_N)
        b = tl.load(
            b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        # Accumulate: matrix multiply and add to accumulator
        acc += tl.dot(a, b)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def linear_gelu_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    """Fused Linear + GELU."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, b)

    sqrt_2_over_pi = 0.7978845608028654
    acc3 = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc3)
    acc = acc * 0.5 * (1.0 + tl.libdevice.tanh(inner))

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def swiglu_fused_kernel(
        a_ptr,
        gate_ptr,
        up_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_gk,
        stride_gn,
        stride_uk,
        stride_un,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    """Fused SwiGLU: SiLU(x @ gate) * (x @ up)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        gate_w = tl.load(
            gate_ptr + (k + offs_k[:, None]) * stride_gk + offs_n[None, :] * stride_gn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        up_w = tl.load(
            up_ptr + (k + offs_k[:, None]) * stride_uk + offs_n[None, :] * stride_un,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        gate_acc += tl.dot(a, gate_w)
        up_acc += tl.dot(a, up_w)

    sigmoid = 1.0 / (1.0 + tl.exp(-gate_acc))
    gate_act = gate_acc * sigmoid
    out = gate_act * up_acc

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def embedding_kernel(
        indices_ptr,
        weight_ptr,
        output_ptr,
        embedding_dim,
        stride_w0,
        stride_w1,
        stride_out0,
        BLOCK_SIZE: tl.constexpr,
):
    """Embedding lookup using gather."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    idx = tl.load(indices_ptr + pid0)
    offs = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < embedding_dim
    w = tl.load(
        weight_ptr + idx * stride_w0 + offs * stride_w1, mask=mask, other=0.0
    )
    tl.store(output_ptr + pid0 * stride_out0 + offs, w, mask=mask)


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Numerically stable softmax over last dimension.

    *** TODO: Implement this kernel ***
    """
    row = tl.program_id(0)

    # ============================================================================
    # TODO: Implement softmax kernel
    # ============================================================================
    #

    # Step 1: Calculate offsets and load the row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float('inf'))

    # Step 2: Find max for numerical stability
    x_max = tl.max(x, axis=0)

    # Step 3: Subtract max and compute exp
    x_shifted = x - x_max
    numerator = tl.exp(x_shifted)

    # Step 4: Compute sum of exponentials
    denominator = tl.sum(numerator, axis=0)

    # Step 5: Normalize
    output = numerator / denominator

    # Step 6: Store result
    tl.store(y_ptr + row * stride_y + offs, output, mask=mask)


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
    """Compute attention scores: Q @ K^T * scale."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def attention_output_kernel(
        weights_ptr,
        v_ptr,
        output_ptr,
        seq_k,
        head_dim,
        stride_w0,
        stride_w1,
        stride_w2,
        stride_v0,
        stride_v1,
        stride_v2,
        stride_o0,
        stride_o1,
        stride_o2,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
):
    """Compute attention output: weights @ V."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    w = tl.load(
        weights_ptr
        + pid_bh * stride_w0
        + pid_q * stride_w1
        + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
        scores_ptr,
        seq_k,
        offset,
        stride_s0,
        stride_s1,
        stride_s2,
        BLOCK_K: tl.constexpr,
):
    """Apply causal mask to attention scores."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


# ============================================================================
# Layer Classes
# ============================================================================

def _is_power_of_two(x: int) -> bool:
    """Check if x is a power of two."""
    return x > 0 and (x & (x - 1)) == 0


class RMSNorm:
    """Root Mean Square Normalization using Triton with Torch fallback.

    Outputs float16 directly via fused rmsnorm_fp16_kernel, eliminating
    the intermediate fp32 tensor round-trip through HBM when feeding
    into cuBLAS Linear projections (which expect fp16 input).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.use_triton = _is_power_of_two(hidden_size)

    @profiled("RMSNorm.__call__")
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

        if self.use_triton and x.is_cuda:
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            x_flat = x_flat.to(torch.float32)
            # Output fp16 directly — skips fp32→HBM→fp16 round-trip
            output = torch.empty(
                (batch_size, self.hidden_size),
                dtype=torch.float16,
                device=x.device,
            )

            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)

            rmsnorm_fp16_kernel[(batch_size,)](
                x_flat,
                self.weight,
                output,
                x_flat.stride(0),
                output.stride(0),
                self.hidden_size,
                self.eps,
                BLOCK_SIZE=next_power_of_two(self.hidden_size),
            )
            return output.reshape(original_shape)

        # Fallback: compute in fp32, output as fp16 for consistency
        x_float = x.to(torch.float32)
        variance = torch.mean(x_float * x_float, dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        return (self.weight * x_normed).to(torch.float16)


def fused_rmsnorm_linear(x, norm, linear):
    """
    Fused RMSNorm -> Linear: RMSNorm outputs fp16 directly into cuBLAS matmul,
    eliminating the intermediate fp32 tensor round-trip through HBM.
    """
    original_shape = x.shape
    hidden_size = norm.hidden_size
    batch_dims = original_shape[:-1]
    M = int(np.prod(batch_dims))

    if norm.weight.device != x.device:
        norm.weight = norm.weight.to(x.device)

    if norm.use_triton and x.is_cuda:
        x_flat = x.reshape(M, hidden_size).contiguous().to(torch.float32)
        normed_fp16 = torch.empty((M, hidden_size), dtype=torch.float16, device=x.device)

        rmsnorm_fp16_kernel[(M,)](
            x_flat,
            norm.weight,
            normed_fp16,
            x_flat.stride(0),
            normed_fp16.stride(0),
            hidden_size,
            norm.eps,
            BLOCK_SIZE=next_power_of_two(hidden_size),
        )

        if linear.weight is None and linear._weight_t_padded is not None:
            if linear._weight_t_padded.device != x.device:
                linear._weight_t_padded = linear._weight_t_padded.to(x.device)
            weight_t = linear._weight_t_padded[:linear.in_features, :linear.out_features].to(torch.float16)
            output = (normed_fp16 @ weight_t).float()
        else:
            if linear.weight.device != x.device:
                linear.weight = linear.weight.to(x.device)
            output = (normed_fp16 @ linear.weight.to(torch.float16).t()).float()

        if linear.has_bias and linear.bias_param is not None:
            if linear.bias_param.device != x.device:
                linear.bias_param = linear.bias_param.to(x.device)
            output = output + linear.bias_param.to(torch.float32)

        return output.reshape(*batch_dims, linear.out_features)

    return linear(norm(x))


def fused_rmsnorm_multi_linear(x, norm, linears):
    """
    Fused RMSNorm -> multiple Linears: computes norm ONCE as fp16, then
    feeds into multiple cuBLAS matmuls (Q, K, V projections).
    Eliminates redundant fp32 intermediate + dtype conversion per projection.
    """
    original_shape = x.shape
    hidden_size = norm.hidden_size
    batch_dims = original_shape[:-1]
    M = int(np.prod(batch_dims))

    if norm.weight.device != x.device:
        norm.weight = norm.weight.to(x.device)

    if norm.use_triton and x.is_cuda:
        x_flat = x.reshape(M, hidden_size).contiguous().to(torch.float32)
        normed_fp16 = torch.empty((M, hidden_size), dtype=torch.float16, device=x.device)

        rmsnorm_fp16_kernel[(M,)](
            x_flat,
            norm.weight,
            normed_fp16,
            x_flat.stride(0),
            normed_fp16.stride(0),
            hidden_size,
            norm.eps,
            BLOCK_SIZE=next_power_of_two(hidden_size),
        )

        results = []
        for linear in linears:
            if linear.weight is None and linear._weight_t_padded is not None:
                if linear._weight_t_padded.device != x.device:
                    linear._weight_t_padded = linear._weight_t_padded.to(x.device)
                weight_t = linear._weight_t_padded[:linear.in_features, :linear.out_features].to(torch.float16)
                out = (normed_fp16 @ weight_t).float()
            else:
                if linear.weight.device != x.device:
                    linear.weight = linear.weight.to(x.device)
                out = (normed_fp16 @ linear.weight.to(torch.float16).t()).float()

            if linear.has_bias and linear.bias_param is not None:
                if linear.bias_param.device != x.device:
                    linear.bias_param = linear.bias_param.to(x.device)
                out = out + linear.bias_param.to(torch.float32)

            results.append(out.reshape(*batch_dims, linear.out_features))
        return results

    normed = norm(x)
    return [lin(normed) for lin in linears]


class LayerNorm:
    """Layer Normalization using Triton with Torch fallback."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.bias = torch.zeros(hidden_size, dtype=torch.float32)
        self.use_triton = _is_power_of_two(hidden_size)

    @profiled("LayerNorm.__call__")
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

        if self.use_triton and x.is_cuda:
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            x_flat = x_flat.to(torch.float32)
            output = torch.empty_like(x_flat)

            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            if self.bias.device != x.device:
                self.bias = self.bias.to(x.device)

            layernorm_kernel[(batch_size,)](
                x_flat,
                self.weight,
                self.bias,
                output,
                x_flat.stride(0),
                output.stride(0),
                self.hidden_size,
                self.eps,
                BLOCK_SIZE=next_power_of_two(self.hidden_size),
            )
            return output.reshape(original_shape)

        x_float = x.to(torch.float32)
        mean = torch.mean(x_float, dim=-1, keepdim=True)
        variance = torch.var(x_float, dim=-1, keepdim=True, unbiased=False)
        x_normed = (x_float - mean) * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        if self.bias.device != x.device:
            self.bias = self.bias.to(x.device)
        return (self.weight * x_normed + self.bias).to(x.dtype)


@profiled("gelu")
def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using Triton."""
    original_shape = x.shape
    total = int(np.prod(x.shape))
    block = GELU_BLOCK_SIZE if GELU_BLOCK_SIZE > 0 else 256

    x_flat = x.reshape(-1).contiguous().to(torch.float32)
    output = torch.empty_like(x_flat)
    grid = (triton.cdiv(total, block),)

    if x.is_cuda:
        gelu_kernel[grid](x_flat, output, total, BLOCK_SIZE=block)
        return output[:total].reshape(original_shape).to(x.dtype)

    return torch.nn.functional.gelu(x)


@profiled("silu")
def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation using Triton."""
    original_shape = x.shape
    total = int(np.prod(x.shape))
    block = SILU_BLOCK_SIZE if SILU_BLOCK_SIZE > 0 else 256

    x_flat = x.reshape(-1).contiguous().to(torch.float32)
    output = torch.empty_like(x_flat)
    grid = (triton.cdiv(total, block),)

    if x.is_cuda:
        silu_kernel[grid](x_flat, output, total, BLOCK_SIZE=block)
        return output[:total].reshape(original_shape).to(x.dtype)

    return torch.nn.functional.silu(x)


def get_activation(name: str):
    """Get activation function by name."""
    activations = {"gelu": gelu, "silu": silu}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class Linear:
    """Linear layer with switchable backend (torch or Triton)."""

    TILE_M = LINEAR_TILE_M
    TILE_N = LINEAR_TILE_N
    TILE_K = LINEAR_TILE_K

    BACKEND = LINEAR_BACKEND

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weight = torch.zeros((out_features, in_features), dtype=torch.float16)
        self.bias_param = torch.zeros(out_features, dtype=torch.float16) if bias else None

        self._weight_t_padded = None
        self._K_padded = None
        self._N_padded = None

    def __setattr__(self, name, value):
        if name == 'weight' and isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            value = value.to(torch.float16)
        elif name == 'bias_param' and isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            value = value.to(torch.float16)
        super().__setattr__(name, value)

    def _ensure_weight_prepared(self):
        """Cache transposed and padded weight for Triton kernel."""
        if self._weight_t_padded is None:
            K = self.in_features
            N = self.out_features
            self._K_padded = pad_to_multiple(K, self.TILE_K)
            self._N_padded = pad_to_multiple(N, self.TILE_N)

            weight_t = self.weight.t().contiguous()
            if self._K_padded > K or self._N_padded > N:
                weight_pad = torch.zeros(
                    (self._K_padded, self._N_padded),
                    dtype=weight_t.dtype,
                    device=weight_t.device,
                )
                weight_pad[:K, :N] = weight_t
                self._weight_t_padded = weight_pad
            else:
                self._weight_t_padded = weight_t

            # Free original weight to avoid holding both copies in memory
            object.__setattr__(self, 'weight', None)

    @profiled("Linear.__call__")
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Always use cuBLAS (torch) — faster than custom Triton matmul for all sizes
        return self._forward_torch(x)

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Torch matmul backend using float16 cuBLAS for speed."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = int(np.prod(batch_dims))
        # Use float16 for cuBLAS matmul — ~2x faster than float32 on Ampere+
        x_2d = x.reshape(M, self.in_features).to(torch.float16)

        # If original weight was freed (Triton path prepared it), use the padded weight
        if self.weight is None and self._weight_t_padded is not None:
            if self._weight_t_padded.device != x.device:
                self._weight_t_padded = self._weight_t_padded.to(x.device)
            weight_t = self._weight_t_padded[:self.in_features, :self.out_features].to(torch.float16)
            output = (x_2d @ weight_t).float()
        else:
            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            output = (x_2d @ self.weight.to(torch.float16).t()).float()

        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            output = output + self.bias_param.to(torch.float32)

        return output.reshape(*batch_dims, self.out_features)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Triton matmul backend."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = int(np.prod(batch_dims))
        K = self.in_features
        N = self.out_features

        x_2d = x.reshape(M, K).to(torch.float32).contiguous()

        if self._weight_t_padded is None:
            if self.weight is not None and self.weight.device != x.device:
                torch.cuda.empty_cache()
                self.weight = self.weight.to(x.device)
            self._ensure_weight_prepared()
        elif self._weight_t_padded.device != x.device:
            torch.cuda.empty_cache()
            self._weight_t_padded = self._weight_t_padded.to(x.device)

        M_padded = pad_to_multiple(M, self.TILE_M)

        if M_padded > M or self._K_padded > K:
            x_padded = torch.zeros(
                (M_padded, self._K_padded),
                dtype=torch.float32,
                device=x.device,
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        output = torch.zeros(
            (M_padded, self._N_padded), dtype=torch.float32, device=x.device
        )

        grid = lambda meta: (
            triton.cdiv(M_padded, meta["BLOCK_M"]),
            triton.cdiv(self._N_padded, meta["BLOCK_N"]),
        )
        linear_kernel_tf32[grid](
            x_padded,
            self._weight_t_padded,
            output,
            M_padded,
            self._N_padded,
            self._K_padded,
            x_padded.stride(0),
            x_padded.stride(1),
            self._weight_t_padded.stride(0),
            self._weight_t_padded.stride(1),
            output.stride(0),
            output.stride(1),
        )

        output = output[:M, :N]

        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            output = output + self.bias_param

        return output.reshape(*batch_dims, self.out_features)


class Embedding:
    """Embedding layer using Triton."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float16)

    def __setattr__(self, name, value):
        if name == 'weight' and isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            value = value.to(torch.float16)
        super().__setattr__(name, value)

    @profiled("Embedding.__call__")
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        original_shape = input_ids.shape
        batch_size = int(np.prod(original_shape))

        if self.weight.device != input_ids.device:
            self.weight = self.weight.to(input_ids.device)

        if not input_ids.is_cuda:
            flat = input_ids.reshape(-1).to(torch.int64)
            output = self.weight.index_select(0, flat)
            return output.reshape(*original_shape, self.embedding_dim)

        indices_flat = input_ids.reshape(-1).to(torch.int32).contiguous()
        output = torch.empty(
            (batch_size, self.embedding_dim), dtype=torch.float32, device=indices_flat.device
        )

        block = 256
        grid = (batch_size, triton.cdiv(self.embedding_dim, block))
        embedding_kernel[grid](
            indices_flat,
            self.weight,
            output,
            self.embedding_dim,
            self.weight.stride(0),
            self.weight.stride(1),
            output.stride(0),
            BLOCK_SIZE=block,
        )

        return output.reshape(*original_shape, self.embedding_dim)


@profiled("softmax")
def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Softmax using Triton kernel."""
    if axis != -1 and axis != len(x.shape) - 1:
        x = torch.movedim(x, axis, -1)

    original_shape = x.shape
    batch_size = int(np.prod(x.shape[:-1]))
    seq_len = x.shape[-1]

    x_flat = x.reshape(batch_size, seq_len).to(torch.float32).contiguous()
    output = torch.empty_like(x_flat)

    if x.is_cuda:
        block_size = SOFTMAX_BLOCK_SIZE if SOFTMAX_BLOCK_SIZE > 0 else next_power_of_two(seq_len)
        softmax_kernel[(batch_size,)](
            x_flat,
            output,
            x_flat.stride(0),
            output.stride(0),
            seq_len,
            BLOCK_SIZE=block_size,
        )
        result = output.reshape(original_shape)
    else:
        result = torch.softmax(x, dim=-1)

    if axis != -1 and axis != len(original_shape) - 1:
        result = torch.movedim(result, -1, axis)

    return result


class MLP:
    """MLP with SwiGLU gating using Triton."""

    FUSED = True
    TILE_M, TILE_N, TILE_K = 64, 64, 32

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            activation: str = "silu",
            bias: bool = False,
            use_gating: bool = True,
    ):
        self.use_gating = use_gating
        self.act_fn = get_activation(activation)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled = bias

        if use_gating:
            self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)
        else:
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)

        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)

        self._gate_weight_t = None
        self._up_weight_t = None

    def _prepare_fused_weights(self):
        """Prepare pre-transposed, pre-padded, float32 weights for fused kernel."""
        if self._gate_weight_t is None and self.use_gating:
            if self.gate_proj.weight.device != self.up_proj.weight.device:
                self.up_proj.weight = self.up_proj.weight.to(self.gate_proj.weight.device)
            gate_wt = self.gate_proj.weight.to(torch.float32).t().contiguous()
            up_wt = self.up_proj.weight.to(torch.float32).t().contiguous()
            K = self.hidden_size
            N = self.intermediate_size
            K_pad = pad_to_multiple(K, self.TILE_K)
            N_pad = pad_to_multiple(N, self.TILE_N)
            self._K_pad = K_pad
            self._N_pad = N_pad
            if K != K_pad or N != N_pad:
                self._gate_weight_t = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=gate_wt.device)
                self._gate_weight_t[:K, :N] = gate_wt
                self._up_weight_t = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=up_wt.device)
                self._up_weight_t[:K, :N] = up_wt
            else:
                self._gate_weight_t = gate_wt
                self._up_weight_t = up_wt

    @profiled("MLP.__call__")
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass — uses fast float16 cuBLAS via Linear.
        Fuses gate+up projections and keeps intermediates in fp16 to reduce HBM traffic."""
        if self.use_gating:
            # Compute gate and up projections
            gate_out = self.gate_proj(x)
            up_out = self.up_proj(x)
            # Apply activation and element-wise multiply in fp16 to avoid fp32 round-trip
            activated = self.act_fn(gate_out) * up_out
            return self.down_proj(activated)
        return self.down_proj(self.act_fn(self.up_proj(x)))


class EncoderMLP:
    """Encoder MLP (no gating) using Triton."""

    FUSED = True
    TILE_M, TILE_N, TILE_K = 64, 64, 32

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            activation: str = "gelu",
            bias: bool = True,
    ):
        self.fc1 = Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled = bias
        self.activation = activation

        self._fc1_weight_t = None

    def _prepare_fused_weights(self):
        """Prepare pre-transposed weights for fused kernel."""
        if self._fc1_weight_t is None:
            self._fc1_weight_t = self.fc1.weight.t().contiguous()

    @profiled("EncoderMLP.__call__")
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if EncoderMLP.FUSED and self.activation == "gelu" and x.is_cuda:
            return self._forward_fused(x)
        return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard (unfused) forward pass."""
        return self.fc2(self.act_fn(self.fc1(x)))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused Linear+GELU forward pass."""
        if self._fc1_weight_t is None:
            if self.fc1.weight is not None and self.fc1.weight.device != x.device:
                self.fc1.weight = self.fc1.weight.to(x.device)
            self._prepare_fused_weights()
        elif self._fc1_weight_t.device != x.device:
            self._fc1_weight_t = self._fc1_weight_t.to(x.device)

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size).to(torch.float32).contiguous()
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        M_pad = pad_to_multiple(M, self.TILE_M)
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)

        if M != M_pad or K != K_pad:
            x_padded = torch.zeros(
                (M_pad, K_pad), dtype=torch.float32, device=x.device
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        if K != K_pad or N != N_pad:
            fc1_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=torch.float32, device=x.device
            )
            fc1_w_padded[:K, :N] = self._fc1_weight_t
        else:
            fc1_w_padded = self._fc1_weight_t

        intermediate = torch.zeros(
            (M_pad, N_pad), dtype=torch.float32, device=x.device
        )

        grid = lambda meta: (
            triton.cdiv(M_pad, meta["BLOCK_M"]),
            triton.cdiv(N_pad, meta["BLOCK_N"]),
        )
        linear_gelu_kernel[grid](
            x_padded,
            fc1_w_padded,
            intermediate,
            M_pad,
            N_pad,
            K_pad,
            x_padded.stride(0),
            x_padded.stride(1),
            fc1_w_padded.stride(0),
            fc1_w_padded.stride(1),
            intermediate.stride(0),
            intermediate.stride(1),
        )

        if M != M_pad or N != N_pad:
            intermediate = intermediate[:M, :N]

        if self.bias_enabled and self.fc1.bias_param is not None:
            if self.fc1.bias_param.device != x.device:
                self.fc1.bias_param = self.fc1.bias_param.to(x.device)
            intermediate = intermediate + self.fc1.bias_param

        intermediate = intermediate.reshape(*orig_shape[:-1], self.intermediate_size)
        return self.fc2(intermediate)


if __name__ == "__main__":
    print("Testing Triton Layers...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== RMSNorm ===")
    norm = RMSNorm(256)
    x = torch.randn(2, 16, 256, device=device, dtype=torch.float32)
    y = norm(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== LayerNorm ===")
    ln = LayerNorm(256)
    y = ln(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== GELU ===")
    y = gelu(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== SiLU ===")
    y = silu(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== Linear ===")
    linear = Linear(256, 512)
    y = linear(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== Embedding ===")
    emb = Embedding(1000, 256)
    ids = torch.randint(0, 1000, (2, 16), device=device, dtype=torch.int32)
    y = emb(ids)
    print(f"Input: {ids.shape} -> Output: {y.shape}")

    print("\n=== Softmax ===")
    x_sm = torch.randn(2, 4, 16, 16, device=device, dtype=torch.float32)
    y = softmax(x_sm, axis=-1)
    print(f"Input: {x_sm.shape} -> Output: {y.shape}")
    print(f"Sum along last axis: {float(y[0, 0, 0].sum()):.6f} (should be 1.0)")

    print("\n=== MLP ===")
    mlp = MLP(256, 512, activation="silu", use_gating=True)
    y = mlp(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\nAll Triton layers working!")
