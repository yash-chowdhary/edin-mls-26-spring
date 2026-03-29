# Arithmetic Intensity: Manual Calculation (encoder N = 1500)

This document is the same methodology as `[ai_manual_calculation.md](ai_manual_calculation.md)`, but uses **encoder self-attention sequence length N = 1500**, which matches `**audio_encoder` output** for our benchmark clip (`test_audio.wav`: mel features **(1, 128, 3000)** after conv subsampling → **seq = 1500**). The earlier **~750** value was not tied to that measurement.

**Script:** Run `[compute_ai_encoder_n.py](compute_ai_encoder_n.py)` to recompute for any N:

```bash
python docs/compute_ai_encoder_n.py --n 1500 --three-kernel
python docs/compute_ai_encoder_n.py --n 750   # compare to old illustrative N
```

**Definition:** Arithmetic intensity = total floating-point operations / total bytes transferred to/from DRAM [Williams et al., 2009].

**Conventions:** (unchanged from `ai_manual_calculation.md`)

- GEMM A(M×K) @ B(K×N): **2MNK** FLOPs  
- fp16 = **2 bytes/element** unless noted  
- Flash tiling model: same as the manual doc — full Q and O footprints once; **K and V each streamed once per Q-tile** (`ceil(N / BLOCK_M)` passes, **BLOCK_M = 128**)

---

## Model Dimensions (updated)


| Parameter                       | Encoder             | Decoder                             |
| ------------------------------- | ------------------- | ----------------------------------- |
| hidden_size                     | 1280                | 2048                                |
| num_heads                       | 20                  | 16 (Q), 4 (KV)                      |
| head_dim                        | 64                  | 128                                 |
| intermediate_size               | 5120                | 6144                                |
| num_layers                      | 32                  | 28                                  |
| **seq_len (this doc, encoder)** | **1500** (measured) | ~200 prefill / 1 decode (unchanged) |


---

## 1. GELU / SiLU (element-wise)

Unchanged **AI = 2.5** FLOP/byte (per-element ratio).

**Encoder example with N = 1500, hidden = 1280:**

- Elements: 1500 × 1280 = **1,920,000**  
- FLOPs: ~10 × 1.92M = **19.2M**  
- Bytes: 4 × 1.92M = **7.68M**  
- AI ≈ **2.5**

---

## 2. RMSNorm / LayerNorm

Same **~4–8 FLOP/byte** class as before (normalization is bandwidth-heavy). Row-count **B** in decoder still comes from prefill shapes; encoder-side norms scale with **1500** positions in the encoder stack.

---

## 3. Flash Attention — Encoder (seq_q = seq_k = **1500**, head_dim = 64)

Per head, per layer:

**FLOPs**

- Q @ K^T: 2 × 1500 × 1500 × 64 = **288,000,000**  
- P @ V: 2 × 1500 × 1500 × 64 = **288,000,000**  
- Softmax ~5 ops per score: 5 × 1500 × 1500 = **11,250,000**  
- **Total ≈ 587,250,000 FLOPs per head** (~4× the N = 750 case, since leading term is **N²**)

**DRAM bytes (tiled flash, BLOCK_M = 128)**

- num_Q_tiles = ceil(1500 / 128) = **12**  
- Q + O footprint: 1500 × 64 × 2 × 2 = **384,000** bytes  
- K streamed once per Q-tile: 12 × (1500 × 64 × 2) = **2,304,000** bytes  
- V streamed once per Q-tile: **2,304,000** bytes  
- **Total ≈ 4,992,000 bytes ≈ 4.99 MB per head**

**AI (flash) = 587,250,000 / 4,992,000 ≈ 118 FLOP/byte** (script prints **117.64**)

Compared to **N = 750** with the same model (~~**109** FLOP/byte in the script, ~**112** in the rounded writeup), AI **rises slightly** because **FLOPs grow as N²** while the tiled byte traffic grows roughly **linearly in N** times **number of Q-tiles** (~~**N × ceil(N/B)**), not as **N²**.

**Classification:** Still **compute-bound** on H200 (ridge ~12.7) and on RTX 5090 (ridge ~58.5).

---

## 3a. Optional — 3-kernel pipeline (fp32 scores), N = 1500

Score tensor **1500 × 1500 × 4** bytes = **9 MB** per head.

- **DRAM ≈ 36,768,000 bytes per head** (~35 MB)  
- **AI ≈ 587.25M / 36.77M ≈ 16 FLOP/byte** 

---

## 3b. Flash Attention — Decoder Decode

**Unchanged** from `ai_manual_calculation.md`: dominated by **seq_q = 1** and full KV read; **AI ~1** FLOP/byte. Decoder KV length is **not** 1500.

---

## 4. Linear (cuBLAS) — encoder Q projection with N = 1500

**x (1500×1280) @ W (1280×1280)**

- FLOPs: 2 × 1500 × 1280 × 1280 = **4,915,200,000**  
- Bytes: (1500×1280 + 1280×1280 + 1500×1280) × 2 = **10,956,800**  
- **AI ≈ 449 FLOP/byte** (higher than N = 750 case ~**362**, because **M** is larger → more reuse of **W**)

Decoder examples (200×2048) unchanged.

---

## 5. Fused SwiGLU

Unchanged if still using **x (200×2048)** prefill — not tied to encoder **N = 1500**.

---

## 6. RoPE — encoder layer, seq = **1500**

**FLOPs** (same formula as manual): 20 × 1500 × 32 × 6 = **5,760,000** per layer  
(2× the N = 750 value **2,880,000**.)

**Bytes:** scale the N = 750 estimate **linearly** with sequence length:  
5,856,000 × (1500 / 750) = **11,712,000** bytes  

**AI ≈ 5.76M / 11.71M ≈ 0.49 FLOP/byte** — still **~0.5**, bandwidth-bound.

---

## Summary Table (N = 1500 encoder vs prior N = 750)


| Operation                 | N = 750 (old doc)  | N = 1500 (this doc)               |
| ------------------------- | ------------------ | --------------------------------- |
| GELU (encoder layer act)  | AI ~2.5            | AI ~2.5                           |
| Flash attn encoder        | ~109–112 FLOP/byte | **~118** FLOP/byte                |
| 3-kernel encoder (fp32 S) | ~16 FLOP/byte      | **~16** FLOP/byte                 |
| Linear enc Q proj         | ~362               | **~449**                          |
| RoPE encoder layer        | ~0.5               | **~0.5** (FLOPs & bytes both ~2×) |


Decoder decode / SwiGLU rows: use `ai_manual_calculation.md` unless you remeasure decoder prefill length **L**.

---

## Note on Methodology

Same caveats as `ai_manual_calculation.md`: analytic minima, **ncu** unavailable on the cluster, real AI may be lower. **BLOCK_M** is set to **128** to match the tiling discussion; if your kernel uses **64-row** Q-tiles, run:

```bash
python docs/compute_ai_encoder_n.py --n 1500 --block-m 64 --three-kernel
```

That changes **num_Q_tiles** and the tiled **DRAM** total.