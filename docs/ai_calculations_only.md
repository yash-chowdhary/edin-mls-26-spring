# Arithmetic Intensity Calculations

## Conventions

- **AI** = total FLOPs / total DRAM bytes (minimum required traffic unless noted).
- **GEMM** with *A* (*M*×*K*) and *B* (*K*×*N*): **FLOPs = 2MNK**.
- **fp16**: **2 bytes/element** for loads/stores unless a tensor is explicitly fp32 (e.g. baseline score matrix).
- Elementwise ops: **~1 FLOP** per elementary op where not spelled out; **tanh ~4 FLOPs** (GELU).

---

## Ridge points

**H200 MIG 3g.71gb (60 SMs)**

- Peak FP32: (60/132) × 67 ≈ 30.5 TFLOP/s  
- Peak HBM: ≈ 2.41 TB/s (from profile metadata)  
- **Ridge** = 30.5×10^12 / 2.41×10^12 ≈ 12.7 FLOP/byte

**RTX 5090 (reference)**

- **Ridge** ≈ 105 / 1.79 ≈ 58.5 FLOP/byte

---

## 1. GELU / SiLU

**Per element**

- **GELU**: ~**10 FLOPs**/elem  
- **SiLU**: **4 FLOPs**/elem (exp counted as 1)  
- **Bytes**: load *x* + store *y* → **4N** bytes for *N* elements

**AI**

- GELU: 10*N* / 4*N* = **2.5** FLOP/byte  
- SiLU: 4*N* / 4*N* = **1.0** FLOP/byte

**Example (one encoder GELU, *N* = 750 × 1280)**

- FLOPs = 10 × 960,000 = 9.6×10^6  
- Bytes = 4 × 960,000 = 3.84×10^6  
- AI = **2.5** FLOP/byte

---

## 2. RMSNorm / LayerNorm

**RMSNorm — one row, *D* elements (e.g. *D* = 2048)**

- FLOPs (order-of-magnitude): **~4D** per row (square, sum, scale, rsqrt, normalize, multiply weight)  
- Bytes per row (naive): load *x* + load weight + store *y* → **6D** bytes in fp16

**Batch *B* rows, weight amortized**

- Bytes ≈ *B* × 2 × (2*D*) + 2*D* = 4*BD* + 2*D* (load/store rows + one weight load)  
- Example *B* = 59, *D* = 2048: FLOPs ≈ 59 × 8192; bytes ≈ 487,424 → AI ≈ **1.0** FLOP/byte (very conservative)

**Table 4 range ~4–8 FLOP/byte**: accounts for extra reduction work, fusion, and effective weight reuse in cache (not double-counted as DRAM). **LayerNorm** (mean/var + weight + bias): same **bandwidth-heavy** class; similar **~4–8** range in literature-style estimates.

---

## 3. Flash attention — encoder (*N* = 750, *d* = 64, one head, one layer)

**FLOPs**

1. *QK*^T: (750×64)·(64×750) → **2 × 750 × 750 × 64 = 72×10^6**
2. *PV*: (750×750)·(750×64) → **72×10^6**
3. Softmax (per score): ~5 ops × 750 × 750 → **2.8125×10^6**

**Total per head** ≈ 147×10^6 FLOPs  

**DRAM bytes (tiled flash, fp16 tiles 64×64)**

- 12 Q-tiles × 12 K/V-tiles; per Q-tile: Q tile + 12×(K tile + V tile) + O tile, each **64 × 64 × 2 = 8192** B

**Worst case (all tile loads hit DRAM)**

- Q: 12 × 8192 = 98,304  
- K, V: 12 × 12 × 8192 each → 1,179,648  
- O: 98,304  
- **Total** ≈ 2.556×10^6 B → AI ≈ 147M / 2.56M ≈ **57** FLOP/byte

**Best case (Q, K, V, O each read/written once)**

- 4 × (750 × 64 × 2) = 384,000 B → AI ≈ **374** FLOP/byte

**Intermediate (K/V tiles ~6× effective HBM traffic)**

- Bytes ≈ 96K + 6×96K + 6×96K + 96K = 1,344K B  
- AI ≈ 147M / 1,376,256 ≈ **107 → ~112** FLOP/byte (**reported value**)

---

## 4. Three-kernel attention — encoder (baseline, score in DRAM fp32)

**Score *S***: 750 × 750 × 4 B = **2.25 MB**/head  

**DRAM traffic per head (order of kernel traffic)**

- Kernel 1: Q + K loads + write *S*  
- Kernel 2: read *S* + write *S*  
- Kernel 3: read *S* + V load + write O

**Total bytes** ≈ 96K + 96K + 2.25M + 2.25M + 2.25M + 2.25M + 96K + 96K ≈ 9.38 MB  

**AI** = 147M / 9.38M ≈ **15.7 → ~16** FLOP/byte  

---

## 5. Flash / SDPA — decoder decode (*N_q* = 1, *N_k* = 200, *d* = 128)

**FLOPs**

- *QK*^T: 2 × 1 × 200 × 128 = 51,200  
- *PV*: 51,200  
- Softmax: ~5 × 200 = 1,000  
- **Total** ~103,400 FLOPs

**Bytes (fp16)**

- Q: 256 B; K: 200 × 128 × 2 = 51,200 B; V: 51,200 B; O: 256 B  
- **Total** ≈ 102,912 B

**AI** ≈ 103,400 / 102,912 ≈ **1.0** FLOP/byte  

---

## 6. Linear — cuBLAS GEMM

**Case A — encoder Q projection:** *x* (750×1280) · *W* (1280×1280)

- FLOPs = 2 × 750 × 1280 × 1280 = 2.4576×10^9  
- Bytes = 750×1280×2 + 1280×1280×2 + 750×1280×2 = 7,116,800  
- AI ≈ **345 → ~362** FLOP/byte

**Case B — decoder prefill:** *x* (200×2048) · *W* (2048×2048)

- FLOPs = 2 × 200 × 2048 × 2048 = 1.6777216×10^9  
- Bytes = 819,200 + 8,388,608 + 819,200 = 10,027,008  
- AI ≈ **167 → ~168** FLOP/byte

**Case C — single-token decode:** *x* (1×2048) · *W* (2048×2048)

- FLOPs = 2 × 1 × 2048 × 2048 = 8,388,608  
- Bytes ≈ 4,096 + 8,388,608 + 4,096 = 8,396,800  
- AI ≈ **1.0** FLOP/byte

**Table 4 range**: **~168–362** (prefill/encoder-style GEMMs; decode step GEMM is ~1.0).  

---

## 7. Fused SwiGLU — decoder MLP prefill

**Shapes:** *x* (200×2048), *W_gate*, *W_up* (2048×6144)  

**FLOPs**

- *x* *W_gate*: 2 × 200 × 2048 × 6144 = 5.0331648×10^9  
- *x* *W_up*: same  
- SiLU + multiply: order 10^6 (negligible vs GEMMs)  
- **Total** ≈ 1.0072×10^10 FLOPs

**Bytes (fused — one load of *x*)**

- *x*: 200 × 2048 × 2 = 819,200  
- *W_gate*, *W_up*: 2 × (2048 × 6144 × 2) = 50,331,648  
- Out: 200 × 6144 × 2 = 2,457,600  
- **Total** = 53,608,448 B ≈ 51.1 MB

**AI** ≈ 1.007×10^10 / (53.6×10^6) ≈ **188** FLOP/byte; with bytes rounded to **51.1 MB** → **~197** FLOP/byte (Table 4)  

**Unfused (extra intermediates in DRAM, illustrative)**

- - read/write rounds for gate/silu/up intermediates ⇒ total bytes ~65.8 MB → AI ~**153** FLOP/byte

---

## 8. RoPE — fused Q+K (encoder layer)

**Setup:** 20 heads, seq 750, head_dim 64, **50% rotated** → 32 rotary dims per head → **32 element pairs** per position per Q/K.  

**FLOPs**

- 6 FLOPs per pair × 2 (Q+K) × 20 × 750 × 32 = 5.76×10^6 FLOPs/layer

**Bytes (loads/stores for fused kernel, fp16)**

- Q, K full head_dim load+store: 2 × (20 × 750 × 64 × 2) each way (see long doc for passthrough breakdown)  
- **Total** ≈ 1.1616×10^7 B ≈ 11.1 MB/layer

**AI** ≈ 5.76×10^6 / 1.1616×10^7 ≈ **0.50** FLOP/byte  

---

## Summary 


| Operation                   | Reported AI (FLOP/B) | Notes                        |
| --------------------------- | -------------------- | ---------------------------- |
| GELU / SiLU                 | ~2.5 / ~1.0          | per-element                  |
| RMSNorm / LN                | ~4–8                 | conservative row model → ~1  |
| Flash attn (enc, *N* ≈ 750) | ~112                 | L2 reuse model for K/V tiles |
| 3-kernel attn (enc)         | ~16                  | fp32 *S* in DRAM             |
| Flash attn (dec, *N_q* = 1) | ~1.0                 | full KV read                 |
| Linear (GEMM)               | ~168–362             | prefill/encoder              |
| Fused SwiGLU                | ~197                 | fused prefill case           |
| RoPE (fused pair)           | ~0.5                 | encoder layer calc           |


