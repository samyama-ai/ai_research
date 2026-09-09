---
id: 02-attention/io-optimal-attention
title: "Hardware-Aware Attention Optimality Under IO Bounds"
topic: 02-attention
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hardware-Aware Attention Optimality Under IO Bounds

> **Topic:** Attention Mechanisms · **ID:** `02-attention/io-optimal-attention` · **Status:** solved-but-impractical

## 1. Problem Statement

Exact softmax attention is compute-cheap relative to its data movement. The question is what the *minimum* movement is, and whether an algorithm hitting that minimum is the fastest algorithm on a real accelerator.

- **Input:** $Q, K, V \in \mathbb{R}^{N \times d}$ resident in slow memory; a fast memory of $M$ words; a machine whose slow-memory transfers cost far more than arithmetic.
- **Output:** $O = \mathrm{softmax}(QK^\top/\sqrt{d})V$, exactly, to the working numeric format.
- **Objective:** minimize $Q_{\mathrm{IO}}$, the number of words moved between fast and slow memory.
- **Decision predicate:** does there exist an algorithm with $Q_{\mathrm{IO}} = o(f(N,d,M))$ for the best known upper bound $f$, and if not, is the matching lower bound the quantity that governs wall-clock?

Three variants that are routinely conflated:

- **Theory variant.** Prove tight upper/lower bounds on $Q_{\mathrm{IO}}$ in the red-blue pebble model. *Largely closed* in the large-cache regime.
- **Method variant.** Build a kernel achieving the bound. *Closed* — FlashAttention and successors.
- **Measurement variant.** Show that an I/O-optimal kernel is runtime-optimal on an actual GPU. *Open, and the interesting part.* On H100, FlashAttention-3 is 1.5–2× faster than FlashAttention-2 while moving the same number of HBM words. Optimality in the model does not pin performance on the machine.

This entry is `solved-but-impractical` for exactly that reason: the theorem is proved, and the theorem's objective is no longer the binding constraint.

## 2. Formal Setting

**Machine model.** Two-level memory (Hong & Kung's red-blue pebble game, STOC 1981): unbounded slow memory, fast memory of $M$ words, arithmetic only on words in fast memory. $Q_{\mathrm{IO}}$ counts red/blue transitions. *Measured as:* HBM read + write bytes for the kernel, divided by word size — obtainable from `ncu` counters `dram__bytes_read.sum + dram__bytes_write.sum`, not from a static formula.

**Attention.** With $S = QK^\top/\sqrt d$, $P = \mathrm{softmax}_{\text{row}}(S)$, $O = PV$. Arithmetic cost $\approx 4N^2d$ FLOPs for the two matmuls (causal masking halves it).

**Standard-algorithm restriction.** Lower bounds are proved for algorithms that compute each $S_{ij}$ as an atomic product in fast memory — no Strassen-style recombination, no low-rank approximation. *Known to be violated in principle:* Alman & Song (NeurIPS 2023) give $n^{1+o(1)}$-time approximate attention when entries are bounded by $o(\sqrt{\log n})$, outside this class.

**Upper bound (FlashAttention).** Tiling with $B_c=\lceil M/4d\rceil$, $B_r=\min(\lceil M/4d\rceil, d)$ and online softmax rescaling gives

$$Q_{\mathrm{IO}}^{\mathrm{FA}} = \Theta\!\left(\frac{N^2 d^2}{M}\right), \qquad \text{versus } \Theta(N^2 + Nd) \text{ for materialized attention.}$$

**Lower bound (Saha & Ye, ICML 2024).** For $M \ge d^2$ (large cache), $Q_{\mathrm{IO}} = \Omega(N^2d^2/M)$ for standard algorithms — FlashAttention is asymptotically optimal. For $M < d^2$ they give an algorithm at $O(N^2 d/\sqrt{M})$, strictly better than FlashAttention there; the matching small-cache lower bound holds under a narrower algorithm class *(verify against the published statement before citing as unconditional)*.

**Roofline.** Arithmetic intensity $I = \text{FLOPs}/\text{bytes moved}$; runtime $\ge \max(\text{FLOPs}/\pi,\ \text{bytes}/\beta)$ for peak $\pi$ and bandwidth $\beta$. Ridge point $\pi/\beta$: **295 FLOP/byte** on H100 SXM (989 TFLOP/s BF16 dense, 3.35 TB/s HBM3); **153 FLOP/byte** on A100 80GB (312 TFLOP/s, 2.04 TB/s).

**Assumptions known to be violated in practice:**

1. **$M$ is a scalar.** It is not: H100 has 256 KB register file + 228 KB shared memory per SM ×132 SMs, a 50 MB L2, then HBM, then NVLink, then network. Transfer costs differ by two orders of magnitude across tiers.
2. **Transfers are synchronous and uniform-cost.** TMA (Tensor Memory Accelerator) copies overlap with tensor-core math; a moved word can be free if it hides under compute.
3. **Arithmetic is free.** At $I \approx 100$–$300$ FLOP/byte for prefill attention, it is not — the kernel is compute-bound.
4. **Exactness.** FP8 attention (FlashAttention-3) changes both the FLOP rate and the byte count, and is not the same function.

## 3. State of the Art

**Theory SOTA (established).**
- Hong & Kung (1981): red-blue pebble framework; $\Omega(n^3/\sqrt M)$ for matrix multiply.
- Saha & Ye (ICML 2024, arXiv:2402.07443): first attention-specific I/O lower bound. Large-cache optimality of FlashAttention is a theorem, not a claim.

**Systems SOTA (established, with public kernels and reproduced benchmarks).**
- FlashAttention (NeurIPS 2022): 2–4× kernel speedup over PyTorch attention; 15% end-to-end GPT-2 (seq 1K) training speedup; 3× BERT-large seq-512 over the MLPerf 1.1 record.
- FlashAttention-2 (ICLR 2024): ~2× over FA1, 50–73% of A100 BF16 peak (~230 TFLOP/s at 200–230 range).
- FlashAttention-3 (NeurIPS 2024): ~740 TFLOP/s FP16 on H100 (≈75% util), ~1.2 PFLOP/s FP8. Same asymptotic $Q_{\mathrm{IO}}$ as FA2. The gain is warp specialization, TMA/WGMMA asynchrony, and softmax–matmul interleaving.

**Claimed but unablated.**
- That FA3's FP8 path preserves training quality at scale — reported RMSE improvements over baseline FP8 attention on synthetic outlier distributions, not an end-to-end pretraining ablation.
- Vendor attention TFLOP/s numbers (cuDNN, TE) are benchmark points at selected $(N,d,\text{batch})$; they do not come with I/O counters, so whether they beat FA3 by moving fewer bytes or by scheduling better is unreported.
- Ring Attention's "near-infinite context" (ICLR 2024) is a device-count scaling claim; the network-tier I/O optimality of ring vs. tree KV exchange is not proved.

## 4. What Is Known

- **Memory:** exact attention needs $O(N)$, not $O(N^2)$, activation memory (Rabe & Staats 2021; online softmax, Milakov & Gimelshein 2018). FA1 measured 10–20× activation-memory reduction at seq 1K–4K.
- **The tiling constant:** with $d=128$, bf16, $M=228$ KB shared memory, $B_c=B_r$ blocks are 64–256 rows in practice; block size is chosen by occupancy, not by the $M/4d$ formula.
- **Prefill is compute-bound on modern parts.** FA2/FA3 sit at 50–75% of tensor-core peak — a memory-bound kernel cannot.
- **Decode is bandwidth-bound.** Batch-1 attention decode has $I \approx 1$–$2$ FLOP/byte against a 295 ridge — roughly $150\times$ inside the memory roof. GQA (Ainslie et al., EMNLP 2023) cuts KV bytes by the query:KV head ratio (typically 4–8×) with reported quality close to MHA on T5-XXL.
- **Fragmentation, not asymptotics, dominated serving.** PagedAttention (SOSP 2023) reported 2–4× throughput over prior systems by eliminating KV-cache waste — no change to $Q_{\mathrm{IO}}$ per token.

## 5. What Is Not Known

- **Theoretically open.** Tight I/O bounds in a *multi-level* hierarchy (registers/SMEM/L2/HBM/NVLink) for attention. Tight bounds for causal, sliding-window, and block-sparse masks — the $N^2d^2/M$ bound assumes dense. Whether the small-cache $\Theta(N^2d/\sqrt M)$ regime is optimal against *all* standard algorithms.
- **Theoretically open.** Lower bounds when approximation is allowed: the Alman–Song regime sits outside the pebble model entirely, and no one has combined an $\epsilon$-approximation budget with an I/O budget in one statement.
- **Empirically open.** Does closing the remaining $O(1)$ gap in $Q_{\mathrm{IO}}$ buy any wall-clock at $N \in [8\mathrm{K}, 1\mathrm{M}]$ on H100/B200? Runnable today; nobody has published the counter-instrumented sweep.
- **Methodologically blocked.** There is no accepted cost model that scores a kernel on asynchrony. $Q_{\mathrm{IO}}$ counts bytes; the machine charges for *un-hidden* bytes. Until "effective I/O" is defined — bytes moved minus bytes overlapped with math — "I/O-optimal" cannot be tested against runtime.

## 6. Why It Is Hard

**The specific obstruction: the evaluation does not measure the thing it names.** "I/O-optimal" is optimality in a model whose objective stopped being the bottleneck for the dominant workload. In the prefill regime a kernel at the proven $Q_{\mathrm{IO}}$ minimum can still be 2× slower than one at the same $Q_{\mathrm{IO}}$, because runtime is set by tensor-core occupancy and pipeline overlap, which the pebble model does not represent. Two secondary obstructions compound it: (i) **confounded measurement** — every real comparison changes tiling, numeric format, and scheduling together, so no published ablation isolates bytes-moved from schedule quality; (ii) **non-identifiability of $M$** — a single cache parameter cannot represent a five-tier hierarchy, so the bound's constant is not tied to any measurable hardware quantity.

## 7. Current Research (as of 2026)

- **Asynchrony-aware kernels.** Dao's group (Princeton/Together), NVIDIA CUTLASS/cuDNN, and the Triton team target Hopper/Blackwell warp specialization and FP8/FP4 attention. The optimization target is pipeline occupancy, not byte count.
- **Compiler-level tiling search.** PyTorch FlexAttention, Mosaic/Pallas on TPU, and TVM-descendants search block shapes empirically rather than from $M/4d$ — implicit admission that the analytic optimum is not the practical one.
- **Multi-tier and distributed I/O.** Ring/Ulysses/context-parallel attention push the bound to the network tier; formal lower bounds across device counts remain unwritten *(frontier — verify)*.
- **Decode-side byte reduction.** MLA (DeepSeek), GQA variants, and KV quantization attack the regime where $Q_{\mathrm{IO}}$ genuinely is the bottleneck — this is where I/O theory still bites.
- **I/O bounds for structured masks and linear/state-space attention** — Saha, Ye, and adjacent theory groups *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** at fixed schedule quality, does reducing HBM bytes reduce attention runtime?

- **Scale:** one H100 SXM, single kernel, head dim $d=128$, batch×heads = 32, $N \in \{2\mathrm{K}, 8\mathrm{K}, 32\mathrm{K}, 128\mathrm{K}\}$, BF16, causal and non-causal.
- **Arms:** (a) FlashAttention-3 stock; (b) FA3 with block sizes forced *away* from the runtime-optimal choice to a tiling that reduces measured HBM bytes by ≥30% (larger $B_c$, lower occupancy); (c) FA3 with tiling that *increases* HBM bytes by ≥30% at unchanged occupancy; (d) **control arm** — FA2 at the same block sizes as (a), which holds $Q_{\mathrm{IO}}$ fixed and varies only asynchrony.
- **Instrumentation:** `ncu` DRAM read/write bytes per arm; wall-clock per arm; achieved TFLOP/s.
- **Deciding number:** the regression slope $\partial(\text{runtime})/\partial(\text{HBM bytes})$ within arms (a)–(c), expressed as a fraction of $1/\beta = 0.30$ ns/KB. **If the slope is below $0.2/\beta$ at $N \ge 8\mathrm{K}$ while arm (d) shows a $\ge 1.4\times$ runtime gap at identical bytes, the I/O objective is empirically decoupled from runtime in prefill, and the open problem should be restated over an asynchrony-aware cost model.** If the slope approaches $1/\beta$, the classical bound remains the right target and the gap is engineering.

## 9. Key References

- **[Foundational]** Jia-Wei Hong, H. T. Kung. *I/O Complexity: The Red-Blue Pebble Game.* STOC, 1981.
- **[Foundational]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135
- **[Foundational]** Maxim Milakov, Natalia Gimelshein. *Online Normalizer Calculation for Softmax.* Technical report, 2018. — arXiv:1805.02867
- **[Foundational]** Markus N. Rabe, Charles Staats. *Self-attention Does Not Need $O(n^2)$ Memory.* 2021. — arXiv:2112.05682
- **[SOTA — theory]** Barna Saha, Christopher Ye. *The I/O Complexity of Attention, or How Optimal is FlashAttention?* ICML, 2024. — arXiv:2402.07443
- **[SOTA — systems]** Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR, 2024. — arXiv:2307.08691
- **[SOTA — systems]** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.* NeurIPS, 2024. — arXiv:2407.08608
- **[Related — hardness]** Josh Alman, Zhao Song. *Fast Attention Requires Bounded Entries.* NeurIPS, 2023. — arXiv:2302.13214
- **[Related — systems]** Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, Shigang Li, Torsten Hoefler. *Data Movement Is All You Need: A Case Study on Optimizing Transformers.* MLSys, 2021. — arXiv:2007.00072
- **[Related — serving]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Related — distributed]** Hao Liu, Matei Zaharia, Pieter Abbeel. *Ring Attention with Blockwise Transformers for Near-Infinite Context.* ICLR, 2024. — arXiv:2310.01889
- **[Related — decode]** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP, 2023. — arXiv:2305.13245
- **[Survey]** Samuel Williams, Andrew Waterman, David Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4), 2009.
- **[Survey]** Sehoon Kim, Coleman Hooper, Thanakul Wattanawong, et al. *Full Stack Optimization of Transformer Inference: A Survey.* 2023. — arXiv:2302.14017

## 10. Worked Example

One attention head, $N=8192$, $d=128$, BF16, non-causal, on H100 SXM ($\beta=3.35$ TB/s, $\pi=989$ TFLOP/s).

**Bytes.** $Q,K,V,O$ are each $8192\times128\times2\,\mathrm{B} = 2.10$ MB. Mandatory traffic: 8.39 MB. Materialized attention additionally writes and re-reads $S$ and $P$: $2\times 8192^2 \times 2\,\mathrm{B} \times 2 = 537$ MB. FlashAttention with $M = 228$ KB $= 116{,}736$ words gives

$$Q_{\mathrm{IO}} \approx \frac{N^2d^2}{M} = \frac{8192^2 \cdot 128^2}{116{,}736} = 9.4\times10^6 \text{ words} = 18.8\ \text{MB}.$$

A **29× reduction**, and by the Saha–Ye theorem ($M = 116{,}736 \ge d^2 = 16{,}384$, so large-cache) no standard algorithm does asymptotically better.

**Time.** FLOPs $= 4N^2d = 3.44\times10^{10}$. Compute floor: $3.44{\times}10^{10}/9.89{\times}10^{14} = 34.8$ µs. I/O floor: $18.8\ \mathrm{MB}/3.35\ \mathrm{TB/s} = 5.6$ µs. Arithmetic intensity $I = 3.44{\times}10^{10}/1.88{\times}10^{7} = 1830$ FLOP/byte against a ridge of 295 — **6.2× inside the compute roof.**

**The obstruction, made visible.** Suppose a hypothetical algorithm halved $Q_{\mathrm{IO}}$ to 9.4 MB, beating the proved-optimal constant. The I/O floor drops from 5.6 µs to 2.8 µs; the runtime floor stays 34.8 µs. Gain: zero. Now hold bytes fixed and change only the schedule: FA2→FA3 at identical $Q_{\mathrm{IO}}$ moves measured throughput from roughly 350–500 to ~740 TFLOP/s on this part — a 1.5–2× wall-clock win the I/O model cannot express, because it prices bytes and the machine prices un-overlapped bytes.

**Where the bound still binds.** Same head at decode, batch 1, one new token, 8192-token KV cache: bytes $= 2\cdot 8192\cdot128\cdot2\,\mathrm{B} = 4.19$ MB; FLOPs $= 4\cdot8192\cdot128 = 4.2$ MFLOP; $I = 1.0$ FLOP/byte — 295× *inside the memory roof*. Here every byte removed is a byte of latency removed, which is why GQA and KV quantization, not better tiling, are the live levers. The theorem is tight; it is tight about the wrong phase of inference.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*