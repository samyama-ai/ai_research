---
id: 32-hardware-and-kernels/cache-oblivious-transformer-inference
title: "Cache-Oblivious Algorithms for Transformer Inference"
topic: 32-hardware-and-kernels
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cache-Oblivious Algorithms for Transformer Inference

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/cache-oblivious-transformer-inference` · **Status:** open

## 1. Problem Statement

Every fast transformer inference kernel in production is **cache-aware**: FlashAttention picks tile sizes $B_r, B_c$ from the SRAM capacity of the target GPU, vLLM picks a KV page size, and each new accelerator generation forces a retune. The question is whether that tuning is *necessary*.

**Input.** A transformer inference workload — prefill of $N$ tokens or decode of one token against a $N$-token KV cache — with head dimension $d$, on a machine whose memory hierarchy parameters (capacity $M$, line size $B$, at each of several levels) are *not given to the algorithm*.

**Output.** A schedule (loop nest / recursion / kernel) that is oblivious to $M$ and $B$.

**Decision predicate.** Does there exist a cache-oblivious schedule whose data movement at *every* level of the hierarchy is within a constant factor of the level-wise I/O lower bound, and whose measured wall-clock is within a small constant (say $1.25\times$) of a per-machine-tuned cache-aware kernel?

Three variants, of very different difficulty:

- **Theory.** Is attention (with online softmax) cache-obliviously optimal, or is there a separation — a problem where every oblivious schedule pays $\omega(1)$ more traffic than the aware optimum? Separations of this kind already exist for sorting (Brodal & Fagerberg, STOC 2003).
- **Method.** Can a recursive, tile-size-free attention kernel be written that a real GPU compiler turns into code competitive with FlashAttention-3?
- **Measurement.** Can per-level traffic on a GPU be attributed to the algorithm at all, given that L2 is shared across SMs, the DRAM controller reorders, and a "cache miss" counter on an SM is not a level in the ideal-cache sense?

## 2. Formal Setting

**Ideal-cache model** (Frigo, Leiserson, Prokop, Ramachandran, FOCS 1999): two levels, cache of $M$ words in lines of $B$ words, fully associative, optimal replacement. Cost is $Q(n; M, B)$, the number of line transfers; work is $W(n)$. An algorithm is *cache-oblivious* if its control flow does not read $M$ or $B$; it is *optimal* if $Q$ matches the Hong–Kung / Aggarwal–Vitter lower bound for all $(M,B)$ simultaneously, which is what makes the two-level result lift to a full hierarchy.

**Attention.** With $Q, K, V \in \mathbb{R}^{N \times d}$,
$$O = \mathrm{softmax}\!\left(\tfrac{1}{\sqrt d} QK^\top\right) V .$$
FlashAttention never materializes the $N \times N$ score matrix; using online softmax rescaling (Milakov & Gimelshein 2018; Rabe & Staats 2021) it achieves
$$Q_{\text{HBM}} = \Theta\!\left(\frac{N^2 d^2}{M}\right)$$
line transfers for $d \le M \le Nd$, which Saha & Ye (ICML 2024) prove is tight in that regime — and show is *not* tight for $M = \Omega(d^2)$, where a better algorithm exists.

**Quantities as measured.**
- $Q_{\text{HBM}}$: NVIDIA Nsight Compute `dram__bytes_read.sum + dram__bytes_write.sum`, divided by 32 B sectors.
- $Q_{\text{L2}}$: `lts__t_sectors_op_read/write` — note this counts *requests to* L2, i.e. traffic across the L1↔L2 boundary, not L2 misses.
- $Q_{\text{SMEM}}$: `l1tex__data_pipe_lsu_wavefronts_mem_shared`, in wavefronts, not bytes — the unit mismatch is real and is part of the measurement problem.
- Arithmetic intensity $I = W/Q_{\text{HBM}}$ in FLOP/byte; ridge point $I^\star = \text{peak FLOP/s} / \text{peak B/s}$ (Williams, Waterman, Patterson, CACM 2009).

**Assumptions, and which are violated.**

| Assumption | Status on a GPU |
|---|---|
| Full associativity | Violated. L1/L2 are set-associative; shared memory is 32-way banked with conflict penalties. |
| Optimal (LRU-competitive) replacement | Violated. L2 uses a hardware policy; shared memory is *software*-managed — there is no replacement policy at all, the programmer is the policy. |
| Tall cache $M = \Omega(B^2)$ | Holds numerically (H100: $M = 228$ KB SMEM, $B = 32$ B) but is meaningless for a scratchpad. Brodal & Fagerberg (STOC 2003) show tall-cache is *necessary* for oblivious sorting optimality. |
| Two levels compose | Weakly violated. L2 (50 MB on H100) is shared by 132 SMs; one SM's working set is not the cache's working set. |
| Cost $\propto$ transfers | Violated for latency-bound decode, where occupancy and memory-level parallelism dominate, not byte count. |

The scratchpad row is the crux: cache-obliviousness is a theory *about caches*. GPU shared memory is not a cache, so the model's central mechanism — automatic replacement doing the right thing when recursion narrows the working set below $M$ — has no hardware counterpart.

## 3. State of the Art

**Theory SOTA.** Frigo et al. (FOCS 1999) give cache-oblivious optimal matrix multiply ($Q = \Theta(n^3/(B\sqrt M))$), matrix transpose, FFT and sorting. Saha & Ye (ICML 2024) give the first tight I/O bounds for attention itself; their algorithms are cache-*aware*. Brodal & Fagerberg (STOC 2003) give the known limits of obliviousness. **Established.**

**Systems SOTA.** FlashAttention (Dao et al., NeurIPS 2022), FlashAttention-2 (Dao, ICLR 2024), FlashAttention-3 (Shah et al., NeurIPS 2024); Flash-Decoding and FlashDecoding++ (Hong et al., 2023) for the decode split-$K$ regime; PagedAttention/vLLM (Kwon et al., SOSP 2023) for KV memory. All are explicitly cache-aware and hand-tuned per architecture; FA-3 rewrote the schedule for Hopper's TMA and warp-specialized async pipeline, which is exactly the retuning cost obliviousness would remove. **Established.**

**Claimed but unablated.** That the recursive/blocked structure of these kernels is "essentially cache-oblivious with tiles pinned for hardware reasons." No paper reports the ablation: same kernel, tile sizes varied over a $4\times$ range, traffic and time measured at each level. Autotuners (TVM/Ansor, Triton `autotune`) are sometimes described as making obliviousness unnecessary; that is a claim about engineering cost, not about the existence of a portable optimal schedule, and it is not benchmarked as such.

**Benchmark-number-only.** FA-3's headline throughput figures (below) are single-shape, single-machine measurements, not ablations of the tiling strategy.

## 4. What Is Known

- **Cache-oblivious MMM is asymptotically optimal but empirically slower.** Yotov, Roeder, Pingali, Gunnels, Gustavson (SPAA 2007) compared cache-oblivious matrix multiply against ATLAS-generated cache-conscious code and found the oblivious version substantially slower — a multiplicative gap, not a constant close to 1 — attributed to recursion overhead and lost register-level scheduling. This is the single most relevant negative datum for the present problem.
- **Data movement dominates transformer time.** Ivanov et al. (MLSys 2021) measured that on BERT, ~37% of runtime is in memory-bound normalization/elementwise ops that hold <1% of the FLOPs; fusing them gave a ~1.3× end-to-end training speedup.
- **FlashAttention numbers.** 3× speedup on GPT-2 training and 15% end-to-end on BERT-large (A100, 2022). FA-2 reaches ~230 TFLOP/s on A100 FP16, ~72% of peak. FA-3 reaches up to ~740 TFLOP/s FP16 on H100 (~75% util) and ~1.2 PFLOP/s in FP8, 1.5–2.0× over FA-2 — measured at $d = 64/128$, $N$ up to 16K, single H100 SXM.
- **Tightness.** Attention's I/O complexity is $\Theta(N^2d^2/M)$ for $d \le M \le Nd$ (Saha & Ye, ICML 2024). FlashAttention is optimal there; it is *not* known to be optimal for large caches.
- **Decode is not compute-bound.** Arithmetic intensity of KV-cache attention during decode is $\approx 1$ FLOP/byte independent of batch, against an H100 ridge point of $\approx 295$ FLOP/byte (see §10).

## 5. What Is Not Known

- **Theoretically open.** Whether a cache-oblivious algorithm attains $\Theta(N^2d^2/M)$ for attention *simultaneously at all levels*, or whether attention joins sorting on the list of problems with a proven oblivious/aware separation. Nobody has proved either direction.
- **Theoretically open.** The I/O lower bound for *decode* attention with a growing KV cache across $T$ steps, as an online problem. Saha & Ye treat the batch (prefill) problem.
- **Empirically open.** Whether a recursive tile-free attention kernel, compiled by Triton or CUTLASS, lands within 25% of FA-3 on H100. The kernel is a few hundred lines; nobody has published the comparison.
- **Methodologically blocked.** Per-level traffic attribution on a GPU. Shared memory has no misses to count; L2 is shared and its counters measure requests, not misses; a "cache complexity" number for a GPU kernel is therefore not currently a well-defined measurement, which is why §8 falls back on a wall-clock predicate plus DRAM bytes.

## 6. Why It Is Hard

The specific obstruction is **the model's mechanism is absent from the hardware**. Cache-obliviousness works because recursion eventually produces subproblems that fit in $M$, and *automatic replacement* then keeps them resident for free. On a GPU, the level that matters (shared memory, 228 KB/SM on H100) is a scratchpad: residency is achieved only by an explicit `cp.async`/TMA copy whose size is a compile-time constant. A tile-size-free kernel is not merely unoptimized — it has no way to express the thing the model assumes is free.

Second obstruction: **the evaluation does not measure what it names.** "Cache complexity" measured by `dram__bytes` conflates the algorithm's traffic with hardware prefetch, L2 sharing across 132 SMs, and eviction from other concurrently-resident kernels. Two schedules with identical ideal-cache $Q$ can differ 2× in measured DRAM bytes.

Third: **the payoff may be in the regime where traffic is irrelevant.** Decode is latency- and bandwidth-bound at $I \approx 1$; there is essentially no reuse to schedule, so an optimal cache-oblivious schedule and a naive streaming one coincide. Obliviousness can only pay in prefill and in large-batch decode — a narrower target than the framing suggests.

## 7. Current Research (as of 2026)

- **Tight I/O bounds for attention variants** — sparse, sliding-window, and linear attention; follow-ons to Saha & Ye at UC San Diego and adjacent theory groups. *(frontier — verify)*
- **Compiler-side automation** — Triton, Mosaic/Pallas (Google), CUTLASS 3.x/CuTe (NVIDIA), and ThunderKittens (Stanford Hazy Research) all pursue *portability by re-specialization*: keep the schedule aware, make retargeting cheap. This is the pragmatic competitor to obliviousness and is currently winning.
- **Hierarchy-aware serving** — KV offload to host DRAM and NVMe pushes the problem to a genuinely multi-level hierarchy (SMEM/L2/HBM/DRAM/SSD) where per-level hand-tuning scales badly and obliviousness would pay most. *(frontier — verify)*
- **Parallel cache-oblivious theory** — work-stealing and multicore cache bounds (Blelloch and collaborators) remain the closest analogue to the many-SM setting.

## 8. Concrete Next Experiment

**Question.** Does a tile-size-free recursive attention kernel come within 25% of a tuned kernel?

**Scale.** Single H100 SXM (80 GB, 3.35 TB/s HBM3). Prefill attention only. $d = 128$, 32 heads, batch 8, $N \in \{2\text{K}, 8\text{K}, 32\text{K}, 128\text{K}\}$, FP16.

**Arms.**
1. *Oblivious arm.* Recursive attention: split the longer of the $Q$-rows / $KV$-rows dimension in half, recurse, combine with online-softmax rescaling; base case = one warp-tile, chosen by the compiler's register allocator, never by $M$. Implement in Triton with no `autotune` and no capacity constant anywhere in the source.
2. *Control arm.* FlashAttention-3, default per-shape tuned configuration, same shapes, same GPU, same clocks (lock SM clocks with `nvidia-smi -lgc`).
3. *Portability control.* Both arms rerun on A100 (192 KB SMEM, 40 MB L2, 2.0 TB/s) **with the oblivious source unchanged and FA-3 retuned**.

**Deciding number.** The ratio $R = t_{\text{oblivious}} / t_{\text{control}}$ at $N = 32\text{K}$, plus its change across A100→H100, $\Delta R$. Report `dram__bytes` for both arms alongside.

- $R \le 1.25$ and $|\Delta R| \le 0.05$: obliviousness is practically viable; the aware/oblivious gap on attention is a constant and portable. Strong result.
- $R \in (1.25, 2]$: viable only where retuning cost dominates (multi-backend serving).
- $R > 2$, matching the Yotov et al. pattern for MMM: attention behaves like matrix multiply, and the tiling constant is doing real work that recursion cannot recover. This is the expected outcome and would justify reclassifying the *method* variant as negative while leaving the theory variant open.

Cost: roughly two GPU-weeks of engineering, under 100 GPU-hours of measurement. This is a small experiment that nobody has published.

## 9. Key References

- **[Foundational]** M. Frigo, C. E. Leiserson, H. Prokop, S. Ramachandran. *Cache-Oblivious Algorithms.* FOCS, 1999.
- **[Foundational]** J.-W. Hong, H. T. Kung. *I/O Complexity: The Red-Blue Pebble Game.* STOC, 1981.
- **[Foundational]** A. Aggarwal, J. S. Vitter. *The Input/Output Complexity of Sorting and Related Problems.* CACM 31(9), 1988.
- **[Theory SOTA]** B. Saha, C. Ye. *The I/O Complexity of Attention, or How Optimal is FlashAttention?* ICML, 2024. — arXiv:2402.07443
- **[Theory]** G. S. Brodal, R. Fagerberg. *On the Limits of Cache-Obliviousness.* STOC, 2003.
- **[SOTA]** T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135
- **[SOTA]** T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR, 2024. — arXiv:2307.08691
- **[SOTA]** J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.* NeurIPS, 2024. — arXiv:2407.08608
- **[SOTA]** W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Empirical]** K. Yotov, T. Roeder, K. Pingali, J. Gunnels, F. Gustavson. *An Experimental Comparison of Cache-Oblivious and Cache-Conscious Programs.* SPAA, 2007.
- **[Empirical]** A. Ivanov, N. Dryden, T. Ben-Nun, S. Li, T. Hoefler. *Data Movement Is All You Need: A Case Study on Optimizing Transformers.* MLSys, 2021. — arXiv:2007.00072
- **[Method]** M. Milakov, N. Gimelshein. *Online Normalizer Calculation for Softmax.* 2018. — arXiv:1805.02867
- **[Method]** M. N. Rabe, C. Staats. *Self-attention Does Not Need $O(n^2)$ Memory.* 2021. — arXiv:2112.05682
- **[Method]** P. Tillet, H. T. Kung, D. Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL, 2019.
- **[Survey]** S. Williams, A. Waterman, D. Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4), 2009.
- **[Survey]** R. Pope et al. *Efficiently Scaling Transformer Inference.* MLSys, 2023. — arXiv:2211.05102

## 10. Worked Example

**Setting.** One decode step, one attention layer. Batch $b = 32$, heads $H = 32$, head dim $d = 128$, KV length $N = 8192$, FP16 (2 B), H100 SXM.

KV bytes touched:
$$2 \cdot b \cdot H \cdot N \cdot d \cdot 2\,\text{B} = 2 \cdot 32 \cdot 32 \cdot 8192 \cdot 128 \cdot 2 = 4.29\ \text{GB}.$$

FLOPs (scores $+$ weighted sum, $2$ FLOP per MAC):
$$2 \cdot 2 \cdot b \cdot H \cdot N \cdot d = 4.29\ \text{GFLOP}.$$

Arithmetic intensity:
$$I = \frac{4.29 \times 10^9}{4.29 \times 10^9} = 1.0\ \text{FLOP/byte}.$$

H100 ridge point: $989\ \text{TFLOP/s} / 3.35\ \text{TB/s} \approx 295$ FLOP/byte. The workload sits $295\times$ to the left of the ridge. Lower bound on time: $4.29\ \text{GB} / 3.35\ \text{TB/s} = 1.28$ ms per layer per token — and every byte of KV is read exactly once, so $Q$ is $\Theta(bHNd/B)$ for *any* schedule, oblivious or not.

**Where the obstruction becomes visible.** In this regime the cache-oblivious question is vacuous: there is no reuse, so recursion buys nothing and the aware/oblivious gap is zero by construction. Now change one number — batch 32 → 512, sharing nothing (each sequence has its own KV) — and $I$ stays at 1.0. Change instead to *prefill* of $N = 8192$: FLOPs scale as $N^2$, bytes as $N$, and $I$ rises to $\approx 8192/2 \cdot$ (reuse factor), landing right of the ridge, where tiling is everything and FA-3 gets ~75% of peak.

So the problem has a narrow live target: **prefill and large-batch shared-prefix decode**, not the token-by-token regime that dominates serving cost. And on exactly that target — dense, compute-bound, reuse-rich matrix work — the one published head-to-head (Yotov et al., SPAA 2007, on MMM) says cache-oblivious loses by a multiplicative factor. The catalog entry stays open because that experiment has never been run on attention, on a GPU, with a scratchpad instead of a cache. The prior is negative; the measurement is missing.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*