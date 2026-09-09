---
id: 32-hardware-and-kernels/io-lower-bounds-exact-attention
title: "I/O Lower Bounds for Exact Attention"
topic: 32-hardware-and-kernels
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# I/O Lower Bounds for Exact Attention

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/io-lower-bounds-exact-attention` · **Status:** partially-solved

## 1. Problem Statement

Given $Q, K, V \in \mathbb{R}^{N \times d}$ and a two-level memory (fast cache of $M$ words, unbounded slow memory), how many words must move between the two levels to compute exact attention $O = \mathrm{softmax}(QK^\top)V$?

Three variants, with different difficulty:

- **Theory variant.** Prove a lower bound $\mathrm{IO}(N,d,M) = \Omega(f)$ valid over a stated class of algorithms, and an algorithm matching it. Open across the full $(N,d,M)$ parameter space; settled up to constants in the small-cache regime.
- **Measurement variant.** Given a real kernel on real hardware, measure the I/O it actually performs and compare to $f$. Blocked by the fact that GPUs have three or four memory levels, not two, and the counter you can read (DRAM sectors) is not the quantity the theorem bounds.
- **Method variant.** Build a kernel whose measured traffic reaches the constant in the lower bound, not just its asymptotic order.

Solving it means: a bound tight in constant factor, over a class of algorithms broad enough to include everything a compiler would emit, plus a kernel that attains it on hardware.

## 2. Formal Setting

**Machine model.** Red-blue pebble game (Jia-Wei & Kung, 1981). Slow memory is unbounded; fast memory holds $M$ words. Cost is the number of load/store operations. An *I/O* is one word moved.

**Measured as.** On an NVIDIA GPU, $M$ = usable shared memory per thread block, in elements: A100 gives 164 KB/block $\Rightarrow M = 82{,}000$ fp16 words; H100 gives 227 KB $\Rightarrow M \approx 113{,}500$. Traffic is measured as `dram__bytes_read.sum + dram__bytes_write.sum` from Nsight Compute, divided by element width.

**Problem.** With $S = QK^\top$, $P_{ij} = \exp(S_{ij} - \max_k S_{ik})$, $O = \mathrm{diag}(P\mathbf{1})^{-1} P V$.

**Baselines.** Materializing $S$ costs $\Theta(N^2 + Nd)$ I/Os. Tiled streaming (FlashAttention) costs

$$\mathrm{IO}_{\text{FA}} = \Theta\!\left(\frac{N^2 d^2}{M}\right)$$

which beats materialization whenever $M \gg d^2$.

**Assumptions, and which break.**

1. *Two memory levels.* Violated: A100 has 40 MB L2, H100 50 MB, plus a register file (256 KB/SM) larger than a tile. For $N \le 4096$, $d=64$, all of $K,V$ (2 MB fp16) fits in L2, so measured DRAM traffic is far below the model's prediction.
2. *No algebraic cancellation.* All known bounds assume every product $Q_i \cdot K_j$ is formed explicitly — the same restriction that makes matmul bounds inapplicable to Strassen. Softmax's nonlinearity makes cancellation implausible but nobody has proved it impossible.
3. *Exactness.* Approximate attention escapes the bound entirely, and is subquadratic when entries are bounded by $o(\sqrt{\log N})$ (Alman & Song, 2023).
4. *Unit-cost I/O.* Violated: coalesced 128-byte transactions cost the same as one 32-byte sector, so word-counting misprices strided access by up to $4\times$.

## 3. State of the Art

**Theory SOTA.** Saha & Ye, *The I/O Complexity of Attention, or How Optimal Is FlashAttention?* (ICML 2024, arXiv:2402.07443). Established: in the **small-cache regime** ($M = \Theta(d)$, i.e. the cache holds a constant number of rows), any algorithm in the standard class needs $\Omega(N^2 d^2 / M)$ I/Os, matching FlashAttention up to constants. Also established in that work: the picture changes for **large caches**, where FlashAttention is *not* optimal and their upper and lower bounds do not meet. The exact large-cache exponent is the open case *(frontier — verify the precise statement against the published version before citing a formula)*.

**Systems SOTA.** FlashAttention-3 (Shah et al., NeurIPS 2024): ~740 TFLOP/s FP16 on H100, about 75% of the 989 TFLOP/s dense peak; ~1.2 PFLOP/s claimed in FP8. FlashAttention-2 (Dao, ICLR 2024): up to 230 TFLOP/s on A100, 50–73% of peak.

**Claimed but unablated.** That FlashAttention-3's gain over FlashAttention-2 is an *I/O* gain. It is not: the mechanism is warp-specialization, TMA async copy, and FP8 — throughput and latency hiding, not fewer bytes. No published ablation reports `dram__bytes` for FA-2 vs FA-3 at fixed shape. The "9× memory-access reduction" figure in the original FlashAttention paper is a benchmark number at one shape (GPT-2, $N=1024$, $d=64$), not a general ratio.

## 4. What Is Known

- **Matmul, as calibration.** $\Omega(n^3/\sqrt{M})$ I/Os (Jia-Wei & Kung 1981; Irony–Toledo–Tiskin 2004), tight constant $2n^3/\sqrt{M}$ (Smith & van de Geijn, 2017). Attention does *not* inherit the $\sqrt{M}$ form, because softmax forbids accumulating over the $N$ dimension in the same way — the $N^2d^2/M$ shape is genuinely different.
- **Upper bound attained.** FlashAttention's $\Theta(N^2d^2/M)$ is realized in code, at $N$ from 512 to 64K, $d \in \{64,128\}$.
- **Small-cache optimality.** Tight up to constants (Saha & Ye, ICML 2024).
- **Quadratic time is hard to beat.** Exact attention in truly subquadratic time refutes SETH (Keles, Wijewardena & Hegde, ALT 2023); approximate attention is subquadratic iff entries are $O(\sqrt{\log N})$ (Alman & Song, NeurIPS 2023).
- **Empirical scale.** FlashAttention: 3× end-to-end on GPT-2 (A100, $N=1024$), 15% over MLPerf 1.1 BERT-large. FlashAttention-2: ~2× over FA-1 at $N=$ 2K–16K on A100.

## 5. What Is Not Known

- **Theoretically open.** The intermediate and large-cache regimes: for $d^2 \ll M \ll Nd$ there is no matching pair of bounds. Also open: whether the lower bound survives dropping the no-cancellation assumption, and whether causal masking (half the work, but an irregular tile shape) changes the constant or only the leading coefficient.
- **Theoretically open.** Any *constant-factor* lower bound. Everything published is $\Omega(\cdot)$ with an unstated constant, so "FlashAttention is optimal" is a statement about exponents only.
- **Empirically open.** Nobody has published measured DRAM traffic for FA-2/FA-3 across a $(N, d, M)$ sweep and fitted the $N^2d^2/M$ curve. The experiment costs a few GPU-hours.
- **Methodologically blocked.** What $M$ *is* on an H100 with a 50 MB L2 and TMA prefetch. The theorem's $M$ is a single number; the hardware has a hierarchy plus an async copy engine that overlaps transfers with compute, so "I/Os performed" and "time spent on I/O" decouple.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus a model mismatch**, not compute cost.

The lower bound counts words crossing one boundary. On real hardware the boundary that matters moves with $N$: at $N=2048$, $d=64$, both $K$ and $V$ (1 MB fp16 total per head) sit in L2 for the entire kernel, so DRAM traffic is $\Theta(Nd)$ — asymptotically *below* the $\Omega(N^2d^2/M)$ bound, without contradicting it, because the bound is about shared memory and the counter reads DRAM. Any experiment that reports DRAM bytes is measuring a different quantity than the one the theorem bounds; any experiment that reports shared-memory traffic (`l1tex__data_pipe_lsu_wavefronts_mem_shared`) measures a quantity with no published lower bound. The evaluation does not measure the thing it names.

Second obstruction: the no-cancellation restriction. Removing it requires a lower bound over arbitrary arithmetic circuits with an $\exp$ gate — a regime where no I/O lower bound technique currently works.

## 7. Current Research (as of 2026)

- **Regime-completion.** Saha, Ye and collaborators (UCSD) extending the ICML 2024 analysis to sparse and causal attention, and to the large-cache gap *(frontier — verify)*.
- **Fine-grained hardness.** Alman & Song (Columbia/Simons) on gradient computation and on limits of subquadratic transformer alternatives (Alman & Yu, ICLR 2025) — time, not I/O, but the same no-fast-algorithm structure.
- **Kernel side.** Tri Dao's group (Princeton/Together), plus Triton and CUTLASS/ThunderKittens work, targeting Blackwell's larger per-SM memory and 5th-gen tensor cores. The interesting question there is whether a larger $M$ shifts the shape into the unresolved regime *(frontier — verify)*.
- **Distributed variants.** Ring Attention (Liu, Zaharia & Abbeel, ICLR 2024) moves the bottleneck to inter-device bandwidth, where a separate and less-studied lower bound applies.

## 8. Concrete Next Experiment

**Question.** Does measured shared-memory↔HBM traffic in FlashAttention-2 scale as $N^2d^2/M$, and what is the empirical constant?

**Scale.** One A100-80GB. Sweep $N \in \{2^{10}, \dots, 2^{16}\}$, $d \in \{32, 64, 128\}$, batch 1, 1 head, fp16, non-causal. Force $M$ by compiling FA-2 with pinned tile sizes at four settings: 32 KB, 64 KB, 100 KB, 164 KB per block.

**Control arm.** Same sweep on a materializing reference kernel (explicit `S = Q @ K.T`, `torch.softmax`, `P @ V`), whose I/O is known to be $\Theta(N^2)$. This calibrates the counter: if the reference does not measure $\Theta(N^2)$, the measurement pipeline is wrong, not the theory. Second control: run every point with L2 persistence disabled (`cudaLimitPersistingL2CacheSize = 0`) and with the L2 fully carved out, to bound how much of the discrepancy L2 absorbs.

**Deciding number.** Fit $\log(\text{bytes}) = \alpha \log N + \beta \log d + \gamma \log M + c$ over the sweep, restricted to $N \ge 16384$ where $K,V$ exceed L2. The prediction is $(\alpha, \beta, \gamma) = (2, 2, -1)$. **The single number is $\hat\gamma$.** If $\hat\gamma \in [-1.15, -0.85]$, the cache-size dependence is confirmed and $e^{c}$ gives the first empirical constant for the bound. If $\hat\gamma > -0.5$, traffic barely responds to $M$, and the two-level model does not describe the machine — which promotes the methodological block from suspicion to result.

## 9. Key References

- **[Foundational]** Hong Jia-Wei, H. T. Kung. *I/O Complexity: The Red-Blue Pebble Game.* STOC, 1981.
- **[Foundational]** Dror Irony, Sivan Toledo, Alexander Tiskin. *Communication Lower Bounds for Distributed-Memory Matrix Multiplication.* Journal of Parallel and Distributed Computing, 2004.
- **[Foundational]** Grey Ballard, James Demmel, Olga Holtz, Oded Schwartz. *Minimizing Communication in Numerical Linear Algebra.* SIAM Journal on Matrix Analysis and Applications, 2011.
- **[SOTA — theory]** Barna Saha, Christopher Ye. *The I/O Complexity of Attention, or How Optimal Is FlashAttention?* ICML, 2024. — arXiv:2402.07443
- **[SOTA — systems]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135
- **[SOTA — systems]** Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR, 2024. — arXiv:2307.08691
- **[SOTA — systems]** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.* NeurIPS, 2024. — arXiv:2407.08608
- **[Related hardness]** Josh Alman, Zhao Song. *Fast Attention Requires Bounded Entries.* NeurIPS, 2023. — arXiv:2302.13214
- **[Related hardness]** Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, Chinmay Hegde. *On the Computational Complexity of Self-Attention.* ALT, 2023.
- **[Related]** Tyler Michael Smith, Robert van de Geijn. *Pushing the Bounds for Matrix-Matrix Multiplication.* 2017 (constant-tight matmul I/O bound).
- **[Survey]** Grey Ballard, Erin Carson, James Demmel, Mark Hoemmen, Nicholas Knight, Oded Schwartz. *Communication Lower Bounds and Optimal Algorithms for Numerical Linear Algebra.* Acta Numerica, 2014.

## 10. Worked Example

**Setting.** A100-80GB, one head, $N = 8192$, $d = 64$, fp16. $M = 164$ KB $= 82{,}000$ elements. $Q,K,V,O$ are each $8192 \times 64 \times 2\,\text{B} = 1.05$ MB.

**Materializing reference.** $S$ is $8192^2$ fp16 = 134 MB. Write $S$, read it, write $P$, read $P$: $4 \times 134 = 536$ MB, plus 4 MB for $Q,K,V,O$. Total ≈ **540 MB**.

**FlashAttention.** Tile widths $B_c = \lceil M/4d \rceil = 320$, $B_r = \min(B_c, d) = 64$. Outer loop over $K,V$ blocks: $T_c = \lceil 8192/320 \rceil = 26$. Each outer step streams all of $Q$ and read-modify-writes all of $O$: $26 \times (1.05 + 2 \times 1.05) = 82$ MB, plus $K,V$ read once (2.1 MB). Total ≈ **84 MB**. Ratio 6.4×, and $N^2d^2/M \times 2\,\text{B} = 2.75\times10^{11}/82000 \times 2 = 6.7$ MB — the asymptotic formula is **12× below** the honest block-count, because it hides the $\Theta(1)$ constant from re-streaming $Q$ and $O$.

**Where the obstruction becomes visible.** Now measure it. `dram__bytes` on this shape reports well under 84 MB: $K$ and $V$ together are 2.1 MB and $Q,O$ 2.1 MB, all of which fit inside the 40 MB L2, so after the first pass every subsequent outer iteration's re-read of $Q$ is an L2 hit and never touches HBM. Measured DRAM traffic collapses toward $\Theta(Nd) \approx 4$ MB — a **20× gap** from the model's prediction, in the direction of *less* traffic than a proven lower bound allows.

Nothing is contradicted: the theorem bounds shared-memory traffic and the counter reads DRAM. But that is exactly the problem. The only regime where the two coincide is $Nd \gg$ L2, i.e. $N \gtrsim 160{,}000$ at $d=64$ — beyond any shape in the FlashAttention papers' benchmarks. Every published claim that FlashAttention is "I/O-optimal in practice" is therefore made at scales where the measurement cannot see the quantity the optimality theorem is about.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*