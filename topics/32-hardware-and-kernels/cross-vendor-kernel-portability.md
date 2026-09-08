---
id: 32-hardware-and-kernels/cross-vendor-kernel-portability
title: "Cross-Vendor Kernel Portability Without Performance Loss"
topic: 32-hardware-and-kernels
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Vendor Kernel Portability Without Performance Loss

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/cross-vendor-kernel-portability` · **Status:** empirically-open

## 1. Problem Statement

**Input:** one source-level description of a deep-learning kernel $k$ (attention, GEMM, fused normalization, MoE dispatch, paged KV-cache read), written once.

**Output:** executable code for each vendor target $v$ in a set $V$ = {NVIDIA Hopper/Blackwell, AMD CDNA3/4, Google TPU v5/v6, Intel Xe/Gaudi, Apple Silicon, ...}.

**Decision predicate:** does there exist a single-source system such that for all $k$ in a workload suite and all $v \in V$,
$$\rho(k,v) \;=\; \frac{T_{\text{expert}}(k,v)}{T_{\text{portable}}(k,v)} \;\ge\; 1-\epsilon$$
with $\epsilon \le 0.05$, where $T_{\text{expert}}$ is the best hand-tuned vendor-native kernel and $T_{\text{portable}}$ is the compiled single-source kernel? "Portability without performance loss" means $\epsilon$ small **on the worst-case pair**, not on the mean.

Three variants, different difficulty:

- **Measurement variant.** Define $T_{\text{expert}}$ so the comparison is fair. Currently unresolved: vendor libraries embed hardware-specific numerics (FP8 scaling, TF32 accumulate) and different levels of engineering investment. This is the blocking variant.
- **Method variant.** Build the compiler/IR/autotuner that achieves $\epsilon \le 0.05$. Empirically open.
- **Theory variant.** Is there a class of kernels for which a hardware-abstract program provably admits a schedule within $(1+\epsilon)$ of optimal on every architecture in a parameterized machine family? Theoretically open; no impossibility result and no positive result.

## 2. Formal Setting

A kernel is a pair $(C, S)$: a **computation** $C$ — an iteration domain, dependences, and arithmetic, as in the polyhedral/Halide model — and a **schedule** $S$ — tiling, memory placement, vectorization, pipelining, instruction selection.

A target $v$ has a machine vector
$$m_v = (P_v,\, B_v,\, \{c_{v,\ell}\},\, \{b_{v,\ell}\},\, W_v,\, \Lambda_v,\, N_v)$$
where $P_v$ = peak FLOP/s at the kernel's dtype (measured, not datasheet: sustained rate of a saturating microbenchmark), $B_v$ = HBM bandwidth (measured by a large-stride stream read), $c_{v,\ell}$ and $b_{v,\ell}$ = capacity and bandwidth of memory level $\ell$ (registers, LDS/SMEM, L2, on-chip vector memory), $W_v$ = matrix-unit tile shape (e.g. $16{\times}8{\times}16$ MMA on Hopper, $16{\times}16{\times}32$ MFMA on CDNA3, $128{\times}128$ systolic on TPU), $\Lambda_v$ = instruction latencies, $N_v$ = number of cores/CUs/tensor cores.

**Measured runtime.** $T(k,v,S)$ = median over 100 iterations after 20 warmup, device-side timing (CUDA events / HIP events / TPU profiler), fixed clocks where the driver permits, one kernel resident on the device, cold L2 flushed between iterations. Report the median and the interquartile range; clock throttling on air-cooled MI300X and H100 SXM moves the mean by 3–10% across a long sweep, so untimed-clock numbers are not comparable.

**Portability gap** for a workload set $K$:
$$\epsilon_{\max}(V,K) = 1 - \min_{k \in K,\, v \in V} \rho(k,v), \qquad
\epsilon_{\text{geo}} = 1 - \Big(\textstyle\prod_{k,v} \rho(k,v)\Big)^{1/|K||V|}.$$

**Roofline normalization** (Williams, Waterman, Patterson, CACM 2009). With arithmetic intensity $I = \text{FLOPs}/\text{bytes moved from HBM}$, the attainable rate is $\min(P_v, I\,B_v)$, giving a vendor-independent efficiency
$$\eta(k,v) = \frac{\text{FLOPs}(k)}{T(k,v)\cdot \min(P_v,\, I B_v)} \in (0,1].$$
$\eta$ is the honest cross-vendor score: it removes the confound that a target is simply faster.

**Assumptions, and where they break.**

1. *$T_{\text{expert}}$ is well defined.* Violated. It is the best kernel anyone has written, which tracks vendor headcount, not hardware. NVIDIA's cuDNN/CUTLASS/cuBLAS receive far more engineering than any AMD or Intel equivalent, so $\rho$ measures staffing as much as compiler quality.
2. *Numerics are held fixed.* Violated. TF32, BF16 accumulate, FP8 E4M3 with per-tensor vs per-block scaling, and flush-to-zero defaults differ across vendors; a "faster" kernel is sometimes a different function.
3. *The machine vector is observable.* Partially violated. Scheduler policy, L2 replacement, and async-copy queue depths are undocumented on every vendor; TPU details are disclosed only at high level.
4. *Kernels compose additively.* Violated. End-to-end model time depends on fusion boundaries and layout choices set outside the kernel, so per-kernel $\rho$ does not integrate to end-to-end $\rho$.

## 3. State of the Art

**Established (reproduced, ablated):**

- **Halide** (Ragan-Kelley et al., PLDI 2013) established the compute/schedule separation: one algorithm, many schedules. Portability comes from rewriting the schedule per target, not from schedule-free portability.
- **TVM** (Chen et al., OSDI 2018) and **Ansor** (Zheng et al., OSDI 2020) established that learned search over schedules beats hand-written vendor libraries on some operators and targets. Ansor reported up to $3.8\times$ over the best baseline on Intel CPU, $2.6\times$ on ARM CPU, $1.7\times$ on NVIDIA GPU across its operator set — but its GPU wins concentrate on non-GEMM ops.
- **Triton** (Tillet, Kung, Cox, MAPL 2019) established a block-level programming model that reaches near-cuBLAS FP16 GEMM on V100 from ~25 lines. Triton is now the backend of PyTorch **Inductor** (Ansel et al., ASPLOS 2024) and has AMD and Intel backends in tree.
- **MLIR** (Lattner et al., CGO 2021) established a reusable multi-level IR; Linalg-on-tensors plus vendor dialects is now the common substrate (IREE, Triton, XLA/OpenXLA, Mojo).
- **Kokkos** (Edwards, Trott, Sunderland, JPDC 2014) and **RAJA** established that single-source C++ portability across CPU/GPU is achievable for HPC stencils and sparse kernels, at a reported few-percent gap on those workloads — a different regime from matrix-unit-bound DL kernels.

**Claimed but unablated, or benchmark-number-only:**

- Triton's cross-vendor parity on AMD. The AMD backend exists and runs; published head-to-head numbers against hand-tuned Composable Kernel / hipBLASLt on the same commit, same numerics, same clocks are scarce. Vendor blog figures are benchmark numbers with no ablation of which schedule choice closed the gap.
- **FlashAttention** (Dao, Fu, Ermon, Rudra, Ré, NeurIPS 2022; FlashAttention-2, Dao 2023; FlashAttention-3, Shah et al., NeurIPS 2024) is the clearest counterexample to portability: each version is retuned per architecture, and FA-3 is explicitly Hopper-specific (warp-specialization, TMA, FP8). The AMD ports (Triton-based, and CK-based) are separate engineering efforts.
- **Hidet** (Ding et al., ASPLOS 2023) reported up to $1.48\times$ over ONNX Runtime/TVM on inference workloads — NVIDIA-only evaluation.
- TileLang, Pallas/Mosaic, and ThunderKittens-style tile abstractions claim expressivity across backends *(frontier — verify: cross-vendor $\rho$ tables are not yet published at parity of effort)*.

## 4. What Is Known

- Attention is where the gap is largest. FlashAttention-2 reached ~50–73% of A100 FP16/BF16 peak (Dao 2023); FlashAttention-3 reported 1.5–2.0$\times$ over FA-2 on H100 with 740 TFLOP/s FP16 (~75% utilization) and ~1.2 PFLOP/s FP8 (Shah et al., NeurIPS 2024). None of these numbers transfer: they depend on Hopper TMA and warp-specialization that CDNA3 does not have.
- Compiler-generated GEMM can match vendor GEMM on a single vendor at fixed shapes: Triton at FP16 on V100 (MAPL 2019) and Inductor's Triton templates in PyTorch 2.x are within a few percent of cuBLAS on many shapes, and lose on small-$K$ and skinny shapes.
- Autotuning cost is the dominant hidden term. TVM/Ansor tuning is reported in the thousands of trials per operator per target — hours to days of device time for a full model.
- CDNA3 (MI300X) has a $64$-wide wavefront, 64 KB LDS per CU, and MFMA tiles of a different shape than Hopper's MMA. A schedule tuned for warp $=32$ and $16{\times}8{\times}16$ tiles is not merely slower on CDNA3; it is often invalid, needing a different tiling to be legal at all.
- Sparse/stencil HPC kernels port at small loss under Kokkos/RAJA; dense matrix-unit kernels do not. The distinguishing feature is that the former are bandwidth-bound (roofline-limited, so $\eta$ is architecture-insensitive) and the latter are matrix-unit-bound.

## 5. What Is Not Known

- **Methodologically blocked (the primary gap).** There is no accepted definition of $T_{\text{expert}}$ that controls for engineering investment. Without it, every reported $\rho$ conflates compiler quality with vendor headcount. No community benchmark fixes numerics, clocks, and effort budget across vendors for DL kernels; MLPerf measures systems end-to-end, not kernel portability.
- **Empirically open.** Whether a single Triton/MLIR-level source can reach $\epsilon_{\max} \le 0.05$ across NVIDIA + AMD + one non-GPU target on a 10-kernel suite. Runnable today on rentable hardware; nobody has published it with matched effort and matched numerics.
- **Theoretically open.** Whether a hardware-abstract schedule language exists whose optimum is within $(1+\epsilon)$ of the per-target optimum for all $m_v$ in a parameterized family. No proof either way; not even a formalization of what "abstract enough to be portable, concrete enough to be fast" means as a constraint on the schedule space.
- Unknown whether the gap is dominated by (a) missing instruction-level features, (b) search-space coverage, or (c) cost-model transfer. These have never been separated by ablation.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement compounded by a non-identifiable attribution.**

The headline number $\rho(k,v)$ has at least four causes mixed into it: hardware capability, vendor library engineering-years, compiler search quality, and numerics differences. Only the third is the object of study. A measured $\rho = 0.6$ on AMD is consistent with "the compiler is bad," "hipBLASLt is bad so the denominator is small and the true gap is worse," and "the FP8 scaling policy differs" — all at once, with no experiment in the literature separating them.

Second obstruction: **the abstraction that enables portability deletes the features that produce the last 30% of peak**. Hopper's TMA + warp-specialized asynchrony, CDNA3's MFMA shapes and LDS banking, and TPU's systolic-array shape all sit below any portable abstraction. Exposing them re-fragments the source; hiding them caps $\eta$.

Third: search cost is multiplicative. $|K| \times |V| \times$ (thousands of autotune trials) makes a clean, effort-matched study cost real GPU-months, which is why the decisive experiment has not been run.

## 7. Current Research (as of 2026)

- **OpenXLA / IREE / StableHLO** — vendor-neutral MLIR compilation to NVIDIA, AMD, TPU, CPU. Established infrastructure; cross-vendor $\rho$ tables at matched effort not published.
- **Triton multi-backend** (OpenAI + AMD + Intel) — one language, three backends in tree. The natural substrate for the deciding experiment.
- **PyTorch Inductor / `torch.compile`** — Triton codegen is the de facto portable path in production.
- **JAX Pallas + Mosaic** — tile-level kernels targeting TPU and GPU from one source *(frontier — verify parity claims)*.
- **Tile-level DSLs**: TileLang, ThunderKittens, CUTLASS 4 / CuTe DSL, AMD Composable Kernel *(frontier — CuTe DSL and CK are explicitly single-vendor; they set the $T_{\text{expert}}$ bar rather than solve portability)*.
- **Mojo / Modular MAX** — claims single-source NVIDIA+AMD kernels at vendor-library parity *(frontier — verify; published comparisons are vendor-run benchmark numbers)*.
- **LLM-driven kernel generation** (KernelBench, Ouyang et al., 2025, and successors) — search over kernel source with an LLM proposer. Promising because it attacks the effort-asymmetry confound directly: equal LLM budget per vendor is a defensible definition of matched effort.

## 8. Concrete Next Experiment

**Question:** is $\epsilon_{\max} \le 0.05$ reachable from one Triton source on NVIDIA + AMD at matched numerics and matched effort?

**Scale.** $|K| = 8$ kernels: FP16 GEMM at three shapes ($4096^3$; $M{=}8192,N{=}8192,K{=}512$; $M{=}1,N{=}8192,K{=}8192$), causal FlashAttention forward and backward at $B{=}8,H{=}32,S{=}8192,d{=}128$, fused RMSNorm+residual at $S{\cdot}B{=}65536, d{=}8192$, top-2 MoE dispatch/combine at 64 experts, paged-KV decode attention at page 16. $|V| = 2$: H100 SXM and MI300X, both clock-locked, both timed with device events, both cold-L2. 16 GPU-hours autotune budget **per kernel per vendor** — equal budget is the effort control.

**Control arms.** (a) $T_{\text{expert}}$: cuBLAS/CUTLASS + FA-3 on H100; hipBLASLt/CK + CK-FlashAttention on MI300X, each at its vendor's recommended settings. (b) A *per-vendor-forked* Triton source, hand-specialized, with the same 16 h budget — this isolates "how much of the gap is the single-source constraint" from "how much is Triton."

**Numerics gate.** Every arm must match the reference within max relative error $10^{-2}$ against an FP32 CPU reference on the same inputs; arms that fail are excluded, not reported as faster.

**Deciding number.** $\epsilon_{\max}$ over the 16 (kernel, vendor) pairs, reported alongside $\eta$ per pair. $\epsilon_{\max} \le 0.05$ ⟹ portability is an engineering matter. $\epsilon_{\max} \ge 0.30$ with the forked-Triton arm at $\le 0.10$ ⟹ the single-source constraint itself is the binding cost, which is the interesting negative result. Cost estimate: $8 \times 2 \times 16 \approx 256$ GPU-hours autotune plus measurement, roughly $1{,}500 on rented capacity — small enough that the absence of this table is a gap in the field, not a resource limit.

## 9. Key References

- **[Foundational]** J. Ragan-Kelley, C. Barnes, A. Adams, S. Paris, F. Durand, S. Amarasinghe. *Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines.* PLDI, 2013.
- **[Foundational]** S. Williams, A. Waterman, D. Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM 52(4), 2009.
- **[Foundational]** T. Chen et al. *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* OSDI, 2018. — arXiv:1802.04799
- **[Foundational]** P. Tillet, H.-T. Kung, D. Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL @ PLDI, 2019.
- **[Foundational]** C. Lattner et al. *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation.* CGO, 2021.
- **[SOTA]** L. Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI, 2020. — arXiv:2006.06762
- **[SOTA]** T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135
- **[SOTA]** T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR, 2024. — arXiv:2307.08691
- **[SOTA]** J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.* NeurIPS, 2024. — arXiv:2407.08608
- **[SOTA]** J. Ansel et al. *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.* ASPLOS, 2024.
- **[SOTA]** Y. Ding et al. *Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs.* ASPLOS, 2023. — arXiv:2210.09603
- **[Related]** H. C. Edwards, C. R. Trott, D. Sunderland. *Kokkos: Enabling Manycore Performance Portability through Polymorphic Memory Access Patterns.* Journal of Parallel and Distributed Computing 74(12), 2014.
- **[Related]** N. Vasilache et al. *Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions.* 2018. — arXiv:1802.04730
- **[Benchmark]** A. Ouyang, S. Guo, S. Arora, A. L. Zhang, W. Hu, C. Ré, A. Mirhoseini. *KernelBench: Can LLMs Write Efficient GPU Kernels?* 2025. — arXiv:2502.10517

## 10. Worked Example

Take causal FlashAttention forward, $B{=}8$, $H{=}32$, $S{=}8192$, $d{=}128$, BF16.

FLOPs (causal, $\approx$ half the dense count): $2 \cdot 2 \cdot B H S^2 d / 2 = 2 \cdot 8 \cdot 32 \cdot 8192^2 \cdot 128 \approx 5.5 \times 10^{15}$ FLOP.

Suppose measured times: H100 SXM with FA-3 at 640 TFLOP/s BF16 → $T = 8.6$ ms. MI300X with CK-FlashAttention at 420 TFLOP/s → $T = 13.1$ ms. A single Triton source, autotuned 16 h on each, gives 590 TFLOP/s on H100 and 250 TFLOP/s on MI300X.

Naively: $\rho_{\text{H100}} = 590/640 = 0.92$, $\rho_{\text{MI300X}} = 250/420 = 0.60$, so $\epsilon_{\max} = 0.40$. The obvious reading — "Triton is much worse on AMD" — is exactly the inference the measurement does not support.

Normalize by roofline instead. Sustained BF16 matrix peak (measured, not datasheet): H100 SXM ≈ 790 TFLOP/s, MI300X ≈ 1050 TFLOP/s. Then $\eta_{\text{Triton,H100}} = 590/790 = 0.75$ and $\eta_{\text{Triton,MI300X}} = 250/1050 = 0.24$, while the expert kernels sit at $0.81$ and $0.40$. So *both* the portable and the expert AMD kernels are far from the machine's own roofline. The 0.60 ratio is measuring a weak denominator as much as a weak numerator: at least $0.40/0.81$ of the shortfall lives in the AMD expert kernel, not in Triton.

Now the non-identifiability bites. Three hypotheses fit these four numbers equally well:

1. Triton's schedule space lacks CDNA3-specific software pipelining across LDS, so the portable kernel is genuinely handicapped.
2. CK-FlashAttention has received roughly one-tenth the engineering of FA-3, so $T_{\text{expert}}$ on MI300X is not "expert."
3. The AMD arm ran with a different FP32-accumulate policy in the softmax, changing both the numerics and the instruction mix.

Nothing in $\rho$, and nothing in $\eta$, distinguishes them. Separating them requires the forked-Triton control arm from §8: if hand-specialized Triton on MI300X reaches 0.65 $\eta$, hypothesis 1 is confirmed and the single-source constraint costs ~0.4 in $\eta$. If it stalls at 0.26, the limit is the hardware/library stack, not the portability abstraction. That single missing control arm is why the problem is empirically open rather than answered.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*