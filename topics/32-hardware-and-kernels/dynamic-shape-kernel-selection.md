---
id: 32-hardware-and-kernels/dynamic-shape-kernel-selection
title: "Kernel Selection Under Dynamic Shapes Without Recompilation"
topic: 32-hardware-and-kernels
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Kernel Selection Under Dynamic Shapes Without Recompilation

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/dynamic-shape-kernel-selection` · **Status:** empirically-open

## 1. Problem Statement

A compiled model ships with a finite set of precompiled kernel binaries. At runtime the shapes vary — batch size, sequence length, number of KV-cache blocks, expert token counts in an MoE router, image resolution. The runtime must pick one binary per operator invocation, in microseconds, with no compiler in the loop.

- **Input:** an operator (e.g. GEMM, attention, grouped GEMM), a runtime shape vector $s \in \mathcal{S}$, and a precompiled library $K = \{k_1,\dots,k_m\}$ of kernel variants (tile sizes, pipeline depth, split-K factor, layout, instruction selection).
- **Output:** an index $\pi(s) \in [m]$, chosen online.
- **Objective:** minimize latency (or its tail) over the runtime shape distribution, subject to $m \le M$ (binary-size / compile-time budget) and a per-call dispatch cost $\le \tau$.
- **Solved** means: for a stated workload and device, a dispatcher exists whose regret against per-shape exhaustive autotuning is bounded and small (say $\le 5\%$ geomean, $\le 15\%$ worst case) at $m$ small enough to compile in minutes, with dispatch overhead under a few hundred nanoseconds — and this holds on shapes not seen during library construction.

Three variants, with different difficulty:

- **Measurement:** what is the achievable regret frontier as a function of $m$? Requires a ground-truth oracle $\min_k T(k,s)$ over a dense shape grid. Runnable today; expensive.
- **Method:** jointly choose the library $K$ and the dispatcher $\pi$. This is a facility-location / $k$-medoids problem over a latency metric — combinatorial, but approximable.
- **Theory:** characterize when a small $K$ suffices — i.e. bound the *covering number* of the shape space under the "same kernel is near-optimal" relation. Essentially untouched.

## 2. Formal Setting

Shape space $\mathcal{S} \subseteq \mathbb{Z}_{>0}^d$ (for a GEMM, $d=3$: $M,N,K$). Runtime shape distribution $\mathcal{D}$ over $\mathcal{S}$, measured as the empirical histogram of shapes observed in a trace of the deployed serving stack, not a synthetic sweep.

Latency $T(k,s) \in \mathbb{R}_{>0}$: median over $R \ge 100$ replays after $\ge 20$ warmup iterations, clocks locked, measured with device-side events (CUDA events / `perf` counters), excluding host launch. Report also $p99$ and the inter-run coefficient of variation; a variant whose CV exceeds 2% is not distinguishable from its neighbours at the margins that matter here.

Oracle and regret:

$$T^\star(s) = \min_{k \in \mathcal{K}} T(k,s), \qquad
r(\pi, K; s) = \frac{T(k_{\pi(s)}, s)}{T^\star(s)} - 1 .$$

Library-and-dispatcher design problem, with $\rho \in \{$mean, $p99\}$:

$$\min_{K \subseteq \mathcal{K},\, |K| \le M} \; \min_{\pi} \;
\rho_{s \sim \mathcal{D}}\big[\, r(\pi, K; s) \,\big]
\quad \text{s.t.} \quad c_{\mathrm{disp}} \le \tau,\; \textstyle\sum_i \mathrm{bytes}(k_i) \le B .$$

With $\pi$ unconstrained and $T$ known, the inner problem collapses to $\pi(s) = \arg\min_{k\in K} T(k,s)$, and the outer problem is $k$-median over the cost matrix $T$ — NP-hard in general, but with a $(1+\epsilon)$-style greedy/local-search approximation since the cost is a nonnegative matrix. The realistic problem is harder because $T$ is unknown off the measured grid and $\pi$ must be a cheap function of $s$.

Hardware structure that any model of $T$ must respect. For a tile $(B_M,B_N)$ on a device with $P$ SMs, the *wave quantization* factor is

$$w(k,s) = \frac{\lceil M/B_M \rceil \lceil N/B_N \rceil}{P \cdot o(k)} \Big/ \left\lceil \frac{\lceil M/B_M \rceil \lceil N/B_N \rceil}{P \cdot o(k)} \right\rceil \in (0,1],$$

with $o(k)$ the CTA occupancy per SM. Achieved throughput scales roughly as $w(k,s)$ times a compute/memory roofline term. This makes $T(\cdot, s)$ **piecewise, sawtoothed, and non-monotone in $s$** — the core difficulty.

Assumptions, and which break:

1. $T(k,s)$ is deterministic given $(k,s)$. **Violated:** DVFS, thermal throttling, and clock drift move latency 3–10% across a long run; on multi-tenant GPUs, co-residency moves it more.
2. Kernels compose additively; per-op choice is globally optimal. **Violated:** layout choices couple adjacent ops, and fusion changes the operator set entirely.
3. $\mathcal{D}$ is stationary. **Violated:** LLM serving shape distributions depend on the scheduler, which depends on latency — a feedback loop.
4. Dispatch is free. **Violated:** a table lookup is ~50 ns; a small MLP predictor is 1–10 µs, which exceeds the runtime of many small kernels.

## 3. State of the Art

**Systems/empirical SOTA — established.**
- **Bucketed specialization + padding.** TensorRT optimization profiles, ONNX Runtime, and `torch.compile` dynamic shapes (Ansel et al., *PyTorch 2*, ASPLOS 2024) all reduce to: compile for a few shape buckets, pad or guard-and-recompile otherwise. Established, universally deployed, and the honest baseline.
- **Nimble** (Shen et al., MLSys 2021): a VM-based runtime with dynamic shape support and shape-generic kernels. Established that dynamic-shape execution need not fall back to an interpreter.
- **DietCode** (Zheng et al., MLSys 2022): constructs a shape-generic *micro-kernel* search space so one tuned artifact covers a range of sequence lengths; reports large tuning-cost reductions versus per-shape Ansor and better latency on dynamic BERT workloads. Established as a benchmark number on BERT-family shapes; not ablated across architectures or vendors.
- **Roller** (Zhu et al., OSDI 2022): constructs near-optimal tile configurations analytically from hardware specs in seconds rather than searching. Directly relevant — it shows the good-config set is *predictable*, not only searchable.
- **MikPoly** (Feng et al., ASPLOS 2024): online micro-kernel polymerization — compose a small set of precompiled micro-kernels into a shape-specific kernel at runtime with negligible cost. The closest existing attack on this exact problem.
- **CUTLASS/cuBLASLt heuristics:** vendor libraries ship a learned/hand-tuned dispatcher over a large kernel set. Effective; the selection logic and its regret are not published.

**Claimed but unablated.** That learned cost models (as in Ansor, Zheng et al., OSDI 2020; TenSet; Hidet, Ding et al., ASPLOS 2023) transfer to unseen shapes well enough to replace measurement. Cost-model accuracy is reported as ranking correlation on held-out *programs*, rarely as end-to-end regret on held-out *shapes*.

**Theory SOTA.** None specific. The nearest formal object is $k$-median/facility location on the latency matrix, plus roofline and wave-quantization analysis — neither yields a covering bound over $\mathcal{S}$.

## 4. What Is Known

- **The sawtooth is real and large.** For a fixed tile, GEMM efficiency drops sharply when the tile grid does not fill an integer number of waves. On an A100 (108 SMs), a $128\times128$ tile on an $N{=}4096$, $M{=}1024$ problem gives $8\times32=256$ CTAs $=2.37$ waves — the trailing 0.37 wave costs ~24% of achievable throughput at occupancy 1. Measured repeatedly in NVIDIA's own tiling/quantization guidance.
- **A small library covers a lot.** Vendor GEMM libraries reach near-peak across a wide shape range with $O(10^2)$–$O(10^3)$ variants, not $O(10^6)$. This is evidence — not proof — that the covering number of $\mathcal{S}$ under near-optimality is small.
- **Per-shape autotuning is expensive.** Ansor-class search needs $\sim 10^3$ measured trials per operator, hours to days per network on one GPU. DietCode's contribution is exactly the observation that this scales linearly in the number of shapes and is therefore infeasible for dynamic workloads.
- **Padding is not free.** Padding sequence length up to the next bucket wastes compute proportional to the bucket gap; at 8 buckets over $[1,4096]$ geometrically spaced, expected waste is ~15–25% of FLOPs depending on $\mathcal{D}$. Ragged/variable-length attention kernels (FlashAttention varlen; vLLM PagedAttention, Kwon et al., SOSP 2023) exist precisely to avoid it.
- **Guard-and-recompile has a visible tail.** In PyTorch 2, a shape that misses the guard set triggers a recompilation costing seconds — a $10^4$–$10^6\times$ p99 spike relative to steady-state step time.

## 5. What Is Not Known

- **Theoretically open.** No bound on the covering number: how many kernel variants $m$ are needed so that some $k \in K$ is within $(1+\epsilon)$ of optimal for all $s$ in a box $[1,S]^d$, as a function of $\epsilon$, $d$, and device parameters $(P, o, \text{cache sizes})$. Also open: whether the wave-quantization structure makes the $k$-median instance easier than general (it plausibly has bounded doubling dimension in a suitable metric — unproven).
- **Empirically open.** The regret-vs-$m$ frontier has never been published for a realistic LLM serving trace. The experiment is a large but ordinary measurement sweep; nobody has run it and released the matrix.
- **Empirically open.** Whether a learned dispatcher beats an analytic one (Roller-style, computed from $B_M,B_N,P,o$) once dispatch cost is charged. Both exist; they have not been compared under a fixed latency budget.
- **Methodologically blocked.** "Dynamic-shape performance" has no agreed metric. Papers report geomean speedup over a padded baseline on a shape set the authors chose. Without a published $\mathcal{D}$ from real traces, results are not comparable — the number reported depends mostly on the shape sampler.

## 6. Why It Is Hard

Two obstructions, both specific.

**Absent ground truth at the required density.** Regret is defined against $\min_k T(k,s)$, which requires benchmarking every variant at every shape. For GEMM alone, a modest grid of $64^3$ shapes $\times\ 10^3$ variants $\times\ 100$ replays $\times\ 1$ ms $\approx 2.6\times10^{10}$ ms $\approx 300$ GPU-days. Every published result therefore substitutes a sparse grid and an assumed interpolation — but the true surface is sawtoothed at period $B_M$, so sparse grids alias the exact structure being studied. This is not a funding problem; it is an aliasing problem.

**Confounded measurement.** Reported "dynamic shape speedups" fold together (a) better kernel selection, (b) avoided padding, (c) avoided recompilation stalls, and (d) fusion differences. These have different fixes and different ceilings. Almost no paper reports the padding-free, recompilation-free, fusion-matched arm that would isolate (a).

## 7. Current Research (as of 2026)

- **Micro-kernel composition at runtime** — MikPoly (ASPLOS 2024) and successors; assemble rather than select. *(frontier — verify current follow-ups.)*
- **Analytic config construction** — Roller-lineage work at MSRA and its integration into industrial compilers; attractive because it makes $\pi$ a closed-form function of $s$ with nanosecond cost.
- **Dynamic-shape industrial compilers** — Alibaba BladeDISC, ByteDance's inference stacks, AWS Neuron; all publish engineering reports rather than controlled ablations.
- **Serving-layer shape shaping** — vLLM/SGLang chunked prefill and continuous batching deliberately *reshape* $\mathcal{D}$ into a narrow set of shapes, sidestepping selection. Arguably the most effective deployed answer, and it changes the problem rather than solving it.
- **Autotuning inside Triton/`torch.compile`** — persistent autotune caches keyed by shape bucket; cache-miss behaviour is the open weak point.
- **MoE grouped GEMM** — expert token counts are data-dependent and unbounded, giving a shape distribution with no useful bucketing; a live pressure point. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Publish the regret-vs-library-size frontier for one operator, one device, one real trace.**

- **Scale.** One NVIDIA H100 (or MI300X). Operator: FP16/BF16 GEMM with $N,K$ fixed to Llama-3-8B's four projection shapes, $M$ (token count) drawn from a real vLLM serving trace — 100k requests, chunked prefill on, capturing the empirical $M$ histogram. Variant set: $|\mathcal{K}| = 512$ CUTLASS configurations (tile $\times$ stages $\times$ split-K $\times$ cluster). Measure $T(k, M)$ for all 512 variants at every distinct $M$ in $[1, 8192]$ — 8192 shapes $\times$ 512 variants $\times$ 50 replays. At ~0.3 ms mean kernel time this is ~2 GPU-days per projection shape, ~8 GPU-days total. Tractable.
- **Arms.** (1) Oracle $T^\star$. (2) *Control:* 8 geometric buckets with padding, the standard deployed approach. (3) $k$-medoids-selected library of size $m \in \{1,2,4,8,16,32,64,128\}$ with exact table dispatch. (4) Analytic dispatcher: closed-form wave-quantization + roofline score, library size $m$, no table. (5) Learned dispatcher: gradient-boosted tree on $(M,N,K)$ features, dispatch cost charged.
- **Deciding number.** $m^\star$ = the smallest library size at which trace-weighted mean regret $\rho_{\mathrm{mean}}[r] \le 0.05$ **and** $p99$ regret $\le 0.15$, with dispatch charged. If $m^\star \le 32$, precompiled selection is sufficient and runtime codegen for dynamic shapes is unnecessary for this class. If $m^\star \ge 256$, or no $m$ reaches the target, the composition/JIT direction is required. Secondary number: the gap between arm (4) and arm (3) at $m = m^\star$ — if under 2 percentage points, the learned dispatcher is unjustified.

## 9. Key References

- **[Foundational]** Tianqi Chen et al. *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* OSDI, 2018. — arXiv:1802.04799
- **[Foundational]** Lianmin Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI, 2020. — arXiv:2006.06762
- **[SOTA]** Bojian Zheng, Ziheng Jiang, Cody Hao Yu, Haichen Shen, Joshua Fromm, Yizhi Liu, Yida Wang, Luis Ceze, Tianqi Chen, Gennady Pekhimenko. *DietCode: Automatic Optimization for Dynamic Tensor Programs.* MLSys, 2022.
- **[SOTA]** Hongyu Zhu, Ruofan Wu, Yijia Diao, Shanbin Ke, Haoyu Li, Chen Zhang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Wei Cui, Fan Yang, Mao Yang, Lidong Zhou, Asaf Cidon, Gennady Pekhimenko. *ROLLER: Fast and Efficient Tensor Compilation for Deep Learning.* OSDI, 2022.
- **[SOTA]** Feng Yu et al. *Optimizing Dynamic-Shape Neural Networks on Accelerators via Online Micro-Kernel Polymerization (MikPoly).* ASPLOS, 2024.
- **[Systems]** Haichen Shen, Jared Roesch, Zhi Chen, Wei Chen, Yong Wu, Mu Li, Vin Sharma, Zachary Tatlock, Yida Wang. *Nimble: Efficiently Compiling Dynamic Neural Networks for Model Inference.* MLSys, 2021. — arXiv:2006.03031
- **[Systems]** Jason Ansel et al. *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.* ASPLOS, 2024.
- **[Systems]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Systems]** Yaoyao Ding, Cody Hao Yu, Bojian Zheng, Yizhi Liu, Yida Wang, Gennady Pekhimenko. *Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs.* ASPLOS, 2023. — arXiv:2210.09603
- **[Foundational]** Philippe Tillet, H. T. Kung, David Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL, 2019.
- **[Related]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135

## 10. Worked Example

One projection, one device, integer arithmetic.

Llama-3-8B QKV projection: $N = 6144$, $K = 4096$, FP16, on an A100-80GB (108 SMs, 312 TFLOP/s dense FP16). $M$ = tokens in the batch. Two precompiled variants:

| variant | tile $(B_M,B_N)$ | CTAs at $M{=}256$ | waves ($o=1$) | efficiency $w$ |
|---|---|---|---|---|
| $k_1$ | $128\times128$ | $2\times48 = 96$ | $96/108 = 0.89$ | $0.89$ |
| $k_2$ | $64\times128$ | $4\times48 = 192$ | $192/108 = 1.78$ | $0.89$ |

At $M=256$ both land near 0.89 — a tie. Now $M = 272$ (a real chunked-prefill remainder):

- $k_1$: $\lceil 272/128 \rceil = 3$ row tiles $\Rightarrow 3\times48 = 144$ CTAs $= 1.33$ waves $\Rightarrow$ 2 waves executed, $w = 0.67$. Padded work: $384/272 = 1.41\times$ the useful FLOPs.
- $k_2$: $\lceil 272/64 \rceil = 5 \Rightarrow 240$ CTAs $= 2.22$ waves $\Rightarrow$ 3 waves, $w = 0.74$. Padded work: $320/272 = 1.18\times$.

Useful FLOPs $= 2 \cdot 272 \cdot 6144 \cdot 4096 = 1.37\times10^{10}$. Ideal time $= 43.9$ µs. Predicted: $k_1 \approx 43.9 \cdot 1.41 / 0.67 \cdot \eta^{-1}$, $k_2 \approx 43.9 \cdot 1.18/0.74 \cdot \eta^{-1}$ — with a shared micro-efficiency $\eta \approx 0.85$, $k_1 \approx 108$ µs, $k_2 \approx 82$ µs. Picking $k_1$ costs 32% regret at a shape 16 tokens away from a shape where the two were tied.

Where the obstruction becomes visible: this analytic estimate is *wrong in the other direction* often enough to be untrustworthy. $k_2$'s smaller tile halves arithmetic intensity per CTA, raising L2 traffic for the $K=4096$ reduction; on real silicon $k_2$ frequently loses despite better quantization. Deciding which effect dominates at $M=272$ requires measuring both. Extend that to all $M \in [1,8192]$ and 512 variants and you have the 8-GPU-day sweep of §8 — for one of four projections, on one device, in one dtype. Nobody has published that matrix, which is exactly why the regret-vs-$m$ frontier, and hence the answer to "is precompilation enough?", remains empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*