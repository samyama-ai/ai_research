---
id: 32-hardware-and-kernels/automatic-kernel-fusion-vs-handwritten
title: "Automatic Kernel Fusion Decisions Beating Hand-Written Kernels"
topic: 32-hardware-and-kernels
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Automatic Kernel Fusion Decisions Beating Hand-Written Kernels

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/automatic-kernel-fusion-vs-handwritten` · **Status:** empirically-open

## 1. Problem Statement

**Input.** A tensor program as a dataflow graph $G=(V,E)$ (operators, shapes, dtypes, layouts), a target accelerator description $H$ (SM count, register file size, shared-memory capacity, tensor-core shapes, HBM bandwidth), and a workload distribution over input shapes.

**Output.** A *partition* of $V$ into fused kernels plus a schedule for each part: tiling, loop order, memory placement, pipelining depth, and (on Hopper/Blackwell-class hardware) warp specialisation and async-copy structure.

**Decision predicate.** Does there exist a compiler that, with no per-operator human tuning, produces end-to-end latency $\le$ the best hand-written kernel library plus hand-written fused kernels, on the *same* hardware and numerics, across a workload set the compiler authors did not choose?

Three variants, of very different difficulty:

- **Measurement.** Define "beating hand-written" so the comparison is not rigged: same numerics, same shapes, same tuning budget disclosed, and hand-written baselines that are current rather than a two-year-old cuDNN release.
- **Method.** Search or learn fusion+schedule decisions that reach the human frontier on *algorithmically nontrivial* fusions — attention, MoE routing, fused optimizer+communication — not just elementwise-after-GEMM epilogues, which are already solved.
- **Theory.** Characterise when a fusion is profitable. Even the sub-question "given a fixed fusion partition, is the optimal tiling computable in polynomial time?" has no clean answer for tensor-core targets with recompute.

Solving it means: a checked-in autotuner beats the best human kernel on a held-out workload suite, with the win attributable to fusion decisions rather than to a better GEMM.

## 2. Formal Setting

Let a fusion plan be a partition $P = \{V_1,\dots,V_k\}$ of $V$ into connected, acyclic-contracting subgraphs, and let $s_i \in \mathcal{S}(V_i)$ be a schedule for part $i$. Measured objective:

$$T(P,s) \;=\; \sum_{i=1}^{k}\Big( t^{\text{launch}} + \hat t(V_i, s_i, H)\Big),$$

where $\hat t$ is **wall-clock kernel time from on-device timing** (CUDA events or CUPTI), median of $\ge 100$ replays after 20 warmup iterations, clocks locked, L2 flushed between replays. Not a cost-model estimate — the cost model is the thing under test.

Per-part traffic and arithmetic, as counted by the profiler (`dram__bytes.sum`, `sm__inst_executed_pipe_tensor`), give measured intensity

$$I(V_i,s_i)=\frac{F(V_i)}{B(V_i,s_i)}\quad[\text{FLOP/byte}],\qquad \hat t \;\ge\; \max\!\Big(\frac{F}{F_{\max}},\ \frac{B}{\beta}\Big),$$

with $F_{\max}$ the achievable tensor-core throughput and $\beta$ the achieved HBM bandwidth (both measured by microbenchmark, not datasheet). Fusion of $u\to v$ removes intermediate traffic $2|X_{uv}|\cdot \text{sizeof}(dtype)$ but adds recompute factor $\rho \ge 1$ and raises per-thread resource use, cutting occupancy $\theta$.

The gap to be closed is

$$g \;=\; \frac{T(P^{\text{auto}},s^{\text{auto}})}{T(P^{\text{human}},s^{\text{human}})},$$

with the claim "solved" iff $g \le 1$ at 95% bootstrap confidence over the workload set.

**Assumptions, and which are violated.**

1. *$\hat t$ is additive across kernels.* Violated: back-to-back kernels share L2 residency and overlap tail effects; measuring kernels in isolation overstates fused wins by a few percent.
2. *Numerics are held fixed.* Routinely violated — FlashAttention-style fusions change accumulation order, and many compiler wins come from silently permitting fp32→tf32 or different reduction trees.
3. *The search space contains the human kernel.* Frequently false. Warp specialisation, TMA descriptor reuse and persistent-kernel scheduling were absent from Halide/TVM-style spaces for years, so the search cannot reach the frontier at any budget.
4. *Static shapes.* Violated by LLM serving, where sequence length and batch vary per step and the tuned plan is selected at compile time.

## 3. State of the Art

**Established (reproduced, ablated).**
- Loop-fusion + autoscheduling for *memory-bound* subgraphs is solved in production. TorchInductor (Ansel et al., ASPLOS 2024) generates Triton kernels for pointwise/reduction fusion and reports geomean speedups over eager PyTorch on 180+ HuggingFace/TIMM/TorchBench models; the wins are dominated by elementwise fusion and are independently reproducible via the public dashboards.
- Ansor (Zheng et al., OSDI 2020) beats hand-tuned TVM templates and vendor libraries on several CPU/GPU operator sets, up to $3.8\times$ on ARM CPU — but on NVIDIA GPUs the comparison predates modern tensor-core libraries.
- Vendor-style epilogue fusion (GEMM + bias + activation) is at parity: CUTLASS-generated epilogues match hand-written ones because the epilogue space is small and enumerable.

**Claimed but unablated.**
- "Compiler beats cuBLAS/cuDNN" numbers in most compiler papers use baselines pinned to a library version contemporaneous with submission, and shapes selected by the authors. The delta between the reported win and a win against the *current* library is usually not reported.
- Superoptimizers over algebraic + schedule space — TASO (Jia et al., SOSP 2019), Mirage (Wu et al., multi-level superoptimization of tensor programs, 2024–25) — report finding fused kernels competitive with or exceeding hand-written attention variants. These are benchmark numbers on selected graphs; there is no held-out workload protocol.
- LLM-generated kernels. KernelBench (Ouyang et al., 2025) is the only public standardised harness; reported `fast_1` rates (correct *and* faster than the PyTorch reference) were in the single-digit-percent range for frontier models at release, and the reference is eager PyTorch, not a hand-written kernel.

**Systems SOTA vs theory SOTA.** Systems SOTA is a portfolio: Triton/TorchInductor for memory-bound fusion, CUTLASS/cuDNN for GEMM/conv, hand-written FlashAttention-3 for attention. There is no theory SOTA: no known approximation guarantee for the joint fusion+tiling problem.

## 4. What Is Known

- **Fusion wins are bandwidth wins, quantified.** FlashAttention (Dao et al., NeurIPS 2022) achieves $3\times$ on GPT-2 ($L=1024$) end-to-end training vs. an unfused HuggingFace baseline, by never materialising the $N \times N$ attention matrix — an $O(N^2)$-to-$O(N)$ HBM traffic reduction. FlashAttention-2 reaches ~50–73% of A100 peak fp16; FlashAttention-3 reports ~75% of H100 fp16 peak (~740 TFLOP/s) and ~1.2 PFLOP/s in fp8.
- **No compiler found FlashAttention.** The tiled-online-softmax formulation is an algebraic rewrite plus a recompute decision; it was discovered by humans (building on Milakov & Gimelshein's online softmax, 2018, and Rabe & Staats, 2021) and then imported into compilers as a pattern.
- **Search budgets are large and reported.** Ansor-class autoscheduling costs hours of GPU time per network; TVM/Ansor papers report on the order of $10^3$ measured trials per operator.
- **Cost models transfer badly.** TenSet (Zheng et al., NeurIPS 2021 Datasets & Benchmarks) — 13M+ measured program–latency pairs — shows learned cost models degrade across hardware targets; ranking accuracy, not absolute error, is what determines search quality.
- **Memory-bound fusion is where the volume is.** Across TorchInductor's benchmark suites, most generated kernels are pointwise/reduction fusions; the compute-bound cores are still dispatched to libraries.

## 5. What Is Not Known

- **Empirically open (the main gap).** Whether *any* existing automatic method beats current hand-written kernels on a held-out, third-party-chosen workload suite under fixed numerics. The experiment is runnable today on 8 GPUs in days. Nobody has published it because every group's benchmark set is also its tuning set.
- **Empirically open.** Whether LLM-driven kernel synthesis, given the same wall-clock budget as a human expert (say 40 hours), matches that human on a novel fused op. No controlled head-to-head exists.
- **Theoretically open.** Complexity of optimal fusion partitioning under a recompute-aware, occupancy-aware cost model. The hypergraph-partitioning reduction suggests NP-hardness, but no published hardness proof or approximation bound covers the tensor-core setting.
- **Methodologically blocked.** "Hand-written baseline" is undefined. cuDNN 8 vs. cuDNN 9 vs. FlashAttention-3 vs. an internal kernel differ by more than typical compiler wins, so $g$ is not a property of the compiler alone.

## 6. Why It Is Hard

**The specific obstruction is a moving, non-reproducible baseline combined with a search space that provably excludes the winner.**

1. *Non-identifiability of the win.* A reported end-to-end speedup mixes (a) fusion decisions, (b) a better GEMM, (c) layout choices, (d) relaxed numerics. Papers rarely ablate (a) alone. Without that ablation, $g$ measures the whole toolchain, not fusion.
2. *Space coverage, not search quality, is the binding constraint.* If warp specialisation or TMA multicast is not expressible in the schedule IR, no amount of search reaches FlashAttention-3. The failure mode looks like "search converged" — it is silent.
3. *Search cost.* Hours of GPU time per operator × shape, times a dynamic-shape workload, times a new architecture every ~18 months. The autotuning amortises poorly for serving, where shapes change per request.
4. *Ground truth is absent.* Nobody knows the optimal $T^\star$ for a fused attention kernel on H100, so "97% of peak" is measured against a roofline that is itself an estimate.

## 7. Current Research (as of 2026)

- **Triton and Mosaic/Pallas as the target IR.** Compilers increasingly emit Triton rather than raw CUDA, moving the fusion decision above the schedule; this narrows the space-coverage gap but does not close it for warp-specialised pipelines *(frontier — verify)*.
- **Superoptimisation over algebraic + schedule space.** Mirage (CMU, Zhihao Jia's group) searches multi-level ($\mu$Graph) rewrites and reports discovering fused kernels for attention variants; the open question is held-out generalisation.
- **LLM kernel agents.** KernelBench (Stanford, Azalia Mirhoseini's group) plus iterative-repair agents; reported gains come largely from execution feedback loops rather than better priors *(frontier — verify)*.
- **Hand-written frontier keeps moving.** ThunderKittens (Stanford Hazy Research) and CUTLASS 3.x/CuTe make expert kernels cheaper to write, raising the bar the compiler must clear.
- **Tile-level abstractions on the hardware side** (TMA, tile-based tensor memory on Blackwell) shift where the fusion boundary sits.

## 8. Concrete Next Experiment

**Question.** Does automatic fusion beat hand-written kernels on workloads the compiler authors did not pick?

**Scale.** 8× H100 SXM (or 8× MI300X), one week of machine time. 60 fused subgraphs drawn by a third party from three sources: (i) 20 attention variants (GQA, sliding-window, ALiBi, MLA), (ii) 20 MoE routing+grouped-GEMM subgraphs, (iii) 20 optimizer/normalisation/quantisation fusions. Shapes sampled from real serving traces, not powers of two. The suite is registered and published *before* any compiler is run.

**Arms.**
- **A (auto):** TorchInductor + Triton autotuning, max-autotune, budget capped at 2 GPU-hours per subgraph.
- **B (auto-super):** a superoptimiser (Mirage-class) with the same 2-GPU-hour cap.
- **Control arm (human):** best available hand-written kernel — FlashAttention-3 / CUTLASS / vendor library, latest release at run date — plus, for the 20 subgraphs with no library kernel, a single expert given 2 hours per subgraph (matched budget, not matched to a career).
- **Numerics gate:** every arm must match a fp64 reference to the same max relative error tolerance; arms that relax accumulation are disqualified or reported separately.

**Deciding number.** The geometric mean of $g = T_{\text{auto}}/T_{\text{human}}$ over the 60 subgraphs, with a bootstrap CI. $g \le 1.0$ ⇒ the problem is solved for this workload class. $1.0 < g \le 1.15$ ⇒ compilers are practically adequate but not superior. $g > 1.5$ on the attention subset specifically ⇒ the space-coverage hypothesis (§6.2) is confirmed, and the next work is IR expressiveness, not better search.

**Required ablation.** Rerun arm A with fusion disabled but scheduling intact. The difference isolates the fusion decision from the schedule.

## 9. Key References

- **[Foundational]** Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, Saman Amarasinghe. *Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines.* PLDI, 2013.
- **[Foundational]** Tianqi Chen et al. *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* OSDI, 2018. — arXiv:1802.04799
- **[SOTA]** Lianmin Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI, 2020. — arXiv:2006.06762
- **[SOTA]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135
- **[SOTA]** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.* NeurIPS, 2024.
- **[SOTA]** Jason Ansel et al. *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.* ASPLOS, 2024.
- **[SOTA]** Zhihao Jia, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, Alex Aiken. *TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions.* SOSP, 2019.
- **[SOTA]** Mengdi Wu, Xinhao Cheng, Shengyu Liu, Chunan Shi, Jianan Ji, Zhihao Jia et al. *Mirage: A Multi-Level Superoptimizer for Tensor Programs.* Carnegie Mellon University, 2024–2025.
- **[Benchmark]** Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini. *KernelBench: Can LLMs Write Efficient GPU Kernels?* Stanford University, 2025.
- **[Dataset]** Lianmin Zheng, Ruochen Liu, Junru Shao, Tianqi Chen, Joseph E. Gonzalez, Ion Stoica, Ameer Haj-Ali. *TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers.* NeurIPS Datasets and Benchmarks, 2021.
- **[Survey]** Mingzhen Li et al. *The Deep Learning Compiler: A Comprehensive Survey.* IEEE TPDS, 2021.
- **[Related]** Philippe Tillet, H. T. Kung, David Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL @ PLDI, 2019.

## 10. Worked Example

**Instance.** Causal multi-head attention, $B=8$, $H=32$, $N=4096$, $d=128$, fp16, one H100 SXM (HBM3 ~3.35 TB/s, fp16 tensor-core peak ~989 TFLOP/s with sparsity off).

*Unfused traffic.* The score matrix is $B\cdot H\cdot N^2 = 8\cdot32\cdot4096^2 \approx 4.29\times10^9$ elements. Softmax alone reads and writes it at least twice ($S$ write, $P$ read/write, $P$ read for $PV$): call it $4\times$ traffic $= 4\cdot 4.29\times10^9\cdot 2\,\text{B} \approx 34.4$ GB. At 3.35 TB/s that is a **~10.3 ms floor from softmax traffic alone**.

*FLOPs.* $2\cdot 2\cdot B H N^2 d = 4\cdot 8\cdot 32\cdot 4096^2\cdot 128 \approx 2.20\times10^{14}$; causal masking halves useful work to $\approx 1.10\times10^{14}$ FLOP. At an achieved 600 TFLOP/s that is **~0.18 ms**.

The ratio is the whole problem: the unfused version is $\sim\!55\times$ off its own compute bound. Fusing softmax into the tiled loop drops HBM traffic to $Q,K,V,O$ only: $4\cdot B H N d\cdot 2\,\text{B} = 4\cdot8\cdot32\cdot4096\cdot128\cdot2 \approx 1.07$ GB, a **~0.32 ms** traffic floor — now below the compute time. Fusion moves the kernel from bandwidth-bound to compute-bound.

**Where the obstruction becomes visible.** A pointwise-fusion compiler *can* see the traffic saving; that part is easy arithmetic. What it cannot do is the enabling rewrite: softmax is a *global* reduction over the row, so tiling it requires replacing $\text{softmax}$ with the online rescaling recurrence

$$m^{(j)}=\max(m^{(j-1)}, \text{rowmax}(S_j)),\quad \ell^{(j)}=e^{m^{(j-1)}-m^{(j)}}\ell^{(j-1)}+\text{rowsum}(e^{S_j-m^{(j)}}),$$

with $O$ rescaled by $e^{m^{(j-1)}-m^{(j)}}$ at each step. This is an algebraic identity outside a loop-fusion legality checker's vocabulary: it changes the numerics (a different, though stable, accumulation order) and introduces per-tile recompute. Add the schedule side — warp specialisation into producer/consumer groups, TMA async loads, softmax/GEMM overlap — and the gap between "compiler finds the fusion" and "compiler reaches 75% of peak" is another $2$–$3\times$.

Net: on this single instance, a good loop-fusion compiler lands near **1.5–2.5 ms** (fusing epilogues, materialising or partly tiling $S$), while FlashAttention-3 lands near **0.25–0.35 ms**. Measured $g \approx 5$–$8$. The number is not evidence that the search was underfunded; it is evidence that the winning program is not in the space being searched.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*