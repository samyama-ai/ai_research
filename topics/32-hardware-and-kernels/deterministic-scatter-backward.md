---
id: 32-hardware-and-kernels/deterministic-scatter-backward
title: "Deterministic Backward Passes for Scatter-Based Operators"
topic: 32-hardware-and-kernels
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Deterministic Backward Passes for Scatter-Based Operators

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/deterministic-scatter-backward` · **Status:** solved-but-impractical

## 1. Problem Statement

Scatter-based operators — `index_add`, `scatter_add`, `embedding_bag`, sparse-dense matmul, message passing in GNNs, MoE token dispatch/combine, and the $dQ$ accumulation in fused attention — write many source elements to the same destination slot. The standard GPU implementation uses `atomicAdd`, which fixes *which* values are summed but not *the order* in which they are summed. Floating-point addition is not associative, so the returned gradient is a function of thread scheduling.

The problem has three variants that are routinely conflated:

- **Measurement.** Define and estimate the cost of run-to-run bitwise nondeterminism in a training pipeline: how much of end-of-training variance is attributable to atomic reordering, separately from data-order and dropout RNG?
- **Method.** Produce a scatter-reduce kernel whose output is bitwise identical across runs, thread counts, and (ideally) GPU SKUs, at a throughput overhead small enough that people leave it on by default. Target: $\le 1.10\times$ end-to-end step time.
- **Theory.** Bound the worst-case and typical divergence between two orderings of the same accumulation as a function of the index multiplicity distribution and the gradient condition number, and bound how that divergence propagates through $T$ optimizer steps.

Solving it means: a scatter backward that is run-to-run bitwise reproducible, run-invariant to batch/thread partitioning, and within a stated overhead budget on a named workload — with the overhead published as a head-to-head ablation, not a microbenchmark.

**Status rationale (`solved-but-impractical`).** Deterministic constructions exist and are correct (sort + segmented reduction; fixed-point superaccumulators; deterministic tree reductions). All of them cost extra memory traffic or extra passes, so they ship behind opt-in flags and are off by default.

## 2. Formal Setting

Let $s \in \mathbb{R}^{N \times d}$ be source rows, $\iota \in \{1,\dots,M\}^N$ an index vector, and the forward op $y_j = \sum_{i:\iota_i=j} s_i$. The backward of a gather is a scatter, so the object of interest is the accumulation

$$
g_j \;=\; \bigoplus_{i \in S_j} v_i, \qquad S_j = \{i : \iota_i = j\}, \quad n_j = |S_j|,
$$

where $\oplus$ is IEEE-754 addition in working precision with unit roundoff $u$ ($u = 2^{-24} \approx 5.96\times10^{-8}$ for fp32, $2^{-11}$ for bf16 accumulate).

**Quantities, as measured.**

- **Multiplicity distribution** $\{n_j\}$: measured by `torch.bincount(idx)`. Report $\max_j n_j$ and $\mathbb{E}[n_j]$ — these, not $N$, set the error.
- **Order-divergence** $\Delta_j = |g_j^{(\pi_1)} - g_j^{(\pi_2)}|$ for two permutations $\pi_1,\pi_2$. Measured by running the same kernel twice on identical inputs and XOR-ing the bit patterns; report $\|\Delta\|_\infty$ and the ULP histogram, not the mean.
- **Standard bound** (Higham 1993): for any summation order,
  $$\left|\hat g_j - g_j\right| \;\le\; \gamma_{n_j-1}\sum_{i\in S_j}|v_i|, \qquad \gamma_n = \frac{nu}{1-nu},$$
  so $\Delta_j \le 2\gamma_{n_j-1}\sum_i |v_i|$. The **condition number** $\kappa_j = \sum_{i\in S_j}|v_i| \,/\, |g_j|$ converts this to relative error; $\kappa_j$ is unbounded when the gradients cancel.
- **Trajectory divergence** $D_T = \|\theta_T^{(1)} - \theta_T^{(2)}\|_2 / \|\theta_T^{(1)}\|_2$ after $T$ steps from identical seeds. Measured by two full training runs with every other RNG source pinned.
- **Overhead** $\rho = t_{\text{det}} / t_{\text{atomic}}$, measured as wall-clock per optimizer step on the full model, not on an isolated kernel.

**Assumptions, and which are violated.**

1. *Same values, different order.* Violated in practice: many kernels also change the **split** (split-K, chunk count) with the launch configuration, so runs differ in the parenthesization tree, not just a permutation. Determinism therefore requires fixing the partition, not just sorting.
2. *Accumulator = storage precision.* Violated: bf16/fp16 scatter with fp32 accumulate is common, and the atomic is often performed in the *storage* type on older paths, giving $u = 2^{-8}$ for bf16 atomics.
3. *Bounded multiplicity.* Violated in real embedding/MoE traffic, which is Zipfian: $\max_j n_j$ can exceed $\mathbb{E}[n_j]$ by three orders of magnitude in one batch.
4. *Cross-device reproducibility.* Not implied by run-to-run determinism. A sorted reduction is stable per-partition; changing SM count or tile size changes the tree.

## 3. State of the Art

**Established (correct and shipping).**

- **Sort + segmented reduction.** PyTorch's `torch.use_deterministic_algorithms(True)` routes `index_add_`, `scatter_add_`, `index_put_` and `embedding_bag` backward to sorted/segmented implementations, or raises for kernels with no deterministic path. Documented behavior, not a research claim.
- **Deterministic split reductions in fused attention.** `flash-attention` exposes `deterministic=True` for the backward pass, which replaces atomic $dQ$ accumulation with a fixed-partition two-stage reduction (Dao et al., FlashAttention-2, 2023, and the reference implementation).
- **Reproducible summation with error-free transforms.** ReproBLAS (Demmel, Ahrens, Nguyen) and ExBLAS (Collange, Defour, Graillat, Iakymchuk, *Parallel Computing* 2015) give order-independent sums via pre-rounding bins or a long fixed-point superaccumulator — bitwise reproducible *and* more accurate than naive summation, at a constant-factor cost.
- **Deterministic segmented primitives.** CUB/Thrust `DeviceSegmentedReduce` and merge-based SpMV (Merrill & Garland, SC 2016) give fixed reduction trees over sorted segments.
- **Hardware-level determinism.** GPUDet (Jooybar et al., ASPLOS 2013) shows a deterministic GPU architecture is buildable; the reported overhead is ~2× and it has never been in a shipped part.

**Claimed but unablated.**

- "Determinism costs almost nothing." Common in issue threads and release notes. The published evidence is nearly all *kernel-level* microbenchmarks on uniform indices. End-to-end $\rho$ on a Zipfian-index model at production scale is largely unpublished.
- "Nondeterminism does not affect final quality." Contradicted at small scale (§4) and untested at large scale.

**Benchmark-number-only.** Most quoted slowdowns for deterministic scatter (commonly cited in the $1.5\times$–$10\times$ range depending on multiplicity) come from single-kernel timings in framework issue trackers, not peer-reviewed head-to-head ablations. Treat them as anecdotes.

## 4. What Is Known

- **Nondeterminism is not negligible at small scale.** Summers & Dinneen (ICML 2021) show that flipping a *single weight's* least-significant bit at initialization produces final-accuracy spread comparable to changing the seed entirely, on CIFAR-10/ImageNet-scale CNNs. Order-of-summation noise is a strictly larger perturbation than one LSB.
- **Tooling-induced variance is measurable.** Zhuang, Zhang, Song & Hooker (MLSys 2022) characterize run-to-run variance from cuDNN algorithm selection and nondeterministic kernels; top-1 accuracy spreads of roughly 0.5–1 point on standard image classifiers, at ResNet-scale.
- **Variance in DL systems is systemic.** Pham et al. (ASE 2020) report top-1 accuracy ranges up to ~2.9 points across identical-configuration runs on CIFAR-scale models with GPU nondeterminism enabled.
- **Error bounds are tight and classical.** Higham (1993) $\gamma_n$ bound above; Kahan compensated summation reduces the bound to $2u + O(nu^2)$ but is *not* by itself order-independent under parallel splitting.
- **Order-independence is achievable exactly.** ExBLAS/ReproBLAS give bit-identical sums independent of order and of the number of threads, at a cost of a wide accumulator (ExBLAS uses a ~2098-bit fixed-point accumulator for fp64) plus periodic normalization.
- **Inference-side analogue is now demonstrated.** Batch-invariant kernels remove run-to-run variation in LLM serving (Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025), at a reported throughput cost — evidence that fixing the *partition*, not just the order, is the operative requirement.

## 5. What Is Not Known

- **Empirically open.** The end-to-end $\rho$ for full deterministic training of a $\ge 10$B-parameter MoE or GNN with realistic Zipfian index traffic. Runnable today on any large cluster; no published number.
- **Empirically open.** Whether $D_T$ from scatter reordering alone (all other RNG pinned) reaches the same magnitude as seed variance at $\ge 1$B parameters, or whether large-batch averaging suppresses it. Both stories are plausible; nobody has run the paired experiment at scale.
- **Theoretically open.** A tight bound on $D_T$ under Adam for a nonconvex objective given a per-step perturbation of size $\gamma_{n_{\max}}\kappa$. Existing bounds are exponential in $T$ via Lipschitz arguments and are vacuous after a few hundred steps.
- **Theoretically open.** Whether an order-independent, single-pass scatter-reduce exists with $O(1)$ extra state per destination slot and $\le 1.1\times$ memory traffic — or a lower bound proving it cannot.
- **Methodologically blocked.** "Reproducibility" has no agreed operational definition across the axes: run-to-run (same binary, same GPU), thread-count-invariant, SKU-invariant, and version-invariant. Papers claim "deterministic" while meaning different things, so results are not comparable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by cost asymmetry**.

The perturbation injected per step is tiny ($\sim 10^{-7}$ relative) but the training dynamics are chaotic, so its effect on the final model is only visible in the *distribution* over runs. Isolating it requires $\ge 10$ paired full-scale runs with every other nondeterminism source pinned — data order, dropout, dataloader worker interleave, cuDNN autotune, NCCL reduction order, and gradient-accumulation order across data-parallel ranks. NCCL's own all-reduce is not order-guaranteed, so at multi-node scale the scatter contribution cannot be separated without also fixing the collective. That is $10\times$ the cost of a single training run to measure a quantity nobody can predict the sign of.

Meanwhile the fix has a cost that is *certain and immediate*: sorting $N$ indices, or holding a wide accumulator per slot, costs memory traffic on exactly the operator that is already bandwidth-bound. A practitioner facing a certain 1.5–3× slowdown on a hot kernel against an uncertain and unmeasured quality benefit rationally leaves the flag off. The problem stays "solved-but-impractical" because the measurement that would justify paying the cost is the expensive one.

## 7. Current Research (as of 2026)

- **Framework-level completion of deterministic coverage.** PyTorch continues to expand `use_deterministic_algorithms` coverage and raise clear errors for gaps; JAX/XLA offers deterministic scatter lowering at similar cost. Engineering, not research.
- **Batch- and partition-invariant kernels.** Following the 2025 inference-determinism work, extension of fixed-split reductions from inference to training backward passes is being pursued in kernel libraries *(frontier — verify)*.
- **Superaccumulator revival on tensor cores.** Using integer/fixed-point accumulation paths on modern tensor cores to get ExBLAS-style order-independence at near-fp32 throughput *(frontier — verify)*.
- **Determinism as a compliance requirement.** Regulated-domain and audit settings (model provenance, reproducible eval) are the main pull; this changes the cost-benefit calculation without changing the kernel math.
- **Groups.** Meta (PyTorch core), NVIDIA (CUB/cuDNN/Transformer Engine), Tri Dao's group (flash-attention deterministic mode), Berkeley (Demmel/Ahrens lineage on reproducible BLAS), Sorbonne/Uppsala (Graillat/Iakymchuk, ExBLAS).

## 8. Concrete Next Experiment

**Question.** Does scatter-reorder nondeterminism, alone, produce end-of-training variance comparable to seed variance — and what does removing it cost end to end?

**Scale.** A 1.3B-parameter MoE (top-2 of 64 experts, so token dispatch/combine is the dominant scatter) trained on 30B tokens, 64×H100, ~2 days per run. Fixed data order, fixed dropout RNG, fixed cuDNN algorithms, deterministic NCCL reduction order, single seed.

**Arms** (5 runs each, 15 runs total, ~$\sim$90 GPU-days):

| Arm | Scatter kernel | Other RNG |
|---|---|---|
| A (treatment) | atomic (nondeterministic) | fully pinned |
| B (control) | sort + segmented reduce (deterministic) | fully pinned |
| C (reference) | atomic | seed varied |

**Deciding number.** The ratio
$$
R \;=\; \frac{\mathrm{sd}_{\text{A}}[\text{val loss}]}{\mathrm{sd}_{\text{C}}[\text{val loss}]}.
$$
Arm B must show $\mathrm{sd}_{\text{B}} = 0$ exactly (bitwise identical checkpoints) — that is the sanity check that the isolation worked. If $R \ge 0.5$, scatter ordering alone accounts for most of run-to-run variance and determinism is worth paying for. If $R \le 0.1$, the field can stop asking.

**Second number, same runs.** $\rho = t_B/t_A$ per step, wall-clock, end-to-end. This is the missing production overhead figure. Report alongside $\max_j n_j$ for the token-routing histogram, since $\rho$ is a function of multiplicity skew, not of model size.

## 9. Key References

- **[Foundational]** Nicholas J. Higham. *The Accuracy of Floating Point Summation.* SIAM Journal on Scientific Computing, 14(4), 1993.
- **[Foundational]** James Demmel, Hong Diep Nguyen. *Fast Reproducible Floating-Point Summation.* IEEE Symposium on Computer Arithmetic (ARITH-21), 2013.
- **[SOTA]** Sylvain Collange, David Defour, Stef Graillat, Roman Iakymchuk. *Numerical Reproducibility for the Parallel Reduction on Multi- and Many-Core Architectures.* Parallel Computing, 49, 2015. (ExBLAS)
- **[SOTA]** Duane Merrill, Michael Garland. *Merge-Based Parallel Sparse Matrix-Vector Multiplication.* SC, 2016.
- **[SOTA]** Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR, 2024. — arXiv:2307.08691 (deterministic backward mode).
- **[Empirical]** Donglin Zhuang, Xingyao Zhang, Shuaiwen Leon Song, Sara Hooker. *Randomness in Neural Network Training: Characterizing the Impact of Tooling.* MLSys, 2022.
- **[Empirical]** Cecilia Summers, Michael J. Dinneen. *Nondeterminism and Instability in Neural Network Optimization.* ICML, 2021.
- **[Empirical]** Hung Viet Pham, Shangshu Qian, Jiannan Wang, Thibaud Lutellier, Jonathan Rosenthal, Lin Tan, Yaoliang Yu, Nachiappan Nagappan. *Problems and Opportunities in Training Deep Learning Software Systems: An Analysis of Variance.* ASE, 2020.
- **[Architecture]** Hadi Jooybar, Wilson W. L. Fung, Mike O'Connor, Joseph Devietti, Tor M. Aamodt. *GPUDet: A Deterministic GPU Architecture.* ASPLOS, 2013.
- **[Systems]** Nathan Whitehead, Alex Fit-Florea. *Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs.* NVIDIA technical whitepaper, 2011.
- **[Systems]** Horace He and the Thinking Machines Lab team. *Defeating Nondeterminism in LLM Inference.* Thinking Machines Lab blog post, 2025.
- **[Context]** Matthias Fey, Jan Eric Lenssen. *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

## 10. Worked Example

**A three-element accumulation where the relative error is unbounded.**

Three gradient contributions land on embedding row $j$, in fp32:

$$v_1 = 1.0,\qquad v_2 = -1.0,\qquad v_3 = 2^{-24} \approx 5.960\times10^{-8}.$$

Exact sum: $g_j = 2^{-24}$.

| Order | Intermediate | Result |
|---|---|---|
| $(v_1 \oplus v_2) \oplus v_3$ | $0.0$ | $5.960\times10^{-8}$ ✔ |
| $(v_1 \oplus v_3) \oplus v_2$ | $\mathrm{fl}(1 + 2^{-24}) = 1.0$ (ties-to-even) | $\mathbf{0.0}$ ✘ |

Two `atomicAdd` schedules on the same data return $5.96\times10^{-8}$ and $0.0$. Relative error: $100\%$. The classical bound is not violated — $\kappa_j = 2/2^{-24} = 3.4\times10^{7}$, so $\gamma_2 \kappa_j \gg 1$ and the bound is simply vacuous. This is the general situation in backward passes, where positive and negative gradients cancel by construction near convergence.

**Scaling to a realistic slot.** Take a token appearing $n_j = 4096$ times in a batch (plausible for a frequent BPE token or a hot MoE expert), with $|v_i| \sim 10^{-3}$ and near-complete cancellation, $|g_j| \sim 10^{-3}\sqrt{4096} = 6.4\times10^{-2}$ against $\sum|v_i| = 4.1$. Then $\kappa_j \approx 64$ and

$$\frac{\Delta_j}{|g_j|} \;\lesssim\; 2\gamma_{4095}\kappa_j \;\approx\; 2 \times (4095 \cdot 5.96\times10^{-8}) \times 64 \;\approx\; 3.1\times10^{-2}.$$

A **3% run-to-run swing** on one embedding row's gradient. Adam then divides by $\sqrt{\hat v}$, and for a rarely-updated row the second-moment estimate is small, so the *update* swing is larger than the gradient swing.

**Where the obstruction becomes visible.** The deterministic fix for this slot is: sort 4096 indices, then reduce each segment with a fixed tree. The sort touches $N$ index entries plus a permutation buffer — roughly $3\times$ the memory traffic of the atomic path, on a kernel that was already bandwidth-bound. That is a **certain** cost, payable every step. The benefit is a 3% wobble on one row, whose effect on the final model after $10^5$ steps is exactly the quantity §8 proposes to measure and that nobody has measured. The kernel problem is solved; the decision problem is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*