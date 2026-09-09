---
id: 32-hardware-and-kernels/gpu-kernel-cost-model-accuracy
title: "Cost Model Accuracy for GPU Kernel Autotuning"
topic: 32-hardware-and-kernels
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cost Model Accuracy for GPU Kernel Autotuning

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/gpu-kernel-cost-model-accuracy` · **Status:** methodologically-blocked

## 1. Problem Statement

A tensor compiler explores a space of schedules (tile sizes, unroll factors, thread/block mappings, pipelining depth, layout) for one operator on one GPU. The space is $10^6$–$10^{10}$ points; on-device measurement costs 1–100 ms each including compilation. Autotuners therefore substitute a **cost model** $\hat f$ that predicts latency or throughput from the schedule, and measure only the top candidates.

Three variants, routinely conflated:

- **Measurement variant.** What is the ground-truth latency of a kernel, and with what reproducibility? Given clock throttling, cache state, and driver version, is there a protocol under which two labs agree to within the error a cost model is asked to beat?
- **Method variant.** Build $\hat f$ that, plugged into a fixed search, reaches within $\epsilon$ of the measured-oracle schedule using $k$ real measurements. Solving it means dominating the current Pareto frontier of (measurements used, final kernel latency) across unseen operators *and* unseen GPUs.
- **Theory variant.** Characterize when the argmax of a noisy learned surrogate over a combinatorial space is near-optimal — a regret bound rather than a regression bound.

The catalog status is **methodologically blocked**: the field reports pointwise regression error (MAPE, $R^2$) on a metric that does not determine search outcome, against measurements whose own dispersion is the same order as the reported error.

## 2. Formal Setting

Let $\mathcal{S}$ be the schedule space for operator $o$ on device $d$. For $s \in \mathcal{S}$, the physical latency is a random variable; a measurement uses $R$ repeats and a reducer $\rho$:

$$ T^{(R,\rho)}_{o,d}(s) \;=\; \rho\big(t_1,\dots,t_R\big), \qquad \rho \in \{\min, \mathrm{median}, \mathrm{mean}\}. $$

Each $t_i$ comes from CUDA events or wall clock around $R$ back-to-back launches, after $W$ warmup launches, at SM clock $\phi$, with L2 either flushed or warm. **The tuple $(R,\rho,W,\phi,\text{cache state},\text{driver},\text{arg values})$ is part of the definition of the target, and is not standardized across papers.**

A cost model is $\hat f_\theta:\mathcal{S}\times\mathcal{D}\to\mathbb{R}$, trained on $\mathcal{T}=\{(s_i,T(s_i))\}_{i=1}^n$. Reported metrics:

$$ \mathrm{MAPE}=\frac{1}{m}\sum_j \frac{|\hat f(s_j)-T(s_j)|}{T(s_j)}, \qquad \tau = \text{Kendall/Spearman rank correlation on a held-out set}. $$

The quantity that actually matters is **search regret** under a budget of $k$ measurements. With search policy $\pi$ producing candidate set $A_k(\hat f,\pi)$:

$$ \mathcal{R}_k(\hat f) \;=\; \frac{\min_{s\in A_k} T(s)}{\min_{s\in\mathcal{S}} T(s)} - 1 . $$

$\mathcal{R}_k$ depends only on $\hat f$'s ordering **near the top of the distribution**, not on its global fit. Measurement noise floor: $\eta_{o,d} = \mathrm{sd}(T^{(R,\rho)}(s))$ over repeated independent measurement sessions of the *same* $s$. A cost-model error claim is meaningful only when $\mathrm{MAPE} \gg \eta/T$.

Assumptions in common use, and their status:

| Assumption | Status |
|---|---|
| $T$ is deterministic given $s$ | **Violated.** DVFS, thermal state, and ECC/clock drift give session-to-session dispersion of a few percent, larger on consumer cards. |
| Training and test schedules are i.i.d. from one distribution | **Violated.** Search concentrates on a shifting, self-selected region; the model is queried off its training distribution by construction. |
| Latency is transferable across same-SKU devices | Partially violated: same-model GPUs differ by binning and cooling. |
| Per-operator latency composes additively into network latency | **Violated** by fusion, layout conversion, and cache carryover between ops. |
| The measured argmin is the true optimum | Unverifiable; $\min_{s\in\mathcal{S}}T(s)$ is never computed, so $\mathcal{R}_k$ is reported against a *best-found* baseline. |

## 3. State of the Art

**Systems/empirical SOTA.**
- *AutoTVM* (Chen et al., NeurIPS 2018) — gradient-boosted trees over loop-context features, trained online per workload, with a **rank loss** rather than regression loss. Established: rank loss beats regression loss for search outcome. This is the single most reproduced finding in the area.
- *Ansor* (Zheng et al., OSDI 2020) — hierarchical sketch space plus a learned model; the search-space change, not the model, carries most of the gain.
- *Halide auto-scheduler* (Adams et al., SIGGRAPH 2019) — hand-designed features into a small MLP, trained on random programs; CPU-first but the recipe transferred.
- *TenSet* (Zheng et al., NeurIPS D&B 2021) — ~52M measurement records across 6 platforms; the first offline dataset allowing pretraining rather than per-workload online fitting.
- *TLP* (Zhai et al., ASPLOS 2023) — schedule-primitive token sequences into a transformer, avoiding hand-crafted features; reports better top-$k$ selection than TenSet-trained MLPs on TenSet.
- *Roller* (Zhu et al., OSDI 2022) — an **analytical** model over "rTiles" aligned to hardware tiling; constructs near-peak kernels in seconds instead of hours. Evidence that a large part of the schedule space is analytically prunable.
- *TpuGraphs* (Phothilimthana et al., NeurIPS D&B 2023) and the earlier TPU learned model (Kaufman et al., MLSys 2021) — the graph-level analogue, on TPUs.

**Adjacent SOTA on the same measurement question (CPU basic blocks, where it is better studied).** *Ithemal* (Mendis et al., ICML 2019) and the *BHive* benchmark (Chen et al., IISWC 2019). BHive's contribution was to show that published throughput-model errors are protocol-dependent — the same models score differently under different measurement harnesses.

**Claimed but unablated.** Nearly every learned-GPU-cost-model paper reports MAPE/top-$k$ on its own dataset with its own measurement harness, and separately reports end-to-end speedup. Almost none ablates *how much of the end-to-end speedup is attributable to the model versus the search space or the sampling policy*. Cross-paper MAPE comparisons are benchmark numbers on incomparable targets, not measurements of a common quantity.

## 4. What Is Known

- **Ranking beats regression.** Replacing squared-error with a pairwise rank objective improves search outcome at fixed budget (AutoTVM, NeurIPS 2018; reproduced in Ansor and TenSet). Measured at the scale of single operators, thousands of measurements per workload.
- **Offline pretraining transfers within a device, weakly across devices.** TenSet (52M records, 6 platforms, 2021) showed models pretrained on one platform lose substantial top-$k$ accuracy when transferred, and need per-target fine-tuning.
- **Analytical models are competitive on dense GEMM/conv.** Roller (OSDI 2022) reaches kernels comparable to hours-long learned search in seconds on such shapes — i.e. on the best-understood operators, the learned model's marginal value over hardware-aligned enumeration is small.
- **Simulators are not a substitute.** Accel-Sim (Khairy et al., ISCA 2020) reports cycle-level correlation against real NVIDIA hardware in the ~85–90% range on standard suites — useful for microarchitecture research, far too slow and too biased for tuning-loop use.
- **CPU precedent for the noise problem.** BHive (IISWC 2019) found published basic-block throughput models had substantially higher error under an independent harness than as originally reported; Ithemal reported roughly single-digit-percent MAPE against IACA/llvm-mca in the high teens to twenties. No equivalent independent audit exists for GPU tensor-program cost models.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No community-standard measurement protocol for GPU kernel latency. Without a fixed $(R,\rho,W,\phi,\text{cache state})$ and a published noise floor $\eta$, a claimed MAPE of 8% versus 12% is not a comparison. No published GPU cost-model paper reports $\eta$ alongside its error.
- **Methodologically blocked (secondary).** The reported metric is the wrong functional. Regret $\mathcal{R}_k$ is a functional of the top tail of the ordering; MAPE and global $\tau$ can move in the opposite direction to $\mathcal{R}_k$.
- **Empirically open.** Whether *any* learned model contributes to end-to-end tuning gain once search space and sampling policy are held fixed. The ablation is runnable — swap $\hat f$ for a random-ranking control inside an identical Ansor/MetaSchedule loop — and is not reported at scale.
- **Empirically open.** Cross-architecture generalization (Ampere → Hopper → Blackwell) with tensor-core-, TMA-, and cluster-level features. No public dataset spans these with a common protocol.
- **Theoretically open.** No regret bound for surrogate-guided combinatorial search under heteroscedastic, distribution-shifted noise. Standard Bayesian-optimization regret results assume a kernel/GP prior that discrete schedule spaces violate.

## 6. Why It Is Hard

**Confounded measurement and absent ground truth, jointly.**

1. *The target is not a number, it is a protocol.* A kernel's latency at locked clocks with a cold L2 differs from its latency in a warm inference loop by tens of percent. Both are "the" latency. Cost models are trained on one and deployed against the other.
2. *No oracle.* $\min_{s\in\mathcal{S}}T(s)$ is never known; exhaustive search over $10^8$ points is infeasible. Every "regret" number is against a best-found baseline that itself came from the method under test.
3. *Distribution shift is endogenous.* The search visits schedules the model rates highly. Model errors steer the data collection that trains the model — errors are self-confirming, and held-out i.i.d. accuracy does not bound this.
4. *The metric mismatch is not fixable by better fitting.* A model with 30% MAPE but correct ordering of the top $10^{-4}$ of the space beats a 5%-MAPE model that misranks that tail.
5. *Compute cost of the honest experiment.* Establishing $\eta$ and running matched-control ablations across operators, GPUs, and drivers is thousands of GPU-hours of pure measurement with no new method at the end — publishable-value mismatch.

## 7. Current Research (as of 2026)

- **Analytical/hybrid pruning.** Roller-style hardware-aligned enumeration with a learned model only for the residual ordering. Microsoft Research and academic compiler groups.
- **Pretrained sequence models over schedule primitives.** TLP-line work; extension to Triton and CUTLASS-style parameter spaces *(frontier — verify)*.
- **LLM kernel generation as a competing paradigm.** KernelBench (Ouyang et al., 2025) evaluates LLM-written GPU kernels by *correctness plus measured speedup*, sidestepping cost models entirely — and inheriting exactly the same unspecified-measurement problem in its speedup metric.
- **Triton/Mojo/Helion-style autotuners in production compilers**, which mostly abandoned learned cost models for cached exhaustive search over small hand-pruned spaces. This is the strongest practical signal about the state of the problem.
- **Measurement-protocol standardization** remains the gap nobody owns; MLPerf standardizes end-to-end inference, not per-kernel latency.

## 8. Concrete Next Experiment

**"Noise floor first, then a random-ranking control."**

*Scale.* 200 operators (GEMM, conv, attention, normalization, elementwise-fused) sampled from a real model workload, on 3 GPUs: A100, H100, and one consumer card (e.g. RTX 4090). Roughly 2,000–4,000 GPU-hours total.

*Arm 0 — noise floor.* Pick 500 schedules per operator. Measure each in 10 independent sessions separated by driver reload and thermal reset, at locked clocks and at default DVFS, with cold and warm L2. Report $\eta/T$ per condition and the **rank churn**: Kendall $\tau$ between the top-100 orderings from two independent sessions.

*Arm 1 — treatment.* Fixed search (MetaSchedule/Ansor evolutionary loop), budget $k=1000$ measurements, learned cost model (TLP-class, pretrained on TenSet, fine-tuned online).

*Arm 2 — control.* Byte-identical search loop, cost model replaced by uniform random scoring — i.e. random sampling of the same sketch space at the same budget.

*Arm 3 — control.* Same loop, analytical Roller-style scoring, no learning.

*The deciding number.* The median over 200 operators of

$$ \Delta \;=\; \frac{T_{\text{Arm 2}}^{\text{best}} - T_{\text{Arm 1}}^{\text{best}}}{T_{\text{Arm 2}}^{\text{best}}} \quad\text{at } k=1000, $$

reported with a confidence interval built from Arm 0's session-to-session dispersion. If $\Delta \le 2\eta/T$ — plausibly a few percent — the learned cost model contributes nothing beyond the search space, and every MAPE improvement in the literature is measuring an irrelevant quantity. If $\Delta \ge 15\%$ and survives on the consumer card, the model is real and the field's metric is merely inefficient, not wrong.

## 9. Key References

- **[Foundational]** Tianqi Chen, Lianmin Zheng, Eddie Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy. *Learning to Optimize Tensor Programs.* NeurIPS, 2018. — arXiv:1805.08166
- **[Foundational]** Tianqi Chen et al. *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* OSDI, 2018. — arXiv:1802.04799
- **[SOTA]** Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI, 2020. — arXiv:2006.06762
- **[SOTA]** Lianmin Zheng, Ruochen Liu, Junru Shao, Tianqi Chen, Joseph E. Gonzalez, Ion Stoica, Ameer Haj-Ali. *TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers.* NeurIPS Datasets & Benchmarks, 2021.
- **[SOTA]** Yi Zhai et al. *TLP: A Deep Learning-based Cost Model for Tensor Program Tuning.* ASPLOS, 2023.
- **[SOTA]** Hongyu Zhu, Ruofan Wu, Yijia Diao, Shanbin Ke, et al. *ROLLER: Fast and Efficient Tensor Compilation for Deep Learning.* OSDI, 2022.
- **[SOTA]** Andrew Adams, Karima Ma, Luke Anderson, Riyadh Baghdadi, et al. *Learning to Optimize Halide with Tree Search and Random Programs.* ACM Transactions on Graphics (SIGGRAPH), 2019.
- **[Measurement]** Charith Mendis, Alex Renda, Saman Amarasinghe, Michael Carbin. *Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation using Deep Neural Networks.* ICML, 2019. — arXiv:1808.07412
- **[Measurement]** Yishen Chen, Ajay Brahmakshatriya, Charith Mendis, Alex Renda, et al. *BHive: A Benchmark Suite and Measurement Framework for Validating x86-64 Basic Block Performance Models.* IISWC, 2019.
- **[Measurement]** Mahmoud Khairy, Zhesheng Shen, Tor M. Aamodt, Timothy G. Rogers. *Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling.* ISCA, 2020.
- **[Adjacent]** Sam Kaufman, Phitchaya Mangpo Phothilimthana, Yanqi Zhou, Charith Mendis, Sudip Roy, Amit Sabne, Mike Burrows. *A Learned Performance Model for Tensor Processing Units.* MLSys, 2021. — arXiv:2008.01040
- **[Adjacent]** Phitchaya Mangpo Phothilimthana et al. *TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs.* NeurIPS Datasets & Benchmarks, 2023.
- **[Frontier]** Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini. *KernelBench: Can LLMs Write Efficient GPU Kernels?* 2025. — arXiv:2502.10517
- **[Survey]** Junru Shao, Xiyou Zhou, Siyuan Feng, Bohan Hou, Ruihang Lai, Hongyi Jin, et al. *Tensor Program Optimization with Probabilistic Programs.* NeurIPS, 2022. (MetaSchedule; contains the clearest statement of the search/model decomposition.)

## 10. Worked Example

One fused $2048\times2048\times2048$ FP16 GEMM + bias + GELU on an A100-SXM4-80GB. Peak FP16 tensor-core throughput is 312 TFLOP/s; the kernel is $2\cdot2048^3 = 17.2$ GFLOP, so the hard floor is

$$ T_{\min} \;=\; \frac{17.2\times10^9}{312\times10^{12}} \;\approx\; 55\ \mu\text{s}. $$

Suppose a tuner's best schedule measures $T=71\ \mu$s (78% of peak, a realistic good result) and the cost model predicts $\hat f = 66\ \mu$s. The reported error is $|66-71|/71 = 7.0\%$ — a good MAPE number.

Now the measurement. Re-measure the same binary across conditions:

```
locked clocks 1410 MHz, cold L2, R=100, rho=min : 71.4 us
locked clocks 1410 MHz, warm L2, R=100, rho=min : 68.9 us
default DVFS, warm L2, R=100, rho=median        : 74.6 us
default DVFS, thermally soaked (60 s loop)      : 78.1 us
```

Spread: 68.9–78.1 $\mu$s, i.e. $\pm 6.2\%$ about the mean of 73.3 $\mu$s. **The dispersion attributable to protocol choice alone is as large as the model error being claimed.** A second model reporting 4.5% MAPE under its own harness cannot be said to be better.

The second half of the obstruction. Take the top 5 schedules by model score and their measured latencies:

| rank by $\hat f$ | $\hat f$ ($\mu$s) | measured ($\mu$s) |
|---|---|---|
| 1 | 62 | 84 |
| 2 | 64 | 71 |
| 3 | 66 | 92 |
| 4 | 67 | 70 |
| 5 | 69 | 88 |

Global MAPE over these five is 18%. But the tuner measures all five and keeps the minimum: 70 $\mu$s. Its regret against a 68 $\mu$s best-found oracle is 2.9%. Halving MAPE by improving predictions on the three *bad* schedules (84, 92, 88) would change the search outcome by exactly zero. Conversely, a model with 40% MAPE that put the 70 $\mu$s schedule first would let the tuner stop after one measurement — a 5× budget saving invisible to every metric currently reported.

The number the field reports and the number the field wants are decoupled, and the measurement they are both computed against is not pinned down. That is the block.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*