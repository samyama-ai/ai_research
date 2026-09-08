---
id: 31-distributed-training/deterministic-reduction-order-cost
title: "Deterministic Reduction Order Cost at Scale"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Deterministic Reduction Order Cost at Scale

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/deterministic-reduction-order-cost` · **Status:** empirically-open

## 1. Problem Statement

Floating-point addition is not associative, so a data-parallel gradient all-reduce returns a bit-pattern that depends on the order in which partial sums are combined. Production collective libraries pick that order dynamically — by message size, channel count, detected topology, protocol (LL/LL128/Simple), and world size — so two runs of the same job on different node allocations, or the same job resumed at a different world size, produce different gradients in the last few mantissa bits, and therefore different final weights.

The problem: **quantify the end-to-end cost of forcing a fixed reduction order, and determine whether that cost buys anything a cheaper mechanism cannot.**

Three variants, of different difficulty:

- **Measurement.** Define a determinism predicate that is checkable and that covers the failure modes practitioners actually hit (elastic resize, node repair, checkpoint restart), not just same-allocation replay.
- **Method.** Build a reduction schedule that is order-invariant across world size and topology, and measure its throughput penalty $\rho$ at $\ge 1024$ accelerators on a real pretraining job.
- **Theory.** Bound the divergence between two runs that differ only in reduction order, as a function of steps $t$, curvature, and unit roundoff $u$. Decide whether the bound is vacuous (divergence saturates at the seed-to-seed baseline) or informative.

Solving it means: a number for $\rho$ at scale, and a statement of whether bitwise reproducibility changes any downstream decision that a variance-over-seeds protocol would not.

## 2. Formal Setting

Let $W$ workers hold local gradients $g^{(1)},\dots,g^{(W)} \in \mathbb{R}^d$. The exact reduction is $s = \sum_{w} g^{(w)}$. A collective implements a *reduction schedule* $T$ — a binary tree over the $W$ inputs per coordinate, induced by the algorithm (ring, recursive halving–doubling, double-binary-tree, NVLS in-switch reduction) and by the number of channels $c$ and chunk size. The realized value is $\hat s_T = \mathrm{fl}_T(\sum_w g^{(w)})$.

**Roundoff.** With unit roundoff $u$ ($u = 2^{-24}$ for fp32 accumulate, $2^{-8}$ for bf16, $2^{-11}$ for fp16) and $\gamma_n = nu/(1-nu)$, the standard summation bound (Higham 2002, Ch. 4) gives, for a tree of depth $h$,
$$\lVert \hat s_T - s\rVert_\infty \le \gamma_h \sum_w \lVert g^{(w)}\rVert_\infty ,\qquad h = W-1 \text{ (sequential)},\ \ h=\lceil \log_2 W\rceil \text{ (tree)} .$$
Two schedules therefore differ by at most $2\gamma_h \sum_w \lVert g^{(w)}\rVert_\infty$ — a *bound on the difference*, not on its consequence.

**Determinism predicate.** For run $A$ and run $B$ with identical data order, seeds, and hyperparameters, define
$$\mathrm{Det}(\mathcal{C}) \;=\; \mathbb{1}\big[\theta^A_t = \theta^B_t \text{ bitwise},\ \forall t \le \tau\big]$$
over a *configuration class* $\mathcal{C}$. Measured classes, in increasing strength: $\mathcal{C}_1$ = same binary, same nodes, same world size (replay); $\mathcal{C}_2$ = same world size, different physical nodes; $\mathcal{C}_3$ = different $W$ with equal global batch size (elastic resize). Reporting "deterministic" without naming $\mathcal{C}$ is the usual source of confusion.

**Divergence.** $D_t = \lVert \theta^A_t - \theta^B_t\rVert_2 / \lVert \theta^A_t\rVert_2$, measured at fixed step, against the control $D^{\text{seed}}_t$ from two different data-order seeds. The question is whether $D_t \ll D^{\text{seed}}_t$ or $D_t \to D^{\text{seed}}_t$.

**Cost.** $\rho = T_{\text{det}}/T_{\text{base}} - 1$ in wall-clock per step, decomposed into (i) collective time, (ii) forced fp32 accumulation traffic, (iii) replacement of atomics-based kernels (`scatter_add`, embedding-bag backward, some fused optimizers) with deterministic variants.

**Assumptions, and which are violated.** (a) *Gradients are bounded and comparable in magnitude* — violated by loss-spike steps and by MoE experts with skewed token counts, where one addend dominates and cancellation is severe. (b) *The reduction schedule is the only nondeterminism* — violated: cuDNN autotuning, atomics, dropout RNG offsets, and asynchronous checkpoint restore all contribute (Zhuang et al., MLSys 2022). (c) *Global batch is order-invariant* — violated under gradient accumulation, where local accumulation order also matters. (d) *Elastic resize preserves the mathematical update* — false in general for ZeRO/FSDP, where partition boundaries change with $W$.

## 3. State of the Art

**Established.**
- Bandwidth-optimal ring all-reduce moves $2\frac{W-1}{W}S$ bytes per rank; recursive halving–doubling and double-binary-tree variants trade bandwidth for $O(\log W)$ latency (Thakur, Rabenseifner & Gropp, *IJHPCA* 2005; Sanders et al. on two-tree broadcast). These are the schedules whose selection is dynamic.
- Reproducible summation with a fixed error bound and *no* dependence on order is a solved algorithmic problem: pre-rounding / indexed-FP summation (Demmel & Nguyen, *ARITH* 2013; *IEEE Trans. Computers* 2015) gives reproducible sums at a small constant-factor cost on CPUs. Intel MKL ships this idea as Conditional Numerical Reproducibility.
- Nondeterminism materially changes single-run outcomes: Summers & Dinneen (*ICML* 2021) show that flipping one bit of one weight at initialization produces final-accuracy spread statistically indistinguishable from changing the whole seed, on CIFAR-10/ImageNet-scale CNNs.

**Claimed but unablated.**
- Vendor and framework guidance that determinism costs "a few percent" (NVIDIA framework-determinism guidance, Riach, GTC 2019) rests on single-node, single-GPU-family measurements; no published multi-node ablation isolates the *reduction-order* component from the atomics and autotuning components.
- Large-model training reports frequently assert bitwise reproducibility for restart-from-checkpoint. These are benchmark statements about $\mathcal{C}_1$, verified by replay on the same allocation, and do not test $\mathcal{C}_2$ or $\mathcal{C}_3$.

**Benchmark-number-only.** NCCL microbenchmark tables (`nccl-tests`) comparing algorithm/protocol settings exist for throughput, but not paired with a determinism predicate, so they cannot be read as a cost of determinism.

## 4. What Is Known

- **Order-dependence magnitude.** For bf16 gradients with fp32 accumulate, per-element relative differences between two reduction orders are $O(10^{-7})$ at $W=512$; with fp16 accumulate they reach $O(10^{-3})$ on ill-conditioned sums. This follows directly from $\gamma_{\log W}$ and is reproduced in practice by any two-schedule diff.
- **Divergence is exponential then saturating.** Independent reports at 100M–1B parameter scale show relative weight divergence growing from $10^{-7}$ to $O(1)$ within $10^2$–$10^4$ steps, after which the two runs are as different as two seeds. Mechanism: positive local Lyapunov exponents in SGD dynamics; consistent with Summers & Dinneen.
- **Non-reduction sources dominate the fix list.** Zhuang et al. (MLSys 2022) attribute most run-to-run variance at ResNet/ImageNet scale to library nondeterminism plus data order, not to collectives; disabling deterministic kernels alone changes accuracy variance more than reduction order does at $W \le 8$.
- **NCCL is deterministic within $\mathcal{C}_1$.** For a fixed count, algorithm, protocol, channel count and topology, the schedule is fixed; run-to-run bit differences at fixed allocation arise from other layers. The break happens on resize/reallocation, i.e. $\mathcal{C}_2$/$\mathcal{C}_3$.
- **Cost of reproducible summation on CPUs:** roughly $1.2\times$–$3\times$ the time of a plain sum for the indexed-FP scheme (Demmel & Nguyen). No equivalent published GPU-collective figure.

## 5. What Is Not Known

- **Empirically open.** The value of $\rho$ for a *reduction-order-only* intervention at $\ge 1024$ accelerators on a $\ge 7$B-parameter pretraining run. Every ingredient exists (fixed algorithm/protocol/channel env vars, fixed-order two-phase reduce-scatter/all-gather, deterministic kernels); nobody has published the paired throughput-and-determinism table at that scale.
- **Empirically open.** Whether $\mathcal{C}_3$ determinism (bitwise-equal updates across world size) is achievable at all for FSDP/ZeRO without reverting to a $W$-independent accumulation, and what it costs in memory traffic.
- **Theoretically open.** A non-vacuous bound on $D_t$ for non-convex training as a function of $u$, $W$, and step count. Existing forward-error bounds control one step; nothing controls the composition through $10^5$ steps except worst-case exponential amplification.
- **Methodologically blocked.** What determinism is *for*. There is no accepted protocol saying which debugging or compliance decisions require bitwise equality versus a distributional guarantee ("the run's loss curve is within the seed band"). Without that, $\rho$ has no threshold to be compared against.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by absent ground truth**.

Confounding: a job made deterministic differs from the baseline in at least four places at once — collective schedule, accumulation precision, kernel selection, and autotuner disablement. Each has its own throughput cost, and the deterministic-kernel cost (embedding/scatter backward) is often larger than the collective cost. Reported "determinism overhead" numbers do not separate them, so the reduction-order term is not identified.

Absent ground truth: there is no correct $\theta_t$ to compare against. Exact-arithmetic training is unavailable at scale, so one cannot say which reduction order is *right* — only that they differ. And since the difference saturates at the seed-to-seed baseline within a few thousand steps, the natural end-to-end metric (final loss) cannot distinguish "reduction order mattered" from "the seed changed," which is exactly the null hypothesis. Any experiment must therefore be a *variance* experiment over many runs, and at 1024-GPU scale a single arm costs six figures.

## 7. Current Research (as of 2026)

- **Deterministic collectives in libraries.** NCCL and RCCL expose algorithm/protocol/channel pinning; work continues on making in-network reduction (SHARP, NVLS) order-stable, where the switch-side tree depends on the job's node placement *(frontier — verify)*.
- **Framework-level determinism.** PyTorch `torch.use_deterministic_algorithms` coverage keeps expanding; the open edge is distributed optimizers and MoE all-to-all dispatch order.
- **Reproducible numerics.** The Demmel–Nguyen line (Berkeley) and ExBLAS-style exact accumulators; GPU ports exist but are not integrated into DL collectives.
- **Elastic/fault-tolerant training** (Meta, DeepSpeed, TorchTitan) increasingly makes $\mathcal{C}_3$ the operationally relevant class, since node repair changes $W$ mid-run *(frontier — verify)*.
- **Regulatory reproducibility.** Model-provenance and audit requirements are pushing toward attestable retraining, which is the one use case that plausibly needs bitwise equality rather than a distributional guarantee.

## 8. Concrete Next Experiment

**Scale.** 1024 H100s (128 nodes), a 7B dense transformer, 20B tokens (~5,000 steps at 4M-token global batch). Cheap enough to run four arms in about 3 GPU-days each per arm at 40% MFU.

**Arms.**
1. **Control (baseline):** stock NCCL autotuning, cuDNN benchmark on, atomics kernels, bf16 grads / fp32 accumulate.
2. **Reduction-order-only:** pin `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, fixed `NCCL_NCHANNELS`, rank order pinned to physical topology. Everything else as control.
3. **Full determinism:** arm 2 plus deterministic kernels and autotuner disabled.
4. **Seed control:** stock config, different data-order seed — establishes $D^{\text{seed}}_t$.

**Procedure.** Run arms 1–3 twice each, the second replica on a *different* node allocation (class $\mathcal{C}_2$). Record per-step wall clock and $D_t$ at steps $\{1, 10, 10^2, 10^3, 5\times10^3\}$.

**The deciding number.** $\rho_{\text{order}} = T_{\text{arm2}}/T_{\text{arm1}} - 1$, the throughput penalty attributable to reduction order alone, with $D_t \equiv 0$ verified across allocations. If $\rho_{\text{order}} < 0.02$, pinned-order all-reduce should be the default for any run that may be resumed or resized, and the debate collapses to the deterministic-kernel cost $\rho_{\text{arm3}} - \rho_{\text{order}}$. If $\rho_{\text{order}} > 0.10$, determinism is a deliberate trade and the field should standardize on seed-band reproducibility instead.

## 9. Key References

- **[Foundational]** Nicholas J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002. — summation error bounds $\gamma_n$.
- **[Foundational]** James Demmel, Hong Diep Nguyen. *Fast Reproducible Floating-Point Summation.* IEEE Symposium on Computer Arithmetic (ARITH), 2013.
- **[SOTA]** James Demmel, Hong Diep Nguyen. *Parallel Reproducible Summation.* IEEE Transactions on Computers, 2015.
- **[Foundational]** Rajeev Thakur, Rolf Rabenseifner, William Gropp. *Optimization of Collective Communication Operations in MPICH.* International Journal of High Performance Computing Applications, 2005.
- **[SOTA]** Cecilia Summers, Michael J. Dinneen. *Nondeterminism and Instability in Neural Network Optimization.* ICML, 2021. — arXiv:2103.04514
- **[SOTA]** Donglin Zhuang, Xingyao Zhang, Shuaiwen Leon Song, Sara Hooker. *Randomness in Neural Network Training: Characterizing the Impact of Tooling.* MLSys, 2022. — arXiv:2106.11872
- **[Foundational]** Sharan Chetlur et al. / Paulius Micikevicius et al. *Mixed Precision Training.* ICLR, 2018. — arXiv:1710.03740 (fp32 master accumulation).
- **[Foundational]** Priya Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* Technical report, 2017. — arXiv:1706.02677
- **[Survey]** Tal Ben-Nun, Torsten Hoefler. *Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis.* ACM Computing Surveys, 2019. — arXiv:1802.09941
- **[Systems]** Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC, 2020. — arXiv:1910.02054
- **[Practice]** Duncan Riach. *Determinism in Deep Learning.* NVIDIA GTC talk, 2019, and the associated `NVIDIA/framework-determinism` repository.

## 10. Worked Example

Take one gradient coordinate on $W=512$ ranks. Suppose 511 ranks contribute $g_w = 1.0\times10^{-4}$ and one outlier rank contributes $g_{512} = -5.11\times10^{-2}$ (a near-total cancellation, typical of a rarely-activated embedding row or an MoE expert). Exact sum: $511\times10^{-4} - 5.11\times10^{-2} = 0$.

- **Schedule A (ring, outlier last):** partial sums grow to $5.11\times10^{-2}$, then the outlier is added. In bf16 accumulate ($u=2^{-8}\approx 3.9\times10^{-3}$), the running sum's ulp near $5\times10^{-2}$ is about $2\times10^{-4}$ — *larger than each addend*. Roughly half the $10^{-4}$ contributions are absorbed entirely. Result: $\hat s_A \approx -1.5\times10^{-3}$.
- **Schedule B (double binary tree):** addends of equal magnitude are paired, so partial sums stay small and the cancellation happens once at the root. Result: $\hat s_B \approx -6\times10^{-6}$.

The two differ by ~250×, and neither equals 0. Switching to fp32 accumulate shrinks both to $O(10^{-9})$ — which is the first thing the obstruction makes visible: **most of the order sensitivity is an accumulation-precision problem, not a schedule problem.** A team that pins the schedule and leaves bf16 accumulation on has fixed the wrong term.

Now the second half. Even with fp32 accumulate, $\hat s_A \ne \hat s_B$ in the last bits, so a 5,000-step run bifurcates. Fitting the observed pattern — $D_t$ rising from $10^{-7}$ to $O(1)$ over ~$10^3$ steps — gives an effective amplification of roughly $10^{7/1000} \approx 1.016$ per step, i.e. ~1.6% growth per step. At step 5,000, $D_t$ has long since saturated at $D^{\text{seed}}_t$.

That is the obstruction stated numerically: the intervention that removes the bit difference costs bandwidth and kernel throughput, and the metric most people would use to justify it — final loss — is provably unable to detect the difference, because the two runs have become statistically identical to two seeds. Justifying determinism requires naming a decision (audit, bisecting a numerical bug, exact restart) that bitwise equality serves and seed-band reproducibility does not. Until that decision is named, $\rho_{\text{order}}$ has a value but no threshold.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*