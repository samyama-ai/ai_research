---
id: 31-distributed-training/distributed-second-order-methods
title: "Communication-Efficient Second-Order Methods at Scale"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Communication-Efficient Second-Order Methods at Scale

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/distributed-second-order-methods` · **Status:** empirically-open

## 1. Problem Statement

Second-order and matrix-preconditioned optimizers (K-FAC, Shampoo, SOAP, Muon) reduce the *number of steps* to a target loss relative to Adam. Each step costs more: extra state, extra collectives, and periodic matrix factorizations. The open question is whether the step-count win survives the systems cost at frontier scale.

**Decision predicate.** Fix a model, dataset, and hardware allocation. Let $T_{\text{opt}}$ be wall-clock time to reach a fixed validation loss under optimizer $\text{opt}$, with each arm tuned under an equal hyperparameter-search budget. Solved means: exhibiting a preconditioned optimizer with
$$\frac{T_{\text{2nd}}}{T_{\text{AdamW}}} \le 0.7$$
at $\ge 10^4$ accelerators and $\ge 10^{11}$ parameters, with the ratio not degrading as device count grows.

Three variants, of very different difficulty:

- **Measurement.** Is the speedup measured at equal tuning budget, equal token budget, and equal hardware — or against an undertuned baseline? Most reported wins fail at least one.
- **Method.** Build a preconditioner whose communication volume and update latency stay sublinear in device count $D$ while retaining the curvature information that produces the step-count win.
- **Theory.** Prove a convergence rate for a *stale, block-diagonal, low-precision* preconditioner that is strictly better than the Adam rate under assumptions that actually hold for transformer training.

## 2. Formal Setting

Minimize $f(w) = \mathbb{E}_{x \sim \mathcal{D}}[\ell(w; x)]$, $w \in \mathbb{R}^P$, over $D$ devices with global batch $B$, per-device batch $B/D$.

**Parameter layout.** Partition $w$ into $L$ layer matrices $W_\ell \in \mathbb{R}^{m_\ell \times n_\ell}$, $P = \sum_\ell m_\ell n_\ell$.

**Preconditioned update.** For Shampoo-family methods,
$$W_\ell \leftarrow W_\ell - \eta \, L_\ell^{-1/4} G_\ell R_\ell^{-1/4}, \quad L_\ell \leftarrow \beta L_\ell + G_\ell G_\ell^\top, \quad R_\ell \leftarrow \beta R_\ell + G_\ell^\top G_\ell,$$
with $G_\ell$ the minibatch gradient. Muon replaces $L^{-1/4}(\cdot)R^{-1/4}$ with an orthogonalization $\mathrm{msign}(G_\ell) = UV^\top$ from the SVD $G_\ell = U\Sigma V^\top$, approximated by 5 Newton–Schulz iterations (5 matmuls, no inverse, no persistent factor state).

**Measured quantities.**

| Symbol | Definition | How measured |
|---|---|---|
| $C_{\text{grad}}$ | gradient allreduce bytes/step | $2\frac{D-1}{D} P b$, $b$ = bytes/element; read from NCCL trace |
| $C_{\text{pre}}$ | preconditioner-state traffic/step | bytes moved by factor allgather / broadcast, amortized over the inverse period $\tau$ |
| $M_{\text{state}}$ | optimizer memory | $\sum_\ell (m_\ell^2 + n_\ell^2)b$ for Shampoo vs $2Pb$ for Adam; peak RSS delta |
| $\tau$ | inverse/root recomputation period, in steps | config |
| $s$ | preconditioner staleness | steps since the factors used were computed; $s \le \tau$ |
| $\rho$ | step-count ratio to target loss | $N_{\text{2nd}}/N_{\text{AdamW}}$ |
| $\kappa$ | per-step wall-clock ratio | median step time ratio, steady state |

Speedup is $\rho\kappa$. The optimizer wins only when $\rho\kappa < 1$.

**Assumptions, and which are violated.**

- *Kronecker factorization of the curvature* ($F_\ell \approx A_\ell \otimes B_\ell$). Violated: cross-layer and attention-head correlations are nonzero; the error is not bounded in practice.
- *Stale preconditioner ≈ fresh preconditioner.* Violated at $\tau = 100$–$1000$ during learning-rate warmup and any sharp loss-landscape transition; nobody measures the induced regret directly.
- *Convexity or bounded-variance smoothness* in the theory. Violated for transformers.
- *Homogeneous devices, fault-free run.* Violated at $10^4$ devices; factor recomputation is a synchronous straggler amplifier.
- *Gradient noise small relative to curvature signal.* At large $B$ this holds better, which is exactly why preconditioners look good in the large-batch regime and the comparison confounds with batch size.

## 3. State of the Art

**Systems/empirical SOTA — established.**

- **Distributed Shampoo (PyTorch)**, Shi et al. 2023 (arXiv:2309.06497): a real, open, ZeRO-style sharded implementation with block-diagonal blocking to cap $m_\ell$. It won the **external tuning track of the AlgoPerf: Training Algorithms benchmark (2024)**, ~28% faster to target across the workload suite than the NAdamW baseline. This is the strongest *methodologically controlled* result in the field — AlgoPerf fixes the hardware, the target metrics, and the tuning budget. Scale: ≤8 V100/A100-class devices per workload, models $\le 10^8$ parameters.
- **Muon at $10^{11}$–$10^{12}$ parameters.** Moonshot's Moonlight (Liu et al. 2025, arXiv:2502.16982) reports matching AdamW quality at ~52% of the training FLOPs on a 16B-parameter MoE (3B active), 5.7T tokens, with Muon optimizer state at half of AdamW's and distributed overhead reported under 1% of step time. Kimi K2 (1T MoE) trained with MuonClip. This is the only $\ge 10^{11}$-parameter evidence.

**Claimed but unablated.** The Moonlight 2× figure is a single training run against one AdamW arm; there is no equal-tuning-budget sweep and no independent reproduction at that scale. K-FAC's ImageNet step-count wins (Ba et al. 2017; KAISA, Pauloski et al. SC 2021, reporting 18–36% time-to-solution improvement at up to 512 GPUs) are convnet results and have not transferred cleanly to transformer pretraining.

**Benchmark-number-only.** SOAP (Vyas et al. 2024) reports ~40% fewer iterations than AdamW at 360M parameters — a benchmark number on a small model, with no distributed-cost accounting.

**Theory SOTA.** Gupta, Koren, Singer (ICML 2018) give a regret bound for Shampoo in the online convex setting, with the preconditioner-error term additive. Bernstein & Newhouse (2024) recast Muon/Shampoo as steepest descent under a spectral or induced operator norm — this explains the update rule but yields no rate separating it from Adam on nonconvex objectives.

## 4. What Is Known

- **Step-count reduction is real at small-to-mid scale.** AlgoPerf 2024: ~28% wall-clock reduction, 8 devices, $\le 10^8$ parameters, equal tuning budget. Reproduced by the competition's independent scoring harness.
- **Optimizer memory need not exceed Adam's.** Muon stores one momentum buffer: $Pb$ vs Adam's $2Pb$ — measured 50% state reduction at 16B parameters (Moonlight).
- **Shampoo state is quadratic without blocking.** For a $4096 \times 4096$ matrix, $L$ and $R$ are $2 \times 4096^2 = 33.6$M entries against $16.8$M parameters — 2× the parameter count in fp32. Blocking to $1024$ cuts this to $0.5\times$.
- **Root computation is the latency bottleneck, not the FLOPs.** Inverse 4th roots via eigendecomposition are $O(n^3)$ but run on one device; at $n=4096$ this is seconds on CPU, hence $\tau \ge 100$ in every deployed system.
- **Staleness tolerance is empirically wide at steady state.** $\tau \in [100, 1000]$ costs little late in training (Anil et al. 2021, ICLR, "Scalable Second Order Optimization for Deep Learning").
- **Newton–Schulz orthogonalization is communication-cheap.** 5 matmuls per layer, no factor state to shard or gather, so $C_{\text{pre}} \approx 0$ — the reason Muon scaled first.

## 5. What Is Not Known

- **Empirically open.** Does $\rho\kappa < 0.7$ hold at $\ge 10^4$ accelerators and $\ge 10^{11}$ parameters *at equal tuning budget*? The experiment is runnable — it costs roughly a frontier pretraining run per arm, which is why it is unrun. Nobody has published a matched-budget Muon-vs-AdamW sweep above 10B parameters.
- **Empirically open.** Does the win persist under Chinchilla-optimal-and-beyond token budgets, or is it a warmup-phase effect that closes by 20T tokens?
- **Theoretically open.** No convergence rate for a preconditioner that is simultaneously block-diagonal, stale by $s$ steps, and quantized. No lower bound showing the communication cost of a Kronecker preconditioner must be $\Omega(\cdot)$ in $D$.
- **Theoretically open.** Whether Muon's spectral-norm steepest-descent interpretation implies any rate advantage over Adam on nonconvex smooth objectives.
- **Methodologically blocked.** "Curvature information retained" has no accepted measure. Comparing an approximate preconditioner to the true Hessian or Fisher at $10^{11}$ parameters is infeasible, so no one can attribute a win to *curvature* rather than to an implicit learning-rate schedule or gradient-norm control.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement compounded by compute cost.**

Any preconditioner changes the effective per-layer learning rate. A tuned AdamW baseline with per-layer learning-rate scaling, a good warmup, and gradient clipping recovers part of the reported gap — but nobody runs that arm at scale, because each arm is a full pretraining run. So the field's headline results compare a heavily engineered second-order arm against a default-recipe first-order arm.

This is non-identifiability, not just cost: at the scales where the answer matters, the number of runs needed to separate "curvature helps" from "you retuned the learning rate" exceeds any single lab's budget for optimizer research. AlgoPerf solves the confound by fixing the tuning budget, but only at $10^8$ parameters and 8 devices — exactly the regime where communication cost is negligible and the systems question does not arise.

## 7. Current Research (as of 2026)

- **Muon variants and scaling laws.** Moonshot AI (Moonlight, Kimi K2), Keller Jordan and collaborators on NanoGPT speedruns, Essential AI on Muon scaling-law comparisons. *(frontier — verify current results.)*
- **Distributed Shampoo hardening.** Meta's PyTorch optimizer team; block size, root-recompute scheduling, fp32-factor-in-bf16-training precision issues.
- **Norm-based optimizer design theory.** Bernstein, Newhouse, and the modular-norm line — deriving updates from a chosen norm on weight space rather than from a curvature estimate.
- **Preconditioner sharding.** Placing factor inverses on distinct devices so root computation parallelizes across layers rather than serializing; overlapping the root computation with the backward pass.
- **AlgoPerf-style benchmarking at larger scale** — the community's stated need; no fixed-budget suite yet exists above $10^9$ parameters. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Is the Muon/Shampoo advantage a curvature effect or a tuning artifact, at a scale where communication cost is visible?

- **Scale.** 8B dense transformer, 512 H100s, 400B tokens (~2× Chinchilla). ~3 days/arm; 4 arms ≈ 150k GPU-hours. This is the smallest configuration where $C_{\text{pre}}$ is non-negligible and the result plausibly extrapolates.
- **Arms.** (1) AdamW, default recipe. (2) **Control arm:** AdamW with per-layer learning rates set to $\eta \cdot \sqrt{\text{fan\_out}/\text{fan\_in}}$ and update-RMS clipping — i.e. the cheap first-order approximation of what Muon's orthogonalization does to update scale, *no curvature*. (3) Muon on all 2-D parameters, AdamW on embeddings/norms. (4) Distributed Shampoo, block size 1024, $\tau = 100$.
- **Budget control.** Identical 12-trial quasi-random search per arm over learning rate and warmup, identical hardware, identical data order.
- **Deciding number.** $\rho\kappa$ for arm 3 **relative to arm 2**, at validation loss 2.10. If arm 3 beats arm 2 by $\le 10\%$ wall-clock, the reported second-order advantage is mostly update-scale control, and communication-efficient curvature estimation is the wrong thing to optimize. If it beats arm 2 by $\ge 25\%$, the curvature signal is real and the systems problem is worth the engineering.
- **Secondary instrumentation.** Log $C_{\text{pre}}/C_{\text{grad}}$ and the step-time p99/p50 ratio per arm to quantify straggler amplification from synchronous factor recomputation.

## 9. Key References

- **[Foundational]** James Martens, Roger Grosse. *Optimizing Neural Networks with Kronecker-factored Approximate Curvature.* ICML, 2015. — arXiv:1503.05671
- **[Foundational]** Jimmy Ba, Roger Grosse, James Martens. *Distributed Second-Order Optimization using Kronecker-Factored Approximations.* ICLR, 2017.
- **[Foundational]** Vineet Gupta, Tomer Koren, Yoram Singer. *Shampoo: Preconditioned Stochastic Tensor Optimization.* ICML, 2018. — arXiv:1802.09568
- **[SOTA]** Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer. *Scalable Second Order Optimization for Deep Learning.* ICLR, 2021 (as "Second Order Optimization Made Practical"). — arXiv:2002.09018
- **[SOTA]** Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, Michael Rabbat. *A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale.* 2023. — arXiv:2309.06497
- **[SOTA]** Jingyuan Liu et al. *Muon is Scalable for LLM Training.* 2025. — arXiv:2502.16982
- **[SOTA]** J. Gregory Pauloski, Qi Huang, Lei Huang, Shivaram Venkataraman, Kyle Chard, Ian Foster, Zhao Zhang. *KAISA: An Adaptive Second-Order Optimizer Framework for Deep Neural Networks.* SC, 2021.
- **[Method]** Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade. *SOAP: Improving and Stabilizing Shampoo using Adam.* 2024. — arXiv:2409.11321
- **[Theory]** Jeremy Bernstein, Laker Newhouse. *Old Optimizer, New Norm: An Anthology.* 2024. — arXiv:2409.20325
- **[Benchmark]** George E. Dahl et al. *Benchmarking Neural Network Training Algorithms.* 2023. — arXiv:2306.07179
- **[Survey]** Jorge Nocedal, Stephen J. Wright. *Numerical Optimization.* Springer, 2nd ed., 2006. — background on quasi-Newton and inexact-Newton methods.

## 10. Worked Example

A 4096-hidden transformer layer, one $4096 \times 11008$ MLP up-projection, $D = 512$ devices, bf16 gradients, fp32 factors.

**Gradient traffic.** $P_\ell = 45.1$M params. Ring allreduce: $2\frac{511}{512}(45.1\text{M})(2\text{B}) = 180$ MB per step for this matrix.

**Shampoo factor state.** $L \in \mathbb{R}^{4096\times4096}$, $R \in \mathbb{R}^{11008\times11008}$. Unblocked: $(16.8\text{M} + 121\text{M}) \times 4\text{B} = 551$ MB — **3.1× the parameter bytes**, for one matrix. Blocking at 1024 replaces $R$ with 11 blocks of $1024^2$: state drops to $(4\times1024^2\!\cdot\!4 + 11\times1024^2\!\cdot\!4)/1\text{e}6 \approx 63$ MB, but the preconditioner is now block-diagonal in a basis nobody chose on curvature grounds — it is the basis induced by the column ordering of a weight matrix, which is arbitrary under permutation.

**Root cost.** Eigendecomposition of a $1024^2$ block: $\sim 10 \cdot 1024^3 \approx 1.1\times10^{10}$ FLOPs. Fifteen blocks, $\sim 1.6\times10^{11}$ FLOPs, on a non-tensor-core path at maybe 2 TFLOP/s effective → 80 ms. Amortized over $\tau = 100$ steps: 0.8 ms/step. Negligible — *if* it overlaps. It does not: the recompute step blocks, so at $D = 512$ that one step is $\sim 80$ ms longer, and every device waits. The p99 step time, not the mean, is what the schedule sees.

**Muon on the same matrix.** 5 Newton–Schulz iterations, each 2 matmuls of $4096\times11008$ shapes: $\sim 5 \times 2 \times 2 \cdot 4096 \cdot 11008 \cdot 4096 \approx 7.4\times10^{12}$ FLOPs — 45× more arithmetic than Shampoo's amortized root, but on tensor cores at ~400 TFLOP/s bf16, i.e. ~18 ms, with **zero extra collective and zero persistent state**.

**The obstruction made visible.** The cheaper method by FLOPs (Shampoo) is the expensive one by wall-clock at $D = 512$, because its cost is latency and synchronization, not arithmetic. And having paid either cost, one still cannot tell whether the resulting speedup came from curvature or from the fact that both updates have a controlled spectral norm — which arm 2 of §8 gets for free.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*