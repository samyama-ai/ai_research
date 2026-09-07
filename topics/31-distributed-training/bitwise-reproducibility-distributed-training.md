---
id: 31-distributed-training/bitwise-reproducibility-distributed-training
title: "Determinism and Bitwise Reproducibility in Distributed Training"
topic: 31-distributed-training
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Determinism and Bitwise Reproducibility in Distributed Training

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/bitwise-reproducibility-distributed-training` · **Status:** solved-but-impractical

## 1. Problem Statement

Two runs of the same training script, same seed, same data order, same code, same cluster. Do they produce the same weights, bit for bit? On a single GPU with deterministic kernels selected, usually yes. Across data-parallel, tensor-parallel and pipeline-parallel workers with collective reductions, usually no — and the divergence grows from one least-significant bit to a measurable difference in final task accuracy.

Three variants, of very different difficulty:

- **Measurement.** Given two runs, quantify *when* and *how much* they diverged, and attribute the divergence to a source (reduction order, autotuner choice, atomics, RNG offset, elastic rescale). Requires a per-step comparison protocol that does not itself perturb the run.
- **Method.** Build a training stack where the bitwise-identity predicate holds for the *realistic* configuration — variable microbatch shapes, failure-triggered restart from checkpoint, changed world size — at an acceptable throughput cost. This is where "solved-but-impractical" bites: order-independent reduction algorithms have existed since 2013, and nobody runs them in a frontier pretraining stack.
- **Theory.** Bound the divergence of two SGD trajectories that differ only by floating-point rounding, as a function of steps $T$, learning rate, and loss curvature. Open in any useful non-convex form.

Solving it means: a stack that satisfies bitwise identity under worker-count change and restart, with measured overhead under ~5% end-to-end, plus a demonstration that when reproducibility is *dropped*, the induced result spread is smaller than the effect sizes people report on.

## 2. Formal Setting

A run is a map $R(\sigma, C) \mapsto (\theta_0, \dots, \theta_T)$ from seed $\sigma$ and configuration $C$ (world size $P$, parallelism degrees, precision, kernel/autotune policy, collective algorithm) to a parameter trajectory. Parameters are IEEE-754 words; identity is on the bit pattern, not the value, so $-0.0 \neq +0.0$ and NaN payloads matter.

**Bitwise reproducibility predicate.** For $C_1, C_2$ in an admissible set $\mathcal{C}$,
$$\mathrm{BR}(\mathcal{C}) \iff \forall\, C_1, C_2 \in \mathcal{C}:\ \mathrm{bits}(\theta_T(\sigma, C_1)) = \mathrm{bits}(\theta_T(\sigma, C_2)).$$
The strength of the claim is entirely the choice of $\mathcal{C}$. Four nested levels, measured by running the pair and diffing checkpoints:
1. $\mathcal{C}_1$: same binary, same $P$, same hardware, rerun. (Weakest; what most "determinism flags" deliver.)
2. $\mathcal{C}_2$: adds restart from checkpoint at an arbitrary step.
3. $\mathcal{C}_3$: adds change of $P$ and of microbatch/global-batch decomposition at fixed effective batch.
4. $\mathcal{C}_4$: adds change of GPU model or collective topology.

**Divergence step.** $\tau = \min\{t : \mathrm{bits}(\theta_t^{(1)}) \neq \mathrm{bits}(\theta_t^{(2)})\}$, measured by hashing (e.g. SHA-256 over the flat parameter buffer) at every step; $\tau = \infty$ means reproducible. Cheap enough to run always: one hash per step per rank.

**Divergence magnitude.** $d_t = \|\theta_t^{(1)} - \theta_t^{(2)}\|_2 / \|\theta_t^{(1)}\|_2$, and $u_t = $ max ULP distance over coordinates. $d_t$ is what grows; $\tau$ is what you control.

**Root cause.** Floating-point addition is non-associative: for a reduction over $P$ shards with per-shard gradient $g_p$,
$$\mathrm{fl}\!\left(\sum_{p \in \pi(1..P)} g_p\right) \ne \mathrm{fl}\!\left(\sum_{p \in \pi'(1..P)} g_p\right)$$
for permutations $\pi \ne \pi'$, with worst-case relative error $\le (P-1)\varepsilon + O(\varepsilon^2)$, $\varepsilon = 2^{-24}$ (fp32) or $2^{-8}$ (bf16 mantissa). NCCL selects ring vs. tree by message size and topology at runtime, so $\pi$ is not fixed by the program.

**Cost.** $\rho = \mathrm{tok/s}(\text{nondeterministic}) / \mathrm{tok/s}(\text{reproducible})$, measured end-to-end at steady state after warmup, not per-kernel.

**Assumptions known to be violated in practice.** (i) That reduction order is a function of the program — false, autotuners and NCCL algorithm selection depend on runtime state and free memory. (ii) That gradient accumulation is exactly associative — false in bf16. (iii) That RNG streams are index-addressed — false where dropout masks are generated per-thread-block, so a different launch geometry gives a different mask. (iv) That kernels are batch-invariant — false for split-reduction attention and matmul kernels, where the per-example result depends on the batch it was run in.

## 3. State of the Art

**Established (theory/numerics).** Order-independent summation is solved. Demmel and Nguyen's reproducible summation (ARITH 2013) and the ReproBLAS line, formalized in Ahrens, Demmel and Nguyen (*ACM TOMS*, 2020), give a reduction whose result is independent of order and of $P$, using a small fixed number of accumulator bins, with a proved error bound and a single extra reduction pass. ExBLAS (Iakymchuk et al.) achieves the same via long accumulators. These are real solutions to the collective-reduction half of the problem and are essentially unused in deep-learning stacks.

**Established (systems).** PyTorch's `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`, cuDNN deterministic algorithm selection, and NVIDIA's `framework-determinism` work (Riach, GTC 2019 onward) reliably deliver $\mathcal{C}_1$ for single-GPU and many multi-GPU data-parallel configurations, and throw on ops with no deterministic implementation. This is genuinely shipped and tested.

**Claimed but unablated.** Vendor and framework statements that a given large-scale stack is "deterministic" are almost always $\mathcal{C}_1$ claims, verified on short runs, with no published $\tau$ under rescale or restart. No public frontier-scale pretraining run has published a $\mathcal{C}_3$ result.

**Benchmark-number-only.** Thinking Machines Lab's *Defeating Nondeterminism in LLM Inference* (2025) built batch-invariant matmul/attention/RMSNorm kernels and demonstrated bitwise-identical LLM outputs across batch sizes — the first clean public attack on the batch-invariance sub-problem. Their throughput cost is reported as a small number of single-configuration measurements (roughly a 1.5–2× slowdown before optimization), not an ablation across shapes and hardware, and the work targets inference, not the backward pass or optimizer state.

## 4. What Is Known

- **Nondeterminism's effect on final accuracy equals a seed change.** Summers and Dinneen (ICML 2021) show that flipping a *single bit* in one initial weight of a ResNet on CIFAR-10 produces the same run-to-run test-accuracy standard deviation (~0.2–0.5 points at ~94% accuracy) as changing the full random seed. Nondeterminism is not a small perturbation; it is a full resample of the trajectory.
- **Confirmed independently at ImageNet scale.** Zhuang et al. (*Randomness in Neural Network Training*, MLSys 2022) find GPU-nondeterminism-only variance statistically indistinguishable from full-seed variance, with ImageNet top-1 spreads of a few tenths of a point across runs.
- **Larger spreads exist in some configurations.** Pham et al. (ASE 2020) report accuracy differences across identical-configuration reruns reaching ~10 points in some vision settings — an outlier relative to the two results above, and configuration-specific.
- **Error growth is bounded per-step, unbounded per-run.** One fp32 all-reduce over $P=1024$ shards differs by at most ~$1023 \cdot 2^{-24} \approx 6\times10^{-5}$ relative. After $10^5$ steps through a non-convex loss with a chaotic Lyapunov exponent, $d_t$ reaches $O(1)$.
- **Deterministic kernels cost something, and the cost is op-specific.** Deterministic cuDNN backward paths for some convolutions and scatter/index ops run measurably slower than the atomics-based default; there is no published end-to-end $\rho$ for a large transformer pretraining run.

## 5. What Is Not Known

- **Empirically open.** $\rho$ for a $\mathcal{C}_3$-reproducible transformer pretraining run at $\ge 10^9$ parameters and $\ge 256$ GPUs. Fully runnable today with reproducible-summation all-reduce plus batch-invariant kernels. Nobody has published it. This is the central gap.
- **Empirically open.** Whether $\mathcal{C}_2$ (restart-identical) alone captures most of the practical debugging value at near-zero overhead, making $\mathcal{C}_3$ unnecessary.
- **Theoretically open.** Any useful bound on $\mathbb{E}[d_T]$ for non-convex SGD under bounded per-step rounding perturbation, without assuming PL/strong convexity. Convex bounds are known and vacuous here.
- **Methodologically blocked.** Attribution. Given $\tau = 3{,}412$, no tool tells you *which* of ~15 candidate sources caused it. Current practice is binary search over disabled features, which costs one full rerun per bisection step.

## 6. Why It Is Hard

The specific obstruction is **cost asymmetry against an unmeasured benefit**. Reproducible summation adds a reduction pass and forbids the fused, atomics-based, autotuned kernels that supply most of the throughput; batch-invariant kernels forbid split-reduction strategies that exist precisely because they win on small batches. Every one of these is a real percentage of tokens/s, paid on every step of a run costing millions of dollars. The benefit — faster debugging, valid A/B comparisons — has never been quantified in the same units.

Second obstruction: **confounded measurement of the benefit**. Because nondeterminism-induced variance equals seed variance (Section 4), you cannot tell a real algorithmic improvement from noise by running once; but the standard fix (run $n$ seeds, compare means) *also* works without determinism, and is cheaper than paying $\rho$ on every run. Determinism only clearly wins for exact-repro debugging of rare failures — a use case whose frequency nobody has published.

## 7. Current Research (as of 2026)

- **Batch-invariant kernels.** Thinking Machines Lab's 2025 inference work is the live thread; extension to backward passes and optimizer states is the obvious next step *(frontier — verify)*.
- **Framework determinism engineering.** PyTorch's deterministic-algorithms coverage and NVIDIA's determinism guidance continue to expand op coverage; this is maintenance-grade, well-established work.
- **Reproducible BLAS revival.** ReproBLAS/ExBLAS-style order-independent reduction has not been ported into NCCL. A NCCL-level reproducible all-reduce is the missing artifact *(frontier — verify)*.
- **Reproducibility in RL and post-training**, where run-to-run variance is far larger and determinism has correspondingly more debugging value (Nagarajan, Warnell and Stone, 2018, is the reference point).

## 8. Concrete Next Experiment

**Question.** What is the end-to-end throughput cost of $\mathcal{C}_3$ bitwise reproducibility for transformer pretraining?

**Scale.** 1.3B-parameter decoder, 20B tokens, 64×H100, tensor-parallel 2 × data-parallel 32, bf16 compute with fp32 master weights. ~2 days per arm.

**Arms.**
- *Control:* stock stack — NCCL default algorithm selection, cuDNN/cuBLAS autotuning on, fused optimizer.
- *Treatment:* reproducible all-reduce (Ahrens–Demmel–Nguyen binned summation over the gradient buffer), batch-invariant matmul/attention/norm kernels, fixed launch geometry, index-addressed RNG, deterministic optimizer reduction order.

**Protocol.** Each arm run three times: (a) rerun at $P=64$, (b) restart from step-5000 checkpoint, (c) rerun at $P=32$ with doubled gradient accumulation (same effective batch). Hash parameters every step; record $\tau$ for each pair.

**Deciding number.** $\rho$, the tokens/s ratio of control to treatment, *conditional on* $\tau = \infty$ for all three treatment pairs. $\rho \le 1.05$ makes always-on reproducibility defensible for production pretraining; $\rho \ge 1.25$ confirms "solved-but-impractical" and moves the field to $\mathcal{C}_2$-only. Secondary readout: $\tau$ for the control arm's pair (c) — expected to be $\tau \le 1$.

## 9. Key References

- **[Foundational]** J. Demmel, H. D. Nguyen. *Fast Reproducible Floating-Point Summation.* IEEE Symposium on Computer Arithmetic (ARITH-21), 2013.
- **[Foundational]** P. Ahrens, J. Demmel, H. D. Nguyen. *Algorithms for Efficient Reproducible Floating Point Summation.* ACM Transactions on Mathematical Software, 2020.
- **[Foundational]** N. J. Higham. *The Accuracy of Floating Point Summation.* SIAM Journal on Scientific Computing, 14(4), 1993.
- **[SOTA]** C. Summers, M. J. Dinneen. *Nondeterminism and Instability in Neural Network Optimization.* ICML 2021. — arXiv:2103.04514
- **[SOTA]** D. Zhuang, X. Zhang, S. Song, S. Hooker. *Randomness in Neural Network Training: Characterizing the Impact of Tooling.* MLSys 2022. — arXiv:2106.11872
- **[SOTA]** H. He et al. *Defeating Nondeterminism in LLM Inference.* Thinking Machines Lab, Connectionism blog, 2025.
- **[Empirical]** H. V. Pham, S. Qian, J. Wang, T. Lutellier, J. Rosenthal, L. Tan, Y. Yu, N. Nagappan. *Problems and Opportunities in Training Deep Learning Software Systems: An Analysis of Variance.* ASE 2020.
- **[Empirical]** P. Nagarajan, G. Warnell, P. Stone. *Deterministic Implementations for Reproducibility in Deep Reinforcement Learning.* 2018. — arXiv:1809.05676
- **[Context]** P. Micikevicius et al. *Mixed Precision Training.* ICLR 2018. — arXiv:1710.03740
- **[Context]** P. Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* 2017. — arXiv:1706.02677
- **[Docs]** PyTorch. *Reproducibility.* Official documentation; and NVIDIA `framework-determinism` (D. Riach), GTC 2019 talk *Determinism in Deep Learning*.

## 10. Worked Example

Data-parallel gradient all-reduce, $P = 64$, fp32, one parameter coordinate. Per-rank gradients are drawn around $10^{-3}$ with a few outliers near $10^{2}$ (a rank that saw a rare long document).

Ring order sums $63$ small values first, accumulating to $\approx 6.3\times10^{-2}$, then adds $10^{2}$. Tree order pairs the two large values early. The two totals differ in the last mantissa bit: relative gap $\approx 2^{-24} \approx 6\times10^{-8}$.

With Adam, $\eta = 3\times10^{-4}$, this becomes a parameter difference of order $10^{-11}$ — a single ULP on a weight of magnitude $10^{-2}$. So $\tau = 1$: the checkpoints differ at step 1.

Now the growth. Empirically, trajectory separation in transformer pretraining behaves chaotically with an effective doubling time of roughly 100–1000 steps in the early phase. Taking a doubling every 500 steps, $10^{-11}$ relative reaches $O(1)$ after $\log_2(10^{11}) \cdot 500 \approx 18{,}000$ steps — well inside a 50k-step run. Summers and Dinneen's result says exactly this: the endpoint is a different sample from the seed distribution, ~0.3 accuracy points away on CIFAR-10, and there is no sense in which the two runs are "the same run with rounding noise".

**Where the obstruction becomes visible.** The fix at the reduction is cheap in FLOPs — binned summation needs about $4\times$ the arithmetic of a plain add, on an operation that is bandwidth-bound anyway. The expensive part is everything the fix forbids downstream: with a fixed reduction order and fixed launch geometry, the autotuner can no longer pick the split-K matmul that is 12% faster at this shape, and NCCL can no longer switch from ring to tree when the message crosses its size threshold. Each is a few percent. Nobody has published the sum of those few percents over a real pretraining run — which is precisely the number in Section 8, and precisely why the problem sits at *solved-but-impractical* rather than *solved*.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*