---
id: 31-distributed-training/bitwise-reproducibility-elastic-training
title: "Bitwise Reproducibility Under Dynamic Rescheduling"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Bitwise Reproducibility Under Dynamic Rescheduling

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/bitwise-reproducibility-elastic-training` · **Status:** open

## 1. Problem Statement

Elastic and fault-tolerant training changes the parallel configuration mid-run: a node is preempted, data-parallel degree drops from 512 to 448, a pipeline stage is re-replicated, ZeRO shards are re-partitioned, the reduction tree is rebuilt. The optimization problem being solved is unchanged. The floating-point program is not.

**The problem:** construct a training system in which the parameter vector after step $t$ is *bit-identical* regardless of how many times, and at which steps, the job was rescheduled — at an accepted throughput cost, for realistic LLM-scale configurations.

Three variants that are routinely conflated:

- **Measurement.** Given two runs of the same job under different reschedule schedules, decide whether the divergence observed is (a) reschedule-induced, (b) baseline hardware nondeterminism (atomics, cuDNN autotune), or (c) genuine algorithmic difference. Currently no standard metric or harness reports this decomposition.
- **Method.** Build reduction, sharding, data-ordering and optimizer-state layouts whose results are invariant to worker count and placement. Engineering-hard, not obviously impossible.
- **Theory.** Bound the horizon over which *approximate* reproducibility survives: if two trajectories differ by 1 ULP at step $t_0$, how does $\|\theta_t - \theta'_t\|$ grow? SGD on non-convex losses is empirically chaotic; no useful non-vacuous bound exists.

A solution to the method variant makes the theory variant moot. That is the point of pursuing bitwise, not statistical, equality.

## 2. Formal Setting

Let $\theta_t \in \mathbb{F}^d$ be parameters in a finite floating-point set $\mathbb{F}$ (bf16 storage, fp32 master weights: $u_{\text{bf16}} = 2^{-8} \approx 3.9\times10^{-3}$, $u_{\text{fp32}} = 2^{-24} \approx 5.96\times10^{-8}$, where $u$ is unit roundoff).

A **schedule** $\sigma$ is the sequence of parallel configurations $\sigma = ((c_0, 0), (c_1, s_1), \dots)$, where $c_k = (D_k, P_k, T_k, \pi_k)$ gives data-parallel degree, pipeline depth, tensor-parallel degree and the physical placement map, and $s_k$ is the step at which reconfiguration $k$ takes effect.

The realized update is $\theta_{t+1} = \mathcal{U}_{c(t)}(\theta_t, B_t)$, where $\mathcal{U}_c$ is the *floating-point* update actually executed under configuration $c$ and $B_t$ is the global batch. Define the reproducibility predicate

$$R(T) \;=\; \mathbb{1}\Big[\forall t \le T:\; \theta_t^{(\sigma)} = \theta_t^{(\sigma')} \text{ bitwise, for all schedules } \sigma, \sigma' \text{ with the same } B_{0:T}\Big].$$

**Measured as:** SHA-256 over the canonically-ordered, denormal-flushed byte serialization of all parameter and optimizer-state shards, gathered to a single rank at step $t$. Two runs match iff hashes match. NaN payloads and $\pm 0$ must be canonicalized or the hash reports spurious mismatches.

Decompose $R$ into three invariances:

1. **Reduction invariance.** For per-rank gradient contributions $g^{(1)},\dots,g^{(D)}$, the executed sum $\hat{S}_D = \text{fl}\!\left(\sum_i g^{(i)}\right)$ must not depend on $D$ or on the ring/tree topology. Floating-point addition is non-associative, so generically $\hat S$ depends on order; the standard bound (Higham 2002) is $|\hat S - S| \le \gamma_{n-1}\sum_i |g^{(i)}|$ with $\gamma_k = ku/(1-ku)$.
2. **Batch/shape invariance.** Kernels must return identical bits for a given logical input regardless of the microbatch or shard shape they are launched with — split-K GEMM reductions and attention split sizes both violate this.
3. **Data-order invariance.** The sample assigned to global index $j$ must be a function of $j$ and the seed alone, never of $D_k$; likewise RNG for dropout must be indexed by (layer, token position, step), not by device-local counters.

Define the **divergence horizon** $T_\varepsilon(\sigma,\sigma') = \min\{t : \|\theta_t^{(\sigma)} - \theta_t^{(\sigma')}\|_2/\|\theta_t\|_2 > \varepsilon\}$, measured by running both schedules to $T$ and logging relative $L_2$ per step.

**Assumptions, and which are violated in practice:**
- *Deterministic kernels available for every op.* Violated: fused attention backward, scatter-add embedding grads, and several sparse-MoE dispatch kernels use atomics; PyTorch's `use_deterministic_algorithms(True)` raises on some of them.
- *Identical hardware across the reschedule.* Violated by design in preemptible-instance training (A100→H100 mixed pools), where tensor-core accumulate widths and default TF32 behavior differ.
- *Gradients are exactly recoverable after a failure.* Violated whenever recovery replays from a checkpoint with in-flight microbatches dropped rather than re-executed.
- *Optimizer state is layout-independent.* Violated by ZeRO-2/3: resharding changes which rank owns which slice, and fp32 moment slices are re-partitioned, altering later reduction order.

## 3. State of the Art

**Systems SOTA (established).**
- **Elastic runtimes** — TorchElastic (PyTorch Distributed Elastic), Horovod Elastic, **Varuna** (Athlur et al., EuroSys 2022), **Bamboo** (Thorpe et al., NSDI 2023), **Oobleck** (Jang et al., SOSP 2023), **Parcae** (Duan et al., NSDI 2024), **ReCycle** (Gandhi et al., SOSP 2024). All establish *statistical* recovery: training continues and final quality is comparable. None claims bitwise schedule-invariance, and none reports a hash-equality experiment.
- **Migration-based elasticity** — **Singularity** (Shukla et al., Microsoft, arXiv:2202.07848) claims transparent preemption/migration/resizing with "no impact on correctness"; the paper argues semantic equivalence of device-proxy replay but does not present bitwise hash comparisons across resize events. *Claimed, unablated.*
- **Deterministic reductions in numerics** — ReproBLAS / reproducible summation (Demmel & Nguyen, ARITH 2013; Ahrens, Demmel, Nguyen, *ACM TOMS* 2020) gives order- and processor-count-independent sums via pre-rounding into fixed-exponent bins. Established, with proofs. It has not been integrated into a production collective-communication library for LLM training.
- **Batch-invariant kernels** — Thinking Machines Lab (He et al., "Defeating Nondeterminism in LLM Inference", 2025) demonstrated batch-size-invariant matmul/attention/RMSNorm kernels giving bit-identical *inference* outputs across batch sizes. Established for inference; the training backward pass was not covered.

**Theory SOTA.** Essentially none specific to this problem. What exists is classical: non-associativity bounds (Higham), and the empirical chaos result below.

## 4. What Is Known

- **1 ULP is enough to fully decorrelate a run.** Summers & Dinneen (*Nondeterminism and Instability in Neural Network Optimization*, ICML 2021, arXiv:2103.04514) showed that changing a single weight by the smallest representable amount produces end-of-training variance statistically indistinguishable from changing all random seeds — measured on CIFAR-10 ResNets and comparable settings. This is the load-bearing result: approximate reproducibility is not a weaker version of bitwise reproducibility, it is a different thing that decays to zero.
- **Run-to-run variance from tooling nondeterminism is non-negligible.** Pham et al. (*Problems and Opportunities in Training Deep Learning Software Systems*, ASE 2020) measured identical-configuration reruns and found top-1 accuracy spreads of several percentage points on some vision tasks. Zhuang et al. (*Randomness in Neural Network Training*, MLSys 2022) attributed the bulk to nondeterministic GPU kernels and reported that enforcing determinism costs runtime in the tens of percent at single-GPU/small-multi-GPU scale.
- **Non-associativity is not a rounding curiosity at scale.** With $D = 512$ ranks and fp32 accumulation, the worst-case reduction-order discrepancy is $\gamma_{511}\sum|g^{(i)}| \approx 511 \cdot 5.96\times10^{-8} \cdot \sum|g^{(i)}| \approx 3\times10^{-5}\sum|g^{(i)}|$ — five orders of magnitude above the per-element $u$.
- **NCCL does not promise cross-topology determinism.** Same-topology, same-version allreduce is in practice repeatable; changing rank count or algorithm (ring vs. tree vs. NVLS) changes the summation order and the result. This is documented behavior, not a bug.
- **Checkpoint-based recovery is cheap enough not to be the bottleneck.** CheckFreq (Mohan et al., FAST 2021) and Gemini (Wang et al., SOSP 2023) reduced checkpoint stall to low single-digit percentages of training time. So "just checkpoint and restore identically" is affordable — but restoring the same *bytes* does not restore the same *schedule*.

## 5. What Is Not Known

- **Empirically open.** No published experiment reports parameter-hash equality across a reconfiguration event at any scale above a toy model. Nobody has measured the throughput cost of a fully schedule-invariant training stack (fixed-tree reproducible allreduce + batch-invariant kernels + index-based data order) at $\ge$ 1B parameters over $\ge$ 10k steps. The experiment is runnable today on 64 GPUs.
- **Methodologically blocked.** There is no accepted metric that separates reschedule-induced divergence from baseline nondeterminism. Papers report final loss/accuracy, which conflates the two and has a variance floor larger than the effect. Until the harness in §2 (canonical hash + per-step $L_2$ against a same-schedule control) is standard, claims of "correctness preserved" are unfalsifiable.
- **Theoretically open.** No non-vacuous bound on $T_\varepsilon$ for SGD/Adam on a transformer loss given an initial 1-ULP perturbation. Nor is it known whether schedule-invariance is achievable *without* fixing a maximum degree $D_{\max}$ — i.e. whether there is a reduction scheme invariant over all $D$ with $O(1)$ rather than $O(\log D_{\max})$ extra accumulator width.

## 6. Why It Is Hard

**Confounded measurement is the primary obstruction.** The natural end-to-end signal (final loss) has a run-to-run standard deviation, from ordinary nondeterminism, that is larger than any plausible reschedule effect. So the null hypothesis "rescheduling changed nothing" is unrejectable in the observable, and the true predicate — bit-equality — is not measured by any standard benchmark. An evaluation that does not measure what it names.

**Second: the invariance must be global, and it is only as strong as its weakest kernel.** Reduction invariance, shape invariance, RNG invariance and data-order invariance must all hold simultaneously; a single atomicAdd in one embedding-gradient kernel destroys the hash for the entire run. There is no partial credit and no way to localize a failure except by bisection over kernels.

**Third: the cost is paid on every step, the benefit appears only on reschedule.** Fixed-tree reductions forbid topology-adaptive collectives; shape-invariant GEMMs forbid split-K autotuning. This is a permanent throughput tax against an event that may never happen, which is why no production stack pays it.

## 7. Current Research (as of 2026)

- **Batch-invariant kernel libraries.** Extension of the Thinking Machines inference result to training backward passes, including attention. *(frontier — verify)*
- **torchft / semi-synchronous fault tolerance** (PyTorch team, Meta) targets availability under failures with DiLoCo-style relaxed synchronization — a direction that moves *away* from bitwise determinism, and is the main competing bet.
- **Reproducible collectives.** Bringing ReproBLAS-style binned summation into NCCL/RCCL-level allreduce. Discussed in the HPC reproducibility community; no released implementation for GPU collectives that we can verify. *(frontier — verify)*
- **Compliance-driven demand.** EU AI Act technical-documentation duties and provenance/audit requirements are cited as a motivation for reproducible training pipelines; whether regulators will demand bitwise rather than documented-procedure reproducibility is unsettled.
- **Deterministic replay for debugging.** Record-replay of collective operations to reproduce loss spikes; adjacent but not the same guarantee.

## 8. Concrete Next Experiment

**Scale.** A 1.3B-parameter decoder-only transformer, 64×H100, tensor-parallel 2, pipeline 1, data-parallel 32, global batch 1M tokens, 10,000 steps (~10B tokens). Roughly 1–2 days on 64 GPUs.

**Arms.**
1. **Control (same-schedule).** Two runs, identical configuration, no reschedule, deterministic kernels on. Measures the floor: does the stack hash-match itself?
2. **Treatment.** One run with three forced reconfigurations at steps 2,000 / 5,000 / 8,000: DP 32 → 28 → 32 → 24, with reduction-tree rebuild and optimizer resharding.
3. **Invariant arm.** Treatment, but with (a) fixed binned/reproducible allreduce over a canonical rank order padded to $D_{\max}=32$, (b) batch-invariant GEMM and attention kernels, (c) data order indexed by global sample id, (d) RNG keyed by (step, layer, position).

**Deciding number.** $t^\star$ = the first step at which the SHA-256 of the canonicalized fp32 master weights differs from the control. Success is $t^\star = \infty$ (no mismatch through step 10,000) for arm 3, together with a throughput cost $\le 10\%$ versus arm 1. Arm 2 is expected to give $t^\star = 2{,}001$. Secondary readout: relative $L_2$ divergence per step for arm 2, to fit the empirical growth rate and give the first data point on $T_\varepsilon$.

If arm 1 itself fails to hash-match, the result is still publishable and important: it means the community's determinism tooling does not deliver determinism at 64-GPU scale, and every "reproducible training" claim above that scale is unverified.

## 9. Key References

- **[Foundational]** Nicholas J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002. — non-associativity and summation error bounds.
- **[Foundational]** David Goldberg. *What Every Computer Scientist Should Know About Floating-Point Arithmetic.* ACM Computing Surveys, 1991.
- **[SOTA, theory]** Peter Ahrens, James Demmel, Hong Diep Nguyen. *Algorithms for Efficient Reproducible Floating Point Summation.* ACM Transactions on Mathematical Software, 2020.
- **[Foundational]** James Demmel, Hong Diep Nguyen. *Fast Reproducible Floating-Point Summation.* IEEE Symposium on Computer Arithmetic (ARITH), 2013.
- **[Key empirical]** Cecilia Summers, Michael J. Dinneen. *Nondeterminism and Instability in Neural Network Optimization.* ICML, 2021. — arXiv:2103.04514
- **[Key empirical]** Donglin Zhuang, Xingyao Zhang, Shuaiwen Leon Song, Sara Hooker. *Randomness in Neural Network Training: Characterizing the Impact of Tooling.* MLSys, 2022.
- **[Key empirical]** Hung Viet Pham, Shangshu Qian, Jiannan Wang, Thibaud Lutellier, Jonathan Rosenthal, Lin Tan, Yaoliang Yu, Nachiappan Nagappan. *Problems and Opportunities in Training Deep Learning Software Systems: An Analysis of Variance.* ASE, 2020.
- **[SOTA, systems]** Insu Jang, Zhenning Yang, Zhen Zhang, Xin Jin, Mosharaf Chowdhury. *Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates.* SOSP, 2023.
- **[SOTA, systems]** John Thorpe, Pengzhan Zhao, Jonathan Eyolfson, Yifan Qiao, Zhihao Jia, Minjia Zhang, Ravi Netravali, Guoqing Harry Xu. *Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs.* NSDI, 2023.
- **[SOTA, systems]** Sanjith Athlur, Nitika Saran, Muthian Sivathanu, Ramachandran Ramjee, Nipun Kwatra. *Varuna: Scalable, Low-cost Training of Massive Deep Learning Models.* EuroSys, 2022.
- **[SOTA, systems]** Swapnil Gandhi, Mark Zhao, Athinagoras Skiadopoulos, Christos Kozyrakis. *ReCycle: Resilient Training of Large DNNs using Pipeline Adaptation.* SOSP, 2024.
- **[Systems]** Dharma Shukla et al. *Singularity: Planet-Scale, Preemptive and Elastic Scheduling of AI Workloads.* Microsoft, 2022. — arXiv:2202.07848
- **[Systems]** Jayashree Mohan, Amar Phanishayee, Vijay Chidambaram. *CheckFreq: Frequent, Fine-Grained DNN Checkpointing.* USENIX FAST, 2021.
- **[Systems]** Zhuang Wang, Zhen Jia, Shuai Zheng, Zhen Zhang, Xinwei Fu, T. S. Eugene Ng, Yida Wang. *Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints.* SOSP, 2023.
- **[Adjacent SOTA]** Horace He and Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference.* Technical report / blog, 2025. — batch-invariant kernels for bit-identical inference.

## 10. Worked Example

Take one parameter's gradient, data-parallel degree 8, fp32 accumulation. Per-rank contributions:

$$g = (1.0,\; 2^{-25},\; 2^{-25},\; 2^{-25},\; 2^{-25},\; 0,\; 0,\; 0).$$

- **Ring order** (sequential, rank 0 first): $\text{fl}(1.0 + 2^{-25}) = 1.0$ — the addend falls below half an ULP of 1.0 and round-to-nearest-even discards it. Repeat four times. Result: exactly $1.0$.
- **Tree order** (pairwise): $2^{-25}+2^{-25} = 2^{-24}$, twice, then $2^{-24}+2^{-24} = 2^{-23}$, then $\text{fl}(1.0 + 2^{-23}) = 1 + 2^{-23}$.

The two results differ by 1 ULP of fp32, $\approx 1.19\times 10^{-7}$. Nothing failed. Both are correctly rounded for their own order.

Now lose two nodes. Elastic recovery rebuilds the collective for $D=6$; NCCL's algorithm selection flips from ring to tree at the smaller size. The gradient for this parameter is now the tree value. With Adam at $\eta = 3\times10^{-4}$, the parameter update differs at step $s_1$ by about $10^{-11}$ — far below any monitoring threshold, invisible in the loss curve, and *permanent*.

What makes the obstruction visible is the next part. By Summers & Dinneen (ICML 2021), a perturbation of exactly this magnitude — one ULP in one weight — is empirically sufficient to make the final model statistically indistinguishable from a run with a completely different seed. So after step $s_1$:

- the two runs are no longer the same experiment;
- the final-loss difference, once it appears, is the same size as ordinary seed noise, so no measurement of loss can attribute it to the reschedule;
- and the only signal that would have caught it — a parameter hash mismatch at step $s_1 + 1$ — is not computed by any production training stack.

The bit that decided it was discarded by a rounding rule in a sum of eight numbers. Recovering the ability to *notice* that is the whole problem; recovering the bit is the solution.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*