---
id: 32-hardware-and-kernels/autotuning-cost-versus-speedup
title: "Kernel Autotuning Search Cost Versus Achieved Speedup"
topic: 32-hardware-and-kernels
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Kernel Autotuning Search Cost Versus Achieved Speedup

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/autotuning-cost-versus-speedup` · **Status:** empirically-open

## 1. Problem Statement

Autotuners (AutoTVM, Ansor, MetaSchedule, Kernel Tuner, Triton `autotune`, cuBLASLt heuristic search) find a fast implementation of a tensor operator by measuring many candidate schedules on real hardware. Every measurement costs GPU-seconds. The tuned kernel then repays that cost over its deployment lifetime.

The problem: **given a workload, a device, and a search budget, predict the achieved speedup before spending the budget — and decide whether to spend it at all.**

Three variants that are routinely conflated:

- **Measurement.** What is the correct denominator for "speedup"? Against an untuned default schedule, autotuners look 10–100×. Against a hand-written vendor library (cuBLAS, CUTLASS, cuDNN, FlashAttention), the same tuners often lose. Reported speedups are not comparable across papers because the baseline is not fixed.
- **Method.** Build a search policy with a stopping rule: halt when the expected remaining gain per additional measurement falls below the amortized value of a measurement. No mainstream autotuner ships such a rule; they run to a fixed trial count (typically 1000–20000 trials per operator).
- **Theory.** Characterize the anytime curve $g(n)$ — best speedup after $n$ measurements — as a function of search-space structure. Is it provably a power law? Is the marginal-return-based stopping rule optimal under any realistic noise model?

A solution to the measurement variant is a baseline protocol that makes cross-paper numbers commensurable. A solution to the method variant is a tuner that, at equal wall-clock, beats fixed-budget tuning on a fixed operator suite. A solution to the theory variant is a regret bound for a search over a combinatorially structured, non-metric schedule space with heteroscedastic timing noise.

## 2. Formal Setting

Let $\mathcal{S}$ be the schedule space for an operator $o$ on device $d$: the finite set of legal configurations (tile sizes, unroll factors, vectorization widths, thread-block shapes, pipeline depths, layouts). $|\mathcal{S}|$ is typically $10^7$–$10^{12}$ under the standard TVM/Ansor sketch grammars.

**Latency, as measured.** For $s \in \mathcal{S}$, one *trial* compiles $s$ and runs it $r$ times ($r \approx 3$–$10$) after $w$ warmup iterations, yielding
$$\hat{T}(s) = \operatorname{median}_{i=1..r} T_i(s), \qquad T_i(s) = T^\star(s) + \varepsilon_i,$$
with $\varepsilon_i$ from clock throttling, DVFS state, cache/TLB residency, and co-tenancy. $\varepsilon$ is **heteroscedastic** — variance grows with $T^\star$ — and **non-stationary** — the mean drifts as the die heats. Compilation is not free: $C_{\text{compile}}(s)$ is 0.2–5 s for a CUDA kernel via NVCC, and for many spaces dominates $r\hat{T}(s)$.

**Cost of a trial.**
$$c(s) = C_{\text{compile}}(s) + (w + r)\,\hat{T}(s) + C_{\text{xfer}},$$
measured in device-seconds. Total search cost after $n$ trials: $C_n = \sum_{i=1}^n c(s_i)$. Reporting $n$ instead of $C_n$ is the standard error — trials are not equal-cost, and pruned/timed-out candidates cost the timeout, not the runtime.

**Anytime speedup curve.** Against a declared baseline latency $T_{\text{base}}$,
$$g(n) = \frac{T_{\text{base}}}{\min_{i \le n} T^\star(s_i)}, \qquad g_\infty = \frac{T_{\text{base}}}{\min_{s \in \mathcal{S}} T^\star(s)}.$$

**Amortization predicate.** With $N$ inferences over the kernel's deployment life, tuning pays off iff
$$\underbrace{N \cdot \left(T_{\text{base}} - T^\star(s^{(n)})\right)}_{\text{saved device-seconds}} \;>\; C_n .$$
Define the **break-even count** $N^\ast(n) = C_n / (T_{\text{base}} - T^\star(s^{(n)}))$. This, not $g(n)$, is the decision-relevant quantity, and it is almost never reported.

**Assumptions, and which are violated.**
1. *$T^\star$ is stationary.* Violated: sustained-clock GPUs drift 5–15% between cold and thermally saturated states.
2. *Isolated measurement.* Violated under MPS/MIG, multi-tenant clusters, and any tuning run sharing a host.
3. *Per-operator latency composes into end-to-end latency.* Violated by fusion, layout-conversion costs between differently-tuned operators, and memory-bound regimes where the operator is not the bottleneck.
4. *The cost model transfers across devices.* Violated: learned cost models trained on one GPU generation degrade sharply on the next (the motivation for TenSet's multi-platform dataset).
5. *Smoothness of $T^\star$ over $\mathcal{S}$.* Violated: register-spill and shared-memory-capacity cliffs make latency discontinuous in tile size.

## 3. State of the Art

**Systems/empirical SOTA.**
- **Ansor** (Zheng et al., OSDI 2020) — hierarchical sketch generation plus evolutionary search with a learned cost model. Reported up to $3.8\times$ over AutoTVM on some networks and beating vendor libraries on selected shapes; searches run in the hours-per-network range on a single GPU. *Established:* the end-to-end latency numbers on the paper's suite. *Claimed but unablated:* how much of the gain is the sketch grammar versus the search policy, at matched device-seconds rather than matched trial count.
- **MetaSchedule / TensorIR** (Shao et al., NeurIPS 2022) — probabilistic schedule programs; the current TVM production path. Comparable quality with better engineering ergonomics; no anytime-cost analysis published.
- **Halide tree-search autoscheduler** (Adams et al., SIGGRAPH 2019) — beam search over a learned cost model trained on randomly generated pipelines; substantially reduces search cost versus pure autotuning on CPU pipelines.
- **Value-based scheduling** (Steiner et al., MLSys 2021) — learned value function over partial schedules, reported to reach comparable performance with far fewer measurements than Halide's beam search.
- **Vendor libraries + Triton.** Hand-tuned CUTLASS/cuBLAS and FlashAttention (Dao et al., NeurIPS 2022) remain the practical bar on dense attention and GEMM. Autotuner papers that beat "TensorFlow default" and not CUTLASS are benchmark numbers, not evidence of a superior search.
- **KernelBench** (Ouyang et al., 2025) — LLM-generated kernels evaluated against PyTorch eager; useful as a fixed baseline protocol, but the eager baseline is weak and its `fast_p` metric is a correctness-gated speedup rate, not an amortized cost.

**Theory SOTA.** Pure-exploration bandit theory gives budget-optimal identification for *unstructured* arms: successive halving (Karnin, Koren, Somekh, ICML 2013) and Hyperband (Li et al., JMLR 2018) with $O(\sum_i \Delta_i^{-2}\log)$-type complexity. None of these bounds apply to $\mathcal{S}$, because $|\mathcal{S}| \gg$ budget and the arms are structurally dependent. There is no published regret or sample-complexity bound for schedule search that predicts $g(n)$.

## 4. What Is Known

- **Learned cost models cut measurement count by roughly an order of magnitude** versus random or genetic search alone, at the cost of an offline training corpus. TenSet (Zheng et al., NeurIPS 2021 Datasets & Benchmarks) contains ~52 million measured program-performance records across six hardware platforms, assembled precisely because collecting measurements per-target is the bottleneck.
- **Search cost is large and mostly not reported as device-seconds.** Tuning a full network with AutoTVM/Ansor is conventionally quoted at hours on one GPU (order $10^3$–$10^4$ trials per operator, $10^2$–$10^3$ operators per network).
- **Returns are strongly diminishing.** Across published anytime plots (Ansor Fig. 10-style curves; OpenTuner, Ansel et al., PACT 2014), most of the final speedup arrives in the first 10–20% of the budget. No paper fits a functional form to this or uses it for stopping.
- **Timing noise is non-trivial at the margin.** Late in search, candidates differ by 1–3%, which is inside the run-to-run spread of a median-of-5 measurement on a thermally unsteady GPU. This is measured folklore in the tuning community rather than a published variance study.
- **Hand-written kernels still win on the hot path.** FlashAttention-class kernels achieve large end-to-end speedups on attention that no general autotuner has matched from its own schedule grammar.

## 5. What Is Not Known

- **Empirically open.** The anytime curve $g(n)$ has never been measured to convergence on a fixed operator suite across at least three GPU generations with cost in device-seconds. Runnable today; nobody has published it.
- **Empirically open.** Whether a marginal-return stopping rule beats fixed-budget tuning at matched device-seconds. The experiment is a one-week single-node run.
- **Methodologically blocked.** "Speedup" has no fixed denominator. Until a baseline protocol exists (vendor library at the same precision and shape, same clocks, same measurement harness), cross-paper speedups are not comparable, and no meta-analysis of cost-vs-gain is possible.
- **Theoretically open.** No sample-complexity or regret bound for structured schedule search under heteroscedastic, drifting noise. Whether $g_\infty - g(n)$ decays polynomially in $n$ for realistic grammars is unproven in both directions.
- **Empirically open.** Transfer: how much of a tuning budget spent on device $d$ is recoverable on device $d'$ of the next generation. TenSet enables the measurement; the decay curve is unpublished.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an undefined baseline**.

1. *The signal is smaller than the noise near the optimum.* The decision "is candidate A faster than B" at the 1% margin requires more repeats than the search budget can afford, so the tuner's own ranking of its top candidates is partly noise. This makes $g(n)$ itself a random variable with variance that grows as the search converges — the opposite of what a clean anytime curve needs.
2. *The denominator is free.* An author picks $T_{\text{base}}$. Choosing an untuned default multiplies the reported speedup by 10–100× at zero methodological cost, and reviewers cannot detect it without re-running against a vendor library.
3. *Cost is reported in the wrong unit.* Trial counts hide that a compile is 0.2–5 s while a kernel run is microseconds; two tuners at equal trials can differ 5× in device-seconds.
4. *Non-identifiability of the win's source.* A tuner is a grammar plus a cost model plus a search policy. Papers vary all three at once, so the observed gain cannot be attributed.

## 7. Current Research (as of 2026)

- **Learned and pretrained cost models with cross-hardware transfer** — the TenSet/TLP line, plus TVM's MetaSchedule; the aim is amortizing search over targets rather than reducing it per-target.
- **LLM-generated kernels** — KernelBench and successors reframe the search as program synthesis with an LLM proposer. *(frontier — verify)* Reported pass-and-speedup rates against PyTorch eager remain modest, and token cost per accepted kernel is rarely converted into device-second-equivalents, so the cost-vs-gain question is reproduced rather than solved.
- **Compiler-side reduction of the space** — Triton and Mojo-style languages shrink $\mathcal{S}$ so a small grid search suffices; the trade is expressiveness for tractability.
- **Vendor heuristic search** — cuBLASLt/oneDNN ship online heuristic selection with sub-second budgets, which is the deployed answer to the amortization question at the cheap end. Not published as an anytime-curve study.

## 8. Concrete Next Experiment

**Question.** Does a marginal-return stopping rule reach the same speedup as fixed-budget tuning at materially lower device-second cost?

**Scale.** 60 operators (GEMM, conv2d, depthwise conv, attention, layernorm — real shapes drawn from a Llama-class model and a ResNet/ViT pair), on 3 GPUs spanning two generations (e.g. A100, H100, plus one consumer card). Single tuner (TVM MetaSchedule) to hold grammar and cost model fixed. One week on one node.

**Instrumentation.** For each operator, run to $n = 20{,}000$ trials, logging per-trial wall-clock, compile time, all $r$ repeats (not just the median), SM clock, and GPU temperature. Fix $T_{\text{base}}$ = best of {cuBLAS/cuDNN/CUTLASS at the same precision and shape} — declare it, and publish the harness.

**Arms.**
- *Control:* fixed budget $n = 2000$ trials, the community default.
- *Treatment:* stop when the fitted marginal gain per device-second over a trailing window of 200 trials falls below $1/N$ for a declared deployment count $N = 10^7$ inferences.
- *Reference:* the $n = 20{,}000$ oracle curve, used offline only.

**Deciding number.** The **device-seconds to reach 99% of oracle speedup**, aggregated over the 60 operators as a geometric mean ratio $R = C^{\text{treat}}_{99}/C^{\text{ctrl}}_{99}$. $R < 0.5$ with a bootstrap CI excluding 1.0 makes the stopping rule the new default. $R \ge 0.9$ says the fixed budget is already near-optimal and the research effort belongs in the grammar, not the stopping rule.

**Secondary output.** Publish $g(n)$ per operator with error bars from the raw repeats. This is the missing dataset regardless of which arm wins.

## 9. Key References

- **[Foundational]** Tianqi Chen et al. *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* OSDI, 2018. — arXiv:1802.04799
- **[Foundational]** Tianqi Chen et al. *Learning to Optimize Tensor Programs.* NeurIPS, 2018. — arXiv:1805.08166
- **[SOTA]** Lianmin Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI, 2020. — arXiv:2006.06762
- **[SOTA]** Junru Shao et al. *Tensor Program Optimization with Probabilistic Programs.* NeurIPS, 2022. — arXiv:2205.13603
- **[SOTA]** Andrew Adams et al. *Learning to Optimize Halide with Tree Search and Random Programs.* ACM Transactions on Graphics (SIGGRAPH), 2019.
- **[Dataset]** Lianmin Zheng et al. *TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers.* NeurIPS Datasets and Benchmarks, 2021.
- **[Method]** Jason Ansel et al. *OpenTuner: An Extensible Framework for Program Autotuning.* PACT, 2014.
- **[Method]** Benoit Steiner, Chris Cummins, Horace He, Hugh Leather. *Value Learning for Throughput Optimization of Deep Learning Workloads.* MLSys, 2021.
- **[Theory]** Zohar Karnin, Tomer Koren, Oren Somekh. *Almost Optimal Exploration in Multi-Armed Bandits.* ICML, 2013.
- **[Theory]** Lisha Li et al. *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.* JMLR 18(185), 2018. — arXiv:1603.06560
- **[Baseline]** Tri Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135
- **[Benchmark]** Anne Ouyang et al. *KernelBench: Can LLMs Write Efficient GPU Kernels?* 2025. — arXiv:2502.10517
- **[Language]** Philippe Tillet, H. T. Kung, David Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL, 2019.
- **[Model]** Samuel Williams, Andrew Waterman, David Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4), 2009.

## 10. Worked Example

One GEMM: $M=4096$, $N=4096$, $K=4096$, fp16 accumulate-fp32, on an A100-80GB (312 TFLOP/s peak fp16 tensor core). Work is $2MNK = 1.37\times10^{11}$ FLOP.

| Implementation | Latency | TFLOP/s | % of peak |
|---|---|---|---|
| Untuned TVM default schedule | 12.0 ms | 11.4 | 3.7% |
| MetaSchedule, 2000 trials | 0.72 ms | 191 | 61% |
| cuBLAS | 0.62 ms | 222 | 71% |

**Two speedups, same tuner.** Against the default schedule: $16.7\times$. Against cuBLAS: $0.86\times$ — the tuner *loses*. Both are honest numbers. Only the first appears in a headline. That gap is the measurement obstruction, made concrete.

**Cost.** 2000 trials, mean compile 1.4 s, mean run cost 5 ms $\times$ 8 repeats:
$$C_{2000} \approx 2000 \times (1.4 + 0.04)\ \text{s} \approx 2880\ \text{s} \approx 48\ \text{GPU-minutes}.$$
Note that 97% of the cost is compilation, not measurement — so a policy that halves *trials* barely halves cost, while one that caches compilations changes the picture entirely. Trial count is the wrong unit.

**Break-even against the real baseline.** Since the tuned kernel is 0.10 ms *slower* than cuBLAS, $T_{\text{base}} - T^\star < 0$ and $N^\ast$ is undefined: **no deployment volume repays this search.** Against the default schedule, $N^\ast = 2880\,\text{s} / (11.28\times10^{-3}\,\text{s}) \approx 2.6\times10^{5}$ calls — trivially repaid. Same search, same numbers, opposite conclusions, decided entirely by an unstandardized choice of denominator.

**Where the gain lives.** Logging best-so-far latency: 1.9 ms at $n=100$, 0.85 ms at $n=400$, 0.75 ms at $n=1000$, 0.72 ms at $n=2000$. The last 1000 trials — 24 GPU-minutes, half the budget — bought 4%. And 4% is within about 2× of the run-to-run spread of a median-of-8 timing on a thermally drifting A100, so a large part of that final increment may not be a real improvement at all. That is the obstruction: the region where a stopping rule must decide is exactly the region where the measurement cannot resolve the difference.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*