---
id: 11-inference-and-serving/latency-accuracy-pareto-frontier
title: "Latency-Accuracy Pareto Frontier Characterization"
topic: 11-inference-and-serving
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Latency-Accuracy Pareto Frontier Characterization

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/latency-accuracy-pareto-frontier` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a fixed model family, a fixed hardware budget, and a request workload, characterize the set of serving configurations that are not dominated on the two axes *latency* and *accuracy* — the Pareto frontier — and predict where a new configuration lands on it without running it.

A configuration is a joint choice over: weight/activation precision, KV-cache precision and eviction policy, batching and chunked-prefill policy, tensor/pipeline parallel degree, speculative decoding draft model and acceptance threshold, early-exit or cascade routing, and test-time compute (samples, reasoning-token budget).

Three variants, of very different difficulty:

- **Measurement variant.** Define latency and accuracy so that "configuration $A$ dominates $B$" is a stable, workload-transferable claim. This is the blocked one. Latency is a distribution over a workload, not a scalar; accuracy is a distribution over a task mixture; both move when the other is changed, and most published comparisons collapse each to a single number chosen after the fact.
- **Method variant.** Search the configuration space efficiently. Runnable today; the obstacle is cost, not definition.
- **Theory variant.** Prove structural properties of the frontier — convexity, monotonicity in a scalar "compute per token", or the existence of a scaling law $A^\star(\ell)$ giving best achievable accuracy at latency $\ell$. Open, with no serious candidate theorem.

Solved would mean: a published protocol under which two independent labs, given the same model weights and the same hardware, report frontiers that agree within stated error bars, plus a predictor whose held-out error on accuracy at a target latency is smaller than the gap between competing methods it is used to rank.

## 2. Formal Setting

Let $\mathcal{M}$ be a base model, $c \in \mathcal{C}$ a serving configuration, and $H$ a hardware+arrival-process context. A workload is a request stream $R = \{(x_i, t_i)\}_{i=1}^{N}$ with prompt $x_i$, arrival time $t_i$, drawn from $\mathcal{D}_{\text{req}}$, at offered load $\lambda$ requests/s.

**Latency, as measured.** Per request $i$ the serving system emits token timestamps $\tau_{i,1} < \dots < \tau_{i,K_i}$. Three primitives:

$$\text{TTFT}_i = \tau_{i,1} - t_i, \qquad \text{TBT}_{i,k} = \tau_{i,k+1} - \tau_{i,k}, \qquad \text{E2E}_i = \tau_{i,K_i} - t_i.$$

Any scalar summary is a functional $L = \Phi(\{\text{TTFT}_i\}, \{\text{TBT}_{i,k}\}, \lambda)$ — e.g. $\Phi = P_{99}(\text{TTFT})$, or $\Phi = \mathbb{E}[\text{E2E}]$. **The choice of $\Phi$ is part of the problem, not a detail.** $K_i$ is itself configuration-dependent: quantization, sampling temperature and reasoning budget change generation length, so $\text{E2E}$ mixes speed with verbosity.

**Accuracy, as measured.** With task mixture $\mathcal{T}$ and scorer $s$,

$$A(c) = \mathbb{E}_{(x,y)\sim\mathcal{T}}\,\mathbb{E}_{\hat y \sim p_c(\cdot\mid x)}\, s(\hat y, y).$$

$p_c$ is the *deployed* distribution: quantized weights, KV eviction, speculative acceptance, batch-dependent kernel reductions. $s$ is exact-match, pass@1, or an LLM judge; judges are not invariant to output length, which the configuration also changes.

**Frontier.** $\mathcal{P}(H,\lambda,\Phi,\mathcal{T}) = \{c : \nexists\, c' \text{ with } L(c') \le L(c),\ A(c') \ge A(c),\ \text{one strict}\}$, and the frontier function $A^\star(\ell) = \sup\{A(c) : L(c) \le \ell\}$.

**Assumptions, and which are violated.**

1. *$L$ and $A$ are separable* — accuracy measured offline at batch size 1 transfers to the served system. **Violated**: continuous batching, chunked prefill, and cache-eviction policies make outputs depend on co-scheduled traffic; floating-point reduction order varies with batch shape.
2. *Stationary arrivals.* **Violated**: real traffic is bursty, and $P_{99}$ latency is dominated by burst behavior, not steady state.
3. *$\mathcal{T}$ is fixed under $c$.* **Violated** for any configuration that changes output length — the effective task difficulty per unit of compute shifts.
4. *A scalar $\Phi$ orders user experience.* **Violated**: TTFT and TBT trade against each other directly under chunked prefill; no scalar captures both.
5. *Deterministic replay.* **Violated**: nondeterministic kernels and scheduling make $A(c)$ itself a random variable across runs, with seldom-reported variance.

## 3. State of the Art

**Systems/empirical SOTA (established).** Orca (Yu et al., OSDI 2022) introduced iteration-level continuous batching; vLLM/PagedAttention (Kwon et al., SOSP 2023) reported 2–4× throughput at matched latency over prior systems; Sarathi-Serve (Agrawal et al., OSDI 2024) made the prefill/decode interference explicit with chunked prefill and stall-free batching, reporting large capacity gains under a stated TBT SLO. These are established, reproduced in open-source form, and — importantly — they are the papers that made the *latency axis multi-dimensional* rather than resolving it.

**Measurement SOTA.** MLPerf Inference (Reddi et al., ISCA 2020) is the only widely adopted protocol that binds accuracy and latency together: results are only valid if accuracy stays within 99% or 99.9% of an FP32 reference, under percentile latency constraints per scenario. It fixes the problem by *fiat* — one reference model, a small closed task set, fixed constraints — rather than characterizing the frontier. Metron (Agrawal et al., 2024) proposed fluidity-index and related fluid-throughput metrics precisely because TTFT/TBT percentiles miss user-visible stalls; adoption is limited.

**Claimed but unablated.** Most accuracy-preservation claims for quantization, KV-cache compression, and pruning are measured offline, batch-size-1, on a handful of benchmarks, then paired with latency numbers from a different harness. The composed Pareto claim ("4-bit is free") is essentially never measured end-to-end in the serving system that produced the latency figure. Speculative decoding is the exception: Leviathan et al. (ICML 2023) and Chen et al. (2023) prove the accepted-token distribution equals the target model's, so accuracy is invariant *by construction* rather than by benchmark.

**Benchmark-number-only results.** Vendor and framework leaderboards reporting "tokens/s at X quality" are single-point measurements at unpublished load and unpublished sampling parameters; they do not establish domination.

## 4. What Is Known

- **Speculative decoding is lossless.** Distributional equality holds exactly (rejection sampling argument, Leviathan et al. 2023); measured 2–3× wall-clock speedup on T5-XXL 11B and 70B-class chat models. This is one point where the frontier is genuinely characterized: a strict vertical improvement.
- **4-bit is Pareto-optimal for bits-vs-accuracy.** Dettmers & Zettlemoyer (ICML 2023) swept 35,000+ zero-shot evaluations across 19M–176B parameters and found 4-bit weights maximize zero-shot accuracy at fixed total model bits — but the axis is *bits*, not latency, and the two are not proportional.
- **Prefill and decode contend.** Sarathi-Serve (OSDI 2024) showed decode stalls caused by long prefills are the dominant tail-latency mechanism at 7B–180B scale on A100s; chunked prefill trades a small TTFT increase for a large TBT-tail reduction. The frontier is therefore *not* one-dimensional in latency.
- **Test-time compute buys accuracy.** Snell et al. (2024) showed optimal test-time compute allocation can beat a 14× larger model on MATH subsets — moving the frontier by spending latency, and confirming that latency and accuracy are genuinely coupled through the same knob.
- **Judge scores are length-biased.** Length-controlled AlpacaEval (Dubois et al. 2024) raised Chatbot Arena Spearman correlation to ~0.98 by regressing out length, quantifying a bias that directly corrupts accuracy measurement for any configuration that changes verbosity.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no agreed definition of $\Phi$ under which the domination relation is stable. Reversing the metric — $P_{99}$ TTFT versus mean TBT versus fluidity-index — reorders published configurations. Nor is there an accepted protocol for measuring $A(c)$ *inside* the serving system at load, with run-to-run variance reported. Until both exist, "Pareto frontier" is not a measurable object.
- **Empirically open.** Nobody has published a dense sweep of the joint space (precision × KV policy × chunk size × speculative depth × parallelism) on one model family, one cluster, and one workload, with accuracy re-measured in-system. The compute is affordable — order $10^3$ GPU-hours for an 8B model — but the harness does not exist.
- **Theoretically open.** Whether $A^\star(\ell)$ is concave, or admits a power-law form $A^\star(\ell) = a - b\ell^{-\alpha}$, is unproven. No theorem forbids non-convex frontiers, and cascade/routing configurations (mixtures of two systems) suggest the achievable set should be convexified by randomization — which, if true, means most reported "frontiers" are strictly interior.

## 6. Why It Is Hard

Two named obstructions.

**Confounded measurement.** Accuracy and latency cannot be measured in the same run without one perturbing the other. Under continuous batching, a request's output depends on which requests share its batch, so accuracy at load is not a property of the request. Measure accuracy offline instead and you have measured a different distribution $p_{c}$ from the one you timed.

**Non-identifiability of the axes.** Latency is a functional of a distribution, not a number, and the ranking depends on the functional. Two configurations with crossing TTFT/TBT curves are incomparable; a scalar $\Phi$ picked to summarize them silently encodes a utility function over user experience that has never been elicited or validated. This is an evaluation that does not measure what it names: "latency" names a scalar that does not exist.

Compute cost is secondary — it makes the sweep expensive, not ill-posed.

## 7. Current Research (as of 2026)

- **SLO-aware schedulers.** Sarathi-Serve's chunked prefill is now standard in vLLM and SGLang (Zheng et al., NeurIPS 2024); the active question is scheduling to explicit per-request TTFT/TBT SLOs rather than to throughput. Microsoft Research India, UW/vLLM, Stanford/LMSYS.
- **In-system accuracy harnesses.** Efforts to re-run benchmarks through the production server rather than a batch-1 script, so quantization and cache-eviction losses are measured under real batching. *(frontier — verify)*
- **Metric design.** Fluidity-index-style metrics that score sustained smoothness rather than percentiles; not yet adopted by leaderboards.
- **Latency-aware test-time compute.** Adaptive reasoning-token budgets and cascades that move along the frontier per request rather than per deployment. *(frontier — verify)*
- **Prediction rather than search.** Fitting cost models that predict $(L, A)$ for unseen configurations from a few probes; published work covers $L$ well and $A$ almost not at all.

## 8. Concrete Next Experiment

**Question.** Does the Pareto ordering of serving configurations survive a change of latency functional and a change of accuracy harness?

**Scale.** One model, Llama-3.1-8B-Instruct, on 1× H100 80GB. Grid of 24 configurations: precision {FP16, FP8, INT4-AWQ} × chunked-prefill token budget {512, 2048} × speculative decoding {off, 1B draft, depth 5} × KV cache {FP16, FP8}. Workload: replayed arrival trace at three offered loads (0.5×, 0.8×, 0.95× of each config's measured saturation), 20k requests each, prompt/output length distribution fixed from a public trace. Cost: roughly 400–700 GPU-hours including three seeds.

**Measurement.** For every configuration, record full token timestamps, and compute accuracy *from the served outputs* on a 4k-item mixture (MMLU, GSM8K, HumanEval, plus a length-controlled judge task). Three seeds; report run-to-run standard deviation of $A$.

**Control arm.** The same 24 configurations evaluated the conventional way: accuracy offline at batch size 1 with greedy decoding, latency from an isolated throughput benchmark. This is the current default practice and is what the experiment must be compared against.

**The deciding number.** Compute the Pareto set $\mathcal{P}$ under each of four latency functionals ($P_{99}$ TTFT, $P_{99}$ TBT, mean E2E, fluidity-index) and under both accuracy harnesses — 8 frontiers. Report the **mean pairwise Jaccard similarity of the Pareto sets**, $\bar J$. If $\bar J \ge 0.8$, the frontier is a robust object and the field can standardize on any reasonable metric. If $\bar J \le 0.5$, published "Pareto-optimal" claims are metric artifacts and the problem is confirmed methodologically blocked; the deliverable then becomes the metric, not the sweep. Secondary number: the gap between in-system and offline accuracy, $|A_{\text{served}} - A_{\text{offline}}|$, versus the seed-to-seed standard deviation — if the gap exceeds 2σ, offline accuracy is invalid for frontier construction.

## 9. Key References

- **[Foundational]** Reddi, V. J., Cheng, C., Kanter, D., et al. *MLPerf Inference Benchmark.* ISCA, 2020. — arXiv:1911.02549
- **[Foundational]** Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., Chun, B.-G. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI, 2022.
- **[SOTA]** Kwon, W., Li, Z., Zhuang, S., et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[SOTA]** Agrawal, A., Kedia, N., Panwar, A., et al. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024. — arXiv:2403.02310
- **[SOTA]** Leviathan, Y., Kalman, M., Matias, Y. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- **[SOTA]** Chen, C., Borgeaud, S., Irving, G., et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. — arXiv:2302.01318
- **[SOTA]** Dettmers, T., Zettlemoyer, L. *The case for 4-bit precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[SOTA]** Pope, R., Douglas, S., Chowdhery, A., et al. *Efficiently Scaling Transformer Inference.* MLSys, 2023. — arXiv:2211.05102
- **[SOTA]** Zheng, L., Yin, L., Xie, Z., et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS, 2024.
- **[Methodology]** Agrawal, A., Kedia, N., Mohan, J., et al. *Metron: Holistic Performance Evaluation Framework for LLM Inference Systems.* 2024. (identifier omitted — verify before citing)
- **[Methodology]** Dubois, Y., Galambosi, B., Liang, P., Hashimoto, T. B. *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators.* 2024. — arXiv:2404.04475
- **[Related]** Snell, C., Lee, J., Xu, K., Kumar, A. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[Survey]** Miao, X., Oliaro, G., Zhang, Z., et al. *Towards Efficient Generative Large Language Model Serving: A Survey from Algorithm to System.* 2023. — arXiv:2312.15234

## 10. Worked Example

Two configurations of an 8B model on one H100, both plausible deployments.

- **Config A — FP16 + speculative decoding, depth 5, 1B draft.** Accuracy invariant by the rejection-sampling theorem (Leviathan et al. 2023): $\Delta A = 0$ exactly. Draft verification runs a large batch per step, so median TBT falls; but a rejected block forces a full-model step, so TBT is *bimodal*.
- **Config B — INT4-AWQ weights, FP8 KV, chunked prefill 512.** Weight-only 4-bit sits on the bits-vs-accuracy Pareto front (Dettmers & Zettlemoyer 2023), with typical reported degradation under one point on MMLU-class tasks. Smaller weights leave more KV headroom, so larger batches fit; chunking keeps prefill from stalling decode, so TBT is *tight and unimodal*.

Now score them. Take the accuracy loss for B as $\Delta A = -0.6$ points, and suppose the timestamp traces give:

| Metric | A (spec) | B (INT4) | Winner |
|---|---|---|---|
| Mean TBT | 18 ms | 22 ms | A |
| $P_{99}$ TBT | 96 ms | 31 ms | B |
| $P_{99}$ TTFT | 640 ms | 410 ms | B |
| Accuracy (MMLU) | 68.2 | 67.6 | A |

Under $\Phi = $ mean TBT, A dominates B: faster *and* more accurate, so B is off the frontier and the frontier has one point. Under $\Phi = P_{99}$ TBT, neither dominates: both are on the frontier, and the deployment decision now depends on a 0.6-point accuracy trade against a 65 ms tail. The same two measurements, the same runs, opposite conclusions — the frontier changed cardinality because of a choice of summary statistic that no paper is required to justify.

Push once more. Config B produces shorter completions on the judge task (quantization slightly reduces verbosity). Under a raw LLM judge, B loses another ~1.5 points; under the length-controlled judge (Dubois et al. 2024), most of that gap disappears. So the accuracy axis also moves under a choice that is orthogonal to serving.

The obstruction is now visible and it is not compute: three defensible choices — which latency functional, whether accuracy is measured in-system, whether the judge is length-controlled — produce three different Pareto sets from one set of runs. Section 8's $\bar J$ is exactly the number that measures how bad this is.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*