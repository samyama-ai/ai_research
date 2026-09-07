---
id: 31-distributed-training/automatic-parallelism-search-at-scale
title: "Automatic Parallelism Search That Beats Hand Tuning at Scale"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Automatic Parallelism Search That Beats Hand Tuning at Scale

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/automatic-parallelism-search-at-scale` · **Status:** empirically-open

## 1. Problem Statement

**Input:** a model graph $G$, a global batch size $B$, a token budget $T$, and a cluster description $C$ (device count, memory per device, intra- and inter-node bandwidth and topology).

**Output:** a *parallelization plan* $\pi$ — a device mesh, a sharding assignment for every tensor, a pipeline partition and schedule, a recomputation set, an optimizer-state sharding level, and the collective operations that reconcile them.

**Objective:** maximize sustained training throughput (tokens/s) subject to per-device memory fitting and to $\pi$ being loss-preserving (identical optimizer semantics to the reference plan).

**Decision predicate.** For a model and cluster *not in the search tool's tuning set*, does the automatically found plan beat the best plan a strong human team produces, at $\ge 1024$ accelerators, on the same hardware and software stack, on end-to-end time-to-fixed-loss?

Three variants, with different difficulty:

- **Measurement.** Can we compare an auto-plan to a hand plan without the comparison being decided by unrelated kernel and communication-library engineering? Currently the weakest link.
- **Method.** Can search produce plans at least as good as hand tuning within a cost that is small relative to the training run?
- **Theory.** Is the plan-selection problem tractable, and how far from optimal are the decompositions (intra-op ILP + inter-op DP) that every practical system uses?

## 2. Formal Setting

Let $G=(V,E)$ be the forward-backward graph with operators $v\in V$. A plan is
$$\pi = \big(\{s_v\}_{v\in V},\ P,\ \sigma,\ R,\ z\big)$$
with $s_v$ the sharding spec of $v$'s output over a mesh $M \in \mathbb{N}^{d}$ ($\prod_i M_i = N$ devices), $P$ a partition of $V$ into $S$ pipeline stages, $\sigma$ the microbatch schedule with $m$ microbatches, $R\subseteq V$ the recomputed set, and $z\in\{0,1,2,3\}$ the optimizer-state sharding level.

**Cost model, as measured.** Per-stage compute time $t^{\text{comp}}_j(\pi)$ is measured by profiling each operator at its assigned shard shape on one device (median of $\ge 20$ timed iterations after warmup, not FLOPs divided by peak). Communication time for a resharding edge $(u,v)$ is measured as $t^{\text{comm}}_{uv} = \alpha_\ell + b_{uv}(\pi)/\beta_\ell$, where $b_{uv}$ is bytes moved, and $(\alpha_\ell,\beta_\ell)$ are the measured latency and achieved bandwidth of the collective on link class $\ell$ (NVLink, intra-rack, cross-rack) at that message size — from a collective sweep, not vendor peak.

Steady-state step time under 1F1B pipelining:
$$T_{\text{step}}(\pi) \;=\; (m + S - 1)\cdot \max_{j\le S} t_j(\pi), \qquad t_j = t^{\text{comp}}_j + t^{\text{comm}}_j - t^{\text{overlap}}_j$$
and throughput $\Theta(\pi) = B/T_{\text{step}}(\pi)$. The **bubble fraction** is $(S-1)/(m+S-1)$, measured directly from a trace as idle device-time over wall device-time.

**Memory feasibility**, measured as peak allocator high-water mark:
$$\underbrace{\frac{|\theta_j|}{d_z}\,c_z}_{\text{params+states}} + \underbrace{A_j(\pi)}_{\text{live activations}} + \underbrace{F_j}_{\text{fragmentation + workspace}} \;\le\; \text{Mem}$$
with $c_z$ bytes/parameter at sharding level $z$ and $A_j$ scaling as $O\!\big(m \cdot |{\rm stage}_j| \cdot b \cdot L_{\text{seq}} \cdot h /\, M_{\text{tp}}\big)$ before recomputation.

Search problem: $\pi^\star = \arg\max_{\pi \in \Pi(G,C)} \hat{\Theta}(\pi)$ s.t. memory. $|\Pi|$ is exponential in $|V|$; Alpa's decomposition solves intra-operator sharding per stage as an integer linear program and inter-operator stage assignment by dynamic programming.

**Assumptions, and which are violated.**
1. *Cost separability* — $t_j$ decomposes into per-operator terms. Violated: kernel fusion, autotuned GEMM selection, and allocator behavior make stage cost non-additive.
2. *Static, uniform bandwidth* — violated by topology-dependent congestion, oversubscribed cross-rack fabric, and interference from other jobs.
3. *Homogeneous, non-failing devices* — violated by stragglers, thermal throttling, and mid-run failures on runs of $10^3$–$10^4$ devices.
4. *Loss invariance* — violated in practice whenever a plan changes reduction order, mixed-precision boundaries, or expert routing.
5. *Steady state dominates* — violated for MoE, where per-step token routing makes expert load stochastic.

## 3. State of the Art

**Established.**
- **Alpa** (Zheng et al., OSDI 2022) formalizes the inter-op/intra-op split, solves intra-op by ILP and inter-op by DP, and on **64 A100s** matches hand-tuned Megatron-LM on GPT and beats DeepSpeed on MoE by up to $\approx 9.7\times$. Matching, not beating, on the model the human baseline was tuned for is the honest reading.
- **GSPMD / XLA SPMD** (Xu et al., 2021) is the compiler substrate: annotation-propagated sharding, used in production for models to $\sim$1T parameters. It *propagates* sharding; it does not search over pipeline plans.
- **Megatron-LM** (Shoeybi et al., 2019; Narayanan et al., SC 2021) is the hand-tuned reference: 502 pFLOP/s on 3072 A100s for a 1T-parameter model, $\approx 52\%$ of peak. Every auto-search claim is measured against this line.
- **FlexFlow** (Jia et al., MLSys 2019) established the SOAP search space plus an execution simulator; **Unity** (Unger et al., OSDI 2022) unified parallelization with algebraic graph transformation.

**Claimed but unablated.** Post-2023 systems — **Galvatron** (VLDB 2023), **Metis** (USENIX ATC 2024, heterogeneous clusters), **Aceso** (EuroSys 2024), **nnScaler** (Lin et al., OSDI 2024, which searches beyond the standard TP/PP/DP space) — all report multi-fold speedups over baselines. These are **benchmark numbers**: mostly $\le 128$ GPUs, baselines that are library defaults rather than an expert-tuned plan for that exact model and cluster, and no ablation separating the plan's contribution from the runtime's fusion and overlap improvements.

**Theory SOTA.** The general device-placement/partition problem is NP-hard by reduction from graph partitioning; **Piper** (Tarnawski et al., NeurIPS 2021) gives an exact-DP algorithm for the pipeline+tensor space under a layered-graph assumption. No approximation-ratio bound is known for the ILP+DP decomposition against the true joint optimum.

## 4. What Is Known

- Hand-tuned 3D parallelism reaches **52% of peak** on 3072 A100s (1T params, Narayanan et al., SC 2021); MFU of 40–55% is the practitioner band at $10^3$ scale.
- ZeRO stage-3 sharding cuts per-device optimizer memory by $\approx N$ (Rajbhandari et al., SC 2020), measured to 400 GPUs; this made a whole axis of the plan space memory-feasible.
- Pipeline bubble is $(S-1)/(m+S-1)$ exactly under 1F1B; interleaved schedules reduce it by the interleaving factor at the cost of more point-to-point traffic (SC 2021, measured at 3072 GPUs).
- Alpa's search cost is minutes-to-hours of CPU ILP time for models at 64-GPU scale — negligible against a multi-week run, but ILP variables grow with $|V|\times$ mesh candidates.
- Simulator-to-hardware error for well-built cost models is single-digit percent on homogeneous 8–64 GPU setups (FlexFlow, Alpa). No published error bar exists for the same simulators at $\ge 1024$ devices across an oversubscribed fabric.

## 5. What Is Not Known

- **Empirically open (the core gap).** Nobody has published a controlled comparison at $\ge 1024$ accelerators where the *same* runtime executes (a) an expert hand plan and (b) an auto-searched plan, for a model outside the search tool's development set. Everything at that scale is hand-tuned; everything auto-searched is $\le 128$ devices. The experiment is runnable — it costs a few thousand GPU-hours — and has not been run.
- **Theoretically open.** No approximation guarantee for the intra-op-ILP $\times$ inter-op-DP decomposition versus the joint optimum; no lower bound showing the joint problem is hard to approximate for realistic transformer graphs.
- **Methodologically blocked.** "Beats hand tuning" is not well defined. The human baseline is a moving, unspecified artifact — its quality depends on how long the team tuned, which is never reported. There is no accepted protocol for fixing human effort (engineer-hours, tuning trials) as a controlled variable.
- **Empirically open.** Whether a searched plan preserves the loss curve. Throughput comparisons dominate; time-to-fixed-loss comparisons are rare.

## 6. Why It Is Hard

**Confounded measurement, principally.** Auto-search papers ship a new runtime alongside the new search. When plan $\pi_{\text{auto}}$ on runtime $A$ beats $\pi_{\text{hand}}$ on runtime $B$, the delta mixes plan quality with kernel fusion, communication overlap, and allocator behavior. Alpa's own GPT result — parity with Megatron-LM rather than a win — is the cleanest evidence that most reported gains come from unfair baselines.

**Absent ground truth.** $\pi^\star$ is unknown. There is no oracle plan to measure regret against, so "5% faster than the baseline" says nothing about the distance to optimal.

**Compute cost of the honest experiment.** Validating one plan at 1024 GPUs to steady state costs $\sim 10^3$ GPU-hours; the space has $10^{10}$+ candidates, so search must trust a simulator whose error at that scale is itself unmeasured.

**Non-stationarity.** The cluster changes under the plan — stragglers, failures, contention — so a plan optimal for the profiled cluster is not optimal for the cluster that runs it.

## 7. Current Research (as of 2026)

- **Compiler-native search**: nnScaler (MSR) and Alpa-lineage work expanding beyond the TP/PP/DP box to arbitrary operator partitioning; JAX/XLA `shard_map` plus Shardy as a substrate where search can emit constraints rather than a full plan.
- **Heterogeneous and elastic clusters**: Metis-style search over mixed device types; plan re-selection after failure. *(frontier — verify)*
- **MoE-aware planning**: expert-parallel placement under stochastic routing load, where the steady-state assumption breaks hardest.
- **Learned cost models** replacing analytic $\alpha$–$\beta$ terms, trained on collective and kernel sweeps per cluster. *(frontier — verify)*
- **Long-context plans**: sequence/context parallelism (Ring/Ulysses-style) adds an axis that hand tuning covers poorly, which is where auto-search is most likely to win first.

## 8. Concrete Next Experiment

**Scale.** 1024 H100s (128 nodes), a 70B-parameter dense transformer at 32k sequence length — a shape outside every published auto-search evaluation set.

**Arms** (all on one runtime — Megatron-Core or an equivalent, identical kernels, identical communication library, identical seed and data order):
1. **Control:** expert hand plan, tuning effort capped and *reported* at 40 engineer-hours and $\le 20$ profiling trials.
2. **Treatment:** auto-searched plan from an ILP+DP searcher with a cost model calibrated by an on-cluster collective and kernel sweep, budget $\le 200$ GPU-hours total including profiling.
3. **Reference:** random feasible plan, to establish the spread of the space.

**Deciding number.** Ratio of time-to-fixed-validation-loss (loss $=1.95$ nats, say), treatment over control, with 3 seeds. **Treatment $\le 0.90$** (a $\ge 10\%$ win) settles the method variant affirmatively. **$0.95$–$1.05$** means parity — search is a labor saver, not a performance win, which is itself a publishable result nobody has established at this scale. Secondary readouts: simulator predicted step time versus measured (reports the first $\ge 1024$-device cost-model error bar), and measured bubble fraction per arm.

## 9. Key References

- **[Foundational]** Zhihao Jia, Matei Zaharia, Alex Aiken. *Beyond Data and Model Parallelism for Deep Neural Networks.* MLSys, 2019. — arXiv:1807.05358
- **[Foundational]** Mohammad Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* 2019. — arXiv:1909.08053
- **[SOTA]** Lianmin Zheng et al. *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.* OSDI, 2022. — arXiv:2201.12023
- **[SOTA]** Colin Unger et al. *Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization.* OSDI, 2022.
- **[SOTA]** Zhiqi Lin et al. *nnScaler: Constraint-Guided Parallelization Plan Generation for Deep Learning Training.* OSDI, 2024.
- **[Established baseline]** Deepak Narayanan et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC, 2021. — arXiv:2104.04473
- **[Theory]** Jakub Tarnawski, Deepak Narayanan, Amar Phanishayee. *Piper: Multidimensional Planner for DNN Parallelization.* NeurIPS, 2021.
- **[Systems]** Yuanzhong Xu et al. *GSPMD: General and Scalable Parallelization for ML Computation Graphs.* 2021. — arXiv:2105.04663
- **[Systems]** Samyam Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC, 2020. — arXiv:1910.02054
- **[Systems]** Deepak Narayanan et al. *PipeDream: Generalized Pipeline Parallelism for DNN Training.* SOSP, 2019.
- **[Systems]** Xupeng Miao et al. *Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism.* VLDB, 2023. — arXiv:2211.13878

## 10. Worked Example

70B dense transformer, 80 layers, $h=8192$, $L_{\text{seq}}=8192$, global batch 4M tokens, 1024 H100s (80 GB), 8 GPUs/node on NVLink, nodes on 400 Gb/s InfiniBand.

**Hand plan:** $M_{\text{tp}}=8$ (inside a node), $S=8$ pipeline stages, data-parallel $=16$, $m=64$ microbatches, ZeRO-1, full activation recompute.
Bubble $=(8-1)/(64+8-1)=9.9\%$. Weights+states at bf16 with fp32 master and Adam: $\approx 16$ B/param; per stage $70\text{B}/8 = 8.75$B params, sharded over $M_{\text{tp}}=8$ and DP-16 for states: params $8.75\text{B}\times 2/8 = 2.2$ GB, states $8.75\text{B}\times14/(8\cdot16)=9.6$ GB, activations with full recompute $\approx 12$ GB, workspace $\approx 6$ GB — total $\approx 30$ GB, fits.

**Candidate auto plan:** trade pipeline depth for tensor width — $M_{\text{tp}}=4$, $S=16$, DP $=16$. Halving TP cuts per-layer all-reduce bytes-on-the-wire per device and removes one NVLink reduction per block; the simulator, using $\alpha=6\,\mu s$, $\beta=210$ GB/s measured intra-node, predicts $-7\%$ step time. But $S=16$ raises the bubble to $(15)/(64+15)=19.0\%$, a $+9\%$ penalty on the same term. Predicted net: $+1.5\%$ *worse*. Flip $m$ to 128 (halving microbatch size) and the bubble drops to $10.5\%$, predicted net $-6\%$ better — except the smaller microbatch drops GEMM efficiency, and the simulator's per-operator profile was taken at the *larger* shape.

**The obstruction, visible.** The decision between these two plans turns on a $\pm 6\%$ effect, while the cost model's own error at 1024 devices is unmeasured and its error at 64 devices is already "single-digit percent". The search cannot rank the candidates more finely than its own noise floor. And measuring both on hardware — $2\times 10^3$ GPU-hours for two plans, times a space with hundreds of near-ties — is exactly the search-by-execution the simulator existed to avoid. That circularity, not the ILP, is what keeps the problem empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*