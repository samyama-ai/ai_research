---
id: 32-hardware-and-kernels/optimal-sharding-search-complexity
title: "Optimal Sharding Strategy Search Complexity"
topic: 32-hardware-and-kernels
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Sharding Strategy Search Complexity

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/optimal-sharding-search-complexity` · **Status:** partially-solved

## 1. Problem Statement

**Input.** A dataflow graph $G=(V,E)$ for one training step of a neural network, a device set $D$ with an interconnect topology, per-device memory capacities, and a global batch size.

**Output.** A *sharding strategy* $s$: for every tensor, which axes are split across which device-mesh axes; for every operator, which device group executes it; plus the pipeline stage assignment, micro-batch count, recomputation policy, and optimizer-state partitioning.

**Objective.** Minimize measured per-step wall-clock time subject to every device staying under its memory capacity, with the computed gradient bit-equivalent (up to reduction order) to the unsharded reference.

Three variants, routinely conflated:

- **Theory variant.** Is finding $s^\star$ NP-hard, and for which graph classes is it in P? *Largely settled* — see §4.
- **Method variant.** Can a planner find a strategy within $\varepsilon$ of the best *measured* strategy in less time than it takes to measure a handful of candidates? *Partially solved.*
- **Measurement variant.** Given two strategies whose predicted step times differ by 3%, can you decide which is faster? *Methodologically blocked* at multi-node scale — the decision noise floor is comparable to the gaps that matter.

"Solved" would mean: a planner that, for an unseen model/cluster pair, returns a strategy whose *measured* step time is within 2% of the best measured strategy in an exhaustively-benchmarked candidate set, with search cost under one training step's worth of GPU-hours.

## 2. Formal Setting

Let $D$ be organized as a logical mesh of shape $(m_1,\dots,m_k)$ with $\prod_j m_j = |D|$. For operator $v\in V$ with output tensor of rank $r_v$, an SPMD annotation is a map $\sigma_v:\{1,\dots,r_v\}\to \{0,1,\dots,k\}$, where $0$ means replicated. The per-operator space has size $(k+1)^{r_v}$; the graph space is $\prod_v (k+1)^{r_v}$ before feasibility pruning, i.e. exponential in $|V|$.

**Cost model.** For strategy $s$, per-device stage time is modelled as
$$t_d(s) \;=\; \sum_{v\in V_d}\frac{F_v(s)}{\rho_d}\;+\;\sum_{e\in E_d}\Big(\alpha_e+\frac{B_e(s)}{\beta_e}\Big)\;-\;O_d(s),$$
with $F_v$ FLOPs assigned to $d$, $\rho_d$ achieved FLOP/s (measured, not peak), $\alpha_e$ collective latency, $B_e$ bytes moved, $\beta_e$ achieved link bandwidth, and $O_d$ the compute/communication overlap credit. Pipelined step time with $p$ stages and $\mu$ micro-batches is
$$T(s)\;=\;(\mu + p - 1)\cdot \max_{d} t_d(s) \;+\; T_{\text{bubble-tail}}(s),$$
and the constraint is $M_d(s)=M^{\text{param}}_d+M^{\text{opt}}_d+M^{\text{act}}_d(\mu,\text{recompute})\le C_d$ for all $d$.

**As measured.** $T(s)$ is the median of steps $[K{+}1,\,K{+}N]$ after $K\approx 20$ warmup steps, $N\ge 50$, on a *fixed* node allocation. $\rho_d$ and $\beta_e$ come from microbenchmarks (GEMM sweep; NCCL all-reduce/all-gather sweep at the exact message sizes the strategy emits) — not from vendor spec sheets. $M_d$ is `torch.cuda.max_memory_allocated` plus allocator fragmentation headroom, which is itself strategy-dependent.

**Assumptions, and which break.**

| Assumption | Status in practice |
|---|---|
| $t_d$ additive over ops | **Violated** — overlap $O_d$ is scheduler-dependent and not a function of $s$ alone |
| Bandwidth $\beta_e$ independent of concurrent traffic | **Violated** — congestion on shared fat-tree/rail links; measured $\beta$ drops 20–50% under concurrent collectives |
| Static shapes | **Violated** for MoE (dynamic expert routing) and variable sequence length |
| Homogeneous devices | Violated in heterogeneous or partially-degraded clusters |
| Step time is deterministic | **Violated** — run-to-run and allocation-to-allocation variance, see §6 |

## 3. State of the Art

**Theory SOTA.** Tarnawski, Phanishayee, Devanur, Mahajan, Nina Paravecino (*Efficient Algorithms for Device Placement of DNN Graph Operators*, NeurIPS 2020) give exact dynamic programs polynomial in $|V|$ for graphs of bounded treewidth with a fixed device count, and hardness for the general case. Piper (Tarnawski et al., *Piper: Multidimensional Planner for DNN Parallelization*, NeurIPS 2021) extends this to joint tensor+pipeline planning via a two-level DP. **Established:** the tractable-subclass boundary. **Not established:** any approximation ratio against *measured* time.

**Systems SOTA (established, ablated).**
- **Alpa** (Zheng et al., OSDI 2022) — hierarchical decomposition: intra-operator sharding as an integer linear program, inter-operator pipelining as a DP. Matches hand-tuned Megatron-LM on GPT and beats DeepSpeed on MoE; the decomposition itself is ablated in the paper.
- **GSPMD / GShard** (Xu et al., 2021; Lepikhin et al., ICLR 2021) — not a search: annotation propagation from a few user hints. Established that propagation *scales*; not established that it finds good strategies without expert hints.
- **FlexFlow** (Jia, Zaharia, Aiken, MLSys 2019) — MCMC over the SOAP space with a simulator; established that a simulator-guided random search beats hand-designed hybrid parallelism on 2019-era models.
- **nnScaler** (Lin et al., OSDI 2024) — constraint-guided plan generation over user-supplied primitives; the constraint interface is the contribution.

**Claimed but unablated / benchmark-only.** Reported end-to-end speedups from Galvatron (VLDB 2023), Aceso (EuroSys 2024), AMP (NeurIPS 2022) and Metis (USENIX ATC 2024) are single-cluster benchmark numbers. None reports the *oracle gap* — the ratio of the planner's chosen strategy to the best strategy in an exhaustively-measured set. Without that, a 1.3× speedup over a weak baseline is consistent with the planner being 1.5× off optimal.

## 4. What Is Known

- **Hardness.** Optimal operator placement/partitioning on general DAGs is NP-hard; the standard reductions go through balanced graph partitioning and 3-PARTITION for the pipeline-balance subproblem. Bounded-treewidth + fixed $|D|$ is polynomial (Tarnawski et al., NeurIPS 2020).
- **Decomposition works for transformers.** Alpa's two-level split reduces the space enough that ILP solves the intra-op problem per stage in seconds-to-minutes; total plan search is reported in the tens of minutes to a few hours for models up to 39B parameters on 64 GPUs.
- **Hand-tuned baselines are near-optimal for dense transformers.** Megatron-LM's $(\text{TP}\le 8 \text{ within node}, \text{PP across nodes}, \text{DP outer})$ heuristic (Shoeybi et al. 2019; Narayanan et al., SC 2021) is hard to beat by more than a few percent on dense decoder models. Automated planners' large wins are on MoE and non-uniform architectures.
- **Interleaved pipeline schedules cut bubble time** by roughly the interleaving factor at the cost of proportionally more point-to-point traffic (Narayanan et al., SC 2021, measured at 175B–1T scale on 3072 A100s).
- **Memory is the binding constraint far more often than communication.** ZeRO stage-3 / FSDP changes which strategies are feasible at all (Rajbhandari et al., SC 2020).

## 5. What Is Not Known

- **Theoretically open.** No approximation algorithm with a proven ratio for the *joint* (sharding, pipeline, recompute, micro-batch) problem under a congestion-aware communication model. All known guarantees assume a link-independent cost model, which §2 flags as violated.
- **Empirically open.** The oracle gap. Nobody has published an exhaustive measured sweep of a restricted-but-honest candidate space at $\ge 64$ GPUs and reported where each planner's pick lands in that ranking. The experiment is runnable today (§8) at a cost of a few thousand GPU-hours.
- **Methodologically blocked.** Whether two strategies differing by <3% predicted time are distinguishable. Step-time distributions are non-Gaussian (stragglers, network jitter, DVFS thermal drift) and the allocation itself is a nuisance variable — the same strategy on a different set of 64 nodes can shift several percent. No standard protocol exists for this comparison.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement at the scale where the answer matters**, compounded by **absent ground truth**.

- The candidate strategies that survive pruning are *clustered*: the top 20 typically fall within 5–8% of each other. The measurement noise floor on a multi-node run is 1–3% within an allocation and can exceed that across allocations.
- So the planner's cost model must be accurate to ~2% to rank correctly, while its two least-reliable inputs — overlap credit $O_d$ and congested bandwidth $\beta_e$ — are each uncertain by 20%+ and *anticorrelated* with the term they perturb.
- Ground truth is absent because the space is too large to enumerate ($\sim10^9$ in the worked example below) and each evaluation costs minutes of whole-cluster time. Planners are therefore evaluated against *other planners* or against a hand-tuned baseline, never against the argmin. A benchmark that reports "1.3× faster than DeepSpeed" does not measure "close to optimal" — this is an evaluation not measuring the thing it names.

## 7. Current Research (as of 2026)

- **Compiler-native partitioning.** PartIR (Alabed et al., Google DeepMind) and the XLA/Shardy line push composable, user-steerable partitioning rather than black-box search — the bet is that a small tactic language plus propagation beats global optimization. *(frontier — verify current Shardy status.)*
- **Learned and profile-in-the-loop cost models.** Replacing analytic $\alpha$-$\beta$ terms with measured collective profiles indexed by (message size, group topology, concurrency). *(frontier — verify.)*
- **Heterogeneous and elastic clusters** (Metis, ATC 2024; follow-ons) where the homogeneity assumption is dropped by construction.
- **MoE-specific planning**, where expert-parallel degree and routing imbalance make static cost models structurally wrong.
- **Long-context / sequence parallelism** (Ring Attention, Ulysses-style variants) adds a mesh axis whose cost is sequence-length dependent, enlarging the space again.

## 8. Concrete Next Experiment

**Question.** What is the oracle gap of published planners?

**Scale.** A 13B-parameter dense decoder (40 layers, $d_{\text{model}}=5120$), 64 H100s (8 nodes × 8, NVLink intra-node, 400 Gb/s IB inter-node), fixed global batch 1024, sequence length 4096.

**Candidate space (deliberately truncated so it is enumerable).** Ordered factorizations $(\text{TP},\text{PP},\text{DP})$ with product 64: 28 of them. × micro-batch $\in\{1,2,4,8\}$ × recompute $\in\{\text{none, selective, full}\}$ × ZeRO stage $\in\{0,1\}$ = **672 configs**. Drop OOM configs (expect ~300–400 to survive). Measure each: 20 warmup + 50 timed steps, 5 repeats, repeats spread across **two different node allocations** to expose allocation as a nuisance variable.

**Control arm.** Megatron-LM's documented heuristic (TP=8 intra-node, PP across nodes, DP outer, micro-batch chosen by memory fit) — a single config, measured under the identical protocol.

**Deciding number.** **Top-1 measured regret** $R = T(s_{\text{planner}})/\min_s T(s) - 1$, computed for Alpa, Galvatron, nnScaler and the Megatron heuristic, with a bootstrap CI over the 5 repeats.

- $R < 0.02$ for at least one planner, with CI excluding 0.05 → the method variant is solved for dense transformers; move the open problem to MoE and heterogeneous clusters.
- $R > 0.10$ for all planners → search is genuinely unsolved and the field's benchmark-relative reporting is masking it.
- CI width $> R$ itself → the measurement variant is confirmed blocked, and protocol standardization is the prerequisite work.

Estimated cost: ~400 configs × 70 steps × 5 repeats × ~1.5 s/step ≈ 58 GPU-hours of timed steps, ~2–4k GPU-hours with setup and compile overhead.

## 9. Key References

- **[Foundational]** Jia, Zaharia, Aiken. *Beyond Data and Model Parallelism for Deep Neural Networks.* MLSys, 2019. — arXiv:1807.05358
- **[Foundational]** Shoeybi, Patwary, Puri, LeGresley, Casper, Catanzaro. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* 2019. — arXiv:1909.08053
- **[Theory SOTA]** Tarnawski, Phanishayee, Devanur, Mahajan, Nina Paravecino. *Efficient Algorithms for Device Placement of DNN Graph Operators.* NeurIPS, 2020.
- **[Theory SOTA]** Tarnawski, Narayanan, Phanishayee. *Piper: Multidimensional Planner for DNN Parallelization.* NeurIPS, 2021.
- **[SOTA]** Zheng, Li, Zhuang, Chen, Xu, Zhuo, Gonzalez, Stoica et al. *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.* OSDI, 2022. — arXiv:2201.12023
- **[SOTA]** Lin, Han, Li, et al. *nnScaler: Constraint-Guided Parallelization Plan Generation for Deep Learning Training.* OSDI, 2024.
- **[Systems]** Xu, Lee, Chen, et al. *GSPMD: General and Scalable Parallelization for ML Computation Graphs.* 2021. — arXiv:2105.04663
- **[Systems]** Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Systems]** Narayanan, Shoeybi, Casper, LeGresley, et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC, 2021. — arXiv:2104.04473
- **[Systems]** Rajbhandari, Rasley, Ruwase, He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC, 2020. — arXiv:1910.02054
- **[Survey/adjacent]** Miao, Wang, Zhang, Cui, et al. *Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism.* VLDB, 2023.

## 10. Worked Example

Take the 13B model above but restore the assumption real planners quietly make.

**Space size with uniform layer assignment.** 28 factorizations × 4 micro-batch × 3 recompute × 2 ZeRO = 672.

**Space size without it.** Pipeline stage boundaries are a composition of 40 layers into $p$ contiguous stages: $\binom{39}{p-1}$. At $p=8$ that is $\binom{39}{7}=15{,}380{,}937$. For a GPT-3-scale 80-layer model at $p=8$ it is $\binom{79}{7}\approx 2.9\times10^9$. Uniform assignment collapses $10^7$–$10^9$ to **one** choice.

**Why the collapsed choice is wrong.** Stage 0 carries the embedding (for 13B: $50257\times5120\approx 2.6\times10^8$ params) and the last stage carries the output projection plus the softmax/cross-entropy activation of shape $(B_\mu, 4096, 50257)$ — at $B_\mu=2$ in bf16 that is $2\times4096\times50257\times2\,\text{B}\approx 0.82$ GB for one tensor, before the logit-gradient buffer. A decoder layer's parameters are $12d^2\approx 3.1\times10^8$ — so the embedding is worth ~0.8 layers of memory but ~0 FLOPs, and the loss head is worth a fraction of a layer of parameters but the single largest activation in the model.

**The consequence.** A uniform 5-layers-per-stage split leaves the last stage's peak memory ~15–25% above the middle stages'. The planner's response is to lower $\mu$ or force full recompute *globally*, which costs ~30% extra forward FLOPs everywhere to fix an imbalance local to one stage. A non-uniform split (4 layers on the last stage, 6 on a middle one) removes the constraint at ~2% throughput cost — but that strategy is **not in the search space** of any planner that assumes uniform stages.

**Where the obstruction becomes visible.** Suppose the analytic model predicts the uniform+full-recompute plan at 1.48 s/step and the non-uniform+selective plan at 1.42 s/step — a 4% gap. Measured over 5 repeats across two allocations, the observed spread on a *single* fixed strategy is ±1.5% within an allocation and ±3% across them. The two plans' confidence intervals overlap. The planner cannot justify the larger search space from measurement, and the measurement cannot adjudicate the planner. That is the loop §5 calls methodologically blocked, and §8 is the smallest experiment that breaks it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*