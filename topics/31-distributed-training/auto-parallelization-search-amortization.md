---
id: 31-distributed-training/auto-parallelization-search-amortization
title: "Automatic Parallelization Search Cost Amortization"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Automatic Parallelization Search Cost Amortization

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/auto-parallelization-search-amortization` · **Status:** open

## 1. Problem Statement

An automatic parallelization system takes a model graph, a cluster description, and a batch/memory budget, and returns an execution plan: how each operator is sharded, how the graph is staged across devices, how microbatches are pipelined, and which collectives run where. Finding a good plan costs something — solver time, profiling time, and sometimes trial runs on the real cluster. Training the model then costs a great deal more. The problem is whether the search cost can be made small relative to the run it optimizes, *without* the plan quality degrading, and whether that ratio holds as clusters and models scale.

Three variants, of different difficulty:

- **Measurement.** Define and report search cost honestly. Today "compilation time" excludes profiling, excludes the human iterations that produced the search space, and excludes failed OOM attempts. There is no agreed accounting.
- **Method.** Build a searcher whose cost amortizes: transfer across models, warm-start from prior plans, or prune with a cost model accurate enough that few real trials are needed.
- **Theory.** Characterize when the joint sharding + staging + scheduling problem admits a polynomial-time constant-factor approximation, and when a learned prior can provably reduce sample complexity.

A solution to the method variant: for a target training job of $C$ device-hours, produce a plan within $1{+}\epsilon$ of the best plan the same search space contains, at total search cost $\le \delta C$ with $\delta \le 0.01$ and $\epsilon \le 0.05$, measured end to end including profiling, on a cluster and model the searcher has not seen.

## 2. Formal Setting

Let $G=(V,E)$ be the model's operator graph, $D$ the device set with topology $T$ (bandwidth matrix $B \in \mathbb{R}^{|D|\times|D|}$, per-device memory $M_d$). A plan $\pi \in \Pi(G,D)$ assigns to each $v\in V$ a sharding $s_v$ over a device mesh, a stage index, and a microbatch schedule.

**Objective.** Per-step latency
$$t(\pi) = \mathbb{E}\big[\text{wall-clock seconds per optimizer step}\big],$$
measured as median over $\ge 50$ steps after $\ge 10$ warmup steps, with the plan feasible: $\max_d \text{peak\_mem}_d(\pi) \le M_d$. Peak memory measured by the allocator's high-water mark, not estimated.

**Search cost.** The quantity usually under-reported:
$$C_{\text{search}} = \underbrace{C_{\text{solve}}}_{\text{ILP/DP/MCMC}} + \underbrace{C_{\text{prof}}}_{\text{operator + collective profiling}} + \underbrace{C_{\text{trial}}}_{\text{real executions, incl. OOM aborts}},$$
each in **device-seconds** (wall-clock $\times$ devices held), not CPU-seconds, because solver time on a node whose GPUs sit idle still burns the allocation.

**Amortization ratio.**
$$\delta = \frac{C_{\text{search}}}{C_{\text{train}}}, \qquad C_{\text{train}} = |D|\cdot N_{\text{steps}}\cdot t(\pi).$$

**Regret.** With $\pi^\star = \arg\min_{\pi\in\Pi} t(\pi)$ over the *same* search space,
$$\epsilon(\hat\pi) = \frac{t(\hat\pi)}{t(\pi^\star)} - 1.$$
$\pi^\star$ is not computable at scale; in practice $t(\pi^\star)$ is replaced by the best plan found by exhaustive search on a small proxy, or by the best hand-tuned baseline (Megatron-LM style). Both are upper bounds on quality, so reported $\epsilon$ is a lower bound on true regret.

**Cost model.** Most searchers optimize a surrogate $\hat t(\pi)$ built from profiled per-operator times and an analytic collective model $\hat t_{\text{coll}}(m,k) = \alpha_k + m/\beta_k$ for message size $m$ on group size $k$. Surrogate fidelity is $\rho = \text{corr}(\hat t, t)$ over sampled plans — rarely reported.

**Assumptions, and where they break:**
- *Additivity*: step time decomposes into per-stage sums. Violated by compute/communication overlap and by kernel fusion changing operator times ($\pm 20\%$ is common).
- *Homogeneity*: all devices identical. Violated on shared/heterogeneous clusters and by stragglers; Metis (ATC 2024) exists precisely because of this.
- *Static graph*: violated by MoE routing, dynamic sequence lengths, and activation-checkpointing policies chosen at runtime.
- *Bandwidth as a constant*: violated under contention — concurrent collectives on the same NVLink/IB fabric interfere non-additively.

## 3. State of the Art

**Established (ablated, reproduced).**
- **Alpa** (Zheng et al., OSDI 2022) decomposes the space into intra-operator (ILP per stage) and inter-operator (DP over stages) levels. The decomposition itself is the ablated contribution: it makes the space tractable, and Alpa matches hand-written Megatron-LM plans on GPT-family models it was not specialized for.
- **FlexFlow** (Jia, Zaharia, Aiken, MLSys 2019) established that MCMC over an SOAP space plus a simulator beats expert plans on some CNN/RNN workloads, and that simulation — not real trials — is what makes search affordable.
- **Piper** (Tarnawski et al., NeurIPS 2021) gives an exact DP over multi-dimensional partitioning with pruning; the algorithmic guarantee is real, the search space is restricted.
- **GSPMD** (Xu et al., 2021) shows annotation + sharding propagation reaches near-hand-tuned throughput with *no search at all*, which is the strongest baseline any searcher must beat on amortization grounds.

**Claimed but unablated.**
- Speedup headlines (Alpa's up-to-$3.5\times$ on MoE and up-to-$9.7\times$ on Wide-ResNet vs. baselines, at $\le 64$ GPUs) are benchmark numbers against baselines chosen by the authors; no paper reports the speedup *net of* search cost on a single job.
- Transfer/warm-start claims — that a plan found for model $A$ helps model $B$ — appear as remarks, not as controlled experiments with a cold-start control arm.
- Cost-model fidelity is typically reported as an aggregate error on plans the model already ranks well, not on the tail where search decisions are made.

**Systems SOTA vs. theory SOTA diverge.** Systems SOTA is compositional decomposition plus solver engineering (Alpa, Unity OSDI 2022, nnScaler OSDI 2024, Aceso EuroSys 2024, Galvatron VLDB 2023). Theory SOTA is polynomial-time exact/approximate algorithms only for restricted graph classes and cost models (Tarnawski et al., NeurIPS 2020).

## 4. What Is Known

- The general device-placement/partitioning problem is NP-hard; polynomial algorithms exist for restricted structures (chain- or tree-like graphs, uniform bandwidth) — Tarnawski et al., NeurIPS 2020.
- Simulator-guided search is orders of magnitude cheaper than trial-based search. FlexFlow's simulator made minutes-to-hours search viable on 4–64 GPUs where RL device placement (Mirhoseini et al., ICML 2017) needed hours of search on ~80 workers per model.
- Two-level decomposition works: Alpa's ILP+DP produces plans competitive with expert Megatron-LM configurations at 64 GPUs on GPT-3-scale slices — the single most reproduced result in this area.
- Annotation-based propagation (GSPMD) reaches high MFU on TPU pods with $C_{\text{search}} \approx 0$, at the price of human-supplied annotations — i.e. the cost moved out of the machine and into the engineer, where it is not measured.
- Constraint-guided space reduction (nnScaler, OSDI 2024) and iterative bottleneck repair (Aceso, EuroSys 2024) both cut plan-finding time by large factors versus exhaustive search, at 64–128 GPU scale, by shrinking the space rather than searching it faster.
- Heterogeneity breaks homogeneous cost models badly enough to require dedicated planners (Metis, ATC 2024; AMP, NeurIPS 2022).

## 5. What Is Not Known

- **Methodologically blocked.** There is no standard for $C_{\text{search}}$. Papers report "compilation time" with inconsistent inclusion of profiling, solver warm starts, and failed runs, and in CPU-seconds vs. device-seconds interchangeably. Until $\delta$ is defined uniformly, cross-paper amortization claims are uncomparable. This is the primary blockage.
- **Empirically open.** Whether search transfers. Nobody has published: search on models $\{A_i\}$, then measure regret and search cost on held-out model $B$ and held-out cluster shape, against a cold-start control. Runnable today on 64–256 GPUs for well under 10k device-hours.
- **Empirically open.** How $\epsilon$ and $C_{\text{search}}$ scale with $|D|$. Published evaluations cluster at 8–128 devices; behavior at $10^3$–$10^4$ devices, where the plan matters most, is extrapolated.
- **Theoretically open.** No approximation guarantee for the joint sharding+staging+overlap problem under a contention-aware communication model. No sample-complexity result for warm-started search.
- **Theoretically open.** Whether the two-level decomposition (intra-op then inter-op) is lossy — no bound on the optimality gap it introduces.

## 6. Why It Is Hard

Four named obstructions.

1. **Absent ground truth.** $\pi^\star$ is unknown; the space is combinatorially large ($|\Pi|$ grows super-exponentially in $|V|$ and $|D|$). Every reported $\epsilon$ is measured against a baseline, so a searcher that is 40% off optimum and one that is 2% off are indistinguishable when both beat Megatron-LM by 10%.
2. **Confounded measurement.** Search cost and plan quality are traded against each other by hyperparameters (time limits, pruning thresholds) that papers tune per-workload. A better $\delta$ can always be bought by loosening the solver, and the resulting $\epsilon$ increase is usually not reported.
3. **Cost-model / reality gap.** The surrogate $\hat t$ is what search optimizes; the fabric's contention behavior, allocator fragmentation, and fusion effects are what determines $t$. Errors are not uniform — they concentrate on plans with heavy overlap, exactly the plans search selects. This makes low simulated regret compatible with high real regret.
4. **Compute cost of the honest experiment.** Establishing true regret needs exhaustive or near-exhaustive enumeration, which is affordable only on proxies small enough that the interesting effects (fabric contention, straggler variance) do not appear.

"Search takes minutes and training takes weeks, so who cares" is the usual dismissal. It fails because search cost is not the only cost: the search space itself is hand-designed per model family, and re-designing it for a new architecture is unbudgeted human time.

## 7. Current Research (as of 2026)

- **Space reduction over space search.** nnScaler (MSRA, OSDI 2024) and Aceso (OSDI/EuroSys line) treat the problem as constraint specification, not optimization. This is where systems effort is going.
- **Heterogeneous and elastic clusters.** Metis (ATC 2024), AMP (NeurIPS 2022), and follow-ups target clusters where the homogeneity assumption is false by construction; re-planning after a device set changes makes $C_{\text{search}}$ recurring rather than one-off, sharpening the amortization question. *(frontier — verify current systems.)*
- **Learned cost models replacing analytic ones**, borrowing from tensor-compiler autotuning (Ansor/TVM lineage) — plausible but the transfer evidence for parallelization plans specifically is thin. *(frontier — verify.)*
- **Compiler-integrated sharding** (XLA/GSPMD, PyTorch DTensor + `torch.compile`) continues to compete by making search unnecessary rather than cheap.
- **MoE and long-context** break the static-graph assumption; planners that assume fixed shapes are increasingly mismatched to frontier workloads. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Does automatic-parallelization search transfer across models, or is it paid in full every time?

**Scale.** 64 A100/H100 GPUs (8 nodes), 3 held-out models: a 6.7B dense decoder, a 7B MoE with top-2 routing, and a 3B encoder-decoder. Sequence length 4096, fixed global batch.

**Arms.**
- **Cold (control):** Alpa-style ILP+DP run from scratch per model. Record $C_{\text{solve}}$, $C_{\text{prof}}$, $C_{\text{trial}}$ in device-seconds, including OOM aborts.
- **Warm:** same searcher, initialized from plans and profiled operator/collective tables harvested from 5 *different* source models on the same cluster. No held-out-model profiling beyond what the arm actually needs (count it if used).
- **No-search floor:** GSPMD/Megatron-style hand annotation by one engineer, human time logged.
- **Quality reference:** best-of-$N$ random search over the same space with $N$ large enough to spend $10\times$ the cold arm's budget; its best $t$ serves as the stand-in $t(\pi^\star)$.

**Deciding number.** The pair $(\delta, \epsilon)$ per arm, reduced to one scalar: **the warm arm's search device-seconds as a fraction of the cold arm's, at matched $\epsilon \le 0.05$ against the best-of-$N$ reference.** Transfer is real if that fraction is $\le 0.25$ on all three held-out models. If it exceeds $0.75$, search does not amortize across models and the field should invest in space reduction, not warm-starting.

**Budget.** ~3 model-days of 64-GPU time per arm-model pair including reference search; roughly 6–9k device-hours total.

## 9. Key References

- **[Foundational]** Jia, Z., Zaharia, M., Aiken, A. *Beyond Data and Model Parallelism for Deep Neural Networks.* MLSys, 2019. — arXiv:1807.05358
- **[Foundational]** Mirhoseini, A., Pham, H., Le, Q. V., Steiner, B., Larsen, R., Zhou, Y., Kumar, N., Norouzi, M., Bengio, S., Dean, J. *Device Placement Optimization with Reinforcement Learning.* ICML, 2017. — arXiv:1706.04972
- **[SOTA]** Zheng, L., Li, Z., Zhang, H., Zhuang, Y., Chen, Z., Huang, Y., Wang, Y., Xu, Y., Zhuo, D., Xing, E. P., Gonzalez, J. E., Stoica, I. *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.* OSDI, 2022. — arXiv:2201.12023
- **[SOTA]** Unger, C., Jia, Z., Wu, W., Lin, S., et al. *Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization.* OSDI, 2022.
- **[SOTA]** Lin, Z., Miao, Y., Zhang, Q., Yang, F., et al. *nnScaler: Constraint-Guided Parallelization Plan Generation for Deep Learning Training.* OSDI, 2024.
- **[SOTA]** Liu, G., Miao, Y., Lin, Z., et al. *Aceso: Efficient Parallel DNN Training through Iterative Bottleneck Alleviation.* EuroSys, 2024.
- **[Theory]** Tarnawski, J., Phanishayee, A., Devanur, N. R., Mahajan, D., Nina Paravecino, F. *Efficient Algorithms for Device Placement of DNN Graph Operators.* NeurIPS, 2020.
- **[Theory]** Tarnawski, J., Narayanan, D., Phanishayee, A. *Piper: Multidimensional Planner for DNN Parallelization.* NeurIPS, 2021.
- **[Baseline]** Xu, Y., Lee, H., Chen, D., Hechtman, B., et al. *GSPMD: General and Scalable Parallelization for ML Computation Graphs.* 2021. — arXiv:2105.04663
- **[Related]** Miao, X., Wang, Y., Jiang, Y., Shi, C., Nie, X., Zhang, H., Cui, B. *Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism.* VLDB, 2023. — arXiv:2211.13878
- **[Heterogeneity]** Um, T., Oh, B., Kang, M., et al. *Metis: Fast Automatic Distributed Training on Heterogeneous GPUs.* USENIX ATC, 2024.
- **[Heterogeneity]** Li, D., Wang, H., Xing, E., Zhang, H. *AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness.* NeurIPS, 2022.
- **[Baseline]** Narayanan, D., Shoeybi, M., Casper, J., et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC, 2021. — arXiv:2104.04473

## 10. Worked Example

A 6.7B decoder, 64 H100s, target 100k steps, measured $t(\pi) = 0.90$ s/step under the found plan.

$$C_{\text{train}} = 64 \times 10^5 \times 0.90\,\text{s} = 5.76\times10^6\ \text{device-s} = 1600\ \text{device-hours}.$$

Search, accounted honestly:

| component | wall-clock | devices held | device-s |
|---|---|---|---|
| operator + collective profiling | 900 s | 64 | 57,600 |
| ILP + DP solve | 1,800 s | 64 (idle, allocated) | 115,200 |
| 3 trial runs, 1 OOM abort | 4 × 180 s | 64 | 46,080 |
| **total** | | | **218,880** |

$$\delta = \frac{2.19\times10^5}{5.76\times10^6} = 0.038.$$

Reported as "30 minutes of compilation on a 1600-device-hour job," this looks like $\delta \approx 0.02$ CPU-only, or effectively zero. Counted in device-seconds with profiling and the OOM abort included, it is 3.8% — nearly four times the 1% target, and it is paid again for every architecture change, sequence-length change, and cluster resize.

Now the part the number hides. Suppose the plan found is 8% slower than the space's optimum ($\epsilon = 0.08$, i.e. $t(\pi^\star) = 0.833$ s). The foregone compute is
$$64 \times 10^5 \times (0.90 - 0.833) = 4.3\times10^5\ \text{device-s},$$
**twice the entire search cost**. Spending $3\times$ more on search to cut $\epsilon$ from 0.08 to 0.02 would be strictly profitable — $\delta$ rises to 0.11, waste falls to $1.1\times10^5$ device-s, net saving $\approx 8\times10^4$ device-s.

The obstruction is that $\epsilon$ is unmeasurable here. Enumerating $\Pi$ for this graph on 64 devices is infeasible, and the only available reference is the Megatron-LM hand plan at 0.94 s/step. Against it the searcher reports a 4% win and declares success. Whether the true gap is 1% or 15% — and therefore whether the correct action is to search less or search ten times harder — the experiment as normally run cannot say.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*