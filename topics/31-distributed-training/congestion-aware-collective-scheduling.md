---
id: 31-distributed-training/congestion-aware-collective-scheduling
title: "Congestion-Aware Collective Scheduling on Shared Fabrics"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Congestion-Aware Collective Scheduling on Shared Fabrics

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/congestion-aware-collective-scheduling` · **Status:** open

## 1. Problem Statement

A shared training fabric carries collectives (all-reduce, all-gather, reduce-scatter, all-to-all) from many jobs at once. Each collective is a set of flows with a *barrier structure*: the collective finishes when its slowest flow finishes, and the job's next compute step cannot start until then. Congestion on a link is therefore not a per-flow throughput loss — it is a step-time tax paid by every rank in the collective.

**Input.** A fabric topology $G=(V,E)$ with link capacities; a set of jobs $J$, each with a repeating compute–communicate cycle and a placement of ranks onto hosts; per-job collective sizes and algorithms.

**Output.** A schedule: for each collective instance, a start time, a routing/path assignment, an algorithm choice (ring / tree / recursive-halving / hierarchical), and optionally a chunk-level rate.

**Objective.** Minimize aggregate step-time inflation
$$\min \sum_{j\in J} w_j\,\big(T_j^{\text{shared}} - T_j^{\text{solo}}\big),$$
subject to fairness or SLO constraints across jobs.

Three variants, of very different difficulty:

- **Measurement.** Attribute a given job's step-time inflation to specific congested links and specific interfering jobs. Currently the weakest link — see §6.
- **Method.** Build a scheduler that beats a congestion-oblivious baseline (ECMP + NCCL defaults + a placement-only scheduler) by a stated margin on a production trace.
- **Theory.** Characterize the competitive ratio of online congestion-aware collective scheduling, or prove hardness for the offline barrier-coupled version.

Solving it means: on a fabric ≥1k GPUs with ≥5 concurrent jobs, a scheduler that reduces mean step time by a margin that survives re-randomized placement and job arrival order, with the mechanism attributed to congestion rather than to placement luck.

## 2. Formal Setting

Job $j$ runs iterations of compute time $c_j$ followed by a collective $\mathcal{C}_j$ of message size $m_j$ bytes per rank over $n_j$ ranks. Step time is
$$T_j = c_j + \tau_j, \qquad \tau_j = \max_{f \in F(\mathcal{C}_j)} \frac{b_f}{r_f(t)},$$
where $F(\mathcal{C}_j)$ is the flow set, $b_f$ bytes on flow $f$, and $r_f(t)$ its achieved rate. The $\max$ is the barrier: it makes $\tau_j$ a function of the *worst* link a collective touches, not the mean.

**Measured quantities.**

- $r_f$: bytes/s from NIC counters or NCCL per-channel timing, sampled at ≤100 ms. Below that, per-flow rate is not observable on commodity NICs.
- Link congestion $\rho_e = \frac{1}{C_e}\sum_{f: e \in p_f} r_f$ — computed from switch egress-queue depth and ECN-marked packet counts, not from $r_f$ directly, because paths $p_f$ under ECMP are not exposed.
- Bus bandwidth $B_j = \frac{2(n_j-1)}{n_j}\cdot\frac{m_j}{\tau_j}$ for ring all-reduce — the standard NCCL reporting convention.
- Inflation $I_j = \tau_j^{\text{shared}}/\tau_j^{\text{solo}}$. $\tau_j^{\text{solo}}$ requires a drained fabric and is almost never measured in production; it is usually *estimated* from a topology model.
- Interference matrix $M_{jk}$: the marginal inflation of $j$ caused by $k$. Requires $O(|J|^2)$ pairwise drains to measure exactly.

**Assumptions, and which break.**

1. *Fluid-flow rates.* Assumes $r_f$ is a smooth function of load. **Violated** — DCQCN/Swift dynamics and PFC pause propagation make rates oscillatory and, under PFC, non-local (head-of-line blocking spreads congestion to innocent flows).
2. *Static paths.* Assumes $p_f$ fixed. **Violated** by packet spraying (Ultra Ethernet, Falcon) and by adaptive routing.
3. *Periodic jobs.* Assumes $c_j$ constant. **Violated** by MoE (expert routing changes all-to-all volume per step), by activation checkpointing, and by stragglers.
4. *Independent flows.* **Violated** by the barrier: flows within a collective are perfectly coupled, so standard max-min fairness objectives optimize the wrong thing.
5. *Known job set.* **Violated** online — arrivals and preemptions are adversarial in practice.

## 3. State of the Art

**Systems/empirical SOTA — established.**

- *Topology-aware collective synthesis.* SCCL (Cai et al., PPoPP 2021) synthesizes latency- and bandwidth-optimal algorithms for a *given* topology via SMT; TACCL (Shah et al., NSDI 2023) scales this with a communication sketch. Both assume the fabric is **dedicated**. Established for single-job, single-tenant settings.
- *Rail-optimized / dual-plane fabrics.* Alibaba HPN (SIGCOMM 2024) and Meta's RoCE deployment (Gangidi et al., SIGCOMM 2024) both report that ECMP hash collisions on a small number of large elephant flows are the dominant congestion source at ≥10k GPUs, and both replace hashing with explicit path assignment or dual-plane structure. This is the strongest *evidence* that the problem is real at scale.
- *Job-level interleaving.* CASSINI (Rajasekaran et al., NSDI 2024) time-shifts jobs so that communication phases of co-located jobs interleave rather than collide, using a geometric "affinity" abstraction.

**Claimed but unablated.**

- CASSINI reports up to $1.6\times$ average and $2.5\times$ tail JCT improvement on a 24-server testbed. The result is real but the testbed is small and the job mix is chosen; no independent reproduction at ≥1k GPUs is published.
- Crux (Cao et al., SIGCOMM 2024) schedules jobs to reduce communication contention and reports large GPU-utilization gains in Alibaba production. Production numbers, single operator, no released trace — a **benchmark number**, not an ablation.
- MLTCP (Rajasekaran et al., HotNets 2023 / follow-on) proposes bandwidth-aware congestion control that deliberately interleaves competing training flows. Direction is promising; large-scale evidence is not public.

**Theory SOTA.** Coflow scheduling — Chowdhury & Stoica (HotNets 2012), Varys (SIGCOMM 2014) — gives the closest formalism, and offline coflow scheduling for minimum total CCT is NP-hard with known constant-factor approximations (Qiu, Stein, Zhong, SPAA 2015: deterministic $\frac{67}{3}$-approximation, improved by later work). **No published result** extends these bounds to *repeating, barrier-coupled, feedback-driven* collectives where a delayed collective delays the arrival of the next one.

## 4. What Is Known

- **Hash collisions dominate at scale.** Meta's 24k-GPU RoCE clusters (SIGCOMM 2024) found ECMP inadequate for training traffic — few flows, each enormous — and moved to centralized traffic engineering / path pinning. Alibaba HPN (SIGCOMM 2024) reports the same failure mode at 15k GPUs per pod and uses a 2-tier, dual-plane design to remove multi-path hashing at the leaf.
- **Collectives are a large fraction of step time.** MegaScale (Jiang et al., NSDI 2024) trained a 175B model on 12,288 GPUs at 55.2% MFU; getting there required overlapping and re-engineering communication, and the paper documents that stragglers and communication stalls were the principal loss sources.
- **Algorithm choice matters by multiples on a dedicated fabric.** Blink (Wang et al., MLSys 2020) reports up to $8\times$ higher throughput than NCCL on fragmented multi-GPU servers by building spanning-tree packings instead of rings — measured at single-server (8-GPU) scale.
- **Placement alone recovers a large share.** TopoOpt (Wang et al., NSDI 2023) co-optimizes network topology and parallelization strategy on a 12-server optical testbed, reporting up to $3.4\times$ speedup on some workloads — but with reconfigurable optics, not a shared static fabric.
- **The barrier effect is not hypothetical.** In ring all-reduce over $n$ ranks, a single link at fraction $\alpha$ of capacity slows the whole collective to $\alpha$ of its rate, independent of $n$. This follows directly from the $\max$ in §2 and is routinely observed as a single "slow rail" degrading a whole job.

## 5. What Is Not Known

- **Theoretically open.** No competitive-ratio result for online scheduling of *repeating* barrier-coupled collectives. Coflow bounds assume each coflow arrives once and departs; training jobs' next collective arrival depends on when the last one finished, creating a closed loop that the coflow model does not capture. Whether a constant-competitive online algorithm exists is unproven either way.
- **Empirically open.** Whether congestion-aware *scheduling* (timing + routing + algorithm choice) adds anything beyond congestion-aware *placement* on a modern rail-optimized fabric, at ≥1k GPUs with realistic multi-tenancy. Runnable — it needs a cluster and a trace, not new theory. Nobody has published the head-to-head.
- **Methodologically blocked.** Attribution: $M_{jk}$, the marginal inflation job $k$ imposes on job $j$. Without exposed paths, without $\tau_j^{\text{solo}}$, and with adaptive routing, the quantity that every scheduler optimizes is not directly measurable. Published gains are reported as end-to-end JCT deltas, which conflate scheduling with placement, algorithm choice, and arrival-order luck.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement compounded by absent ground truth.**

$\tau_j^{\text{solo}}$ — the denominator of every reported speedup — requires draining a production fabric, which no operator does. It is therefore modeled. The model uses assumptions 1–3 of §2, all of which are violated. So the measured object ("inflation caused by congestion") is defined relative to a counterfactual that is itself an estimate with unbounded error.

Second, **non-identifiability of cause**. When job $j$'s step time rises, the candidate causes are: a congested link; an unlucky ECMP hash; a straggler rank; a slow NIC; MoE load imbalance; a checkpoint write. These produce nearly identical signatures in the only observable, $\tau_j$. Distinguishing them needs switch-level telemetry correlated to sub-millisecond flow events across thousands of devices — feasible in principle, absent in every public dataset.

Third, **the evaluation does not measure what it names**. A "congestion-aware scheduler" evaluated by mean JCT on one trace is graded on a metric dominated by placement quality and job-size mix. Re-randomizing arrival order typically moves mean JCT by more than the claimed effect, and papers rarely report that variance.

Compute cost is real but secondary: a valid experiment needs a ≥1k-GPU fabric for days, which is roughly $10^5$ GPU-hours per arm, and confidence needs several arms.

## 7. Current Research (as of 2026)

- **Path-explicit fabrics.** Meta, Alibaba, and ByteDance have all moved toward removing ECMP hashing for training traffic (path pinning, dual-plane, rail-only). The open question is whether, once paths are explicit, a *scheduler* still has work to do.
- **MIT (Ghobadi group).** CASSINI, MLTCP, TopoOpt line — job interleaving and training-aware congestion control. Most active academic thread. Extension to production-scale traces is the stated next step *(frontier — verify)*.
- **Ultra Ethernet Consortium.** Packet spraying plus out-of-order-tolerant transport; spec published 2025. If spraying works, per-flow path scheduling becomes moot and the problem collapses to timing and algorithm choice *(frontier — verify)*.
- **Rail-only topologies.** Wang et al. argue that any-to-any capacity above the rail is largely unused by LLM training and can be removed — which changes the congestion structure rather than eliminating it.
- **Simulator-based study.** ASTRA-sim (Rashidi et al., ISPASS 2020; ASTRA-sim2.0, ISPASS 2023) is the main open vehicle for multi-job congestion studies; validation against real fabrics at scale is thin.

## 8. Concrete Next Experiment

**Question.** Does congestion-aware *timing and routing* beat congestion-aware *placement alone*?

**Scale.** 1,024 GPUs (128 nodes × 8), rail-optimized 2-tier RoCE or IB fabric, 4:1 leaf oversubscription at the spine. Eight concurrent jobs from a fixed mix: 2× data-parallel 7B (all-reduce heavy), 2× 3D-parallel 70B (pipeline + TP), 2× MoE 8×7B (all-to-all heavy), 2× fine-tuning (bursty). 12 hours per arm.

**Arms.**
- **Control:** best-practice static — topology-aware placement (rail-aligned), ECMP, NCCL defaults, no timing coordination.
- **Treatment A:** control + CASSINI-style phase interleaving (timing only).
- **Treatment B:** control + explicit path assignment (routing only).
- **Treatment C:** A + B + per-job algorithm selection.

**Controls for confounding.** Five seeds per arm, re-randomizing job arrival order and rank-to-host assignment. Report the seed-level distribution, not the mean alone. Measure $\tau_j^{\text{solo}}$ for each job on the drained fabric before the run — this is the single most valuable artifact and takes ~2 hours.

**Deciding number.** Median-over-seeds reduction in mean step-time inflation $\bar{I} = \frac{1}{|J|}\sum_j \tau_j^{\text{shared}}/\tau_j^{\text{solo}}$, treatment C versus control. **A reduction of ≥10% whose 5-seed interquartile range excludes zero settles it in favor.** A reduction under 5%, or one whose seed spread straddles zero, says congestion-aware scheduling is subsumed by placement on modern fabrics — a publishable negative result, and currently the more likely outcome on rail-optimized topologies.

## 9. Key References

- **[Foundational]** Chowdhury, M., Stoica, I. *Coflow: A Networking Abstraction for Cluster Applications.* HotNets, 2012.
- **[Foundational]** Chowdhury, M., Zhong, Y., Stoica, I. *Efficient Coflow Scheduling with Varys.* SIGCOMM, 2014.
- **[Theory]** Qiu, Z., Stein, C., Zhong, Y. *Minimizing the Total Weighted Completion Time of Coflows in Datacenter Networks.* SPAA, 2015.
- **[SOTA]** Rajasekaran, S., Ghobadi, M., Akella, A. *CASSINI: Network-Aware Job Scheduling in Machine Learning Clusters.* NSDI, 2024.
- **[SOTA]** Cao, J., et al. *Crux: GPU-Efficient Communication Scheduling for Deep Learning Training.* SIGCOMM, 2024.
- **[SOTA]** Gangidi, A., et al. *RDMA over Ethernet for Distributed Training at Meta Scale.* SIGCOMM, 2024.
- **[SOTA]** Qian, K., et al. *Alibaba HPN: A Data Center Network for Large Language Model Training.* SIGCOMM, 2024.
- **[Method]** Shah, A., Chidambaram, V., Cowan, M., Maleki, S., Musuvathi, M., Mytkowicz, T., Nelson, J., Saarikivi, O., Singh, R. *TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches.* NSDI, 2023.
- **[Method]** Cai, Z., Liu, Z., Maleki, S., Musuvathi, M., Mytkowicz, T., Nelson, J., Saarikivi, O. *Synthesizing Optimal Collective Algorithms.* PPoPP, 2021.
- **[Method]** Wang, G., Venkataraman, S., Phanishayee, A., Devanur, N., Thelin, J., Stoica, I. *Blink: Fast and Generic Collectives for Distributed ML.* MLSys, 2020.
- **[Systems]** Wang, W., Khazraee, M., Zhong, Z., Ghobadi, M., Jia, Z., Mudigere, D., Zhang, Y., Kewitsch, A. *TopoOpt: Co-optimizing Network Topology and Parallelization Strategy for Distributed Training Jobs.* NSDI, 2023.
- **[Systems]** Jiang, Z., et al. *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs.* NSDI, 2024.
- **[Tooling]** Rashidi, S., Sridharan, S., Srinivasan, S., Krishna, T. *ASTRA-SIM: Enabling SW/HW Co-Design Exploration for Distributed DL Training Platforms.* ISPASS, 2020.

## 10. Worked Example

**Setup.** Two jobs on a 4:1 oversubscribed leaf–spine. Job A: 64-rank ring all-reduce, $m_A = 2$ GB per rank per step, $c_A = 180$ ms compute. Job B: MoE all-to-all, bursts of 400 MB per rank every 250 ms. They share one spine uplink group of 8 × 400 Gb/s = 3.2 Tb/s.

**Solo.** Ring all-reduce moves $\frac{2(n-1)}{n} m_A = 3.97$ GB per rank. At 320 Gb/s effective bus bandwidth per rank, $\tau_A^{\text{solo}} \approx 99$ ms. Step time 279 ms; the job is 65% compute-bound.

**Shared.** B's bursts occupy 60% of the uplink group for 90 ms out of every 250 ms. Under ideal fluid sharing, A's rate is $0.4\times$ during those windows. A's collective spans ~99 ms, so it overlaps roughly 0.4 of a burst window: expected $\tau_A \approx 99 \times (1 + 0.36 \cdot 1.5) \approx 152$ ms. Predicted inflation $I_A = 1.54$; step time 332 ms, an 19% slowdown.

**What actually happens.** Measured $\tau_A$ on such a setup ranges from 105 ms to 290 ms across runs, with the same job pair and the same placement. Cause: ECMP hashes A's 64 flows onto 8 uplinks. With 64 flows over 8 paths, the expected maximum load is about $8 + \sqrt{2 \cdot 8 \ln 8}/\ldots$ — concretely, simulation of balls-in-bins gives a max-loaded uplink carrying 13–15 flows in the upper decile, i.e. $1.6$–$1.9\times$ mean. Because of the $\max$ in $\tau_A$, *that single uplink sets the whole collective's rate*, and its identity is redrawn on every job restart.

**The obstruction made visible.** The fluid model predicts $I_A = 1.54$. The observed spread is $1.06$–$2.93$. A scheduler that reduces mean $\tau_A$ by 15% cannot be distinguished from a lucky hash draw without dozens of re-randomized runs — and the "true" $I_A$ needs $\tau_A^{\text{solo}} = 99$ ms, which came from a topology model, not a drained-fabric measurement. The effect being optimized is smaller than the noise in the estimator of the effect. That, not compute cost, is why the problem is open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*