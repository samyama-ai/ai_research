---
id: 31-distributed-training/decentralized-sparse-topology-training
title: "Decentralized Training Over Sparse Topologies at Frontier Scale"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Decentralized Training Over Sparse Topologies at Frontier Scale

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/decentralized-sparse-topology-training` · **Status:** empirically-open

## 1. Problem Statement

Train a single frontier-scale language model ($\geq 10^{10}$ parameters, $\geq 10^{12}$ tokens) on $n$ compute sites connected by a **sparse, possibly time-varying communication graph** — each site exchanges parameters or updates with $O(1)$ or $O(\log n)$ peers per round, never with all of them — and match the loss-versus-tokens curve of centralized synchronous data-parallel training at lower wall-clock or lower dollar cost.

Three variants, different difficulty:

- **Theory.** Bound the excess loss of gossip-averaged SGD as a function of the graph spectral gap $1-\lambda_2(W)$, gradient heterogeneity $\zeta^2$, and local-step count $H$, in the *non-convex, non-IID, large-$H$* regime that actually applies. Mostly settled for $H=1$; open for the DiLoCo regime.
- **Method.** Choose the topology, mixing weights, sync period, and outer optimizer so that the sparse run's token-efficiency penalty is small enough that the bandwidth saving wins.
- **Measurement.** Report a comparison that isolates topology. Most published comparisons vary topology, batch size, and step budget simultaneously.

**Solved** means: a $\geq 10$B-parameter run on a graph of maximum degree $\leq \log_2 n$ reaching the same validation loss as an all-reduce control at $\leq 1.05\times$ the token budget and $< 1.0\times$ the wall-clock.

## 2. Formal Setting

$n$ workers, worker $i$ holds parameters $x_i^{(t)} \in \mathbb{R}^d$ and local distribution $\mathcal{D}_i$. Objective:

$$f(x) = \frac{1}{n}\sum_{i=1}^n f_i(x), \qquad f_i(x) = \mathbb{E}_{\xi\sim\mathcal{D}_i}[\ell(x;\xi)].$$

Communication at round $t$ is a doubly-stochastic mixing matrix $W^{(t)} \in \mathbb{R}^{n\times n}$ with $W^{(t)}_{ij}\neq 0$ only if $(i,j)\in E^{(t)}$. Gossip step: $x_i \leftarrow \sum_j W_{ij} x_j$.

**Measured quantities.**

| Symbol | Definition | How measured |
|---|---|---|
| $p$ | mixing rate: $\mathbb{E}\|XW-\bar X\|_F^2 \le (1-p)\|X-\bar X\|_F^2$ | eigendecompose $W$; for time-varying, empirically fit decay of $\|X-\bar X\|_F$ on random $X$ |
| $\zeta^2$ | heterogeneity $\frac{1}{n}\sum_i\|\nabla f_i(x)-\nabla f(x)\|^2$ | two-shard gradient estimate at a shared checkpoint, averaged over $\geq 32$ batches |
| $\sigma^2$ | within-worker gradient variance | difference of independent micro-batch gradients on the same worker |
| $C^{(t)}$ | consensus distance $\frac{1}{n}\sum_i\|x_i^{(t)}-\bar x^{(t)}\|_2^2$ | logged per outer step; requires one extra all-reduce, so instrument on a sampled schedule |
| $H$ | local steps between gossip rounds | config |
| $u$ | compute utilization $=\frac{\text{FLOPs issued}}{\text{peak FLOPs}\times\text{wall-clock}}$ | end-to-end, including stragglers and restarts |

**Assumptions, and which break.** $L$-smoothness (fine as an idealization); bounded $\zeta^2$ (**violated** — heterogeneity is not bounded uniformly and drifts as data shards are consumed in different orders); doubly-stochastic $W$ with symmetric links (**violated** under packet loss, node churn, and asymmetric WAN bandwidth — push-sum is needed); IID sampling within a worker (holds if shards are shuffled); homogeneous per-worker speed (**violated**: straggler tails dominate WAN fleets); the noise model $\sigma^2$ being constant across training (**violated** — gradient noise scale grows through training, which is exactly the regime where large $H$ becomes safer).

## 3. State of the Art

**Theory SOTA (established).** Koloskova, Loizou, Boyd, Stich, Jaggi, *A Unified Theory of Decentralized SGD with Changing Topology and Local Updates* (ICML 2020) gives, for non-convex $f$, a rate whose leading term $\sigma^2/(n\epsilon^2)$ is **topology-independent** and whose topology cost enters only in lower-order terms scaling as $p^{-1}$ and $p^{-1/2}$. Lu and De Sa (ICML 2021) give matching lower bounds for the class. Ying et al. (NeurIPS 2021) prove the one-peer exponential graph over $n=2^k$ workers achieves *exact* averaging every $\log_2 n$ steps at degree 1.

**Systems SOTA (established).** Assran et al., *Stochastic Gradient Push* (ICML 2019), trained ResNet-50/ImageNet to baseline accuracy on 256 GPUs with directed, time-varying graphs. Ryabinin et al., *SWARM Parallelism* (ICML 2023), trained a 1B model over preemptible, unreliably-connected nodes. Yuan et al. (NeurIPS 2022) scheduled pipeline+data parallelism over heterogeneous WAN links.

**Claimed but unablated.** DiLoCo (Douillard et al., 2023) and OpenDiLoCo (Jaghouar et al., 2024) report ~500$\times$ communication reduction at 150M–1.1B scale — but these are *fully connected* at the outer level (global all-reduce every $H$ steps), not sparse topologies. INTELLECT-1 (Prime Intellect, 2024) is a 10B model trained over 30 geographically distributed nodes at reported 83–96% utilization; the run has **no all-reduce control arm at the same scale**, so its token-efficiency penalty is a benchmark number, not an ablation. DeMo (Peng, Quesnelle, Kingma, 2024) reports decoupled momentum with heavy compression matching AdamW at ~1B; the topology is still all-to-all-ish and the large-scale ablation is absent.

**No published run** combines $\geq 10$B parameters, degree $\leq \log_2 n$ gossip, and a matched centralized control.

## 4. What Is Known

- **Linear speedup with $n$ despite sparsity.** Lian et al. (NeurIPS 2017) proved D-PSGD's leading term matches centralized SGD; verified at 112 GPUs on ResNet/CIFAR and ImageNet.
- **Topology cost is second-order but not negligible.** Ring on $n$ nodes has $p = \Theta(1/n^2)$; static exponential graph has $p = \Theta(1/\log n)$ at degree $\log_2 n$ (Ying et al., NeurIPS 2021, up to 64 nodes).
- **Consensus distance predicts the gap.** Kong et al., *Consensus Control for Decentralized Deep Learning* (ICML 2021): when $C^{(t)}$ is held below a critical value, decentralized matches centralized; above it, the gap opens. Measured on ResNet-20/CIFAR-10 and up to 64-worker ImageNet.
- **Large $H$ works at small scale.** DiLoCo: 8 workers, $H=500$, 150M–400M params on C4 — matches or beats the synchronous baseline in final perplexity with ~500$\times$ less communication. Streaming DiLoCo (Douillard et al., 2025) reports comparable quality at 1B with partial-parameter sync and further bandwidth reduction.
- **WAN runs are feasible.** OpenDiLoCo: 1.1B params, workers on three continents, ~90–95% reported utilization. INTELLECT-1: 10B, 1T tokens, 30 nodes.
- **Compression composes with locality.** PowerSGD-style low-rank and DeMo-style momentum decoupling both retain quality at $\leq 1$B.

## 5. What Is Not Known

- **Empirically open (primary).** Does the token-efficiency penalty of a degree-$O(\log n)$ topology grow, shrink, or stay flat as parameters scale $10^9 \to 10^{11}$? No matched-control experiment exists above ~1B. Both directions are arguable: larger models tolerate more update staleness (larger critical batch), but $d$ grows so consensus distance per byte of communication worsens.
- **Theoretically open.** No tight non-convex bound covering *sparse topology* $\times$ *large $H$* $\times$ *unbounded heterogeneity* $\times$ *momentum-based outer optimizer* jointly. Existing analyses handle at most three of the four, and none covers Nesterov outer momentum on a time-varying directed graph.
- **Theoretically open.** Whether a critical consensus distance $C^\star$ exists as a function of $(d, \text{batch}, \text{LR})$ — Kong et al. observed it, no one has derived it.
- **Methodologically blocked.** "Communication cost" has no agreed unit. Bytes-on-the-wire, $p$-normalized bytes, and critical-path seconds rank methods differently, and papers pick whichever favors them. Until one is fixed, cross-paper comparison is not meaningful.

## 6. Why It Is Hard

**The confound is structural, not sloppy.** A sparse-topology run and an all-reduce run cannot be held equal on both step count and wall-clock — that is the entire point of the change. Fixing steps hides the speedup; fixing wall-clock changes the token budget. Every published comparison silently picks one, and the two orderings can disagree.

**The decisive experiment costs a control arm at frontier scale.** Detecting a 5% token-efficiency penalty at 10B/1T tokens needs a centralized run of the same size — roughly $2\times 10^{23}$ FLOPs, about 4,000 H100-months of the budget spent purely to have something to compare against. That is why INTELLECT-1 has no control: nobody pays twice.

**Heterogeneity is unmeasurable at scale in the form the theory needs.** $\zeta^2$ is defined at a common point $x$; once workers diverge there is no common point, so what gets logged is a proxy whose relationship to the theoretical quantity is unestablished.

## 7. Current Research (as of 2026)

- **Prime Intellect** — INTELLECT-1/-2 line, protocol-level decentralized RL and pretraining over volunteer hardware *(frontier — verify current scale claims)*.
- **Google DeepMind** — DiLoCo, Streaming DiLoCo, async local SGD; direction is bandwidth reduction under a full-mesh outer step, not sparse graphs.
- **EPFL MLO (Jaggi group)** — consensus-distance theory, unified decentralized analyses.
- **Nous Research** — DeMo and successors; compression $\times$ locality composition.
- **HKUST / Alibaba (Ying, Yuan, Yan)** — exponential-graph topology design and equivalence results.
- **Yandex/HSE (Ryabinin) lineage** — fault-tolerant model-parallel swarms over unreliable links.
- Open direction *(frontier — verify)*: learned or bandwidth-aware topology selection, where $W$ is optimized against measured link latency rather than fixed a priori.

## 8. Concrete Next Experiment

**Scale.** 7B-parameter decoder-only transformer, 300B tokens (Chinchilla-ish), 64 nodes $\times$ 8 H100. Fixed data order seed, fixed LR schedule, fixed global batch 4M tokens.

**Arms** (all identical except the outer communication):
1. **Control:** synchronous all-reduce every step (single datacenter).
2. **Full-mesh DiLoCo:** global outer all-reduce, $H=100$, Nesterov outer optimizer.
3. **Sparse:** one-peer exponential graph, degree 1, same $H=100$, same outer optimizer.
4. **Sparse-ring:** ring, degree 2, $H=100$ — the low-$p$ stress arm.

**Instrumentation.** Log $C^{(t)}$ every 10 outer steps; log bytes-on-wire and critical-path seconds separately.

**The deciding number.** Tokens needed by arm 3 to reach the validation loss arm 1 reaches at 300B tokens, expressed as $R = T_3/T_1$. $R \le 1.05$ means sparse topology is free at 7B and the frontier claim becomes credible; $R \ge 1.25$ means gossip sparsity does not survive the jump from 1B and the field should spend its bandwidth budget on compression instead. Arm 4 versus arm 3 isolates whether the gap tracks $p$ as theory predicts.

Cost: about $5\times10^{22}$ FLOPs per arm, ~4 arms — roughly 2,500 H100-months total. This is the cheapest scale at which a 5% effect is separable from seed noise (run 2 seeds of arm 1 to establish that noise floor first).

## 9. Key References

- **[Foundational]** X. Lian, C. Zhang, H. Zhang, C.-J. Hsieh, W. Zhang, J. Liu. *Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent.* NeurIPS 2017. — arXiv:1705.09056
- **[Foundational]** A. Nedić, A. Ozdaglar. *Distributed Subgradient Methods for Multi-Agent Optimization.* IEEE Trans. Automatic Control, 2009.
- **[Theory SOTA]** A. Koloskova, N. Loizou, S. Boyd, S. U. Stich, M. Jaggi. *A Unified Theory of Decentralized SGD with Changing Topology and Local Updates.* ICML 2020. — arXiv:2003.10422
- **[Theory SOTA]** Y. Lu, C. De Sa. *Optimal Complexity in Decentralized Training.* ICML 2021.
- **[Theory]** B. Ying, K. Yuan, Y. Chen, H. Hu, P. Pan, W. Yin. *Exponential Graph is Provably Efficient for Decentralized Deep Training.* NeurIPS 2021.
- **[Theory/Empirical]** L. Kong, T. Lin, A. Koloskova, M. Jaggi, S. U. Stich. *Consensus Control for Decentralized Deep Learning.* ICML 2021.
- **[Systems SOTA]** M. Assran, N. Loizou, N. Ballas, M. Rabbat. *Stochastic Gradient Push for Distributed Deep Learning.* ICML 2019. — arXiv:1811.10792
- **[Systems SOTA]** M. Ryabinin, T. Dettmers, M. Diskin, A. Borzunov. *SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient.* ICML 2023.
- **[Systems]** B. Yuan et al. *Decentralized Training of Foundation Models in Heterogeneous Environments.* NeurIPS 2022.
- **[SOTA — claimed]** A. Douillard, Q. Feng, A. A. Rusu, R. Chhaparia, Y. Donchev, A. Kuncoro, M. Ranzato, A. Szlam, J. Shen. *DiLoCo: Distributed Low-Communication Training of Language Models.* 2023. — arXiv:2311.08105
- **[SOTA — claimed]** S. Jaghouar, J. M. Ong, J. Hagemann. *OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training.* 2024. — arXiv:2407.07852
- **[SOTA — claimed]** Prime Intellect team. *INTELLECT-1 Technical Report.* 2024.
- **[Survey]** T. Ben-Nun, T. Hoefler. *Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis.* ACM Computing Surveys, 2019.

## 10. Worked Example

**Setup.** 70B parameters, bf16 = 140 GB of state to exchange. 64 nodes, 8 H100 each, 1 Gbps symmetric WAN per node (= 125 MB/s). Effective compute 400 TFLOP/s per GPU (≈40% MFU) → 3.2 PFLOP/s per node. Global batch 4M tokens ⇒ 65,536 tokens per node per local step.

**Compute per local step.** $6 \times 70\times10^9 \times 65{,}536 = 2.75\times10^{16}$ FLOPs $\div\ 3.2\times10^{15}$ FLOP/s $= 8.6$ s.

**Communication per gossip round, degree 1.** Send 140 GB and receive 140 GB concurrently on a duplex link: $140\times10^9 / 125\times10^6 = 1{,}120$ s.

**Utilization versus $H$:**

| $H$ | compute (s) | comm (s) | $u$ if not overlapped |
|---|---|---|---|
| 100 | 860 | 1,120 | 43% |
| 500 | 4,300 | 1,120 | 79% |
| 1000 | 8,600 | 1,120 | 88% |

**Where the obstruction becomes visible.** Degree 1 — the sparsest possible graph — still leaves utilization at 43% for $H=100$. The fix is to raise $H$, not to sparsify further: topology is *not* the binding constraint at 70B; the sync period is. But raising $H$ to 500 means each node takes $500\times65{,}536 = 33$M tokens of independent gradient steps between contacts, and the only evidence that this preserves token efficiency comes from 400M-parameter runs. Meanwhile the one-peer exponential graph needs $\log_2 64 = 6$ gossip rounds to mix a perturbation across the fleet, so a bad update on one node is $6\times500 = 3{,}000$ local steps old before every node sees it.

So the sparse-topology question does not even bind until bandwidth per node is high enough that $H$ can be small — and at that point the all-reduce control becomes cheap enough to be worth running. The regime where sparsity is decisive is exactly the regime where nobody can afford the control arm. That is the empirical gap, in one table.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*