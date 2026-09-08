---
id: 31-distributed-training/cross-datacenter-training-equivalence
title: "Low-Bandwidth Cross-Datacenter Pretraining Equivalence"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Low-Bandwidth Cross-Datacenter Pretraining Equivalence

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/cross-datacenter-training-equivalence` · **Status:** empirically-open

## 1. Problem Statement

A single datacenter caps out on power, not silicon. Splitting a pretraining run across $M$ sites replaces a 400–3200 Gb/s intra-cluster fabric with a wide-area link of 1–100 Gb/s and 10–100 ms RTT — three to four orders of magnitude less bandwidth per step. Local-SGD-family methods (DiLoCo and descendants) communicate only every $H$ inner steps and claim to close the gap.

The question: **does a low-communication run reach the same loss and the same downstream capability as fully synchronous data parallelism, at the same token budget, at frontier scale?**

Three variants, different difficulty:

- **Measurement.** What is "the same"? A token-matched loss gap inside the seed-to-seed band is one predicate; matched downstream evals, matched scaling-law slope, and matched post-training tractability are different and stronger predicates. Which one the field means is not settled.
- **Method.** Given a bandwidth budget $B$ and $M$ sites, build a scheme whose token-matched loss gap is zero. Open past ~10B parameters.
- **Theory.** Prove (or refute) that local SGD with $H$ inner steps and an adaptive inner optimizer attains the same loss-per-token as synchronous SGD in the single-epoch, non-convex, over-parameterized regime that LLM pretraining actually occupies. Existing bounds do not cover this regime.

Solving it means: a stated $(M, H, B)$ region in which equivalence holds, a stated boundary beyond which it fails, and a mechanism for the failure.

## 2. Formal Setting

$M$ replicas (one per site), each holding parameters $\theta^{(m)}_t \in \mathbb{R}^N$. Inner loop: $H$ steps of AdamW on local data shards. Outer loop, every $H$ inner steps:

$$\Delta^{(m)}_k = \theta^{(m)}_{kH} - \bar{\theta}_{(k-1)H}, \qquad \bar{\theta}_{kH} = \mathrm{OuterOpt}\!\left(\bar{\theta}_{(k-1)H},\ \tfrac{1}{M}\sum_{m=1}^{M}\Delta^{(m)}_k\right)$$

with OuterOpt = Nesterov momentum, $\eta_{\text{outer}}\approx 0.7$, $\mu = 0.9$ (DiLoCo's setting). Setting $H=1$, $\eta_{\text{outer}}=1$, $\mu=0$ recovers synchronous data parallelism.

**Quantities, as actually measured.**

- Token budget $D$: tokens consumed, counted once globally, not per replica. Equivalence claims are void unless $D$ matches.
- Loss $L$: mean cross-entropy in nats/token on a fixed held-out shard disjoint from all $M$ shards, evaluated on $\bar\theta$ (the outer average), not on any $\theta^{(m)}$.
- Seed band $\sigma_L$: standard deviation of $L$ over $n \ge 3$ full runs differing only in seed. Everything hinges on this and it is rarely reported.
- Gap $\Delta L = L_{\text{lowbw}}(D) - L_{\text{sync}}(D)$. **Equivalence predicate:** $\Delta L \le 2\sigma_L$ *and* the mean downstream score gap over a fixed suite is within its own seed band.
- Wide-area bytes per token: $C = \dfrac{2 P_{\text{bytes}} (M-1) \, \lceil D/(H B_{\text{tok}})\rceil}{D}$ for ring all-reduce of $P_{\text{bytes}}$ bytes of parameters, $B_{\text{tok}}$ tokens per inner step. Measure it at the switch, not from the formula — quantization, retries and overlap change it.
- Utilization: MFU, and separately the *stall fraction* $\rho$ = wall-clock in outer communication not overlapped with compute.

**Assumptions, and which are violated.**

1. *IID shards across replicas.* Violated whenever sites hold different data (the usual reason to split sites).
2. *Bounded gradient dissimilarity* $\frac{1}{M}\sum_m \|\nabla F_m(\theta) - \nabla F(\theta)\|^2 \le \zeta^2$, uniformly in $\theta$. Unmeasured at scale; almost certainly grows with $H$ as replicas drift.
3. *Smoothness $L$-Lipschitz gradients.* Violated at loss spikes, which is exactly when replicas diverge.
4. *Multi-epoch / repeated data*, which most local-SGD theory assumes. Frontier pretraining is roughly single-epoch, so the variance-reduction argument that makes local SGD work does not obviously apply.
5. *Homogeneous, fault-free replicas.* Violated: cross-site runs see straggling and preemption; async variants exist precisely because of this.

## 3. State of the Art

**Established (ablated, reproduced).**

- **DiLoCo** (Douillard et al., 2023): 150M-parameter transformers on C4, $M=8$, $H=500$. Matches or beats the synchronous baseline at equal tokens while communicating ~500× less. Ablations over $H$, $M$, outer optimizer are in the paper.
- **OpenDiLoCo** (Jaghouar et al., 2024): independent reproduction of DiLoCo at 150M, then a 1.1B run across three continents at 90–95% compute utilization. This is the strongest independent replication.
- **Scaling laws for DiLoCo** (Charles et al., 2025): sweeps to 10B parameters. Finds DiLoCo's advantage over data-parallel is *not* a fixed penalty — it varies predictably with $M$ and model size, and DiLoCo tolerates a substantially larger optimal global batch size. This is the only systematic scaling study.

**Claimed but unablated, or benchmark-number-only.**

- **INTELLECT-1** (Prime Intellect, 2024): 10B parameters, ~1T tokens, 30 nodes across continents. A real artifact, but there is **no token-matched synchronous control run** — the headline is a benchmark table, not an equivalence test.
- **Streaming DiLoCo** (Douillard et al., 2025): overlapping communication plus 4-bit outer transmission, reported ~400× bandwidth reduction at ≤1B scale with no loss penalty. Compelling but at small scale.
- **DeMo** (Peng, Quesnelle, Kingma, 2024): decoupled momentum, orders-of-magnitude communication reduction, evaluated to ~1B; the claim that it *exceeds* AdamW is a benchmark number without a seed band.
- **Frontier-lab multi-datacenter runs.** Google's Gemini reports describe training across multiple datacenters; the mechanism, bandwidth, and any loss penalty are undisclosed. Treat as existence proof, not evidence for the local-SGD hypothesis specifically — a fat private WAN with synchronous sharding is a different system.

## 4. What Is Known

- Local SGD converges at the same asymptotic rate as minibatch SGD for smooth strongly-convex objectives with $H$ up to $O(T^{1/2}/M^{3/2})$ (Stich, ICLR 2019). The regime is convex and multi-epoch; LLM pretraining is neither.
- Post-local SGD (Lin et al., ICLR 2020) improves generalization over large-batch synchronous SGD on CIFAR/ImageNet — a *positive* gap, measured at ResNet scale, not at LLM scale.
- DiLoCo at 150M: $H \in [50, 500]$ all work; degradation appears as $H$ grows past ~1000, and $M > 8$ degrades at fixed token budget.
- 10B-scale sweep (Charles et al., 2025): the DiLoCo-vs-data-parallel loss difference follows a fitted scaling law rather than exploding — the single best piece of evidence for optimism, and it stops at 10B.
- Async local SGD (Liu et al., 2024) at ~150M: naive asynchrony *hurts*; a delayed-Nesterov outer optimizer with dynamic local updates recovers most of the gap. Momentum in the outer optimizer is the fragile part.
- Real WAN numbers from OpenDiLoCo's 1.1B run: all-reduce of 1.1B parameters in fp16 (~2.2 GB) intercontinentally in ~500 ms–1.5 s, amortized over $H$ steps, gives ≥90% utilization.

## 5. What Is Not Known

- **Empirically open (the main gap).** Nobody has published a token-matched, seed-banded comparison of DiLoCo-family training against synchronous data parallelism at ≥70B parameters and ≥5T tokens. Every equivalence claim is ≤10B; every ≥10B artifact lacks a control. The experiment is runnable — it costs a frontier pretraining budget twice over.
- **Empirically open.** Whether equivalence survives *non-IID* site shards, which is the realistic case, at any scale above 1B.
- **Theoretically open.** No convergence result for local SGD with an adaptive inner optimizer plus a momentum outer optimizer, in the non-convex single-epoch regime, that gives an $H$-dependent bound tight enough to predict the observed $H \approx 500$.
- **Theoretically open.** Whether the extra outer-loop noise acts as regularization (positive gap, as in post-local SGD) or as damage, and which one dominates as $N$ grows.
- **Methodologically blocked.** "Equivalent" has no agreed operational definition. Loss gaps of 0.005 nats are reported without seed bands; downstream evals at these gaps are noise-dominated. Until $\sigma_L$ is routinely reported at scale, positive and negative results are not distinguishable.

## 6. Why It Is Hard

**The obstruction is that the control arm costs as much as the treatment, at a scale where nobody runs anything twice.** A 70B/5T-token run is $\sim 2\times10^{24}$ FLOPs; the equivalence test needs it twice, plus $n\ge3$ seeds for the band, so ~6–8 full runs. No one who can afford it has an incentive to publish the negative arm.

Two secondary obstructions:

- **Confounded measurement.** Low-bandwidth methods change the *effective batch size* and the optimal learning-rate schedule simultaneously. A measured $\Delta L$ conflates the communication scheme with an under-tuned hyperparameter, and the tuning sweep at 70B is itself unaffordable. Charles et al. show the optimal batch size genuinely differs — so any comparison at matched batch size is unfair to one arm, and at matched-optimal batch size is no longer a controlled comparison.
- **Evaluation that does not measure what it names.** Downstream benchmark deltas at frontier scale have run-to-run noise comparable to the effect size, so "matches on MMLU" is not evidence of equivalence.

## 7. Current Research (as of 2026)

- **Google DeepMind** (Douillard, Szlam, Charles, Rush): DiLoCo scaling laws, Streaming DiLoCo, DiPaCo modular paths. The only group publishing systematic $N$-sweeps.
- **Prime Intellect**: OpenDiLoCo, INTELLECT-1, and INTELLECT-2 (decentralized RL, 2025) — engineering-first, artifact-driven, control arms absent by design.
- **Nous Research**: DeMo and successor optimizers; distributed runs over consumer-grade links.
- **Flower Labs / Cambridge** (Sani, Lane et al.): federated pretraining ("Photon"), the non-IID-shard case.
- **Yandex / HSE lineage** (Ryabinin, Borzunov): SWARM parallelism and pipeline-over-WAN — an orthogonal axis to local SGD.
- *(frontier — verify)* Multi-site synchronous training over dedicated fiber at Google and Microsoft/OpenAI, reportedly at 100s of Gb/s between campuses. If true, the frontier answer may be "buy bandwidth," making the low-bandwidth question academic for labs and decisive for everyone else.

## 8. Concrete Next Experiment

**Scale.** 8B parameters (Llama-3-8B architecture), 1T tokens, $M=4$ replicas. ~$4\times10^{23}$ FLOPs per arm; ~4,000 H100-days per arm. Large enough to be above the 10B-adjacent published frontier in *token budget*, small enough to run 6 arms.

**Arms.**
1. **Control:** fully synchronous data parallel, global batch 4M tokens, tuned LR. Three seeds → gives $\sigma_L$.
2. **Treatment:** DiLoCo, $H=250$, Nesterov outer, IID shards. Two seeds.
3. **Stress:** identical to (2) but with non-IID shards — each site gets a disjoint domain mixture (code / web / books / multilingual). One seed.

**The deciding number.** $\Delta L = L_{\text{treat}} - L_{\text{ctrl}}$ in nats/token on a held-out mixed shard at exactly 1T tokens, reported against $2\sigma_L$ from arm 1. Prior small-scale runs put $\sigma_L \approx 0.002$–$0.005$ nats; the prediction to falsify is $\Delta L \le 0.005$. **If $\Delta L > 0.02$ nats at 8B when it was $\approx 0$ at 1B, the gap grows with scale and the whole approach is bounded — that single number is the result.** The non-IID arm gives the second number: $\Delta L_{\text{non-IID}} - \Delta L_{\text{IID}}$, isolating shard heterogeneity from the communication scheme.

Report wide-area bytes/token measured at the switch and the stall fraction $\rho$ alongside, so the efficiency claim is auditable independently of the loss claim.

## 9. Key References

- **[Foundational]** Sebastian U. Stich. *Local SGD Converges Fast and Communicates Little.* ICLR 2019. — arXiv:1805.09767
- **[Foundational]** Tao Lin, Sebastian U. Stich, Kumar Kshitij Patel, Martin Jaggi. *Don't Use Large Mini-Batches, Use Local SGD.* ICLR 2020. — arXiv:1808.07217
- **[Foundational]** Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc'Aurelio Ranzato, Arthur Szlam, Jiajun Shen. *DiLoCo: Distributed Low-Communication Training of Language Models.* 2023. — arXiv:2311.08105
- **[SOTA]** Zachary Charles, Gabriel Teston, Lucio Dery, Keith Rush, Nova Fallen, Zachary Garrett, Arthur Douillard, Arthur Szlam. *Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo.* 2025. — arXiv:2503.09799
- **[SOTA]** Arthur Douillard et al. *Streaming DiLoCo with Overlapping Communication: Towards a Distributed Free Lunch.* 2025. — arXiv:2501.18512
- **[Systems]** Sami Jaghouar, Jack Min Ong, Johannes Hagemann. *OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training.* 2024. — arXiv:2407.07852
- **[Systems]** Prime Intellect team. *INTELLECT-1 Technical Report.* 2024. — arXiv:2412.01152
- **[Method]** Bo Liu, Rachita Chhaparia, Arthur Douillard, Satyen Kale, Andrei A. Rusu, Jiajun Shen, Arthur Szlam, Marc'Aurelio Ranzato. *Asynchronous Local-SGD Training for Language Modeling.* 2024. — arXiv:2401.09135
- **[Method]** Bowen Peng, Jeffrey Quesnelle, Diederik P. Kingma. *DeMo: Decoupled Momentum Optimization.* 2024. — arXiv:2411.19870
- **[Method]** Max Ryabinin, Tim Dettmers, Michael Diskin, Alexander Borzunov. *SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient.* ICML 2023. — arXiv:2301.11913
- **[Method]** Thijs Vogels, Sai Praneeth Karimireddy, Martin Jaggi. *PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization.* NeurIPS 2019.
- **[Survey]** Jianyu Wang, Gauri Joshi. *Cooperative SGD: A Unified Framework for the Design and Analysis of Local-Update SGD Algorithms.* JMLR 2021.

## 10. Worked Example

An 8B model, fp16 outer transmission: $P_{\text{bytes}} = 1.6\times10^{10}$. Ring all-reduce across $M=4$ sites moves $2 P_{\text{bytes}}(M-1)/M \approx 24$ GB per site per outer step.

Inner step at global batch 4M tokens across 4 sites, ~1,024 H100s total, MFU 0.45: $6ND = 6\cdot 8\times10^9 \cdot 4\times10^6 \approx 1.9\times10^{17}$ FLOPs, so $1.9\times10^{17} / (1024 \cdot 9.9\times10^{14} \cdot 0.45) \approx 0.42$ s.

- **Synchronous, $H=1$:** 24 GB per 0.42 s $\Rightarrow$ 457 Gb/s sustained per site. Not a WAN link. This is the wall.
- **DiLoCo, $H=250$:** 24 GB per 105 s $\Rightarrow$ **1.83 Gb/s**. A commodity leased line. On a 10 Gb/s link the transfer takes ~19 s, a stall fraction $\rho = 19/105 = 18\%$ — unacceptable, which is exactly why Streaming DiLoCo overlaps and quantizes: 4-bit shards drop it to ~4.6 s, $\rho \approx 4\%$, and overlap drives it toward 0.

So the bandwidth arithmetic works — 250× reduction turns an impossible link into a cheap one. **The obstruction is not here.** It is that $H=250$ lets each replica take 250 AdamW steps, ~1B tokens, before resynchronizing. At 1T total tokens that is 4,000 outer steps: the outer optimizer sees a 4,000-step trajectory, roughly the length of a *short* training run, and the replicas have each moved through a full learning-rate-schedule segment in isolation. Whether the outer average of four such trajectories lands where synchronous training would land is a claim verified at 150M and 1B and **assumed** at 8B and above. The 1.83 Gb/s number is measurable in an afternoon. The $\Delta L$ number costs 8,000 H100-days, and nobody has paid it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*