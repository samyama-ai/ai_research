---
id: 31-distributed-training/heterogeneous-accelerator-training
title: "Heterogeneous-Accelerator Training Without Throughput Collapse"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Heterogeneous-Accelerator Training Without Throughput Collapse

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/heterogeneous-accelerator-training` · **Status:** open

## 1. Problem Statement

**Input.** A pool of accelerators that differ in FLOP rate, memory capacity, and interconnect — e.g. a mix of H100 NVL nodes, A100 40 GB nodes, and consumer 4090s spread over three datacenters — plus a model, a dataset, and a target loss.

**Output.** A training plan: device grouping, parallelism assignment (data / tensor / pipeline / expert), microbatch and stage sizes, communication schedule, and optimizer/synchronization rule.

**Objective.** Minimize wall-clock time to reach a fixed validation loss, subject to per-device memory limits and a fixed pool.

**Decision predicate.** Solved if, for pools with a FLOP-rate spread of $\ge 4\times$, a plan achieves $\eta \ge 0.8$ where $\eta$ is measured against the FLOP-weighted ideal (Section 2) **and** reaches the same target loss in the same number of tokens as a homogeneous run of equal aggregate FLOPs.

Three variants, routinely conflated:

- **Measurement.** What is the correct denominator for "efficiency" on a heterogeneous pool? There is no agreed ideal-throughput reference, so published speedups are mostly against a weak baseline.
- **Method.** Find the plan. This is a combinatorial scheduling problem over a search space with a stochastic, non-stationary objective.
- **Theory.** Does relaxing synchrony to absorb heterogeneity cost sample efficiency, and by how much as a function of the speed spread? Convergence rates for asynchronous and local-update SGD exist, but not tight ones tied to a device-heterogeneity parameter at LLM scale.

## 2. Formal Setting

Devices $\mathcal{D}=\{1,\dots,N\}$. For device $i$:

- $\rho_i$ — **sustained** throughput in model FLOP/s, measured by running the target model's forward+backward on that device alone at the largest microbatch that fits, over $\ge 200$ steps, taking the median. Not vendor peak FLOPs.
- $M_i$ — free HBM in bytes after CUDA context and allocator fragmentation, measured at steady state.
- $B_{ij}, L_{ij}$ — achieved pairwise bandwidth (bytes/s) and RTT, measured with the collective actually used, not `iperf`.

A plan $\pi$ induces per-step time $T_{\text{step}}(\pi)$ and per-step token count $b(\pi)$. Realized throughput $\Theta(\pi) = b(\pi)/T_{\text{step}}(\pi)$.

**Ideal reference.** Let $C$ be model FLOPs per token. The FLOP-weighted ideal is

$$\Theta^\star = \frac{1}{C}\sum_{i=1}^{N}\rho_i , \qquad \eta(\pi) = \frac{\Theta(\pi)}{\Theta^\star}.$$

**Collapse.** For synchronous data parallelism with equal microbatches, every device waits on the slowest:

$$\Theta_{\text{sync}} = \frac{N\min_i \rho_i}{C}, \qquad \eta_{\text{sync}} = \frac{N\min_i\rho_i}{\sum_i \rho_i}.$$

One 4090 ($\rho \approx 1.2\times10^{14}$) in a pool of 63 H100s ($\rho\approx 6\times10^{14}$) gives $\eta_{\text{sync}}\approx 0.20$. That is the collapse.

**Statistical side.** Let $\mathcal{L}(n)$ be validation loss after $n$ tokens. Define the **token tax**

$$\tau = \frac{n_{\text{het}}(\ell^\star)}{n_{\text{hom}}(\ell^\star)} - 1,$$

the extra tokens the heterogeneous plan needs to reach loss $\ell^\star$. Total figure of merit: $\eta/(1+\tau)$. Papers report $\eta$; almost none report $\tau$.

**Assumptions, and where they break.**

- *$\rho_i$ is constant.* Violated: thermal throttling, MPS/MIG co-tenancy, and preemption on spot pools make $\rho_i$ time-varying by 10–30%.
- *Bandwidth is static and symmetric.* Violated across WAN links and under congestion from other jobs.
- *Devices are numerically identical.* Violated: bf16 accumulate order, TF32 vs FP16 tensor-core paths, and different cuDNN kernels give different rounding, so gradients are not bitwise reproducible across device classes.
- *Failures are rare.* Violated at 1000+ heterogeneous nodes and on preemptible pools, where MTBF is minutes to hours.

## 3. State of the Art

**Established (ablated, reproduced).**

- **Pipeline rebalancing by device speed.** HetPipe (Park et al., USENIX ATC 2020) partitions layers across "virtual workers" of unequal GPUs and reports up to ~49% faster convergence than Horovod on whimpy-GPU clusters — at ResNet/VGG scale, not LLM scale.
- **Cost-model search over device meshes.** Alpa (Zheng et al., OSDI 2022) established that automated inter/intra-operator partitioning matches hand-tuned plans; it assumes homogeneity. Metis (Um et al., USENIX ATC 2024) extends the search to heterogeneous GPU types and reports plans found in minutes that beat heterogeneity-unaware baselines.
- **Low-communication outer optimization.** DiLoCo (Douillard et al., 2023) shows 8 islands with 500 local steps between syncs match a fully synchronous baseline at ~150M–400M params with ~500× less communication. Independently reproduced by OpenDiLoCo (Jaghouar et al., 2024) and scaled in INTELLECT-1 (10B, 2024).
- **Preemption tolerance.** Bamboo (Thorpe et al., NSDI 2023) and Oobleck (Jang et al., SOSP 2023) show redundant-computation and pipeline-template reconfiguration keep throughput within a modest factor under frequent node loss.

**Claimed but unablated.**

- SWARM Parallelism (Ryabinin et al., ICML 2023) trains a 1B model on preemptible, geographically split T4/V100s with randomized stage routing. The routing's effect on *sample* efficiency versus a same-token homogeneous control is not isolated.
- Decentralized training over heterogeneous networks (Yuan et al., NeurIPS 2022) reports up to ~4.8× over heterogeneity-unaware baselines. The headline is a **benchmark number** against a baseline that makes no attempt to schedule; it is not $\eta$ against $\Theta^\star$.
- Mixed vendor (NVIDIA + AMD, or GPU + TPU) single-run training at frontier scale: vendor blog claims exist; no peer-reviewed loss-curve comparison to a homogeneous control.

**Theory SOTA.** Asynchronous SGD converges under bounded staleness $s$ with rate degrading roughly as $O(\sqrt{s})$ in the constant-factor term (Lian et al., ICML 2018, AD-PSGD; Stich, ICLR 2019, for local SGD). No bound is stated in terms of the device-speed spread $\max_i\rho_i/\min_i\rho_i$.

## 4. What Is Known

- **Collapse is arithmetic, not empirical.** $\eta_{\text{sync}} = N\min\rho_i/\sum\rho_i$ holds exactly for equal-microbatch synchronous DP. Measured on 8 GPUs (4×A100 + 4×V100, ~2.5× spread), $\eta \approx 0.55$ — matching prediction to within a few points.
- **Microbatch rebalancing recovers most of DP collapse.** Assigning per-device microbatch $\propto \rho_i$ restores $\eta$ toward $0.9$ *when memory permits*, at 100M–1B scale. It fails when the slow device is also the small-memory device.
- **Local-step methods buy heterogeneity tolerance cheaply at small scale.** DiLoCo at 150M params, 8 workers, $H=500$: matched baseline perplexity with 500× less communicated bytes. At 10B (INTELLECT-1, 30 nodes, 3 continents) the run completed with reported ~83–96% compute utilization within islands — but no same-token homogeneous control was run.
- **Straggler tails dominate over means.** In production DL clusters, per-step time is heavy-tailed; scheduling on mean $\rho_i$ underperforms scheduling on the p95 (Gavel, Narayanan et al., OSDI 2020, at 100+ GPU scale).
- **Pipeline parallelism is the natural home for heterogeneity** because stage width is a free variable, but bubble fraction grows as $(P-1)/(m+P-1)$ for $P$ stages and $m$ microbatches, and heterogeneous stages inflate the critical path.

## 5. What Is Not Known

- **Empirically open.** Whether $\eta \ge 0.8$ *and* $\tau \le 0.05$ are simultaneously achievable at $\ge 30$B params on a pool with $\ge 4\times$ FLOP spread. Runnable today on any large mixed cluster; nobody has published the paired control.
- **Empirically open.** Whether the DiLoCo family's loss-neutrality survives to $\ge 100$B params and $\ge 100$ islands. Scaling laws for outer-optimizer methods exist only up to ~10B.
- **Theoretically open.** A convergence bound for local/async SGD parameterized by the speed spread $\kappa = \max_i\rho_i/\min_i\rho_i$ that is tight enough to predict $\tau$. Current bounds are staleness-based and vacuous at realistic $s$.
- **Theoretically open.** Complexity of the plan-search problem. Heterogeneous pipeline partitioning with memory constraints is NP-hard by reduction from bin packing, but no approximation ratio is known for the version with communication costs.
- **Methodologically blocked.** $\Theta^\star$ itself. $\rho_i$ depends on the plan (a device given a wider tensor-parallel shard runs at a different efficiency), so the denominator is not plan-independent. Without a fixed denominator, "$\eta = 0.8$" is not comparable across papers.
- **Methodologically blocked.** Attribution of loss differences to heterogeneity versus to non-determinism. Cross-vendor bf16 rounding differences alone move final loss by a small but nonzero amount; no one has measured that floor.

## 6. Why It Is Hard

The binding obstruction is **confounded measurement compounded by control-arm cost**. Deciding the question requires two runs at the same token count — heterogeneous and homogeneous — at a scale where the effect is visible ($\ge 10$B). That is two full pretraining runs, and the homogeneous control needs the *same aggregate FLOPs*, which means holding a large uniform cluster idle-equivalent for the duration. Groups with mixed pools (decentralized collectives) cannot afford the homogeneous arm; groups with homogeneous pools have no incentive to build the heterogeneous one. So the published record is systems throughput numbers with no statistical control attached.

Secondary: **non-identifiability of the denominator**. $\rho_i$ is a function of the plan, so $\eta$ can be inflated by choosing a plan that makes each device look individually slow. And **evaluation drift** — "utilization" (MFU) measures FLOPs burned, not tokens usefully consumed; a plan that recomputes activations aggressively raises MFU while lowering $\Theta$.

## 7. Current Research (as of 2026)

- **Outer-optimizer scaling.** Google DeepMind's DiLoCo line, including Streaming DiLoCo with overlapped partial sync (Douillard et al., 2025), targets reduced peak bandwidth. Scaling-law work on the number of islands is active *(frontier — verify)*.
- **Decentralized production runs.** Prime Intellect (OpenDiLoCo, INTELLECT-1, and later RL-focused runs) is the main group actually training on globally heterogeneous, permissionless pools *(frontier — verify current model sizes)*.
- **Automated heterogeneous plan search.** Metis (ATC 2024), HAP (EuroSys 2024) and successors push cost-model-driven search over mixed device types; the open sub-problem is search under time-varying $\rho_i$.
- **Cross-vendor collectives.** NCCL/RCCL interop and UCC-based paths for mixed NVIDIA/AMD pools; correctness of reductions across differing accumulate orders is the live issue *(frontier — verify)*.
- **Asynchronous pipeline with bounded staleness** for slow-tier devices as prefetch/embedding workers rather than as full replicas.

## 8. Concrete Next Experiment

**Question.** At what speed spread $\kappa$ does the token tax $\tau$ become nonzero for a DiLoCo-style plan?

**Scale.** 7B-parameter decoder-only model (Llama-architecture), 150B tokens, fixed data order and seed set of 3.

**Arms.**
1. **Control (homogeneous).** 64×H100, standard FSDP + TP, synchronous. Record $\mathcal{L}(n)$ and $\Theta$.
2. **Heterogeneous, $\kappa=2$.** 32×H100 + 64×A100-80GB, equal aggregate FLOPs to arm 1, DiLoCo islands with $H=100$, per-island plan from Metis-style search.
3. **Heterogeneous, $\kappa=5$.** 32×H100 + 128×L40S, equal aggregate FLOPs.
4. **Ablation.** Arm 3 run *synchronously* with speed-proportional microbatches, to separate the effect of relaxing synchrony from the effect of the device mix.

Hold aggregate FLOPs, tokens, batch size in tokens, and LR schedule identical across all arms. Measure $\rho_i$ per arm in isolation first to fix $\Theta^\star$.

**Deciding number.** $\tau$ at $\ell^\star = $ arm-1 final loss, reported with a seed-variance error bar. If $\tau \le 0.03$ at $\kappa=5$ while $\eta \ge 0.8$, heterogeneous training is a solved engineering problem at this scale and the open question moves to $\ge 100$B. If $\tau \ge 0.15$ at $\kappa=5$, the throughput gain is illusory — $\eta/(1+\tau) < \eta_{\text{sync}}$ — and the field should stop reporting $\eta$ alone.

Cost estimate: roughly 4 × 7B × 150B-token runs ≈ $6\times10^{21}$ model FLOPs total, order 2–4 weeks on the stated pools.

## 9. Key References

- **[Foundational]** Park, Yun, Chang, et al. *HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism.* USENIX ATC, 2020.
- **[Foundational]** Narayanan, Harlap, Phanishayee, et al. *PipeDream: Generalized Pipeline Parallelism for DNN Training.* SOSP, 2019.
- **[Foundational]** Lian, Zhang, Zhang, Liu. *Asynchronous Decentralized Parallel Stochastic Gradient Descent.* ICML, 2018. — arXiv:1710.06952
- **[Foundational]** Stich. *Local SGD Converges Fast and Communicates Little.* ICLR, 2019. — arXiv:1805.09767
- **[SOTA]** Zheng, Li, Zhuang, et al. *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.* OSDI, 2022. — arXiv:2201.12023
- **[SOTA]** Yuan, He, Davis, et al. *Decentralized Training of Foundation Models in Heterogeneous Environments.* NeurIPS, 2022. — arXiv:2206.01288
- **[SOTA]** Ryabinin, Dettmers, Diskin, Borzunov. *SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient.* ICML, 2023. — arXiv:2301.11913
- **[SOTA]** Douillard, Feng, Rusu, et al. *DiLoCo: Distributed Low-Communication Training of Language Models.* 2023. — arXiv:2311.08105
- **[SOTA]** Jaghouar, Ong, Hagemann. *OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training.* 2024. — arXiv:2407.07852
- **[SOTA]** Jang, Yang, Kim, et al. *Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates.* SOSP, 2023.
- **[SOTA]** Thorpe, Zhao, Eyolfson, et al. *Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs.* NSDI, 2023.
- **[SOTA]** Um, Lee, Wang, et al. *Metis: Fast Automatic Distributed Training on Heterogeneous GPUs.* USENIX ATC, 2024.
- **[Survey]** Narayanan, Santhanam, Kazhamiaka, Phanishayee, Zaharia. *Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads.* OSDI, 2020.
- **[Systems]** Athlur, Saran, Sivathanu, Ramjee, Kwatra. *Varuna: Scalable, Low-cost Training of Massive Deep Learning Models.* EuroSys, 2022. — arXiv:2111.04007

## 10. Worked Example

A pool of 32 H100-80GB and 32 L40S-48GB. Measured sustained throughput on a 7B model, bf16, activation checkpointing on:

| Class | $n$ | $\rho_i$ (model TFLOP/s) | $M_i$ free |
|---|---|---|---|
| H100 | 32 | 420 | 74 GB |
| L40S | 32 | 105 | 43 GB |

$\sum \rho_i = 32(420+105) = 16{,}800$ TFLOP/s. Model FLOPs per token $C \approx 6\times 7\times10^9 = 4.2\times10^{10}$. So $\Theta^\star = 1.68\times10^{16}/4.2\times10^{10} \approx 400{,}000$ tokens/s.

**Arm A — naive synchronous FSDP, uniform microbatch.** Every rank waits on an L40S. $\Theta = 64\times105/420 \dots$ more directly, $\eta_{\text{sync}} = 64\times105/16{,}800 = 0.40$. Realized $\approx 160{,}000$ tokens/s. Note: dropping the L40S entirely gives $32\times420/4.2\times10^{10} = 320{,}000$ tokens/s. **Adding 32 GPUs halved throughput.**

**Arm B — speed-proportional microbatches.** Give H100s 4× the microbatch of L40S. Compute balances; $\eta$ should hit ~0.95. It does not. FSDP shards optimizer state uniformly, so each L40S must hold $\tfrac{1}{64}$ of AdamW state ($7\text{B}\times12\text{B} / 64 = 1.31$ GB) plus its shard of parameters and gradients — fine — but the all-gather of full-precision parameters for the current layer requires a 2-byte 7B-wide transient the L40S can only fit at reduced prefetch depth. Measured $\eta \approx 0.71$: memory, not FLOPs, is the binding constraint.

**Arm C — DiLoCo, 2 islands (H100 island, L40S island), $H=200$.** Each island runs at its own rate. $\eta$ rises to 0.93 — the H100 island completes ~4 outer rounds while the L40S island completes 1.

**Where the obstruction becomes visible.** In Arm C the two islands now contribute unequal numbers of outer updates from unequal amounts of data. The final loss at 150B tokens comes out 0.021 nats above Arm A's. Is that:

1. the token tax $\tau$ from asynchrony,
2. the effective batch-size change (Arm C's outer batch is not Arm A's),
3. bf16 accumulate-order differences between Ada and Hopper tensor cores, or
4. seed noise (single-seed run-to-run spread at 7B is on the order of 0.01–0.02 nats)?

The measurement cannot separate them. That is the problem: 0.021 nats is inside the noise floor of a single-seed comparison, and the seed-replicate control costs another three 7B runs. Every published heterogeneous-training result to date reports the $\eta = 0.93$ and omits the 0.021.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*