---
id: 15-mixture-of-experts/all-to-all-lower-bounds-moe
title: "All-to-All Communication Lower Bounds for MoE Training"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# All-to-All Communication Lower Bounds for MoE Training

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/all-to-all-lower-bounds-moe` · **Status:** open

## 1. Problem Statement

Expert-parallel MoE training moves every token's hidden vector to the device holding its selected expert and moves the expert output back. This is two all-to-all collectives per MoE layer per direction of the backward pass. The question: **what is the minimum communication volume and minimum latency any correct expert-parallel MoE training step must incur, as a function of the routing distribution, the expert placement, and the memory per device?**

Three variants, with different difficulty:

- **Theory variant.** Fix a device count $P$, per-device memory $M$ words, and a routing assignment $\sigma$ mapping tokens to experts. Over all algorithms that compute the exact MoE layer output, prove a lower bound on words communicated per device, $W_{\min}(P, M, \sigma)$, and exhibit a matching algorithm. Solving it means a bound tight to within a constant, not $\Omega(1)$-vacuous.
- **Method variant.** Design placement + schedule + replication policies that provably approach $W_{\min}$, rather than beating the previous system by a measured factor.
- **Measurement variant.** Define, and instrument, the *achieved fraction of the bound* for a running job — an all-to-all analogue of Model FLOP Utilization. Today systems papers report speedup over a named baseline, which does not say how much headroom remains.

The problem is open in all three variants; the theory variant is the load-bearing one, since without $W_{\min}$ the other two have no denominator.

## 2. Formal Setting

**Objects.** $P$ devices, grouped into $N$ nodes of $G = P/N$ devices. $E$ routed experts, expert-parallel degree $P_E \le P$, so each device holds $E/P_E$ experts. Model width $d_m$, expert hidden width $d_{ff}$, top-$k$ routing, per-device micro-batch of $B$ tokens, element size $b$ bytes ($b=2$ for bf16, $1$ for fp8 dispatch).

**Routing.** Router logits $z_t = W_r h_t \in \mathbb{R}^{E}$; the assignment is $S_t = \mathrm{top}\text{-}k(z_t)$. Define the empirical load vector
$$\ell_e \;=\; \frac{1}{Bk}\sum_{t=1}^{B}\mathbb{1}[e \in S_t], \qquad \textstyle\sum_e \ell_e = 1 .$$
Imbalance is measured as $\lambda = P_E \max_{p} \sum_{e \in \mathcal{E}_p} \ell_e$, the max-to-mean device load ratio ($\lambda = 1$ is perfect balance). Measure $\lambda$ from the dispatch histogram per micro-batch, not averaged over a step — the straggler is per-collective.

**Volume.** Naive dispatch sends, per device per MoE layer forward,
$$V_{\text{disp}} \;=\; B\,k\,d_m\,b\,\Big(1 - \tfrac{1}{P_E}\Big) \ \text{bytes},$$
with $V_{\text{comb}} = V_{\text{disp}}$ for the combine, and the backward pass repeating both. Measure it from NIC/NVLink counters (`nvidia-smi nvlink`, IB port counters), not from the arithmetic — padding to capacity factor $c$ inflates real bytes by up to $c$.

**Time.** Under a Hockney model per link, $T = \alpha \, s + V/\beta$ for $s$ messages, latency $\alpha$, bandwidth $\beta$. An all-to-all on $P$ ranks has $s = P-1$, so the $\alpha$ term grows linearly and dominates when $B k d_m b / P$ falls below the per-link message-size knee (typically 64–256 KB on InfiniBand).

**Bisection argument.** If the network has bisection bandwidth $\beta_{\text{bis}}$ and routing is uniform, at least half the payload crosses any balanced cut in expectation, giving
$$T \;\ge\; \frac{B\,k\,d_m\,b\,P/2}{\beta_{\text{bis}}}.$$
This is the only lower bound anyone actually invokes, and it is loose: it ignores memory $M$, ignores replication, and ignores that $\sigma$ is not uniform.

**Assumptions, and which are violated.** (i) *Routing is data-dependent and unknown before the layer runs* — true, but violated in spirit by systems that predict routing one layer ahead (Lina, Janus). (ii) *Experts are not replicated* — violated by DeepSpeed-MoE and by production hot-expert replication. (iii) *Exact computation* — violated by every capacity-factor implementation, which drops tokens; a dropping system is solving a different problem than the exact one the bound is stated for. (iv) *Uniform link bandwidth* — badly violated: intra-node NVLink is ~$10\times$ inter-node InfiniBand per GPU, so the flat all-to-all model misprices the real cost by an order of magnitude.

## 3. State of the Art

**Theory SOTA (established, but not about MoE).** Communication lower bounds for dense linear algebra are settled: Hong & Kung's red-blue pebble game (STOC 1981) gives $\Omega(n^3/\sqrt{M})$ for matmul; Irony, Toledo & Tiskin (JPDC 2004) give the distributed-memory per-processor form $\Omega(n^3/(P\sqrt{M}))$; Ballard, Demmel, Holtz & Schwartz (SIMAX 2011) generalize to most of direct linear algebra; Solomonik & Demmel (Euro-Par 2011) give 2.5D algorithms that trade memory for communication and attain the bound. **None of these apply to MoE dispatch**, because the cost there is a data-dependent permutation, not an arithmetic dependence graph — there is no reuse to exploit, and the pebbling lower bounds go vacuous.

**Systems SOTA (established as engineering, unablated as optimality).** GShard (Lepikhin et al., ICLR 2021) and Switch Transformer (Fedus, Zoph & Shazeer, JMLR 2022) established the capacity-factor all-to-all pattern. DeepSpeed-MoE (Rajbhandari et al., ICML 2022) introduced hierarchical all-to-all plus expert replication. Tutel (Hwang et al., MLSys 2023) added adaptive parallelism switching and 2D hierarchical all-to-all. FasterMoE (He et al., PPoPP 2022) added expert shadowing and a performance roofline model. MegaBlocks (Gale et al., MLSys 2023) removed token dropping with block-sparse kernels. Lina (Li et al., USENIX ATC 2023) and Janus (Jiang et al., SIGCOMM 2023) prioritize or invert the data movement (move experts, not tokens). DeepSeek-V3 (2024) uses node-limited routing — each token reaches at most $M_{\text{node}}=4$ nodes — plus DualPipe to overlap communication with computation.

**Claimed but unablated.** Every one of these reports speedup over a chosen baseline. None reports distance from an optimum, because no optimum is known. The near-total overlap claimed for DualPipe is a benchmark number on one cluster topology and one model shape; it has not been shown to be a property of the schedule rather than of that hardware's compute-to-bandwidth ratio.

## 4. What Is Known

- All-to-all is the dominant non-compute cost at scale. DeepSpeed-MoE and subsequent measurements put MoE communication at roughly 30–60% of step time for multi-node expert parallelism before overlap; FasterMoE reported all-to-all exceeding half of iteration time in its 64-GPU measurements.
- Node-limited routing gives a hard, non-empirical cap: cross-node bytes per token drop from $k\,d_m b$ to $M_{\text{node}} d_m b$. At DeepSeek-V3's $k=8$, $M_{\text{node}}=4$, that is exactly $2\times$, independent of load balance.
- Hierarchical (intra-node then inter-node) all-to-all reduces inter-node messages from $P-1$ to $N-1$ per device. Tutel reported single-MoE-layer speedups up to $5.75\times$ over a flat baseline and about 40% end-to-end gain training SwinV2-MoE on 2048 A100s.
- Load imbalance is the binding constraint, not average volume. All-to-all completes at the straggler, so step time scales with $\lambda$, not with $\mathbb{E}[\ell]$. Auxiliary-loss balancing (Switch) and DeepSeek-V3's auxiliary-loss-free bias adjustment both target $\lambda$ directly.
- Dropless routing (MegaBlocks) removes the capacity-factor padding overhead; reported ~40% end-to-end speedup over Tutel-style dMoE at the scales tested (up to 64 GPUs), with no quality loss from dropping.
- fp8 dispatch halves $V$ relative to bf16 and is used in production (DeepSeek-V3). This is a constant-factor win on a bound whose exponent is unknown.

## 5. What Is Not Known

- **Theoretically open.** No non-trivial lower bound on MoE all-to-all volume as a function of $(P, M, \sigma)$. Specifically: is there a memory-communication tradeoff for MoE dispatch analogous to the $1/\sqrt{M}$ of matmul, purchased by expert replication? Given $r$ replicas per expert costing $r \cdot |{\rm expert}|$ memory and $r$-way gradient reduction, the optimal $r^\star(\beta_{\text{intra}}, \beta_{\text{inter}}, \ell)$ is unproven. Also open: whether any placement can beat the bisection bound in expectation when $\ell$ is non-uniform and known.
- **Empirically open.** No published head-to-head of hierarchical all-to-all, expert replication, node-limited routing, and expert-migration on a *single* cluster with a *single* model. Every comparison is cross-paper, cross-hardware, cross-model. The experiment is runnable on 256–512 GPUs; nobody has run it.
- **Methodologically blocked.** There is no accepted "communication utilization" metric. MFU has a denominator (peak FLOP/s); all-to-all has none, because the achievable minimum depends on the routing realization. Until $W_{\min}$ exists, "we cut communication by 40%" is uninterpretable — it may be 40% of a $10\times$ gap or 40% of a $1.4\times$ one.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth compounded by a data-dependent adversary**. The bound must be stated over routing distributions $\sigma$, but $\sigma$ is produced by the model being trained and drifts across training: early-training routing is near-uniform, late-training routing is specialized and heavy-tailed. A bound proved for uniform $\sigma$ is not the bound that binds at step 100k, and a bound proved for worst-case $\sigma$ is vacuous (all tokens to one expert).

Second, the measurement is confounded by overlap. Modern systems hide all-to-all behind computation, so wall-clock step time is insensitive to communication volume until the hidden cost exceeds the compute shadow — at which point it becomes fully visible. Any measured "communication time" is therefore a step function of a quantity nobody reports: the compute-to-communication ratio of the specific cluster. Two systems with identical schedules will report opposite conclusions on an H100/NVLink pod versus a 200 Gbps Ethernet cluster.

## 7. Current Research (as of 2026)

- **Fine-grained overlap.** Kernel-level fusion of dispatch with expert GEMM, so communication is decomposed to tile granularity rather than layer granularity — the Comet line of work from ByteDance and similar efforts *(frontier — verify)*. This attacks the schedule, not the volume.
- **Routing-aware placement.** Expert-parallel load balancers that re-place and replicate experts from observed load histograms, including DeepSeek's open-sourced EPLB. Established as engineering; the placement optimality question is untouched.
- **Topology-aware collectives.** NVSHMEM/DeepEP-style device-initiated all-to-all removing the CPU from the critical path *(frontier — verify)*.
- **Theory side.** The relevant tools — massively-parallel-computation round/communication bounds (Beame, Koutris & Suciu, PODS 2013) and memory-independent bounds (Ballard et al., SPAA 2012) — have not been brought to bear on MoE. This is the clearest unexploited opening in the problem.

## 8. Concrete Next Experiment

**Establish the empirical denominator before attempting the theorem.**

- **Scale.** 256 GPUs (32 nodes × 8), a 40B-total / 4B-active MoE, $d_m = 4096$, $d_{ff}=1408$, $E=128$, $k=8$, $B=4096$ tokens/device, bf16, trained to 50B tokens. Fixed cluster, fixed model, fixed data order.
- **Arms.** (1) *Control*: flat all-to-all, capacity factor 1.25, no replication. (2) Hierarchical 2D all-to-all. (3) Node-limited routing, $M_{\text{node}}=4$. (4) Top-8 hot-expert replication, $r=2$. (5) Arms 2+3+4 combined. (6) *Oracle floor*: replay the recorded routing traces offline and solve the placement/schedule as a min-makespan flow problem with a solver, ignoring the cost of solving. Arm 6 is the point of the experiment.
- **Deciding number.** $\rho = T_{\text{a2a}}^{\text{arm}} / T_{\text{a2a}}^{\text{oracle}}$, the per-micro-batch all-to-all critical-path time divided by the offline oracle's, reported at 10%, 50% and 100% of training. If the best practical arm has $\rho \le 1.2$, current systems are near-optimal and the theory question is of academic interest only. If $\rho \ge 2$, there is a factor of two on the table and the lower-bound question is worth the proof effort. Publishing the routing traces makes arm 6 reproducible by others without the 256 GPUs.

## 9. Key References

- **[Foundational]** Hong, J.-W., Kung, H. T. *I/O Complexity: The Red-Blue Pebble Game.* STOC, 1981.
- **[Foundational]** Irony, D., Toledo, S., Tiskin, A. *Communication Lower Bounds for Distributed-Memory Matrix Multiplication.* Journal of Parallel and Distributed Computing, 2004.
- **[Foundational]** Ballard, G., Demmel, J., Holtz, O., Schwartz, O. *Minimizing Communication in Numerical Linear Algebra.* SIAM Journal on Matrix Analysis and Applications, 2011.
- **[Foundational]** Solomonik, E., Demmel, J. *Communication-Optimal Parallel 2.5D Matrix Multiplication and LU Factorization Algorithms.* Euro-Par, 2011.
- **[Foundational]** Shazeer, N. et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Lepikhin, D. et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[SOTA]** Fedus, W., Zoph, B., Shazeer, N. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[SOTA]** Rajbhandari, S. et al. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* ICML, 2022. — arXiv:2201.05596
- **[SOTA]** Hwang, C. et al. *Tutel: Adaptive Mixture-of-Experts at Scale.* MLSys, 2023. — arXiv:2206.03382
- **[SOTA]** He, J. et al. *FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models.* PPoPP, 2022.
- **[SOTA]** Gale, T., Narayanan, D., Young, C., Zaharia, M. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[SOTA]** Li, J. et al. *Accelerating Distributed MoE Training and Inference with Lina.* USENIX ATC, 2023.
- **[SOTA]** Jiang, J. et al. *Janus: A Unified Distributed Training Framework for Sparse Mixture-of-Experts Models.* ACM SIGCOMM, 2023.
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Survey]** Chan, E., Heimlich, M., Purkayastha, A., van de Geijn, R. *Collective Communication: Theory, Practice, and Experience.* Concurrency and Computation: Practice and Experience, 2007.
- **[Survey]** Ivanov, A. et al. *Data Movement Is All You Need: A Case Study on Optimizing Transformers.* MLSys, 2021. — arXiv:2007.00072

## 10. Worked Example

A 512-GPU job (64 nodes × 8), $d_m = 7168$, $d_{ff} = 2048$, $k=8$, $B = 4096$ tokens/GPU, bf16.

Dispatch payload per GPU per MoE layer forward:
$$4096 \times 8 \times 7168 \times 2 \ \text{B} = 4.70\times10^{8}\ \text{B} \approx 470\ \text{MB},$$
and the same again for combine: **0.94 GB per layer, forward only**. With 58 MoE layers that is 54 GB per GPU per forward pass.

Under uniform routing, the off-node fraction is $1 - 8/512 = 98.4\%$, so essentially all of it crosses InfiniBand. At 400 Gbps ($\approx 50$ GB/s) egress per GPU, dispatch alone costs $470\,\text{MB} / 50\,\text{GB/s} = 9.4$ ms per layer.

Now apply node-limited routing with $M_{\text{node}}=4$. Each token's vector crosses to at most 4 nodes regardless of $k$:
$$4 \times 4096 \times 7168 \times 2\ \text{B} = 2.35\times10^{8}\ \text{B} = 235\ \text{MB} \;\Rightarrow\; 4.7\ \text{ms}.$$

Expert compute on the receiving side, SwiGLU (3 matmuls), $32{,}768$ token-expert pairs per GPU:
$$3 \times 2 \times 7168 \times 2048 \times 32768 = 2.89\ \text{TFLOP} \ \Rightarrow\ 7.2\ \text{ms at } 400\ \text{TFLOP/s effective.}$$

So the shadow is 7.2 ms and the communication is 4.7 ms — overlap works, with 1.53$\times$ of margin. **The obstruction:** that margin is a property of this cluster's 50 GB/s-per-GPU-to-400 TFLOP/s ratio, not of the algorithm. Move to a fabric at 25 GB/s per GPU and communication becomes 9.4 ms against a 7.2 ms shadow — the same schedule flips from fully hidden to 30% exposed, and the reported "communication cost" jumps from 0 to 2.2 ms per layer with no change to the code. Worse, load imbalance $\lambda = 1.3$ — mild by production standards — pushes the straggler to $4.7 \times 1.3 = 6.1$ ms and eats most of the remaining margin.

Nobody can say whether the 235 MB is within 10% or within 200% of the true minimum for this routing realization, because $W_{\min}$ is unproven and the offline oracle of §8 has never been computed. Both the reported win from node-limiting ($2\times$) and the reported win from overlap ($\infty$, apparently) are measured against baselines, not against a floor.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*