---
id: 32-hardware-and-kernels/compute-communication-overlap-limits
title: "Compute-Communication Overlap Limits in Pipeline Parallelism"
topic: 32-hardware-and-kernels
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Communication Overlap Limits in Pipeline Parallelism

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/compute-communication-overlap-limits` · **Status:** partially-solved

## 1. Problem Statement

A pipeline-parallel training step splits a model into $P$ stages on $P$ devices and a global batch into $M$ microbatches. Devices idle in two ways: **bubbles** (dependency stalls in the schedule) and **exposed communication** (point-to-point activation/gradient transfers, plus collectives from tensor- and data-parallel dimensions, not hidden behind compute). The question is how much of that idle time is removable.

Three variants, with different difficulty:

- **Measurement.** Given a real training run, attribute each device-microsecond to compute, bubble, exposed communication, or memory stall — and do it without the attribution changing under the profiler. Currently the weakest link.
- **Method.** Given a model, a device count, a memory budget $B$, and an interconnect topology, produce a schedule plus a communication decomposition that minimizes step time. Zero-bubble schedules solve the dependency half under generous memory; the communication half is solved only per-kernel and per-topology.
- **Theory.** Prove a lower bound on step time as a function of $(P, M, B, \text{bandwidth}, \text{overlap capability})$, and show whether it is attainable. Open in the memory-constrained, bandwidth-limited regime.

Solving it means: a schedule search that provably matches a lower bound, or a bound proving the residual idle fraction observed in production runs (typically 10–25%) is irreducible under the stated memory and bandwidth constraints.

## 2. Formal Setting

Stage $p \in \{1..P\}$, microbatch $m \in \{1..M\}$. Measured quantities:

- $t^F_p, t^B_p$ — forward and input-gradient time for one microbatch on stage $p$, measured as CUDA-event deltas on an isolated stream with no concurrent communication.
- $t^W_p$ — weight-gradient time, separable from $t^B_p$ because $\partial L/\partial W$ has no successor in the dependency graph (the split exploited by zero-bubble schedules).
- $c_p = S_p / \beta$ — activation transfer time, $S_p$ the boundary tensor bytes, $\beta$ the achieved (not nominal) link bandwidth from a `nccl-tests` `sendrecv` at that message size.
- $B$ — peak activation memory, in bytes, measured as `torch.cuda.max_memory_allocated` minus parameter/optimizer state.
- $\alpha \in [0,1]$ — **overlap efficiency**: with compute time $T_c$ alone and comm time $T_m$ alone, the co-scheduled time is $T_{co}$, and
$$\alpha = \frac{T_c + T_m - T_{co}}{\min(T_c, T_m)}.$$
$\alpha = 1$ is perfect hiding, $\alpha = 0$ is full serialization; $\alpha < 0$ is possible and common.

Step time and bubble fraction:
$$T_{\text{step}} = \sum_{m,p \to \text{device } p} (t^F_p + t^B_p + t^W_p) + T_{\text{bubble}} + T_{\text{exposed}}, \qquad \phi = \frac{T_{\text{bubble}}}{T_{\text{step}}}.$$
For GPipe-style fill-drain with uniform stages, $\phi = (P-1)/(M + P - 1)$; for interleaved 1F1B with $v$ virtual stages per device, $\phi \approx (P-1)/(vM)$ at $v\times$ the P2P traffic.

Assumptions, with the ones known to be violated marked:

1. Stage times are uniform and deterministic. **Violated**: embedding/LM-head stages differ by 1.5–3$\times$; MoE stages vary per microbatch with routing; clock throttling adds 3–8% jitter.
2. Communication and compute occupy disjoint resources. **Violated**: NCCL kernels consume SMs, and NVLink/PCIe traffic contends with HBM bandwidth. This is exactly why $\alpha < 1$.
3. $\alpha$ is a constant of the hardware. **Violated**: it depends on message size, kernel arithmetic intensity, and SM allocation — it is a function, not a scalar.
4. Memory is the only capacity constraint. **Violated**: recomputation policy trades $B$ against $t^F$, coupling the two axes the theory usually separates.

## 3. State of the Art

**Established (schedules).** GPipe (Huang et al., NeurIPS 2019) sets the $(P-1)/(M+P-1)$ baseline. 1F1B (PipeDream, Narayanan et al., SOSP 2019) keeps the bubble but caps activation memory at $O(P)$ microbatches. Interleaved 1F1B (Narayanan et al., SC 2021) divides the bubble by $v$ and is the production default in Megatron-LM; its cost — $v\times$ P2P messages — is measured, not assumed. Zero Bubble (Qi et al., ICLR 2024) splits $B$ into $B$ and $W$ and reaches a theoretically empty bubble with ZB-H2 under $\sim 2\times$ the 1F1B activation memory; ZB-V matches 1F1B memory with near-zero bubble. Chimera (Li & Hoefler, SC 2021) uses bidirectional pipelines at $2\times$ weight memory.

**Established (communication).** Decomposing a matmul-plus-collective into chunks that pipeline against each other (Wang et al., "Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models", ASPLOS 2023) is a real, reproduced technique on TPUs; PyTorch's `async-TP` and FLUX (Chang et al., 2024) are the GPU analogues.

**Claimed but unablated.** DeepSeek-V3's DualPipe (technical report, 2024) claims near-full overlap of all-to-all expert routing with compute on 2048 H800s, but the report gives no bubble-versus-exposed-comm decomposition and no ablation isolating DualPipe from the FP8 pipeline and the warp-specialized kernels shipped alongside it. Zero Bubble's headline "up to 23% throughput over 1F1B" is a benchmark number on specific model/device configurations, not a scaling law. Vendor MFU figures (MegaScale, Narayanan SC'21) are end-to-end and do not separate bubble from exposed communication at all.

**Theory SOTA.** Zero Bubble's ILP/heuristic scheduler is optimal only under uniform, deterministic stage times and free communication. No lower bound is known that jointly binds $P$, $M$, $B$, $\beta$, and $\alpha < 1$.

## 4. What Is Known

- Interleaving works: Megatron-LM on 3072 A100s reached 502 PFLOP/s, ~52% of peak, on a 1T-parameter model (Narayanan et al., SC 2021), with interleaved 1F1B contributing a reported ~10% throughput gain over non-interleaved at that scale.
- MegaScale (Sun et al., NSDI 2024) reports 55.2% MFU for a 175B model on 12,288 GPUs — the highest independently-published large-scale figure, still leaving ~45% unaccounted.
- Zero-bubble schedules deliver a measured 15–23% step-time improvement over 1F1B at 8–32 pipeline stages with matched memory (Qi et al., ICLR 2024) — a real gain, at small pipeline depth.
- Decomposed overlap of collectives with dependent matmuls yields 1.2–1.4$\times$ on the affected layers (ASPLOS 2023, TPU v4); FLUX reports comparable GPU numbers.
- $\alpha < 1$ is robust: NCCL kernels reserve SMs (`NCCL_MAX_NCHANNELS` trades comm bandwidth against compute occupancy), and enabling overlap typically slows the compute kernel by 5–15% at the same time it hides communication. This is measured on A100/H100 by multiple groups and is the reason naïve overlap sometimes loses.
- DeepSeek-V3 trained 671B parameters (37B active) on 14.8T tokens for 2.788M H800 GPU-hours — an efficiency figure consistent with, but not proof of, near-complete overlap.

## 5. What Is Not Known

- **Theoretically open.** A lower bound on $T_{\text{step}}$ under a memory cap $B$, finite bandwidth $\beta$, and overlap efficiency $\alpha<1$. Zero-bubble optimality assumes free communication; nobody has shown whether zero bubble and full overlap are simultaneously achievable, or whether they trade off.
- **Empirically open.** Whether zero-bubble schedules retain their 15–23% advantage at $P \geq 64$ with cross-node P2P, MoE routing imbalance, and recomputation on. Runnable on a 512-GPU cluster; not published.
- **Methodologically blocked.** The attribution itself. There is no accepted way to split a device-microsecond between "bubble" and "exposed communication" when a NCCL kernel is resident but stalled on the network, and profilers that can tell them apart (Nsight Systems with NVTX plus CUPTI) perturb the schedule by 2–10%. Reported bubble fractions across papers are therefore not comparable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus non-identifiability**. Compute and communication share SMs, HBM bandwidth, and the copy engines. When a step is 20% slower than the sum of its isolated parts, that 20% has at least four indistinguishable causes: dependency stall, network latency, SM contention from NCCL, and HBM contention. The single observable — wall-clock step time — is one equation in four unknowns, and the standard intervention (turn off overlap and re-measure) changes the kernels' occupancy, so the counterfactual is not the same system. Compounding it: the decisive experiment needs $P \geq 64$ across nodes, i.e. 512+ GPUs for days, which is why the regime where the answer differs is exactly the regime nobody ablates.

## 7. Current Research (as of 2026)

- Zero-bubble and its descendants (Sea AI Lab; ZB-V, and hybrid ZB + MoE schedules) — extending the $B/W$ split to expert-parallel all-to-all *(frontier — verify)*.
- DualPipe-style bidirectional MoE schedules with computation-communication co-design, following DeepSeek-V3; several open reimplementations exist in Megatron-LM forks *(frontier — verify)*.
- Compiler-level decomposition: XLA/Mosaic and Triton-level fusion of collectives into matmul epilogues; PyTorch `async-TP` and SymmetricMemory make this routine on NVLink domains.
- Automatic parallelism search (Alpa, OSDI 2022, and successors) now increasingly includes overlap as a cost-model term rather than an afterthought.
- Hardware direction: NVLink-domain scale-up (GB200 NVL72) shrinks $c_p$ enough that pipeline parallelism may be replaced by tensor parallelism at these depths — which would dissolve the problem rather than solve it.

## 8. Concrete Next Experiment

**Question.** Does the zero-bubble advantage survive cross-node pipeline depth, and how much of the residual is exposed communication rather than bubble?

**Scale.** 512 H100s, 64-way pipeline (8 nodes deep on cross-node P2P) $\times$ 8-way data parallel, a 70B dense transformer, $M = 128$ microbatches, activation recomputation off, matched activation memory across arms.

**Arms.**
1. Control: interleaved 1F1B, $v = 4$ (Megatron default).
2. ZB-V zero-bubble at matched memory.
3. ZB-V plus decomposed overlap of the stage-boundary P2P into 4 chunks.

**Instrumentation.** Per-rank CUPTI traces; classify each idle interval as bubble (no work enqueued) or exposed comm (NCCL kernel resident). Report $\alpha$ per arm from a paired isolated-kernel run.

**Deciding number.** The **residual idle fraction** $\phi + T_{\text{exposed}}/T_{\text{step}}$ in arm 3. If it falls below 5%, the problem is method-solved at production depth and only the bound is missing. If it stays above 15% while arm 2 shows $\phi < 2\%$, then bubbles are solved and the open problem is entirely exposed communication — a different problem than the literature currently names.

## 9. Key References

- **[Foundational]** Y. Huang et al. *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.* NeurIPS, 2019. — arXiv:1811.06965
- **[Foundational]** D. Narayanan et al. *PipeDream: Generalized Pipeline Parallelism for DNN Training.* SOSP, 2019.
- **[Foundational]** D. Narayanan et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC, 2021. — arXiv:2104.04473
- **[SOTA]** P. Qi, X. Wan, G. Huang, M. Lin. *Zero Bubble Pipeline Parallelism.* ICLR, 2024. — arXiv:2401.10241
- **[SOTA]** S. Li, T. Hoefler. *Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines.* SC, 2021.
- **[SOTA]** S. Wang et al. *Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models.* ASPLOS, 2023.
- **[Systems]** Z. Sun et al. *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs.* NSDI, 2024. — arXiv:2402.15627
- **[Systems]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Systems]** L. Zheng et al. *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.* OSDI, 2022. — arXiv:2201.12023
- **[Related]** S. Fan et al. *DAPPLE: A Pipelined Data Parallel Approach for Training Large Models.* PPoPP, 2021. — arXiv:2007.01045

## 10. Worked Example

70B dense transformer, 80 layers, $P = 16$ stages (5 layers each), hidden $h = 8192$, sequence $L = 4096$, microbatch size 1, bf16. Boundary activation per microbatch:
$$S = L \cdot h \cdot 2\ \text{bytes} = 4096 \times 8192 \times 2 = 67\ \text{MB}.$$

Measured on H100: $t^F_p \approx 21$ ms, $t^B_p \approx 21$ ms, $t^W_p \approx 21$ ms per stage-microbatch. Cross-node InfiniBand at an achieved $\beta = 45$ GB/s gives $c_p = 67/45{,}000 \approx 1.5$ ms — 2.4% of the 63 ms stage time. Communication looks free.

With $M = 64$: 1F1B bubble fraction $\phi = 15/(64+15) = 19.0\%$. Interleaved with $v=4$: $\phi \approx 4.7\%$, but P2P messages go from 1 to 4 per stage-microbatch — $4 \times 1.5 = 6$ ms per boundary against 15.75 ms of compute per virtual stage.

Now the obstruction. Suppose the schedule hides all 6 ms behind compute. The measurement says the step should be $63 \times 64 / \text{stage} \times (1 + 0.047) = 4.22$ s. Real runs at this configuration land near 4.6–4.8 s. Where did 400–560 ms go?

Set $T_c = 15.75$ ms, $T_m = 6$ ms per virtual stage. If $\alpha = 1$, $T_{co} = 15.75$. Observed $T_{co} \approx 17.4$ ms implies
$$\alpha = \frac{15.75 + 6 - 17.4}{6} = 0.725.$$
So 27.5% of the communication — 1.65 ms per virtual stage, 422 ms per step — is *not* hidden. But there is no way to tell from this measurement whether that 1.65 ms is network latency the schedule failed to cover, or SM contention slowing the matmul while the network was actually idle. The two hypotheses predict identical wall-clock and demand opposite fixes: more decomposition chunks in the first case, fewer NCCL channels in the second. Zero-bubble scheduling addresses neither — it removes the 4.7%, leaving the 10% that this page is about.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*