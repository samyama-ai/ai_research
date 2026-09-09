---
id: 13-parameter-efficient-adaptation/multi-adapter-serving-throughput
title: "Multi-Adapter Serving Throughput Limits"
topic: 13-parameter-efficient-adaptation
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multi-Adapter Serving Throughput Limits

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/multi-adapter-serving-throughput` · **Status:** solved-but-impractical

## 1. Problem Statement

**Input.** One frozen base model $W$, a store of $N$ LoRA-style adapters $\{\Delta W_i\}_{i=1}^N$ with ranks $\{r_i\}$, and a request stream where each request carries an adapter id drawn from some (usually skewed, usually non-stationary) popularity distribution.

**Output.** A serving schedule — batching, adapter placement across HBM/host DRAM/disk, kernel choice — that maximises token goodput subject to latency SLOs.

**Decision predicate.** Given hardware $H$ and a workload $(N, \{r_i\}, \text{popularity } \pi, \text{arrival rate})$, what is the maximum achievable goodput $T^\star$, and how far below the single-adapter (dedicated-model) baseline $T_1$ must it fall?

Three variants, different difficulty:

- **Measurement.** Define and report *multi-adapter goodput* in a way that is comparable across systems. Blocked: published speedups are quoted against different baselines with different adapter-popularity distributions.
- **Method.** Build a system that approaches $T_1$ under realistic skew. Largely solved for the homogeneous-rank, HBM-resident case (Punica, S-LoRA, dLoRA); unsolved for heterogeneous ranks, cold adapters, and adapter-count-per-batch $\gg 100$.
- **Theory.** Prove the ceiling: an upper bound on $T^\star$ as a function of distinct adapters per batch $m$, ranks $r_i$, and the hardware ridge point. Open.

The status is **solved-but-impractical** because the published solutions *are* real and do work — but only inside an operating envelope (few distinct adapters per batch, uniform rank, everything resident) that production adapter fleets routinely leave.

## 2. Formal Setting

Base linear layer $W \in \mathbb{R}^{d \times k}$; adapter $i$ contributes $\Delta W_i = B_i A_i$ with $A_i \in \mathbb{R}^{r_i \times k}$, $B_i \in \mathbb{R}^{d \times r_i}$ (Hu et al., ICLR 2022). Batch $\mathcal{B}$ holds $n$ tokens; token $t$ maps to adapter $a(t)$; the *distinct-adapter count* is
$$m = |\{a(t) : t \in \mathcal{B}\}|, \qquad n_i = |\{t : a(t) = i\}|.$$

**Measured quantities.**

- **Goodput** $T$: tokens/s completed within SLO, measured over a $\geq 10^4$-request replay at fixed arrival rate, excluding a warm-up prefix. Not average latency; not tokens/s of a saturated closed loop.
- **Adapter bytes resident**: $M_A = \sum_{i \in \mathcal{R}} 2 L \kappa\, r_i (d + k) \cdot b$ for $L$ layers, $\kappa$ adapted modules per layer, $b$ bytes/element, resident set $\mathcal{R}$.
- **Adapter arithmetic intensity** in a decode step, weight-stationary base, per-adapter weights read once:
$$I_A(\mathcal{B}) = \frac{\sum_t 2\, r_{a(t)} (d+k)}{\sum_{i \in A(\mathcal{B})} r_i (d+k) \, b} \;\approx\; \frac{2}{b}\cdot\frac{n}{m}.$$
The whole problem is in that last ratio: intensity depends on **tokens per distinct adapter**, not on rank, not on $N$.
- **Ridge point** $I^\star = \text{FLOP/s}_{\text{peak}} / \text{BW}_{\text{HBM}}$. Adapter work is bandwidth-bound whenever $I_A < I^\star$.

**Assumptions, and which break.**

| Assumption | Status in practice |
|---|---|
| Uniform rank $r_i \equiv r$ | Violated — fleets mix $r \in \{4,\dots,128\}$; SGMV/BGMV kernels pad or bucket |
| Adapter set stationary | Violated — popularity is non-stationary and long-tailed |
| All adapters HBM-resident | Violated above a few thousand adapters or with large $r$ |
| Adapters applied to same module set | Violated — different targets ($q,v$ vs. all-linear) change $\kappa$ |
| Base model weight-stationary across batch | Holds for decode; prefill is compute-bound and behaves differently |

## 3. State of the Art

**Systems SOTA (established, with public code).**

- **Punica** (Chen et al., MLSys 2024): SGMV — Segmented Gather Matrix–Vector multiply — computes all adapters in one batched kernel with a single base-weight pass. Established: the kernel exists and runs at near-uniform cost in $m$ for small $m$. Reported ~12$\times$ throughput over then-current multi-tenant baselines; that number is a benchmark figure under the authors' popularity distribution, not an ablation of skew.
- **S-LoRA** (Sheng et al., MLSys 2024): unified paging of KV cache and adapter weights, host-DRAM offload, custom MBGMM/MBGMV kernels. Established: thousands of adapters served from one GPU. Claimed 4$\times$ over vLLM-with-LoRA-merged and up to 30$\times$ over HuggingFace PEFT — the 30$\times$ compares against a baseline that re-merges weights per request, so it measures the baseline's defect more than the system's ceiling.
- **dLoRA** (Wu et al., OSDI 2024): dynamic switching between merged and unmerged execution plus cross-replica adapter-aware request migration. Reports up to 57.9$\times$ over vLLM and 26.0$\times$ over S-LoRA on their traces. Benchmark numbers; the merge/unmerge crossover point is reported but not independently reproduced.
- **PetS** (Zhou et al., USENIX ATC 2022): the earliest unified PET-serving framework; established the shared-base / per-task-operator decomposition.
- **LoRAX** (Predibase, 2023–) and vLLM's multi-LoRA path are the deployed implementations; production numbers are not published with controlled workloads.

**Theory SOTA.** Nothing beyond roofline reasoning. No published bound on $T^\star(m, \{r_i\}, \pi)$, no scheduling-optimality result for adapter-aware batching, no competitive-ratio result for adapter cache eviction (which is a weighted caching problem with non-uniform object sizes $\propto r_i$ — a setting where classical results exist but have not been applied here).

## 4. What Is Known

- **Adapter FLOPs are negligible; adapter bytes are not.** For Llama-2-7B, $r=16$ on $q,v$: adapter GEMMs are ~0.1% of base decode FLOPs, but one adapter's weights are ~16.8 MB in bf16 — 0.12% of the 13.5 GB base weight read *per distinct adapter per step*. At $m=512$ the adapter read exceeds the base read. Measured scale: A100-80GB, 2039 GB/s.
- **Merging is throughput-optimal at $m=1$ and catastrophic at $m>1$.** Merged $W + B_iA_i$ costs zero extra bytes; per-request merge costs a full weight rewrite. dLoRA's contribution is exactly this crossover.
- **Adapter memory is small relative to KV cache at moderate $N$.** 1000 adapters at $r=16$, $\kappa=2$, $L=32$ ≈ 16.8 GB — fits alongside a 13.5 GB 7B model in 80 GB, leaving ~50 GB for KV. At $r=64$ all-linear ($\kappa=7$), one adapter is ~235 MB and 1000 adapters do not fit.
- **Continuous batching is a precondition.** Orca (OSDI 2022) iteration-level scheduling and vLLM PagedAttention (SOSP 2023) are assumed by every multi-adapter system; chunked prefill (Sarathi-Serve, OSDI 2024) changes the prefill/decode mix and therefore $n/m$.
- **Rank matters little for quality over a wide range**, which is why fleets are rank-heterogeneous at all: LoRA Land (Predibase, 2024) fine-tuned 310 adapters, mostly low rank, on 7B bases.

## 5. What Is Not Known

- **Theoretically open.** No upper bound on goodput as a function of $(m, \{r_i\})$ for a given ridge point. No proof that any online adapter-placement policy is within a constant factor of the offline optimum under non-stationary skew. No optimality result for rank-heterogeneous segment packing.
- **Empirically open.** The scaling curve $T(m)$ with $n$ held fixed — swept from $m=1$ to $m=n$, on one hardware platform, one base model, one system — has not been published. Every paper reports a point, not a curve. Likewise the rank-heterogeneity penalty: how much does mixing $r\in\{8,64\}$ in one batch cost versus bucketing into two batches?
- **Methodologically blocked.** There is no standard multi-adapter serving benchmark. Adapter popularity distributions are chosen per paper (uniform, Zipf with unstated $\alpha$, trace-derived from unreleased traces), so cross-system speedups are not comparable. Without a fixed $\pi$, "throughput" names a quantity that differs between papers.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement compounded by a bandwidth cliff that is invisible in the reported metric.**

Multi-adapter throughput is a function of $n/m$, but $m$ is an *emergent* property of the arrival process, the batching policy, and the SLO — not a knob the experimenter sets. Two systems on "the same workload" can run at different $m$ because their schedulers admit different requests. A 26$\times$ speedup can therefore be a scheduling artifact that lowered $m$, not a kernel improvement. Nobody publishes $m$.

Secondary: rank heterogeneity makes the adapter GEMM a ragged-segment problem. Padding to $\max_i r_i$ wastes bandwidth proportional to rank variance; bucketing by rank raises $m$ per bucket-batch. Both are bad, and which is less bad is workload-dependent — that is a non-identifiability, not a tuning gap.

## 7. Current Research (as of 2026)

- **Kernel side:** FlashInfer (Ye et al., MLSys 2025) is absorbing adapter-aware batched GEMM into a general attention/GEMM engine; grouped-GEMM paths shared with MoE routing are the natural home for SGMV. *(frontier — verify)* NVIDIA and vLLM contributors are converging multi-LoRA and MoE expert dispatch onto the same grouped-GEMM primitive.
- **Placement side:** CaraServe (Chen, Lin, Zhang, Wu, 2024) uses CPU-assisted prefill for cold adapters to hide load latency; rank-aware scheduling is its second contribution.
- **Cluster side:** dLoRA's adapter-affinity migration (Peking University — Xin Jin's group) is the main line on multi-replica placement.
- **Quality side:** whether adapters can be *unified* — merged into a shared low-rank basis so $m$ collapses — is an active PEFT question that would dissolve the systems problem if it worked. No result yet shows basis-shared adapters matching per-task LoRA at fleet scale. *(frontier — verify)*

## 8. Concrete Next Experiment

**The $T(m)$ sweep.** This is the missing curve.

- **Scale.** Llama-2-7B (or Llama-3-8B) bf16 on one A100-80GB. Adapter store $N = 1024$, all $r=16$ on $q,v$ (16.8 MB each, 17.2 GB total, fully HBM-resident — this removes offload as a confound). Fixed decode batch $n = 512$ tokens/step, enforced by a closed-loop harness. Sweep $m \in \{1, 2, 4, 8, 16, 32, 64, 128, 256, 512\}$ by *constructing* the batch, not by sampling a popularity distribution. 3 runs, 2000 steps each after warm-up.
- **Control arm.** $m=1$ with the adapter pre-merged into $W$ — the dedicated-model ceiling $T_1$. Second control: $m=1$ unmerged, isolating SGMV's fixed cost from its $m$-scaling.
- **Deciding number.** $m_{1/2}$ — the distinct-adapter count at which decode throughput falls to $0.5\,T_1$. The roofline prediction is $m_{1/2} \approx M_{\text{base}} / M_{\text{adapter}} = 13.5\,\text{GB} / 16.8\,\text{MB} \approx 800$. If measured $m_{1/2}$ is within 2$\times$ of 800, multi-adapter serving is bandwidth-bound and the theory is a one-line roofline. If $m_{1/2} < 100$, the loss is kernel-launch and gather overhead, and there is real systems work left. Current systems' published operating points sit at $m \lesssim 64$, so the answer is not currently known either way.

**Extension (one extra day):** repeat with ranks drawn from $\{8,16,32,64\}$ uniformly at $m=32$, versus rank-bucketed batches. Report the padding-waste fraction.

## 9. Key References

- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Zhe Zhou, Xuechao Wei, Jiejing Zhang, Guangyu Sun. *PetS: A Unified Framework for Parameter-Efficient Transformers Serving.* USENIX ATC, 2022.
- **[SOTA]** Lequn Chen, Zihao Ye, Yongji Wu, Danyang Zhuo, Luis Ceze, Arvind Krishnamurthy. *Punica: Multi-Tenant LoRA Serving.* MLSys, 2024. — arXiv:2310.18547
- **[SOTA]** Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica. *S-LoRA: Serving Thousands of Concurrent LoRA Adapters.* MLSys, 2024. — arXiv:2311.03285
- **[SOTA]** Bingyang Wu, Ruidong Zhu, Zili Zhang, Peng Sun, Xuanzhe Liu, Xin Jin. *dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving.* USENIX OSDI, 2024.
- **[Systems context]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Systems context]** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* USENIX OSDI, 2022.
- **[Systems context]** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* USENIX OSDI, 2024.
- **[Related]** Shaoyuan Chen, Yutong Lin, Mingxing Zhang, Yongwei Wu. *CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference.* Preprint, 2024.
- **[Survey]** Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR, 2024. — arXiv:2403.14608

## 10. Worked Example

Llama-2-7B, bf16, A100-80GB (2039 GB/s HBM, 312 TFLOP/s bf16 dense; ridge point $I^\star \approx 153$ FLOP/byte). LoRA $r=16$ on $q,v$ only: $L=32$, $\kappa=2$, $d=k=4096$.

**Adapter size.** $2 \times 32 \times 2 \times 16 \times 4096 \times 2\,\text{B} = 16.8$ MB.

**One decode step, $n=512$ tokens.**

```
base weight read      13.5 GB   ->  6.62 ms at 2039 GB/s
base FLOPs            2*512*7e9 = 7.17 TFLOP -> 22.9 ms?  no:
                      decode is memory-bound; 7.17 TFLOP / 312e12 = 0.023 ms
adapter FLOPs         2*512*16*8192*64 = 8.6 GFLOP -> 0.028 us
adapter bytes         m * 16.8 MB
```

| $m$ | adapter bytes | adapter time | step time | throughput vs. $T_1$ |
|---|---|---|---|---|
| 1 | 16.8 MB | 0.008 ms | 6.63 ms | 1.00 |
| 32 | 538 MB | 0.26 ms | 6.88 ms | 0.96 |
| 128 | 2.15 GB | 1.05 ms | 7.67 ms | 0.86 |
| 512 | 8.6 GB | 4.22 ms | 10.84 ms | 0.61 |

**Where the obstruction becomes visible.** At $m=512$ the adapter arithmetic intensity is $I_A = 2 \cdot (512/512)/2 = 1$ FLOP/byte against a ridge point of 153 — off the roofline by 150$\times$. The adapters contribute **0.12%** of the step's FLOPs and **39%** of its bytes. No kernel improvement can fix this; SGMV already reads each adapter exactly once. The only levers are lowering $m$ (adapter-aware batching, which trades against latency SLOs) or shrinking $r$ (which trades against quality).

Now the measurement problem. A system that reports "10.8 ms/step, 512 tokens" alongside another reporting "6.9 ms/step, 512 tokens" looks 1.6$\times$ slower. It may simply have been handed $m=512$ instead of $m=32$ by an arrival process the paper does not describe. Every published multi-adapter speedup is compatible with this explanation, because none of them report $m$. That is why the problem is solved-but-impractical rather than solved: the kernels are right, the envelope is narrow, and the reported numbers do not tell you where the envelope ends.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*