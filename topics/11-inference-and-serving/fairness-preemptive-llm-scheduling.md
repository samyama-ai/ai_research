---
id: 11-inference-and-serving/fairness-preemptive-llm-scheduling
title: "Fairness Across Requests in Preemptive LLM Schedulers"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Fairness Across Requests in Preemptive LLM Schedulers

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/fairness-preemptive-llm-scheduling` · **Status:** open

## 1. Problem Statement

An LLM serving engine runs continuous batching: at each iteration it picks a subset of live requests to advance by one step, and may **preempt** a running request by evicting its KV cache (recomputed or swapped on resume). Preemption is what makes the scheduler expressive — and what lets it starve a client indefinitely.

- **Input.** A stream of requests $r_i = (c_i, a_i, p_i)$ — client $c_i$, arrival $a_i$, prompt length $p_i$ — with output length $o_i$ revealed only when the request emits EOS.
- **Output.** A per-iteration schedule: which requests run, which are preempted, which are admitted.
- **Objective.** Bound the *service disparity* between backlogged clients while keeping throughput and tail latency within a stated factor of the throughput-optimal schedule.

Three variants, of very different difficulty:

- **Measurement.** What is "service"? Requests differ in prompt length, output length, prefix-cache hit rate, and LoRA adapter. Every candidate unit (requests, tokens, GPU-seconds, weighted tokens) gives a different fairness verdict on the same trace. This variant is *methodologically blocked*.
- **Method.** Build a scheduler with a proven bound on disparity under a *given* cost function. **Solved** for the token-cost model by VTC (Sheng et al., OSDI 2024).
- **Theory.** Characterise the achievable fairness/throughput/tail-latency frontier when output length is unknown and preemption is not free. **Open.**

Solving it means: a cost function that is measurable online and invariant to the scheduler's own choices, plus an algorithm with a disparity bound under that function, plus a measured throughput loss.

## 2. Formal Setting

Time is discretised into engine iterations $t = 1, 2, \dots$. Let $S_t \subseteq R$ be the running set at iteration $t$, constrained by KV memory:

$$\sum_{i \in S_t} \big(p_i + d_i(t)\big)\, \kappa \;\le\; M,$$

where $d_i(t)$ is tokens decoded so far, $\kappa$ is bytes of KV per token (measured: $\kappa = 2 L H B$ for $L$ layers, hidden size $H$, $B$ bytes/element — 800 KiB/token for Llama-2-13B in fp16), and $M$ is free HBM after weights and activations.

**Service.** Client $c$'s accumulated service by time $T$:

$$U_c(T) \;=\; \sum_{i:\,c_i=c} \Big( w_p \cdot p_i \cdot \mathbb{1}[\text{prefilled by } T] \;+\; w_o \cdot d_i(T) \Big).$$

Measured by counting tokens at the engine, not by timing. $w_p, w_o$ are the free parameters — everything below turns on them.

**Fairness predicate (max–min / WFQ style).** For any two clients $c, c'$ backlogged throughout $[T_1, T_2]$:

$$\big| U_c(T_2) - U_c(T_1) - (U_{c'}(T_2) - U_{c'}(T_1)) \big| \;\le\; \Delta,$$

with $\Delta$ constant in $T_2 - T_1$. This is the Parekh–Gallager GPS guarantee transposed to tokens.

**Preemption cost.** Resuming a preempted request costs $\min(\text{recompute}, \text{swap})$: recompute $\approx 2 N (p_i + d_i)$ FLOPs for an $N$-parameter model; swap $\approx 2 \kappa (p_i + d_i) / \beta$ seconds at PCIe bandwidth $\beta \approx 25$ GB/s effective on Gen4 ×16.

**Throughput price.** $\rho = \text{tok/s}(\text{fair}) / \text{tok/s}(\text{throughput-optimal})$, measured on the same trace and hardware.

**Assumptions, and which are violated:**

| Assumption | Status |
|---|---|
| Per-token cost is constant, so token counts proxy for GPU time | **Violated.** Prefill is compute-bound, decode memory-bound; the ratio moves with batch size (§10). |
| Prefill cost is linear in $p_i$ | **Violated** above ~4K tokens, where attention's $O(p^2)$ term bites. |
| Requests are independent | **Violated** by prefix caching: a shared system prompt makes one client's prefill free because another paid for it. |
| Output length unknown (non-clairvoyance) | Holds; length predictors are noisy, not oracles. |
| Single homogeneous replica | **Violated** under disaggregated prefill/decode and multi-replica migration. |

## 3. State of the Art

**Established.**
- **VTC** — Sheng, Cao, Li, Zhu, Li, Zhuo, Gonzalez, Stoica, *Fairness in Serving Large Language Models*, OSDI 2024. Defines the token-based fairness predicate above, proves VTC's service difference between backlogged clients is bounded by a constant proportional to the largest per-request token cost, and shows FCFS, requests-per-minute limiting, and least-attained-service all admit unbounded divergence. Measured on Llama-2-7B/13B on A100s. This is the reference result.
- **Chunked prefill** — Agrawal et al., *Taming the Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024. Stall-free batching removes the head-of-line block a long prefill imposes on decodes; reported up to 2.6× serving capacity at SLO for Mistral-7B on an A100 and 5.6× for Falcon-180B on 8×A100. Fairness is a side effect, not measured as such.
- **Preemption mechanism** — Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM), SOSP 2023: recompute-or-swap preemption, 2–4× throughput over Orca (Yu et al., OSDI 2022) at matched latency.

**Claimed but unablated for fairness.**
- **FastServe** (Wu et al., arXiv:2305.05920): skip-join MLFQ with proactive KV swapping; reports up to 5.1× lower average job completion time vs. vLLM. Benchmark number only — no per-client disparity is reported, and MLFQ demotion is exactly the mechanism that starves long generations.
- **Andes** (Chung et al., arXiv:2404.16283): QoE-aware token-delivery scheduling for streaming; preempts requests whose token buffer is ahead of reading speed. QoE is per-request, not cross-client; no disparity bound.
- **Llumnix** (Sun et al., OSDI 2024): live migration of requests across replicas for load and priority. Reports large P99 latency reductions; migration changes the cost accounting VTC assumes and no fairness bound is given for the multi-replica case.
- **QLM** (Patke et al., VLDB 2025): queue-management with request-group SLOs. SLO attainment, not max–min fairness.

**Theory SOTA (from classical scheduling, directly applicable).** Non-clairvoyant total flow time admits no $o(n^{1/3})$-competitive deterministic and no $o(\log n)$-competitive randomized algorithm (Motwani, Phillips, Torng, TCS 1994). With $(1+\epsilon)$ resource augmentation, SETF is $O(1+1/\epsilon)$-competitive (Kalyanasundaram & Pruhs, JACM 2000). Weighted flow time admits no $O(1)$-competitive online algorithm (Bansal & Chan, SODA 2009). None of these model preemption cost that grows with attained service — the defining feature of KV-cache eviction.

## 4. What Is Known

- VTC's bound is real and reproduced in the paper's own artifact: under a 2-client trace where one client sends 8× more requests, FCFS gives that client ~8× the tokens; VTC holds the ratio at 1 with disparity bounded by roughly one request's token cost. Scale: Llama-2-13B, single A100-80GB, ~100–200 req/s synthetic and ShareGPT traces.
- The weight choice $w_p=1, w_o=2$ in VTC was fit empirically on A100-class hardware. It is a fitted constant, not derived.
- Preemption is not rare. On ShareGPT at high load, vLLM preempts a double-digit percentage of requests; recomputation is cheaper than swapping for short sequences and the crossover moves with $p_i + d_i$ and PCIe generation.
- Chunked prefill converts head-of-line blocking into a uniform per-iteration tax: the largest decode stall drops from the length of a full prefill (~1.1 s for 8K tokens on a 13B model at 60% MFU) to the chunk time (~30–60 ms at 512-token chunks).
- KV footprint dominates admission: 800 KiB/token for 13B fp16 means a 40 GB free budget holds only ~50K tokens — about 6 requests at 8K context. Fairness at that batch width is a discrete, not fluid, allocation.

## 5. What Is Not Known

- **Methodologically blocked.** There is no scheduler-invariant definition of "service". Token cost in GPU-seconds depends on batch composition, which the scheduler chooses (§10). Any fairness certificate stated in tokens is conditional on weights that the scheduler's own behaviour moves. No published work measures fairness in GPU-seconds directly, because per-request GPU-seconds inside a fused batch are not separable.
- **Theoretically open.** No lower bound on the fairness/throughput frontier under non-clairvoyant output length *and* attained-service-proportional preemption cost. Whether $\Delta = O(1)$ disparity is achievable at $\rho \ge 1-\epsilon$, or whether the two trade off strictly, has no proof either way.
- **Empirically open.** The throughput price of VTC-style fairness has not been measured at production scale (multi-node, disaggregated prefill/decode, prefix caching on, 10+ tenants, real diurnal traces). Every published number is single-replica with 2–8 synthetic clients.
- **Empirically open.** Fairness under prefix sharing: whether a cache-hitting client should be charged for tokens it did not compute is undecided, and no scheduler implements either policy with a bound.

## 6. Why It Is Hard

**The obstruction is non-identifiability of the cost metric, made worse by endogeneity.** The quantity fairness should equalise is GPU-seconds. GPU-seconds per request are not measurable inside a continuous batch — one fused kernel serves 64 requests and the marginal cost of each depends on the other 63. So the field substitutes weighted token counts. But the weight that makes tokens proportional to GPU-seconds is a function of batch size, and batch size is chosen by the scheduler being evaluated. The measurement is a function of the treatment. §10 shows the implied $w_o/w_p$ swinging from 2 to 16 on identical hardware and model.

Secondary: preemption cost scales with attained service, so the cheapest request to preempt is the one closest to being finished — the opposite of what fairness wants, and outside every classical competitive-analysis model.

## 7. Current Research (as of 2026)

- **Berkeley Sky Computing / vLLM** — VTC's authors; follow-on work on multi-tenant and multi-LoRA fairness. Whether a VTC variant ships in vLLM's V1 scheduler is *(frontier — verify)*.
- **Microsoft Research** — fairness-under-diverse-applications work extending token-cost accounting to heterogeneous app mixes; treat specific claims as *(frontier — verify)*.
- **CMU / Michigan (Andes lineage)** — QoE-shaped preemption for streaming and agentic workloads, where a "request" is a multi-turn session and the fairness unit is arguably the session, not the request. *(frontier — verify)*
- **Disaggregated serving** (DistServe, Zhong et al., OSDI 2024, and successors) — splits prefill and decode onto separate pools, which makes $w_p$ and $w_o$ prices of *different* resources. No fairness formulation exists for the two-pool case. Genuinely open.
- **Agentic/long-horizon serving** — fairness across agents that issue bursts of dependent requests; the backlogged-client assumption underlying WFQ-style bounds does not hold for dependency-blocked agents.

## 8. Concrete Next Experiment

**Question.** Does token-based fairness (VTC) deliver GPU-second fairness, and what does it cost in throughput?

- **Scale.** Llama-3.1-8B on 1×H100-80GB, plus a 70B run on 4×H100 for the tensor-parallel check. 8 tenants, 4 hours of replayed ShareGPT + LMSYS-Chat-1M traffic at 85% of saturation. Tenant mix deliberately adversarial: 2 long-prompt/short-output (RAG, 8K in / 200 out), 2 short-prompt/long-output (200 in / 2K out), 4 mixed. Prefix caching on and off as a crossed factor.
- **Arms.** (a) VTC with published $w_p=1, w_o=2$. (b) VTC with $w$ re-fit per hour from measured hardware counters. (c) FCFS. (d) Throughput-optimal control arm: vLLM default continuous batching with no fairness mechanism, which sets $\rho = 1$.
- **Ground truth for GPU-seconds.** Shapley-style attribution by ablation: for a sampled 1% of iterations, re-run the identical batch with request $i$ removed and attribute the time delta to $i$. Expensive but exact enough to bound the error.
- **The deciding number.** $\max_{c,c'} |G_c - G_{c'}| / \bar{G}$ — the largest pairwise gap in attributed GPU-seconds per backlogged client, normalised by the mean, over the run. If VTC-(a) keeps this below 0.10 while token disparity is bounded, token fairness is a sound proxy and the problem's measurement variant closes. If it exceeds 0.30 — the prediction, given the 8× weight swing in §10 — token-based fairness is certifying the wrong quantity, and the field needs a cost model, not a better queue. Report $\rho$ for each arm alongside it.

## 9. Key References

- **[SOTA]** Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica. *Fairness in Serving Large Language Models.* OSDI 2024. — arXiv:2401.00588
- **[Foundational]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. — arXiv:2309.06180
- **[Foundational]** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022.
- **[SOTA]** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, Ramachandran Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI 2024. — arXiv:2403.02310
- **[SOTA]** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI 2024. — arXiv:2401.09670
- **[SOTA]** Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, Wei Lin. *Llumnix: Dynamic Scheduling for Large Language Model Serving.* OSDI 2024. — arXiv:2406.03243
- **[Claimed]** Bingyang Wu, Yinmin Zhong, Zili Zhang, Gang Huang, Xuanzhe Liu, Xin Jin. *Fast Distributed Inference Serving for Large Language Models.* arXiv, 2023. — arXiv:2305.05920
- **[Claimed]** Jiachen Liu, Zhiyu Wu, Jae-Won Chung, Fan Lai, Myungjin Lee, Mosharaf Chowdhury. *Andes: Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services.* arXiv, 2024. — arXiv:2404.16283
- **[Foundational]** Abhay K. Parekh, Robert G. Gallager. *A Generalized Processor Sharing Approach to Flow Control in Integrated Services Networks: The Single-Node Case.* IEEE/ACM Transactions on Networking, 1993.
- **[Foundational]** Alan Demers, Srinivasan Keshav, Scott Shenker. *Analysis and Simulation of a Fair Queueing Algorithm.* SIGCOMM 1989.
- **[Foundational]** Rajeev Motwani, Steven Phillips, Eric Torng. *Non-clairvoyant Scheduling.* Theoretical Computer Science, 1994.
- **[Foundational]** Bala Kalyanasundaram, Kirk Pruhs. *Speed Is as Powerful as Clairvoyance.* Journal of the ACM, 2000.
- **[Foundational]** Nikhil Bansal, Ho-Leung Chan. *Weighted Flow Time Does Not Admit O(1)-Competitive Algorithms.* SODA 2009.
- **[Foundational]** Ali Ghodsi, Matei Zaharia, Benjamin Hindman, Andy Konwinski, Scott Shenker, Ion Stoica. *Dominant Resource Fairness: Fair Allocation of Multiple Resource Types.* NSDI 2011.

## 10. Worked Example

Llama-2-13B, fp16, one A100-80GB. Two clients, both permanently backlogged.

- **Client A (RAG):** 8192-token prompts, 200-token outputs.
- **Client B (chat):** 200-token prompts, 2000-token outputs.

**Measured costs.**

```
Prefill, 8192 tok:  2 * 13e9 * 8192      = 2.13e14 FLOP
                    / (312 TFLOP/s * 0.60) = 1.14 s
                    -> 139 us per prefill token

Decode step:        26 GB weights / 1.5 TB/s = 17.3 ms  (memory-bound,
                    near-independent of batch size)
  at batch 64:      17.3 ms / 64  = 270 us per decode token
  at batch  8:      17.3 ms /  8  = 2163 us per decode token
```

**The implied fair weight.**

$$\frac{w_o}{w_p} = \frac{\text{cost/decode token}}{\text{cost/prefill token}} = \begin{cases} 270/139 \approx 1.9 & \text{batch } 64\\ 2163/139 \approx 15.6 & \text{batch } 8\end{cases}$$

VTC's published $w_o/w_p = 2$ is correct at batch 64 and wrong by 8× at batch 8.

**Consequence.** Suppose the scheduler equalises $U_A = U_B$ with $w_p{=}1, w_o{=}2$ over a window where each client completes $n$ requests:

$$U_A = n(8192 + 2\cdot200) = 8592n, \qquad U_B = n(200 + 2\cdot2000) = 4200n.$$

To equalise, the scheduler serves B about $8592/4200 = 2.05$ requests for every one of A's. Now price it in GPU-seconds. At batch 64: A costs $8192(139) + 200(270) = 1.19$ s/req; B costs $200(139) + 2000(270) = 0.568$ s/req. Ratio $2.05 \times 0.568 = 1.16$ s vs A's $1.19$ s — within 3%. Token fairness works.

But B's long decodes and A's large KV footprint push the running batch to ~8 when A is admitted (6 requests × 8192 tokens ≈ 40 GB of KV). At batch 8: A costs $8192(139) + 200(2163) = 1.57$ s; B costs $200(139) + 2000(2163) = 4.35$ s. Now B receives $2.05 \times 4.35 = 8.9$ GPU-seconds for A's 1.57 — a **5.7× disparity**, while VTC's certificate reports zero disparity, because in tokens the two are exactly equal.

**The obstruction, visible.** The scheduler's own admission decision — how many of A's 8K-context requests to hold resident — sets the batch size, which sets the weight, which sets what "equal" means. There is no fixed $w$ that is correct across the operating range, and the scheduler cannot be evaluated against a metric it moves. That is why the method variant is solved and the measurement variant is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*