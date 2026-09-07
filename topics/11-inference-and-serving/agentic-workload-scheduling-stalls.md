---
id: 11-inference-and-serving/agentic-workload-scheduling-stalls
title: "Agentic Workload Scheduling With Tool-Call Stalls"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Agentic Workload Scheduling With Tool-Call Stalls

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/agentic-workload-scheduling-stalls` · **Status:** open

## 1. Problem Statement

An agentic request is not one generation. It is a sequence of generation segments separated by **stalls**: the model emits a tool call, the engine hands off to an external executor (search, code sandbox, database, another model, a human), and waits. Stall durations span roughly $10^{-1}$ s to $10^{2}$ s and are not known when the segment starts.

The scheduler must decide, at each stall: **evict** the request's KV cache (freeing memory, paying recompute on resume), **swap** it to host memory (paying PCIe bandwidth), or **retain** it (paying occupancy that blocks admission of other requests). It must also decide the batch composition and preemption order across a mixed population of stalling and non-stalling requests.

Three variants, commonly conflated:

- **Measurement.** Define an SLO for multi-turn agentic requests. Per-token TPOT and per-request TTFT are both wrong: end-to-end latency is dominated by stall time the engine does not control, and normalizing it away hides the scheduling decision. No agreed metric exists.
- **Method.** Given a stall-duration predictor and a memory budget, find a policy minimizing mean weighted flow time. Runnable today; largely unrun at production trace scale because the traces are proprietary.
- **Theory.** Characterize the competitive ratio of online policies for jobs with unknown, alternating service/stall phases and a hard shared-memory constraint. Open.

**Solved** means: a policy with a stated competitive or regret bound *and* an independent reproduction showing goodput gain over a strong retain-everything baseline on a public agentic trace, at fixed hardware and fixed task success rate.

## 2. Formal Setting

Requests $r_i$ arrive at times $a_i$. Each is a chain of $K_i$ segments; segment $k$ has decode length $\ell_{ik}$ tokens and is followed by stall $s_{ik} \ge 0$ seconds. $K_i$, $\ell_{ik}$, $s_{ik}$ are all revealed only on completion.

**Measured as:** $\ell_{ik}$ = tokens emitted between segment start and the stop token that opens a tool call (engine counter). $s_{ik}$ = wall clock from the engine dispatching the tool result request to its return, including client-side network — measured at the engine boundary, not the tool boundary, because retry and queueing at the client are part of the occupancy cost.

KV footprint of request $i$ after $t$ tokens on a model with $L$ layers, $H_{kv}$ KV heads, head dim $d$, dtype bytes $b$:
$$m_i(t) = 2\,L\,H_{kv}\,d\,b\,t .$$
Total device KV budget $M$. Feasibility at time $t$: $\sum_{i \in \mathcal{R}(t)} m_i(t) \le M$, where $\mathcal{R}(t)$ is the resident set.

Objective — **agentic goodput**, tasks completed per GPU-second subject to a tail constraint:
$$G = \frac{|\{i : C_i - a_i - S_i \le \tau\}|}{T\cdot N_{\text{gpu}}}, \qquad S_i = \sum_k s_{ik},$$
with $C_i$ the completion time. Subtracting $S_i$ gives **engine-attributable latency**: the part the scheduler can affect. This is the key definitional choice and it is not standard.

Cost of resuming an evicted request: $\min(c_{\text{recompute}}\,t,\; m_i(t)/B_{\text{pcie}})$ where $c_{\text{recompute}}$ is prefill time per token (with prefix-cache hits, far less). The decision is a comparison of $s_{ik}$ against the break-even stall
$$s^{*}_{ik} = \frac{\text{resume cost}}{\text{marginal value of }m_i(t)\text{ to the rest of the batch}},$$
and the numerator is known while the denominator and $s_{ik}$ are not.

**Assumptions known violated in practice:** (i) stalls independent of content — false, long-running tools correlate with long prior reasoning segments; (ii) stalls independent across requests — false, a slow upstream API stalls every agent hitting it simultaneously, producing correlated bursts; (iii) Poisson arrivals — false for agentic traffic, which is bursty and self-similar; (iv) $m_i$ additive across requests — false with prefix sharing, where a shared system prompt or subtree is charged once (RadixAttention, SGLang, NeurIPS 2024).

## 3. State of the Art

**Systems, established (independently used and reproduced):**
- Continuous/iteration-level batching — Orca (Yu et al., OSDI 2022).
- Paged KV with preemption by swap or recompute — vLLM/PagedAttention (Kwon et al., SOSP 2023): 2–4× throughput over Orca at matched latency, 13B on A100.
- Chunked prefill to bound decode stalls — Sarathi-Serve (Agrawal et al., OSDI 2024).
- Prefill/decode disaggregation — DistServe (Zhong et al., OSDI 2024), Splitwise (Patel et al., ISCA 2024).
- Prefix-sharing cache — SGLang RadixAttention (Zheng et al., NeurIPS 2024).

None of these is stall-aware. All treat a request as a single monotone generation.

**Stall-aware, claimed but under-ablated:**
- **InferCept** (Abhyankar et al., ICML 2024) is the first system to make the retain/swap/discard choice explicitly per interception, minimizing wasted GPU time. Headline throughput gains are single-system benchmark numbers; the ablation isolating the *policy* from the improved memory manager is not public.
- **Parrot** (Lin et al., OSDI 2024) exposes application-level dependency structure ("semantic variables") to the scheduler, enabling DAG-aware ordering. Gains are reported on hand-built LLM applications, not on a released production trace.
- **Autellix** (Luo et al., 2025) schedules whole agent *programs* rather than requests, using non-clairvoyant program-level priorities. Reported large throughput gains over vLLM on agentic workloads *(frontier — verify; no independent reproduction known)*.
- **Preble** (Srivatsa et al., 2024) schedules across replicas for prefix locality — relevant because eviction destroys the locality it exploits.

**Theory SOTA is separate and much weaker.** Nothing in the queueing literature covers the exact model. The nearest results: SRPT optimality for mean flow time with known sizes (Schrage, 1968); Gittins-index optimality for M/G/1 with unknown sizes; SOAP as a unified analysis of age-based policies (Scully, Harchol-Balter, Scheller-Wolf, SIGMETRICS 2018). None handles a hard shared-capacity constraint whose per-job demand *grows monotonically with service received* — the defining feature here.

## 4. What Is Known

- **Non-clairvoyance is provably expensive.** For mean flow time, every deterministic online algorithm is $\Omega(n^{1/3})$-competitive and every randomized one $\Omega(\log n)$-competitive (Motwani, Phillips, Torng, *Nonclairvoyant scheduling*, TCS 1994). Stall durations make agentic scheduling strictly non-clairvoyant.
- **Output length is partly predictable.** Perception-based length prediction gives useful ordering signal (Zheng et al., NeurIPS 2023); learning-to-rank on relative order, rather than absolute length, closes much of the gap to an oracle SJF scheduler (Fu et al., 2024). Both measured on single-turn chat, 7B–13B scale — *not* on tool-call chains.
- **Preemption is cheap when prefixes are shared, expensive otherwise.** Recompute cost falls with prefix-cache hit rate; agent loops with a fixed system prompt and growing scratchpad have high hit rates on the prefix and zero on the tail.
- **KV memory, not FLOPs, is the binding constraint at agentic context lengths.** For a 70B model with GQA at 8 KV heads ($L{=}80$, $d{=}128$, fp16), $m_i \approx 0.32$ MB/token; a 100 K-token agent context is ~32 GB — nearly half an 80 GB device for one stalled request.
- **Public agentic benchmarks measure task success, not serving cost.** $\tau$-bench (Yao et al., 2024) and SWE-bench report pass rates; neither publishes stall-duration distributions or arrival traces.

## 5. What Is Not Known

- **Theoretically open.** No competitive-ratio result for online scheduling of alternating service/stall jobs under a shared capacity constraint with service-monotone demand. It is not even known whether a constant-competitive algorithm exists with $O(1)$ resource augmentation in memory.
- **Empirically open.** Whether stall-aware eviction beats a well-tuned oblivious LRU-by-arrival baseline by a margin that survives fixed hardware and fixed task success. The experiment is runnable on 8×H100; the blocker is trace availability, not compute.
- **Methodologically blocked.** The metric. Subtracting $S_i$ (§2) assumes engine and tool time are separable, but eviction changes resume time, which changes when the *next* tool call fires, which changes tool-side queueing. Under contention the two are coupled and no accepted decomposition exists.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent ground truth**. Every published stall-aware system changes the memory manager and the policy together, so the reported gain is not attributable. And the ground truth needed to evaluate a policy — a real distribution of $(\ell_{ik}, s_{ik}, K_i)$ under real arrivals — exists only inside a handful of companies. Synthetic stalls (exponential, fixed mean) destroy the property that makes the problem hard: heavy tails and cross-request correlation. A policy tuned on exponential stalls is tuned on a distribution where the residual-life predictor is constant, i.e. where prediction is worthless and the policy question vanishes.

Second obstruction: **non-identifiability of the objective**. Retaining a stalled request lowers its own latency and raises everyone else's. Without a stated exchange rate between per-request tail latency and system goodput, two policies can both be optimal for different unstated weightings, and papers pick the weighting after seeing the results.

## 7. Current Research (as of 2026)

- **Program-level scheduling** — treating the agent trajectory, not the HTTP request, as the schedulable unit (Autellix; Parrot lineage). Berkeley Sky Computing / UCSD-adjacent groups *(frontier — verify)*.
- **KV offload hierarchies** — Mooncake (Qin et al., FAST 2025) and successors make swap cheap enough that eviction becomes the default rather than a fallback; this may dissolve the policy question at the cost of DRAM and NIC bandwidth *(frontier — verify)*.
- **Stall-duration prediction from partial output** — extending length-prediction work to predict tool latency from the emitted call itself (a `sleep 30` and a cache-hit key lookup are trivially distinguishable from the argument text). Little published.
- **Speculative continuation across stalls** — decode the likely post-tool continuation while waiting, discard on mismatch. Attractive on paper; correctness under tool side effects is unresolved.

## 8. Concrete Next Experiment

**Question:** does a stall-aware eviction policy beat an oblivious one at fixed hardware and fixed task success?

**Scale.** One 8×H100 node, Llama-3.3-70B-Instruct (or equivalent open 70B), vLLM ≥0.8 as the substrate so the memory manager is held constant across arms. Workload: 2,000 $\tau$-bench-retail and SWE-bench-verified episodes replayed as a Poisson-modulated arrival process at 4 load levels (0.5×, 0.8×, 0.95×, 1.1× of saturating rate). Record real tool latencies; do not synthesize them. Runtime ≈ 30 GPU-hours per arm.

**Arms.**
1. *Control:* retain-all until memory pressure, then vLLM default preemption (recompute, FCFS victim).
2. Oblivious threshold: evict any request stalled > 2 s.
3. Predictive: evict if $\hat{s}_{ik} > s^{*}_{ik}$, with $\hat{s}$ from a small classifier on the tool name plus arguments, trained on a held-out 20% of episodes.
4. Oracle: true $s_{ik}$ replayed, giving the achievable ceiling.

**Deciding number.** Agentic goodput $G$ (§2) at 0.95× saturating load, with the P95 engine-attributable latency constraint $\tau = 30$ s, and task success rate required to stay within ±1 pp of control. **Arm 3 must reach ≥ 60% of the (arm 4 − arm 1) gap to justify prediction; if arm 2 already reaches 90% of it, the prediction line of work is dead.** Report the gap in absolute goodput, not ratio.

## 9. Key References

- **[Foundational]** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI, 2022.
- **[Foundational]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Foundational]** Rajeev Motwani, Steven Phillips, Eric Torng. *Nonclairvoyant Scheduling.* Theoretical Computer Science, 1994.
- **[SOTA]** Reyna Abhyankar, Zijian He, Vikranth Srivatsa, Hao Zhang, Yiying Zhang. *InferCept: Efficient Intercept Support for Augmented Large Language Model Inference.* ICML, 2024.
- **[SOTA]** Chaofan Lin, Zhenhua Han, Chengruidong Zhang, Yuqing Yang, Fan Yang, Chen Chen, Lili Qiu. *Parrot: Efficient Serving of LLM-based Applications with Semantic Variable.* OSDI, 2024.
- **[SOTA]** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024.
- **[SOTA]** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI, 2024.
- **[SOTA]** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS, 2024.
- **[Theory]** Ziv Scully, Mor Harchol-Balter, Alan Scheller-Wolf. *SOAP: One Clean Analysis of All Age-Based Scheduling Policies.* ACM SIGMETRICS, 2018.
- **[Related]** Zangwei Zheng, Xiaozhe Ren, Fuzhao Xue, Yang Luo, Xin Jiang, Yang You. *Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline.* NeurIPS, 2023.
- **[Benchmark]** Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024.
- **[Systems]** Ruoyu Qin, Zheming Li, Weiran He, et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* USENIX FAST, 2025.

## 10. Worked Example

Llama-3.3-70B, fp16, $L=80$, $H_{kv}=8$, $d=128$: $m_i(t) = 2 \cdot 80 \cdot 8 \cdot 128 \cdot 2 \cdot t = 327{,}680$ B/token $\approx 0.31$ MB/token.

An agent at turn 6 holds 40,000 tokens $\Rightarrow$ **12.5 GB**. On 8×H100 with tensor parallelism, KV budget after weights is roughly $8\cdot80 - 140 \approx 500$ GB, so 40 such agents saturate the cache.

It stalls on a web-search call. Two policies:

- **Retain.** Cost = 12.5 GB held for $s$ seconds. At a marginal admission value of ~1 additional concurrent decode per 12.5 GB, and ~35 tok/s/request, the opportunity cost is $\approx 35s$ tokens forgone.
- **Evict + recompute.** The system prompt and tool schema (2,000 tokens) hit the radix prefix cache; the remaining 38,000 tokens must be re-prefilled. At an aggregate ~9,000 prefill tok/s on this configuration, resume costs $\approx 4.2$ s of exclusive GPU time — about 147 decode-token-equivalents.

Break-even: $35 s = 147 \Rightarrow s^{*} \approx 4.2$ s. Evict if the stall exceeds ~4 s.

**Where the obstruction becomes visible.** Measured web-search tool latency in agent traces is heavy-tailed: a plausible distribution has median 0.9 s but mean 6.1 s, driven by a ~7% tail beyond 30 s. The median is below $s^*$ and the mean is above it — so a mean-based policy evicts, a median-based policy retains, and **both are wrong most of the time**. The correct decision needs $P(s > 4.2 \mid \text{tool}, \text{arguments})$, which requires the conditional stall distribution. That distribution is not published for any real agent deployment. The example is fully specified in every quantity the engine controls and undetermined in the one quantity that decides it — which is precisely why the problem is open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*