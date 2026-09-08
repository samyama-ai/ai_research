---
id: 11-inference-and-serving/continuous-batching-scheduling-optimality
title: "Continuous Batching Scheduling Optimality"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continuous Batching Scheduling Optimality

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/continuous-batching-scheduling-optimality` · **Status:** open

## 1. Problem Statement

Continuous (iteration-level) batching lets an LLM server admit and retire requests between forward passes rather than at batch boundaries. At every iteration the scheduler picks which waiting prefills to admit, which running decodes to continue, and which to preempt or evict — under a hard KV-cache memory constraint. The question: **what is the optimal such policy, and how far from it are deployed schedulers?**

Three variants, with different difficulty:

- **Measurement.** Given a trace and a hardware target, how far is a given scheduler from the achievable frontier of (throughput, TTFT, inter-token latency)? Requires a credible optimum to compare against; today only pairwise A/B numbers exist.
- **Method.** Build a policy that dominates FCFS + chunked prefill on realistic traces without a length oracle. Runnable now.
- **Theory.** Prove a competitive ratio (or a lower bound) for online iteration-level scheduling with unknown output lengths and a memory constraint that is itself a function of scheduling history. Open.

Solving it means: a policy with a proved bound relative to an offline clairvoyant optimum, *and* an implementation whose measured SLO-goodput matches that bound's prediction on production traces.

## 2. Formal Setting

Requests $r_i = (a_i, p_i, o_i)$: arrival time $a_i$, prompt length $p_i$ (observed at arrival), output length $o_i$ (**not** observed until the request emits EOS). Let $d_i(t)$ be tokens decoded by time $t$.

At iteration $t$ the scheduler chooses a batch $B_t$ and, for each admitted prefill, a chunk size $c_{j,t}$. Constraints as actually enforced by a server:

$$\sum_{j \in B_t} \big(p_j + d_j(t)\big) \cdot s_{\text{kv}} \;\le\; M, \qquad \sum_{j \in B_t} c_{j,t} + |D_t| \;\le\; \Lambda$$

where $s_{\text{kv}}$ is bytes of KV per token (measured: $2 \cdot L \cdot H_{kv} \cdot d_h \cdot b$), $M$ is free HBM after weights and activations, $D_t$ the decoding subset, and $\Lambda$ the token budget (a tunable, e.g. vLLM's `max_num_batched_tokens`).

Iteration latency, measured by timing the forward pass:

$$T(B_t) \;\approx\; \alpha + \beta \sum_{j} c_{j,t} + \gamma \sum_{j \in B_t}\big(p_j + d_j(t)\big)$$

$\alpha$ = fixed kernel-launch and weight-read cost (decode is memory-bound: $\alpha \approx W/\text{BW}$), $\beta$ = per-prefill-token compute cost, $\gamma$ = per-KV-token attention read cost.

Per-request metrics as measured at the client:
- TTFT$_i = f_i - a_i$ (first token emitted at $f_i$);
- TBT/ITL: the distribution of gaps between successive tokens, usually reported as P99, not mean;
- SLO-goodput $G = \frac{1}{n}\sum_i \mathbb{1}[\text{TTFT}_i \le \tau_1 \wedge \mathrm{P99}_i(\text{TBT}) \le \tau_2]$.

Objective: maximize $G$, or minimize $\sum_i w_i (\text{TTFT}_i + \lambda \cdot \text{TBT}_i^{P99})$.

**Assumptions known to be violated in practice:** (i) $T(\cdot)$ is linear — false near the compute-bound knee and under tensor-parallel collectives; (ii) arrivals are Poisson — production traces are bursty and correlated by tenant; (iii) preemption is free — recompute costs a full prefill, swap costs PCIe bandwidth; (iv) $o_i$ is drawn i.i.d. from a known distribution — output length correlates strongly with prompt and with the application; (v) single replica — real fleets route across replicas with prefix-cache affinity, which changes the objective.

## 3. State of the Art

**Established (reproduced, ablated).**
- *Orca* (Yu et al., OSDI 2022) introduced iteration-level scheduling with selective batching. Reported $36.9\times$ throughput at the same latency vs FasterTransformer on GPT-3 175B — a single-system benchmark number, not an independently reproduced one, but the mechanism is now universal.
- *PagedAttention / vLLM* (Kwon et al., SOSP 2023) showed that KV fragmentation, not compute, binds batch size; $2$–$4\times$ throughput over Orca at equal latency. Independently reproduced many times; the memory-management claim is solid.
- *Sarathi-Serve* (Agrawal et al., OSDI 2024) established the **stall-free batching** result: splitting prefill into chunks and piggybacking decodes removes the generation stall. Reported up to $2.6\times$ capacity within SLO for Mistral-7B on one A100 and $\sim 5.6$–$6.9\times$ for Falcon-180B on 8 A100s. The mechanism is ablated; the exact multipliers are trace-specific.
- *Prefill/decode disaggregation*: DistServe (Zhong et al., OSDI 2024) reports $7.4\times$ more requests or $12.6\times$ tighter SLO; Splitwise (Patel et al., ISCA 2024) reports $1.4\times$ throughput at $20\%$ lower cost. Established that the two phases have different optimal parallelism; **not** established which of disaggregation vs chunked prefill wins, and under what arrival rate.

**Claimed but unablated.**
- Length-prediction-driven SJF/SRPT (Qiu et al., 2024; FastServe, Wu et al., 2023) reports large latency wins, but almost always on ShareGPT-derived traces where output length is unusually predictable. The gain attributable to prediction *accuracy* versus to simply avoiding head-of-line blocking is rarely separated.
- Token-budget $\Lambda$ auto-tuning: every major server exposes it; no published policy adapts it online with an ablation.

**Theory SOTA is classical and only loosely applicable.** SRPT is optimal for total flow time on one machine with known sizes; nonclairvoyant scheduling has an $\Omega(n^{1/3})$ competitive lower bound for total flow time (Motwani, Phillips & Torng, 1994); $(1+\epsilon)$-speed augmentation restores $O(1)$-competitiveness (Kalyanasundaram & Pruhs, JACM 2000). None of these models the memory constraint whose feasibility depends on the scheduler's own past choices.

## 4. What Is Known

- **Head-of-line blocking is real and large.** FCFS with unbounded prefill inflates P99 TBT by roughly an order of magnitude when a long prompt lands mid-decode (Sarathi-Serve, A100, Mistral-7B / Yi-34B).
- **The chunked-prefill tax is measurable.** Chunking costs extra attention reads and repeated weight loads; reported prefill-throughput loss is on the order of $10$–$25\%$ at chunk sizes of 512–1024 tokens on A100/H100 for 7B–70B models.
- **KV memory, not FLOPs, sets the batch ceiling** for most 7B–70B deployments at 4k–32k context. Llama-3-8B with GQA: $128$ KB/token; a 70B MHA-style model is $\sim 10\times$ worse per token.
- **Preemption-by-recompute usually beats swapping** at small block counts, and loses at large ones (vLLM ablation).
- **Fairness and throughput conflict.** VTC (Sheng et al., OSDI 2024) shows FCFS and naive continuous batching starve low-rate clients; a work-conserving fair-share counter fixes it at some throughput cost.

## 5. What Is Not Known

- **Theoretically open.** No competitive-ratio result for online scheduling where (a) job sizes are unknown, (b) jobs occupy a resource that grows monotonically with service received, and (c) admission is irrevocable except by paying a recompute penalty. No lower bound either. Nor is the *offline* clairvoyant version proved NP-hard for this exact formulation, despite being widely assumed so.
- **Empirically open.** No public study computes the offline optimum (via an ILP or simulator with oracle lengths, e.g. Vidur) on a production trace and reports the gap of deployed schedulers to it. The experiment is runnable on existing simulators; nobody has published the number.
- **Empirically open.** The chunked-prefill vs disaggregation crossover as a function of arrival rate, prompt/decode ratio and interconnect bandwidth.
- **Methodologically blocked.** "Latency" is not one number. TTFT, mean ITL, P99 ITL and jitter trade against each other, and user-perceived quality (Andes, 2024) is not a monotone function of any of them. Without a settled scalar objective, "optimal" is undefined — which is why most papers report a Pareto plot instead.

## 6. Why It Is Hard

**The specific obstruction is non-identifiability of the objective compounded by a state-dependent feasible set.**

1. The constraint set is endogenous: admitting a request now shrinks $M$ later by an amount that depends on the request's unknown $o_i$. Classical flow-time scheduling has no analogue — a job's resource footprint there is fixed.
2. Every reported speedup is a ratio between two points on different Pareto surfaces. A $2.6\times$ "capacity within SLO" is a function of the chosen $(\tau_1, \tau_2)$; changing $\tau_2$ from P99 to mean can reverse the ordering of two schedulers. This is an evaluation that does not measure the thing it names.
3. The absent ground truth: no one publishes the clairvoyant offline optimum, so all comparisons are relative. A scheduler that is $1.3\times$ better than FCFS may be at $40\%$ or at $95\%$ of optimum, and the literature cannot distinguish these.
4. Compute cost of a fair sweep: the policy space is $\Lambda \times$ chunk size $\times$ preemption rule $\times$ admission rule $\times$ length predictor, over several traces and model sizes. A full sweep on real hardware is thousands of GPU-hours; simulators cut this but inherit the linear-$T(\cdot)$ error.

## 7. Current Research (as of 2026)

- **Simulation-first scheduling research.** Vidur (Microsoft Research India / Georgia Tech, MLSys 2024) fits per-kernel latency models to enable large policy sweeps. Direction: use it to compute optimality gaps, not just to tune configs. *(frontier — verify)*
- **Prefill/decode disaggregation at fleet scale** (Microsoft, PKU, Alibaba, Moonshot's Mooncake line of work) — moving the scheduling question from one replica to a two-pool system with KV transfer costs.
- **Cross-replica scheduling with prefix-cache affinity** (SGLang RadixAttention, vLLM router work). The objective gains a cache-hit term, and the single-replica optimum is no longer the right target. *(frontier — verify)*
- **Learned/length-aware admission** — proxy-model length prediction, ranking rather than regression, and risk-aware SJF that degrades gracefully when the predictor errs.
- **QoE-shaped objectives** (Andes and successors): deliberately slowing fast streams to free capacity, since a user reads slower than the model generates.

## 8. Concrete Next Experiment

**Question:** how far from clairvoyant-offline optimal is production-grade continuous batching?

**Scale.** Llama-3-8B (fp16) on one A100-80GB and Llama-3-70B (fp16) on 4×H100, single replica. Trace: 20,000 requests replayed from a public production-shaped trace (Azure LLM inference trace, or ShareGPT with an arrival process resampled to CV$^2 = 3$), at load factors $\rho \in \{0.5, 0.7, 0.9\}$.

**Arms.**
1. *Control:* vLLM default — FCFS + chunked prefill, $\Lambda$ at library default.
2. Best tuned $\Lambda$ and chunk size found by a 20-point sweep.
3. Oracle-SRPT: same server, output lengths revealed at admission.
4. **Offline bound:** ILP/DP over the simulator with all $(a_i, p_i, o_i)$ known, maximizing SLO-goodput at $\tau_1 = 2$ s TTFT, $\tau_2 = 100$ ms P99 TBT. Solve to within a 5% gap on 2,000-request windows and stitch.

**Deciding number.** The ratio $G_{\text{control}} / G_{\text{offline}}$ at $\rho = 0.9$. If it is $\ge 0.9$, scheduling is a closed problem and effort belongs in disaggregation, kernels and cache reuse. If it is $\le 0.7$, there is a $\ge 30\%$ goodput overhang and the gap between arm 3 and arm 4 tells you how much of it needs a length oracle.

Secondary number: $G_{\text{oracle-SRPT}} / G_{\text{offline}}$ — the price of memory-constraint myopia alone, with clairvoyance held fixed.

## 9. Key References

- **[Foundational]** Yu, Jeong, Kim, Kim, Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI, 2022.
- **[Foundational]** Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[SOTA]** Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov, Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024. — arXiv:2403.02310
- **[SOTA]** Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI, 2024. — arXiv:2401.09670
- **[SOTA]** Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA, 2024. — arXiv:2311.18677
- **[SOTA]** Sheng, Cao, Li, Zhu, Li, Zhuo, Gonzalez, Stoica. *Fairness in Serving Large Language Models.* OSDI, 2024. — arXiv:2401.00588
- **[Systems]** Sun, Huang, Zhao, Chen, Cheng, Zhang, Lin, Chen, Jin. *Llumnix: Dynamic Scheduling for Large Language Model Serving.* OSDI, 2024. — arXiv:2406.03243
- **[Tooling]** Agrawal, Kedia, Mohan, Panwar, Kwatra, Gulavani, Ramjee, Tumanov. *Vidur: A Large-Scale Simulation Framework for LLM Inference.* MLSys, 2024. — arXiv:2405.05465
- **[Theory]** Motwani, Phillips, Torng. *Nonclairvoyant Scheduling.* Theoretical Computer Science, 1994.
- **[Theory]** Kalyanasundaram, Pruhs. *Speed is as Powerful as Clairvoyance.* Journal of the ACM, 2000.
- **[Theory]** Leonardi, Raz. *Approximating Total Flow Time on Parallel Machines.* STOC, 1997.
- **[Related]** Liu, Li, Cheng, Ray, Huang, Zhang, Du, Yao, Lu, Ananthanarayanan, Maire, Hoffmann, Holtzman, Jiang. *Andes: Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services.* 2024. — arXiv:2404.16283
- **[Survey]** Miao, Oliaro, Cheng, Zhang, et al. *Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems.* 2023. — arXiv:2312.15234

## 10. Worked Example

**Setup.** Llama-3-8B, fp16, A100-80GB. Weights $W = 16$ GB; HBM bandwidth $\approx 1.6$ TB/s; dense fp16 peak $312$ TFLOP/s, realized MFU on prefill $\approx 60\%$. KV per token: $2 \times 32 \text{ layers} \times 8 \text{ KV heads} \times 128 \times 2\text{ B} = 128$ KB. Free KV budget $\approx 56$ GB $\Rightarrow$ $\sim 460{,}000$ tokens.

**Steady state.** 32 requests decoding, 2,000 KV tokens each = 64k tokens = 8 GB of KV.
- Weight read: $16/1600 \approx 10$ ms. KV read: $8/1600 = 5$ ms.
- Iteration $\approx 15$ ms $\Rightarrow$ ITL = 15 ms, comfortably under a 100 ms SLO.

**A 2,048-token prompt arrives.**
- Prefill FLOPs $\approx 2 \times 8\times10^9 \times 2048 = 3.3\times10^{13}$. At $0.6 \times 312$ TFLOP/s: **175 ms**.
- Run it unchunked (Orca-style): that iteration takes $175 + 15 = 190$ ms. All 32 in-flight requests see a single 190 ms gap. P99 TBT jumps $12.7\times$ and blows the 100 ms SLO for 32 users to serve 1.

**Chunk it at $\Lambda_{\text{prefill}} = 512$ (Sarathi-style).**
- Per chunk: $\approx 44$ ms compute $+ 15$ ms of co-batched decode work $= 59$ ms.
- 4 chunks: TBT capped at 59 ms — SLO held. But wall-clock prefill is $4 \times 59 = 236$ ms vs 175 ms: **a 35% TTFT tax**, and the 32 decoders now advance 4 tokens in 236 ms (59 ms/token) instead of 4 tokens in 60 ms.

**Where the obstruction shows.** Pick $\Lambda_{\text{prefill}} = 256$: TBT falls to $\approx 37$ ms, TTFT tax rises to $\sim 69\%$. Pick 1024: TBT $\approx 103$ ms (SLO violated on the P99), TTFT tax $\sim 17\%$. The correct $\Lambda$ depends on the ratio of arriving prefill tokens to in-flight decode tokens over the next few hundred milliseconds — which depends on the unknown $o_i$ of every request currently decoding. A static $\Lambda$ is provably wrong for *some* segment of any bursty trace; no published policy sets it online, and no one has computed how much goodput the static choice costs against the offline optimum. That missing number is the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*