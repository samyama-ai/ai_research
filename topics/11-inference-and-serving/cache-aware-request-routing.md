---
id: 11-inference-and-serving/cache-aware-request-routing
title: "Cache-Aware Request Routing in Multi-Replica Clusters"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cache-Aware Request Routing in Multi-Replica Clusters

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/cache-aware-request-routing` · **Status:** open

## 1. Problem Statement

A cluster runs $R$ replicas of the same LLM. Each replica holds a private KV cache of finite size. A router sees a stream of requests and must assign each to one replica, immediately and irrevocably (or with a bounded-cost migration). Sending a request to a replica that already holds a prefix of its tokens skips that much prefill; sending it there anyway may queue it behind other work.

- **Input:** request stream $\sigma = (r_1, r_2, \dots)$, each $r_i$ a token sequence with unknown output length; per-replica cache state; per-replica queue state.
- **Output:** an assignment $a_i \in \{1,\dots,R\}$ made at arrival time.
- **Objective:** minimise a latency functional (p99 time-to-first-token, or mean end-to-end latency) subject to a throughput floor — equivalently, maximise goodput at fixed SLO attainment.

Three variants, routinely conflated:

- **Measurement.** Is cache-hit rate the right proxy for the objective at all? A router that maximises hit rate can lose to a load-balancing router, because the saved prefill is a *shared* resource with decode.
- **Method.** Find a routing policy that beats both endpoints (pure locality, pure load balance) across workload mixes, without per-workload tuning.
- **Theory.** Bound the competitive ratio of any online policy for joint caching-and-load-balancing with prefix-structured reuse. No such bound is known.

Solved means: a policy with a proved competitive ratio, or an empirically dominant policy that is not tuned per workload and holds under cache eviction pressure and skewed prefix popularity.

## 2. Formal Setting

Replicas $j \in [R]$, each with KV budget $B_j$ blocks of $b$ tokens. Cache state $C_j(t) \subseteq$ set of cached blocks, maintained as a radix/prefix tree over token blocks.

For request $r_i$ with prompt length $L_i$, the **hit length** on replica $j$ is
$$h_{ij} = b \cdot \max\{k : \text{first } k \text{ blocks of } r_i \in C_j(t_i)\}.$$
Measured as: block-hash lookups in the router's replica-state mirror, verified after the fact against the replica-reported `num_cached_tokens` field (vLLM/SGLang both expose it). The mirror is stale; disagreement between predicted and reported $h_{ij}$ is itself a quantity to log.

Prefill service time on a compute-bound replica:
$$T^{\text{pre}}_{ij} = \alpha (L_i - h_{ij}) + \beta (L_i^2 - h_{ij}^2) + \gamma \cdot \mathbb{1}[h_{ij}>0],$$
with $\alpha$ (per-token linear term, GEMM-bound), $\beta$ (attention quadratic term), $\gamma$ (cache-lookup/copy overhead). Fit $\alpha,\beta,\gamma$ by regression on isolated single-request prefills at fixed batch size.

Queueing: replica $j$ has pending work $W_j(t)$, measured as summed estimated prefill plus decode-token-slots, not request count. TTFT for request $i$:
$$\text{TTFT}_i = q_{ij} + T^{\text{pre}}_{ij}, \qquad q_{ij} = \text{waiting time under the replica's continuous-batching scheduler.}$$
$q_{ij}$ is not a simple function of $W_j$: continuous batching interleaves prefill chunks with decode, so $q$ depends on the chunked-prefill budget and the decode population.

Objective over horizon $T$:
$$\min_{a} \; \Pr\!\big[\text{TTFT}_i > \tau\big] \quad \text{s.t.}\quad \text{throughput} \ge \Theta_0,$$
with the router's realised decision quality reported as regret against an offline optimum computed by an MILP on the logged trace.

**Assumptions, and which break.**
1. *Cache state is known to the router.* Violated: replicas evict asynchronously (LRU over the radix tree) and preempt under memory pressure; the mirror drifts. Drift grows with load — exactly when routing matters.
2. *Output length unknown at arrival.* True, and it determines the decode footprint that evicts other prefixes. Length prediction is itself an open problem.
3. *Prefix reuse is a tree.* Violated by RAG, where reusable chunks appear at non-prefix positions; standard prefix caching returns $h_{ij}=0$ for a document reordering that reuses 95% of tokens.
4. *Replicas are homogeneous.* Violated in disaggregated clusters (separate prefill and decode pools) and in mixed-hardware fleets.
5. *Migration is free or forbidden.* Both are wrong: KV transfer costs bytes $\propto 2 \cdot n_{\text{layers}} \cdot n_{\text{kv heads}} \cdot d_{\text{head}} \cdot h \cdot \text{dtype}$, real but finite over RDMA.

## 3. State of the Art

**Established (ablated, reproduced).**
- *Prefix caching works within a replica.* PagedAttention/vLLM (Kwon et al., SOSP 2023) made block-level sharing practical; RadixAttention in SGLang (Zheng et al., NeurIPS 2024) added LRU over a radix tree and reported up to $6.4\times$ throughput on structured/multi-call workloads. The intra-replica gain is reproduced widely and is not in dispute.
- *Cache reuse across turns cuts TTFT.* AttentionStore/CachedAttention (Gao et al., USENIX ATC 2024) reports up to 87% TTFT reduction and ~70% prefill cost reduction on multi-turn conversation traces by keeping KV in a memory hierarchy.
- *Consistent hashing with bounded loads* (Mirrokni, Thorup, Zadimoghaddam, SODA 2018) gives an $O(1)$-factor load guarantee with $O(1)$ amortised reassignments — the standard fallback when locality and balance conflict, but it has no notion of *partial* (prefix) hits.

**Claimed but not independently ablated.**
- Preble (Srivatsa et al., 2024) proposes distributed prompt scheduling with an E2 (explore/exploit) policy and reports large average and p99 latency reductions over per-replica-independent baselines on 2–8 GPU clusters. The comparison is against load-balancing routers; the counterfactual against a *tuned* hybrid score has not been published by a third party.
- Mooncake (Qin et al., FAST 2025) makes the KV cache a first-class disaggregated pool with a global store, reporting large throughput gains, part of which are simulated rather than measured on the production cluster.
- Prefix-cache-aware routing in the Kubernetes Gateway API Inference Extension and the vLLM production-stack router is deployed and measured in vendor blog posts only *(frontier — verify)*. These are benchmark numbers, not ablations: the load term and the locality term are changed together.

**Theory SOTA.** There is no competitive-ratio result for the joint problem. The pieces exist separately: deterministic paging is $k$-competitive and no better (Sleator & Tarjan, CACM 1985); randomized paging is $\Theta(\log k)$ (Fiat et al., J. Algorithms 1991); balls-into-bins with two choices gives $\log\log n$ max load (Azar, Broder, Karlin, Upfal, SICOMP 1999); static data placement with capacities is NP-hard with a constant-factor approximation (Baev, Rajaraman, Swamy, SICOMP 2008).

## 4. What Is Known

- Prefill dominates TTFT for long prompts. At $L = 8$k tokens on a 70B-class model, prefill is seconds; a full prefix hit reduces it to a KV copy plus one block of compute. Cache-hit economics are visible in public API pricing: cached input tokens are billed at roughly 0.1× uncached across major providers, a vendor-revealed estimate of the cost ratio.
- Reuse in real traffic is heavy-tailed. Published serving traces (Mooncake's released trace, FAST 2025) show a small set of long system prompts and document prefixes accounting for a large share of prompt tokens — the regime where locality routing has anything to win.
- Locality and balance genuinely conflict. Routing every request sharing a hot prefix to one replica concentrates load; SGLang/Preble-style systems all report needing an explicit load term.
- Non-prefix reuse is recoverable at a quality cost: CacheBlend (Yao et al., EuroSys 2025) fuses independently cached chunks by recomputing a small fraction of tokens, reporting several-fold TTFT reduction on RAG workloads with small generation-quality change — measured at 7B–70B scale on RAG datasets.
- Migration is feasible: Llumnix (Sun et al., OSDI 2024) live-migrates requests between instances with near-zero downtime, showing the irrevocable-assignment constraint is a design choice, not a physical limit.

## 5. What Is Not Known

- **Theoretically open.** No competitive ratio — upper or lower — for online routing with per-replica prefix caches, partial hits, and queueing. Even the offline version's complexity with the quadratic prefill term is unsettled. Whether any policy is $O(\text{polylog})$-competitive against an offline optimum is unproved either way.
- **Empirically open.** Nobody has published a controlled comparison of routing policies with *cache state and load held separately variable*, on a shared trace, at $R \ge 16$ replicas, under cache pressure (working set $\gg \sum_j B_j$). Every published comparison changes multiple factors at once and runs at $R \le 8$.
- **Methodologically blocked.** There is no agreed definition of "cache hit rate" for a cluster. Candidates — hit tokens / total prompt tokens, hit requests / requests, prefill FLOPs saved — rank policies differently, and none is monotone in the latency objective. Until the metric is fixed, cross-paper numbers are not comparable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by feedback**. Routing changes cache state, which changes future hit rates, which changes future routing. The policy and the workload are not independent: an A/B comparison of two routers on the same trace does not compare them on the same cache states after the first few seconds. This breaks the usual serving-benchmark methodology, which assumes the system under test does not shape its own input distribution.

Two consequences. First, hit rate is an *endogenous* metric — a router can inflate it by starving the tail (route only easy, hot-prefix requests promptly), so a hit-rate win can coexist with a p99 loss. Second, the offline optimum needed for regret is expensive: it requires solving assignment jointly with eviction over the whole trace, and eviction policy is itself part of the replica, not the router.

Compute cost is secondary but real: a meaningful experiment needs $\ge 16$ replicas of a production-scale model for hours, which is a five-figure cloud bill per policy arm.

## 7. Current Research (as of 2026)

- **KV-cache disaggregation.** Moonshot AI (Mooncake), and the LMCache/vLLM production-stack line, treat KV as a shared tiered store so routing becomes a *placement* problem rather than an affinity problem *(frontier — verify current status)*.
- **Router as a control plane.** Kubernetes Gateway API Inference Extension, with prefix-aware and load-aware scorers combined by weights; vLLM's `kv-events` stream publishes block insert/evict so a router can mirror cache state. Active, mostly engineering, weights hand-tuned *(frontier — verify)*.
- **Learned routing.** RL and bandit formulations over the (hit length, queue depth, predicted output length) state. Published results are small-scale simulation.
- **Non-prefix reuse.** CacheBlend-style selective recompute, and position-independent caching, which if it lands changes the routing objective from "longest prefix" to "largest cached token set".
- **Theory.** Nobody appears to be working on the competitive analysis directly; the nearest active line is online algorithms with predictions (learning-augmented caching and load balancing), which supplies the right template but not the result.

## 8. Concrete Next Experiment

**Question.** Does a locality-weighted router beat load balancing on p99 TTFT once the working set exceeds aggregate cache capacity, and at what weight?

**Scale.** $R = 16$ replicas of an 8B model on single A100/H100 GPUs (fits one commodity node group; 8B keeps the bill under ~$2k for a 4-hour sweep). Replay the released Mooncake trace, rescaled so aggregate prompt working set is $3\times \sum_j B_j$. Run each arm 30 min after a 5-min warm-up, three seeds.

**Arms.** Score $s_{ij} = h_{ij} - \lambda \hat{W}_j$; sweep $\lambda \in \{0, 0.25, 0.5, 1, 2, 4, \infty\}$ in token-equivalent units.

**Control arm.** $\lambda = \infty$ (pure least-loaded, cache-blind) — the deployed default. Second control: $\lambda = 0$ (pure longest-prefix), which isolates the locality endpoint.

**Deciding number.** p99 TTFT of the best $\lambda$ divided by p99 TTFT of the $\lambda=\infty$ control, at matched throughput. If $\min_\lambda$ ratio $\ge 0.9$ under cache pressure, cache-aware routing does not pay at this scale and the reported gains are artifacts of undersubscribed caches. If the ratio is $\le 0.6$ but the argmin $\lambda$ moves by more than $4\times$ across the three trace segments, the problem is confirmed as open in the *method* sense: the gain exists but is not achievable without per-workload tuning.

**Required secondary logging.** Predicted vs. replica-reported $h_{ij}$ per request (mirror drift), and eviction counts per replica. Without these, a negative result is uninterpretable.

## 9. Key References

- **[Foundational]** Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Foundational]** Sleator, Tarjan. *Amortized Efficiency of List Update and Paging Rules.* Communications of the ACM, 1985.
- **[Foundational]** Fiat, Karp, Luby, McGeoch, Sleator, Young. *Competitive Paging Algorithms.* Journal of Algorithms, 1991.
- **[Foundational]** Azar, Broder, Karlin, Upfal. *Balanced Allocations.* SIAM Journal on Computing, 1999.
- **[SOTA]** Zheng, Yin, Xie, Sun, Huang, Yu, Cao, Kozyrakis, Stoica, Gonzalez, Barrett, Sheng. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS, 2024. — arXiv:2312.07104
- **[SOTA]** Srivatsa, He, Abhyankar, Li, Zhang. *Preble: Efficient Distributed Prompt Scheduling for LLM Serving.* 2024. — arXiv:2407.00023
- **[SOTA]** Qin, Cheng, Zhao, Chen, Sheng, Zhang, et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* USENIX FAST, 2025. — arXiv:2407.00079
- **[SOTA]** Gao, Dai, Peng, Zhang, et al. *Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention.* USENIX ATC, 2024. — arXiv:2403.19708
- **[SOTA]** Yao, Li, Zhang, Cheng, Jiang, et al. *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion.* EuroSys, 2025. — arXiv:2405.16444
- **[SOTA]** Sun, Huang, Zhao, Zheng, Lin, et al. *Llumnix: Dynamic Scheduling for Large Language Model Serving.* USENIX OSDI, 2024. — arXiv:2406.03243
- **[Theory]** Mirrokni, Thorup, Zadimoghaddam. *Consistent Hashing with Bounded Loads.* SODA, 2018. — arXiv:1608.01350
- **[Theory]** Baev, Rajaraman, Swamy. *Approximation Algorithms for Data Placement Problems.* SIAM Journal on Computing, 2008.

## 10. Worked Example

Two replicas, 8B model, GPU KV budget $B_j = 40{,}000$ tokens each. Two hot system prefixes, $A$ and $B$, each 4,000 tokens, plus a 1,000-token user suffix per request. Traffic: 60% $A$, 40% $B$. Measured constants on one H100: linear prefill $\approx 0.09$ ms/token at this scale, so a cold 5,000-token prefill $\approx 450$ ms; with a 4,000-token hit, $\approx 90$ ms plus $\approx 10$ ms lookup.

**Pure locality.** All $A$ → replica 1, all $B$ → replica 2. Hit rate = 80% of prompt tokens. But replica 1 gets 1.5× the requests of replica 2. At arrival rate 20 req/s, replica 1 sees 12 req/s of 100 ms work (1.2 utilisation of a single-stream lane) and queues; replica 2 sees 8 req/s (0.8). Measured p99 TTFT is set by replica 1's queue, not by prefill: the 360 ms saved per request is swamped by a queue that grows without bound while replica 2 idles 20% of the time.

**Pure balance.** Round-robin. Both prefixes land on both replicas; each replica caches both ($8{,}000 \le 40{,}000$ tokens — no eviction), so after warm-up the hit rate is *also* 80%. Locality routing wins nothing.

That is the visible obstruction. Locality only pays when the working set exceeds cache capacity. Add 30 distinct 4,000-token document prefixes with Zipf($s{=}1.1$) popularity: working set $= 128$k tokens against 80k of capacity. Now round-robin duplicates every prefix across both replicas, halving effective capacity to $\sim$40k of distinct coverage and evicting the tail; locality routing partitions the 30 prefixes and covers $\sim$80k. Expected hit rate: $\sim$62% (balance) vs $\sim$84% (locality) under a static LRU estimate.

But the *measured* outcome depends on which effect the trace triggers, and the two regimes differ only in a parameter — working-set-to-capacity ratio — that no published routing paper reports. Papers that benchmark in the first regime measure the queue; papers in the second measure the cache. Both report "cache-aware routing wins $N\times$". This is why the deciding number in §8 must be p99 TTFT at fixed throughput *with the capacity ratio stated*, not hit rate.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*