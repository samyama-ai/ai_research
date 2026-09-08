---
id: 11-inference-and-serving/cross-tenant-prefix-cache-sharing
title: "Optimal Prompt Cache Sharing Across Tenants"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Prompt Cache Sharing Across Tenants

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/cross-tenant-prefix-cache-sharing` · **Status:** open

## 1. Problem Statement

A multi-tenant LLM serving cluster holds a finite KV-cache. Prefix reuse — serving a request whose prompt shares a prefix with an earlier one by reusing that prefix's KV blocks — removes prefill FLOPs proportional to the shared length. Reuse *within* a tenant is uncontroversial. Reuse *across* tenants is where the problem lives: it raises hit rate (system prompts, tool schemas, few-shot blocks, and RAG chunks are shared by construction), and it simultaneously creates a timing channel that reveals what other tenants sent.

**Input.** A stream of requests $(t_j, u_j, p_j, \ell_j)$ — arrival time, tenant id, prompt token sequence, decode length — and a cache of capacity $B$ bytes.

**Output.** An online policy: for each request, which prefix blocks to reuse, which to admit, which to evict, and across which tenant boundaries reuse is permitted.

**Decision predicate.** Does there exist a sharing policy that attains a $(1-o(1))$ fraction of the hit rate of unrestricted global sharing while keeping cross-tenant distinguishing advantage below a stated $\epsilon$ — and is that frontier computable online?

Three variants, different difficulty:
- **Measurement:** define and estimate the leakage of a deployed cache. Partly blocked (§5).
- **Method:** build a policy on the frontier. Empirically open.
- **Theory:** characterize the achievable (hit rate, leakage) region for online caching under a distinguishability constraint. Theoretically open.

## 2. Formal Setting

Tenants $i \in [n]$. Prompts live in a radix trie $T$ over token sequences; node $v$ holds $\mathrm{len}(v)$ tokens of KV state costing
$$
\mathrm{sz}(v) = \mathrm{len}(v)\cdot 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b
$$
bytes, with $L$ layers, $H_{kv}$ KV heads, head dim $d_h$, bytes per element $b$. **Measured as:** allocated block count × block bytes, read from the allocator, not computed from the formula (paged allocators round up; the last block is partial).

Policy $\pi$ maps history to a cache state $C_t \subseteq V(T)$ with $\sum_{v \in C_t}\mathrm{sz}(v) \le B$. Hit length $h_j(\pi)$ is the number of leading tokens of $p_j$ covered by a *contiguous* root path in $C_{t_j}$ — contiguity matters, because a gap invalidates every subsequent KV entry unless a blending method (§3) recomputes it.

**Utility.** Prefill cost is superlinear in length; measure it, don't model it. Let $\hat{c}(m)$ be the measured prefill milliseconds for $m$ tokens on the target hardware. Saved work:
$$
U(\pi) = \sum_j w_{u_j}\big[\hat c(|p_j|) - \hat c(|p_j| - h_j(\pi))\big],\qquad w_i \ \text{a per-tenant weight}.
$$
Report also the delivered metric: p50/p99 TTFT (time to first token) and cluster goodput at fixed SLO attainment.

**Leakage.** Fix a tenant-$a$ adversary and a candidate secret prefix $s$. Let $b \in \{0,1\}$ be whether some victim tenant submitted $s$ in window $W$. The adversary issues probes and observes latencies. Leakage is the distinguishing advantage
$$
\mathrm{Adv}_\pi(s, W) = \big|\Pr[\hat b = 1 \mid b=1] - \Pr[\hat b=1 \mid b=0]\big|,
$$
**measured as** an empirical AUC over $\ge 10^3$ paired probe trials, with a bootstrap CI — not asserted from a mechanism argument. A policy is $\epsilon$-private if $\sup_{s,W}\mathrm{Adv} \le \epsilon$.

**Objective.** $\max_\pi U(\pi)$ s.t. $\mathrm{Adv}_\pi \le \epsilon$, online, with no knowledge of future arrivals.

**Assumptions, and which break.**
1. *Prefix locality is stable* — violated: system prompts get versioned, and hit rate drops discontinuously at each deploy.
2. *Reuse is exactly equivalent to recompute* — violated in general: numerics differ, and cross-request reuse of *non-prefix* chunks changes outputs unless blended (§3).
3. *Timing is the only channel* — violated: memory-pressure and admission-control side effects, and shared-batch scheduling, leak too.
4. *Tenants are the privacy unit* — violated: one tenant's cache serves many end users, so intra-tenant sharing can leak between *their* users.

## 3. State of the Art

**Systems SOTA (established).** PagedAttention/vLLM (Kwon et al., SOSP 2023) made block-level sharing practical via copy-on-write page tables. RadixAttention in SGLang (Zheng et al., NeurIPS 2024) maintains an LRU radix trie of KV blocks with cache-aware scheduling. Mooncake (Qin et al., FAST 2025) disaggregates prefill from decode with a cluster-wide KV store. Hydragen (Juravsky et al., 2024) decomposes attention over a shared prefix so batching over it is compute-efficient, not just memory-efficient. All are reproduced in open code.

**Beyond prefixes (claimed, partly unablated).** PromptCache (Gim et al., MLSys 2024) reuses modular, non-prefix segments via position-tolerant attention masks. CacheBlend (Yao et al., EuroSys 2025) recomputes a selected minority of tokens to repair cross-attention when concatenating independently cached chunks. Both report large TTFT reductions with small quality deltas on their own benchmark suites; the quality claim is *benchmark-number-level*, not an ablation across model families and long-context tasks.

**Fairness, not privacy.** VTC (Sheng et al., OSDI 2024) gives a virtual-token-counter scheduler with a proven bounded service difference between backlogged clients. It fairly divides *service*; it says nothing about who benefits from whose cached prefix — the sharing-externality accounting problem is untouched.

**Attack side (established).** Cache-timing recovery of other users' prompts is demonstrated: InputSnatch (Zhu et al., 2024/25) reconstructs inputs from response-time differences; "I Know What You Asked" (Song et al., NDSS 2025) leaks prompts via shared KV caches. Auditing Prompt Caching in Language Model APIs (Gu, Song, Hashimoto et al., 2025) runs statistical hypothesis tests against commercial endpoints and finds cross-user global caching in production providers, plus a cache-timing signal that distinguishes architectural properties.

**No SOTA exists** for the constrained optimum itself: no deployed system reports a measured $(U, \mathrm{Adv})$ pair.

## 4. What Is Known

- **Offline optimum is Belady's rule** for unit-size, unit-cost items; KV blocks are neither, and weighted caching with arbitrary sizes is NP-hard offline.
- **Online lower bound:** deterministic paging is at best $k$-competitive and randomized at best $H_k$-competitive (Sleator & Tarjan, CACM 1985; Fiat et al., 1991), with $k$ = cache size in pages. LRU is $k$-competitive. These bounds transfer to the *unconstrained* KV setting and are the only hard guarantee currently in hand.
- **Hit-rate gains at production scale:** SGLang reports up to ~$6.4\times$ throughput over prior systems on structured workloads with heavy prefix reuse (7B–70B class, A10G/A100). Commercial caching discounts imply large real hit rates: providers price cached input tokens at roughly $0.1\times$ of uncached.
- **Timing gap is large and easy to measure.** For a 1k-token shared prefix on a 7B model at A100 class, prefill is tens of milliseconds versus sub-millisecond lookup — a signal far above network jitter for a colocated prober. The Auditing paper detects it through the public internet.
- **Leakage is not hypothetical:** at least one major provider changed its caching scope after cross-user sharing was reported publicly (2025).

## 5. What Is Not Known

- **Theoretically open.** No competitive-ratio result for online weighted caching under a distinguishability constraint. Unknown whether an $O(\log k)$-competitive $\epsilon$-private policy exists, or whether the price of privacy is $\Omega(k)$.
- **Theoretically open.** Whether $\mathrm{Adv} \le \epsilon$ can be achieved with per-tenant delay padding at $o(1)$ latency overhead, or whether padding must cost the full prefill it hides.
- **Empirically open.** The frontier itself. Nobody has published a curve of hit rate versus measured $\mathrm{Adv}$ for real multi-tenant traces. The experiment is runnable on 8 GPUs today (§8).
- **Empirically open.** How much of global sharing's benefit survives restriction to a *public-corpus allowlist* (system prompts, tool schemas, popular documents) — the obvious safe policy whose value nobody has quantified.
- **Methodologically blocked.** There is no accepted leakage metric. Candidates — distinguishing advantage, mutual information between cache state and prompts, DP-style $\epsilon$ over a neighbouring-request relation — disagree on which policies are safe, and none has a standard adversary model, probe budget, or secret distribution. Comparisons across papers are therefore not meaningful.

## 6. Why It Is Hard

**The primary obstruction is that the evaluation does not measure the thing it names.** Reported "cache hit rate" is a property of a trace and a cache size, and it is silently confounded with (a) tenant mix, (b) system-prompt version churn, and (c) admission control that drops requests under load — a policy can raise hit rate by shedding cache-miss traffic. Two papers reporting 60% hit rate may differ by $3\times$ in delivered TTFT.

**Second: absent ground truth for leakage.** Advantage is defined against an adversary and a secret distribution, both chosen by the evaluator. A policy tuned against known probes is not a bound; it is a fitted defense. Without a canonical adversary class the "private" axis is unfalsifiable.

**Third: non-identifiability of the benefit split.** When tenants $a$ and $b$ share a prefix, the FLOPs saved on $b$'s request exist because $a$ paid for the prefill. Any attribution — for billing, for weights $w_i$, for fairness — is a cost-allocation problem with no unique answer (the Shapley value is the natural candidate and is exponential to compute online).

## 7. Current Research (as of 2026)

- **Cache-aware scheduling and disaggregation:** SGLang and vLLM communities; Mooncake-style cluster KV stores; prefix-aware routing (Preble, Srivatsa et al.). Focus is throughput, privacy scoping is per-deployment configuration.
- **Non-prefix reuse:** CacheBlend-style selective recompute, and KV compression combined with reuse. *(frontier — verify)* whether blended caches change the leakage picture: partial recompute may blur timing signatures for free.
- **Auditing as a discipline:** Stanford (Hashimoto group) and follow-ons treating black-box cache detection as statistical testing. This is the most likely source of a standard metric.
- **DP-flavored serving:** *(frontier — verify)* proposals to add calibrated noise to TTFT or to randomize admission. No published measured frontier.

## 8. Concrete Next Experiment

**Question.** How much throughput does $\epsilon$-privacy cost?

**Scale.** One 8×H100 node, Llama-3.1-8B and 70B, SGLang or vLLM with the radix cache. Replay 24 h of a multi-tenant trace (or a synthetic mix: 200 tenants, Zipf($\alpha{=}1.1$) prompt popularity, 20% of prefix mass genuinely public, mean prompt 2.5k tokens), at load sweeping 60–95% of saturation.

**Arms.**
1. **Control A — global sharing**, unrestricted cross-tenant radix cache.
2. **Control B — strict isolation**, per-tenant cache partitions.
3. Allowlist sharing: only prefixes registered as public.
4. $k$-anonymous admission: a block becomes shareable only after $k \ge 5$ distinct tenants have independently produced it.
5. Global sharing + constant-time response for cache-eligible prefixes (pad to the miss latency).

**Measurement.** For each arm: goodput at 95% SLO attainment on p99 TTFT; and $\mathrm{Adv}$ from $10^4$ paired probe trials by a colocated adversary tenant against 100 planted secret prefixes, reported as AUC with bootstrap 95% CI.

**The deciding number.** Goodput of arm 4 as a fraction of arm 1, at the largest $k$ for which $\mathrm{Adv} \le 0.55$ AUC (upper CI). **If that fraction is $\ge 0.9$, $k$-anonymous sharing is the practical answer and the theory question becomes secondary. If it is $\le 0.6$, the privacy/throughput tension is real and the frontier is worth characterizing.** Arm 3 versus arm 1 gives the second number: how much of the win is just public prefixes.

## 9. Key References

- **[Foundational]** W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, I. Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[SOTA]** L. Zheng, L. Yin, Z. Xie, C. Sun, et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS, 2024. — arXiv:2312.07104
- **[SOTA]** R. Qin, Z. Li, W. He, M. Zhang, et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* USENIX FAST, 2025. — arXiv:2407.00079
- **[SOTA]** I. Gim, G. Chen, S. Lee, N. Sarda, A. Khandelwal, L. Zhong. *Prompt Cache: Modular Attention Reuse for Low-Latency Inference.* MLSys, 2024. — arXiv:2311.04934
- **[SOTA]** J. Yao, H. Li, Y. Liu, S. Ray, Y. Cheng, Q. Zhang, K. Du, S. Lu, J. Jiang. *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion.* EuroSys, 2025. — arXiv:2405.16444
- **[Foundational]** D. Sleator, R. E. Tarjan. *Amortized Efficiency of List Update and Paging Rules.* Communications of the ACM 28(2), 1985.
- **[SOTA]** Y. Sheng, S. Cao, D. Li, B. Zhu, Z. Li, D. Zhuo, J. Gonzalez, I. Stoica. *Fairness in Serving Large Language Models.* USENIX OSDI, 2024. — arXiv:2401.00588
- **[Security]** C. Gu, X. L. Li, R. Kuditipudi, P. Liang, T. Hashimoto. *Auditing Prompt Caching in Language Model APIs.* 2025. — arXiv:2502.07776
- **[Security]** L. Song et al. *I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving.* NDSS, 2025.
- **[Security]** X. Zhu et al. *InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks.* 2024. — arXiv:2411.18191
- **[Related]** J. Juravsky, B. Brown, R. Ehrlich, D. Fu, C. Ré, A. Mirhoseini. *Hydragen: High-Throughput LLM Inference with Shared Prefixes.* 2024. — arXiv:2402.05099

## 10. Worked Example

Llama-3.1-8B, GQA with $H_{kv}=8$, $d_h=128$, $L=32$, fp16. Per token:
$$
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} = 128\ \text{KiB}.
$$
An 80 GB H100 with ~16 GB of weights and activation headroom leaves ~60 GB of KV, i.e. **~480k tokens** resident, about 192 prompts of 2.5k tokens.

Now the sharing case. A 1,024-token shared system prompt costs $1024 \times 128\ \text{KiB} = 128$ MiB — 0.2% of the cache — and its prefill is roughly $2 \times 8\text{e}9 \times 1024 \approx 1.6\times10^{13}$ FLOPs, about **25 ms** at an achieved 650 TFLOP/s. Cached, the lookup is well under 1 ms. So the prefix pays for itself after one reuse and saves 25 ms on every subsequent hit.

**Where the obstruction becomes visible.** That same 25 ms is the leak. An adversary tenant guesses a candidate system prompt, submits it, and reads TTFT: ~26 ms if some other tenant already sent it, ~51 ms if not. The gap is $25$ ms against colocated jitter of order 1–3 ms — an AUC near 1.0 with a handful of trials. Token-by-token extension of the guess turns detection into reconstruction.

Try to close it by padding to the miss latency: every hit now costs 51 ms, and the cache's entire TTFT benefit is gone, though the FLOPs saving (and hence throughput) survives — which is exactly why the deciding number in §8 is goodput, not latency. Try $k$-anonymity instead: with Zipf($1.1$) over 200 tenants, the head prefix reaches 5 distinct tenants quickly and shares fine; the tail — where a prefix is unique to one tenant and therefore actually secret — never shares, which is the correct outcome. The open question is what fraction of the 25 ms × hit-count saving lives in that head. Nobody has measured it on a real trace.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*