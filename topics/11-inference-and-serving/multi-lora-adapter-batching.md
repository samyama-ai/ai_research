---
id: 11-inference-and-serving/multi-lora-adapter-batching
title: "Optimal Multi-LoRA Adapter Batching"
topic: 11-inference-and-serving
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Multi-LoRA Adapter Batching

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/multi-lora-adapter-batching` · **Status:** partially-solved

## 1. Problem Statement

A server holds one frozen base model $W$ and $N$ LoRA adapters (often $N \in [10^2, 10^4]$), only a few of which fit in GPU memory at once. Requests arrive online, each tagged with the adapter it must be served by. The scheduler must decide, at every iteration, **which requests to place in the next batch** — equivalently, which set of adapters to co-resident and co-execute.

- **Input:** request stream $\{(t_i, a_i, p_i, \hat{o}_i)\}$ (arrival time, adapter id, prompt length, unknown output length); adapter ranks $\{r_j\}$; GPU memory budget $M$; host↔device bandwidth $\beta$.
- **Output:** an iteration-level schedule — batch membership, adapter residency set, and eviction/prefetch actions.
- **Objective:** maximize throughput subject to per-request SLOs, typically time-to-first-token (TTFT) and time-per-output-token (TPOT) tail percentiles.
- **Solved** would mean: a scheduling policy with a proven competitive ratio against the offline optimum under a validated cost model, *and* a kernel whose cost is provably insensitive to adapter heterogeneity within a batch.

Three variants, different difficulty:

- **Measurement:** what is the true marginal cost of adding the $d$-th distinct adapter to a batch of size $B$? Currently fitted per-kernel, not derived.
- **Method:** find the online policy. Heuristics exist (S-LoRA, dLoRA, Chameleon); none carries a guarantee.
- **Theory:** the offline version is a batch-scheduling problem with family setup times and a cache; no approximation ratio is known for the online form with unknown output lengths.

## 2. Formal Setting

LoRA replaces a frozen $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with $W + \tfrac{\alpha}{r} B_j A_j$, $A_j \in \mathbb{R}^{r_j \times d_{\text{in}}}$, $B_j \in \mathbb{R}^{d_{\text{out}} \times r_j}$ (Hu et al., ICLR 2022). Merging $B_jA_j$ into $W$ costs nothing at inference but forbids batching across adapters, so multi-tenant serving keeps them **unmerged**:

$$y_i = W x_i + \tfrac{\alpha}{r_{a_i}} B_{a_i} A_{a_i} x_i .$$

For a batch $\mathcal{B}$ with $B = |\mathcal{B}|$ tokens and $d = |\{a_i : i \in \mathcal{B}\}|$ distinct adapters, the per-layer cost decomposes as

$$C(\mathcal{B}) = \underbrace{C_{\text{base}}(B)}_{\text{one dense GEMM}} + \underbrace{C_{\text{lora}}(B, d, \{r_j\})}_{\text{gathered / segmented GEMM}} + \underbrace{\sum_{j \in \text{fetch}} \frac{2 r_j (d_{\text{in}}+d_{\text{out}}) s}{\beta}}_{\text{adapter transfer, } s \text{ bytes/param}} .$$

**How each quantity is measured.**

- $C_{\text{base}}, C_{\text{lora}}$: isolated CUDA-graph-captured kernel time, median of $\geq 100$ replays, clocks locked. Not derived from FLOPs — the LoRA term is memory-bound, so FLOP counts mispredict it badly.
- **Adapter-cost slope** $\kappa = \partial C_{\text{lora}} / \partial d$ at fixed $B$: the number that decides whether heterogeneous batches are free. Measured by sweeping $d \in \{1,\dots,B\}$ with $B$ fixed.
- **Effective throughput** $T = \sum_i \mathbb{1}[\text{SLO}_i \text{ met}] / \text{wall time}$, not raw tokens/s. Reporting tokens/s without the SLO indicator is the single most common confound in this literature.
- **Memory**: $M \geq M_{\text{base}} + M_{\text{KV}}(t) + \sum_{j \in \mathcal{R}(t)} 2 r_j (d_{\text{in}}+d_{\text{out}}) s$, with $\mathcal{R}(t)$ the resident set. KV cache and adapter pool compete for the same bytes; S-LoRA's unified paging is exactly a response to this.

**Assumptions, and which break.**

| Assumption | Status |
|---|---|
| Output length $\hat o_i$ known or predictable | **Violated.** Length prediction is the dominant scheduling error term. |
| Requests per adapter are i.i.d. | **Violated.** Real traces are bursty and heavy-tailed over adapters. |
| All ranks equal ($r_j = r$) | **Violated.** Mixed-rank pools force padding to $r_{\max}$ or ragged kernels. |
| $\kappa \approx 0$ (adapter count free) | **Approximately true** for decode at large $B$, **false** for prefill and small $B$. |
| Base weights fixed across tenants | Holds by construction; breaks if quantization differs per tenant. |

The offline decision problem — partition requests into batches minimizing makespan when a batch pays a setup cost per adapter family and residency is capacity-limited — contains bin packing with conflicts as a special case, hence NP-hard. This is folklore in the systems papers, not a stated theorem in any of them.

## 3. State of the Art

**Systems/empirical SOTA (established, with released code and independent reuse):**

- **Punica** (Chen et al., MLSys 2024): introduces **SGMV** (segmented gather matrix-vector multiply) — one kernel applying different adapters to different rows of a batch. Established: SGMV makes decode-phase cost nearly flat in $d$. Reported ~12× throughput over prior serving systems; that headline is a benchmark number against baselines with no multi-LoRA support, so it measures the absence of a feature, not a scheduling advance.
- **S-LoRA** (Sheng et al., MLSys 2024): unified paging for KV cache and adapter weights, plus tensor-parallel-compatible custom kernels. Established: ~2,000 adapters served from one A100-80GB with host-memory offload; reported up to ~4× over vLLM with naive adapter support and far larger factors over HuggingFace PEFT. The 4× is a benchmark number under a synthetic Poisson-per-adapter workload.
- **dLoRA** (Wu et al., OSDI 2024): dynamically switches between **merged** and **unmerged** execution and migrates requests/adapters across replicas. Established by ablation: the merge/unmerge switch itself is worth a large share of the gain when the adapter distribution is skewed. Reported up to ~1.8× over S-LoRA and much larger factors over vLLM/PEFT.
- **LoRAX** (Predibase, open source, 2023–) and vLLM's multi-LoRA path: production implementations; no peer-reviewed ablation.
- **CaraServe** (Li et al., 2024): CPU-assisted prefill for cold adapters, rank-aware scheduling. Claimed but not independently reproduced.
- **Chameleon** (Iliakopoulou et al., 2024): adapter caching plus a multi-queue non-preemptive scheduler; claimed large P99 TTFT reductions under skew. Claimed, not independently ablated.

**Theory SOTA:** essentially absent. No competitive-ratio result exists for online adapter-aware batching. The closest formal neighbours are scheduling with family setup times and online paging/caching ($k$-competitiveness for LRU), neither of which covers the joint batch-and-cache decision with unknown job lengths.

## 4. What Is Known

- **Unmerged LoRA is cheap in decode, expensive in prefill.** For $r=16$, $d_{\text{model}}=4096$, the LoRA path adds $O(r/d_{\text{model}}) \approx 0.4\%$ of base FLOPs but a much larger share of wall time because it is bandwidth-bound. Measured on A100/A10G in Punica and S-LoRA.
- **Kernel cost is near-flat in adapter count during decode.** SGMV/BGMV keep decode latency roughly constant as $d$ grows from 1 to the batch size, at 7B scale on A100-40/80GB.
- **Adapter memory is small; the pool is not.** One rank-16 adapter on Llama-2-7B (all attention projections) is roughly 20 MB in fp16; 2,000 of them is ~40 GB — larger than the base model, which is why host-memory paging, not on-GPU residency, is the operating point.
- **Merging wins under low skew-free concentration.** When one adapter dominates a replica's traffic, merging it into $W$ removes the LoRA path entirely; dLoRA's gain over S-LoRA comes largely from detecting that regime online.
- **Continuous batching is the substrate.** Iteration-level scheduling (Orca, OSDI 2022) and PagedAttention (vLLM, SOSP 2023) are prerequisites; multi-LoRA policies are layered on them, and chunked-prefill interference (Sarathi-Serve, OSDI 2024) applies unchanged.
- **Adapter quality is real, so the workload is real.** LoRA Land (Predibase, 2024) fine-tuned 310 adapters on one 7B base and reported task-level parity with much larger models — evidence that $N$ in the hundreds is a genuine deployment shape, not a synthetic premise.

## 5. What Is Not Known

- **Theoretically open.** No competitive ratio for online adapter-aware batching with a capacity-limited adapter cache and unknown output lengths. Not even a lower bound separating any online policy from the offline optimum. Nobody has published the NP-hardness reduction for the offline case either, though it is straightforward.
- **Empirically open.** How much headroom remains above the current heuristics? No paper reports the **offline optimum** (or an LP/ILP relaxation bound) on a real trace, so every reported speedup is relative to another heuristic. Runnable today at 7B scale on a single A100; simply unrun.
- **Empirically open.** Mixed-rank pools. Published evaluations mostly fix $r$ or use a narrow range; the cost of ragged-rank batching versus padding to $r_{\max}$ has no clean measurement.
- **Methodologically blocked.** There is no shared multi-LoRA workload trace. Each system evaluates on its own synthetic adapter-popularity distribution (Poisson arrivals, Zipf or uniform over adapters), and the ranking of policies is known to depend on the skew parameter. The measurement "system A beats system B" is therefore not yet well defined across papers.
- **Methodologically blocked.** Quality-under-batching. Whether numerically fused multi-adapter kernels are bit-comparable to per-adapter execution is asserted, rarely audited; no benchmark reports task accuracy per adapter under multi-tenant batching versus isolated serving.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent ground truth**.

- *Confounded:* a multi-LoRA speedup mixes at least four effects — kernel design (SGMV vs. loop-over-adapters), memory management (paged vs. static), admission/scheduling policy, and merge/unmerge decisions. Papers vary all four between arms. Punica's 12× and S-LoRA's 4× are not measuring the same variable.
- *Absent ground truth:* there is no computed offline optimum on any trace, so "optimal" in the problem title is unanchored. Every claim is a heuristic-vs-heuristic delta.
- *Non-stationarity:* output length is unknown at admission, so the batch composition chosen at iteration $t$ commits GPU memory for a duration the scheduler cannot see. This is what blocks importing the clean paging results — the "page" has an unknown lifetime.
- Compute cost is *not* the obstruction: the decisive experiments run on one or two GPUs.

## 7. Current Research (as of 2026)

- **Rank-aware and heterogeneous-adapter kernels** — ragged-rank SGMV variants avoiding padding to $r_{\max}$; extensions to DoRA and other PEFT forms *(frontier — verify)*.
- **Cache-and-schedule co-design** — Chameleon-style adapter caching with SLO-aware queueing; the open question is whether caching and batching can be decoupled without loss *(frontier — verify)*.
- **Disaggregated prefill/decode with adapters** — adapter residency differs sharply between the two phases; prefill-decode disaggregation changes the placement problem qualitatively. Active in the vLLM and SGLang communities *(frontier — verify)*.
- **Speculative decoding × multi-LoRA** — draft models shared across adapters; interaction with batch composition unstudied *(frontier — verify)*.
- Groups: UW (Punica lineage), UC Berkeley Sky Computing / LMSYS (S-LoRA, vLLM, SGLang), Peking University (dLoRA), ETH Zurich (Chameleon), Microsoft Research, Predibase.

## 8. Concrete Next Experiment

**Establish the missing ground truth: measure the optimality gap of current schedulers.**

- **Scale:** Llama-2-7B (or Llama-3-8B) on a single A100-80GB. $N = 500$ adapters, ranks drawn from $\{8, 16, 32, 64\}$. One 60-minute request trace, ~50k requests, adapter popularity Zipf with $\alpha \in \{0.0, 0.6, 1.2\}$ (uniform → heavy skew), arrivals replayed at three load levels (50%, 80%, 95% of saturation).
- **Treatment arms:** S-LoRA policy; dLoRA policy; a simple adapter-agnostic FCFS continuous-batching baseline.
- **Control arm — the point of the experiment:** an **offline oracle** given the full trace *and* true output lengths, solved as a mixed-integer program (or a Lagrangian/LP relaxation, which suffices as a bound) over iteration-level batch composition and adapter residency, using a cost model $C(B, d, \{r_j\})$ fitted from measured kernel timings on the same GPU. Report the relaxation bound if the MIP does not close.
- **Deciding number:** the **optimality gap** $g = 1 - T_{\text{policy}} / T_{\text{oracle}}$, where $T$ is SLO-attaining throughput (TTFT P99 $\leq$ 2 s, TPOT P99 $\leq$ 50 ms), reported per skew level.
  - $g < 0.10$ at every skew level ⇒ the method variant is effectively closed; move the field's effort to kernels and to the mixed-rank case.
  - $g > 0.30$ at any skew level ⇒ a real scheduling gap exists, and the target for new policies is quantified rather than rhetorical.
- **Secondary measurement, one afternoon:** sweep $d$ from 1 to $B$ at fixed $B \in \{32, 128, 512\}$, prefill and decode separately, and publish $\kappa = \partial C_{\text{lora}}/\partial d$ in µs per additional distinct adapter. Every scheduler in this area assumes $\kappa \approx 0$; nobody publishes it.

## 9. Key References

- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI, 2022.
- **[Foundational]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[SOTA]** Lequn Chen, Zihao Ye, Yongji Wu, Danyang Zhuo, Luis Ceze, Arvind Krishnamurthy. *Punica: Multi-Tenant LoRA Serving.* MLSys, 2024. — arXiv:2310.18547
- **[SOTA]** Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica. *S-LoRA: Serving Thousands of Concurrent LoRA Adapters.* MLSys, 2024. — arXiv:2311.03285
- **[SOTA]** Bingyang Wu, Ruidong Zhu, Zili Zhang, Peng Sun, Xuanzhe Liu, Xin Jin. *dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving.* OSDI, 2024.
- **[Related]** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024. — arXiv:2403.02310
- **[Related]** Zhengxin Zhang, Dan Zhao, Xupeng Miao, et al. *CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference.* arXiv preprint, 2024.
- **[Related]** Nikoleta Iliakopoulou, Jovan Stojkovic, Chloe Alverti, Tianyin Xu, Hubertus Franke, Josep Torrellas. *Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments.* arXiv preprint, 2024.
- **[Related]** Zhen-Zhe Zhou, Xuanzhe Liu, et al. *PetS: A Unified Framework for Parameter-Efficient Transformers Serving.* USENIX ATC, 2022.
- **[Workload evidence]** Justin Zhao, Timothy Wang, Wael Abid, Geoffrey Angus, Arnav Garg, Jeffery Kinnison, Alex Sherstinsky, Piero Molino, Travis Addair, Devvret Rishi. *LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report.* arXiv preprint, 2024. — arXiv:2405.00732

## 10. Worked Example

Llama-2-7B, fp16, one A100-80GB. Base weights 13.5 GB; ~62 GB left for KV cache plus adapters. A rank-16 adapter on $q,k,v,o$ across 32 layers is $32 \times 4 \times 2 \times 16 \times 4096 \times 2\,\text{B} \approx 33$ MB.

Two adapters: $a$ (rank 16, 90% of traffic) and $b$ (rank 64, 10%). Decode batch $B = 64$.

**Arm 1 — merge $a$ into $W$, serve $b$ unmerged.** Requests for $a$ pay zero LoRA cost. But merging is a 13.5 GB weight rewrite; at ~1.5 TB/s effective read+write that is roughly 18 ms per switch, and it must be undone when $a$'s share falls. If the traffic mix flips every 2 s, the server spends ~1.8% of wall clock merging — acceptable. If it flips every 100 ms, it spends ~36% — catastrophic. dLoRA's contribution is detecting which regime it is in.

**Arm 2 — both unmerged via SGMV.** Cost is flat in $d$, but the kernel must handle $r=16$ and $r=64$ in one call. Padding both to $r_{\max}=64$ inflates the rank-16 rows' LoRA work by $4\times$. That work is small in absolute terms — but it is $4\times$ larger on 90% of the rows, and it is memory-bound, so the inflation lands directly on latency, not on spare FLOPs.

**Where the obstruction becomes visible.** To choose between the arms you need $\kappa$ and the padding penalty at *this* $B$ and *this* rank mix, and you need the flip rate of the traffic mix. None of the three is published. Worse, the two arms cannot be compared on any shared trace: Punica-style and dLoRA-style evaluations use different adapter-popularity distributions, and the ranking of the two arms inverts as Zipf $\alpha$ moves from 0 to 1.2 — merging wins under skew, SGMV wins under uniformity. So a practitioner facing this exact 2-adapter configuration cannot read the answer off the literature, and there is no oracle number to say how far either arm is from optimal. That gap is the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*