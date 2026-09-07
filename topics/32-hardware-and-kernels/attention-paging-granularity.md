---
id: 32-hardware-and-kernels/attention-paging-granularity
title: "Optimal Paging Granularity for Attention Memory Management"
topic: 32-hardware-and-kernels
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Paging Granularity for Attention Memory Management

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/attention-paging-granularity` · **Status:** partially-solved

## 1. Problem Statement

Paged KV-cache allocation splits each sequence's key/value cache into fixed-size blocks of $B$ tokens, indexed by a per-sequence block table. $B$ is a free parameter. Every deployed system picks one by hand: vLLM defaults to 16, TensorRT-LLM to 32–64, vAttention to a CUDA virtual-memory page (2 MiB, or 64/256 KiB with a patched driver), SGLang's radix cache to 1 for sharing and a larger value for the kernel.

The problem: **given a model, a hardware target, and a request-arrival process, choose $B$ to maximize serving goodput under a fixed memory budget — and say whether a single $B$ can be optimal across workloads at all.**

Three variants, with different difficulty:

- **Measurement.** Attribute an end-to-end throughput delta to $B$, decomposing it into internal fragmentation, prefix-sharing loss, kernel-level gather/TLB cost, and scheduler admission effects. Currently confounded: changing $B$ moves all four at once.
- **Method.** Produce a selection rule $B^*(\text{model}, \text{GPU}, \text{workload})$, or a design that removes the tradeoff (variable-size pages, virtual-memory-backed contiguity, two-level paging).
- **Theory.** Prove bounds on the achievable memory-utilization/throughput frontier for online paged allocation with unknown output lengths — the online-bin-packing-with-unknown-item-sizes structure has no tight analysis in this setting.

Solved would mean: a rule that, for an unseen workload, picks $B$ within 2% of the best value found by exhaustive sweep, plus a decomposition showing which term binds.

## 2. Formal Setting

Model: $L$ layers, $n_{kv}$ KV heads, head dim $d_h$, element size $s$ bytes. Per-token KV footprint, all layers:

$$c \;=\; 2\,L\,n_{kv}\,d_h\,s \quad \text{bytes/token}.$$

Measured as: allocated bytes per token reported by the allocator, not the analytic value — quantized or MLA-compressed caches diverge.

Requests $i=1..N$ with prompt length $p_i$ and decode length $g_i$; live length $\ell_i(t)$. Allocation at page size $B$:

$$M(t;B) \;=\; c \sum_{i \in \mathcal{A}(t)} B\left\lceil \frac{\ell_i(t)}{B} \right\rceil, \qquad \text{waste } W(t;B)=M(t;B)-c\textstyle\sum_i \ell_i(t).$$

For lengths uniform mod $B$, $\mathbb{E}[W] \approx c\,|\mathcal{A}|\,(B-1)/2$. Measured as: peak allocated blocks minus tokens actually resident, sampled per scheduler step.

Prefix sharing. If two requests share a prefix of $q$ tokens, a page-granular radix cache shares only $B\lfloor q/B\rfloor$. Sharing efficiency

$$\eta(B) \;=\; \mathbb{E}_q\!\left[\frac{B\lfloor q/B\rfloor}{q}\right] \;\approx\; 1 - \frac{B-1}{2\,\mathbb{E}[q]}.$$

Measured as: cached-token hit count from the prefix-cache counter, divided by the hit count at $B=1$ replayed on the same trace.

Kernel term. Paged attention reads $\lceil \ell/B \rceil$ block pointers per sequence per layer. Let $T_\text{attn}(B,\ell,b)$ be measured decode-step attention latency at batch $b$. The empirical shape is $T_\text{attn}$ falling steeply from $B=1$ to $B\approx 16$ (indirection amortized, coalesced loads) then flat or mildly rising (tail-block waste inside the kernel, worse load balance across CTAs).

Objective — goodput under an SLO:

$$B^* \;=\; \arg\max_B \; \frac{\mathbb{E}[\text{completed requests}]}{\text{wall clock}} \quad \text{s.t.} \quad P(\text{TPOT} > \tau) \le \epsilon,\; \max_t M(t;B) \le M_\text{gpu}.$$

Assumptions and their violations:
- *Uniform per-token KV size across layers.* Violated by sliding-window/full hybrids (Gemma-2, Mistral), by MLA (DeepSeek-V2/V3: ~576 latent dims/token vs. thousands), and by per-layer KV quantization.
- *Lengths independent of $B$.* Violated: larger $B$ shrinks the admitted batch, which changes preemption and recompute, which changes effective lengths.
- *Sharing is prefix-only.* Violated by CacheBlend-style non-prefix reuse.
- *Fragmentation is internal only.* True for uniform $B$; false the moment pages are variable-sized, where external fragmentation returns.

## 3. State of the Art

**Established.**
- PagedAttention (Kwon et al., SOSP 2023) cut KV waste from 60–80% in contiguous-allocation systems to under 4%, with 2–4× throughput over Orca at equal latency. Its block-size ablation is the only public sweep over $B$ in the original setting: performance degrades at $B \le 4$ (insufficient parallelism, pointer overhead) and at $B \ge 64$ under parallel sampling/beam search (fragmentation plus lost sharing). $B \in \{16,32\}$ was best. This is a single-model, single-GPU-generation result.
- vAttention (Prabhu et al., ASPLOS 2025) established that the *paging mechanism itself* costs kernel throughput: retaining virtual contiguity via CUDA VMM and mapping physical pages on demand lets unmodified FlashAttention kernels run, recovering the gap that the paged kernel variants pay. Reported prefill throughput gains up to ~1.2–1.97× over vLLM depending on setting.
- SGLang's RadixAttention (Zheng et al., NeurIPS 2024) established that token-granular prefix reuse gives large hit-rate gains on structured/multi-turn workloads — evidence that $\eta(B)$ matters, though the paper does not sweep $B$ against it.
- FlashInfer (Ye et al., MLSys 2025) established that a block-sparse row (BSR) formulation makes attention kernels page-size-agnostic in code, with a load-balancing scheduler, and reports inter-token-latency reductions of 29–69% versus prior serving kernels.

**Claimed but unablated.** That 16 is a good default for modern GQA models on H100/H200/B200 — inherited from a 2023 MHA-era measurement, not re-derived. That larger pages are "better for long context" — plausible from TLB/indirection arguments, not isolated from the batch-size change large pages force.

**Benchmark-number-only.** Vendor defaults (TRT-LLM `tokens_per_block`, cuDNN paged-attention block sizes) are tuning artifacts with no published decomposition.

## 4. What Is Known

- Waste under contiguous allocation: 60–80% of KV memory; PagedAttention $\le 4$%, measured on OPT-13B/OPT-175B and LLaMA-13B, A100-40GB (SOSP 2023).
- The $B$ sweep in that paper spans $\{1,2,4,8,16,32,64,128\}$; the U-shape is real and reproduced by practitioners, but the flat region's width is model-dependent.
- Analytic fragmentation is exact and small at small $B$: for Llama-3-8B ($c = 128$ KiB/token), $B=16$ costs on average 7.5 tokens/sequence ≈ 0.94 MiB — 0.15% of a 64 GiB cache at 256 concurrent sequences. At $B=256$ the same figure is ~4 GiB, ~6.4%.
- Sharing loss is first-order for short shared prefixes: with $\mathbb{E}[q]=1000$ and $B=256$, $\eta \approx 0.87$; with $B=16$, $\eta \approx 0.99$.
- Chunked-prefill scheduling (Sarathi-Serve, OSDI 2024) changes the memory-pressure regime that $B$ operates in — the optimum is not schedule-independent.
- MLA and GQA shrink $c$ by 4–60×, which shrinks the absolute cost of a page and therefore permits much larger $B$ at the same byte-waste. Known analytically; not measured as a $B$ retuning study.

## 5. What Is Not Known

- **Empirically open.** No published sweep of $B$ on a modern GQA/MLA model, on Hopper/Blackwell, under a realistic arrival trace, with the four cost terms measured separately. The experiment is entirely runnable — a few hundred GPU-hours — and nobody has published it.
- **Empirically open.** Whether $B^*$ is stable across (model, GPU, workload) or varies enough that a static default costs >5% goodput.
- **Methodologically blocked.** Attributing a goodput delta to fragmentation versus kernel efficiency: no serving system exposes counters that separate "blocks allocated but unfilled" from "kernel time lost to indirection". Both move when $B$ moves.
- **Theoretically open.** Competitive ratio for online paged KV allocation with unknown output lengths and prefix sharing. Classical bin-packing bounds do not apply (items grow after placement; items share).
- **Theoretically open.** Whether a variable-granularity scheme (small pages near the tail, large pages for the body) is strictly dominant, or whether its metadata and external fragmentation cancel the gain.

## 6. Why It Is Hard

**The measurement is confounded by an admission-control feedback loop.** Changing $B$ changes free-block count, which changes the maximum admitted batch, which changes both arithmetic intensity per attention call and the preemption/recompute rate. So a throughput difference between $B=16$ and $B=64$ is never a clean kernel measurement — the two arms are running different batch sizes on different sequence-length mixes. Holding batch size fixed to de-confound destroys the thing being optimized (memory-limited concurrency is the entire point of paging). There is no ground-truth counterfactual "same workload, same batch, different page size", because page size *is* the batch-size control knob.

Second obstruction: **the kernel term is not a function of $B$ alone.** It depends on kernel implementation (BSR vs. hand-written paged kernel), on whether virtual contiguity is preserved (vAttention), and on hardware page/TLB behavior. A sweep is therefore a sweep over an implementation, not over the abstraction.

## 7. Current Research (as of 2026)

- **Removing the tradeoff rather than tuning it.** vAttention (Microsoft Research India) and successors keep pages at OS/driver granularity and recover token-granularity by demand mapping. Open question: whether sub-2 MiB CUDA pages become a supported API rather than a driver patch *(frontier — verify)*.
- **Page-size-agnostic kernels.** FlashInfer (UW/CMU/NVIDIA lineage, now upstreamed into vLLM and SGLang) makes $B$ a runtime parameter, which finally makes the sweep cheap to run.
- **Heterogeneous KV.** Jenga and related work on per-layer/per-model heterogeneous KV sizes (hybrid attention, MLA, sliding window) implies per-layer page sizes; no published optimum *(frontier — verify)*.
- **Disaggregated and tiered caches.** Mooncake (FAST 2025), DistServe (OSDI 2024), LMCache push KV to CPU/SSD/remote memory, where the transfer granularity is a *second*, larger page size — and the two granularities interact.
- **Prefix-cache-aware routing.** Making $\eta(B)$ a routing objective rather than an allocator side effect.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B (GQA, $c=128$ KiB/token) and DeepSeek-V2-Lite (MLA), single H100-80GB, vLLM or SGLang with the FlashInfer backend so $B$ is a runtime flag. Trace: 30 minutes of ShareGPT arrivals plus a synthetic multi-turn arm with mean shared prefix 1000 tokens, replayed identically per arm.

**Arms.** $B \in \{1, 8, 16, 32, 64, 128, 256\}$, crossed with prefix caching on/off.

**Control arm.** vAttention-style virtually-contiguous allocation (or `B = ∞` with static per-sequence reservation at the 95th-percentile length). This is the "no paging tax" upper bound on kernel efficiency and the lower bound on memory efficiency; every paged arm must be read against it.

**Instrumentation (the point of the experiment).** Per scheduler step log: allocated blocks, resident tokens (→ $W$), prefix-cache hit tokens (→ $\eta$), admitted batch size, and attention-kernel time from CUDA events separated from the rest of the step.

**Deciding number.** $\Delta = \dfrac{\text{goodput}(B^*_\text{measured})}{\text{goodput}(B=16)} - 1$ at a fixed P99 TPOT SLO of 50 ms. If $\Delta < 0.02$ on both models and both workload arms, the default is vindicated and the problem closes to "solved by convention". If $\Delta > 0.05$ on any arm — or if $B^*$ differs between the two models — the static default is a real loss and $B$ must become an autotuned parameter.

## 9. Key References

- **[Foundational]** Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Foundational]** Yu, Jeong, Kim, Kim, Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI, 2022.
- **[SOTA]** Prabhu, Nayak, Mohan, Ramjee, Panwar. *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention.* ASPLOS, 2025. — arXiv:2405.04437
- **[SOTA]** Ye, Chen, Ye, Lin, Xia, Wu, Zhai, Chen, Ceze et al. *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.* MLSys, 2025. — arXiv:2501.01005
- **[SOTA]** Zheng, Yin, Xie, Sun, Huang, Yu, Cao, Kozyrakis, Stoica, Gonzalez, Barrett, Sheng. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS, 2024. — arXiv:2312.07104
- **[Related]** Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov, Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024. — arXiv:2403.02310
- **[Related]** Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI, 2024.
- **[Related]** Qin, Cheng, Zhao, Chen, Cheng, Wang, Xu et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* FAST, 2025. — arXiv:2407.00079
- **[Related]** Gim, Chen, Ko, Yang, Yin, Hu, Ali, Zhong. *Prompt Cache: Modular Attention Reuse for Low-Latency Inference.* MLSys, 2024. — arXiv:2311.04934
- **[Related]** Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP, 2023.
- **[Survey]** Miao, Oliaro, Zhang, Cheng, Jin, Chen, Jia. *Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems.* 2023. — arXiv:2312.15234

## 10. Worked Example

Llama-3.1-8B on one H100-80GB. Weights bf16 = 16 GB; runtime overhead ~4 GB; KV budget $\approx 60$ GiB. Per-token KV: $2 \times 32 \text{ layers} \times 8 \text{ KV heads} \times 128 \times 2\,\text{B} = 128$ KiB. Capacity: $60\,\text{GiB} / 128\,\text{KiB} = 491{,}520$ tokens.

Workload: mean live length 2048 tokens, mean shared prefix 1000 tokens (multi-turn chat).

| $B$ | Frag. tokens/seq (avg 7.5·B/16) | Seqs at capacity | $\eta(B)$ | Effective unique tokens stored |
|---|---|---|---|---|
| 16 | 7.5 | 239 | 0.992 | 491,520 |
| 64 | 31.5 | 236 | 0.968 | 483,000 |
| 256 | 127.5 | 226 | 0.872 | 452,000 |

Going from $B=16$ to $B=256$ costs 13 concurrent sequences (5.4%) and 12% of prefix-cache hits. To be a win, the kernel must get more than ~5% faster.

Now the obstruction. Suppose the measured decode step at $B=256$ *is* 6% faster than at $B=16$. That measurement was taken at batch 226, not 239 — attention at batch 226 has lower memory-bandwidth pressure per step, so part of the 6% is the smaller batch, not the larger page. Rerun $B=16$ at batch 226 and it also speeds up. The two arms cannot be equalized on batch size without deleting the memory effect that motivated the comparison, and no counter in vLLM or SGLang separates "faster because fewer block-table indirections" from "faster because 13 fewer sequences". The 6% is real and uninterpretable at the same time — which is why the field ships $B=16$ by inheritance rather than by measurement.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*