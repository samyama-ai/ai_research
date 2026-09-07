---
id: 16-state-space-models/batched-state-sharing-serving
title: "Recurrent Models and Multi-Query Batched State Sharing"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Recurrent Models and Multi-Query Batched State Sharing

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/batched-state-sharing-serving` · **Status:** open

## 1. Problem Statement

A serving workload where one long context $c$ is followed by many short, mutually independent queries $q_1,\dots,q_B$ (document QA fan-out, agent branching, best-of-$n$ sampling, tree search) is the case where Transformer inference has a clean answer: the prefix KV cache is append-only, so it is computed once and read by all $B$ branches (PagedAttention, Hydragen, RadixAttention).

Recurrent models — Mamba, Mamba-2, RWKV, Griffin/RecurrentGemma, hybrids like Jamba — replace the growing KV cache with a **fixed-size state** $h$ that is *destructively updated*. Forking is therefore a copy, not a reference. The problem:

- **Systems variant.** Given $B$ branches sharing prefix $c$, what is the achievable throughput and peak memory of a recurrent serving stack, and where is the crossover against a prefix-sharing Transformer of matched quality?
- **Method variant.** Is there a state representation or scheduling scheme that lets $B$ branches share one materialized state — factored, delta-encoded, or low-rank — instead of $B$ full copies?
- **Measurement/quality variant.** The state is a *lossy, query-agnostic* summary of $c$. Does answer quality degrade as the branch set diversifies, i.e. is there a query-diversity load beyond which one shared state is provably insufficient where a KV cache is not?

Solved would mean: a public serving system with a stated crossover curve in $(P, B, S)$ = (prefix length, branch count, suffix length), plus a quality curve showing whether shared-state serving loses accuracy relative to per-branch prefill.

## 2. Formal Setting

Model with $L$ layers. Recurrent state per sequence:

$$d_s = L\,(d_{\text{inner}} N + d_{\text{inner}}(d_{\text{conv}}-1)),$$

with $d_{\text{inner}}$ the expanded channel width, $N$ the SSM state dimension, $d_{\text{conv}}$ the short-conv width. **Measured** as bytes of resident device memory per live sequence at steady state (`torch.cuda.max_memory_allocated` delta with $B$ vs $B+1$ sequences), not from the formula — allocator padding and the conv ring buffer are real.

Transformer baseline: bytes per token $d_{kv} = 2 L_T d_{\text{head}} n_{kv} \cdot b$, $b$ = dtype bytes.

Peak cache memory for the fan-out workload:

$$M_{\text{rec}}(P,B,S) = B\,d_s, \qquad M_{\text{tf}}(P,B,S) = P\,d_{kv} + B\,S\,d_{kv}.$$

$M_{\text{rec}}$ is $P$-independent; $M_{\text{tf}}$ pays the prefix **once** under prefix sharing. Recurrent memory wins iff

$$B\,d_s < (P + BS)\,d_{kv} \iff \frac{d_s}{d_{kv}} < \frac{P}{B} + S.$$

Throughput is measured as decoded tokens/s at fixed batch, with time-to-first-token (TTFT) reported at P50/P95 under a Poisson arrival trace, not offline.

Quality: for branch $i$, $A_i^{\text{shared}}$ = accuracy decoding from the forked state versus $A_i^{\text{fresh}}$ = accuracy from a fresh prefill of $c \Vert q_i$. The gap $\Delta = \mathbb{E}_i[A_i^{\text{fresh}} - A_i^{\text{shared}}]$ should be $0$ up to numerics; it is not always, because the fork point may fall mid-chunk in a chunked-scan kernel.

**Assumptions, and where they break.**
1. *States are forkable at any token.* Violated: most kernels checkpoint only at chunk boundaries (Mamba-2 chunk length 64/256), so the reusable prefix is quantized to chunks — Marconi's central observation.
2. *Prefix reuse requires exact match.* Violated in the useful direction only for Transformers: KV caches support token-granular partial reuse; a recurrent state supports **no** suffix editing — deleting the last $k$ tokens of $c$ requires recompute from the last checkpoint.
3. *State is query-agnostic and sufficient.* Known false in the worst case (copying/recall lower bounds, §4).
4. *Fixed state ⇒ flat memory.* True per sequence, false per *branch*: $B$ forks cost $B d_s$.

## 3. State of the Art

**Established (Transformer side).** PagedAttention/vLLM (Kwon et al., SOSP 2023) makes prefix sharing a first-class allocator operation. Hydragen (Juravsky et al., 2024) decomposes attention over a shared prefix into prefix and suffix parts and reports up to $32\times$ end-to-end decoding throughput over vLLM baselines at CodeLlama-13B with long shared prefixes — reproduced in spirit by RadixAttention/SGLang (Zheng et al., 2024). These are ablated: the mechanism (batched GEMM over the shared prefix) is identified, not just a benchmark delta.

**Established (recurrent side).** Constant-memory, constant-per-token decoding for Mamba/Mamba-2 (Gu & Dao, COLM 2024; Dao & Gu, ICML 2024) and $5\times$ decode throughput over a same-size Transformer at 1.4B–2.8B is reproduced across implementations. Jamba (Lieber et al., 2024) reports a 256K context fitting in a single 80GB A100 KV budget for a 52B hybrid.

**Claimed but unablated / benchmark-only.** Marconi (Pan et al., MLSys 2025) builds prefix caching for hybrid Mamba-Transformer models with checkpoint-aware admission and FLOP-aware eviction, reporting up to $34.4\times$ higher token hit rate and large P95 TTFT reductions versus SGLang-style baselines. This is the closest existing work and the numbers are a single-paper benchmark on the authors' traces; the decomposition of the gain into (admission policy) vs (eviction policy) vs (checkpoint granularity) is not independently reproduced.

**Absent.** No published system does *pure* recurrent fan-out — one state, $B$ branches — with a measured crossover surface against Hydragen. There is no recurrent analogue of Hydragen's algebraic decomposition, and it is unclear one exists (§6).

## 4. What Is Known

- **Memory constant per sequence.** Mamba-2 2.7B (64 layers, $d_{\text{inner}}=5120$, $N=128$, bf16): $\approx 84$ MB of SSM state per sequence, independent of context length. A 2.7B MHA Transformer (32 layers, $d=2560$, bf16) costs $\approx 0.33$ MB/token, so $d_s/d_{kv} \approx 256$ tokens.
- **Recall is provably harder.** Jelassi et al., *Repeat After Me* (ICML 2024): constant-state models need $\Omega(n)$ state to copy $n$-token strings; a small Transformer matches or beats a much larger SSM on copy/retrieval. Arora et al. (Zoology, Based; ICML 2024) tie associative-recall accuracy to recurrent state size with an explicit accuracy–state tradeoff curve at 355M–1.3B.
- **Hybrids buy back recall.** Jamba and Zamba-class models with a minority of attention layers close most of the recall gap — but they reintroduce a growing KV cache, so the fan-out arithmetic becomes mixed.
- **Chunked kernels quantize checkpoints.** Mamba-2's SSD formulation is chunkwise; states are naturally available at chunk boundaries, which is exactly the granularity mismatch Marconi has to engineer around.

## 5. What Is Not Known

- **Empirically open.** The crossover surface $B \cdot d_s$ vs $(P + BS)d_{kv}$ under a real serving stack, at quality parity, has not been measured. Every input is available; nobody has published the sweep.
- **Empirically open.** Whether $\Delta$ (shared-state vs fresh-prefill accuracy) is zero. Fork-point numerics and chunk-boundary reconstruction could make it nonzero, and it is never reported.
- **Theoretically open.** Whether a *sublinear-in-$B$* shared representation exists: can $B$ divergent branches from one state be served in $o(Bd_s)$ memory with bounded error? No lower bound and no construction. Hydragen's decomposition works because attention is linear in the value cache; for a nonlinear-in-time gated recurrence there is no known analogous factorization.
- **Methodologically blocked.** "Query diversity" — the load parameter that should predict when one lossy state fails — has no accepted measurement. Without it, degradation cannot be predicted, only observed after the fact.

## 6. Why It Is Hard

The specific obstruction is **destructive state update**: a KV cache is a *set* that only grows, so sharing is aliasing and costs nothing; a recurrent state is a *register* overwritten every token, so sharing requires copying. This is not an engineering gap — it is a consequence of the recurrence, and it converts the headline advantage (state size independent of $P$) into a liability at high fan-out, because memory scales with $B$ instead of with unique content.

Second obstruction: **confounded measurement**. Recurrent and Transformer models of equal parameter count are not of equal quality on retrieval-heavy fan-out tasks (§4), so any throughput comparison silently trades accuracy for speed unless quality is pinned first — which most serving papers do not do.

## 7. Current Research (as of 2026)

- Hybrid prefix caching: Marconi (Princeton/UChicago-affiliated authors, MLSys 2025) and follow-on work integrating checkpoint-aware caching into vLLM/SGLang *(frontier — verify)*.
- Constant-memory long-context serving in production hybrids (AI21 Jamba, Zyphra Zamba, TII Falcon-Mamba, Google RecurrentGemma).
- Attention-to-recurrence distillation (MOHAWK/*Mamba in the Llama*, Wang et al./Bick et al., 2024), which makes quality-matched arms cheap to build and thus makes the crossover experiment feasible.
- Compressed context artifacts trained per-document (Cartridges-style, Stanford Hazy Research, 2025) — an orthogonal attack on the same workload *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Mamba-2 2.7B vs a quality-matched 2.8B Transformer (or the distilled pair from *Mamba in the Llama*), single 80GB H100. Workload: $P = 8192$-token document, $B \in \{1,4,16,64,256\}$ branches, $S = 64$ and $S = 512$ generated tokens, 200 documents from a long-doc QA set.

**Arms.** (A) Recurrent, prefill once, fork state per branch. (B) **Control:** Transformer with vLLM prefix sharing + Hydragen decomposition. (C) Recurrent with per-branch full prefill (quality reference for $\Delta$).

**Deciding number.** The branch count $B^\*$ at which arm A's peak cache memory exceeds arm B's at equal decoded-token throughput. The arithmetic predicts $B^\* \approx P/(d_s/d_{kv} - S) = 8192/(256-64) \approx 43$ at $S=64$, and *no crossover* ($S > 256$) at $S=512$. If measured $B^\*$ lands within $\pm 25\%$ of prediction, the problem reduces to engineering; if it does not, the resident-state model is wrong and the memory accounting needs revision. Secondary gate: report $\Delta = A^{\text{fresh}} - A^{\text{shared}}$; anything above 0.5 points means forking is not free and the systems result is void.

## 9. Key References

- **[Foundational]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Foundational]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[SOTA]** Woosuk Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[SOTA]** Jordan Juravsky, Bradley Brown, Ryan Ehrlich, Daniel Y. Fu, Christopher Ré, Azalia Mirhoseini. *Hydragen: High-Throughput LLM Inference with Shared Prefixes.* 2024. — arXiv:2402.05099
- **[SOTA]** Lianmin Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS, 2024. — arXiv:2312.07104
- **[SOTA]** Rui Pan et al. *Marconi: Prefix Caching for the Era of Hybrid LLMs.* MLSys, 2025. — arXiv:2411.19379
- **[Theory]** Samy Jelassi, David Brandfonbrener, Sham Kakade, Eran Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Theory]** Simran Arora et al. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML, 2024. — arXiv:2402.18668
- **[Systems]** Opher Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* 2024. — arXiv:2403.19887
- **[Method]** Junxiong Wang, Daniele Paliotta, Avner May, Alexander M. Rush, Tri Dao. *The Mamba in the Llama: Distilling and Accelerating Hybrid Models.* NeurIPS, 2024. — arXiv:2408.15237

## 10. Worked Example

One 8192-token contract, 64 questions asked in parallel, 64 tokens each.

**Recurrent (Mamba-2 2.7B).** State per sequence $\approx 84$ MB. Prefill once, fork 64 times:

$$M_{\text{rec}} = 64 \times 84\ \text{MB} = 5.4\ \text{GB}.$$

**Transformer (2.8B MHA, prefix-shared), $d_{kv} = 0.33$ MB/token.**

$$M_{\text{tf}} = 8192 \times 0.33\ \text{MB} + 64 \times 64 \times 0.33\ \text{MB} = 2.70 + 1.35 = 4.05\ \text{GB}.$$

The recurrent model — the one advertised as constant-memory — uses **33% more cache memory** than the Transformer it was supposed to beat. Raise the fan-out to $B=256$: recurrent 21.5 GB, Transformer $2.70 + 5.40 = 8.1$ GB, a $2.7\times$ loss. Drop to $B=1$: recurrent 0.084 GB, Transformer 2.72 GB, a $32\times$ win.

The obstruction is visible in the two terms. The Transformer pays $P d_{kv}$ **once** because the prefix cache is append-only and can be aliased. The recurrent model pays $d_s$ **per branch** because the state is overwritten in place and forking is a `memcpy`. Constant state size buys independence from $P$ and gives up independence from $B$ — and fan-out workloads are exactly the ones where $B$, not $P$, is what grows. Nobody has published this curve on real hardware; the numbers above are arithmetic from published architecture configs, which is precisely why §8 is worth running.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*