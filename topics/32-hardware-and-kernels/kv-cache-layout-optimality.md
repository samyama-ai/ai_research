---
id: 32-hardware-and-kernels/kv-cache-layout-optimality
title: "KV Cache Memory Layout Optimality for Paged Attention"
topic: 32-hardware-and-kernels
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# KV Cache Memory Layout Optimality for Paged Attention

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/kv-cache-layout-optimality` · **Status:** partially-solved

## 1. Problem Statement

Paged attention stores the key/value cache in fixed-size blocks scattered across GPU memory, with a per-sequence block table mapping logical token positions to physical blocks. This buys near-zero external fragmentation and cheap prefix sharing. It costs the attention kernel a level of indirection and the loss of guaranteed contiguity.

The problem: **given a model shape, a serving workload, and a GPU, choose the physical layout of the KV cache — block size, axis order within a block, K/V interleaving, and the allocation policy — that minimizes end-to-end serving cost.** Input: model config $(L, H_{kv}, d_h, \text{dtype})$, request trace, hardware descriptor. Output: a layout $\Lambda$. Objective: minimize token latency at fixed throughput, or maximize throughput at fixed SLO.

Three variants, different difficulty:

- **Measurement.** Attribute a measured latency delta between two layouts to layout alone, not to kernel implementation maturity. Currently the weak link.
- **Method.** Search or auto-tune $\Lambda$ per (model, hardware, workload). Partially solved by hand-tuning; no published autotuner covers the joint space.
- **Theory.** Prove a lower bound on data movement for paged attention decode and show a layout attains it. Open.

Status is *partially-solved*: paging beats contiguous pre-allocation on memory efficiency by a large, reproduced margin; whether current layouts are near-optimal on the kernel side is not established.

## 2. Formal Setting

Model: $L$ layers, $H_{kv}$ KV heads, head dim $d_h$, $b$ bytes per element. Per-token KV footprint:

$$m_{\text{tok}} = 2 L H_{kv} d_h b \ \text{bytes}$$

Cache is partitioned into blocks of $B$ tokens. A sequence of length $s$ occupies $\lceil s/B \rceil$ blocks per layer. **Internal fragmentation**, measured as allocated-minus-used bytes divided by allocated:

$$\phi(B, s) = \frac{\lceil s/B\rceil B - s}{\lceil s/B\rceil B} \approx \frac{B-1}{2s} \ \text{in expectation over } s \bmod B$$

A layout $\Lambda$ is an injective map
$$\Lambda: (\ell, \text{block}, h, t, j) \mapsto \text{byte offset}$$
over layer, block index, head, intra-block token, and head-dim coordinate. vLLM's K cache uses `[n_blocks, H_kv, d_h/x, B, x]` with $x = 16/b$ (8 for fp16) so that a warp lane issues a 16-byte vector load; V uses `[n_blocks, H_kv, d_h, B]`. FlashInfer instead exposes NHD and HND variants of a unified block-sparse-row format.

Cost model. Decode attention over batch $N$, mean length $\bar s$, on a GPU with HBM bandwidth $\beta$:

$$T_{\text{ideal}} = \frac{N \bar s\, m_{\text{tok}}}{\beta}, \qquad T_{\text{measured}} = \frac{T_{\text{ideal}}}{\eta(\Lambda)}$$

where $\eta(\Lambda) \in (0,1]$ is **achieved bandwidth efficiency** — measured as bytes of KV touched (analytically known) divided by kernel wall time divided by $\beta$ (measured with `ncu`, `dram__bytes_read.sum` and `gpu__time_duration.sum`). Indirection overhead is the extra instruction and latency cost of the block-table gather, measured as $\eta(\Lambda_{\text{paged}})/\eta(\Lambda_{\text{contig}})$ with the *same* kernel skeleton.

End-to-end objective, with $U$ the fraction of HBM usable for KV after weights and activations:

$$\max_{\Lambda}\ \text{throughput} \quad \text{s.t.}\quad N \le \frac{U \cdot C_{\text{HBM}}}{\bar s\, m_{\text{tok}} (1 + \phi)}$$

**Assumptions, and where they break.**
- *Uniform HBM cost per byte.* Violated: L2 (50 MB on H100) captures shared prefixes; hit rate depends on layout and on block-table order, so $\eta$ is not layout-separable.
- *Blocks are independent.* Violated: TMA (Tensor Memory Accelerator) bulk copies and `cp.async.bulk` want contiguity larger than $B{=}16$ tokens; a layout that satisfies paging can forfeit the fastest copy path.
- *Fixed $B$ across requests.* Violated by prefix-sharing workloads where a large $B$ coarsens sharing granularity.
- *Decode is purely memory-bound.* Violated under chunked prefill and speculative decoding, where the attention kernel sees a non-trivial query length and becomes partly compute-bound.

## 3. State of the Art

**Established (reproduced, ablated).**
- PagedAttention (Kwon et al., SOSP 2023) reports that pre-paging serving systems wasted 60–80% of KV memory to internal and external fragmentation, and that vLLM reduces waste to under 4%. Throughput gain of 2–4× over Orca/FasterTransformer at equal latency. The memory-waste measurement has been reproduced widely; it is the load-bearing result.
- Prefix sharing via block-table aliasing (RadixAttention in SGLang, Zheng et al., NeurIPS 2024) is a direct consequence of paging and is independently reproduced.

**Claimed but not fully ablated.**
- vAttention (Prabhu et al., ASPLOS 2025) argues paging is the wrong mechanism: use CUDA virtual-memory APIs (`cuMemCreate`/`cuMemMap`) to keep the KV cache *virtually contiguous* while backing it with physical pages on demand. Reports vLLM's paged kernel up to **2.85× slower** than an equivalent non-paged FlashAttention kernel, and up to 1.97× higher prefill throughput end to end. The 2.85× is a kernel-pair comparison, not a controlled layout ablation — the two kernels differ in more than indirection.
- FlashInfer (Ye et al., MLSys 2025) unifies paged, ragged and radix-tree KV under block-sparse row format with a compile-time layout choice, reporting 29–69% inter-token-latency reduction against Triton-based backends. These are benchmark numbers against a specific baseline stack; the paper does not isolate layout from scheduler and JIT effects.
- Block size defaults are folklore. vLLM ships $B{=}16$; TensorRT-LLM has used 32/64/128. No published sweep isolates $B$ at fixed kernel across context lengths.

**Theory SOTA.** None specific to paged attention. The nearest tool is the red-blue pebble game I/O lower bound (Hong & Kung, STOC 1981), which gives $\Omega(n^3/\sqrt{M})$ for dense matmul but has not been instantiated for the gather-then-reduce structure of paged decode.

## 4. What Is Known

- **Fragmentation is analytically negligible at long context.** Llama-3-8B, $\bar s = 8192$: $\phi(16, 8192) \approx 0.09\%$; $\phi(256, 8192) \approx 1.6\%$. Even $B{=}256$ costs ~1.6% of KV memory. The original 60–80% waste figure came from *per-request contiguous pre-allocation to max length*, not from block granularity.
- **Decode is bandwidth-bound at realistic batch.** Arithmetic intensity of GQA decode is $\approx 2 G$ FLOP per byte with group size $G = H_q/H_{kv}$; for Llama-3-8B ($G{=}4$, fp16) that is ~4, far below H100's ~590 FLOP/byte ridge point. Layout matters only through $\eta$.
- **KV shrinkage moves the goalposts.** MQA (Shazeer, 2019) and GQA (Ainslie et al., EMNLP 2023) cut $H_{kv}$ by 4–8×; DeepSeek-V2's MLA (2024) reports 93.3% KV reduction vs. DeepSeek-67B. Smaller $m_{\text{tok}}$ means a block of $B$ tokens is a smaller contiguous run, which makes coalescing *harder*, not easier.
- **Shared-prefix workloads are a different regime.** Hydragen (Juravsky et al., 2024) reports up to 32× throughput gains by decomposing attention over a shared prefix and per-sequence suffixes at 8B scale — a restructuring the block table enables but the layout does not itself deliver.
- **Layout interacts with scheduling.** Sarathi-Serve (Agrawal et al., OSDI 2024) shows chunked prefill changes the query length the attention kernel sees, and therefore which layout is favored.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on HBM traffic (or on achievable $\eta$) for paged decode as a function of $B$, cache size $M$, and prefix-sharing structure. No proof that any published layout is within a constant factor of optimal.
- **Empirically open.** The clean ablation — one kernel, one hardware, one model, sweep $B \in \{1,16,32,64,128,256,512\}$ and axis order, holding scheduler and everything else fixed — has not been published at frontier scale. Runnable today on a single H100 node. Likewise, whether the vAttention 2.85× survives a matched-kernel control is unresolved.
- **Methodologically blocked.** "Cost of paging" has no agreed operational definition. Papers report kernel-pair wall-clock deltas that fold in indirection cost, tiling strategy, TMA use, JIT specialization, and scheduler differences. Until the community fixes a control kernel, the numbers are not comparable across papers.

## 6. Why It Is Hard

**Confounded measurement.** Every published paged-vs-contiguous comparison changes the kernel and the layout simultaneously. FlashAttention's non-paged kernel uses TMA bulk copies and a tile schedule that assumes contiguity; the paged kernel does not. The measured 2.85× therefore bounds *(indirection + lost TMA + implementation gap)*, and the decomposition is unreported. Building the control — a single kernel parameterized by layout, otherwise identical — is a few thousand lines of CUTLASS/Triton and is exactly the work nobody has published.

**Non-identifiability from L2.** Two layouts can have identical DRAM traffic and differ 20% in wall time via L2 hit rate on shared prefixes, which depends on the *request trace*, not the layout. So $\eta(\Lambda)$ is workload-conditional; there is no single optimum to find.

Compute cost is not the obstruction. A full $B$ sweep is hours on one GPU.

## 7. Current Research (as of 2026)

- **Virtual-memory KV management.** vAttention (Microsoft Research India) and follow-ons using CUDA VMM to decouple contiguity from allocation. NVIDIA has been moving TensorRT-LLM toward similar VMM-backed pools *(frontier — verify)*.
- **Composable attention formats.** FlashInfer (UW / CMU / NVIDIA; Ye, Ceze et al.) treating layout as a compile-time parameter over block-sparse formats — the closest thing to an autotuner over $\Lambda$.
- **Layout co-design with attention variants.** MLA and sparse/selective-KV attention (DeepSeek, Moonshot) change $m_{\text{tok}}$ by an order of magnitude and reopen block-size choice *(frontier — verify)*.
- **Disaggregated prefill/decode serving** (Mooncake, DistServe lineage) gives prefill and decode different KV layouts and requires a transfer format between them — an unstudied third layout.

## 8. Concrete Next Experiment

**Question:** how much of the paged-vs-contiguous kernel gap is indirection, and what is the optimal $B$?

- **Scale.** Llama-3-8B (32 layers, $H_{kv}=8$, $d_h=128$, fp16, $m_{\text{tok}} = 128$ KiB) on one H100 SXM (80 GB, $\beta = 3.35$ TB/s). Batch 32, context lengths $\{2048, 8192, 32768\}$. Decode-only, ShareGPT trace replay plus a synthetic 90%-shared-prefix trace.
- **Arms.** Single Triton/CUTLASS decode kernel, layout as a compile-time template parameter. Sweep $B \in \{16, 32, 64, 128, 256, 512\}$ × axis order $\in$ {NHD, HND, vLLM-`x`-split}.
- **Control.** $B = \infty$: same kernel, block table present but all entries pointing into one contiguous per-sequence arena. This holds indirection *instructions* fixed while removing scatter, isolating locality from pointer-chase.
- **Deciding number.** $\eta(\Lambda) = \text{DRAM bytes read} / (\beta \cdot t_{\text{kernel}})$ from `ncu`, reported per arm. If $\max_B \eta(\Lambda_{\text{paged}}) \ge 0.95\,\eta(\Lambda_{\text{contig}})$ at all three context lengths, the layout question is settled — paging is free and the published 2.85× is an implementation gap. If the ratio stays below 0.85, indirection is a real cost and virtual-memory approaches are justified. Secondary readout: the $B$ maximizing $\eta$, and whether it is workload-invariant.

Cost: ~200 GPU-hours including kernel development.

## 9. Key References

- **[Foundational]** Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. — arXiv:2309.06180
- **[Foundational]** Hong, Kung. *I/O Complexity: The Red-Blue Pebble Game.* STOC 1981.
- **[SOTA]** Prabhu, Nayak, Mohan, Ramjee, Panwar. *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention.* ASPLOS 2025. — arXiv:2405.04437
- **[SOTA]** Ye, Chen, Zheng, Xia, et al. *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.* MLSys 2025. — arXiv:2501.01005
- **[SOTA]** Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024. — arXiv:2307.08691
- **[Related]** Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. — arXiv:2205.14135
- **[Related]** Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. — arXiv:2305.13245
- **[Related]** Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. — arXiv:1911.02150
- **[Related]** Zheng, Yin, Xie, et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS 2024. — arXiv:2312.07104
- **[Related]** Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov, Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI 2024. — arXiv:2403.02310
- **[Related]** Juravsky, Brown, Ehrlich, Fu, Ré, Mirhoseini. *Hydragen: High-Throughput LLM Inference with Shared Prefixes.* 2024. — arXiv:2402.05099
- **[Survey]** Miao, Oliaro, Cheng, et al. *Towards Efficient Generative Large Language Model Serving: A Survey from Algorithm to System.* 2023. — arXiv:2312.15234

## 10. Worked Example

Llama-3-8B, fp16, H100 80 GB. Per-token KV: $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B $= 128$ KiB.

Batch 32 at 8192 tokens: $32 \times 8192 \times 128\ \text{KiB} = 32$ GiB of KV. Weights 16 GB; ~30 GB headroom. Fine.

**Ideal decode-attention time**, one step, all KV read once:
$$T_{\text{ideal}} = \frac{34.4 \times 10^9}{3.35 \times 10^{12}} = 10.3\ \text{ms}$$

**Fragmentation at $B=16$:** each of 32 sequences wastes on average 7.5 tokens $\times$ 128 KiB $= 0.96$ MiB; total 30 MiB out of 32 GiB — **0.09%**. At $B=256$: 1.5% . So on the *memory* axis, $B$ is nearly free to raise by 16×.

**Contiguity at $B=16$:** one block, one layer, one head, K only, is $16 \times 128 \times 2 = 4096$ B. That is 32 HBM sectors (128 B) — plenty for coalescing. But the per-layer K block across all 8 heads is $32$ KiB, and the *sequence* is 512 such blocks scattered across HBM. TMA's bulk-copy path wants a single descriptor over a contiguous tile; with 512 disjoint 32 KiB runs the kernel issues 512 descriptors, or falls back to `cp.async`.

**Here the obstruction becomes visible.** Suppose measurement gives $t_{\text{paged}} = 14.7$ ms and $t_{\text{contig}} = 11.1$ ms, i.e. $\eta = 0.70$ vs $0.93$, a 32% gap. Nothing in that pair tells you the split between (a) 512 block-table loads per sequence-layer, (b) the lost TMA path, (c) a tile schedule tuned for one arm. Raise $B$ to 256 and the run length becomes 512 KiB, TMA is viable, fragmentation is still only 1.5% — but if the kernel was written assuming $B{=}16$ tiles, the measured time may not improve, and the experimenter concludes "block size doesn't matter."

That conclusion would be an artifact of the kernel, not a fact about layout. The $B = \infty$ control arm in §8 exists precisely to break this confound: same instruction stream, same tiles, only the scatter removed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*