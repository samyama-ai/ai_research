---
id: 14-long-context/million-token-inference-cost-floor
title: "Energy and Cost Floor for Million-Token Inference"
topic: 14-long-context
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Energy and Cost Floor for Million-Token Inference

> **Topic:** Long Context · **ID:** `14-long-context/million-token-inference-cost-floor` · **Status:** open

## 1. Problem Statement

Serving one query over a $10^6$-token context currently costs on the order of a GPU-hour and a kilowatt-hour. The question is whether that is a property of today's implementations or a floor.

**Input.** A context $x_{1:n}$ with $n \approx 10^6$ tokens, a query, and a quality target $\epsilon$ defined against an exact-attention reference model.

**Output.** The response, plus the measured energy $E$ (joules) and marginal dollar cost $C$ of producing it.

**Decision predicate.** Does there exist an algorithm/hardware pair achieving quality within $\epsilon$ of the exact reference at $E = o(n^2)$, and further at $E = \tilde{O}(n)$?

Three variants, of very different difficulty:

- **Measurement.** Report $E$ and $C$ per million-token query in a way that is comparable across serving stacks. Currently blocked — see §5.
- **Method.** Build a serving system that cuts $E$ by $10\times$ at fixed quality. Empirically open; the gap between claimed and ablated compression is the live issue.
- **Theory.** Prove a lower bound on energy (equivalently, on memory traffic and FLOPs) for any algorithm meeting the quality constraint. Partially settled for *exact* attention, wide open for *approximate*.

Solving it means: a lower bound with matching construction, or a system that provably closes the gap to the bound.

## 2. Formal Setting

Model: $L$ layers, hidden width $d$, $h$ query heads, $h_{kv}$ KV heads, head dim $d_h$, $P$ total parameters, $n$ context tokens, $m$ generated tokens, precision $b$ bytes/element.

**KV cache footprint**, the quantity that actually binds:

$$M_{\mathrm{KV}} = 2 \, L \, h_{kv} \, d_h \, b \, n \quad \text{bytes}$$

Measured as peak allocator high-water mark (`torch.cuda.max_memory_allocated` or the vLLM block-manager count), not as an analytic estimate — paging and fragmentation add 5–20%.

**Prefill FLOPs**, split into the term linear in $n$ and the attention term:

$$F_{\mathrm{pre}} \approx \underbrace{2 P n}_{\text{dense}} + \underbrace{2 L n^2 d}_{\text{attention, causal}}$$

Measured with hardware counters (Nsight Compute `sm__sass_thread_inst_executed_op_*`), not derived, because kernels pad, recompute (FlashAttention recomputes the backward-free softmax tiles), and skip masked blocks.

**Decode memory traffic**, the binding term after prefill:

$$B_{\mathrm{dec}} \approx m \left( P b + M_{\mathrm{KV}} \right) / \beta_{\text{batch}}$$

with $\beta_{\text{batch}}$ the number of concurrent requests sharing a parameter read. Measured via DRAM read counters, not modelled.

**Energy.** Board power integrated over the request window, plus a datacenter overhead factor:

$$E = \gamma \int_{t_0}^{t_1} \sum_{g} p_g(t)\, dt , \qquad p_g \ \text{from NVML at} \ \geq 10\,\mathrm{Hz}$$

$\gamma \approx 1.1$–$1.6$ is PUE (power usage effectiveness — total facility power over IT power). $E$ must be measured at the PDU or via NVML board sensors, never as $\text{TDP} \times \text{time}$.

**Cost.** $C = \sum_g \tau_g \cdot r_g$, GPU-seconds times spot/reserved rate, plus host and network. This is a market price, not a physical quantity, and moves $\pm 3\times$ across providers.

**Objective.** $E^\star(n,\epsilon) = \inf \{ E(\mathcal{A}) : \mathcal{A} \ \text{achieves quality within}\ \epsilon \}$.

**Assumptions, and which are violated:**

| Assumption | Status in practice |
|---|---|
| Power draw roughly constant across a request | **Violated.** Prefill is compute-bound near TDP; decode is memory-bound and draws 40–60% of TDP. A single average hides a $2\times$ swing. |
| Per-query cost is well defined | **Violated.** Prefix caching amortizes prefill across many queries; marginal cost depends on hit rate, which is workload-dependent. |
| Quality is batch-independent | Holds for exact attention; **violated** for dynamic sparse/eviction methods, where per-request budgets interact with scheduling. |
| Reference model is available | **Violated at $n=10^6$** for most labs — the exact-attention control arm is itself the expensive thing. |
| Embodied carbon negligible | Unmeasured. Excluded by convention, not by evidence. |

## 3. State of the Art

**Theory SOTA (established).**
- Keles, Wijewardena & Hegde (ALT 2023) show exact self-attention admits no $O(n^{2-\delta})$ algorithm under SETH (Strong Exponential Time Hypothesis).
- Alman & Song (NeurIPS 2023) give a sharp threshold: $n^{1+o(1)}$-time *approximate* attention exists iff entries are bounded by $B = o(\sqrt{\log n})$; above that, SETH-hard. This is the strongest existing statement about the cost floor, and it is about time, not energy.
- Jelassi et al. (ICML 2024) prove constant-size state-space models need $\Omega(n)$ state to copy length-$n$ strings — a lower bound on any "linear attention removes the KV cache" claim.

**Systems SOTA (established, reproduced).** FlashAttention-2 (Dao, ICLR 2024) removes the $O(n^2)$ *memory* term. PagedAttention/vLLM (Kwon et al., SOSP 2023) cuts KV fragmentation waste to <4%. GQA (Ainslie et al., EMNLP 2023) and MLA (DeepSeek-V2, 2024) shrink $M_{\mathrm{KV}}$ by $8\times$ and more. Chunked prefill (Sarathi-Serve, OSDI 2024) and prefill/decode disaggregation (DistServe, OSDI 2024) raise goodput at fixed SLO. Ring Attention (Liu et al., 2023) makes $n=10^6$ feasible at all by sharding context across devices.

**Claimed but unablated.** KV-eviction and sparse-selection methods — H2O (NeurIPS 2023), StreamingLLM (ICLR 2024), MInference (NeurIPS 2024), KVQuant (NeurIPS 2024) — report 4–20$\times$ savings at "negligible" quality loss. The ablations are almost always at $n \le 128$k, on needle-style retrieval, against a same-family baseline. Whether the savings hold at $n = 10^6$ on multi-hop tasks is a **benchmark number, not an established result**.

**Energy SOTA.** There is none. No serving paper in this list reports joules.

## 4. What Is Known

Numbers, with scale named.

- **KV cache dominates at $10^6$.** A 70B-class model with 80 layers, 8 KV heads, $d_h=128$, fp16 gives $2\cdot 80\cdot 8\cdot 128\cdot 2 = 327{,}680$ bytes/token $\approx 320$ KiB. At $n=10^6$: **328 GB**, more than four H100-80GB.
- **Attention dominates prefill FLOPs at $10^6$.** Same model, $d=8192$: dense term $2Pn = 1.4\times10^{17}$; attention term $2Ln^2d = 1.31\times10^{18}$. Attention is **9.4$\times$ the dense cost** — the reverse of the $n=4$k regime.
- **Crossover.** Attention overtakes dense FLOPs when $n \gtrsim P/(Ld) \approx 10^5$ tokens for this shape. Measured indirectly and consistent with reported prefill latencies.
- **Hardware.** H100 SXM: 700 W, 80 GB HBM3, 3.35 TB/s, ~990 BF16 TFLOP/s dense. Realized MFU (model FLOPs utilization) on long-context prefill: 35–50%.
- **Deployment energy, small-context.** Luccioni, Jernite & Strubell (FAccT 2024) measured 1000 inferences across 88 models; text generation was the most energy-intensive task class by roughly two orders of magnitude over classification. Their contexts were short — extrapolating to $10^6$ is unjustified.
- **Training-side reference point.** Patterson et al. (2021) and the BLOOM footprint study (Luccioni et al., JMLR 2023) established the methodology for reporting AI energy; inference at long context has no equivalent.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on *energy* for approximate long-context inference at a stated quality tolerance. Alman–Song bounds time for a worst-case matrix; real attention matrices are structured (sink-heavy, locally banded), and no one has shown whether that structure is enough to escape the hard regime or is itself an artifact of training on short contexts.
- **Theoretically open.** Whether a memory-traffic lower bound $\Omega(f(n,\epsilon))$ exists for any decoder meeting a retrieval-plus-reasoning spec. The Jelassi copying bound covers exact recall only.
- **Empirically open.** Whether the 4–20$\times$ compression claims survive at $n=10^6$ against an exact-attention control. The experiment is runnable — it costs a few thousand GPU-hours — and has not been run at that scale with the control arm included.
- **Methodologically blocked.** Per-query energy is not currently well defined. Prefix caching, continuous batching, and speculative decoding all make marginal cost a function of the *co-resident workload*, not the request. There is no agreed accounting convention, so two honest measurements of "the same" query differ by $10\times$.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an absent control arm**.

Energy per query is not a function of the query. Under continuous batching, a request's marginal energy depends on how many other requests share each parameter read; under prefix caching, on cache hit rate. Report a number and you have reported a scheduler configuration.

Second, the control arm is the expensive object. To prove a sparse method loses <1 point at $n=10^6$, you must run exact attention at $n=10^6$ — the very thing whose cost you object to. So compression papers evaluate at $128$k and assume monotone extrapolation. That assumption is untested and there is reason to doubt it: eviction error compounds with sequence length, and the tasks that discriminate methods (multi-hop, aggregation over the full context) are exactly the ones where a fixed token budget must degrade.

Third, benchmarks do not measure what they name. Needle-in-a-haystack scores near 100% for methods that keep only a sink plus a recent window, because a single retrievable span survives almost any eviction policy. It certifies retrievability, not the ability to use the context.

## 7. Current Research (as of 2026)

- **Hardware-side amortization.** Prefill/decode disaggregation is now standard in production stacks (vLLM, SGLang, TensorRT-LLM); the open piece is scheduling KV across HBM/DRAM/NVMe tiers, following InfiniGen (OSDI 2024). *(frontier — verify current production defaults.)*
- **Sub-quadratic architectures.** Hybrid attention/SSM stacks (Mamba lineage, YOCO-style single-cache designs) trade a constant-size recurrent state against a retained global cache. The Jelassi and "RNNs are not Transformers (Yet)" results bound how far the pure-recurrent direction can go.
- **Learned KV compression** with quality-aware budgets rather than fixed token counts. *(frontier — verify.)*
- **Energy accounting for inference.** Strubell/Luccioni-adjacent work is extending the FAccT 2024 methodology toward serving systems; no long-context measurement has been published. *(frontier — verify.)*
- **Lower-bound theory.** Alman and collaborators continue extending fine-grained hardness to subquadratic *alternatives*, not just to attention itself.

## 8. Concrete Next Experiment

**"Joules per correct answer at $10^6$."**

- **Scale.** One open 70B-class model with a validated $10^6$-token context. 8×H100 node. 200 queries drawn from a task suite that requires aggregation over the full context (count/sort/compare over $\geq 50$ dispersed facts), not single-needle retrieval.
- **Control arm.** Exact attention (FlashAttention-2 + Ring Attention), full fp16 KV, no eviction, no quantization, batch size 1, prefix caching disabled. Budget ≈ 200 × 1 GPU-hour × 8 = 1600 GPU-hours. This arm is mandatory and is the reason the experiment has not been run.
- **Treatment arms.** (a) KV quantized to 4-bit; (b) sparse prefill selection at 10% density; (c) eviction to a 128k-token budget; (d) GQA→MLA-style latent cache.
- **Instrumentation.** NVML board power at 100 Hz, PDU cross-check, DRAM and FLOP counters. Report joules including idle-in-window power.
- **The deciding number.** $J/\text{correct}$ — total measured joules divided by number of correct answers, for each arm, at $n=10^6$. A method is real if $J/\text{correct}$ drops by $\geq 4\times$ versus control. If every compression arm's $J/\text{correct}$ is within $1.5\times$ of control — because accuracy falls as fast as energy — then compression at $10^6$ buys latency, not efficiency, and the field's headline claims are artifacts of $\leq 128$k evaluation.

## 9. Key References

- **[Foundational]** Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. — arXiv:1706.03762
- **[Theory]** Keles, Wijewardena & Hegde. *On the Computational Complexity of Self-Attention.* ALT 2023.
- **[Theory]** Alman & Song. *Fast Attention Requires Bounded Entries.* NeurIPS 2023. — arXiv:2302.13214
- **[Theory]** Jelassi, Brandfonbrener, Kakade & Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[SOTA/systems]** Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024. — arXiv:2307.08691
- **[SOTA/systems]** Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. — arXiv:2309.06180
- **[SOTA/systems]** Pope et al. *Efficiently Scaling Transformer Inference.* MLSys 2023. — arXiv:2211.05102
- **[SOTA/systems]** Agrawal et al. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI 2024. — arXiv:2403.02310
- **[SOTA/systems]** Zhong et al. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving.* OSDI 2024. — arXiv:2401.09670
- **[SOTA/context]** Liu, Zaharia & Abbeel. *Ring Attention with Blockwise Transformers for Near-Infinite Context.* 2023. — arXiv:2310.01889
- **[Compression]** Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. — arXiv:2305.13245
- **[Compression]** Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023. — arXiv:2306.14048
- **[Compression]** Xiao et al. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[Compression]** Jiang et al. *MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention.* NeurIPS 2024. — arXiv:2407.02490
- **[Energy]** Luccioni, Jernite & Strubell. *Power Hungry Processing: Watts Driving the Cost of AI Deployment?* ACM FAccT 2024. — arXiv:2311.16863
- **[Energy]** Patterson et al. *Carbon Emissions and Large Neural Network Training.* 2021. — arXiv:2104.10350
- **[Survey/benchmark]** Reddi et al. *MLPerf Inference Benchmark.* ISCA 2020. — arXiv:1911.02549

## 10. Worked Example

70B-class model, $L=80$, $d=8192$, $h_{kv}=8$, $d_h=128$, fp16. One query, $n=10^6$ context, $m=1000$ output tokens, 8×H100.

**Prefill.**
$$F_{\mathrm{pre}} = 2Pn + 2Ln^2d = 1.4\times10^{17} + 1.31\times10^{18} = 1.45\times10^{18}\ \text{FLOP}$$
At 990 TFLOP/s $\times$ 0.40 MFU = 396 TFLOP/s per GPU, aggregate 3.17 PFLOP/s:
$$t_{\mathrm{pre}} = 1.45\times10^{18} / 3.17\times10^{15} = 458\ \text{s}$$
GPU-seconds: $8 \times 458 = 3664$. Energy at 700 W measured board power, $\gamma = 1.2$:
$$E_{\mathrm{pre}} = 1.2 \times 3664 \times 700 = 3.08\ \mathrm{MJ} = 0.85\ \mathrm{kWh}$$
Cost at $2/H100-hour: $3664/3600 \times \$2 = \$2.04$.

**Decode.** $M_{\mathrm{KV}} = 328$ GB, read once per token. Aggregate HBM bandwidth $8 \times 3.35 = 26.8$ TB/s at 70% efficiency = 18.8 TB/s:
$$t_{\mathrm{dec}} = 1000 \times 0.328\,\mathrm{TB} / 18.8\,\mathrm{TB/s} = 17.4\ \text{s}$$
At 55% of TDP (memory-bound): $E_{\mathrm{dec}} = 1.2 \times 8 \times 17.4 \times 385 = 0.064$ MJ = 0.018 kWh.

**Result.** 0.87 kWh, $2.11, and prefill is **97.9% of the energy**. Roughly the energy of running a household refrigerator for a day, for one question.

**Where the obstruction becomes visible.** Apply 10% sparse prefill. FLOPs fall to $1.4\times10^{17} + 0.131\times10^{18} = 2.7\times10^{17}$, a $5.4\times$ reduction, so $E \to 0.17$ kWh. That is the number such papers report. But the accuracy control is missing: on a 50-fact aggregation task, if the sparse arm answers 42% correctly against the exact arm's 78%, then

$$\frac{J}{\text{correct}}: \quad \text{exact} = \frac{3.08\ \mathrm{MJ}}{0.78} = 3.95\ \mathrm{MJ}, \qquad \text{sparse} = \frac{0.57\ \mathrm{MJ}}{0.42} = 1.36\ \mathrm{MJ}$$

$2.9\times$, not $5.4\times$ — and if accuracy instead falls to 18%, sparse costs $3.17$ MJ per correct answer and the saving is gone. Nobody has measured which of these two worlds we are in at $n=10^6$, because measuring it requires paying the 1600 GPU-hours for the control arm. That, and not the algorithm design, is what keeps the floor unknown.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*