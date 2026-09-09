---
id: 09-model-design/inference-cost-optimal-architecture
title: "Inference-Cost-Optimal Architecture Design"
topic: 09-model-design
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Inference-Cost-Optimal Architecture Design

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/inference-cost-optimal-architecture` · **Status:** open

## 1. Problem Statement

Given a quality target, a serving workload, and a hardware platform, choose the architecture and training-token budget that minimize **total lifetime cost** — training plus all inference — rather than training compute alone.

- **Input:** quality target $L^\*$ (or a downstream score), a serving workload $W$ (request length distribution, arrival rate, latency SLOs), a hardware/price model $h$, and an expected inference volume $N_{\text{inf}}$ tokens.
- **Output:** an architecture $a$ (depth, width, attention variant, KV-head count, MoE granularity, sequence mixer) and token budget $D$.
- **Predicate:** $a$ solves the problem if no alternative $(a', D')$ meeting $L^\*$ on $W$ and $h$ has lower total dollar cost.

Three variants, of very different difficulty:

- **Measurement:** define and measure "inference cost" of an architecture. Currently the field mostly reports parameters or FLOPs, which are known to misrank real systems.
- **Method:** search $\mathcal{A} \times D$ efficiently, without training every candidate to $L^\*$.
- **Theory:** prove a scaling law of the form $L(a, N, D)$ whose architecture-dependent terms are identifiable and extrapolate across two orders of magnitude.

The measurement variant is the binding one. It is why the method and theory variants have not been settled.

## 2. Formal Setting

Architecture $a$ fixes layer count $n_l$, model width $d$, attention heads $n_q$, KV heads $n_{kv}$, head dim $d_h$, MLP ratio, and expert count $E$ with top-$k$ routing. Non-embedding parameters $N(a)$; active parameters per token $N_{\text{act}}(a) = N(a)$ for dense, $\approx N(a) \cdot k/E$ for MoE (modulo shared layers).

**Training compute** (as measured, not as approximated): $C_{\text{train}} = \tau_{\text{step}} \cdot n_{\text{steps}} \cdot F \cdot G$ GPU-seconds $\times$ price, where $F$ is peak FLOP/s and $G$ the GPU count. The usual proxy $C_{\text{train}} \approx 6 N D$ is only exact under constant model-FLOPs utilization (MFU), which varies with shape.

**Inference cost** is the reciprocal of goodput. On one accelerator with peak $F$ FLOP/s and HBM bandwidth $B$ B/s, a decode step at batch $b$ obeys a roofline:

$$t_{\text{step}}(a,b,\ell) \;=\; \max\!\left(\frac{\Phi(a,b,\ell)}{F \cdot \eta},\; \frac{M(a,b,\ell)}{B}\right)$$

with $\Phi$ FLOPs per step, $M$ bytes moved, and $\eta \in (0,1]$ realized kernel efficiency. Arithmetic intensity $I = \Phi/M$; the ridge point is $I^\* = F/B$. For an H100 SXM in bf16, $F \approx 9.9\times10^{14}$, $B \approx 3.35\times10^{12}$, so $I^\* \approx 295$ FLOP/byte.

Bytes moved split into weights and KV cache:

$$M \;=\; \underbrace{p_w N_{\text{read}}(a,b)}_{\text{weights}} \;+\; \underbrace{2\,p_c\, n_l\, n_{kv}\, d_h\, b\, \ell}_{\text{KV cache}}$$

$p_w, p_c$ are bytes per element (2 for bf16, ~1 for FP8). $N_{\text{read}}$ equals $N$ for dense models but for MoE grows from $N_{\text{act}}$ toward $N$ as $b$ rises and the union of routed experts saturates.

**Objective.** Let $\pi$ be price per GPU-second and $T(a,W,h)$ SLO-feasible tokens/s/GPU:

$$\min_{a,D} \;\; \pi\!\left[C_{\text{train}}(a,D) + \frac{N_{\text{inf}}}{T(a,W,h)}\right] \quad \text{s.t.} \quad L(a,D) \le L^\*.$$

**Assumptions, and which are violated.**
- *Quality is a scalar $L$.* Violated: architectures with equal validation loss differ on long-context and reasoning evals (loss is a lossy sufficient statistic).
- *$\eta$ is architecture-independent.* Violated: head dims that are not multiples of the tensor-core tile, odd expert counts, and small $d$ cost 20–50% of achievable MFU.
- *Serving point is known at design time.* Violated: $b$ and $\ell$ are chosen by the scheduler at runtime, and the optimal $a$ depends on them.
- *Hardware is fixed.* Violated: models outlive the accelerator generation they were designed for; $I^\*$ has roughly doubled per generation.
- *Inference volume $N_{\text{inf}}$ is known.* Violated: it is a demand forecast, and test-time-compute methods make it endogenous to $a$.

## 3. State of the Art

**Theory SOTA.** Hoffmann et al. (Chinchilla, NeurIPS 2022) give $L(N,D) = E + A N^{-\alpha} + B D^{-\beta}$ and the $\approx 20$ tokens/parameter compute-optimal ratio — training only. Sardana et al. (*Beyond Chinchilla-Optimal*, ICML 2024) add an inference term and solve for the minimizer of training-plus-inference FLOPs; this is the only widely cited closed-form treatment. It is **established as an optimization result but its cost model is FLOP-based**, so it does not capture memory-bandwidth-bound decoding, and it holds architecture fixed. Clark et al. (*Unified Scaling Laws for Routed Language Models*, ICML 2022) fit a law in expert count $E$; Kumar et al. (*Scaling Laws for Precision*, 2024) add bit-width. No published law jointly parameterizes depth/width, $n_{kv}$, and $E$ with an identifiable, extrapolating fit.

**Systems/empirical SOTA.** The realized cost frontier is set by serving-system and architecture co-design: PagedAttention/vLLM (Kwon et al., SOSP 2023), FlashAttention (Dao et al., NeurIPS 2022), Pope et al. (*Efficiently Scaling Transformer Inference*, MLSys 2023), GQA (Ainslie et al., EMNLP 2023), MLA (DeepSeek-V2, 2024), and speculative decoding (Leviathan et al., ICML 2023). These are established with ablations.

**Claimed but unablated.** Frontier-lab architecture choices (deep-narrow vs wide, exact $n_{kv}$, expert granularity) are reported in model cards as decisions, almost never as iso-quality ablations with a serving benchmark. Vendor throughput multipliers — e.g. DeepSeek-V2's reported $5.76\times$ maximum generation throughput over DeepSeek-67B — are **benchmark numbers on the authors' own stack**, confounding architecture with kernel and scheduler changes. Hardware-aware NAS (MnasNet, CVPR 2019; Once-for-All, ICLR 2020) solved a structurally identical problem for vision at $\sim10^8$ params; nothing comparable has been run at LLM scale.

## 4. What Is Known

- **Compute-optimal ratio.** $\approx 20$ tokens/param at up to 70B/1.4T (Hoffmann 2022). The replication by Besiroglu et al. (2024) recovers exponents near $0.5/0.5$ but with materially wider confidence intervals than originally reported — the *point estimate* is more fragile than its use suggests.
- **Overtraining is already the practice.** Llama 3 8B was trained on ~15T tokens, ~$75\times$ the Chinchilla ratio — an inference-cost decision made without a published cost model.
- **Batch-1 decoding runs at ~0.3% of peak FLOPs.** Dense decode moves 2 bytes/param and does 2 FLOP/param/token, so $I \approx 1$ against $I^\* \approx 295$. Parameter count and FLOPs are therefore *not* the cost driver in the latency-critical regime; bytes are.
- **KV-cache reduction is real and cheap.** GQA with 8 groups gives an $8\times$ KV reduction on Llama-2-70B-shaped models and lands within a fraction of a point of MHA quality on T5-XXL, with uptraining at ~5% of pretraining compute (Ainslie 2023).
- **Cost metrics disagree in rank order.** Dehghani et al. (*The Efficiency Misnomer*, ICLR 2022) show params, FLOPs, and throughput induce different orderings over the same model set — the single most important negative result for this problem.
- **Architecture rankings do not transfer across scale.** Tay et al. (*Scaling Laws vs Model Architectures*, 2022) fit scaling curves for ~10 architectures and find small-scale ranking is a poor predictor of large-scale ranking; also *Scale Efficiently* (ICLR 2022) finds deep-narrow Pareto-dominates on several tasks at fixed params.
- **Serving software moves the frontier by $2$–$4\times$** at fixed architecture (vLLM, SOSP 2023), which is the same magnitude as the architecture effects being studied.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no agreed definition of an architecture's inference cost that is independent of the serving stack, batch size, and scheduler. Until "cost" is a well-defined functional of $a$ — or is reported as a curve over serving points rather than a scalar — the optimization in §2 is not well posed.
- **Theoretically open.** Whether a scaling law with architecture-dependent terms is *identifiable*: no proof that $L(a,N,D)$ admits a parameterization whose architecture coefficients can be estimated from small-scale runs and extrapolate. The Tay result is evidence against, not a proof.
- **Empirically open.** The iso-quality, iso-serving-point comparison — train several architectures to matched loss and benchmark all on one serving stack across batch sizes — is runnable at $10^{21}$ FLOPs for well under $10^6$ USD. Nobody has published it.
- **Empirically open.** How test-time compute (Snell et al., 2024) changes the optimum: reasoning models make $N_{\text{inf}}$ a function of $a$, and no published cost model closes that loop.

## 6. Why It Is Hard

**The measurement is confounded and the objective is not a scalar function of the architecture.** Three specific obstructions:

1. **Rank inversion across the serving point.** MoE reads $\sim N_{\text{act}}$ bytes at $b=1$ and $\sim N$ bytes at large $b$; dense reads $N$ at both. The cost ordering of two architectures therefore flips with batch size (§10). A single "inference cost" number does not exist.
2. **Architecture is confounded with kernel maturity.** A new mixer is measured against attention with five years of fused-kernel engineering behind it. The measured $\eta$ difference is attributed to the architecture. This confound cannot be removed by careful benchmarking, only by equalizing engineering effort — which nobody funds.
3. **Compute cost of the ground truth.** A single point on the frontier requires training to convergence *and* a full serving sweep. A $3\times3\times3$ grid at 7B scale is $\sim10^{24}$ FLOPs. So the field extrapolates from small scale, which the Tay result says is unsound.

## 7. Current Research (as of 2026)

- **Inference-aware scaling laws.** Extensions of Sardana et al. to memory-bound cost models and to quantized/distilled deployment; MosaicML/Databricks and academic groups. *(frontier — verify)*
- **KV-cache-shaped architecture.** MLA (DeepSeek), sliding-window/hybrid-attention stacks (Mistral, Character.AI-lineage work), and cross-layer KV sharing. Established direction; iso-quality ablations remain thin.
- **Subquadratic and hybrid mixers.** Mamba/Mamba-2 (Gu & Dao) and attention–SSM hybrids, motivated by constant-size decode state. Quality-per-cost at frontier scale is still contested.
- **Fine-grained MoE.** Very small experts with high $E$ and shared experts (DeepSeek-V3 lineage, Qwen). The design target is explicitly $N_{\text{act}}$-vs-$N$ trade.
- **Co-designed serving.** Disaggregated prefill/decode and speculative decoding as architecture constraints rather than post-hoc optimizations.
- **Hardware-aware NAS revival at LLM scale.** Small but growing; the vision-era methods (Once-for-All supernets) have not been made to work for autoregressive decoding cost. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does FLOP-based ranking of architectures predict dollar-cost ranking at realistic serving points?

- **Scale.** Nine architectures at $\approx 1.5$B active parameters, each trained on 60B tokens ($\approx 5\times10^{20}$ FLOPs each, $\sim 4.5\times10^{21}$ total; ~2,000 H100-hours, under \$10k at spot). Grid: depth/width aspect ratio $\in \{$deep-narrow, standard, wide-shallow$\}$ $\times$ KV configuration $\in \{$MHA, GQA-4, MLA$\}$. Two seeds for the standard/GQA-4 cell to establish noise.
- **Control arm.** Fixed data order, fixed tokenizer, fixed optimizer/LR schedule (tuned per-shape via muP so LR is not the confound), and **one serving stack** (vLLM, same commit, same kernels, FP8 KV off) for all nine. Control metric: parameter-matched, FLOP-matched ranking computed analytically.
- **Measurement.** For each model: validation loss to matched precision; then measured tokens/s/GPU at three serving points — $(b{=}1,\ \ell{=}4k)$, $(b{=}32,\ \ell{=}4k)$, $(b{=}256,\ \ell{=}1k)$ — under a fixed p95 TPOT SLO of 50 ms.
- **Deciding number.** Kendall $\tau$ between the FLOP-based cost ranking and the measured-throughput ranking, at each serving point. **If $\tau < 0.5$ at any serving point**, FLOPs are refuted as an architecture-selection objective and the catalog problem is confirmed methodologically blocked in the strong sense. Secondary: whether the *measured* ranking itself is stable across the three serving points ($\tau$ between serving points $> 0.8$ would mean a scalar cost is recoverable after all).

This is small enough to run in a week and its outcome is binary.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Sardana, Portes, Doubov, Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML, 2024. — arXiv:2401.00448
- **[SOTA]** Ainslie, Lee-Thorp, de Jong, et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP, 2023. — arXiv:2305.13245
- **[SOTA]** Kwon, Li, Zhuang, et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[SOTA]** Pope, Douglas, Chowdhery, et al. *Efficiently Scaling Transformer Inference.* MLSys, 2023. — arXiv:2211.05102
- **[Key negative result]** Dehghani, Tay, Arnab, Beyer, Vaswani. *The Efficiency Misnomer.* ICLR, 2022. — arXiv:2110.12894
- **[Key negative result]** Tay, Dehghani, Abnar, et al. *Scaling Laws vs Model Architectures: How Does Inductive Bias Influence Scaling?* Findings of EMNLP, 2023. — arXiv:2207.10551
- **[Replication]** Besiroglu, Erdil, Barnett, You. *Chinchilla Scaling: A Replication Attempt.* 2024. — arXiv:2404.10102
- **[Method]** Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- **[Method]** Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. — arXiv:1911.02150
- **[Method]** Cai, Gan, Wang, Zhang, Han. *Once-for-All: Train One Network and Specialize It for Efficient Deployment.* ICLR, 2020. — arXiv:1908.09791
- **[Related]** Clark, de las Casas, Guy, et al. *Unified Scaling Laws for Routed Language Models.* ICML, 2022. — arXiv:2202.01169
- **[Related]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[Survey]** Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM, 2009.

## 10. Worked Example

Compare a sparse model (Mixtral-8x7B shape: $N = 46.7$B total, $N_{\text{act}} = 12.9$B) against a dense 13B, in bf16 on one H100 SXM ($B = 3.35$ TB/s, $F = 989$ TFLOP/s bf16). Ignore KV cache to isolate the weight term.

**Batch 1 decode.**

| | bytes read | roofline time | tok/s |
|---|---|---|---|
| MoE (2 of 8 experts routed) | 25.8 GB | $25.8/3350 = 7.7$ ms | 130 |
| Dense 13B | 26.0 GB | $7.8$ ms | 128 |

Effectively tied — and the MoE is the far stronger model. FLOPs per token are also near-identical ($2N_{\text{act}} \approx 25.8$ GFLOP vs $26$ GFLOP), so a FLOP-based comparison says "tied" and is right here.

**Batch 256 decode.** Now the 256 tokens in flight route to essentially all experts, so the whole weight set is read once per step:

- MoE: $M = 93.4$ GB $\Rightarrow 27.9$ ms; $\Phi = 256 \times 25.8$ GFLOP $= 6.6$ TFLOP $\Rightarrow 6.7$ ms. Bound by memory: $27.9$ ms $\Rightarrow$ **9,180 tok/s**.
- Dense 13B: $M = 26.0$ GB $\Rightarrow 7.8$ ms; $\Phi = 6.7$ TFLOP $\Rightarrow 6.7$ ms. $\Rightarrow$ **32,800 tok/s**.

**The obstruction.** The FLOP count is identical at both batch sizes and does not move. The measured cost ratio moves from $1.0\times$ to $3.6\times$ in favor of dense, purely from where the routed-expert union saturates. Arithmetic intensity tells the story the FLOP count hides: MoE goes from $I = 1.0$ to $I = 71$; dense goes from $I = 1.0$ to $I = 258$, close to the $I^\* = 295$ ridge.

So "which architecture is inference-cost-optimal?" has no answer without naming $b$ — and $b$ is set at runtime by the scheduler, not at design time by the architect. That is the methodological block in §5, in three lines of arithmetic.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*