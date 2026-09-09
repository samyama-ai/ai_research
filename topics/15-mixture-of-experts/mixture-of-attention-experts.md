---
id: 15-mixture-of-experts/mixture-of-attention-experts
title: "MoE Attention Layer Sparsification"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# MoE Attention Layer Sparsification

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/mixture-of-attention-experts` · **Status:** open

## 1. Problem Statement

Sparse MoE is deployed almost exclusively in the feed-forward block. Mixtral, DeepSeek-V3, Qwen3-MoE and OLMoE all route the MLP and leave attention dense. The question is whether attention can be sparsified the same way — a router picks $k$ of $H$ attention experts per token — at a real gain, and if not, why not.

Three variants, with different difficulty:

- **Measurement.** Given a routed-attention model and a dense control, does routing move the loss-vs-cost frontier? "Cost" must be fixed: total parameters, active parameters, training FLOPs, inference wall-clock, or KV-cache bytes per token. Routed attention wins on some and loses on others; which one is quoted is usually the whole result.
- **Method.** Build a routed attention layer whose measured decode throughput at fixed quality beats grouped-query attention (GQA) plus a dense-attention/MoE-MLP baseline on the same hardware. No published method has shown this at $\geq$ 7B parameters with a matched control.
- **Theory.** Does head-level routing have a scaling exponent distinct from MLP-expert routing? Clark et al.'s routed-model scaling laws (ICML 2022) cover MLP routing; no analogue exists for attention.

Solving it means: a routed-attention recipe that, at $\geq$ 7B active parameters and $\geq$ 1T training tokens, matches or beats the dense-attention control on loss at equal training FLOPs **and** shows a wall-clock or KV-byte win on a documented accelerator.

## 2. Formal Setting

A dense multi-head attention layer over $T$ tokens, model width $d$, $H$ heads of width $d_h$:

$$\mathrm{MHA}(X) = \sum_{h=1}^{H} \mathrm{softmax}\!\left(\frac{XW_Q^h (XW_K^h)^\top}{\sqrt{d_h}} + M\right) X W_V^h W_O^h .$$

A routed layer replaces the sum over all $H$ with a sum over a per-token subset. Let $r(x_t) = \mathrm{softmax}(W_r x_t) \in \Delta^{E-1}$ be the router over $E$ attention experts, and $\mathcal{S}_t = \mathrm{top}\text{-}k(r(x_t))$:

$$\mathrm{MoA}(x_t) = \sum_{e \in \mathcal{S}_t} \frac{r_e(x_t)}{\sum_{j\in\mathcal{S}_t} r_j(x_t)} \, \mathrm{Attn}_e(x_t; K_{\leq t}, V_{\leq t}).$$

An "expert" is one of: (a) a full $(W_Q,W_K,W_V,W_O)$ head (MoA, Zhang et al. 2022); (b) only the value/output projections with shared $Q,K$ (SwitchHead, Csordás et al. 2024); (c) a head-selection mask over existing heads (MoH, Jin et al. 2024); (d) a block of keys in the sequence, so routing is over *positions* rather than parameters (MoBA, NSA).

Measured quantities, as they are actually instrumented:

- **Active parameters per token** $P_a = P_{\text{shared}} + k \cdot P_{\text{expert}}$ — counted from the code, not the config file.
- **Training FLOPs** $C \approx 6 P_a N$ for $N$ tokens, plus the $O(T^2 d)$ attention term, which is *not* reduced by variants (a)–(c) unless $K,V$ are also routed.
- **KV bytes per token** $B_{kv} = 2 \, n_{kv} \, d_h \, L \, s$ for $n_{kv}$ KV heads, $L$ layers, $s$ bytes/element. This is the number that decides serving cost at long context, and variants (b) and (c) do not reduce it.
- **Realized sparsity** $\rho = k/E$, versus **effective sparsity** — the entropy of the empirical expert-usage histogram, $\exp(\mathbb{H}[\bar r])/E$, measured over $\geq 10^6$ held-out tokens. Collapse shows up here and nowhere in the loss curve.
- **Load imbalance** $\mathrm{CV} = \sigma/\mu$ over per-expert token counts within a batch; the padding waste in an expert-parallel kernel scales with $\max_e n_e$, not $\mu$.
- **Decode wall-clock**: median ms/token at batch $B$ and context $T$, one named accelerator, kernels named.

Assumptions, with the ones known to be violated marked:

1. Router decisions are per-token and independent — **violated**: attention outputs at $t$ depend on $K,V$ written by earlier tokens, so a routing choice at $t' < t$ changes the cache that token $t$ reads. Routed attention is not a per-token-decomposable sparsity, unlike a routed MLP.
2. FLOP reduction transfers to wall-clock — **violated**: gather/scatter and per-expert GEMM padding cost 20–40% of the nominal saving at moderate $E$; small-$k$ attention GEMMs are memory-bound, not FLOP-bound.
3. Top-$k$ is differentiable enough for the router to learn — **partly violated**: gradient reaches only selected experts; auxiliary balance losses are what actually keep unselected experts alive.
4. Train-time and inference-time routing distributions match — **violated** under batch-level balancing, where a token's expert depends on the rest of its batch (Switch Transformer, Fedus et al. 2022, notes this for MLPs).

## 3. State of the Art

**Established (with ablation).**
- **Mixture of Attention Heads** (Zhang et al., EMNLP 2022) — the first per-token head router; matches or beats dense Transformer baselines on masked LM and machine translation at $<$1B scale, with an interpretability claim (experts specialize by syntactic role) supported by attribution analysis.
- **SwitchHead** (Csordás, Piękos, Irie, Schmidhuber, NeurIPS 2024) — routes only $V$ and $O$ projections, keeping $Q,K$ shared. Matches parameter-matched dense baselines at 47M and 262M parameters while using fewer attention matrices; ablations isolate which projections must stay dense. This is the cleanest published evidence that attention sparsification is *not free but is possible*.
- **Head redundancy** — Michel, Levy & Neubig (NeurIPS 2019) and Voita et al. (ACL 2019) show most heads are prunable post hoc: Voita et al. remove 38 of 48 encoder heads with about 0.15 BLEU loss. This is the prior that motivates routing, and it is about *static* pruning, not per-token routing.

**Claimed but unablated, or benchmark-number-only.**
- **JetMoE-8B** (Shen et al., 2024) uses MoA in attention and MoE in the MLP and reports Llama2-7B-level benchmarks for about $0.1$M training dollars. The attention-routing contribution is not separated from the MLP-routing contribution, the data recipe, or the two-phase schedule. There is no dense-attention control at the same token budget.
- **MoH** (Jin et al., 2024) reports gains from treating heads as experts across LMs and ViTs, including continued-tuning of LLaMA3-8B. Reported as benchmark deltas; no matched-FLOP loss curve, no throughput measurement.
- **MoSA** (Piękos, Csordás, Schmidhuber, 2025) routes tokens into per-head sparse sets; reported at small scale *(frontier — verify)*.

**Position-routing SOTA is separate and stronger.** NSA (Yuan et al., ACL 2025) and MoBA (Lu et al., 2025) route over *key blocks*, not parameters, and do report real decode speedups at 64K–1M context with production deployment. These succeed because they cut the $O(T^2)$ and KV-read terms — the terms parameter-routed attention leaves untouched.

## 4. What Is Known

- Head-level redundancy is large and reproduced: 38/48 heads removable at 0.15 BLEU cost (Transformer-base WMT, ACL 2019); most layers reduce to one head at test time with small degradation (NeurIPS 2019).
- Routed-value attention matches dense at equal parameters at 47M and 262M (SwitchHead, NeurIPS 2024). No published replication above 1B.
- Routing $Q$ and $K$ is worse than routing $V$ and $O$ at the same sparsity — the shared-$QK$ design in SwitchHead is an ablation result, not a convenience.
- MLP-side routing scales predictably: granularity enters the loss law as a fitted exponent (Ludziejewski et al., ICML 2024); routed models follow a power law in expert count (Clark et al., ICML 2022). Both fits are MLP-only.
- Attention is a minority of parameters. In a 7B LLaMA-style model with $d=4096$, $L=32$, attention projections are about 2.1B of 6.7B parameters ($\sim$32%); with GQA at $n_{kv}=8$ it is closer to 1.3B ($\sim$20%). The upper bound on savings is bounded by this fraction.
- Production MoE models — Mixtral 8x7B (2024), DeepSeek-V3 (2024) — all keep attention dense and instead attack KV cost with GQA or MLA. That is a revealed-preference datum, not a proof.

## 5. What Is Not Known

- **Empirically open.** Whether routed attention holds at $\geq$ 7B and $\geq$ 1T tokens. Every clean ablation is $\leq$ 262M; every $\geq$ 8B result (JetMoE, MoH) lacks a matched control. Runnable today for roughly 10–20k accelerator-hours per arm.
- **Empirically open.** Whether per-token head routing beats *static* head pruning at equal active parameters. The 2019 pruning results are a strong, cheap baseline that routed-attention papers almost never run.
- **Methodologically blocked.** "Attention expert specialization" has no agreed measurement. Papers report qualitative head-role clusters; there is no metric with a null distribution under a randomly-initialized router, so specialization claims are untestable as stated.
- **Theoretically open.** No scaling law with attention-routing granularity as a free variable. No proof or counterexample that head-routed attention has the same asymptotic exponent as MLP routing.
- **Theoretically open.** Whether routed attention is expressive-equivalent to dense attention at matched active parameters — the sequential dependence of routing on the cache (assumption 1) breaks the standard mixture-approximation argument used for MLPs.

## 6. Why It Is Hard

The obstruction is **confounded cost accounting compounded by an unreduced bottleneck**.

At inference, long-context decode is bound by KV-cache reads, not attention projection FLOPs. Parameter-routed attention (MoA, SwitchHead, MoH) reduces projection work while leaving $B_{kv}$ unchanged — so a 2x FLOP reduction can produce a <5% throughput change, and the paper reports the FLOP number. Meanwhile, routing $K$ and $V$ per token — the thing that *would* cut $B_{kv}$ — makes the cache content depend on past routing decisions, so a token's representation is no longer a function of its own routing choice, and quality falls.

Second obstruction: **the baseline is a moving target**. GQA already cuts KV bytes 8x at near-zero loss (Ainslie et al., EMNLP 2023), and MLA (DeepSeek-V2, 2024) cuts further. Routed attention must beat a control that is itself a cheap, well-tuned form of attention sparsification. Most papers compare to full multi-head attention, which no one deploys.

Third: **compute cost of a decisive run**. The question is a scaling question, and the sub-1B regime where clean ablations exist is exactly the regime where attention is a smaller share of both parameters and time.

## 7. Current Research (as of 2026)

- **Position routing over parameter routing.** DeepSeek (NSA) and Moonshot (MoBA) have shifted effort to routing over key blocks with hardware-aligned kernels; this is where deployed speedups now come from.
- **Router-and-cache co-design** — sharing $K,V$ across experts so the cache stays dense while $V/O$ projections route (SwitchHead lineage; IDSIA/Stanford, Csordás and collaborators).
- **Sparse-attention scaling laws** — fitting loss as a function of context length, sparsity and model size, so sparse-attention configurations can be chosen rather than tuned *(frontier — verify; several 2025 preprints, none yet replicated)*.
- **Attention-expert interpretability** — whether routed heads correspond to identifiable circuits, currently blocked on the measurement gap in §5.

## 8. Concrete Next Experiment

**Question:** at 7B active parameters, does routing the attention $V/O$ projections beat spending the same active parameters on dense GQA attention?

**Scale.** Three arms, each 7B active parameters, 300B tokens from one fixed corpus, identical tokenizer, schedule, and data order. ~15k H100-hours per arm.

**Arms.**
1. **Control:** dense attention with GQA ($n_{kv}=8$), MoE MLP (8 experts, top-2). The deployed configuration.
2. **Treatment:** identical, but the attention layer routes $V/O$ over $E=8$ experts at top-2 with shared $Q,K$ (SwitchHead scaled up), total attention parameters raised so $P_a$ matches arm 1.
3. **Cheap baseline:** dense attention, statically pruned to the same active head count using the Michel et al. head-importance score, then continued-pretrained.

**Deciding number.** Median decode latency in ms/token at $B=32$, $T=32{,}768$, on H100 with FlashAttention-3 and a fused grouped-GEMM expert kernel, reported **only among arms within 0.01 nats of the control's held-out loss**. Routed attention is worth deploying iff arm 2 is $\geq$ 10% faster than arm 1 at that loss parity. If arm 3 matches arm 2, per-token routing adds nothing over static pruning and the direction is dead.

**Secondary readouts.** Effective sparsity $\exp(\mathbb{H}[\bar r])/E$ at 10%, 50%, 100% of training — collapse below 0.5 predicts the loss gap. Load CV per batch. $B_{kv}$, which should be identical across arms 1 and 2, making the null result interpretable rather than mysterious.

## 9. Key References

- **[Foundational]** Paul Michel, Omer Levy, Graham Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS, 2019. — arXiv:1905.10650
- **[Foundational]** Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL, 2019. — arXiv:1905.09418
- **[Foundational]** Xiaofeng Zhang, Yikang Shen, Zeyu Huang, Jie Zhou, Wenge Rong, Zhang Xiong. *Mixture of Attention Heads: Selecting Attention Heads Per Token.* EMNLP, 2022. — arXiv:2210.05144
- **[SOTA]** Róbert Csordás, Piotr Piękos, Kazuki Irie, Jürgen Schmidhuber. *SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention.* NeurIPS, 2024. — arXiv:2312.07987
- **[SOTA]** Jingcheng Yuan et al. (DeepSeek-AI). *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* ACL, 2025. — arXiv:2502.11089
- **[SOTA]** Enzhe Lu et al. (Moonshot AI). *MoBA: Mixture of Block Attention for Long-Context LLMs.* 2025. — arXiv:2502.13189
- **[Related]** Peng Jin, Bo Zhu, Li Yuan, Shuicheng Yan. *MoH: Multi-Head Attention as Mixture-of-Head Attention.* 2024. — arXiv:2410.11842
- **[Related]** Yikang Shen et al. *JetMoE: Reaching Llama2 Performance with 0.1M Dollars.* 2024. — arXiv:2404.07413
- **[Baseline]** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP, 2023. — arXiv:2305.13245
- **[Theory]** Aidan Clark et al. *Unified Scaling Laws for Routed Language Models.* ICML, 2022. — arXiv:2202.01169
- **[Theory]** Jakub Krajewski, Jan Ludziejewski et al. *Scaling Laws for Fine-Grained Mixture of Experts.* ICML, 2024. — arXiv:2402.07871
- **[Survey]** William Fedus, Jeff Dean, Barret Zoph. *A Review of Sparse Expert Models in Deep Learning.* 2022. — arXiv:2209.01667

## 10. Worked Example

A 7B LLaMA-style model: $d = 4096$, $L = 32$, $H = 32$, $d_h = 128$, GQA with $n_{kv} = 8$.

Attention parameters per layer: $W_Q$ is $4096\times4096 = 16.8$M; $W_K$ and $W_V$ are $4096 \times 1024 = 4.2$M each; $W_O$ is $16.8$M. Total $42$M per layer, $1.34$B over 32 layers — 20% of the model.

Now route $V$ and $O$ over $E=8$ experts at top-2, holding active parameters fixed. Active $V/O$ work per token stays at $2/8$ of the enlarged bank, so the FLOP saving relative to a dense layer of the same *total* size is 4x on those two projections. Per token, $V+O$ is $21$M of the $42$M attention parameters; the layer's active attention FLOPs fall from $2 \times 42\text{M} = 84$ MFLOP to about $2 \times (21 + 21/4)\text{M} \approx 52$ MFLOP — a 38% cut on attention projections, or about 8% of the whole forward pass.

Now the decode-time reality. At $T = 32{,}768$, $B = 32$, fp16: $B_{kv} = 2 \times 8 \times 128 \times 32 \times 2 = 131{,}072$ bytes per token, so the cache is $32 \times 32768 \times 131072 \approx 137$ GB — more than one H100's 80 GB, and read in full every decode step. Routing $V/O$ changes this number by exactly zero. The attention-projection GEMMs that shrank by 38% were already a small share of a step dominated by KV traffic.

The obstruction, in one line: the treatment cuts an 8% slice of forward FLOPs while the 137 GB of KV reads that set decode latency are untouched — and it adds a gather/scatter and expert padding on top. A 38% projection-FLOP saving that yields <2% end-to-end throughput is the expected outcome, and it is why NSA and MoBA route positions instead. Any paper reporting attention-MoE speedups in FLOPs rather than ms/token has not measured the quantity that decides deployment.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*