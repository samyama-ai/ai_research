---
id: 02-attention/attention-logit-rank-bottleneck
title: "Softmax Bottleneck in Attention Logit Rank"
topic: 02-attention
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Softmax Bottleneck in Attention Logit Rank

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-logit-rank-bottleneck` · **Status:** open

## 1. Problem Statement

A single attention head computes its $n \times n$ logit matrix as a product through a $d_h$-dimensional bottleneck, so $\operatorname{rank}(L) \le d_h$. In standard configurations $d_h = d/h = 64$ while $n$ is $2{,}048$–$1{,}000{,}000$. The question: **does this rank constraint bind on anything a language model actually needs to do, and if so where?**

Three variants, routinely conflated:

- **Theory.** Characterize the set of row-stochastic matrices realizable as $\operatorname{softmax}(L)$ with $\operatorname{rank}(L) \le d_h$, *as a function of an allowed logit magnitude budget* $\|L\|_\infty \le B$. Without a magnitude budget the question is nearly vacuous (Section 10).
- **Measurement.** Define a statistic on trained models that says how close a head is to saturating its rank budget, and that is not confounded by logit scale, causal masking, or cross-head/cross-layer compensation.
- **Method.** Exhibit an architecture that raises attainable logit rank at matched parameters, FLOPs, and KV-cache bytes, and show a loss improvement that survives ablation against a compute-matched baseline.

Solving it means: a separation theorem with a magnitude budget, plus one reproduced experiment where increasing $d_h$ alone (nothing else) buys a measurable loss reduction that increasing $h$ or $d_{\text{ff}}$ by the same FLOPs does not.

## 2. Formal Setting

Input $X \in \mathbb{R}^{n \times d}$, $n$ tokens, model width $d$. One head with $W_Q, W_K \in \mathbb{R}^{d \times d_h}$:

$$L = \frac{1}{\sqrt{d_h}} X W_Q W_K^\top X^\top + M, \qquad A = \operatorname{softmax}_{\text{row}}(L),$$

with $M$ the causal mask ($0$ / $-\infty$). Ignoring $M$, $\operatorname{rank}(L) \le \min(d_h, \operatorname{rank}(X))$.

**Quantities as measured.**

- *Logit rank budget:* $d_h$, read off the config. In multi-head attention $d_h = d/h$ by convention only.
- *Realized rank:* singular values $\sigma_1 \ge \dots$ of $L$ on the **unmasked lower triangle only** — the masked entries are $-\infty$, so rank must be taken on $L$ before masking, or on a submatrix. Report the stable rank $\operatorname{srank}(L) = \|L\|_F^2 / \sigma_1^2$ and the effective rank $\exp(-\sum_i p_i \log p_i)$, $p_i = \sigma_i / \sum_j \sigma_j$ (Roy & Vetterli, EUSIPCO 2007). Both are scale-invariant; plain numerical rank is not.
- *Target-attainability residual:* for a desired pattern $P$ (positive, row-stochastic), $\log P$ is realizable iff $\log P = L + c\mathbf{1}^\top$ for some $\operatorname{rank}(L) \le d_h$ — softmax is invariant to per-row shifts, so the object to rank-test is $\log P$ **after** removing its best rank-one row-shift component. Measure $\varepsilon_r(P) = \min_{\operatorname{rank}(L)\le r,\, c} \|\log P - L - c\mathbf{1}^\top\|_F$.
- *Magnitude budget:* $B = \max_{ij} |L_{ij}|$, measured in the model's activation dtype (bf16 logits saturate a softmax at differences $\gtrsim 30$).

**Assumptions and their violations.**

1. *$X$ has rank $\ge d_h$.* Usually true at $n \gg d$, but violated in early layers where token embeddings are near-duplicated, and progressively violated by depthwise rank collapse (Dong et al., ICML 2021).
2. *A head must realize its target alone.* Violated: $h$ heads sum after $W_O$, and residual + FFN layers compose across depth; a rank-$d_h$ head deficit can be repaired two layers later.
3. *Logit magnitude is free.* Violated: weight decay, LayerNorm, and bf16 all bound $B$ in practice. This is the assumption that makes almost every "rank bottleneck" claim ambiguous.
4. *RoPE and ALiBi do not change rank.* ALiBi adds a fixed rank-$\le n$ bias (it can *raise* attainable rank for free); RoPE keeps rank $\le d_h$ but restricts the achievable subspace.

## 3. State of the Art

**Theory (established).** Bhojanapalli, Yun, Rawat, Reddi & Kumar, *Low-Rank Bottleneck in Multi-head Attention Models*, ICML 2020: for $d_h < n$ there exist positive row-stochastic $P$ that no $W_Q, W_K$ can realize for a given full-rank $X$; $d_h \ge n$ suffices. Yun et al., ICLR 2020, prove transformers are universal sequence-to-sequence approximators *with* $d_h = 1$ — the two results coexist because universality is achieved by depth, not per-head rank. Sanford, Hsu & Telgarsky, NeurIPS 2023, give the sharpest binding separation: the sparse-averaging task needs embedding dimension $\tilde\Omega(n)$ for a one-layer attention model, while two layers solve it at $O(\log n)$.

**Theory (counterweight).** Likhosherstov, Choromanski & Weller show a *fixed* self-attention module can approximate arbitrary sparse attention patterns with head size logarithmic in $n$ — i.e. approximation, unlike exact realization, is cheap. Grivas, Bogoychev & Lopez (ACL 2022) show the analogous output-layer claim: low-rank softmax has unargmaxable classes in theory but almost none in practice.

**Empirical (claimed but under-ablated).** Bhojanapalli et al. decouple $d_h$ from $d/h$ (fixed $d_h = 64$ or $d_h = n$ with more heads) and report GLUE/SQuAD gains under $1$ point — benchmark numbers, not a controlled loss-vs-FLOPs curve. Talking-Heads Attention (Shazeer et al., 2020) mixes logits across heads, raising attainable logit rank to $\le h \cdot d_h$ at small cost, and reports perplexity/GLUE gains — again benchmark numbers, with no rank measurement showing the mechanism is rank.

**Counter-evidence (systems SOTA).** DeepSeek-V2 (2024) Multi-head Latent Attention compresses KV to a low-rank latent and *improves* quality-per-byte; GQA/MQA cut key diversity hard with negligible loss. Frontier practice moves in the direction of *less* rank, not more.

## 4. What Is Known

- Exact rank ceiling: BERT-base ($d{=}768$, $h{=}12$, $d_h{=}64$, $n{=}512$) has logit rank $\le 64$ against a $512 \times 512$ matrix — $12.5\%$ of full rank. Llama-3-8B: $d_h = 128$, $n = 8{,}192$ → $1.6\%$.
- Increasing $d_h$ from $64$ to $512$ at fixed $d$ (fewer heads) does not monotonically help; Bhojanapalli et al. find head count and head size trade off, with best configs at moderate both — measured at BERT-base scale, $\le 110$M params.
- Rank collapse of *representations* (not logits) is doubly exponential in depth for pure attention, and skip connections plus FFNs are what prevent it (Dong et al., ICML 2021), measured on BERT/ALBERT/XLNet-scale models.
- Softmax attention provably disperses: for a fixed logit budget, the max attention weight decays as $n$ grows, so sharp retrieval degrades out of distribution (Veličković et al., 2024). This is a **magnitude**, not a rank, limit — and it is the one demonstrated to bite at long context.
- The output-layer analogue is real and quantified: mixture-of-softmaxes gave $\approx 1.6$ perplexity on PTB and $\approx 2.4$ on WikiText-2 over a matched AWD-LSTM (Yang et al., ICLR 2018), at $\sim 20$M params; Chang & McCallum (ACL 2022) tie the same bottleneck to multi-modal word distributions in GPT-2-scale models.

## 5. What Is Not Known

- **Theoretically open.** No separation theorem of the form: *any* $d_h < f(n)$ head, under logit budget $B$, incurs error $\ge \varepsilon$ on a task a real LM performs — the existing lower bounds are either magnitude-unbounded (Bhojanapalli) or about width rather than head size (Sanford et al.). Also open: whether depth-$k$ composition strictly recovers everything rank-$d_h$ heads lose, at what depth cost.
- **Empirically open.** Nobody has published a compute-matched scaling sweep isolating $d_h$ at fixed $d$, $h \cdot d_h$, FLOPs, and KV bytes, at $\ge 1$B params and $n \ge 8{,}192$. The experiment is a few thousand GPU-hours; it is unrun, not infeasible.
- **Methodologically blocked.** There is no accepted measure of "how much rank a head *needed*." Realized $\operatorname{srank}(L)$ in trained models is typically well below $d_h$, which is consistent both with "the budget is ample" and with "training cannot exploit it." Separating those requires a counterfactual the current toolkit does not provide.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability between rank and logit magnitude**, compounded by cross-layer compensation. Any target pattern is approximable at rank $r$ by paying in logit scale (Section 10), and any residual error is repairable by a later layer. So a measured low $\operatorname{srank}$ is not evidence of a slack constraint, and a measured loss gain from larger $d_h$ is not evidence of rank — larger $d_h$ also changes the per-head parameter count, the $1/\sqrt{d_h}$ temperature, and the optimization conditioning. Every published $d_h$ intervention changes at least two of these at once. That is a **confounded measurement**, not a compute wall.

## 7. Current Research (as of 2026)

- **Sharpness/dispersion over rank.** Following Veličković et al. (2024), work at DeepMind and Oxford on adaptive-temperature and $\log n$-scaled attention treats long-context failure as a magnitude problem. *(frontier — verify)*
- **Latent-attention variants.** DeepSeek MLA derivatives and cross-layer KV sharing push effective key rank down for cache reasons; the open question is where the quality knee is. *(frontier — verify)*
- **Circuit-level rank probes.** Mechanistic-interpretability groups measure per-head QK-circuit spectra on Pythia/Llama checkpoints; induction and copy heads look near-rank-one, retrieval heads less so. Reproduced informally, not published as a controlled study. *(frontier — verify)*
- **Formal expressivity.** Sanford/Hsu/Telgarsky-line work on depth–width–head-size trade-offs, and communication-complexity lower bounds for attention.

## 8. Concrete Next Experiment

**Scale.** Four decoder-only LMs, $d = 1{,}024$, 24 layers, $\approx 400$M non-embedding params, trained on 20B tokens of a fixed corpus, context $n = 8{,}192$, identical data order and optimizer. Cost $\approx 2{,}000$ H100-hours total.

**Arms** (all with total attention FLOPs and KV bytes matched to within $2\%$ by adjusting $h$ and using GQA groups):
- A0 **control**: $h = 16$, $d_h = 64$ (standard).
- A1: $h = 4$, $d_h = 256$.
- A2: $h = 16$, $d_h = 64$ **+ talking-heads** logit mixing (raises attainable logit rank to $\le 1{,}024$, adds $\le 0.1\%$ params).
- A3 **confound control**: $h = 16$, $d_h = 64$, with $d_{\text{ff}}$ raised to absorb the FLOPs of A1's extra QK params — same compute, no rank change.

**Deciding number.** Validation cross-entropy in nats. The rank hypothesis predicts $\mathcal{L}(\text{A1}), \mathcal{L}(\text{A2}) < \mathcal{L}(\text{A3}) \approx \mathcal{L}(\text{A0})$. **Decide on $\Delta = \mathcal{L}(\text{A3}) - \min(\mathcal{L}(\text{A1}), \mathcal{L}(\text{A2}))$: if $\Delta \ge 0.01$ nats (roughly $1\%$ perplexity, $\approx 5\times$ seed noise at this scale, which is $\pm 0.002$ over 3 seeds), logit rank binds; if $|\Delta| < 0.005$, it does not at $n = 8{,}192$.** Secondary readout: needle-in-a-haystack retrieval accuracy at 8k, and $\operatorname{srank}(L)$ per head in each arm — if A1 uses no more stable rank than A0, any gain is not rank.

## 9. Key References

- **[Foundational]** Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen. *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model.* ICLR 2018. — arXiv:1711.03953
- **[Foundational]** Srinadh Bhojanapalli, Chulhee Yun, Ankit Singh Rawat, Sashank Reddi, Sanjiv Kumar. *Low-Rank Bottleneck in Multi-head Attention Models.* ICML 2020. — arXiv:2002.07028
- **[SOTA — theory]** Clayton Sanford, Daniel Hsu, Matus Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS 2023.
- **[SOTA — theory]** Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, Sanjiv Kumar. *Are Transformers Universal Approximators of Sequence-to-Sequence Functions?* ICLR 2020.
- **[Related]** Yihe Dong, Jean-Baptiste Cordonnier, Andreas Loukas. *Attention Is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.* ICML 2021. — arXiv:2103.03404
- **[Related]** Petar Veličković, Christos Perivolaropoulos, Federico Barbero, Razvan Pascanu. *softmax is not enough (for sharp out-of-distribution).* 2024. — arXiv:2410.01104
- **[Method]** Noam Shazeer, Zhenzhong Lan, Youlong Cheng, Nan Ding, Le Hou. *Talking-Heads Attention.* 2020. — arXiv:2003.02436
- **[Method]** DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* 2024. — arXiv:2405.04434
- **[Related]** Andreas Grivas, Nikolay Bogoychev, Adam Lopez. *Low-Rank Softmax Can Have Unargmaxable Classes in Theory but Rarely in Practice.* ACL 2022.
- **[Related]** Haw-Shiuan Chang, Andrew McCallum. *Softmax Bottleneck Makes Language Models Unable to Represent Multi-mode Word Distributions.* ACL 2022.
- **[Related]** Valerii Likhosherstov, Krzysztof Choromanski, Adrian Weller. *On the Expressive Power of Self-Attention Matrices.* 2021.
- **[Method]** Olivier Roy, Martin Vetterli. *The Effective Rank: A Measure of Effective Dimensionality.* EUSIPCO 2007.
- **[Survey]** Tianyang Lin, Yuxin Wang, Xiangyang Liu, Xipeng Qiu. *A Survey of Transformers.* AI Open, 2022.

## 10. Worked Example

**Task.** $n = 512$ tokens; each query must attend to one distinct key — the target is a permutation, $P \to$ identity. Then $\log P$ (after removing the row-shift) is $\tau I_{512}$ up to scale: **rank 512**. With $d_h = 64$ this is unrealizable exactly. That is the Bhojanapalli conclusion, and taken alone it looks fatal.

**Now add the magnitude budget.** Realize $L = \tau \Phi\Phi^\top$ with $\Phi \in \mathbb{R}^{512 \times 64}$ unit rows — rank 64 by construction. Correct retrieval needs, per row,

$$e^{\tau} \;>\; \sum_{j \ne i} e^{\tau \mu_{ij}} \quad\text{with}\quad \mu = \max_{i\neq j}|\langle \phi_i,\phi_j\rangle|,$$

so it suffices that $\tau(1-\mu) > \ln 511 = 6.24$.

- Full rank ($d_h = 512$, $\mu = 0$): $\tau > 6.24$.
- Rank 64, random Gaussian $\Phi$: $\mu \approx \sqrt{2\ln(n^2)/d_h} = 0.62$ → $\tau > 16.4$.
- Rank 64, Welch-bound-optimal frame: $\mu \ge \sqrt{\tfrac{n-d_h}{d_h(n-1)}} = \sqrt{448/32704} = 0.117$ → $\tau > 7.07$.

**The obstruction, made visible.** The exact-rank theorem says "impossible." The actual price of the $8\times$ rank deficit, for an optimally packed key set, is a **13% larger logit scale** ($7.07$ vs $6.24$) — well inside bf16 and inside observed logit norms. The bottleneck is therefore only as real as the bound on $\tau$, and $\tau$ is set by weight decay, LayerNorm and numerics, none of which appear in the rank statement. A random (untrained) key geometry costs $2.6\times$ the logit scale instead of $1.13\times$, so the binding quantity is **key-set coherence $\mu$**, which is a training outcome, not an architectural constant. Any measurement that reports $d_h$, or even $\operatorname{srank}(L)$, without $\mu$ and $\tau$ is not measuring the thing it names.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*