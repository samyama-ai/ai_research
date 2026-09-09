---
id: 09-model-design/feedforward-expansion-ratio
title: "Expansion Ratio of Feedforward Blocks"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Expansion Ratio of Feedforward Blocks

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/feedforward-expansion-ratio` · **Status:** empirically-open

## 1. Problem Statement

Every Transformer block contains a position-wise feedforward network (FFN) that projects the residual width $d$ up to an inner width $d_{\text{ff}}$ and back. The **expansion ratio** is $r = d_{\text{ff}}/d$. Vaswani et al. (2017) set $r=4$ with no ablation reported; the field has largely kept it, with gated variants shifting to $r=8/3$ to hold parameters fixed.

The problem has three variants.

- **Measurement.** Given a compute budget $C$ and a token budget $D$, does the optimal $r^\star$ exist as a well-defined quantity, and what is it? Solving this means an $r^\star(N, D, \text{hardware})$ curve with error bars, not a single number.
- **Method.** Is $r$ a free parameter that should be tuned per deployment (as depth/width aspect ratio arguably is), or is the loss surface flat in $r$ over $[2, 8]$ so that any choice in that band is within noise once parameters and tokens are matched?
- **Theory.** Is there a principled account — from memory capacity, superposition, or approximation theory — that predicts $r^\star$ from properties of the data distribution rather than from a sweep?

Solving it means: a practitioner picking $d$, $n_{\text{layers}}$, and $r$ under a fixed budget can compute $r$ instead of copying Llama.

## 2. Formal Setting

A decoder block at width $d$, sequence length $L$, with gated FFN (SwiGLU):
$$\text{FFN}(x) = W_{\text{down}}\big(\sigma(W_{\text{gate}} x) \odot W_{\text{up}} x\big), \quad W_{\text{up}}, W_{\text{gate}} \in \mathbb{R}^{d_{\text{ff}} \times d},\ W_{\text{down}} \in \mathbb{R}^{d \times d_{\text{ff}}}.$$

**Quantities as measured.**

- Parameters per layer, $g \in \{2,3\}$ matrices (non-gated / gated): $P_\ell = 4d^2 + g\,r\,d^2 = d^2(4 + gr)$. Measured by counting tensor elements, excluding embeddings and norms.
- Forward FLOPs per token per layer: $F_\ell \approx 2P_\ell + 4Ld$, the second term attention scores. Measured, not estimated, via a profiler; the $4Ld$ term is what makes $r^\star$ depend on $L$.
- Compute: $C = 6ND$ for $N$ non-embedding parameters and $D$ tokens (Kaplan et al. 2020), valid only when $L \ll d/ r$-ish, i.e. attention FLOPs are a small share.
- Objective: token-averaged cross-entropy $\mathcal{L}(r) = -\frac{1}{D_{\text{val}}}\sum \log p_\theta(x_t \mid x_{<t})$ on a held-out split from the same corpus. Reported with seed variance $\hat\sigma$ over $\ge 3$ seeds; without $\hat\sigma$ the comparison is uninterpretable.
- The decision predicate: $r^\star(C) = \arg\min_r \min_{d, n_\ell:\ C(d,n_\ell,r)=C} \mathcal{L}$. Note the inner minimisation — $r$ is only defined jointly with the depth/width allocation.
- Inference cost: decode-time arithmetic intensity $I = \text{FLOPs}/\text{bytes read}$. At batch 1 the FFN is bandwidth-bound, so $r$ trades throughput against KV-cache pressure, and $r^\star$ for training $\ne r^\star$ for serving.

**Assumptions, and which are violated.**

1. *$r$ is continuous.* Violated: kernels want $d_{\text{ff}}$ a multiple of 128 or 256, and tensor-parallel sharding wants divisibility by TP degree. Llama-3-8B's $d_{\text{ff}}=14336$ gives $r=3.5$, not $8/3=2.67$, largely for this reason.
2. *One $r$ for all layers.* Violated by MoE models and by pruning studies showing per-layer FFN utilisation varies strongly with depth.
3. *Loss transfers to downstream capability.* Violated in the specific regime of interest: matched-loss models can differ in knowledge recall, which is FFN-mediated.
4. *Attention FLOPs negligible.* Violated at long context, where increasing $r$ is comparatively cheaper than it looks at $L=2048$.

## 3. State of the Art

**Established.**

- Shazeer (2020), *GLU Variants Improve Transformer*, showed SwiGLU/GEGLU FFNs beat ReLU/GELU FFNs on T5 span-corruption at matched parameters and matched FLOPs, holding $d\cdot d_{\text{ff}}\cdot g$ fixed by setting $r=8/3$. This is a real ablation. It fixes the *ratio* only as a bookkeeping device to make the gating comparison fair — it is not evidence that $8/3$ is optimal.
- Kaplan et al. (2020) swept aspect ratios and reported loss is very weakly dependent on shape at fixed $N$: varying $d_{\text{ff}}/d$ over roughly an order of magnitude moved loss by a few percent, far less than changing $N$.

**Claimed but unablated.**

- The $r=4$ default itself. The original Transformer paper gives no sweep over $d_{\text{ff}}$; the value appears as a table entry.
- Gemma-2's high effective ratio (9B: $d=3584$, $d_{\text{ff}}=14336$, $r=4$; Gemma-7B: $d=3072$, $d_{\text{ff}}=24576$, $r=8$) is a released configuration, not an ablation. No public loss curve isolates $r$.
- MobileLLM (Liu et al., ICML 2024) argues deep-and-thin beats wide-and-shallow at sub-billion scale — a shape result that constrains $r$ indirectly, at 125M–350M only.

**Benchmark-number-only.** The spread across frontier open weights — Llama-3-8B $r=3.5$, Qwen2.5-7B $r\approx5.29$ ($3584 \to 18944$), OLMo-2-7B $r\approx2.69$, Gemma-7B $r=8$ — is evidence of no consensus, not evidence about $r^\star$. None of these labs published a matched-budget $r$ sweep.

## 4. What Is Known

- **Shape is a second-order term.** Kaplan et al. (2020), models $10^5$–$10^9$ params on WebText2: loss varies by $\lesssim$ a few percent across wide shape variation at fixed $N$, while $N$ itself drives order-of-magnitude changes. Scale measured: up to 1.5B.
- **Architecture ranking is scale-robust but shape is not the same as architecture.** Tay et al. (2022), *Scaling Laws vs. Model Architectures*, ICLR/EMNLP-era work over 12 architectures: relative ranking at small scale often fails to predict large scale. Scale: up to ~2.6B.
- **FFNs are the memory.** Geva et al. (2021), *Transformer Feed-Forward Layers Are Key-Value Memories* (EMNLP): FFN inner neurons act as pattern-keyed memories. This makes $d_{\text{ff}}$ the natural capacity knob, and gives $r$ a mechanism, not just a hyperparameter.
- **Capacity is roughly linear in parameters.** Allen-Zhu & Li, *Physics of Language Models 3.3: Knowledge Capacity Scaling Laws* (2024), report ~2 bits of stored knowledge per parameter across architectures and sizes (up to a few hundred million params on synthetic biography data), largely insensitive to shape. If that holds, $r$ should not matter for recall at fixed $N$ — a testable prediction, tested only on synthetic corpora.
- **Compute-optimal token counts are known and $r$-independent in current practice.** Hoffmann et al. (2022), Chinchilla: $\approx 20$ tokens/param. No published version conditions on $r$.

## 5. What Is Not Known

- **Empirically open** (the main gap). No public, seed-replicated sweep over $r \in \{1.5, 2.67, 4, 6, 8\}$ at matched $N$ *and* matched $D$, at $\ge 1$B params with $\ge 20$ tokens/param, with depth/width re-optimised inside each $r$. The experiment is runnable today for well under $10^{21}$ FLOPs. Nobody has published it.
- **Empirically open.** Whether $r^\star$ drifts with $D/N$ (over-training regime), with context length $L$, or with data repetition. All three are plausible and none measured.
- **Methodologically blocked.** Whether $r$ affects *capability* independent of loss. There is no accepted measurement that separates "same loss, different knowledge recall" from evaluation noise at the 0.5-point level typical of MMLU-style benchmarks.
- **Theoretically open.** No bound predicting $r^\star$ from data statistics. Existing depth-vs-width separations (e.g. Levine et al., *Limits to Depth Efficiency of Self-Attention*, NeurIPS 2020) constrain $d$ vs $n_\ell$, not $d_{\text{ff}}/d$.

## 6. Why It Is Hard

**Non-identifiability under a budget constraint.** $r$ cannot be varied alone. Holding $N$ fixed, raising $r$ forces $d$ down or $n_\ell$ down; holding $d$ and $n_\ell$ fixed, raising $r$ raises $N$. Every observed $r$ effect is therefore a *joint* effect of $r$ and whichever axis absorbed the budget. Published comparisons almost never state which axis was moved, so the direction of the confound is unrecoverable from the paper.

Second, **the effect is smaller than the confounds around it**. If the true loss gap between $r=2.67$ and $r=4$ at 1B params is ~0.005 nats, it sits below the ~0.01-nat spread from data order, learning-rate schedule, and initialisation seed — so a single-seed run cannot resolve it, and a three-seed run at 1B is a real compute bill.

Third, **the metric that decides it is not the metric people optimise**. Labs choose $r$ for kernel efficiency and TP sharding, then report loss. Loss-optimal $r$ and wall-clock-optimal $r$ can differ by 2× and both are correct answers to different questions.

## 7. Current Research (as of 2026)

- **Shape-aware scaling laws.** Extending Chinchilla-style fits with shape terms, so $\mathcal{L}(N, D, r)$ rather than $\mathcal{L}(N,D)$. Pursued in open-science pretraining efforts (AI2/OLMo, EleutherAI, HuggingFace) where full configs and losses are released. *(frontier — verify)*
- **MoE reframing.** In mixture-of-experts models $r$ splits into expert inner width and expert count; DeepSeek's fine-grained-expert line makes small-$r$ experts the norm, which is an implicit claim that low per-expert $r$ is fine. Unablated against dense equivalents at matched active params. *(frontier — verify)*
- **Hardware co-design.** Selecting $d_{\text{ff}}$ for tile alignment and TP degree first, loss second — normal practice at every frontier lab, and the reason published ratios cluster at ugly numbers like 14336 and 18944.
- **Interpretability-driven capacity work.** Superposition and sparse-autoencoder results (Anthropic, Google DeepMind) suggest FFN width sets the number of representable features; nobody has yet turned a feature-count measurement into an $r$ recommendation. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Five models at $N \approx 1.3$B non-embedding params, $D = 26$B tokens (20 tokens/param), $L = 4096$, SwiGLU, identical data order per seed, 3 seeds each — 15 runs, roughly $15 \times 6ND \approx 3\times10^{21}$ FLOPs, about 2–3 days on 64 H100s.

**Arms.** $r \in \{1.5,\ 2.67,\ 4,\ 6,\ 8\}$. For each $r$, hold $N$ fixed to within 0.5% by re-solving $n_\ell \cdot d^2(4+3r) = N$, and *inside* each arm pick the $(d, n_\ell)$ pair that minimises loss on a cheap 100M-param proxy sweep first (this is the step that removes the depth/width confound). Round $d_{\text{ff}}$ to multiples of 128 and report the residual ratio error.

**Control arm.** $r = 2.67$ at the standard Llama-shaped $(d, n_\ell)$, three seeds — this both anchors against current practice and measures $\hat\sigma$.

**Deciding number.** $\Delta\mathcal{L} = \mathcal{L}(r=2.67) - \min_r \mathcal{L}(r)$ in nats, against the seed standard deviation $\hat\sigma$. If $\Delta\mathcal{L} < 2\hat\sigma$ (expected $\hat\sigma \approx 0.003$ nats at this scale), the ratio is flat over $[1.5, 8]$ and should be chosen purely on kernel efficiency — a real, publishable negative. If $\Delta\mathcal{L} > 0.01$ nats, $r$ is a first-class hyperparameter and every model that copied $4$ left measurable loss on the table.

## 9. Key References

- **[Foundational]** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. *Attention Is All You Need.* NeurIPS, 2017. — arXiv:1706.03762
- **[Foundational]** Shazeer. *GLU Variants Improve Transformer.* arXiv, 2020. — arXiv:2002.05202
- **[SOTA]** Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei. *Scaling Laws for Neural Language Models.* arXiv, 2020. — arXiv:2001.08361
- **[SOTA]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Tay, Dehghani, Abnar, Chung, Fedus, Rao, Narang, Tran, Yogatama, Metzler. *Scaling Laws vs Model Architectures: How Does Inductive Bias Influence Scaling?* Findings of EMNLP, 2023. — arXiv:2207.10551
- **[Mechanism]** Geva, Schuster, Berant, Levy. *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP, 2021. — arXiv:2012.14913
- **[Mechanism]** Allen-Zhu, Li. *Physics of Language Models 3.3: Knowledge Capacity Scaling Laws.* arXiv, 2024. — arXiv:2404.05405
- **[Related]** Levine, Wies, Sharir, Bata, Shashua. *Limits to Depth Efficiency of Self-Attention.* NeurIPS, 2020.
- **[Related]** Liu et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* ICML, 2024. — arXiv:2402.14905
- **[Related]** Tay, Dehghani, Rao, Fedus, Abnar, Chung, Narang, Yogatama, Vaswani, Metzler. *Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers.* ICLR, 2022. — arXiv:2109.10686

## 10. Worked Example

Take $d = 2048$, $n_\ell = 24$, gated FFN ($g=3$), $r = 8/3$.

Per-layer params: $d^2(4 + 3 \cdot 8/3) = 12 d^2 = 12 \times 2048^2 = 50.3$M. Total non-embedding: $24 \times 50.3\text{M} = 1.21$B.

Now try $r = 8$ at the same budget. Per-layer: $d^2(4 + 24) = 28 d^2$. Two ways to pay for it:

| Move | Result | What also changed |
|---|---|---|
| Keep $d=2048$, cut depth | $1.21\text{B} / (28 \times 2048^2) = 10.3 \to 10$ layers | depth cut 2.4× |
| Keep $n_\ell=24$, cut width | $d^2 = 1.21\text{B}/(24 \times 28) = 1.80\times10^6 \to d = 1342 \to 1344$ | width cut 1.5×, head dim changed |

Neither is an $r$ ablation. The first is mostly a depth ablation and, by Levine et al. (2020) and MobileLLM, depth at ~1B is not free. The second changes $d$, which changes the attention FLOP share ($4Ld$ at $L=4096$: $4 \times 4096 \times 2048 = 33.6$M vs $22.0$M per layer) and forces a new head configuration. Rounding $d$ to 1344 also perturbs $N$ by ~0.3%, comparable to the effect being measured.

Then hardware bites: $r=8$ at $d=1344$ gives $d_{\text{ff}} = 10752 = 84 \times 128$, fine for TP=8 but $10752/8 = 1344$ per shard, which is not a multiple of 256 — so a real engineer rounds again, and $r$ becomes 8.19 or 7.81.

**The obstruction, made visible:** the quantity we want to vary is the only one that cannot be varied. Every arm of the sweep is a two-factor change, and the rounding noise in $N$ is the same order as the loss gap the sweep is trying to detect. This is why the experiment in §8 must re-optimise depth/width *inside* each $r$ arm and report residual ratio error — otherwise it measures the confound, not the ratio.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*