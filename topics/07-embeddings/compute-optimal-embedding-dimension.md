---
id: 07-embeddings/compute-optimal-embedding-dimension
title: "Compute-Optimal Embedding Dimension Scaling"
topic: 07-embeddings
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Embedding Dimension Scaling

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/compute-optimal-embedding-dimension` · **Status:** empirically-open

## 1. Problem Statement

Given a compute budget $C$ (FLOPs) and a task, choose the embedding dimension $d$ that minimises loss. Two distinct quantities are both called "embedding dimension" and the literature routinely conflates them:

- **$d_{\text{model}}$** — the residual-stream width of a transformer, which fixes token-embedding table size $Vd_{\text{model}}$, attention projections, and MLP fan-in. Chosen at pretraining time; changing it changes everything.
- **$d_{\text{emb}}$** — the dimension of the *output* vector used for retrieval, recommendation, or nearest-neighbour search. Chosen at head/objective time; downstream serving cost is $O(nd_{\text{emb}})$ per query over $n$ items.

Three variants of the problem, in increasing difficulty:

- **Measurement.** Is there a stable, reproducible curve $d^{\star}(C)$ — the width that minimises validation loss at fixed FLOPs — or is loss flat in $d$ over the range practitioners care about? Runnable today; the answer is contested.
- **Method.** Given a target budget, predict $d^{\star}$ from small-scale fits without training the large model. Requires the curve to have a stable functional form and extrapolate.
- **Theory.** Derive $d^{\star}(C)$ from properties of the data (intrinsic dimension, spectral decay of the co-occurrence or ideal-similarity matrix) rather than fitting it. Open.

Solving it means: a rule $d^{\star} = f(C, V, \mathcal{D})$ that, held out, predicts the empirical loss-minimising width to within the run-to-run noise floor at a scale at least $10\times$ beyond the fitting range.

## 2. Formal Setting

Model $M_\theta$ with $L$ layers, width $d$, vocabulary $V$, trained on $D$ tokens. Non-embedding parameters, as counted in practice:

$$N_{\text{nv}} \approx 12Ld^2, \qquad N_{\text{emb}} = Vd \;(\text{tied}) \;\text{ or }\; 2Vd \;(\text{untied}).$$

Training compute, as measured (forward+backward, ignoring attention-quadratic terms):

$$C \approx 6\,(N_{\text{nv}} + N_{\text{emb}})\,D.$$

The compute-optimal width is

$$d^{\star}(C) \;=\; \arg\min_{d}\;\Big[\min_{L,D:\;6(12Ld^2+Vd)D \le C} \mathcal{L}(d,L,D)\Big],$$

with $\mathcal{L}$ measured as cross-entropy in nats/token on a held-out corpus disjoint from training by document hash, not by shuffle. **Aspect ratio** $\rho = d/L$.

For retrieval, given queries $q$, corpus $\{x_i\}$, encoder $E_d:\mathcal{X}\to\mathbb{R}^d$, the objective is a ranking metric $R(d)$ (nDCG@10, Recall@100) at serving cost $S = nd$ floats. The trade-off is $\max_d R(d)$ s.t. $nd \le B$.

Assumptions, and their status:

- **Loss is separable / shape-independent** — Kaplan et al. report loss depends only weakly on shape at fixed $N$. *Violated at extremes*: at $\rho \lesssim 20$ or $\rho \gtrsim 500$ the penalty is large, and the flat basin's width is itself a function of $N$.
- **$C \approx 6ND$ with $N$ = non-embedding params.** Violated whenever $Vd$ is a large fraction of $N$ — small models with $V \ge 128\text{k}$, where the embedding table can exceed 30% of parameters and contributes FLOPs only in the output projection.
- **$V$ fixed while $d$ varies.** Violated in practice: $V^{\star}$ itself scales with $N$ (§4).
- **Downstream quality is monotone in pretraining loss.** Known false in cases: equal-loss models differ on retrieval and reasoning benchmarks.

## 3. State of the Art

**Established.**
- Kaplan et al. (2020), *Scaling Laws for Neural Language Models*: within roughly $\rho \in [10,300]$, loss at fixed non-embedding $N$ varies by only a few percent — the "shape doesn't matter much" result. Reproduced widely.
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla): fixes $N/D$ but says **nothing** about how to split $N$ into $d$ and $L$. This is the gap this page names.
- Yin & Shen (2018), *On the Dimensionality of Word Embedding* (NeurIPS): PIP loss gives a genuine bias–variance optimum in $d$ for matrix-factorisation embeddings, derived from the signal matrix spectrum. This is the only widely-cited *theory* of an optimal $d$, and it applies to static embeddings, not transformers.
- Bhojanapalli et al. (2020), *Low-Rank Bottleneck in Multi-Head Attention Models* (ICML): if head dimension $d/h < $ sequence length, attention cannot represent arbitrary context distributions — a hard lower bound on $d$ given $h$.
- Kusupati et al. (2022), *Matryoshka Representation Learning* (NeurIPS): nested prefixes of a $d=2048$ representation retain accuracy at much smaller $d$; up to $\sim 14\times$ smaller embeddings for ImageNet-1k accuracy parity. Independently reproduced and shipped in production embedding APIs.

**Claimed but unablated.**
- Tay et al. (2022), *Scale Efficiently* (ICLR): "DeepNarrow" — narrower/deeper models Pareto-dominate on downstream transfer at equal parameters. Measured on T5-style encoder-decoders, one data recipe; not shown to hold for decoder-only models at Chinchilla-optimal token counts.
- Alabdulmohsin et al. (2023), *Getting ViT in Shape* (NeurIPS): a fitted procedure that jointly optimises width, depth and patch size for ViT. Vision only; the transfer to language is asserted, not tested.
- Tao et al. (2024), *Scaling Laws with Vocabulary* (NeurIPS): $V^{\star}$ grows sublinearly with $N$; they estimate a 3B-parameter model wants $\sim$200k vocabulary rather than 32k. The $V$–$d$ interaction is fit, not ablated at scale.

**Benchmark-number-only.** MTEB leaderboard positions across $d \in \{384, 768, 1024, 4096\}$ are confounded by different training data, objectives, and base models — they support no statement about $d$.

## 4. What Is Known

- **The flat basin is real but bounded.** Kaplan et al.: at $N \approx 10^8$ non-embedding parameters, loss varies $<2\%$ for $\rho$ across roughly an order of magnitude; outside that, degradation is steep. Measured at $10^6$–$10^9$ params on WebText2.
- **Production models cluster in a narrow $\rho$ band.** GPT-3 175B: $d{=}12288$, $L{=}96$, $\rho{=}128$. Chinchilla 70B: $d{=}8192$, $L{=}80$, $\rho{=}102$. Gopher 280B: $d{=}16384$, $L{=}80$, $\rho{=}205$. Empirically $\rho \in [100, 210]$ across three independently designed frontier models — a strong regularity nobody has derived.
- **Embedding-table share is a small-model phenomenon.** At $V{=}256\text{k}$, $d{=}2048$: $Vd = 5.2\times10^8$ against $12Ld^2 \approx 1.2\times10^9$ for $L{=}24$ — 30% of parameters in the table. At $d{=}12288$, $L{=}96$ the same $V$ is under 3%.
- **Factorising the table is nearly free.** ALBERT (Lan et al., ICLR 2020) decouples $d_{\text{emb-in}}$ from $d_{\text{model}}$ ($128$ vs $768/1024$) with small loss cost, showing the input table's rank requirement is much lower than $d_{\text{model}}$.
- **Retrieval quality saturates in $d$.** Across MTEB-style evaluations, moving $d_{\text{emb}}$ from 768 to 1536 typically buys $\lesssim 1$ nDCG@10 point while doubling index cost; MRL shows most of the loss from truncation is recoverable by training for it.

## 5. What Is Not Known

- **Theoretically open.** No derivation of $d^{\star}(C)$ from data statistics for transformers. Yin & Shen's PIP-loss argument has no known transformer analogue, because the "signal matrix" whose spectrum sets the optimum is undefined once the representation is contextual and trained end-to-end. No proof that the loss basin in $\rho$ is convex, or that $\rho^\star$ is even asymptotically constant.
- **Empirically open.** Whether $\rho^{\star}$ drifts with $C$. Every published width/depth sweep is at $\le 10^{10}$ FLOPs-equivalent scale or at non-Chinchilla token ratios; the sweep at $10^{22}$–$10^{23}$ FLOPs with $D/N \ge 20$ has not been published. Runnable — costs roughly one frontier pretraining budget for a 6-point sweep.
- **Empirically open.** The joint $(d, V)$ optimum. Tao et al. vary $V$ at conventional $\rho$; nobody has run the $2$-D grid.
- **Methodologically blocked.** "The right $d_{\text{emb}}$ for retrieval." There is no agreed cost model that prices index memory, ANN recall degradation, and quality on one axis, so "optimal" is undefined without a stated exchange rate. Papers pick the exchange rate implicitly and report the winner.

## 6. Why It Is Hard

Three specific obstructions, in order of bite:

1. **The signal is inside the noise floor.** The claimed effect ($<2\%$ loss over a wide $\rho$ range) is comparable to seed-to-seed and data-order variance at the scales where sweeps are affordable. Deciding $d^\star$ needs multiple seeds per cell, which multiplies an already-large sweep.
2. **Confounded measurement via the optimiser.** Optimal learning rate, warmup, and initialisation scale all depend on $d$. A width sweep at fixed hyperparameters measures "width interacted with a miscalibrated LR", not width. muP-style transfer removes most of this but is itself an assumption that must be checked at each $d$.
3. **Non-identifiability of the objective.** Loss-optimal and downstream-optimal $d$ need not coincide, and there is no ground truth for "the representation is the right size" independent of a chosen task. Wider models with equal loss have measurably different retrieval geometry; which one is "better" depends on the serving budget, which is not part of the pretraining objective.

## 7. Current Research (as of 2026)

- **Joint architecture-shape scaling laws** — fitting $(d, L, V, D)$ simultaneously rather than $(N, D)$; Google DeepMind and the ViT-shape line (Alabdulmohsin, Zhai) are the visible thread. *(frontier — verify)*
- **Vocabulary–width co-scaling** following Tao et al., now relevant because frontier vocabularies moved from 32k to 128k–256k. Interacts directly with the $C \approx 6ND$ approximation.
- **Adaptive-dimension serving** — MRL descendants, learned truncation, and product-quantisation-aware training; the practical answer to the retrieval variant is "train one model, serve many $d$".
- **muP / spectral-condition parameterisation** as the control for obstruction 2 — makes width sweeps interpretable but adds an assumption. *(frontier — verify whether muP transfer holds at $d > 8192$ with modern normalisation.)*

## 8. Concrete Next Experiment

**Question.** Does $\rho^{\star} = d/L$ drift with compute, or is it constant?

**Scale.** Six IsoFLOP shells: $C \in \{10^{19}, 3\times10^{19}, 10^{20}, 3\times10^{20}, 10^{21}, 3\times10^{21}\}$ FLOPs. In each shell, train 5 models at $\rho \in \{32, 64, 128, 256, 512\}$, holding $N_{\text{nv}}$ fixed within the shell and setting $D$ Chinchilla-optimally ($D \approx 20N$). Fixed $V = 128\text{k}$, tied embeddings, muP so LR transfers across $d$. 3 seeds per cell in the two smallest shells to establish the noise floor; 1 seed elsewhere. Total $\approx 5\times10^{21}$ FLOPs — roughly one 7B-scale pretraining run.

**Control arm.** The same grid at fixed (non-muP) hyperparameters tuned at $\rho=128$. If the two arms disagree on $\rho^\star$, the published width literature is measuring optimiser miscalibration, and that is itself the finding.

**The deciding number.** Fit $\log \rho^{\star} = a + b\log C$ across shells. **$b$ decides it.** If $|b| < 0.05$ with a confidence interval excluding $0.1$, aspect ratio is compute-invariant and the practitioner rule is "pick $\rho \approx 128$ and stop thinking". If $b \ge 0.1$, current frontier models are systematically too narrow or too deep at scale, and the extrapolated $\rho^\star$ at $10^{25}$ FLOPs is the number to publish. Secondary readout: the width of the $\rho$ interval within $1\%$ of the shell minimum, reported per shell — if it shrinks with $C$, shape matters more at scale, which is the outcome that would most change practice.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, et al. *Scaling Laws for Neural Language Models.* arXiv, 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Yin, Shen. *On the Dimensionality of Word Embedding.* NeurIPS, 2018. — arXiv:1812.04224
- **[SOTA]** Alabdulmohsin, Zhai, Kolesnikov, Beyer. *Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design.* NeurIPS, 2023.
- **[SOTA]** Tao, Liu, Ni, et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024. — arXiv:2407.13623
- **[SOTA]** Kusupati, Bhatt, Rege, et al. *Matryoshka Representation Learning.* NeurIPS, 2022. — arXiv:2205.13147
- **[Method]** Tay, Dehghani, Rao, et al. *Scale Efficiently: Insights from Pretraining and Finetuning Transformers.* ICLR, 2022. — arXiv:2109.10686
- **[Theory]** Bhojanapalli, Yun, Rawat, Reddi, Kumar. *Low-Rank Bottleneck in Multi-Head Attention Models.* ICML, 2020.
- **[Method]** Lan, Chen, Goodman, Gimpel, Sharma, Soricut. *ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations.* ICLR, 2020. — arXiv:1909.11942
- **[Survey/Benchmark]** Muennighoff, Tazi, Magne, Reimers. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023. — arXiv:2210.07316

## 10. Worked Example

Budget: $C = 10^{20}$ FLOPs, $V = 128\text{k}$, tied embeddings, Chinchilla ratio $D = 20N$.

From $C = 6ND = 120N^2$: $N = \sqrt{10^{20}/120} \approx 9.1\times10^{8}$, so $D \approx 1.8\times10^{10}$ tokens.

Now split $N$. Take $N = 12Ld^2 + Vd$ and solve for two candidate shapes:

| $\rho = d/L$ | $d$ | $L$ | $12Ld^2$ | $Vd$ | $Vd$ share |
|---|---|---|---|---|---|
| 64 | 1536 | 24 | $8.15\times10^8$ | $1.97\times10^8$ | 19% |
| 128 | 2048 | 16 | $8.05\times10^8$ | $2.62\times10^8$ | 25% |
| 256 | 2560 | 10 | $7.86\times10^8$ | $3.28\times10^8$ | 29% |

The obstruction is visible in the last column. Moving $\rho$ from 64 to 256 moves $10\%$ of the parameter budget out of the layers and into the embedding table — parameters that contribute FLOPs only at the output projection, not at every layer. So the three rows are **not** IsoFLOP even though they are iso-parameter, and they are not iso-parameter-in-the-layers even though they are iso-$N$. Which of the three is "the same model at a different aspect ratio" depends on which accounting you adopt, and the three accountings pick different winners.

Concretely: if the true loss gap between $\rho{=}64$ and $\rho{=}256$ is the $\sim1.5\%$ that Kaplan-style curves predict, that is roughly $0.03$ nats at $\mathcal{L}\approx 2.0$. A single-seed run at this scale has seed variance of order $0.01$–$0.02$ nats. The effect is one to three noise units. That is why the question is empirically open rather than merely unasked: the experiment in §8 is not hard to design, it is hard to *resolve*, and every existing sweep is underpowered for the effect it reports.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*