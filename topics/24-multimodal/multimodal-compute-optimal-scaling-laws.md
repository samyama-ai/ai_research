---
id: 24-multimodal/multimodal-compute-optimal-scaling-laws
title: "Compute-Optimal Scaling Laws for Multimodal Pretraining"
topic: 24-multimodal
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Scaling Laws for Multimodal Pretraining

> **Topic:** Multimodal Models · **ID:** `24-multimodal/multimodal-compute-optimal-scaling-laws` · **Status:** empirically-open

## 1. Problem Statement

Chinchilla answers one question for text: given a FLOP budget $C$, pick $(N, D)$ — parameters and training tokens — to minimize final loss. The multimodal version adds at least three more free variables: the **mixture** (what fraction of the token stream is image/audio/video), the **encoding rate** (how many tokens one image costs, set by patch size and resolution), and the **architectural split** (how much of $N$ sits in a vision encoder versus a shared decoder).

- **Input:** budget $C$, a modality set $\mathcal{M}$, a data pool per modality.
- **Output:** $(N^\*, D^\*, \boldsymbol{\alpha}^\*, r^\*)$ — size, tokens, mixture weights, tokens-per-image.
- **Decision predicate:** does the compute-optimal ratio $D^\*/N^\*$ depend on $\boldsymbol{\alpha}$, and if so, how?

Three variants, of very different difficulty:

- **Measurement:** is there a loss that is comparable across modalities, so that "the compute-optimal point" is even a well-posed argmin? Currently no.
- **Method:** fit an IsoFLOP surface over $(N, D, \boldsymbol{\alpha})$ and extract a fitting rule. Runnable today; unrun at frontier scale with proper controls.
- **Theory:** derive why exponents should or should not be modality-invariant. Untouched.

## 2. Formal Setting

A model $f_\theta$, $\theta \in \mathbb{R}^N$, is trained on a stream of tokens drawn from modality-labelled sources $m \in \mathcal{M} = \{\text{text}, \text{image}, \dots\}$ with mixture $\boldsymbol{\alpha}$, $\sum_m \alpha_m = 1$.

**Compute.** $C = 6ND$ FLOPs (forward+backward, dense transformer, excluding attention quadratic terms). Measured as: parameter count from the checkpoint, token count from the dataloader counter, cross-checked against wall-clock $\times$ measured MFU $\times$ device peak.

**Per-modality loss.** $L_m = -\frac{1}{|T_m|}\sum_{t \in T_m} \log p_\theta(x_t \mid x_{<t})$, in nats per token, measured on a held-out shard disjoint at the document level (not the sample level — near-duplicate captions leak).

**The aggregate.** Every published multimodal law optimizes some
$$L_{\text{agg}}(\boldsymbol{w}) = \sum_{m} w_m L_m, \qquad \sum_m w_m = 1,$$
and $\boldsymbol{w}$ is a free choice. Usually $w_m = \alpha_m$ (i.e. total stream loss), which makes the objective depend on the mixture being evaluated.

**The fit.** The Hoffmann parametric form, extended:
$$L(N, D, \boldsymbol{\alpha}) = E(\boldsymbol{\alpha}) + \frac{A(\boldsymbol{\alpha})}{N^{a}} + \frac{B(\boldsymbol{\alpha})}{D^{b}},$$
with the open question being whether $a, b$ are functions of $\boldsymbol{\alpha}$ or constants. Under constant $a,b$, minimizing at fixed $C=6ND$ gives $N^\* \propto C^{\frac{b}{a+b}}$, $D^\* \propto C^{\frac{a}{a+b}}$, so the *token/parameter ratio* moves with $\boldsymbol{\alpha}$ only through $A/B$, not through the exponents.

**Encoding rate.** An image at resolution $R$ with patch size $p$ costs $r = (R/p)^2$ tokens. $D$ is therefore not a property of the data — it is a property of the tokenizer. The invariant quantity is bits per *sample*: $\mathcal{L}_{\text{img}} = r \cdot L_{\text{image}} / \ln 2$ bits per image.

**Assumptions, and which are violated:**

| Assumption | Status |
|---|---|
| $C = 6ND$ | Violated for vision encoders (high FLOP/param), MoE, and long-context attention. |
| Tokens are exchangeable units of information | **Badly violated.** One image token ≈ 0.05–0.3 nats of new content; one text token ≈ 2–3 nats. |
| $L_m$ commensurable across $m$ | Violated. Discrete text CE and continuous-image CE over a learned VQ codebook have unrelated units and unrelated irreducible terms $E_m$. |
| Single epoch, infinite data | Violated for image-text: LAION-scale pools are reused 5–30×. |
| Loss is a monotone proxy for capability | Violated at the level of interest: zero-shot retrieval and VQA exponents differ from loss exponents. |

## 3. State of the Art

**Established (fitted, ablated, and reproduced at least in part):**

- **Hoffmann et al. (NeurIPS 2022)** — text-only baseline: $a \approx b$, $D^\*/N^\* \approx 20$, verified by the 70B/1.4T Chinchilla beating the 280B/300B Gopher. This is the control arm every multimodal claim is measured against.
- **Cherti et al. (CVPR 2023)** — CLIP scaling laws over 5 model scales up to ViT-H/14 and up to 34B samples seen. Key established finding: the power-law exponent depends on the *pretraining set* (LAION-2B vs WIT-400M) and on the *downstream task*, with a crossover — no single exponent describes "CLIP scaling".
- **Alabdulmohsin et al. (NeurIPS 2023, SoViT)** — shape scaling for ViTs: width, depth and MLP dim should scale at different rates. SoViT-400m/14 matches ViT-g/14 (1.1B params) at equal compute. Ablated.

**Claimed but not independently ablated:**

- **Aghajanyan et al. (ICML 2023)** — mixed-modal scaling laws over 7 modality pairs and models spanning ~8M to 30B parameters. Proposes an explicit **competition/synergy** term: the mixed-modal loss departs from the sum of the unimodal laws by an amount that is positive (competition) below a scale-dependent threshold and can turn negative (synergy) above it. The functional form is fitted, not derived; the threshold has not been reproduced by an independent group.
- **Shukor et al. (2025, native multimodal)** — early-fusion models scale at least as well as late-fusion (encoder + LLM) and are cheaper at small $N$; the reported compute-optimal $N$-vs-$D$ trade-off is close to text-only. Broad sweep, but the mixture-ratio and encoding-rate axes are entangled with architecture in the same grid.

**Benchmark-number-only:** MM1 (McKinzie et al., ECCV 2024) reports a mixture ratio near 5:5:1 (captions : interleaved : text) and a learning-rate scaling rule fitted at ≤30B. Those are grid-search outcomes on a fixed recipe, not scaling laws — no exponent is estimated and no IsoFLOP surface is published.

## 4. What Is Known

- **Text control point:** $D^\*/N^\* \approx 20$ at $C \approx 5.76\times10^{23}$ FLOPs (70B params, 1.4T tokens). Besiroglu et al. (2024) re-fit Hoffmann's Approach 3 and found the published confidence intervals implausibly tight and inconsistent with Approaches 1–2; the ~20:1 headline survives, the exponent precision does not.
- **Cross-modal exponent similarity:** Henighan et al. (2020) fit autoregressive laws on text, image, video and image↔text and found optimal model size growing as roughly $C^{0.7}$ in each — measured over ~7 orders of magnitude of compute but at small absolute scale (≤1B params) and with per-modality tokenizers whose losses are not comparable.
- **Data quality changes the exponent:** Goyal et al. (CVPR 2024) show the compute-optimal *filtering aggressiveness* for image-text pairs shifts with budget — aggressive filtering wins at small $C$ and loses at large $C$. So there is at least one axis where the multimodal optimum provably is not scale-invariant.
- **Repetition penalty:** Muennighoff et al. (NeurIPS 2023) find text tokens repeated up to ~4 epochs are near-free, decaying to worthless by ~40. No equivalent curve exists for image tokens, which are the ones actually repeated in practice.
- **Encoder cost:** in late-fusion VLMs, the vision tower is typically 3–8% of parameters but 20–50% of pretraining FLOPs at 336–448px, so $C=6ND$ misprices the mixture by tens of percent.

## 5. What Is Not Known

- **Methodologically blocked.** Whether "the compute-optimal multimodal model" is well defined. It is an argmin of $\sum_m w_m L_m$, and no principled $\boldsymbol{w}$ exists. Different reasonable choices (uniform, mixture-weighted, bits-per-sample) move the reported optimum in different directions. This blocks the other two.
- **Empirically open.** Whether $a$ and $b$ depend on $\boldsymbol{\alpha}$. The IsoFLOP sweep — 5 mixtures × 6 model sizes × 4 budgets — is runnable on ~1,000 H100-days. Nobody has published it with encoding rate held fixed.
- **Empirically open.** Whether the Aghajanyan competition-to-synergy crossover is real or an artifact of tokenizer mismatch across their modality pairs.
- **Theoretically open.** No proof either way that mixing two sources with different intrinsic dimension leaves the data exponent $b$ invariant. Existing theory (e.g. Bahri et al.'s resolution-limited/variance-limited regimes, PNAS 2024) predicts $b$ from data-manifold dimension — which *differs* across modalities — so the null hypothesis of invariance has a theoretical reason to be false, and no one has tested it.

## 6. Why It Is Hard

**The specific obstruction is non-identifiability between mixture and encoding rate.** Increasing the image token fraction $\alpha_{\text{img}}$ and increasing tokens-per-image $r$ are the same intervention on $D$ and on $C$, but different interventions on information content. A sweep over $\alpha$ at fixed $r$ and a sweep over $r$ at fixed image count produce the same $(N, D, C)$ triples and different losses, so the fitted coefficients $A(\boldsymbol{\alpha}), B(\boldsymbol{\alpha})$ absorb the tokenizer's choices. Every published multimodal law fixes $r$ implicitly and reports $\alpha$ — the two are confounded by construction.

Second: **the aggregate loss is not comparable across arms.** Change $\alpha$ and the evaluation distribution changes, so the numbers being minimized in two arms of an IsoFLOP sweep are not on the same scale. This is not a nuisance, it changes the sign of conclusions.

Third: cost. A credible frontier fit needs the largest models to be ~10× above the smallest, and Chinchilla-scale extrapolation errors are dominated by the low-$N$ end.

## 7. Current Research (as of 2026)

- **Apple (Shukor, El-Nouby, Susskind)** — native/early-fusion scaling, MoE-sparse multimodal laws. Most direct attack on the mixture axis.
- **Meta FAIR** — mixed-modal autoregressive scaling continued from Chameleon; token-space unification for image and text.
- **LAION / Tübingen (Cherti, Schuhmann, Jitsev) and DataComp (Gadre, Ilharco, Schmidt)** — open scaling ladders where the data pool is the controlled variable.
- **Google DeepMind (Zhai, Alabdulmohsin, Beyer)** — shape scaling and encoder/decoder allocation.
- *(frontier — verify)* Reports that compute-optimal image resolution grows sub-linearly with $C$, i.e. that spending marginal compute on more images beats higher resolution up to ~$10^{23}$ FLOPs. Circulating; no published IsoFLOP surface backs it.

## 8. Concrete Next Experiment

**The mixture-invariance IsoFLOP grid.**

- **Scale:** dense decoder-only models at $N \in \{160\text{M}, 410\text{M}, 1\text{B}, 2.8\text{B}, 6.9\text{B}\}$; budgets $C \in \{3\times10^{19}, 10^{20}, 3\times10^{20}, 10^{21}\}$ FLOPs; image fractions $\alpha_{\text{img}} \in \{0, 0.25, 0.5, 0.75\}$. 60–80 runs, ~1,000 H100-days.
- **Held fixed:** tokens-per-image $r = 256$ in *every* arm, one VQ tokenizer, one data pool, single epoch, cosine schedule terminated at the target budget (not truncated from a longer schedule — that biases the fit).
- **Control arm:** $\alpha_{\text{img}} = 0$, same tokenizer and pipeline, which must reproduce $D^\*/N^\* \approx 20 \pm 3$. If it does not, the rig is broken and no multimodal conclusion follows.
- **Report:** loss in **bits per sample** (text: bits per document; image: $256 \cdot L_{\text{img}}/\ln 2$ per image), not nats per token.
- **The deciding number:** the ratio
$$\rho = \frac{(D^\*/N^\*)\big|_{\alpha=0.5}}{(D^\*/N^\*)\big|_{\alpha=0}}.$$
Bootstrap over IsoFLOP fits, 1,000 resamples. If the 95% CI for $\rho$ lies inside $[0.85, 1.18]$, mixture-invariance holds and Chinchilla transfers to multimodal pretraining unchanged. If the CI excludes 1, the token/parameter rule is modality-dependent and every VLM recipe fitted from text laws is misallocated.
- **Second, cheap arm:** repeat $\alpha=0.5$ at $r = 64$ (patch 28px, same images). If $\rho$ moves more with $r$ than with $\alpha$, the confounder in §6 is the dominant effect and the field is measuring the tokenizer, not the modality.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Henighan, Kaplan, Katz, et al. *Scaling Laws for Autoregressive Generative Modeling.* 2020. — arXiv:2010.14701
- **[SOTA]** Aghajanyan, Yu, Conneau, et al. *Scaling Laws for Generative Mixed-Modal Language Models.* ICML, 2023. — arXiv:2301.03728
- **[SOTA]** Cherti, Beaumont, Wightman, et al. *Reproducible Scaling Laws for Contrastive Language-Image Learning.* CVPR, 2023. — arXiv:2212.07143
- **[SOTA]** Alabdulmohsin, Zhai, Kolesnikov, Beyer. *Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design.* NeurIPS, 2023. — arXiv:2305.13035
- **[SOTA]** Shukor, Fini, Turrisi da Costa, Cord, Susskind, El-Nouby. *Scaling Laws for Native Multimodal Models.* 2025. — arXiv:2504.07951
- **[Empirical]** McKinzie, Gan, Fauconnier, et al. *MM1: Methods, Analysis and Insights from Multimodal LLM Pre-training.* ECCV, 2024. — arXiv:2403.09611
- **[Empirical]** Goyal, Maini, Lipton, Raghunathan, Kolter. *Scaling Laws for Data Filtering — Data Curation Cannot Be Compute Agnostic.* CVPR, 2024.
- **[Empirical]** Muennighoff, Rush, Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Replication]** Besiroglu, Erdil, Barnett, You. *Chinchilla Scaling: A Replication Attempt.* 2024. — arXiv:2404.10102
- **[Theory]** Bahri, Dyer, Kaplan, Lee, Sharma. *Explaining Neural Scaling Laws.* PNAS, 2024.
- **[Survey]** Li, Li, Zhang, et al. *Multimodal Foundation Models: From Specialists to General-Purpose Assistants.* 2023. — arXiv:2309.10020

## 10. Worked Example

Take $C = 10^{22}$ FLOPs. Anchoring on Chinchilla ($N=70$B at $C=5.76\times10^{23}$, $N^\*\propto C^{1/2}$):

$$N^\* = 70\text{B}\sqrt{\tfrac{10^{22}}{5.76\times10^{23}}} = 70\text{B}\times 0.132 \approx 9.2\text{B}, \qquad D^\* = 20 N^\* \approx 184\text{B tokens}.$$

Now make it multimodal at $\alpha_{\text{img}} = 0.5$: 92B image tokens.

| Tokenizer | $r$ (tokens/image) | Images bought | Same $N$, same $C$? |
|---|---|---|---|
| 224px, patch 14 | 256 | 359M | yes |
| 224px, patch 28 | 64 | **1.44B** | yes |

Both rows sit at the identical $(N, D, C) = (9.2\text{B}, 184\text{B}, 10^{22})$ point. The scaling law cannot tell them apart — but one shows the model 4× more of the world. A plausible measured outcome: per-token image CE of 1.10 nats at $r=256$ and 3.20 nats at $r=64$. Aggregate stream loss looks *worse* for the coarse tokenizer, so the naive optimizer picks $r=256$. Converted to the invariant unit:

$$256 \times 1.10 / \ln 2 = 406 \text{ bits/image}, \qquad 64 \times 3.20 / \ln 2 = 295 \text{ bits/image}.$$

The coarse tokenizer compresses each image *better* in bits and buys 4× the images, while scoring worse on the quantity the field actually reports. **That inversion is the obstruction.** Until the aggregate is defined in bits per sample rather than nats per token, an IsoFLOP sweep over multimodal mixtures fits the tokenizer's patch size and reports it as a property of the modality.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*