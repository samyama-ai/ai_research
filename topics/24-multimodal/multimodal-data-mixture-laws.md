---
id: 24-multimodal/multimodal-data-mixture-laws
title: "Data Mixture Laws for Interleaved Multimodal Corpora"
topic: 24-multimodal
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data Mixture Laws for Interleaved Multimodal Corpora

> **Topic:** Multimodal Models · **ID:** `24-multimodal/multimodal-data-mixture-laws` · **Status:** empirically-open

## 1. Problem Statement

Text-only pretraining has a working recipe for choosing corpus proportions: train many small models on many mixtures, fit a parametric map from mixture weights to validation loss, extrapolate to the target scale (Ye et al. 2024; Liu et al. 2025). The question here is whether that recipe survives the move to **interleaved multimodal corpora** — documents such as OBELICS or MMC4 in which images and text alternate inside a single document.

- **Input:** a set of $K$ pretraining sources (interleaved web documents, image–caption pairs, text-only web, OCR/document data, video), a model size $N$, a compute or token budget $D$, and a target evaluation set.
- **Output:** mixture weights $w \in \Delta^{K-1}$ predicted to minimise target loss at $(N, D)$.
- **Decision predicate:** does a law fit on cheap runs predict held-out validation loss at the target scale to within an error smaller than the loss spread between candidate mixtures? If the prediction error exceeds the spread, the law is decorative.

Three variants, different difficulty:

- **Measurement.** Is "mixture weight" even well defined when a single document carries both modalities in a ratio the trainer does not control, and when the image-token count per image is an architecture choice? *Currently ill-posed.*
- **Method.** Fit a predictive functional form for multimodal mixtures. *Runnable, largely unrun at scale.*
- **Theory.** Prove when a low-dimensional mixture-to-loss map exists and when cross-modal transfer is monotone in mixture weight. *Open.*

## 2. Formal Setting

Sources $s_1,\dots,s_K$. Modalities $m \in \mathcal{M}$ (text, image, and where present audio/video). Each source has a **modality composition** $\phi_k \in \Delta^{|\mathcal{M}|-1}$: the fraction of that source's *training tokens* belonging to each modality, measured after tokenisation, not from raw bytes.

Sampling weights $w \in \Delta^{K-1}$ are the fraction of training tokens drawn from each source. The induced modality mixture is

$$\psi(w) \;=\; \sum_{k=1}^{K} w_k \,\phi_k \;\in\; \mathrm{conv}\{\phi_1,\dots,\phi_K\}.$$

**Key structural fact:** $\psi$ ranges over the convex hull of the $\phi_k$, a strict and often small subset of $\Delta^{|\mathcal{M}|-1}$. Interleaved corpora fix $\phi_k$ inside the document; the trainer can only move between sources.

Per-domain validation losses $L_i(w; N, D)$, measured in nats/token on a held-out split of domain $i$, aggregated as $L(w) = \sum_i \lambda_i L_i(w)$ with evaluation weights $\lambda$ chosen by the practitioner.

The working parametric form, transferred from text (Ye et al. 2024):

$$L_i(w) \;=\; c_i \;+\; k_i \exp\!\Big(\sum_{j=1}^{K} t_{ij}\, w_j\Big),$$

fit per target domain $i$, then composed with $N$- and $D$-scaling in Chinchilla form $L(N,D) = E + A N^{-\alpha} + B D^{-\beta}$ (Hoffmann et al. 2022).

The mixed-modal alternative (Aghajanyan et al. 2023) adds an explicit interaction term: the loss of a jointly trained model relative to the sum of unimodal losses defines a **competition/synergy coefficient**, negative when modalities help each other and positive when they contend for capacity.

**How each quantity is actually measured.**

- $\phi_k$: count text tokens with the model's text tokenizer; count image tokens as $n_{\text{img}}$ per image, where $n_{\text{img}}$ is set by the visual encoder (64 for a Perceiver Resampler as in Flamingo; 256–1024 for ViT patch grids; 1024 for the VQ tokenizer in Chameleon).
- $D$: total tokens consumed, mixing text and image tokens in one count.
- $L_i$: next-token cross-entropy. For image tokens this is only defined if images are discretely tokenized; for continuous-embedding architectures there is no image-token likelihood at all, and $L_i$ is restricted to text.

**Assumptions, and which are violated.**

1. *A token is a token* — $D$ is modality-neutral. **Violated.** Halving $n_{\text{img}}$ changes $\phi_k$ and $\psi(w)$ with zero change to the data.
2. *Sources are exchangeable draws from fixed distributions.* **Violated** for interleaved data: image–text alignment within a document is the training signal, and shuffling destroys it.
3. *$\phi_k$ is fixed by the corpus.* **Violated:** image-dropping, resolution schedules and any-resolution tiling change $\phi_k$ mid-run.
4. *Loss predicts downstream capability.* **Weak.** Few-shot in-context ability tracks interleaved data far more than it tracks text loss (MM1, 2024).

## 3. State of the Art

**Text-only mixture laws — established.**
- **DoReMi** (Xie et al., NeurIPS 2023): group-DRO proxy model reweights The Pile; 8B model reaches baseline perplexity with $2.6\times$ fewer steps. Reproduced by third parties.
- **Data Mixing Laws** (Ye et al., ICML 2024): the exponential-of-linear form above; mixture optimised on small proxies and transferred to a 1B model trained on 100B tokens, matching the baseline's 100B-token loss in about 73B tokens.
- **RegMix** (Liu et al., ICLR 2025): 512 proxy models at 1M params / 1B tokens, regression over mixtures, transferred to 1B/7B models. Rank correlation between proxy and target ordering ~0.9.
- **Contested.** **Aioli** (Chen et al., ICLR 2025) reports that several published mixture methods do not reliably beat a simple stratified/proportional baseline once tuning budget is equalised. Treat single-method wins as unablated.

**Multimodal — thinner.**
- **Scaling Laws for Generative Mixed-Modal Language Models** (Aghajanyan et al., 2023): the only systematic mixed-modal scaling study, over text, image, speech, code and paired modalities up to ~30B params. Establishes that competition dominates at small scale and diminishes with $N$, and documents training instability specific to mixed-modal runs. *Established for discretely tokenized modalities only.*
- **MM1** (McKinzie et al., ECCV 2024): mixture ablations at 1.2B params; settles on roughly **45% captioning / 45% interleaved / 10% text-only**. Finding: interleaved data drives few-shot and text-only performance; caption data drives zero-shot. *Established at 1.2B; the ratio itself is a benchmark number at one scale, not a fitted law.*
- **Idefics2 / OBELICS** (Laurençon et al., NeurIPS 2023 D&B; 2024): interleaved-vs-caption ablations showing interleaved corpora are necessary for in-context learning. *Established directionally; no functional form.*
- **Claimed but unablated:** that MM1-style ratios transfer across model scale, vision encoder, and image-token budget. No published sweep varies $n_{\text{img}}$ and mixture jointly.

**No published work fits and validates a predictive mixture law for interleaved multimodal corpora.** That is the gap.

## 4. What Is Known

- Chinchilla-form scaling holds for the text loss of multimodal models over the ranges tested (Hoffmann et al. 2022 for text; Aghajanyan et al. 2023 for mixed-modal, with modality-specific exponents).
- Mixed-modal competition shrinks with scale: unimodal-vs-joint loss gaps that are large below ~1B params narrow toward ~30B (Aghajanyan et al. 2023).
- Interleaved data is not substitutable by captions for few-shot ability: MM1 at 1.2B shows caption-only mixtures collapse in the 4- and 8-shot regime while remaining competitive at 0-shot.
- Adding ~10% text-only data preserves text benchmarks at small cost to VQA (MM1, 1.2B).
- Corpus scales: OBELICS — 141M documents, 115B text tokens, 353M images. MMC4 — 101.2M documents, 43B text tokens, 571M images.
- Text-only mixture laws extrapolate across roughly $100\times$ in proxy-to-target params (RegMix, ICLR 2025).

## 5. What Is Not Known

- **Methodologically blocked.** The definition of a multimodal mixture weight. $\psi(w)$ depends on $n_{\text{img}}$, an architecture parameter. No convention exists for a modality-neutral budget (tokens? FLOPs? images?), so mixtures are not comparable across papers.
- **Methodologically blocked.** A scalar target loss. Continuous-embedding VLMs give no image-token likelihood; the fitted objective is text loss on multimodal context, which is not the quantity practitioners optimise.
- **Empirically open.** Whether the exponential-of-linear form fits multimodal mixtures at all, and whether coefficients $t_{ij}$ fit at 100M params transfer to 8B. Runnable today for well under 100k GPU-hours; unrun publicly.
- **Empirically open.** Whether the MM1 5:5:1 ratio is scale-invariant or a 1.2B artifact.
- **Theoretically open.** Conditions under which $L_i(w)$ is convex in $w$, hence has a unique optimum. Aghajanyan's competition/synergy sign flip with $N$ suggests it is not, and no proof exists either way.
- **Theoretically open.** Identifiability: given only mixtures achievable inside $\mathrm{conv}\{\phi_k\}$, when are per-modality effects separable from per-source effects?

## 6. Why It Is Hard

The obstruction is **non-identifiability under a constrained mixture polytope, compounded by an architecture-dependent unit of measurement.**

Interleaved documents bind image and text content at a ratio the trainer cannot vary. Two candidate causes of a loss change — "more image tokens" and "more OBELICS-style web content" — move together along the only axis available. Their coefficients are confounded, and no amount of extra runs separates them without a source whose $\phi_k$ differs while its content distribution is held fixed. Such a source does not exist naturally; constructing one (same documents, images subsampled) changes the alignment signal that made the corpus useful.

Layered on top: $n_{\text{img}}$ rescales $\phi_k$ without touching the data, so a fitted law is a property of the *(corpus, encoder)* pair, not the corpus. Every published multimodal mixture result uses a different pair.

Compute is a secondary obstruction, not the primary one: 30–60 proxy runs at 100–400M params are affordable. The primary obstruction is that the fitted coefficients do not mean what the paper says they mean.

## 7. Current Research (as of 2026)

- Extension of RegMix-style regression to vision–language mixtures at proxy scale; groups at Sea AI Lab, Shanghai AI Lab and academic labs are the natural pursuers. *(frontier — verify)*
- Apple's MM1/MM1.5 line and HuggingFace's Idefics line continue mixture ablations, but report ratios rather than laws.
- Early-fusion tokenized models (Chameleon-lineage) make image-token likelihood available, reviving the Aghajanyan mixed-modal law programme with modern data. *(frontier — verify)*
- Data-filtering scaling laws (Goyal et al., CVPR 2024) supply the closest existing formalism for a quality–quantity tradeoff in multimodal data, and are being adapted to mixture selection. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does a mixture law fit at small scale predict the loss ordering of multimodal mixtures at 10–30× larger scale, and is that law stable under a change of image-token budget?

**Scale.** Three model sizes: 160M, 410M, 1.4B non-embedding params. Sources $K=4$: OBELICS (interleaved), a caption corpus (COYO/DataComp), text-only web (DCLM), and OCR/document data. Sample 24 mixtures on the simplex by a space-filling design; train each at 160M and 410M for 20B tokens ($\approx$ 24 × 2 runs). Fit $L_i(w) = c_i + k_i \exp(\sum_j t_{ij} w_j)$ per target domain, compose with an $N$-scaling term, and predict the 1.4B result.

**Control arms.** (a) the MM1 5:5:1 caption:interleaved:text ratio, held fixed; (b) uniform stratified sampling, the Aioli baseline. Train both at 1.4B for 20B tokens.
**Second factor.** Repeat the whole design at $n_{\text{img}} = 64$ and $n_{\text{img}} = 256$ with identical data.

**Deciding number.** Mean absolute error of predicted validation loss at 1.4B on 4 held-out mixtures, in nats/token, compared against the observed best-to-worst spread across the 24 mixtures. The law is useful iff

$$\mathrm{MAE}_{\text{held-out}} \;<\; 0.25 \times \big(\max_w L(w) - \min_w L(w)\big).$$

**Second deciding number.** The rank correlation between the mixture ordering at $n_{\text{img}}=64$ and at $n_{\text{img}}=256$. If Spearman $\rho < 0.7$, mixture laws are not corpus properties and every published ratio is encoder-specific.

## 9. Key References

- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Aghajanyan, Yu, Conneau, Hsu, Hambardzumyan, Zhang, Roller, Goyal, Levy, Zettlemoyer. *Scaling Laws for Generative Mixed-Modal Language Models.* ICML, 2023. — arXiv:2301.03728
- **[Foundational]** Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS, 2022. — arXiv:2204.14198
- **[SOTA]** Ye, Liu, Zhang, Yu, Wang, Zhou, Qiu. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* ICML, 2024. — arXiv:2403.16952
- **[SOTA]** Liu, Zeng, He, Pang, Lin. *RegMix: Data Mixture as Regression for Language Model Pre-training.* ICLR, 2025. — arXiv:2407.01492
- **[SOTA]** Xie, Pham, Dong, Du, Liu, Lu, Liang, Le, Ma, Yu. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS, 2023. — arXiv:2305.10429
- **[SOTA]** McKinzie et al. *MM1: Methods, Analysis and Insights from Multimodal LLM Pre-training.* ECCV, 2024. — arXiv:2403.09611
- **[Contrarian]** Chen, Sala et al. *Aioli: A Unified Optimization Framework for Language Model Data Mixing.* ICLR, 2025.
- **[Dataset]** Laurençon et al. *OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.16527
- **[Dataset]** Zhu et al. *Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved with Text.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2304.06939
- **[Related]** Goyal, Maini, Lipton, Raghunathan, Kolter. *Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic.* CVPR, 2024. — arXiv:2404.07177

## 10. Worked Example

Two sources, real counts. OBELICS: 115B text tokens, 353M images. A caption corpus: ~20 text tokens per caption, one image each.

At $n_{\text{img}} = 256$:

$$\phi_{\text{OBELICS}}^{\text{img}} = \frac{353\text{M} \times 256}{115\text{B} + 353\text{M}\times 256} = \frac{90.4}{205.4} = 0.44, \qquad \phi_{\text{cap}}^{\text{img}} = \frac{256}{276} = 0.93.$$

Achievable image-token fractions from these two sources: $[0.44,\,0.93]$. Adding text-only web ($\phi^{\text{img}}=0$) opens $[0,\,0.93]$ — but only by co-varying text *content*, since the text-only source is not the text inside OBELICS.

Now switch to a Perceiver Resampler, $n_{\text{img}} = 64$, byte-identical data:

$$\phi_{\text{OBELICS}}^{\text{img}} = \frac{22.6}{137.6} = 0.164, \qquad \phi_{\text{cap}}^{\text{img}} = \frac{64}{84} = 0.76.$$

The same $w = (0.5, 0.5)$ now yields $\psi^{\text{img}} = 0.46$ instead of $0.69$. A mixture law fit in the first setting reports coefficients $t_{ij}$ against a $w$-axis whose physical meaning has moved by 23 percentage points of image content.

**Where the obstruction becomes visible.** Suppose the fit says loss improves as $w_{\text{OBELICS}}$ rises. Two readings are observationally equivalent inside this polytope: (i) the model needs more *interleaved structure*; (ii) the model needs *fewer image tokens per text token*, and OBELICS is simply the more text-heavy source. Under $n_{\text{img}}=256$ these are the same direction; under $n_{\text{img}}=64$ they still are, because both sources move together. To separate them you need a source with OBELICS's content and the caption corpus's $\phi$ — which means deleting images from OBELICS documents, destroying the interleaving that hypothesis (i) is about.

That is the concrete reason this is empirically open rather than merely unrun: the natural design has one axis and two hypotheses on it, and the unit of the axis is set by the encoder rather than the data.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*