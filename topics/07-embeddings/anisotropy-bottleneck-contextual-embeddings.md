---
id: 07-embeddings/anisotropy-bottleneck-contextual-embeddings
title: "Isotropy and the Anisotropy Bottleneck in Contextual Embeddings"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Isotropy and the Anisotropy Bottleneck in Contextual Embeddings

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/anisotropy-bottleneck-contextual-embeddings` · **Status:** open

## 1. Problem Statement

Contextual embeddings from trained Transformers occupy a narrow cone: two randomly sampled tokens have expected cosine similarity far above $0$, and the spectrum of the embedding covariance is dominated by a few directions. The open problem is whether this **anisotropy is a bottleneck** — a property that causally limits downstream representational quality — or an **artifact of the metric**, harmless once you stop measuring with raw cosine similarity.

Three variants, with different difficulty:

- **Measurement.** Define a scalar $I(\cdot)$ on a set of embeddings that is invariant to the things that should not matter (rescaling, mean shift, sample size, tokenizer frequency skew) and sensitive to the thing that should. Current metrics disagree with each other on the same model.
- **Method.** Find a transformation or training objective that raises $I$ *and* improves a task metric, with an ablation showing the gain comes from the isotropy change rather than from the extra parameters, extra data, or extra normalization that accompanied it.
- **Theory.** Prove or refute: under a softmax output layer with tied embeddings and standard cross-entropy training, is a common dominant direction (a) necessary for the optimum, (b) merely a stable attractor of gradient descent, or (c) neither?

Solving it means: a metric with a stated invariance group, plus a causal experiment where isotropy is intervened on and nothing else is.

## 2. Formal Setting

Let $f_\theta$ be a Transformer, $\ell \in \{1,\dots,L\}$ a layer, and $\mathcal{D}$ a corpus. Sampling a document, then a position within it, gives contextual vectors $h \in \mathbb{R}^d$ drawn from an induced distribution $P_\ell$. Measured quantities, as actually computed:

**Empirical mean and covariance.** From $N$ sampled token vectors $\{h_i\}$ (in practice $N \sim 10^4$–$10^6$, sampled by *token occurrence*, so frequent tokens dominate):
$$\hat\mu = \tfrac{1}{N}\sum_i h_i,\qquad \hat\Sigma = \tfrac{1}{N}\sum_i (h_i-\hat\mu)(h_i-\hat\mu)^\top .$$

**Mean random cosine (Ethayarajh).** $\mathrm{AvgCos}_\ell = \mathbb{E}_{i\ne j}\left[\frac{\langle h_i,h_j\rangle}{\|h_i\|\|h_j\|}\right]$, estimated on random token pairs from *different* contexts.

**Partition-function isotropy (Arora/Mu–Viswanath).** With $Z(a)=\sum_i \exp(a^\top h_i)$ over unit $a$,
$$I_{\mathrm{PF}} = \frac{\min_{a \in \mathcal{A}} Z(a)}{\max_{a \in \mathcal{A}} Z(a)} \in [0,1],$$
where $\mathcal{A}$ is in practice the set of eigenvectors of $\hat\Sigma$ — a *lower*-dimensional surrogate for the true min/max over the sphere, which is not tractable.

**IsoScore (Rudman et al.).** PCA-rotate, take the diagonal variance vector $\sigma \in \mathbb{R}^d_{\ge 0}$, normalize $\hat\sigma = \sqrt{d}\,\sigma/\|\sigma\|_2$, and set $\delta = \|\hat\sigma - \mathbf{1}\|_2/\sqrt{2(d-\sqrt d)}$, then map to a fraction-of-dimensions-used score in $(0,1]$. Unlike $\mathrm{AvgCos}$ and $I_{\mathrm{PF}}$, it is mean-centered and rotation-invariant by construction.

**Rogue-dimension share.** For dimension $k$, its contribution to the expected inner product,
$$\rho_k = \frac{\mathbb{E}_{i,j}\big[(h_{i,k}-\hat\mu_k)(h_{j,k}-\hat\mu_k)\big]}{\mathbb{E}_{i,j}\big[\langle h_i-\hat\mu, h_j-\hat\mu\rangle\big]}.$$

Assumptions, and which are violated:

- *Embeddings are i.i.d. samples from one distribution.* Violated: tokens within a document are strongly correlated, and the space is a union of frequency- and token-identity clusters, not one blob (Cai et al., ICLR 2021).
- *Isotropy is a global property.* Violated: spaces that are globally anisotropic are close to isotropic **within** clusters and on local manifolds.
- *The relevant metric is cosine on raw activations.* Violated in deployment: retrieval stacks apply LayerNorm, whitening, or a trained projection before scoring.
- *$N$ is large enough that $\hat\Sigma$ is well conditioned.* Often violated: $d=768$–$4096$ with $N$ in the low $10^4$ makes the small eigenvalues, which dominate IsoScore and $I_{\mathrm{PF}}$, sample-noise estimates.

## 3. State of the Art

**Established (reproduced independently).** Anisotropy of contextual spaces is a robust measurement across BERT, GPT-2, ELMo, and later decoder-only models (Ethayarajh, EMNLP 2019; Cai et al., ICLR 2021; Godey et al., EACL 2024). Post-hoc removal of the mean plus a few top principal directions ("all-but-the-top", Mu & Viswanath, ICLR 2018) and whitening (Su et al., 2021) reliably raise cosine-based STS correlations for BERT-family encoders. Contrastive sentence training (SimCSE, Gao et al., EMNLP 2021) both raises uniformity in the Wang & Isola (ICML 2020) sense and improves STS — this pairing is reproduced widely.

**Claimed but unablated.** That isotropy *causes* the downstream gain. In nearly every published method the isotropy change is bundled with something else: BERT-flow adds an invertible flow trained on target-domain text; SimCSE adds dropout-augmented contrastive supervision; cluster-based correction (Rajaee & Pilehvar, ACL-IJCNLP 2021) adds a clustering step fit on the evaluation domain. No published arm isolates "same objective, same data, isotropy held fixed."

**Contrary evidence, and it is strong.** Timkey & van Schijndel (EMNLP 2021) show a handful of "rogue" dimensions dominate cosine similarity, and that standardizing them away recovers most of the benefit attributed to isotropy. Ablating the isotropy of static-word-embedding baselines does not transfer cleanly. Rajaee & Pilehvar (Findings EMNLP 2021) report that fine-tuning *decreases* isotropy on several tasks while task performance rises.

**Benchmark-number-only results.** Most isotropy claims rest on STS-B / SentEval Spearman deltas alone. STS is a cosine-similarity benchmark, so a transformation that reshapes cosine geometry is being scored by a metric definitionally sensitive to it. Retrieval (BEIR, MTEB) numbers for isotropy corrections specifically are sparse.

## 4. What Is Known

- **GPT-2, last layer, random token pairs:** $\mathrm{AvgCos} > 0.95$ (Ethayarajh, EMNLP 2019, 117M–345M models). BERT-base layer 12: roughly $0.4$–$0.6$; ELMo layer 2 near $0.6$. Lower layers of all three are markedly more isotropic than upper layers.
- **Rogue dimensions:** in GPT-2, a *single* dimension can account for over $50\%$ of the expected inner product between two random vectors; in BERT the top ~3 dimensions carry a comparable share (Timkey & van Schijndel, EMNLP 2021, base-size models). Standardizing (z-scoring per dimension) before cosine raises Spearman correlation with human word-similarity judgments by tens of points on GPT-2 — larger than most isotropy-correction gains.
- **Metric disagreement:** IsoScore assigns BERT- and GPT-2-scale contextual spaces values well below $0.1$ — i.e. under 10% of dimensions effectively used — while $I_{\mathrm{PF}}$ on the same spaces has been reported above $0.6$ and read as "nearly isotropic" (Rudman et al., Findings ACL 2022, $d = 768$). Two accepted metrics, opposite verdicts, same model.
- **Local structure:** contextual spaces decompose into clusters (by token identity and frequency) that are individually far more isotropic than the whole; removing cluster means recovers most of the global isotropy deficit (Cai et al., ICLR 2021, BERT/GPT-2 base).
- **Origin:** degeneration is traceable to the softmax output layer with rare tokens pushed to a shared direction (Gao et al., ICLR 2019), and anisotropy also appears in the *attention* internals independent of the output layer, including in non-language modalities (Godey et al., EACL 2024).

## 5. What Is Not Known

- **Methodologically blocked.** Which metric to believe. There is no agreed invariance group for $I(\cdot)$: whether the mean should be removed, whether token-frequency-weighted or type-weighted sampling is correct, and how to estimate small eigenvalues at $N \ll d^2$. Until this is fixed, "model A is more isotropic than model B" is not a well-posed claim.
- **Empirically open.** The causal test — hold objective, data, architecture, and parameter count fixed; vary only a term that controls isotropy; measure task quality. Runnable today at 1B parameters for well under $10^4$ GPU-hours. Nobody has published it with a matched control arm.
- **Theoretically open.** Whether anisotropy is *necessary* at the optimum of tied-embedding softmax cross-entropy, or only an attractor of the optimization path. Gao et al. give a degeneration mechanism for rare tokens; there is no theorem stating the minimizer must be anisotropic, nor one showing an isotropic minimizer with equal loss exists. Relatedly open: whether the softmax bottleneck (Yang et al., ICLR 2018) and anisotropy are the same constraint expressed twice.

## 6. Why It Is Hard

The core obstruction is **confounded measurement compounded by non-identifiability of the intervention**.

1. *The evaluation measures the transformation, not the representation.* STS scores cosine similarity; isotropy corrections are cosine-geometry edits. A gain is partly definitional. Under a trained bilinear or learned-projection scorer, most of the gap closes — which suggests the information was present and merely mis-metricized.
2. *You cannot vary isotropy alone.* Every known lever (whitening, flows, contrastive loss, cosine regularizers, output-layer changes) alters the loss landscape or adds fitted parameters. Attributing the effect to isotropy requires a control arm that matches capacity and data exactly, and that arm is usually absent.
3. *Estimation is ill-conditioned.* IsoScore and $I_{\mathrm{PF}}$ depend on the smallest eigenvalues of $\hat\Sigma$, which need $N \gg d$ samples of genuinely independent tokens. Sampling by occurrence from documents delivers far fewer effective samples than the nominal $N$.

## 7. Current Research (as of 2026)

- **Attention-internal anisotropy.** Godey, de la Clergerie, Sagot (Inria/ALMAnaCH) locate anisotropy in self-attention itself, not only in the output embedding, and report it in vision and audio Transformers — decoupling it from the tied-softmax story.
- **Rogue dimensions / outlier features.** Continuing overlap with the quantization literature: emergent outlier channels that break INT8 inference are plausibly the same dimensions that dominate cosine. Whether the two phenomena are one is *(frontier — verify)*.
- **Embedding-model practice.** Production text embedders (E5, GTE, BGE, Matryoshka-style models) ship contrastive training plus L2 normalization and are evaluated on MTEB; isotropy is treated as an implicit side-effect, not a target. No leading MTEB entry cites an isotropy objective.
- **Superposition framing.** Interpretability work (Elhage et al., Anthropic, 2022) reframes non-uniform geometry as feature packing in a low-dimensional space — under which some anisotropy is functional, not pathological. Connecting superposition capacity bounds to isotropy metrics is *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does raising isotropy, holding everything else fixed, change downstream quality?

**Scale.** Two 1.4B-parameter decoder-only models, identical architecture, identical 100B-token corpus, identical seed and schedule. ~$10^3$–$10^4$ A100-hours total.

**Treatment arm.** Add to the LM loss a cosine-regularization term on hidden states, $\lambda \cdot \mathbb{E}_{i \ne j}\big[\cos(h_i,h_j)\big]^2$ at the final layer, with $\lambda$ tuned so IsoScore at layer $L$ lands at $\ge 0.5$ (versus $< 0.1$ untreated).

**Control arm.** Same $\lambda$-scaled auxiliary term with the *sign of the pairing shuffled* so it has matched gradient norm and matched parameter/compute cost but no systematic effect on the covariance spectrum. This is the arm that is missing from the literature; without it the treatment is confounded with "extra regularization".

**Deciding number.** nDCG@10 on BEIR, averaged over 13 public datasets, scored with a **trained linear probe** (not raw cosine), treatment minus control. Pre-register a threshold: a difference below $\pm 1.0$ nDCG point falsifies the bottleneck claim at this scale; a gain above $+2.0$ points supports it. Report $\mathrm{AvgCos}$, IsoScore, $I_{\mathrm{PF}}$, and $\max_k \rho_k$ for both arms so the metric disagreement is on record. Secondary readout: does the treated model still develop rogue dimensions ($\max_k \rho_k > 0.3$)? If yes, isotropy and rogue dimensions are separable phenomena and the metrics are measuring different things.

## 9. Key References

- **[Foundational]** Jiaqi Mu, Pramod Viswanath. *All-but-the-Top: Simple and Effective Postprocessing for Word Representations.* ICLR, 2018. — arXiv:1702.01417
- **[Foundational]** Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-Yan Liu. *Representation Degeneration Problem in Training Natural Language Generation Models.* ICLR, 2019. — arXiv:1907.12009
- **[Foundational]** Kawin Ethayarajh. *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.* EMNLP, 2019. — arXiv:1909.00512
- **[SOTA / contrary]** William Timkey, Marten van Schijndel. *All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality.* EMNLP, 2021. — arXiv:2109.04404
- **[SOTA / measurement]** William Rudman, Nate Gillman, Taylor Rayne, Carsten Eickhoff. *IsoScore: Measuring the Uniformity of Embedding Space Utilization.* Findings of ACL, 2022. — arXiv:2108.07344
- **[SOTA / structure]** Xingyu Cai, Jiaji Huang, Yuchen Bian, Kenneth Church. *Isotropy in the Contextual Embedding Space: Clusters and Manifolds.* ICLR, 2021.
- **[SOTA / method]** Tianyu Gao, Xingcheng Yao, Danqi Chen. *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* EMNLP, 2021. — arXiv:2104.08821
- **[Method]** Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, Lei Li. *On the Sentence Embeddings from Pre-trained Language Models.* EMNLP, 2020. — arXiv:2011.05864
- **[Theory-adjacent]** Tongzhou Wang, Phillip Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML, 2020. — arXiv:2005.10242
- **[Recent]** Nathan Godey, Éric de la Clergerie, Benoît Sagot. *Anisotropy is Inherent to Self-Attention in Transformers.* EACL, 2024. — arXiv:2401.12143
- **[Analysis]** Sara Rajaee, Mohammad Taher Pilehvar. *How Does Fine-tuning Affect the Geometry of Embedding Space: A Case Study on Isotropy.* Findings of EMNLP, 2021.
- **[Related bottleneck]** Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen. *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model.* ICLR, 2018. — arXiv:1711.03953

## 10. Worked Example

Take BERT-base, layer 12, $d = 768$, and $N = 20{,}000$ token vectors sampled from Wikipedia. Suppose the measured statistics are:

- $\mathrm{AvgCos} = 0.52$ → reads as "severely anisotropic".
- After subtracting $\hat\mu$ only: $\mathrm{AvgCos} \approx 0.04$.

The mean alone explains most of the raw number. Now decompose the *centered* inner product. Say the top dimension has $\rho_1 = 0.31$ and the top three sum to $\rho_{1:3} = 0.48$. Then $\mathrm{AvgCos}$ measured on centered vectors is still driven by 3 of 768 coordinates. Z-score each dimension and the residual similarity structure changes again.

Now the obstruction. Compute the two isotropy scores on the *same* centered sample:

- $I_{\mathrm{PF}}$ over the eigenvector set: with eigenvalues spread over three orders of magnitude but the exponential $Z(a)$ dominated by the bulk, values near $0.7$ are typical — "acceptably isotropic".
- IsoScore on the same $\hat\Sigma$: with $\lambda_1/\sum_k \lambda_k \approx 0.3$ and a long tail of near-zero eigenvalues, the score lands below $0.1$ — "under 10% of the space used".

Both are peer-reviewed metrics. They disagree by roughly a factor of seven on the same matrix, because $I_{\mathrm{PF}}$ is an exponential-average dominated by the *large* eigenvalues while IsoScore is an $\ell_2$ deviation dominated by the *small* ones — and the small ones, at $N = 20{,}000$ with $d = 768$ and heavily correlated within-document samples, are largely estimation noise. Doubling $N$ moves IsoScore materially and $I_{\mathrm{PF}}$ barely at all.

So the question "is BERT-base anisotropic?" has no stable answer before the sampling scheme, the centering convention, and the eigenvalue estimator are pinned down. Any experiment that intervenes on isotropy inherits that instability in its outcome variable — which is why the Section 8 design reports all four statistics and decides on a downstream number instead.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*