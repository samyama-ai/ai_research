---
id: 07-embeddings/intrinsic-dimension-estimation-representations
title: "Intrinsic Dimension Estimation for High-Dimensional Neural Representations"
topic: 07-embeddings
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Intrinsic Dimension Estimation for High-Dimensional Neural Representations

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/intrinsic-dimension-estimation-representations` · **Status:** methodologically-blocked

## 1. Problem Statement

**Input.** A finite set of activation vectors $X = \{x_1,\dots,x_N\} \subset \mathbb{R}^D$ drawn by running a trained network on $N$ inputs and reading one layer, with $D$ typically $10^3$–$10^4$ (ResNet penultimate $D=2048$; Llama-family residual stream $D=4096$–$8192$).

**Output.** A scalar $\hat d(X) \in [0, D]$ claimed to be the *intrinsic dimension* (ID) — the number of coordinates needed to parameterize the set the representations actually occupy.

Three variants, routinely conflated:

- **Measurement.** Given $X$, estimate $\hat d$ with quantified error. This is where the field is stuck: the estimand is only defined relative to a manifold assumption the data violates, and no estimator reports a calibrated interval.
- **Method.** Build an estimator that is unbiased and stable for $d \gtrsim 20$ at feasible $N$, and that degrades gracefully when the support is not a manifold.
- **Theory.** Prove sample-complexity bounds: how many points are needed to distinguish $d$ from $d+1$ at a given confidence, and what the minimax rate is under noise.

**What would count as solving it.** An estimator plus a certificate: on a synthetic family with known $d$ *and* the nuisance structure real representations have (curvature, non-uniform density, anisotropic noise, multiple components of different dimension), it recovers $d$ to $\pm 10\%$ at $N \le 10^5$ for $d \le 50$; and on real representations, two estimators from different families agree within their stated intervals.

## 2. Formal Setting

Let $f_\ell : \mathcal{X} \to \mathbb{R}^{D_\ell}$ be the map from input to layer-$\ell$ activations, $\mu$ the input distribution, and $\nu_\ell = (f_\ell)_\\# \mu$ the pushforward. The **generative estimand** assumes $\operatorname{supp}(\nu_\ell) = \mathcal{M}$, a $d$-dimensional $C^2$ submanifold, and defines $d = \dim \mathcal{M}$.

**As measured.** Every practical estimator instead reads a scaling exponent from nearest-neighbour distances. Let $r_k(x)$ be the distance from $x$ to its $k$-th neighbour in $X$.

Levina–Bickel MLE (2004), the most-used estimator:

$$\hat d_{\mathrm{MLE}}^{-1} = \frac{1}{N}\sum_{i=1}^{N} \left[\frac{1}{k-1}\sum_{j=1}^{k-1} \log \frac{r_k(x_i)}{r_j(x_i)}\right]$$

TwoNN (Facco et al., 2017) uses only $\mu_i = r_2(x_i)/r_1(x_i)$; under local uniformity $\mu_i \sim \mathrm{Pareto}(d)$, giving

$$\hat d_{\mathrm{TwoNN}} = \frac{N}{\sum_i \log \mu_i}.$$

Correlation dimension (Grassberger–Procaccia, 1983) reads the slope of $\log C(\varepsilon)$ against $\log\varepsilon$, $C(\varepsilon) = \binom{N}{2}^{-1}\sum_{i<j}\mathbf{1}[\|x_i-x_j\|<\varepsilon]$.

**Assumptions, and their status in practice:**

| Assumption | Status on neural activations |
| --- | --- |
| Support is a single manifold of constant $d$ | **Violated.** Class-conditional components have different dimensions; Doimo et al. (2020) show hierarchical cluster structure, not one sheet. |
| Density locally uniform inside $r_k$ | **Violated.** Activation densities are strongly non-uniform; TwoNN's Pareto derivation is exactly this assumption. |
| Noise is negligible at scale $r_k$ | **Violated.** Off-manifold noise inflates $\hat d$ at small $\varepsilon$; curvature deflates it at large $\varepsilon$. There is no principled scale selection. |
| Euclidean metric on raw activations is the right one | **Unjustified.** LayerNorm, per-channel scale, and outlier "massive activation" dimensions make $\hat d$ preprocessing-dependent. |

The estimand is therefore not identified by the data alone: $\hat d$ is a function of $(k, N, \text{scale}, \text{metric})$ as much as of $\nu_\ell$.

## 3. State of the Art

**Empirical SOTA (established).**
- Facco et al., *Estimating the intrinsic dimension of datasets by a minimal neighborhood information*, Scientific Reports 2017 — TwoNN; robust to curvature at the scale of two neighbours; the default in the neural-representation literature.
- Ansuini, Laio, Macke & Zoccolan, NeurIPS 2019 — measured ID layer-by-layer in CNNs; found the "hunchback" profile (ID rises then falls) and that last-hidden-layer ID correlates with generalization error across architectures. Independently reproduced.
- Denti, Doimo, Laio & Mira, *The generalized ratios intrinsic dimension estimator*, Scientific Reports 2022 — Gride: TwoNN generalized to neighbour ranks $(n, 2n)$, giving an explicit scale knob and a scale–ID curve rather than a point estimate. The most honest current tool.
- Bac, Mirkes, Gorban, Tyukin & Zinovyev, *scikit-dimension*, Entropy 2021 — the reference implementation set; makes estimator disagreement easy to reproduce.

**Claimed but unablated.**
- Pope, Zhu, Abdelkader, Goldblum & Goldstein, ICLR 2021 — ID of natural image *datasets* $\approx 26$–$43$; validated on GAN samples with controlled latent dimension. The validation is on GAN manifolds, which satisfy the manifold assumption by construction; transfer of that calibration to activations is asserted, not shown.
- Aghajanyan, Gonzalez & Zettlemoyer, ACL 2021 — "intrinsic dimensionality" of fine-tuning, e.g. RoBERTa on MNLI reaching 90% of full performance in a random subspace of $\sim 10^2$–$10^3$ dimensions. This is a *task-subspace* quantity, not the geometric ID of a representation; the shared name causes persistent conflation.
- Tulchinskii et al., NeurIPS 2023 — persistent-homology dimension (PHD) of text embeddings separates human from AI text ($\approx 9$ vs $\approx 7$ average). Reported as a benchmark AUROC; no ablation isolating dimension from density or length effects.

**Theory SOTA.** Minimax rates for manifold estimation exist (Genovese, Perone-Pacifico, Verdinelli & Wasserman, Annals of Statistics 2012) but assume reach bounded below and known noise — conditions no one has verified for activations. There is no finite-sample confidence interval for $\hat d$ on non-manifold support.

## 4. What Is Known

- **Extreme compression is real.** Ansuini et al. (2019): ResNet-50 last hidden layer on ImageNet gives $\hat d_{\mathrm{TwoNN}} \approx 12$–$25$ against $D = 2048$; PCA linear dimension for the same 90% variance is $\gtrsim 10^2$. Scale: ImageNet-1k, $N \approx 10^4$ per estimate.
- **Nonlinear $\ll$ linear.** The gap between $\hat d$ and the PCA rank is an order of magnitude across CNNs and transformers — one of the few findings reproduced across labs, architectures and modalities.
- **Layerwise profile.** Transformers show ID peaking in early-to-middle layers then falling; Valeriani, Doimo, Cuturello, Laio, Ansuini & Cazzaniga, NeurIPS 2023, report this for protein and image transformers, with a low-ID "semantic" plateau in the second half.
- **Sample-size sensitivity is large and systematic.** $\hat d$ grows with $N$ and shrinks with $k$; Levina–Bickel MLE has a known negative bias for large $d$, corrected by MacKay & Ghahramani's (2005) inverse-averaging fix — which changes reported values by 10–20% at $d\approx 20$.
- **A hard ceiling.** The Eckmann–Ruelle (1992) argument gives $N \gtrsim 10^{d/2}$ points for a reliable correlation-dimension estimate: $d = 20$ needs $10^{10}$ samples. Every published estimate with $\hat d > 15$ at $N = 10^4$ is below this bar.
- **Estimators disagree on the same data.** Camastra & Staiano's survey (Information Sciences, 2016) documents saturation of most estimators above $d \approx 10$–$20$; in scikit-dimension runs, spreads of 2–3$\times$ across estimator families on identical activation matrices are routine.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no agreed definition of ID for supports that are not manifolds — unions of components with different dimensions, fractal or filamentary structure, heavy off-manifold noise. Since neural representations are demonstrably of this kind, published $\hat d$ values name a quantity that is not defined for the object measured. No estimator ships a confidence interval that accounts for manifold-assumption failure.
- **Methodologically blocked.** Metric choice. Whether to normalize, whiten, or drop outlier dimensions changes $\hat d$ by tens of percent, and no principle selects among them.
- **Empirically open.** Whether the $\hat d$–generalization correlation of Ansuini et al. survives at $10^9$–$10^{11}$ parameters, across pretraining data scale, with $N \ge 10^6$ probe points. Runnable today; unrun at that scale.
- **Empirically open.** Whether layerwise ID profiles predict anything actionable — layer choice for probing, LoRA rank, pruning ratio — beyond correlating with it post hoc.
- **Theoretically open.** Minimax sample complexity for distinguishing $d$ from $d+1$ under additive full-rank noise of variance $\sigma^2$. No matching upper and lower bounds.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability compounded by absent ground truth**.

Non-identifiability: for any finite $N$, a $d$-manifold with noise $\sigma$ and a $(d+m)$-manifold with smaller noise produce nearest-neighbour statistics that are indistinguishable within sampling error. The scaling exponent is only defined in a limit $\varepsilon \to 0$ that finite $N$ never reaches — and at small $\varepsilon$ the noise dimension dominates. Gride makes the scale dependence visible but does not resolve it.

Absent ground truth: there is no dataset of real neural activations with known $d$. Synthetic benchmarks (spheres, Swiss rolls, GAN latents) satisfy exactly the assumptions that activations violate, so passing them is uninformative — an estimator can be calibrated on manifolds and still be arbitrarily wrong on a union of 1000 class-conditional components.

Compute is *not* the obstruction. TwoNN on $10^5\times 4096$ activations is minutes on one GPU. The obstruction is that running it more does not make the number mean more.

## 7. Current Research (as of 2026)

- **Scale-resolved estimation.** Laio's group (SISSA) — Gride, DADApy toolkit, density-peak clustering combined with ID. Direction: report $\hat d(\varepsilon)$ curves and heterogeneous per-point ID rather than one scalar. Most likely route out of the block.
- **Local ID (LID).** Houle's LID formalism and its use in adversarial-example and OOD detection (Ma et al., ICLR 2018) — treats ID as a per-point property, sidestepping the constant-$d$ assumption. *(frontier — verify)* extension to per-token LID in LLM residual streams.
- **ID of LLM representations across training.** Groups at EPFL, Meta and SISSA tracking ID through pretraining and fine-tuning; claims of ID collapse during instruction tuning. *(frontier — verify)* — reported at small scale, not independently reproduced.
- **Topological estimators.** Persistent-homology dimension as an alternative exponent (Tulchinskii et al., 2023; Adams et al., JMLR 2020 on PH dimension estimation). Different bias profile, same scale-selection problem.
- **Manifold hypothesis testing.** Fefferman, Mitter & Narayanan (JAMS 2016) gave a testing procedure for the manifold hypothesis; applying it as a *precondition* before quoting $\hat d$ is proposed but, as far as is public, not done on activations.

## 8. Concrete Next Experiment

**Question.** Does any current ID estimator return a number that survives the failure of the constant-dimension assumption?

**Scale.** Pythia-1.4B and Llama-3-8B, layers $\{4, 8, 16, 24, 32\}$, $N = 2\times10^5$ token activations sampled from The Pile validation split. Estimators: TwoNN, Levina–Bickel MLE ($k=5,10,20$), Gride across ranks $n \in \{1,2,4,\dots,256\}$, PHD. All via scikit-dimension / DADApy.

**Control arm.** A *matched synthetic mixture*: 1000 components, component $j$ a $d_j$-dimensional Gaussian-noise-corrupted patch with $d_j$ drawn from a known distribution (mean 20, spread 5–60), component sizes and pairwise distances matched to the empirical class/cluster statistics of the real activations, plus isotropic noise at the empirically measured off-cluster variance. Ground truth known by construction. Second control: single manifold, $d = 20$, same $N$ — the easy case every estimator already passes.

**Deciding number.** On the matched mixture, the relative error $|\hat d - \bar d_{\text{true}}| / \bar d_{\text{true}}$, where $\bar d_{\text{true}}$ is the density-weighted mean component dimension.

- If every estimator exceeds **30%** relative error on the mixture while staying under 10% on the single manifold, the field's ID numbers for real representations carry no calibrated meaning, and the block is confirmed as a definition problem, not a tuning problem.
- If one estimator (most plausibly Gride at rank chosen by plateau detection) stays under 15%, that estimator becomes the reference and the block downgrades to *empirically open*.

Cost: under 500 GPU-hours, dominated by activation extraction.

## 9. Key References

- **[Foundational]** E. Levina, P. Bickel. *Maximum Likelihood Estimation of Intrinsic Dimension.* NeurIPS, 2004.
- **[Foundational]** P. Grassberger, I. Procaccia. *Measuring the strangeness of strange attractors.* Physica D, 1983.
- **[Foundational]** C. Fefferman, S. Mitter, H. Narayanan. *Testing the manifold hypothesis.* Journal of the AMS, 2016. — arXiv:1310.0425
- **[SOTA]** E. Facco, M. d'Errico, A. Rodriguez, A. Laio. *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Scientific Reports 7:12140, 2017.
- **[SOTA]** F. Denti, D. Doimo, A. Laio, A. Mira. *The generalized ratios intrinsic dimension estimator.* Scientific Reports 12:20005, 2022.
- **[SOTA]** A. Ansuini, A. Laio, J. H. Macke, D. Zoccolan. *Intrinsic dimension of data representations in deep neural networks.* NeurIPS, 2019. — arXiv:1905.12784
- **[SOTA]** P. Pope, C. Zhu, A. Abdelkader, M. Goldblum, T. Goldstein. *The Intrinsic Dimension of Images and Its Impact on Learning.* ICLR, 2021. — arXiv:2104.08894
- **[SOTA]** L. Valeriani, D. Doimo, F. Cuturello, A. Laio, A. Ansuini, A. Cazzaniga. *The geometry of hidden representations of large transformer models.* NeurIPS, 2023.
- **[Related]** A. Aghajanyan, L. Zettlemoyer, S. Gupta. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL, 2021. — arXiv:2012.13255
- **[Tooling]** J. Bac, E. Mirkes, A. Gorban, I. Tyukin, A. Zinovyev. *Scikit-Dimension: A Python Package for Intrinsic Dimension Estimation.* Entropy 23(10):1368, 2021.
- **[Survey]** F. Camastra, A. Staiano. *Intrinsic dimension estimation: Advances and open problems.* Information Sciences, 2016.

## 10. Worked Example

Take ResNet-50 penultimate activations on ImageNet validation, $D = 2048$, $N = 5\times10^4$.

**Step 1 — the headline.** TwoNN returns $\hat d \approx 20$. Compression factor 100$\times$ against $D$. This is the number that gets cited.

**Step 2 — the sample-size check.** Subsample to $N = 5\times10^3$ and re-run. $\hat d$ drops to roughly 14–16, a consistent effect of the shrinking neighbourhood scale. The estimate is not converged: it is still a function of how many points were drawn.

**Step 3 — the feasibility check.** Eckmann–Ruelle requires $N \gtrsim 10^{d/2}$. At $\hat d = 20$ that is $10^{10}$ points. We have $5\times10^4$, six orders of magnitude short. The estimator returns 20 regardless — it has no way to signal that the estimate is unsupported.

**Step 4 — the assumption check.** Split by class. Estimating per class ($N \approx 50$ each) gives values scattered from about 5 to above 30, with enormous variance at that sample size. Whatever the pooled 20 is, it is not "the dimension of the manifold": it is a scale-dependent average over components whose dimensions differ by $6\times$.

**The obstruction, made visible.** Three numbers — 20, 14, and a per-class spread of 5–30 — come from the same activations and the same estimator, differing only in $N$ and in whether the support was assumed connected. None is wrong; there is no defined quantity for them to be wrong about. That is why the status is *methodologically blocked* rather than *empirically open*: more compute produces more numbers, not a better-defined one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*