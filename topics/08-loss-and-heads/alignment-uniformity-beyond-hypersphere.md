---
id: 08-loss-and-heads/alignment-uniformity-beyond-hypersphere
title: "Alignment-Uniformity Decomposition Beyond Hypersphere Embeddings"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Alignment-Uniformity Decomposition Beyond Hypersphere Embeddings

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/alignment-uniformity-beyond-hypersphere` · **Status:** open

## 1. Problem Statement

Wang & Isola (ICML 2020) showed that the InfoNCE contrastive loss, in the infinite-negatives limit on the unit hypersphere $S^{d-1}$, decomposes into two interpretable terms: **alignment** (positive pairs map close) and **uniformity** (the marginal embedding distribution spreads out). Both terms are directly optimizable and each correlates with downstream linear-probe accuracy.

The decomposition is not geometry-agnostic. It rests on $S^{d-1}$ being compact, homogeneous, and carrying a unique normalized Haar measure that is the *unique* minimizer of the Gaussian potential energy. Embedding heads increasingly do not live there: hyperbolic (Poincaré/Lorentz) heads for hierarchy, product-of-spaces heads for mixed curvature, unnormalized Euclidean heads (VICReg, Barlow Twins, SimSiam), and low-rank/whitened heads.

The problem, in three variants:

- **Theory.** For an embedding space $\mathcal{M}$ that is not a compact homogeneous manifold, does the contrastive objective admit an alignment $+$ "spread" decomposition whose spread term has a *unique, attained* minimizer, and is that minimizer independent of the kernel? Solving it means a theorem stating the class of $(\mathcal{M}, \text{kernel})$ pairs for which this holds, plus a counterexample class where it provably fails.
- **Measurement.** What replaces $\mathcal{L}_{\text{unif}}$ as a scalar diagnostic on $\mathcal{M}$, such that it is (a) finite, (b) invariant to the isometry group of $\mathcal{M}$, and (c) comparable across curvatures? Currently no such quantity exists for $\mathcal{M}=\mathbb{H}^d$.
- **Method.** Does explicitly optimizing the correct $\mathcal{M}$-specific pair of terms beat InfoNCE-on-$\mathcal{M}$ on downstream tasks, as it does on the sphere?

## 2. Formal Setting

Encoder $f_\theta: \mathcal{X} \to \mathcal{M}$, augmentation distribution $p_{\text{pos}}(x,x^+)$, marginal $p_{\text{data}}$, induced pushforward $\mu_\theta = f_{\theta\\#}p_{\text{data}}$ on $\mathcal{M}$.

**Sphere baseline** ($\mathcal{M} = S^{d-1}$, geodesic-equivalent chordal distance):

$$\mathcal{L}_{\text{align}}(\theta;\alpha) = \mathbb{E}_{(x,x^+)}\big[\|f_\theta(x)-f_\theta(x^+)\|_2^\alpha\big], \qquad
\mathcal{L}_{\text{unif}}(\theta;t) = \log \mathbb{E}_{x,y \sim p_{\text{data}}}\big[e^{-t\|f_\theta(x)-f_\theta(y)\|_2^2}\big]$$

Measured as: $\mathcal{L}_{\text{align}}$ = mean over a held-out set of $B \geq 10^4$ augmentation pairs, $\alpha=2$; $\mathcal{L}_{\text{unif}}$ = log of the mean over all $\binom{B}{2}$ off-diagonal pairs in the same batch pool, $t=2$. Both are reported in nats and are sensitive to $B$ only through $O(B^{-1/2})$ Monte-Carlo error.

**General manifold.** Let $d_\mathcal{M}$ be the geodesic metric and $K: \mathcal{M}\times\mathcal{M}\to\mathbb{R}$ a kernel. Define spread energy

$$\mathcal{E}_K[\mu] = \iint K(u,v)\, d\mu(u)\, d\mu(v).$$

The sphere result is: for $K(u,v) = e^{-t\|u-v\|^2}$ (or any completely monotonic function of squared chordal distance), the unique minimizer of $\mathcal{E}_K$ over probability measures on $S^{d-1}$ is the normalized surface measure $\sigma$ — Cohn & Kumar's universal optimality (JAMS 2007). This gives $\mathcal{L}_{\text{unif}}$ its meaning: it is a kernel-independent test for "$\mu_\theta = \sigma$".

Assumptions the sphere derivation uses, and their status off-sphere:

| Assumption | On $S^{d-1}$ | Off-sphere status |
|---|---|---|
| $\mathcal{M}$ compact, finite volume | holds | **violated** for $\mathbb{H}^d$, $\mathbb{R}^d$ (infinite volume, no uniform probability measure) |
| Unique isometry-invariant probability measure | holds | **violated** for $\mathbb{H}^d$; holds on compact factors of product spaces |
| Minimizer of $\mathcal{E}_K$ independent of $K$ | holds (universal optimality, $d \in \{1,2,3,8,24\}$ proven strongest; general $d$ for completely monotonic kernels) | **violated** in general: for Riesz $s$-energy on a compact rectifiable $\mathcal{M}$ of dimension $m$, minimizers converge weakly to normalized Hausdorff measure only in the hypersingular regime $s \geq m$; for $s<m$ the equilibrium measure depends on $s$ and on curvature (Borodachov–Hardin–Saff, 2019) |
| Infinite-negatives limit is a good approximation | $M \geq 4096$ typical | same |
| Embedding capacity unconstrained (any $\mu$ realizable) | assumed, false | assumed, false; dimensional collapse (Jing et al., ICLR 2022) |

## 3. State of the Art

**Theory SOTA (established).** Wang & Isola's asymptotic decomposition on $S^{d-1}$; Cohn–Kumar universal optimality supplying uniqueness; Borodachov–Hardin–Saff for the general-manifold energy picture (a mathematics result, not yet imported into the ML literature). HaoChen et al. (NeurIPS 2021) give a *different* route — spectral contrastive loss with a downstream error bound via augmentation-graph eigenvalues — that does not require the sphere but also does not produce an alignment/uniformity pair.

**Empirical SOTA (established, ablated).** SimCSE (EMNLP 2021) uses the alignment–uniformity plane as its explanatory instrument for sentence embeddings; unsupervised SimCSE-BERT-base reaches 76.25 average Spearman across the seven STS tasks versus 56.70 for average BERT-base embeddings, and the alignment/uniformity trajectory is reported per-checkpoint. VICReg (ICLR 2022) and Barlow Twins (ICML 2021) replace uniformity with variance/covariance statistics in unnormalized $\mathbb{R}^d$ — ablated, and the variance term is *not* claimed to be the unique minimizer of anything.

**Claimed but unablated.** Hyperbolic heads report downstream wins — Hyperbolic ViT (CVPR 2022) on CUB/Cars/SOP retrieval, MERU (ICML 2023) for image–text — and MERU reports zero-shot ImageNet accuracy comparable to a matched CLIP baseline on RedCaps-12M. **These are benchmark numbers only**: no paper reports a hyperbolic analogue of $\mathcal{L}_{\text{unif}}$, and none ablates whether the gain comes from curvature, from the entailment-cone regularizer, or from the changed effective temperature. The claim "hyperbolic embeddings trade uniformity for hierarchy" appears in discussion sections with no measured quantity behind it.

## 4. What Is Known

- **Decomposition is exact in the limit.** As negatives $M \to \infty$, InfoNCE on $S^{d-1}$ converges (up to constants) to $\mathcal{L}_{\text{align}} + \mathcal{L}_{\text{unif}}$; optimizing the two terms directly matches or exceeds InfoNCE on CIFAR-10, CIFAR-100, STL-10 and a NYU-Depth-V2 protocol at ResNet-scale encoders (Wang & Isola, ICML 2020).
- **The uniformity number is predictive on the sphere.** Across dozens of pretrained encoders at CIFAR/STL scale, the $(\mathcal{L}_{\text{align}}, \mathcal{L}_{\text{unif}})$ pair correlates with linear-probe accuracy; neither alone does.
- **Exact reference value.** On $S^2$ with $t=2$, under $\sigma$, $\|u-v\|^2 \sim \mathrm{Unif}[0,4]$, so $\mathcal{L}_{\text{unif}} = \log\frac{1-e^{-8}}{8} = -2.0794$ nats. Any measured value above this on $S^2$ is a certified deviation from uniform.
- **Identifiability holds on the sphere.** Contrastive learning with a vMF conditional recovers the latent up to orthogonal transform (Zimmermann et al., ICML 2021) — a compactness-dependent result.
- **Uniformity is not sufficient for downstream success.** Saunshi et al. (ICML 2022) construct encoders with near-identical alignment and uniformity and widely different linear-probe accuracy; inductive bias of the function class matters. Measured at CIFAR/ImageNet-subset scale.
- **Dimensional collapse is real and not visible in $\mathcal{L}_{\text{unif}}$ at small $t$** (Jing et al., ICLR 2022): the embedding spectrum has many near-zero singular values while pairwise-distance statistics look acceptable.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no defined uniformity statistic on $\mathbb{H}^d$. Infinite volume ⟹ no normalized invariant measure ⟹ nothing for "uniform" to name. Every hyperbolic-embedding paper that discusses spread does so informally. Until a finite, isometry-invariant, curvature-comparable statistic is defined, the empirical question cannot even be posed.
- **Theoretically open.** For which $(\mathcal{M}, K)$ is the minimizer of $\mathcal{E}_K$ unique, attained, and $K$-independent? Conjecture: only for compact two-point-homogeneous spaces (spheres, projective spaces). No proof either way for product manifolds $S^{a}\times\mathbb{H}^{b}\times\mathbb{R}^{c}$.
- **Theoretically open.** Whether an alignment/spread decomposition of InfoNCE-on-$\mathcal{M}$ exists at all when the score is $-d_\mathcal{M}(u,v)$ rather than an inner product; the $M\to\infty$ limit requires the log-partition term to be finite, which fails on non-compact $\mathcal{M}$ without an explicit norm penalty.
- **Empirically open.** Whether hyperbolic heads actually gain from curvature or from an implicit radius regularizer. The ablation (hyperbolic head vs. Euclidean head with matched radial-norm penalty, matched compute) is runnable today and has not been run.

## 6. Why It Is Hard

The obstruction is **non-existence of the target object, not compute**. On $S^{d-1}$ the uniformity loss is a proper scoring rule for a well-posed target ($\sigma$). On $\mathbb{H}^d$ the analogous objective is *unbounded below and its infimum is not attained*: a minimizing sequence pushes all embeddings toward the ideal boundary and the energy tends to $-\infty$ (Section 10). So gradient descent on any naive hyperbolic uniformity term optimizes radius, not spread — an evaluation that does not measure what it names.

Two secondary obstructions compound it: (i) **non-identifiability of curvature and temperature** — rescaling curvature $c \to \lambda^2 c$ and temperature $\tau \to \tau/\lambda$ leaves the softmax over geodesic distances nearly invariant in the small-radius regime, so any "curvature helps" claim is confounded with a temperature sweep unless both are swept jointly; (ii) on a general compact $\mathcal{M}$ the equilibrium measure is $K$-dependent, so the diagnostic's value depends on an arbitrary hyperparameter $t$ in a way it does not on the sphere.

## 7. Current Research (as of 2026)

- **Hyperbolic representation learning at scale** — Meta (Nickel, Desai and collaborators, MERU line), and the hyperbolic-vision group around Mettes (Amsterdam) and Oseledets/Sebe (Skoltech/Trento). Focus is downstream benchmarks and entailment structure, not spread diagnostics.
- **Mixed-curvature / product-space heads** — descendants of Gu, Sala, Gunel & Ré (ICLR 2019); learning per-factor curvature end-to-end. *(frontier — verify)* whether any 2025–26 work reports per-factor uniformity.
- **Non-contrastive spread surrogates** — VICReg/Barlow-Twins-style covariance regularizers, which sidestep the measure question entirely by targeting second moments; the open item is whether these have a variational characterization analogous to Cohn–Kumar.
- **Energy-on-manifolds mathematics** — the Borodachov–Hardin–Saff program on Riesz energies over rectifiable sets is the correct tool and is largely uncited by the ML community. *(frontier — verify)* ML-side adoption.

## 8. Concrete Next Experiment

**Question.** Does an isometry-invariant, finite spread statistic on $\mathbb{H}^d$ predict downstream accuracy the way $\mathcal{L}_{\text{unif}}$ does on $S^{d-1}$?

**Proposed statistic (constant-radius conditioning).** Fix the radial marginal, then measure angular spread. For embeddings $u_i \in \mathbb{H}^d$ (Lorentz model), condition on the empirical radius shell and define

$$\widetilde{\mathcal{L}}_{\text{unif}} = \log \mathbb{E}_{i\neq j}\Big[\exp\big(-t\, \delta(u_i,u_j)^2\big)\Big], \qquad \delta(u,v) = d_{\mathbb{H}}(u,v) - |r_i - r_j|,$$

with $r_i = d_{\mathbb{H}}(o, u_i)$ from a learned origin $o$. $\delta$ is invariant under isometries fixing $o$ and is bounded on any radius shell, so the statistic is finite.

**Scale.** ViT-B/16, RedCaps-12M or CC12M, 32 epochs, batch 4096, ~8 A100-days per arm. Four arms:
1. Euclidean/spherical CLIP head (control).
2. Hyperbolic head, InfoNCE over geodesic distances.
3. Hyperbolic head, explicit $\mathcal{L}_{\text{align}} + \lambda\widetilde{\mathcal{L}}_{\text{unif}}$.
4. **Control arm that matters:** Euclidean head with a radial-norm penalty tuned to match arm 2's radius histogram (Wasserstein-1 distance between radius marginals $< 0.05$).

Sweep $\tau \in \{0.01, 0.03, 0.07\}$ and curvature $c \in \{0.1, 0.5, 1.0\}$ *jointly* in arms 2–3 to break the $(c,\tau)$ confound.

**The deciding number.** Spearman correlation $\rho$ between $\widetilde{\mathcal{L}}_{\text{unif}}$ measured at each of ~40 checkpoints (across arms 2–3 and hyperparameters) and zero-shot ImageNet top-1. If $|\rho| \geq 0.7$, the statistic is a working diagnostic and the decomposition transfers. If $|\rho| < 0.3$, it does not, and the hyperbolic-spread story should be retired. Secondary check: arm 4 minus arm 2 zero-shot top-1; if $|\Delta| < 0.5$ points, curvature contributes nothing beyond radial regularization.

## 9. Key References

- **[Foundational]** Tongzhou Wang, Phillip Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML, 2020. — arXiv:2005.10242
- **[Foundational]** Henry Cohn, Abhinav Kumar. *Universally optimal distribution of points on spheres.* Journal of the American Mathematical Society 20(1), 2007.
- **[Foundational]** Sergiy Borodachov, Douglas Hardin, Edward Saff. *Discrete Energy on Rectifiable Sets.* Springer Monographs in Mathematics, 2019.
- **[Foundational]** Maximilian Nickel, Douwe Kiela. *Poincaré Embeddings for Learning Hierarchical Representations.* NeurIPS, 2017. — arXiv:1705.08039
- **[SOTA]** Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, Ramakrishna Vedantam. *Hyperbolic Image-Text Representations.* ICML, 2023. — arXiv:2304.09172
- **[SOTA]** Aleksandr Ermolov, Leyla Mirvakhabova, Valentin Khrulkov, Nicu Sebe, Ivan Oseledets. *Hyperbolic Vision Transformers: Combining Improvements in Metric Learning.* CVPR, 2022. — arXiv:2203.10833
- **[SOTA]** Tianyu Gao, Xingcheng Yao, Danqi Chen. *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* EMNLP, 2021. — arXiv:2104.08821
- **[Theory]** Jeff Z. HaoChen, Colin Wei, Adrien Gaidon, Tengyu Ma. *Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss.* NeurIPS, 2021. — arXiv:2106.04156
- **[Theory]** Nikunj Saunshi, Jordan Ash, Surbhi Goel, Dipendra Misra, Cyril Zhang, Sanjeev Arora, Sham Kakade, Akshay Krishnamurthy. *Understanding Contrastive Learning Requires Incorporating Inductive Biases.* ICML, 2022. — arXiv:2202.14037
- **[Theory]** Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, Wieland Brendel. *Contrastive Learning Inverts the Data Generating Process.* ICML, 2021. — arXiv:2102.08850
- **[Related]** Li Jing, Pascal Vincent, Yann LeCun, Yuandong Tian. *Understanding Dimensional Collapse in Contrastive Self-supervised Learning.* ICLR, 2022. — arXiv:2110.09348
- **[Related]** Adrien Bardes, Jean Ponce, Yann LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR, 2022. — arXiv:2105.04906
- **[Related]** Albert Gu, Frederic Sala, Beliz Gunel, Christopher Ré. *Learning Mixed-Curvature Representations in Product Spaces.* ICLR, 2019.
- **[Survey]** Pascal Mettes, Mina Ghadimi Atigh, Martin Keller-Ressel, Jeffrey Gu, Serena Yeung. *Hyperbolic Deep Learning in Computer Vision: A Survey.* IJCV, 2024. — arXiv:2305.06611

## 10. Worked Example

Take $M=8$ embeddings, $t=2$, and compare the two geometries directly.

**On $S^2$.** The uniform measure gives $\mathcal{L}_{\text{unif}} = \log\frac{1-e^{-8}}{8} = -2.0794$ nats. Eight points in the square-antiprism configuration sit within a few hundredths of this; the minimum is *attained*, and the number is a meaningful target.

**On $\mathbb{H}^2$ (curvature $-1$).** Place the 8 points equally spaced on a hyperbolic circle of radius $R$ about the origin. The nearest-neighbour geodesic distance satisfies $\sinh(d/2) = \sinh R \cdot \sin(\pi/8)$, $\sin(\pi/8)=0.3827$:

```
R = 1:  sinh R = 1.175  ->  d = 0.872,  d^2 = 0.760,  e^{-2d^2} = 2.19e-1
R = 2:  sinh R = 3.627  ->  d = 2.274,  d^2 = 5.171,  e^{-2d^2} = 3.2e-5
R = 5:  sinh R = 74.20  ->  d = 8.080,  d^2 = 65.29, e^{-2d^2} = 1e-57
```

Using $\mathcal{L}_{\text{unif}} = \log \frac{1}{M(M-1)}\sum_{i\neq j} e^{-t d^2}$ and keeping only nearest-neighbour terms as an upper bound:

```
R = 1:   L_unif  <~  log(2.19e-1)   = -1.52
R = 2:   L_unif  <~  log(3.2e-5)    = -10.35
R = 5:   L_unif  <~  log(1e-57)     = -131
R -> inf: L_unif -> -inf
```

**The obstruction, visible.** On the sphere the loss bottoms out at a finite $-2.0794$ and the optimizer is forced to arrange points. In $\mathbb{H}^2$ the same loss has no finite minimum: it decreases without bound as $R$ grows, and the angular arrangement stops mattering once $R \gtrsim 3$ (all cross terms are numerically zero, so the gradient with respect to angle vanishes to float precision). Alignment is unaffected by the inflation — translating a positive pair outward along a geodesic while holding their separation fixed leaves $\mathcal{L}_{\text{align}}$ constant. So the composite objective's only active gradient is "increase radius," and in float32 the run terminates in `inf`/NaN or in a saturated boundary configuration. Any hyperbolic uniformity number reported without radius conditioning is reporting the radius, not the spread.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*