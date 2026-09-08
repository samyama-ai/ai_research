---
id: 07-embeddings/contrastive-sample-complexity
title: "Contrastive Learning Sample Complexity Bounds"
topic: 07-embeddings
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Contrastive Learning Sample Complexity Bounds

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/contrastive-sample-complexity` · **Status:** partially-solved

## 1. Problem Statement

How many unlabeled pairs does contrastive pretraining need to reach a target downstream error, and how does that count depend on the number of negatives, the augmentation distribution, the embedding dimension, and the encoder class?

Three variants, routinely conflated:

- **Theory variant.** Give a bound on downstream excess risk $\varepsilon$ of the form $M \ge \Phi(\varepsilon, k, d, \mathcal{F}, \mathcal{A})$ — unlabeled pairs $M$, negatives $k$, dimension $d$, encoder class $\mathcal{F}$, augmentation family $\mathcal{A}$ — that is (a) non-vacuous at ImageNet scale and (b) correct in its *direction* on $k$ and $d$. Existing bounds fail (a); several fail (b).
- **Measurement variant.** Estimate the empirical sample-complexity curve $\varepsilon(M)$ with everything else fixed, and separate the contribution of *more images* from *more augmentation pairs per image*.
- **Method variant.** Design an objective or sampling scheme whose $\varepsilon(M)$ curve is provably or measurably steeper than InfoNCE at fixed compute.

Solved means: a bound whose predicted $M$ for 5% ImageNet linear-probe excess error is within one order of magnitude of the measured $M$, and whose $\partial M/\partial k$ sign matches experiment.

## 2. Formal Setting

Latent classes $c \sim \rho$ over $\mathcal{C}$, $|\mathcal{C}| = C$; class-conditional data distribution $\mathcal{D}_c$ over $\mathcal{X}$. A positive pair is $(x, x^+) \sim \mathcal{D}_c^{\otimes 2}$; negatives $x_1^-,\dots,x_k^-$ are drawn marginally. The unlabeled sample is $S = \{(x_i, x_i^+, x_{i,1:k}^-)\}_{i=1}^M$. **Measured as:** $M$ = number of distinct anchor images times passes, $k$ = in-batch negatives $B-1$ for batch $B$, or queue length for MoCo.

Encoder $f \in \mathcal{F}$, $f: \mathcal{X} \to \mathbb{R}^d$. Contrastive (logistic/InfoNCE) risk:

$$L_{\mathrm{un}}(f) = \mathbb{E}\left[ -\log \frac{e^{f(x)^\top f(x^+)/\tau_T}}{e^{f(x)^\top f(x^+)/\tau_T} + \sum_{j=1}^k e^{f(x)^\top f(x_j^-)/\tau_T}} \right]$$

with temperature $\tau_T$. Downstream risk uses the **mean classifier** $\mu_c = \mathbb{E}_{x\sim\mathcal{D}_c} f(x)$, scored by $\langle f(x), \mu_c\rangle$; $L_{\sup}(f)$ is its multiclass logistic risk, $\varepsilon$ its excess 0/1 error over the best in class. **Measured as:** frozen-encoder linear probe trained to convergence on the full labeled set — an upper bound on $L_{\sup}$ that is *not* the same object, since the probe is fit, not plugged in.

The **collision probability** $\tau = \Pr[c^- = c^+]$ (a negative that shares the anchor's class) is $\sum_c \rho(c)^2$; for uniform $\rho$, $\tau = 1/C$. Capacity enters as empirical Rademacher complexity $\mathcal{R}_S(\mathcal{F})$.

Assumptions, and their status in practice:

| Assumption | Status |
|---|---|
| Positives are conditionally i.i.d. given a latent class | **Violated** — positives are augmentations of one image, not independent draws |
| Negatives drawn from the class marginal | **Violated** — in-batch negatives are sampled without replacement from a curated, deduplicated corpus |
| Downstream classes = latent contrastive classes | **Violated** — ImageNet's 1000 labels are neither the augmentation-graph clusters nor $\mathcal{C}$ |
| Bound is encoder-agnostic given $\mathcal{R}_S(\mathcal{F})$ | **Violated in effect** — Saunshi et al. (2022) show two encoders with identical $L_{\mathrm{un}}$ and different downstream error |

## 3. State of the Art

**Theory SOTA (established).**
- Arora et al. (ICML 2019): $L_{\sup}(f) \le \frac{1}{1-\tau}\left(L_{\mathrm{un}}(f) - \tau\right)$ plus a generalization term $O\!\left(k\,\mathcal{R}_S(\mathcal{F})/M + \sqrt{\log(1/\delta)/M}\right)$. First end-to-end guarantee; explicitly linear in $k$.
- Tosh, Krishnamurthy & Hsu (ALT 2021; JMLR 2021): under multi-view redundancy, contrastive features make the Bayes-optimal predictor approximately *linear*, with $\tilde{O}(1/\varepsilon^2)$ sample dependence — a landmark-free result that does not need latent-class structure.
- HaoChen et al. (NeurIPS 2021), spectral contrastive loss: error bounded by the augmentation-graph spectral gap; representation dimension $d$ must exceed the number of well-separated clusters. Ties sample complexity to a *measurable* graph quantity.
- Lei, Yang, Ying & Zhou (ICML 2023): generalization bounds with only logarithmic dependence on $k$, removing Arora's $O(k)$ factor.
- Awasthi, Dikkala & Kamath (ICML 2022): with the collision term handled correctly, more negatives do **not** necessarily hurt — the $\tau$-degradation in Arora's bound is an artifact of the analysis, not a theorem about the algorithm.

**Empirical SOTA.** SimCLR (Chen et al., ICML 2020), MoCo v3, DINOv2 (Oquab et al., TMLR 2024), OpenCLIP scaling laws (Cherti et al., CVPR 2023).

**Claimed but unablated.** That "more negatives help" is a general law — SimCLR's batch-size curve is confounded with total optimization steps and with BatchNorm statistics; the effect largely vanishes at long training in the original ablation. That InfoNCE is a mutual-information bound in any operative sense — the bound is loose by orders of magnitude and MI estimates at these scales are unreliable (McAllester & Stratos, AISTATS 2020; Tschannen et al., ICLR 2020). **Benchmark-number-only:** essentially every claim about how much data a contrastive method "needs" — reported as ImageNet-1k linear probe at one dataset size, never as a fitted $\varepsilon(M)$ curve.

## 4. What Is Known

- **Linear probe anchors.** SimCLR ResNet-50, 1.28M ImageNet images, batch 4096, 1000 epochs: 69.3% top-1. MoCo v2 with a 65 536-entry queue: 71.1%. Batch 256 SimCLR at 1000 epochs is within ~1–2 points of batch 8192 — the negatives effect is small once steps are equalized (Chen et al., ICML 2020, Fig. 9).
- **Negatives are not monotone.** Ash et al. (AISTATS 2022) show a U-shaped downstream error in $k$ on CIFAR-10/ImageNet subsets, with the optimum in the tens-to-hundreds, not thousands.
- **Data scaling is a power law, and the exponent depends on the corpus.** Cherti et al. (CVPR 2023) fit zero-shot ImageNet error vs. compute for OpenCLIP on LAION-400M/2B up to ~34B samples seen; exponents differ measurably between LAION and OpenAI's WIT at matched compute. Same architecture, same loss, different $\varepsilon(M)$ — so sample complexity is not a property of the objective alone.
- **Identifiability under strong assumptions.** Zimmermann et al. (ICML 2021): with a uniform latent on the hypersphere and von Mises–Fisher conditionals, InfoNCE recovers the generative latents up to orthogonal transform in the infinite-data limit.
- **Loss value does not determine downstream error.** Saunshi et al. (ICML 2022) construct and exhibit encoders with near-identical $L_{\mathrm{un}}$ and large downstream gaps; inductive bias is a necessary term in any tight bound.

## 5. What Is Not Known

- **Theoretically open.** No non-vacuous finite-sample bound for a realistic encoder class. No lower bound: nobody has proved that $\Omega(g(\varepsilon, C, d))$ unlabeled pairs are *necessary* for a stated augmentation family. No bound that includes the inductive-bias term Saunshi et al. proved is required.
- **Empirically open.** The $\varepsilon(M)$ curve for a fixed recipe with *compute held constant* — vary $M \in \{10^5, 10^6, 10^7, 10^8\}$ at fixed total gradient steps and fixed $k$. Runnable today for roughly $10^4$–$10^5$ GPU-hours; not published as a clean single-variable sweep.
- **Methodologically blocked.** "Downstream error" has no canonical referent. Linear probe, $k$-NN, full finetune, and zero-shot transfer rank encoders differently, and the theory's mean classifier matches none of them. Also blocked: counting a "sample" — an image seen with 200 augmentations is not 200 independent pairs, and no accepted effective-sample-size correction exists.

## 6. Why It Is Hard

The specific obstruction is a **capacity term that dominates everything else**. Arora-style bounds split into a surrogate gap (tight to within a small constant, measurable) and $O(k\,\mathcal{R}_S(\mathcal{F})/M)$. For any encoder anyone actually uses, the best available complexity estimate for $\mathcal{F}$ exceeds $M$, so the bound exceeds 1 and says nothing. Improving the contrastive-specific part of the analysis — better $k$-dependence, better collision handling — does not touch the term that makes the bound vacuous. That term is a deep-learning generalization problem wearing a contrastive-learning label.

Second obstruction: **absent ground truth for $\mathcal{C}$**. $\tau$, $C$ and the augmentation graph's spectral gap are all unobservable; substituting ImageNet labels for latent classes assumes exactly the correspondence the theory needs to prove.

## 7. Current Research (as of 2026)

- Augmentation-graph and spectral analyses extending HaoChen et al. to transfer and to multimodal pairs (Stanford / Tengyu Ma's group and successors).
- Inductive-bias-aware bounds following Saunshi et al. — function-class-dependent rather than loss-value-dependent guarantees *(frontier — verify)*.
- Data-centric scaling: DataComp (Gadre et al., NeurIPS 2023) and data-pruning results (Sorscher et al., NeurIPS 2022) reframe the question as *which* $M$ pairs, showing corpus composition moves the exponent more than objective choice.
- Sharper statistical rates for contrastive risk minimization with mild $k$-dependence (Lei et al. and follow-ups) *(frontier — verify)*.
- Sample complexity of multimodal contrastive learning (CLIP-type) as a distinct object, where positives are cross-modal and the redundancy assumption is more nearly true *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** is the contrastive $\varepsilon(M)$ curve governed by distinct images or by total positive pairs seen?

- **Scale.** ResNet-50 SimCLR, ImageNet-1k. Five arms, $M \in \{64\text{k}, 128\text{k}, 256\text{k}, 640\text{k}, 1.28\text{M}\}$ distinct images, class-balanced subsets. **Every arm gets identical total optimization: 1000 ImageNet-equivalent epochs of gradient steps (≈312k steps at batch 4096), fixed $k = 4095$, fixed LR schedule.** Small arms therefore see each image many more times. ~5×1000 ResNet-50 epochs ≈ 3–5k A100-hours.
- **Control arm.** Same five $M$ values at *fixed epochs over the subset* (the conventional, compute-confounded protocol). The contrast between arms isolates data-diversity from optimization.
- **Deciding number.** The fitted exponent $\alpha$ in $\varepsilon(M) = a M^{-\alpha} + \varepsilon_\infty$ under the compute-matched arm. If $\alpha \ge 0.10$, distinct images carry the sample complexity and pair-count is a poor proxy. If $\alpha \le 0.03$ (curve nearly flat once steps are matched), then published "data scaling" for contrastive learning is mostly compute scaling, and every bound stated in $M$ is measuring the wrong variable. Report $\alpha$ with a bootstrap CI over 3 seeds; the arms are 5 points, so the CI matters.

## 9. Key References

- **[Foundational]** Arora, Khandeparkar, Khodak, Plevrakis, Saunshi. *A Theoretical Analysis of Contrastive Unsupervised Representation Learning.* ICML, 2019. — arXiv:1902.09229
- **[Foundational]** Tosh, Krishnamurthy, Hsu. *Contrastive learning, multi-view redundancy, and linear models.* ALT, 2021. — arXiv:2008.10150
- **[SOTA-theory]** HaoChen, Wei, Gaidon, Ma. *Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss.* NeurIPS, 2021. — arXiv:2106.04156
- **[SOTA-theory]** Saunshi, Ash, Goel, Misra, Zhang, Arora, Kakade, Krishnamurthy. *Understanding Contrastive Learning Requires Incorporating Inductive Biases.* ICML, 2022. — arXiv:2202.14037
- **[SOTA-theory]** Awasthi, Dikkala, Kamath. *Do More Negative Samples Necessarily Hurt in Contrastive Learning?* ICML, 2022. — arXiv:2205.01789
- **[SOTA-theory]** Lei, Yang, Ying, Zhou. *Generalization Analysis for Contrastive Representation Learning.* ICML, 2023.
- **[Empirical]** Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML, 2020. — arXiv:2002.05709
- **[Empirical]** Ash, Goel, Krishnamurthy, Misra. *Investigating the Role of Negatives in Contrastive Representation Learning.* AISTATS, 2022. — arXiv:2106.09943
- **[Empirical]** Cherti, Beaumont, Wightman, Wortsman, Ilharco, Gordon, Schuhmann, Schmidt, Jitsev. *Reproducible Scaling Laws for Contrastive Language-Image Learning.* CVPR, 2023. — arXiv:2212.07143
- **[Related]** Zimmermann, Sharma, Schneider, Bethge, Brendel. *Contrastive Learning Inverts the Data Generating Process.* ICML, 2021. — arXiv:2102.08850
- **[Survey]** Balestriero, Ibrahim, Sobal, Morcos, Shekhar, Goldstein, Bordes, Bardes, Mialon, Tian, Schwarzschild, Wilson, Geiping, Garrido, Fernandez, Bar, Pirsiavash, LeCun, Goldblum. *A Cookbook of Self-Supervised Learning.* arXiv, 2023. — arXiv:2304.12210

## 10. Worked Example

CIFAR-10, SimCLR with ResNet-18, $M = 50{,}000$ images, uniform classes so $C = 10$ and $\tau = 1/C = 0.1$. Evaluate Arora's bound with $k = 1$.

Measured $k{=}1$ contrastive risk of a converged encoder: $L_{\mathrm{un}} \approx 0.45$ nats. The surrogate step gives

$$L_{\sup} \le \frac{1}{1-\tau}\left(L_{\mathrm{un}} - \tau\right) = \frac{0.45 - 0.10}{0.90} = 0.389 \text{ nats}.$$

Converting logistic loss to 0/1 error via $\ell_{0\text{-}1} \le \ell_{\log}/\log 2$:

$$\varepsilon_{\text{bound}} \le 0.389 / 0.693 = 0.56.$$

Measured linear-probe error for this setup is about $0.085$ (91.5% top-1). So the **population** part of the bound is loose by $6.6\times$ — imperfect, but informative.

Now add the finite-sample term, $O(k\,\mathcal{R}_S(\mathcal{F})/M)$. ResNet-18 has $1.1 \times 10^7$ parameters against $M = 5 \times 10^4$ samples. Every norm-based Rademacher estimate available for this network on CIFAR-10 yields $\mathcal{R}_S(\mathcal{F})/M \gg 1$; the additive term alone exceeds 1, so the *stated theorem* certifies $\varepsilon \le 1$ — vacuous, while the network sits at $0.085$.

That is the obstruction in one calculation. The contrastive-specific machinery — collision probability, negatives, the surrogate inequality — is loose by a single-digit factor. The capacity term is loose by more than an order of magnitude and swallows the entire bound. Better theorems about $k$ will not fix a page whose binding constraint is $\mathcal{R}_S(\mathcal{F})$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*