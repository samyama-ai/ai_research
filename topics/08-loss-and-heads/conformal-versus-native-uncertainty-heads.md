---
id: 08-loss-and-heads/conformal-versus-native-uncertainty-heads
title: "Conformal Wrappers Versus Native Uncertainty Heads"
topic: 08-loss-and-heads
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Conformal Wrappers Versus Native Uncertainty Heads

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/conformal-versus-native-uncertainty-heads` · **Status:** empirically-open

## 1. Problem Statement

Two ways exist to make a network report uncertainty.

- **Native head:** train the model to emit a distribution or interval directly — Gaussian mean/variance head, quantile head under pinball loss, evidential head, deep ensemble, MC-dropout. Uncertainty is a *learned output*.
- **Conformal wrapper:** train any point predictor, then calibrate a scalar threshold on a held-out set so the resulting set/interval covers the truth with probability $\ge 1-\alpha$. Uncertainty is a *post-hoc geometric* construction with a finite-sample guarantee.

The open question is not which is *valid* — conformal is valid by construction under exchangeability, native heads are not. It is **whether the native head buys anything conformal cannot get more cheaply**, once both are held to the same marginal coverage.

Three variants, of different difficulty:

- **Measurement:** is there a scoring rule under which a native head, after conformalization, beats a conformalized point predictor? Efficiency (interval width, set size) at matched coverage is the obvious candidate, but it is not the only one, and adaptivity — conditional coverage — is not measured by it.
- **Method:** can the training loss and the conformal score be co-designed so the score is *learned* to be tight, rather than picked from a fixed menu (CQR, APS, RAPS)?
- **Theory:** under what conditions does a well-specified native head yield the conformal-optimal score, i.e. when is the Bayes-optimal predictive density's level set also the width-minimal set at matched coverage?

Solving it means: a stated regime (data size, dimension, heteroscedasticity, distribution shift) in which one family provably or reproducibly dominates, plus the crossover point.

## 2. Formal Setting

Data $(X_i, Y_i)_{i=1}^n \sim P$ on $\mathcal{X}\times\mathcal{Y}$, exchangeable. Split into $\mathcal{D}_{\text{tr}}$ ($n_{\text{tr}}$) and calibration $\mathcal{D}_{\text{cal}}$ ($n_{\text{cal}}$). Miscoverage level $\alpha$.

**Native head.** A model $f_\theta:\mathcal{X}\to\Theta$ parameterizing $q_\theta(y\mid x)$. Trained by minimizing a proper scoring rule, e.g. Gaussian NLL
$$\mathcal{L}_{\text{NLL}}(\theta) = \frac{1}{n_{\text{tr}}}\sum_i \left[\frac{(Y_i-\mu_\theta(X_i))^2}{2\sigma^2_\theta(X_i)} + \tfrac12\log\sigma^2_\theta(X_i)\right],$$
or pinball loss at levels $\alpha/2, 1-\alpha/2$ for a quantile head $\hat q_{\alpha/2}, \hat q_{1-\alpha/2}$. Its native interval is $C^{\text{nat}}_\alpha(x)=[\hat q_{\alpha/2}(x), \hat q_{1-\alpha/2}(x)]$, or $\mu\pm z_{1-\alpha/2}\sigma$.

**Conformal wrapper.** Nonconformity score $s(x,y)\in\mathbb{R}$. Let $\hat s_{(k)}$ be the $k$-th smallest of $\{s(X_i,Y_i)\}_{i\in\mathcal{D}_{\text{cal}}}$ with $k=\lceil (n_{\text{cal}}+1)(1-\alpha)\rceil$. Then $C^{\text{cp}}_\alpha(x)=\{y: s(x,y)\le \hat s_{(k)}\}$ satisfies
$$1-\alpha \;\le\; \Pr[Y_{n+1}\in C^{\text{cp}}_\alpha(X_{n+1})] \;\le\; 1-\alpha+\tfrac{1}{n_{\text{cal}}+1}.$$
CQR (Romano–Patterson–Candès 2019) uses $s=\max\{\hat q_{\alpha/2}(x)-y,\ y-\hat q_{1-\alpha/2}(x)\}$; the wrapper and the native head then share a backbone and differ only by the additive offset $\hat s_{(k)}$.

**Measured quantities.**

- Marginal coverage: $\widehat{\text{Cov}} = \frac{1}{m}\sum_{j=1}^m \mathbf{1}[Y_j\in C(X_j)]$ on a test set of size $m$; binomial standard error $\sqrt{\alpha(1-\alpha)/m}$ — at $\alpha=0.1$, $m=2000$, that is $0.67$ pp.
- Efficiency: mean width $\frac{1}{m}\sum_j |C(X_j)|$ (regression) or mean set size (classification).
- Conditional coverage proxy: worst-slice coverage over $B$ bins of $\hat\sigma(x)$ or of a chosen feature, $\min_b \widehat{\text{Cov}}_b$; also SSC (size-stratified coverage, Angelopoulos et al. 2021).
- Compute: training FLOPs plus calibration cost $O(n_{\text{cal}})$ forward passes.

**Assumptions, and which break.**

- *Exchangeability of calibration and test.* Violated under any covariate shift, temporal drift, or active/online deployment. This is the assumption conformal actually rests on, and it is the one that fails first.
- *Well-specification of $q_\theta$.* Violated essentially always; Gaussian heads on skewed residuals are the standard counterexample.
- *Split independence.* Full conformal and CV+ relax it at $O(K)$ or $O(n)$ training cost.
- *Continuity of scores* (for the upper coverage bound). Ties in discrete $\mathcal{Y}$ need randomized tie-breaking.

## 3. State of the Art

**Established.**

- Split conformal's finite-sample marginal guarantee (Vovk et al. 2005; Lei et al., *JASA* 2018) is a theorem, not a benchmark number, and holds for *any* score including a broken native head's.
- CQR (NeurIPS 2019) established on 11 UCI regression sets that conformalizing a quantile head yields both exact coverage and narrower intervals than conformalizing a mean-only predictor with a constant-width score.
- Impossibility of distribution-free *conditional* coverage: Vovk (2012), Lei & Wasserman (*JRSS-B* 2014), Barber, Candès, Ramdas, Tibshirani (*IMA IAI* 2021) — any method with finite-length intervals and distribution-free conditional validity is essentially vacuous. This bounds what *either* family can promise.
- RAPS (Angelopoulos et al., ICLR 2021) reduced ImageNet top-1 conformal set sizes from ~10+ (APS) to ~2 at 90% coverage on ResNet-152, with size-stratified coverage reported.

**Claimed but unablated.**

- That evidential regression heads (Amini et al., NeurIPS 2020) give calibrated uncertainty — criticized by Meinert, Gawlikowski & Lavin (2023) and by Bengs, Hüllermeier & Waegeman (NeurIPS 2022), who show the evidential objective does not identify the second-order distribution; the head's "epistemic" output is not a consistent estimator of anything.
- That native heads dominate under shift. Ovadia et al. (NeurIPS 2019) show deep ensembles degrade *least* under shift among native methods, but that study contains no conformal arm.
- Head-to-head "conformal vs. native at matched coverage, matched backbone, matched compute" tables exist mostly as single-benchmark numbers inside papers proposing one side. No independent reproduction sweeps both across scale.

## 4. What Is Known

- **Coverage.** Split conformal hits $1-\alpha$ within $1/(n_{\text{cal}}+1)$ by construction; with $n_{\text{cal}}=1000$, $\alpha=0.1$, the guaranteed band is $[0.900, 0.901]$. Native Gaussian NLL heads on UCI-scale data miss nominal coverage by single-digit to tens of percentage points depending on split; MC-dropout and single-model NLL are the worst offenders in Ovadia et al.'s shift sweep (CIFAR-10-C, ImageNet-C, ResNet/CNN scale).
- **Ensembles help, sublinearly.** Lakshminarayanan et al. (NeurIPS 2017): 5-member deep ensembles beat single NLL heads on NLL and Brier at MNIST/SVHN/ImageNet scale, at 5× training cost.
- **Conformalizing a good head is better than conformalizing a bad one.** CQR's UCI results: conformalized quantile intervals are narrower than conformalized-residual intervals at identical 90% coverage, on datasets of $n\approx 10^3$–$10^5$. The wrapper does not remove the value of the head; it removes the head's *coverage* claim.
- **Calibration-set noise is the floor.** Coverage variance across calibration draws is $\Theta(1/n_{\text{cal}})$ (Vovk 2012 beta distribution of conditional coverage); with $n_{\text{cal}}=500$, the coverage sd is ~1.3 pp at $\alpha=0.1$. Any claimed efficiency win smaller than the corresponding width jitter is not measurable at that $n_{\text{cal}}$.
- **Shift breaks the guarantee, and it is patchable.** Weighted conformal (Tibshirani et al., NeurIPS 2019) restores validity under known covariate shift with a likelihood ratio; adaptive conformal (Gibbs & Candès, NeurIPS 2021) restores long-run coverage online without exchangeability, at the price of interval widths that can transiently blow up to $\infty$.

## 5. What Is Not Known

- **Empirically open.** The clean factorial — {native head type} × {conformalize: yes/no} × {matched backbone, matched total FLOPs} × {in-distribution, mild shift, severe shift} × {model scale $10^7 \to 10^{10}$ params} — has not been run and published as a single controlled sweep. Every ingredient exists; the experiment is a compute job, not a research obstacle.
- **Empirically open.** Whether native-head advantages that survive conformalization exist for LLM token/sequence-level uncertainty at all, where the "interval" is a set of sequences and set size is not a natural cost.
- **Theoretically open.** Conditions under which a well-specified native predictive density's $\alpha$-level set is the *minimum-volume* set at matched coverage among all conformal scores derived from that backbone. Known for the oracle density (highest-density region is volume-optimal); unknown once the density is estimated with error $\varepsilon$ — no sharp excess-width bound in terms of $\varepsilon$.
- **Methodologically blocked.** "Adaptivity" has no agreed measurement. SSC, worst-slice coverage, and conditional-coverage regression each rank methods differently, and none is distribution-free-attainable (Barber et al. 2021). Until adaptivity is pinned to a chosen slicing, "the native head is more adaptive" is not a falsifiable claim.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus a fixed impossibility ceiling.**

- Once both arms are conformalized, marginal coverage is *identical by construction* — it carries zero information. The comparison collapses onto efficiency and adaptivity.
- Efficiency is confounded with backbone quality and with $n_{\text{cal}}$. A quantile head trained on $n_{\text{tr}}$ and a mean head trained on the same $n_{\text{tr}}$ do not have the same effective capacity per output; matching FLOPs does not match statistical efficiency.
- Adaptivity, the quantity that would actually distinguish them, is the one Barber et al. proved no distribution-free method can deliver. So any measured adaptivity gap is a statement about the chosen slicing, not about the methods.
- The natural tie-breaker — behavior under shift — is exactly where the conformal guarantee is void, so the arms are compared with neither holding a guarantee, and results become dataset-idiosyncratic.

Compute is a secondary cost: the honest sweep is ~5 head types × 2 wrapper conditions × 4 shift levels × 5 seeds × 4 scales ≈ 800 training runs.

## 7. Current Research (as of 2026)

- **Conformal risk control and beyond-coverage guarantees** — Angelopoulos, Bates, Jordan, Malik, Candès and collaborators (Berkeley/Stanford): controlling expected loss, not just miscoverage. Moves the comparison off interval width, which partly dissolves the confound in §6.
- **Conformal prediction for LLMs** — set-valued and factuality-filtering guarantees (Mohri & Hashimoto, ICML 2024, conformal factuality). Native-head analogues are logit-based confidence and verbalized uncertainty; head-to-head remains thin. *(frontier — verify)*
- **Learned nonconformity scores** — training the score end-to-end with a differentiable surrogate for set size (Stutz, Dvijotham, Cemgil, Doucet, ICLR 2022, "Learning Optimal Conformal Classifiers"). This is the method variant of §1 and is the most direct route to collapsing the two families into one.
- **Critiques of evidential/second-order heads** — Hüllermeier, Bengs, Waegeman; identifiability of epistemic uncertainty. Ongoing.
- **Online/adaptive conformal under drift** — Gibbs & Candès follow-ups, Zaffran et al. (ICML 2022) for time series. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** at matched marginal coverage and matched training FLOPs, does a native head's conformalized interval beat a mean-only backbone's conformalized interval by more than calibration noise — and does that gap survive scale?

**Scale.** Tabular regression at $n=10^5$ (5 UCI/OpenML sets, heteroscedastic) plus one image-regression task (age-from-face, ~200k images) at three backbone sizes: 10M, 300M, 3B params.

**Arms** (identical backbone, identical FLOP budget, 5 seeds each):
1. Mean head + absolute-residual split conformal *(control arm)*.
2. Gaussian NLL head, native interval (no wrapper).
3. Gaussian NLL head, conformalized with normalized score $s=|y-\mu(x)|/\sigma(x)$.
4. Quantile head, native interval.
5. Quantile head + CQR.
6. 5-member ensemble at 1/5 the per-member FLOPs, conformalized.

Fix $\alpha=0.1$, $n_{\text{cal}}=5000$ (coverage sd $\approx 0.42$ pp), test $m=20{,}000$.

**Deciding number.** Relative mean-width reduction of arm 5 over arm 1 (control), averaged over seeds:
$$\Delta = 1 - \frac{\overline{|C^{(5)}|}}{\overline{|C^{(1)}|}}.$$
Decision rule: the native head earns its keep iff $\Delta > 5\%$ with a seed-level 95% CI excluding zero at *every* backbone scale. If $\Delta$ shrinks monotonically with scale — the plausible outcome, since larger backbones estimate conditional mean and spread more accurately from the residual alone — the native head is a small-data device, and the page's status changes to partially-solved with a stated crossover.

Secondary readout: worst-decile coverage over $\hat\sigma(x)$ deciles, reported for all arms, with the explicit caveat that it is a slicing-dependent statistic.

## 9. Key References

- **[Foundational]** Vladimir Vovk, Alexander Gammerman, Glenn Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[Foundational]** Jing Lei, Max G'Sell, Alessandro Rinaldo, Ryan Tibshirani, Larry Wasserman. *Distribution-Free Predictive Inference for Regression.* JASA, 2018. — arXiv:1604.04173
- **[Foundational]** Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS, 2017. — arXiv:1612.01474
- **[SOTA]** Yaniv Romano, Evan Patterson, Emmanuel Candès. *Conformalized Quantile Regression.* NeurIPS, 2019. — arXiv:1905.03222
- **[SOTA]** Anastasios Angelopoulos, Stephen Bates, Jitendra Malik, Michael I. Jordan. *Uncertainty Sets for Image Classifiers using Conformal Prediction.* ICLR, 2021. — arXiv:2009.14193
- **[SOTA]** David Stutz, Krishnamurthy Dvijotham, Ali Taylan Cemgil, Arnaud Doucet. *Learning Optimal Conformal Classifiers.* ICLR, 2022. — arXiv:2110.09192
- **[Theory]** Rina Foygel Barber, Emmanuel Candès, Aaditya Ramdas, Ryan Tibshirani. *The limits of distribution-free conditional predictive inference.* Information and Inference, 2021. — arXiv:1903.04684
- **[Theory]** Ryan Tibshirani, Rina Foygel Barber, Emmanuel Candès, Aaditya Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS, 2019. — arXiv:1904.06019
- **[Theory]** Isaac Gibbs, Emmanuel Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS, 2021. — arXiv:2106.00170
- **[Critique]** Viktor Bengs, Eyke Hüllermeier, Willem Waegeman. *Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation.* NeurIPS, 2022. — arXiv:2203.06102
- **[Empirical]** Yaniv Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[Survey]** Anastasios Angelopoulos, Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* 2021. — arXiv:2107.07511

## 10. Worked Example

Heteroscedastic 1-D problem: $X\sim U[0,10]$, $Y = \sin X + \varepsilon$, $\varepsilon\sim\mathcal{N}(0, (0.1+0.3X)^2)$. Oracle 90% interval width at $x$: $2\times 1.645\times(0.1+0.3x)$, ranging from $0.33$ at $x=0$ to $10.2$ at $x=10$; oracle mean width $= 2\times1.645\times(0.1+1.5)=5.26$.

Take $n_{\text{tr}}=2000$, $n_{\text{cal}}=1000$, $\alpha=0.1$.

- **Arm 1 (mean head + absolute residual).** The score is $|y-\hat\mu(x)|$, so the band is *constant width*. To cover 90% marginally it must be wide enough for the $x\approx 10$ region: the marginal $|{\varepsilon}|$ 90th percentile over $X\sim U[0,10]$ is ≈ $2.9$, giving width ≈ $5.8$. Coverage is exactly 90% marginally — but by decile of $x$ it runs ~100% at $x<2$ and ~55–65% at $x>9$.
- **Arm 5 (quantile head + CQR).** Width tracks $0.1+0.3x$; measured mean width lands near $5.4$–$5.6$ once the finite-sample offset $\hat s_{(k)}$ is added. $\Delta \approx 4$–$7\%$ against arm 1.

**Where the obstruction shows.** The width gap is small — single-digit percent — while the *conditional* coverage gap is enormous: arm 1's worst decile sits near 60% against a nominal 90%; arm 5's near 85%. The number that would justify the native head is therefore the conditional one, and that is the number no distribution-free method is entitled to report as a guarantee (Barber et al. 2021). Worse, the "worst decile of $x$" statistic was chosen by someone who already knew $\sigma$ depended on $x$. Slice instead by $\hat\mu(x)$ — a choice an analyst without the generative model might plausibly make — and, because $\sin x$ is non-monotone, the arm-1 deficit partly averages out and the gap shrinks by roughly half. Same two methods, same data, different verdict, purely from the slicing. That is the methodological block in §5 made concrete: the decisive quantity is defined only relative to a slicing that the experimenter picks, and the width number that *is* well defined is too small to carry the decision.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*