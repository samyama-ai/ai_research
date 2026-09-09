---
id: 33-uncertainty-calibration/distribution-free-conditional-coverage
title: "Distribution-Free Conditional Coverage Impossibility"
topic: 33-uncertainty-calibration
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distribution-Free Conditional Coverage Impossibility

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/distribution-free-conditional-coverage` · **Status:** partially-solved

## 1. Problem Statement

Conformal prediction returns a set $\hat C(X)$ that contains the label with probability $1-\alpha$ **on average over the population**. Users read it as a per-input guarantee: "for *this* patient, 90% coverage." That reading is false, and the gap is not a bug to be engineered away — it is a theorem.

Three variants, with different difficulty:

- **Theory.** Does there exist a procedure with $P(Y \in \hat C(X) \mid X = x) \ge 1-\alpha$ for a.e. $x$, uniformly over all distributions $P$, and finite expected set size? **Answer: no** (Vovk 2012; Lei & Wasserman 2014; Barber, Candès, Ramdas & Tibshirani 2021). The theory variant is closed.
- **Method.** What is the strongest relaxation still achievable distribution-free, and how tight is it? Coverage conditional on a *pre-specified* finite-dimensional class of covariate functions is achievable (Gibbs, Cherian & Candès 2023); coverage on *all* sets of probability $\ge \delta$ costs sample complexity in $1/\delta$. The frontier is the exchange rate between the richness of the conditioning class, the sample size, and the set width. **Open.**
- **Measurement.** Given a deployed predictor, how do you *estimate* its conditional coverage deviation, when the target is a function of $x$ and you see one label per $x$? **Methodologically blocked** — every current estimator (worst-slab, group-conditional, binned) is a proxy with its own bias, and no two papers use the same one.

Solving the practically live version means: a procedure plus an estimator such that the estimator's reported conditional-coverage gap is unbiased under a known null, and the procedure provably shrinks it without unbounded width inflation.

## 2. Formal Setting

Data $(X_i, Y_i)_{i=1}^{n} \stackrel{\text{iid}}{\sim} P$ on $\mathcal{X}\times\mathcal{Y}$, $\mathcal{X}\subseteq\mathbb{R}^d$, plus a test point $(X_{n+1},Y_{n+1})\sim P$. A procedure maps the sample to a set-valued map $\hat C_n:\mathcal{X}\to 2^{\mathcal{Y}}$.

**Marginal coverage** (what split conformal delivers, measured as the fraction of held-out points inside the set):
$$P\big(Y_{n+1}\in \hat C_n(X_{n+1})\big)\ \ge\ 1-\alpha .$$

**Conditional coverage** (what users assume; *not directly measurable*, since each $x$ is seen once):
$$P\big(Y_{n+1}\in \hat C_n(X_{n+1}) \mid X_{n+1}=x\big)\ \ge\ 1-\alpha \quad \text{for } P_X\text{-a.e. } x .$$

**Relaxation A — set-conditional.** For a class $\mathcal{A}$ of measurable $A\subseteq\mathcal{X}$ with $P(A)\ge\delta$: $P(Y\in\hat C(X)\mid X\in A)\ge 1-\alpha$. Measured by restricting the held-out set to $A$; standard error $\approx\sqrt{\alpha(1-\alpha)/(m\delta)}$ on $m$ test points.

**Relaxation B — $\mathcal{F}$-conditional** (Gibbs et al.). For a linear class $\mathcal{F}$ of functions $f:\mathcal{X}\to\mathbb{R}$,
$$\mathbb{E}\big[f(X)\big(\mathbb{1}\{Y\in\hat C(X)\}-(1-\alpha)\big)\big]=0\qquad \forall f\in\mathcal{F}.$$
$\mathcal{F}=\{1\}$ recovers marginal coverage; $\mathcal{F}=$ all measurable $f$ recovers full conditional coverage. Measured as the empirical moment on held-out data, per basis function.

**Reported metric — worst-slab coverage** (WSC; Cauchois, Gupta & Duchi 2021):
$$\mathrm{WSC}_\delta=\inf_{v\in S^{d-1},\,a<b,\ P(a\le v^\top X\le b)\ge\delta} P\big(Y\in\hat C(X)\ \big|\ a\le v^\top X\le b\big),$$
estimated by maximizing over slabs on one split and evaluating on another. Without the split it is biased downward by the maximization; with the split it is a valid but conservative estimate.

**Assumptions and their violations.** (i) Exchangeability of calibration and test data — violated under any temporal or covariate shift, the common deployment case (Tibshirani et al. 2019; Barber et al. 2023). (ii) $P_X$ non-atomic — this is what *powers* the impossibility; with a finite $\mathcal{X}$ and enough data per atom, conditional coverage is attainable, so the theorem is a statement about continuous features. (iii) The conditioning class is fixed before seeing data — routinely violated when analysts pick the group that looks worst.

## 3. State of the Art

**Theory SOTA (established).** Barber, Candès, Ramdas & Tibshirani, *The limits of distribution-free conditional predictive inference*, Information and Inference 10(2), 2021: if $\hat C$ has distribution-free conditional coverage at level $1-\alpha$, then for any $P$ with non-atomic $P_X$, $\mathbb{E}[\mathrm{Leb}(\hat C(x))]=\infty$ at $P_X$-a.e. $x$. Vovk (ACML 2012) proved the equivalent "object-conditional validity is trivial" statement for conformal predictors; Lei & Wasserman (JRSS-B 2014, Lemma 1) proved the finite-length version. Same paper gives the positive side: coverage conditional on all sets with $P(A)\ge\delta$ is attainable, with sample complexity and width degrading as $\delta\downarrow 0$.

**Method SOTA (established).** Gibbs, Cherian & Candès (arXiv:2305.12616, 2023) give finite-sample exact $\mathcal{F}$-conditional coverage for any finite-dimensional linear $\mathcal{F}$, via a quantile-regression dual — the cleanest positive result since Mondrian conformal prediction. Jung, Noarov, Ramalingam & Roth (ICLR 2023) and Bastani et al. (NeurIPS 2022) give multivalid conformal prediction: simultaneous approximate coverage over an arbitrary (possibly overlapping) collection of groups, with error scaling in $\sqrt{\log|\mathcal{G}|/n_g}$ per group, and in the adversarial/online setting with no distributional assumption at all.

**Empirical SOTA (claimed, partly unablated).** CQR (Romano, Patterson & Candès, NeurIPS 2019) and APS/RAPS (Romano, Sesia & Candès, NeurIPS 2020) are the default adaptive scores. Their conditional-coverage advantage over split conformal is reported as WSC or group-coverage numbers on 9–11 UCI/MEPS regression sets and CIFAR/ImageNet — **benchmark numbers, not ablations**. The confound: better scores also produce shorter intervals, and no standard protocol matches length before comparing conditional coverage, so "adaptivity" and "sharpness" are not separated in the published tables.

## 4. What Is Known

- **The impossibility is exact, not asymptotic.** Infinite expected Lebesgue measure at a.e. $x$, at every $n$, for every non-atomic $P_X$ (Barber et al. 2021).
- **Atoms rescue it.** With discrete $X$ taking $k$ values, Mondrian/group conformal gives exact conditional coverage at $\approx \lceil (1-\alpha)(n_k+1)\rceil$ calibration points per cell; at $\alpha=0.1$ that needs $n_k\ge 9$ per cell for the guarantee to be non-vacuous, and $n_k\approx 1000$ for the realized coverage to sit within $\pm 1$ point.
- **Marginal coverage is tight and reproduced everywhere.** Split conformal at $\alpha=0.1$ returns $90\pm 1\%$ empirical coverage across all standard regression and classification benchmarks at $n_{\text{cal}}\ge 1000$; this replicates without exception.
- **Conditional gaps are large at benchmark scale.** On MEPS-19/20/21 (health-expenditure regression, $n\approx 15{,}000$–$35{,}000$), split conformal at a 90% marginal target shows group-coverage differences of several points between race-defined subgroups, and CQR narrows but does not close them (Romano, Barber, Sabatti & Candès, HDSR 2020). WSC at $\delta=0.2$ on standard regression benchmarks is reported 10–20 points below the marginal target for constant-width bands, and roughly halved by CQR *(numbers as reported; the length confound in §3 is not controlled)*.
- **Calibration inherits the same wall.** Gupta, Podkopaev & Ramdas (NeurIPS 2020) show distribution-free conditional calibration of a continuous-output probability predictor is likewise impossible; approximate conditional coverage and threshold calibration are two faces of one obstruction.

## 5. What Is Not Known

- **Theoretically open.** The exchange rate. Given a conditioning class $\mathcal{F}$ of dimension $p$ (or a group collection $\mathcal{G}$), what is the minimax excess expected width relative to the oracle band, as a function of $(p, n, \alpha, \delta)$? Gibbs et al. give achievability; a matching lower bound over rich classes is missing. Also open: whether the $1/\delta$ dependence in set-conditional coverage is tight for adaptive (data-dependent) $\mathcal{A}$.
- **Empirically open.** Whether any adaptive method beats split conformal on conditional coverage *at matched interval length*. The experiment is a few thousand GPU-hours at most; nobody has run it as a controlled comparison.
- **Methodologically blocked.** There is no accepted estimator of the conditional-coverage gap with a calibrated null. WSC, group coverage, binned coverage and $\mathcal{F}$-moment violation disagree in ranking, and none reports what value a *perfectly conditionally valid* oracle would score at finite $n$ — so a reported "12-point WSC gap" cannot be separated from estimator bias.

## 6. Why It Is Hard

**Absent ground truth at the point level, compounded by a maximization bias.** The target $P(Y\in\hat C(X)\mid X=x)$ is a function of $x$, and each $x$ in the test set carries exactly one Bernoulli draw. Estimating it requires pooling neighbours — which reintroduces the very averaging the definition forbids. Every practical metric therefore maximizes a deviation over a family of pooled regions, and the maximum of $|\mathcal{A}|$ noisy estimates is upward-biased by $\Theta(\sqrt{\log|\mathcal{A}|/(m\delta)})$. At $m=2000$, $\delta=0.2$, $|\mathcal{A}|$ the slab family, that bias is several coverage points — the same order as the effects being reported. The evaluation does not cleanly measure the thing it names.

The second obstruction is non-identifiability of the cause: a large WSC gap is consistent with (a) a genuinely non-adaptive band, (b) a well-adapted band on a heteroskedastic problem where the oracle itself has finite-sample slack, or (c) estimator bias. Without an oracle arm, the three are not separable.

## 7. Current Research (as of 2026)

- **Conditional-guarantee conformal.** Candès' group (Stanford) — $\mathcal{F}$-conditional calibration, randomized variants achieving exact rather than conservative coverage, and extensions to covariate shift as a special case of $\mathcal{F}$.
- **Multicalibration/multivalidity.** Roth and collaborators (Penn) — batch and adversarial multivalid conformal prediction, connecting conditional coverage to the multicalibration literature; open question is group-collection richness versus per-group sample size.
- **Localized and weighted conformal.** Guan (Biometrika 2023) and successors — kernel-localized scores, which buy approximate conditional coverage at a bandwidth-dependent, non-distribution-free price.
- **Beyond exchangeability.** Barber, Candès, Ramdas & Tibshirani (Annals of Statistics 2023) — coverage bounds degrading in a total-variation drift term; conditional versions of this bound are being pursued *(frontier — verify)*.
- **LLM applications.** Conformal set prediction for generation and abstention, where the conditional gap is most acute (coverage on easy prompts near 1, on hard prompts far below target) and the least well measured *(frontier — verify)*.

## 8. Concrete Next Experiment

**The question:** does any adaptive conformal method reduce the conditional-coverage gap at matched interval length, beyond what estimator bias explains?

**Scale.** 11 standard regression datasets (9 UCI + MEPS-19/21), $n_{\text{cal}}=2000$, $n_{\text{test}}=4000$, $\alpha=0.1$, 50 random splits each. Add one classification arm: ImageNet-1k, ResNet-50 logits, $n_{\text{cal}}=10{,}000$.

**Arms.** (1) Split conformal, absolute residual. (2) CQR. (3) Gibbs et al. $\mathcal{F}$-conditional with $\mathcal{F}$ = 100 random Fourier features. (4) Batch multivalid conformal, 20 quantile-defined groups. (5) **Control arm — the oracle**: on two synthetic datasets with known $P(Y\mid X)$, the exact conditional band, which has *zero* true gap. Every arm is width-matched by inflating the shorter arms' sets to the largest arm's median length before evaluation.

**The deciding number.** Held-out WSC gap at $\delta=0.2$, **minus the oracle arm's measured WSC gap at the same $n_{\text{test}}$ and $\delta$** (the estimator's null). A method counts as a genuine advance if this bias-corrected gap is $\ge 5$ points below split conformal's, at matched median length, with the 95% bootstrap interval over 50 splits excluding zero. Prediction: the oracle null alone is 4–8 points, and at matched length arms (2)–(4) beat (1) by under 3 points — i.e. most published adaptivity gains are length gains in disguise.

## 9. Key References

- **[Foundational]** Vovk. *Conditional validity of inductive conformal predictors.* ACML (PMLR 25), 2012. — arXiv:1209.2673
- **[Foundational]** Lei & Wasserman. *Distribution-free prediction bands for non-parametric regression.* JRSS-B 76(1), 2014.
- **[Foundational]** Barber, Candès, Ramdas & Tibshirani. *The limits of distribution-free conditional predictive inference.* Information and Inference 10(2), 2021. — arXiv:1903.04684
- **[Foundational]** Vovk, Gammerman & Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[SOTA]** Gibbs, Cherian & Candès. *Conformal prediction with conditional guarantees.* 2023. — arXiv:2305.12616
- **[SOTA]** Jung, Noarov, Ramalingam & Roth. *Batch multivalid conformal prediction.* ICLR, 2023. — arXiv:2209.15145
- **[SOTA]** Bastani, Gupta, Jung, Noarov, Ramalingam & Roth. *Practical adversarial multivalid conformal prediction.* NeurIPS, 2022. — arXiv:2206.01067
- **[Method]** Romano, Patterson & Candès. *Conformalized quantile regression.* NeurIPS, 2019. — arXiv:1905.03222
- **[Method]** Romano, Sesia & Candès. *Classification with valid and adaptive coverage.* NeurIPS, 2020. — arXiv:2006.02544
- **[Measurement]** Cauchois, Gupta & Duchi. *Knowing what you know: valid and validated confidence sets in multiclass and multilabel prediction.* JMLR 22, 2021. — arXiv:2004.10181
- **[Measurement]** Romano, Barber, Sabatti & Candès. *With malice toward none: assessing uncertainty via equalized coverage.* Harvard Data Science Review, 2020.
- **[Related]** Gupta, Podkopaev & Ramdas. *Distribution-free binary classification: prediction sets, confidence intervals and calibration.* NeurIPS, 2020. — arXiv:2006.10564
- **[Related]** Tibshirani, Foygel Barber, Candès & Ramdas. *Conformal prediction under covariate shift.* NeurIPS, 2019. — arXiv:1904.06019
- **[Related]** Guan. *Localized conformal prediction: a generalized inference framework for conformal prediction.* Biometrika 110(1), 2023.
- **[Survey]** Angelopoulos & Bates. *A gentle introduction to conformal prediction and distribution-free uncertainty quantification.* 2021. — arXiv:2107.07511

## 10. Worked Example

Take $X\sim \mathrm{Unif}[0,1]$, $Y\mid X=x \sim \mathcal{N}(0, x^2)$. The fitted mean is exactly right, $\hat\mu\equiv 0$; only the *spread* varies with $x$. Split conformal with score $S=|Y|$ picks the 90th percentile of $X|Z|$, $Z\sim\mathcal{N}(0,1)$:
$$P(X|Z|\le t)=\int_0^1\big(2\Phi(t/x)-1\big)\,dx = 0.90 \ \Rightarrow\ t\approx 0.93 .$$

The band is the constant interval $[-0.93, 0.93]$. Its conditional coverage is $2\Phi(0.93/x)-1$:

| $x$ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| conditional coverage | 1.000 | 0.998 | 0.938 | 0.815 | 0.699 | 0.648 |

Marginal coverage is exactly 90%. Conditional coverage runs from 100% to **64.8%**. The oracle band $\pm 1.645x$ has 90% coverage at every $x$ and *shorter* mean length ($1.645$ vs $1.86$) — so here nothing is bought by the constant band at all.

Now the obstruction. Suppose you deploy the split-conformal band and want to detect the failure from data alone, with $m=2000$ test points and no knowledge of the generative model. Bin $x$ into width-0.05 bins: $\approx100$ points per bin, standard error $\sqrt{0.9\cdot 0.1/100}=3.0$ points. The worst bin's gap here is 25 points, so this case is detectable. But scale the heteroskedasticity down — $\sigma(x)=1+0.15x$ instead — and the worst-bin gap falls to about 4 points, below the 3-point noise floor, while the *maximum over 20 bins* is upward-biased by roughly $2.5\sigma\approx 7$ points under the null. The estimator now reports a 7-point gap for a band that is essentially fine, and a 4-point gap for one that is not. Same reported number, opposite truths. That is the methodological block in §5 made concrete: the impossibility theorem tells you the gap exists, and the available instruments cannot tell you how big it is at the sizes that matter.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*