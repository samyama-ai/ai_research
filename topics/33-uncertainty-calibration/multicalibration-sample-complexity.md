---
id: 33-uncertainty-calibration/multicalibration-sample-complexity
title: "Multicalibration Sample Complexity for Rich Group Families"
topic: 33-uncertainty-calibration
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multicalibration Sample Complexity for Rich Group Families

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/multicalibration-sample-complexity` · **Status:** partially-solved

## 1. Problem Statement

Multicalibration asks a predictor to be calibrated not just marginally but simultaneously on every subpopulation in a family $\mathcal{C}$ of group indicators. The question here is **how many labelled samples that costs** as $\mathcal{C}$ grows rich — overlapping, intersectional, or defined by a learned model class rather than a handful of protected attributes.

Three variants, with different difficulty:

- **Theory variant.** Give matching upper and lower bounds on $n(\alpha, \lambda, \mathcal{C}, \delta)$: the number of i.i.d. samples needed to output an $(\alpha,\lambda)$-multicalibrated predictor with probability $1-\delta$. Upper bounds are known; they are not tight, and the tight characterization is tied to an open problem in convex geometry.
- **Measurement variant.** Given a finite sample, *estimate* the multicalibration error of a fixed predictor. This is where the catalog entry earns its status: for rich $\mathcal{C}$ the estimator's own variance can exceed the quantity being estimated, so a reported "multicalibration error" number is often not a measurement of multicalibration error.
- **Method variant.** Do post-processing algorithms (HKRR-style boosting) beat plain ERM on realistic data at realistic $n$? Recent evidence says usually no, which reframes the sample-complexity question as *when is the extra data spend justified at all*.

A solution: matching bounds in the theory variant, and a bias-corrected estimator with a proven noise floor in the measurement variant.

## 2. Formal Setting

Domain $\mathcal{X}$, binary label $y\in\{0,1\}$, distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$, ground truth $p^*(x)=\Pr[y=1\mid x]$. Predictor $f:\mathcal{X}\to[0,1]$. Group family $\mathcal{C}\subseteq\{0,1\}^{\mathcal{X}}$ (or $[0,1]^\mathcal{X}$ for weighted groups).

**Multicalibration error.** Discretize $[0,1]$ into $\lceil 1/\lambda\rceil$ bins $I_v=[v, v+\lambda)$. For $c\in\mathcal{C}$ and bin $v$, the cell mass and the cell bias are
$$\gamma_{c,v}=\Pr_{\mathcal{D}}[c(x)=1,\ f(x)\in I_v],\qquad \Delta_{c,v}=\mathbb{E}_{\mathcal{D}}\big[(y-f(x))\,\big|\,c(x)=1, f(x)\in I_v\big].$$
$f$ is $(\alpha,\lambda)$-multicalibrated if $\gamma_{c,v}\,|\Delta_{c,v}|\le\alpha$ for all $c,v$ (the mass-weighted convention of Hébert-Johnson et al., 2018).

**Measured quantity.** On a held-out sample of size $n$, with $n_{c,v}$ points in cell $(c,v)$,
$$\widehat{\mathrm{MCE}}=\max_{c\in\mathcal{C}}\ \sum_v \frac{n_{c,v}}{n}\,\Big|\tfrac{1}{n_{c,v}}\textstyle\sum_{i\in(c,v)}(y_i-f(x_i))\Big|.$$
This is the number a paper reports. It is **upward-biased**: each $|\cdot|$ has expectation at least $\sqrt{2/\pi}\cdot\sigma_{c,v}$ under a zero-bias null, with $\sigma_{c,v}\approx\sqrt{p(1-p)/n_{c,v}}$, and the outer $\max$ over $|\mathcal{C}|$ groups adds a further $\Theta(\sigma\sqrt{\log|\mathcal{C}|})$.

**Multiaccuracy** is the $\lambda=1$ (single-bin) relaxation: $|\mathbb{E}[c(x)(y-f(x))]|\le\alpha$. **Degree-$t$ multicalibration** (Gopalan, Kim, Singhal, Zhao, COLT 2022) interpolates by asking for correctness of the first $t$ moments of $f$ against $\mathcal{C}$.

Complexity of $\mathcal{C}$ is measured by VC dimension $d$ for binary groups, or by the graph dimension / fat-shattering dimension $\mathrm{fat}_\epsilon(\mathcal{C})$ for real-valued groups.

**Assumptions and their violations.**
- *i.i.d. sampling from the deployment distribution* — violated whenever multicalibration is invoked for its main selling point, transfer to a shifted target (Kim et al., PNAS 2022).
- *Labels are unbiased draws from $p^*$* — violated under label noise correlated with group membership, which is exactly the intersectional case.
- *$\mathcal{C}$ fixed before seeing data* — violated when groups are mined from the same data (adaptive-analysis inflation of $\widehat{\mathrm{MCE}}$).
- *Every cell has non-trivial mass $\gamma$* — violated by construction for intersectional families; the tail of tiny cells drives both bounds and noise.

## 3. State of the Art

**Theory SOTA (established).**
- Hébert-Johnson, Kim, Reingold, Rothblum (ICML 2018) — the defining paper. Their algorithm converges in $O\!\left(1/(\alpha^2\gamma)\right)$ rounds by a squared-error potential argument, each round one weak-agnostic-learning call over $\mathcal{C}$.
- Shabat, Cohen, Mansour (NeurIPS 2020) — the reference sample-complexity result: uniform convergence of multicalibration error over $\mathcal{C}$ with sample complexity polynomial in the graph/fat-shattering dimension $d$ of $\mathcal{C}$ and roughly $\tilde{O}(d/(\alpha^2\lambda^2))$ in the accuracy parameters, plus lower bounds that do not match the upper bounds in the $\lambda$ dependence.
- Hu, Peale, Reingold (ALT 2023) — sample complexity of outcome indistinguishability (the generalization of multicalibration, Dwork et al., STOC 2021) is characterized by a metric-entropy duality; tightness reduces to Pisier's duality conjecture in convex geometry, which is open.
- Gopalan, Kim, Singhal, Zhao (COLT 2022) — low-degree multicalibration achieves much of the downstream benefit (omniprediction, fairness) at multiaccuracy-like sample cost.
- Haghtalab, Jordan, Zhao (NeurIPS 2023) — recasts multicalibration algorithms as no-regret game dynamics, unifying the rate analyses.

**Empirical SOTA (established, and partly negative).**
- Hansen, Devic, Nakkiran, Sharan (NeurIPS 2024), *When is multicalibration post-processing necessary?* — the largest systematic study: across tabular, image, and language datasets, ERM-trained models are typically already multicalibrated to within evaluation noise, and post-processing gains are frequently within error bars.
- Detommaso, Bertran, Fogliato, Roth (ICML 2024) — multicalibration for LLM confidence scoring, reporting large ECE reductions on QA benchmarks.

**Claimed but unablated.** That multicalibration post-processing improves *downstream decision quality* (not just calibration metrics) is asserted far more often than it is ablated against a temperature-scaling or Platt-scaling control at matched held-out data spend. The LLM-confidence results exist mainly as benchmark numbers on TriviaQA/MMLU-style sets and have not been separated from the effect of simply spending the same calibration split on a global recalibrator.

## 4. What Is Known

- **Round complexity.** HKRR converges in at most $\lceil 1/(4\alpha^2\gamma)\rceil$ update rounds: each successful update reduces $\mathbb{E}[(p^*-f)^2]\le 1/4$ by at least $\alpha^2\gamma$. Established, scale-free.
- **Sample complexity is polynomial in group-class dimension**, not in $|\mathcal{C}|$ — Shabat et al. (NeurIPS 2020). This is what makes rich $\mathcal{C}$ formally tractable.
- **Multiaccuracy is cheap, multicalibration is not.** The $1/\lambda$ bins multiply the effective hypothesis count; empirically, moving from 1 bin to 10 bins on the same data raises the smallest-cell count requirement by roughly an order of magnitude.
- **Empirical numbers, at the scale measured.** Hansen et al. (NeurIPS 2024) evaluate on ACS/folktables tabular tasks ($n\approx 1.5$–$2\times10^5$ per state, ACSIncome California $\approx 1.96\times10^5$ rows), plus vision and language sets, and find ERM baselines' multicalibration error largely indistinguishable from post-processed models once evaluation noise is accounted for.
- **Calibration-error estimation is itself biased.** Błasiok, Gopalan, Hu, Nakkiran (STOC 2023) show binned ECE is not a consistent distance-to-calibration; the same pathology applies per cell in $\widehat{\mathrm{MCE}}$, only worse because cells are smaller.

## 5. What Is Not Known

- **Theoretically open.** The tight sample complexity for multicalibration over a class of fat-shattering dimension $d$. Upper and lower bounds differ by polynomial factors in $1/\lambda$ and $1/\alpha$. The general outcome-indistinguishability version is equivalent to a metric-entropy duality question (Hu, Peale, Reingold, ALT 2023) that is open in convex geometry — so this is not merely unproven, it is blocked on a hard external conjecture.
- **Theoretically open.** Whether $(\alpha,\lambda)$-multicalibration requires $\Omega(1/\lambda^2)$ samples or whether $1/\lambda$ suffices with a smarter, non-uniform-convergence algorithm.
- **Empirically open.** At what $n$, and for which $\mathcal{C}$, post-processing beats ERM by more than noise. The Hansen et al. sweep is at $n\sim10^5$ tabular; nobody has run the equivalent at $n\sim10^7$ with $|\mathcal{C}|\sim10^4$ intersectional groups.
- **Methodologically blocked.** There is no accepted, bias-corrected estimator of multicalibration error with a stated noise floor. Papers report $\widehat{\mathrm{MCE}}$ with no correction for the $\max$ over groups or the small-cell bias, so cross-paper comparison of "multicalibration error" numbers is not meaningful.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement in the small-cell regime**. Multicalibration is a maximum over $|\mathcal{C}|/\lambda$ cells; making $\mathcal{C}$ rich makes cells small; small cells make each per-cell bias estimate noisy; and a maximum over many noisy estimates is dominated by noise, not by signal. Formally, the null-hypothesis expectation of $\widehat{\mathrm{MCE}}$ under a perfectly multicalibrated $f$ scales as $\Theta\!\big(\sqrt{\log(|\mathcal{C}|/\lambda)}\cdot\mathbb{E}_c[\sqrt{\gamma_{c,v}/n}]\big)$, which for realistic intersectional families at $n\sim10^5$ exceeds the $\alpha\approx0.01$–$0.05$ targets papers aim at.

The consequence: **the evaluation does not measure the thing it names**. An improvement in reported $\widehat{\mathrm{MCE}}$ can come from smoothing predictions into fewer occupied bins (which shrinks the max) rather than from any gain in conditional correctness. This also blocks the theory question empirically — you cannot check whether a $1/\lambda$ or $1/\lambda^2$ bound is right if the estimator floor is above both.

## 7. Current Research (as of 2026)

- **Penn (Roth, Noarov, Jung, Bertran)** — multivalid conformal prediction and decision-theoretic multicalibration; the shift is toward calibrating *for a downstream decision loss* rather than for a group-conditional metric, which sidesteps the small-cell max. *(frontier — verify current membership)*
- **Stanford/Apple/Harvard (Gopalan, Nakkiran, Hu, Błasiok)** — consistent distance-to-calibration measures and low-degree relaxations, aimed exactly at the estimator problem in §6.
- **Berkeley/Cornell (Haghtalab, Zhao)** — game-dynamics view; open question is whether the no-regret framing yields better-than-uniform-convergence sample bounds.
- **USC/UCSD (Sharan, Devic, Hansen)** — continued empirical auditing of whether post-processing is necessary; extension to LLM confidence is *(frontier — verify)*.
- **LLM confidence calibration** — multicalibration over prompt-derived group features (topic, length, self-reported confidence) is an active applied thread; almost all results are benchmark numbers without matched-budget controls.

## 8. Concrete Next Experiment

**Question:** is reported multicalibration improvement above the estimator's noise floor?

**Scale.** ACSIncome (folktables) pooled across all US states, $n\approx1.66\times10^6$ rows. Group family $\mathcal{C}$ = all conjunctions of race (9) × sex (2) × age bucket (5) × education bucket (8) = 720 groups, $\lambda=0.1$ (10 bins) → 7,200 cells. Split 60/20/20 train/calibrate/test.

**Arms.**
1. ERM only (gradient-boosted trees), no post-processing.
2. HKRR multicalibration post-processing on the calibration split.
3. **Control arm:** temperature scaling on the same calibration split — same data spend, no group structure.
4. **Null arm:** ERM predictor evaluated against *label-permuted-within-cell* test data, giving the empirical distribution of $\widehat{\mathrm{MCE}}$ under a truly multicalibrated predictor.

**Deciding number.** $\widehat{\mathrm{MCE}}(\text{arm 2}) - \widehat{\mathrm{MCE}}(\text{arm 3})$, compared against the 95th percentile of the arm-4 null spread. If the gap does not exceed the null spread, multicalibration post-processing is not measurable at $n=1.66\times10^6$ with 720 groups, and the sample-complexity question is empirically vacuous below that scale. Sweep $n\in\{10^4,10^5,10^6\}$ to locate the crossover $n^*$ where the gap first clears the null — $n^*$ is the number this experiment produces.

## 9. Key References

- **[Foundational]** Úrsula Hébert-Johnson, Michael P. Kim, Omer Reingold, Guy N. Rothblum. *Multicalibration: Calibration for the (Computationally-Identifiable) Masses.* ICML 2018.
- **[SOTA — theory]** Eliran Shabat, Lee Cohen, Yishay Mansour. *Sample Complexity of Uniform Convergence for Multicalibration.* NeurIPS 2020.
- **[SOTA — theory]** Lunjia Hu, Charlotte Peale, Omer Reingold. *Metric Entropy Duality and the Sample Complexity of Outcome Indistinguishability.* ALT 2023.
- **[Theory]** Parikshit Gopalan, Michael P. Kim, Mihir Singhal, Shengjia Zhao. *Low-Degree Multicalibration.* COLT 2022.
- **[Theory]** Cynthia Dwork, Michael P. Kim, Omer Reingold, Guy N. Rothblum, Gal Yona. *Outcome Indistinguishability.* STOC 2021.
- **[Theory]** Parikshit Gopalan, Adam Tauman Kalai, Omer Reingold, Vatsal Sharan, Udi Wieder. *Omnipredictors.* ITCS 2022.
- **[Theory]** Nika Haghtalab, Michael I. Jordan, Eric Zhao. *A Unifying Perspective on Multi-Calibration: Game Dynamics.* NeurIPS 2023.
- **[Measurement]** Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC 2023.
- **[SOTA — empirical]** Dutch Hansen, Siddartha Devic, Preetum Nakkiran, Vatsal Sharan. *When is Multicalibration Post-Processing Necessary?* NeurIPS 2024.
- **[Applied]** Gianluca Detommaso, Martin Bertran, Riccardo Fogliato, Aaron Roth. *Multicalibration for Confidence Scoring in LLMs.* ICML 2024.
- **[Applied]** Michael P. Kim, Christoph Kern, Shafi Goldwasser, Frauke Kreuter, Omer Reingold. *Universal Adaptability: Target-Independent Inference that Competes with Propensity Scoring.* PNAS, 2022.
- **[Applied]** Christopher Jung, Georgy Noarov, Ramya Ramalingam, Aaron Roth. *Batch Multivalid Conformal Prediction.* ICLR 2023.

## 10. Worked Example

Take ACSIncome California, $n=195{,}665$ rows, 20% held out for evaluation → $n_{\text{test}}=39{,}133$. Groups: 720 conjunctions as above; 10 prediction bins → 7,200 cells.

Cell occupancy is extremely skewed. Mean cell size is $39{,}133/7{,}200\approx5.4$, and even restricting to cells with $\ge30$ points leaves only a few hundred cells covering most mass. Take a mid-sized cell with $n_{c,v}=50$ and $p\approx0.5$. The per-cell standard error is
$$\sigma_{c,v}=\sqrt{p(1-p)/n_{c,v}}=\sqrt{0.25/50}=0.0707.$$
Under a *perfectly* multicalibrated predictor, $\mathbb{E}|\hat\Delta_{c,v}| = \sqrt{2/\pi}\,\sigma \approx 0.056$. Taking a maximum over even 200 such cells adds roughly $\sigma\sqrt{2\ln 200}\approx 0.0707\times3.26\approx0.23$ in the unweighted convention.

So the reported unweighted worst-group calibration gap for a *flawless* predictor is $\approx 0.23$ — versus a target $\alpha$ of $0.01$–$0.05$. The mass weighting $\gamma_{c,v}$ shrinks this (a 50-point cell has $\gamma\approx0.0013$), which is why the weighted convention is used, but the shrinkage is doing the work: the weighted score is dominated by a handful of large cells and is nearly blind to exactly the small intersectional groups multicalibration was introduced to protect.

The obstruction, made concrete: at $n_{\text{test}}=39{,}133$ with 720 groups, **the unweighted estimator's noise floor is 5–20× the target, and the weighted estimator suppresses the groups of interest**. Either way, a paper reporting "multicalibration error 0.03" on this setup has reported a property of its binning and weighting choices, not a measurement of multicalibration. Fixing this needs the null arm of §8 run as standard practice, not a tighter theorem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*