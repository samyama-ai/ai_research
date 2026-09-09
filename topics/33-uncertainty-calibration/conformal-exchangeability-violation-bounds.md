---
id: 33-uncertainty-calibration/conformal-exchangeability-violation-bounds
title: "Exchangeability Violation Bounds for Conformal Prediction"
topic: 33-uncertainty-calibration
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Exchangeability Violation Bounds for Conformal Prediction

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/conformal-exchangeability-violation-bounds` · **Status:** partially-solved

## 1. Problem Statement

Split and full conformal prediction give $\mathbb{P}(Y_{n+1}\in\hat C_n(X_{n+1}))\ge 1-\alpha$ using only one assumption: the sequence $Z_1,\dots,Z_{n+1}$ is exchangeable. Real deployments break it — temporal drift, active data collection, label shift, retraining on one's own predictions. The problem is to say what the guarantee degrades to.

Three variants, with different difficulty:

- **Theory variant.** Find a coverage-gap bound $\mathbb{P}(Y_{n+1}\in\hat C_n)\ge 1-\alpha-\Delta$ where $\Delta$ is a functional of the joint law, and prove it tight (matching lower bound over a nontrivial class).
- **Measurement variant.** Estimate $\Delta$ from the observed data. This is the binding constraint: the known bounds are stated in quantities that are *not identifiable from one realized trajectory*.
- **Method variant.** Design a procedure whose realized gap is small under a stated shift family, and show the improvement is not just the trivial one of widening intervals.

Solving it means: a computable, non-vacuous certificate $\hat\Delta$ with $\mathbb{P}(\text{gap}\le\hat\Delta)\ge 1-\delta$, at a $\hat\Delta$ small enough to act on ($\hat\Delta \ll \alpha$).

## 2. Formal Setting

Data $Z_i=(X_i,Y_i)\in\mathcal{X}\times\mathcal{Y}$, $i=1,\dots,n+1$, with joint law $P$ on $(\mathcal{X}\times\mathcal{Y})^{n+1}$; $Z_{n+1}$ is the test point. A score function $s:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}$ (measured as, e.g., $s=|y-\hat\mu(x)|$ from a model fit on a disjoint split) gives $R_i=s(Z_i)$.

**Weighted (nonexchangeable) conformal.** Fix weights $w_i\in[0,1]$, $\tilde w_i=w_i/(1+\sum_j w_j)$, $\tilde w_{n+1}=1/(1+\sum_j w_j)$. Output
$$\hat C_n(x)=\Big\{y: s(x,y)\le \mathrm{Quantile}_{1-\alpha}\big(\textstyle\sum_i \tilde w_i\delta_{R_i}+\tilde w_{n+1}\delta_{+\infty}\big)\Big\}.$$

**The violation functional.** Let $\sigma_i$ be the transposition swapping $i$ and $n+1$, and $Z^{\sigma_i}$ the permuted sequence. Define
$$\Delta \;=\; \sum_{i=1}^{n}\tilde w_i\, d_{\mathrm{TV}}\!\left(Z,\,Z^{\sigma_i}\right),$$
measured, in principle, as a total-variation distance between two laws on $(\mathcal X\times\mathcal Y)^{n+1}$. Under exchangeability every term is $0$.

**Covariate-shift special case.** $P_{\text{test}}(x,y)=w(x)P_{\text{train}}(x,y)$ with $w=\mathrm{d}P^X_{\text{test}}/\mathrm{d}P^X_{\text{train}}$; measured in practice as $\hat w$ from a train-vs-test discriminator, so the residual quantity is $\mathbb{E}|\hat w-w|$.

**Online case.** Coverage measured as the realized frequency $\frac1T\sum_{t\le T}\mathbb{1}\{Y_t\notin\hat C_t\}$ over a single trajectory, and as *local* coverage on every window $I\subseteq[T]$.

**Assumptions known to be violated in practice.** (i) Exchangeability — violated by construction here. (ii) Score-function independence from the calibration set — violated whenever the model is retrained or the score is tuned. (iii) $w$ known — violated; $\hat w$ is estimated. (iv) $\Delta$ known to the user — violated always; see §6. (v) The test point's marginal is the one weights were built for — violated under feedback loops where $P_{\text{test}}$ depends on $\hat C$.

## 3. State of the Art

**Theory SOTA (established).**
- *Nonexchangeable conformal*, Barber, Candès, Ramdas, Tibshirani (Annals of Statistics, 2023): coverage $\ge 1-\alpha-\Delta$ with $\Delta$ as above, for any weights fixed in advance, any score, no distributional assumption. This is the general-purpose result and holds for both full and split variants.
- *Weighted conformal under covariate shift*, Tibshirani, Barber, Candès, Ramdas (NeurIPS 2019): exact $1-\alpha$ coverage when the likelihood ratio $w$ is known and shift is covariate-only.
- *Split conformal under $\beta$-mixing/drift*, Oliveira, Orenstein, Ramos, Romano (JMLR, 2024): expected-coverage gap bounded by mixing coefficients plus a drift term, for stationary-ish dependent data.
- *Online adaptive conformal inference (ACI)*, Gibbs & Candès (NeurIPS 2021): deterministic long-run coverage regardless of any distribution shift, adversarial included.
- *Impossibility*, Vovk (2012); Lei & Wasserman (JRSS-B, 2014); Barber, Candès, Ramdas, Tibshirani (Information and Inference, 2021): distribution-free $X$-conditional coverage forces infinite-length intervals on non-atomic $P^X$.

**Claimed but unablated.** That the exponential weights $w_i=\gamma^{\,n+1-i}$ used throughout the applied literature are near-optimal — no paper minimizes the bound over the weight class and shows the minimizer beats $\gamma$-decay. That the TV bound is tight for realistic shifts — the published tightness statements are worst-case over adversarial $P$, not for AR/changepoint families.

**Benchmark-only results.** The comparative rankings of ACI / DtACI / AgACI / SAOCP / Conformal PID on ELEC2, stock volatility, and weather series are benchmark numbers on a handful of series, not ablations isolating which mechanism (learning rate, expert aggregation, scorecaster) causes the gain.

## 4. What Is Known

- **The gap bound is not vacuous but is loose.** In Barber et al. (2023), the bound holds at any $n$; their own simulations on AR(1)-type and changepoint data show realized gaps several times smaller than $\Delta$. Verified in §10 at $n=200$: bound $0.259$ vs true gap $0.074$ — a factor of $3.5$.
- **ACI's coverage certificate is exact and finite-sample.** Gibbs & Candès (2021), Prop. 4.1: for step size $\gamma>0$,
  $$\Big|\tfrac1T\textstyle\sum_{t=1}^{T}\mathrm{err}_t-\alpha\Big|\;\le\;\frac{\max\{\alpha_1,\,1-\alpha_1\}+\gamma}{T\gamma}.$$
  At $\gamma=0.005$, $T=10{,}000$, $\alpha_1=0.1$: gap $\le 0.018$. This is *long-run marginal only* — it says nothing about any sub-window.
- **Local coverage needs a different algorithm.** SAOCP (Bhatnagar, Wang, Xiong, Bai, ICML 2023) attains strongly adaptive regret $\tilde O(\sqrt{|I|})$ on every interval $I$; ACI provably does not control interval-wise coverage.
- **Estimated weights cost first-order error.** Under covariate shift with $\hat w\ne w$, coverage loss scales with $\mathbb{E}|\hat w - w|$ (Lei & Candès, JRSS-B 2021, for counterfactual/ITE sets); doubly robust constructions (Yang, Kuchibhotla, Tchetgen Tchetgen, JRSS-B 2024) reduce this to a product of two estimation errors.
- **Weighting trades bias for variance in a measurable way.** Effective sample size $(\sum_i w_i)^2/\sum_i w_i^2$; at $\gamma=0.95$, $n=200$, ESS $=39$, so the $1-\alpha$ quantile carries roughly $\sqrt{\alpha(1-\alpha)/39}\approx 4.8$ percentage points of Monte Carlo noise in coverage.

## 5. What Is Not Known

- **Methodologically blocked (the central gap).** $\Delta$ is defined as a TV distance between the joint law and its permutation. From one trajectory of length $n$, there is exactly one draw from each law; $d_{\mathrm{TV}}$ between two distributions on $\mathbb{R}^{n+1}$ is not estimable from a single sample without structural assumptions. No published method returns a data-driven upper confidence bound on $\Delta$. Until it exists, the theorem is a *conditional* guarantee whose condition nobody can check.
- **Theoretically open.** Tightness. Is there a matching lower bound showing the $\sum_i\tilde w_i d_{\mathrm{TV}}$ form cannot be improved for structured (Markov, changepoint, bounded-drift) families, or does an $f$-divergence or Wasserstein functional give a strictly smaller valid $\Delta$? Also open: optimal weights, i.e. $\arg\min_w \{\Delta(w) + \text{width penalty}(w)\}$, even for AR(1).
- **Theoretically open.** Guarantees under *feedback* — when $P$ at time $t+1$ depends on $\hat C_t$. Fannjiang et al. (PNAS 2022) handle a designed feedback covariate shift; general closed-loop deployment has no bound.
- **Empirically open.** How large is the bound/realized-gap ratio across a wide corpus of real non-exchangeable series? Runnable today on 1,000+ series at negligible compute; nobody has published the distribution of that ratio.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the violation functional from the observed data**. Exchangeability is a symmetry of a joint law on $(n+1)$ points; a deployment gives one draw. Testing it is possible (permutation/martingale tests: Vovk et al.) but *quantifying* it to the precision $\alpha$ requires resolving TV distance to within, say, $0.02$ between two $(n{+}1)$-dimensional laws from one sample. That is information-theoretically impossible without a structural model — and once you assume the structural model (AR, changepoint, bounded drift), the whole appeal of distribution-free inference is gone.

A second obstruction is **an evaluation that does not measure what it names**: papers report marginal coverage averaged over a test trajectory, but the object at risk under non-exchangeability is *conditional* coverage in the current regime. A method can hit $0.90$ marginally while covering $0.99$ in the calm half and $0.81$ in the shifted half — the failure the user cares about, invisible in the reported number.

## 7. Current Research (as of 2026)

- **Online conformal as online convex optimization.** Gibbs & Candès (DtACI, JMLR 2024), Angelopoulos–Candès–Tibshirani (Conformal PID, NeurIPS 2023), Zaffran et al. (AgACI, ICML 2022), Bhatnagar et al. (SAOCP, ICML 2023). Direction: replace assumption-based bounds with regret-based ones, so validity holds adversarially. Groups: Stanford (Candès), Berkeley (Jordan/Angelopoulos), CMU (Ramdas), INRIA/Sorbonne (Zaffran, Josse).
- **Structured $\Delta$ certificates.** Bounding $\Delta$ via mixing coefficients or explicit changepoint models rather than leaving it abstract (Oliveira et al.; Chicago, Foygel Barber). *(frontier — verify)* whether a computable confidence bound on $\Delta$ under a $\beta$-mixing assumption has appeared.
- **Robust/distributionally-robust conformal.** Cauchois, Gupta, Ali, Duchi (JASA 2024): guarantee coverage for every $P'$ in an $f$-divergence ball of radius $\rho$ by inflating the quantile level; shifts $\Delta$-estimation into $\rho$-elicitation.
- **LLM-specific settings.** Conformal sets over token/answer spaces where retraining and RLHF break exchangeability between calibration and deployment. *(frontier — verify)*: no published bound handles calibration-then-finetune sequencing.

## 8. Concrete Next Experiment

**Question.** Is the Barber et al. (2023) bound tight enough to act on, or is it a factor of $\ge 10$ loose in the regimes people deploy in?

**Scale.** Simulate 3 shift families where $\Delta$ is computable in closed form: (a) Gaussian AR(1) scores, $\rho\in\{0.5,0.8,0.9,0.95,0.99\}$; (b) single mean-shift changepoint at fraction $\pi\in\{0.25,0.5,0.75\}$, jump $\mu\in\{0.25,0.5,1,2\}$; (c) linear drift. For each cell: $n\in\{200,1000,5000\}$, weights $w_i=\gamma^{n+1-i}$, $\gamma\in\{1,0.99,0.95,0.9\}$, $\alpha=0.1$, $10^4$ Monte Carlo trajectories. Total $\approx 10^7$ conformal fits — under one CPU-day.

**Control arm.** Unweighted split conformal ($\gamma=1$) on i.i.d. data from the same marginal, where $\Delta=0$ and the realized gap must be $\le 1/(n+1)$. This validates the harness: any measured gap above $1/(n+1)$ in the control means the estimator, not the theory, is at fault.

**Deciding number.** The *slack ratio* $\kappa=\Delta_{\text{bound}}/\text{gap}_{\text{realized}}$, reported as a median over trajectories per cell. Decision rule: if $\mathrm{median}(\kappa)\le 3$ across the majority of cells, the bound is a usable engineering certificate and effort should go to estimating $\Delta$; if $\mathrm{median}(\kappa)\ge 10$ in the practically relevant cells ($\rho\ge0.9$, $\gamma\in\{0.99,0.95\}$), the TV functional is the wrong currency and the open problem is finding a smaller valid divergence.

## 9. Key References

- **[Foundational]** Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer, 2005 (2nd ed. 2022).
- **[Foundational/SOTA]** Barber, Candès, Ramdas, Tibshirani. *Conformal prediction beyond exchangeability.* Annals of Statistics 51(2), 2023. — arXiv:2202.13415
- **[Foundational]** Tibshirani, Foygel Barber, Candès, Ramdas. *Conformal prediction under covariate shift.* NeurIPS, 2019. — arXiv:1904.06019
- **[SOTA]** Gibbs, Candès. *Adaptive conformal inference under distribution shift.* NeurIPS, 2021. — arXiv:2106.00170
- **[SOTA]** Gibbs, Candès. *Conformal inference for online prediction with arbitrary distribution shifts.* JMLR, 2024.
- **[SOTA]** Bhatnagar, Wang, Xiong, Bai. *Improved online conformal prediction via strongly adaptive online learning.* ICML, 2023.
- **[SOTA]** Angelopoulos, Candès, Tibshirani. *Conformal PID control for time series prediction.* NeurIPS, 2023.
- **[SOTA]** Oliveira, Orenstein, Ramos, Romano. *Split conformal prediction and non-exchangeable data.* JMLR, 2024.
- **[SOTA]** Cauchois, Gupta, Ali, Duchi. *Robust validation: confident predictions even when distributions shift.* JASA, 2024.
- **[Limits]** Barber, Candès, Ramdas, Tibshirani. *The limits of distribution-free conditional predictive inference.* Information and Inference, 2021.
- **[Applied]** Zaffran, Féron, Goude, Josse, Dieuleveut. *Adaptive conformal predictions for time series.* ICML, 2022.
- **[Applied]** Fannjiang, Bates, Angelopoulos, Listgarten, Jordan. *Conformal prediction under feedback covariate shift for biomolecular design.* PNAS 119(43), 2022.
- **[Survey]** Angelopoulos, Bates. *A gentle introduction to conformal prediction and distribution-free uncertainty quantification.* 2021. — arXiv:2107.07511
- **[Survey]** Angelopoulos, Barber, Bates. *Theoretical Foundations of Conformal Prediction.* Cambridge University Press, 2025.

## 10. Worked Example

**Setup.** Scores are drawn directly: $R_1,\dots,R_{100}\sim N(0,1)$, then $R_{101},\dots,R_{200},R_{201}\sim N(1,1)$ — a single mean shift halfway through, test point post-shift. $\alpha=0.1$, uniform weights $w_i=1$.

**The bound.** For $i>100$, the swap $\sigma_i$ moves one $N(1,1)$ point past another, so $d_{\mathrm{TV}}=0$. For $i\le 100$, using $q/p(t)=e^{t-1/2}$,
$$d_{\mathrm{TV}}\big(P_0\!\otimes\! P_1,\,P_1\!\otimes\! P_0\big)=\tfrac12 e^{-1/2}\,\mathbb{E}\big|e^{X}-e^{Y}\big| = 2\Phi(1/\sqrt2)-1=0.520 .$$
So $\Delta = \frac{100\times 0.520}{201}=\mathbf{0.259}$. Guaranteed coverage: $0.9-0.259=0.641$.

**The truth.** The calibration scores are a 50/50 mixture; the conformal threshold is the $181$st of $200$ order statistics, i.e. the $0.9005$ mixture quantile $q$ solving $\tfrac12\Phi(q)+\tfrac12\Phi(q-1)=0.9005$, giving $q=1.94$. Test coverage $=\Phi(1.94-1)=\Phi(0.94)=\mathbf{0.826}$. Realized gap $0.074$; slack ratio $\kappa=3.5$.

**Where the obstruction shows.** The bound would let coverage fall to $0.64$; the truth is $0.83$. Now switch to $\gamma=0.95$ decay. Pre-shift weight mass is $0.112$ against a total of $19.0$, so $\Delta=0.112\times0.520/20.0=\mathbf{0.0029}$ — a 90× tighter certificate. But the effective sample size drops to $(\sum w)^2/\sum w^2 = 19.0^2/9.26=39$, so the empirical $0.9$ quantile carries about $\pm 4.8$ points of coverage noise — larger than the entire bound it bought. And choosing $\gamma=0.95$ over $\gamma=1$ required knowing that a changepoint sat 100 steps back. The certificate is tight only when you already know the shift; when you do not, the honest bound ($0.259$) is below any coverage level anyone would deploy on.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*