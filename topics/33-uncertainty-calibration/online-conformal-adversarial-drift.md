---
id: 33-uncertainty-calibration/online-conformal-adversarial-drift
title: "Online Conformal Prediction Under Adversarial Drift"
topic: 33-uncertainty-calibration
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Online Conformal Prediction Under Adversarial Drift

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/online-conformal-adversarial-drift` · **Status:** partially-solved

## 1. Problem Statement

A model emits prediction sets $C_t \subseteq \mathcal{Y}$ online. Nature supplies $(x_t, y_t)$ with no distributional assumption — the sequence may be chosen adversarially, including in response to the algorithm's past outputs. The target is coverage $1-\alpha$.

Three variants, with sharply different difficulty:

- **Measurement.** What is the right validity criterion under drift? Long-run marginal coverage is achievable but vacuous (§10). Conditional coverage is impossible distribution-free with finite sets. The open measurement question is which intermediate criterion — coverage on all sub-intervals, on all subgroups, or under a betting/e-value test — is both achievable and non-trivial.
- **Method.** Given a criterion, build an algorithm that meets it *and* keeps sets small, without tuning a learning rate to the (unknown, time-varying) drift rate.
- **Theory.** Prove a lower bound tying local coverage error to set size — i.e. show that no algorithm can get $\varepsilon$-coverage on every window of length $w$ without inflating expected set size by some quantified factor.

Solving it means: an algorithm with a deterministic, assumption-free guarantee on a criterion that a degenerate predictor cannot satisfy, plus a matching lower bound.

## 2. Formal Setting

At round $t=1,\dots,T$: nature reveals $x_t \in \mathcal{X}$; the learner outputs $C_t$; nature reveals $y_t$. Define the miscoverage indicator, which is what is actually logged:

$$\mathrm{err}_t = \mathbf{1}\{y_t \notin C_t\}.$$

Sets are built from a **nonconformity score** $s_t: \mathcal{X}\times\mathcal{Y}\to\mathbb{R}$ fitted on data up to $t-1$ (e.g. $|y - \hat f_{t-1}(x)|$, or the CQR score $\max\{\hat q_{lo}(x)-y,\; y-\hat q_{hi}(x)\}$), and a threshold $\hat q_t$:

$$C_t = \{y : s_t(x_t,y) \le \hat q_t\}.$$

**Adaptive Conformal Inference (ACI)** maintains a level $\theta_t$ and sets $\hat q_t = \mathrm{Quantile}_{1-\theta_t}(\{s_i\}_{i \in W_t})$ over a calibration window $W_t$, updating

$$\theta_{t+1} = \theta_t + \gamma(\alpha - \mathrm{err}_t), \qquad \gamma > 0.$$

Measured quantities:

- **Long-run coverage error** $\;\big|\tfrac1T\sum_t \mathrm{err}_t - \alpha\big|$ — a single scalar over the run.
- **Local coverage error** $\;\mathrm{LCE}(w) = \max_{I:|I|=w} \big|\tfrac{1}{w}\sum_{t\in I}\mathrm{err}_t - \alpha\big|$ — worst over all contiguous windows of length $w$. Computed by a sliding max; $w$ must be reported, since $\mathrm{LCE}$ is monotone in $1/w$.
- **Efficiency** $\;\bar{\lambda} = \tfrac1T\sum_t |C_t|$ (Lebesgue measure for regression, cardinality for classification). Degenerate rounds must be reported separately: $|C_t| = \infty$ when $\theta_t \le 0$ and $|C_t| = 0$ when $\theta_t \ge 1$; means over runs with infinite sets are undefined, so report the *rate* of each.
- **Regret** against the best fixed threshold in hindsight under pinball loss $\rho_\alpha$.

**Assumptions and their status.** Exchangeability of $(x_t,y_t)$: assumed by split conformal, known false under drift — this is the premise of the whole line. Bounded scores $s_t \in [0,B]$: assumed by most regret bounds, violated by heavy-tailed residuals in finance and energy load. Score-model independence from the update (the model is refit online, so $s_t$ depends on past $\theta$): violated in every deployed system and not covered by existing proofs. Oblivious adversary: assumed in most analyses; a real adversary sees $C_t$ before choosing $y_t$.

## 3. State of the Art

**Theory SOTA (established).**
- Gibbs & Candès (NeurIPS 2021) — ACI. For $\theta_1\in[0,1]$ and any sequence whatever, $\big|\tfrac1T\sum_t \mathrm{err}_t - \alpha\big| \le \frac{1+\gamma}{\gamma T}$. Deterministic, no probability, no exchangeability.
- Gibbs & Candès (JMLR 2024) — DtACI, an expert-aggregation over a grid of $\gamma$; dynamic-regret guarantee against the best time-varying $\theta$ sequence.
- Bhatnagar, Wang, Xiong, Wang (ICML 2023) — SF-OGD and SAOCP; *strongly adaptive* regret $\tilde O(\sqrt{|I|})$ on every contiguous interval $I$, which is the first result addressing local, not just long-run, behaviour.
- Angelopoulos, Barber, Bates (ICML 2024) — decaying step sizes $\gamma_t \propto t^{-1/2-\varepsilon}$ give both long-run coverage and convergence of $\theta_t$ under stationarity, removing the need for a fixed $\gamma$.
- Bastani, Gupta, Jung, Noarov, Ramalingam, Roth (NeurIPS 2022) — MVP: threshold calibration valid simultaneously on a collection of groups against an adaptive adversary.
- Barber, Candès, Ramdas, Tibshirani (*Information and Inference*, 2021) — impossibility: distribution-free conditional coverage forces uninformative sets. This is the wall every conditional variant hits.

**Empirical SOTA (established by ablation).** Conformal PID (Angelopoulos, Candès, Tibshirani, NeurIPS 2023) adds an integrator and a "scorecaster" to the ACI update and is ablated on COVID-19 death forecasting, electricity demand, and stock volatility; the P/I/D components are separated in the paper's own ablation.

**Claimed but unablated.** Most papers report set width without separating three confounded sources: the base model's online adaptation, the calibrator's step size, and the score function. Reported width improvements are rarely attributed. Reported *local* coverage is almost always a plot, not a worst-window number; where LCE appears at all it is at a single window length chosen post hoc — a benchmark number, not a measurement protocol.

## 4. What Is Known

- The ACI bound is tight in form: with $\gamma=0.05$, $T=1000$, the slack is $\frac{1.05}{50} = 0.021$; at $T=10{,}000$ it is $0.0021$. Long-run marginal coverage is essentially free.
- The bound is achieved by a degenerate algorithm. Outputting $C_t=\mathbb{R}$ with probability $1-\alpha$ and $C_t=\emptyset$ otherwise gives exactly $\alpha$ long-run miscoverage. Any criterion satisfied by this predictor is not a validity criterion (§10).
- $\theta_t$ leaves $[0,1]$ in practice. On the volatility and time-series benchmarks used in this literature, small $\gamma$ produces runs with a nonzero rate of infinite-width sets during regime changes; this is reported by the ACI papers themselves as a known failure mode, not a bug.
- The $\gamma$ trade-off is monotone and measured: larger $\gamma$ shrinks recovery time after a changepoint linearly in $1/\gamma$ (§10) and inflates threshold variance, hence average width. No published rule selects $\gamma$ from data with a guarantee; DtACI and AgACI (Zaffran, Féron, Goude, Dieuleveut, Josse, ICML 2022) sidestep it by aggregating over a grid.
- Strongly adaptive regret $\tilde O(\sqrt{|I|\log T})$ (SAOCP, ICML 2023) does *not* imply a local coverage bound of the same order; the translation from pinball regret to $\mathrm{LCE}(w)$ passes through the score density and is lossy where the density is small.
- Scale of the evidence: essentially all of it is univariate or low-dimensional time series with $T$ in the $10^3$–$10^5$ range. There is no adversarial-drift result at the scale of a deployed LLM-scoring pipeline.

## 5. What Is Not Known

- **Theoretically open.** No lower bound relating $\mathrm{LCE}(w)$ to expected set size. Nobody has proved that an algorithm achieving $\mathrm{LCE}(w) \le \varepsilon$ for all $w \ge w_0$ must pay a quantified width penalty, nor exhibited an algorithm that avoids one. The impossibility result of Barber et al. covers conditional coverage, not windowed coverage — the gap between them is unmapped.
- **Theoretically open.** All guarantees assume the score function is fixed or exogenous. Coupled analysis, where the base model is refit online using data whose selection depends on past $C_t$, has no theory.
- **Empirically open.** Whether PID/DtACI/SAOCP differ on worst-window local coverage at matched average width. Runnable today; not run, because papers report different metrics on different datasets.
- **Methodologically blocked.** There is no agreed benchmark for "adversarial drift". Existing evaluations use historical series whose drift is fixed and, by now, implicitly tuned against. Constructing an adversary that is strong but not degenerate — it must not simply force $|C_t|=\infty$ — is itself unsolved, and without it the field cannot measure adversarial robustness at all.

## 6. Why It Is Hard

**The evaluation does not measure the thing it names.** "Coverage" in this literature means the time-average of $\mathrm{err}_t$, a statistic invariant to *when* the errors land. A method that misses every point during each regime change and over-covers between them scores identically to one that is uniformly calibrated. So the guarantee that is easy to prove is exactly the one that carries no information about the failure mode practitioners care about, and the metric that would carry that information ($\mathrm{LCE}(w)$) has a free parameter $w$ with no principled setting: as $w \to 1$ it is maximal for every algorithm producing finite sets, as $w \to T$ it collapses to the vacuous long-run number.

Compounding this: the two natural knobs (step size $\gamma$, window length $|W_t|$) both trade local responsiveness against width, and their effects are **non-identifiable** from a single run — a wide interval could mean a cautious calibrator or a bad base model, and the logged data cannot separate them without a counterfactual re-run with the calibrator frozen.

## 7. Current Research (as of 2026)

- **Betting and e-value calibration.** Testing-by-betting reformulations of the ACI update (Podkopaev, Xu, Lee, *Adaptive Conformal Inference by Betting*, ICML 2024) replace the fixed step size with a wealth process, removing the $\gamma$ tuning problem. Whether this improves local coverage or only removes a hyperparameter is not yet ablated *(frontier — verify)*.
- **Conditional guarantees as a linear program.** Gibbs, Cherian, Candès — coverage conditional on a chosen function class, threading between marginal and full conditional validity. Extension to the online adversarial setting is active *(frontier — verify)*.
- **Risk control beyond coverage.** Feldman, Bates, Romano (*Achieving Risk Control in Online Learning Settings*, TMLR 2023) — Rolling RC, controlling general losses online.
- **Drift detection coupled to calibration.** Podkopaev & Ramdas (ICLR 2022) on sequential detection of harmful shift; combining a detector with a reset of $\theta_t$ is a live but under-evaluated direction.
- **LLM applications.** Conformal abstention and factuality filtering apply these updates to token-level or claim-level scores where the exchangeability violation is severe and unquantified *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** At matched average set width, do PID, DtACI, and SAOCP differ in worst-window local coverage?

**Scale.** Six series, $T \approx 20{,}000$ each: electricity demand (ELEC2), a stock-volatility series, COVID-19 death forecasts, and three synthetic streams with injected changepoints of known magnitude at known times. $\alpha = 0.1$. One fixed base model per series, frozen — the calibrator is the only thing that varies. 5 seeds. Cost: single CPU, under 100 core-hours.

**Control arm.** Two controls, both mandatory. (1) Fixed-threshold split conformal calibrated on the first 20% — no adaptation. (2) The **degenerate predictor**: $C_t = \mathbb{R}$ w.p. 0.9, $\emptyset$ w.p. 0.1. Any metric on which the degenerate arm is not worst is disqualified as a validity metric. This second arm is the point of the experiment.

**Deciding number.** $\mathrm{LCE}(100)$ — worst absolute coverage deviation over any 100-round window — after re-tuning each method's hyperparameters so all methods land within $\pm 2\%$ of the same $\bar{\lambda}$. A separation of $\ge 0.05$ in $\mathrm{LCE}(100)$ between the best and worst adaptive method, consistent across $\ge 5$ of 6 series, establishes that local coverage discriminates between these algorithms. A spread below $0.02$ establishes the opposite — that the published differences are width effects, not calibration effects — which would redirect the field to the lower-bound question in §5.

## 9. Key References

- **[Foundational]** Isaac Gibbs, Emmanuel Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS, 2021. — arXiv:2106.00170
- **[Foundational]** Rina Foygel Barber, Emmanuel Candès, Aaditya Ramdas, Ryan Tibshirani. *The limits of distribution-free conditional predictive inference.* Information and Inference, 2021.
- **[SOTA]** Isaac Gibbs, Emmanuel Candès. *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts.* JMLR, 2024.
- **[SOTA]** Aadyot Bhatnagar, Huan Wang, Caiming Xiong, Yu Bai. *Improved Online Conformal Prediction via Strongly Adaptive Online Learning.* ICML, 2023.
- **[SOTA]** Anastasios Angelopoulos, Emmanuel Candès, Ryan Tibshirani. *Conformal PID Control for Time Series Prediction.* NeurIPS, 2023.
- **[SOTA]** Anastasios Angelopoulos, Rina Foygel Barber, Stephen Bates. *Online conformal prediction with decaying step sizes.* ICML, 2024.
- **[SOTA]** Osbert Bastani, Varun Gupta, Christopher Jung, Georgy Noarov, Ramya Ramalingam, Aaron Roth. *Practical Adversarial Multivalid Conformal Prediction.* NeurIPS, 2022.
- **[Related]** Margaux Zaffran, Olivier Féron, Yannig Goude, Aymeric Dieuleveut, Julie Josse. *Adaptive Conformal Predictions for Time Series.* ICML, 2022.
- **[Related]** Shai Feldman, Stephen Bates, Yaniv Romano. *Achieving Risk Control in Online Learning Settings.* TMLR, 2023.
- **[Related]** Aleksandr Podkopaev, Aaditya Ramdas. *Tracking the risk of a deployed model and detecting harmful distribution shifts.* ICLR, 2022.
- **[Survey]** Anastasios Angelopoulos, Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* Foundations and Trends in Machine Learning, 2023.
- **[Survey]** Anastasios Angelopoulos, Rina Foygel Barber, Stephen Bates. *Theoretical Foundations of Conformal Prediction.* Cambridge University Press, 2025.

## 10. Worked Example

$\alpha = 0.1$, $\gamma = 0.05$, $T = 2000$, ACI with $\theta_{t+1} = \theta_t + \gamma(\alpha - \mathrm{err}_t)$.

**Step 1 — the guarantee.** $\big|\tfrac1T\sum\mathrm{err}_t - 0.1\big| \le \frac{1.05}{0.05 \times 2000} = 0.0105$. Empirical coverage lands in $[0.8895, 0.9105]$ for *any* sequence.

**Step 2 — a changepoint.** At $t = 1000$ the score distribution shifts so that the threshold correct for the new regime sits at level $\theta^\star = \theta_{1000} - 0.4$. Until $\theta_t$ travels that distance, every point is missed, so $\mathrm{err}_t = 1$ and each step moves $\theta$ by $\gamma(\alpha - 1) = 0.05 \times (-0.9) = -0.045$. Recovery takes

$$\lceil 0.4 / 0.045 \rceil = 9 \text{ rounds.}$$

Local coverage on that window: $0/9 = 0\%$ against a 90% target, so $\mathrm{LCE}(9) = 0.9$ — the maximum possible.

**Step 3 — the guarantee does not notice.** Nine total misses out of 2000 is $0.45\%$ of the run. The long-run coverage is unchanged to three decimals. The certificate in Step 1 is fully intact while the predictor was maximally wrong for a contiguous block.

**Step 4 — the fix costs.** Setting $\gamma = 0.2$ cuts recovery to $\lceil 0.4/0.18 \rceil = 3$ rounds. But the stationary fluctuation of $\theta_t$ scales roughly with $\gamma$: the level wanders over a band of order $\pm\gamma\sqrt{\alpha(1-\alpha)/\gamma} \approx \pm 0.12$ instead of $\pm 0.06$, and the resulting threshold jitter inflates average width. On the flat parts of the run $\theta_t$ also spends more time near 0, raising the rate of infinite-width sets.

**The obstruction, visible.** Now run the degenerate control: $C_t = \mathbb{R}$ w.p. 0.9, $\emptyset$ w.p. 0.1. Its long-run coverage is $0.9 \pm O(T^{-1/2})$ — it passes Step 1. Its $\mathrm{LCE}(9)$ is small, since misses are spread uniformly by construction. So on both published metrics it *beats* ACI at the changepoint, while carrying zero information. The field's headline guarantee is satisfied by a predictor that never looks at $x_t$, and the natural repair ($\mathrm{LCE}$) is only meaningful once width is held fixed — which is precisely the matched-width protocol §8 demands and no published comparison has used.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*