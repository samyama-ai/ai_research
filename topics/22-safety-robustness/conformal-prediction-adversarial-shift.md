---
id: 22-safety-robustness/conformal-prediction-adversarial-shift
title: "Conformal Prediction Under Adversarial Exchangeability Violation"
topic: 22-safety-robustness
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Conformal Prediction Under Adversarial Exchangeability Violation

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/conformal-prediction-adversarial-shift` · **Status:** partially-solved

## 1. Problem Statement

Split conformal prediction converts any black-box model into a set predictor with finite-sample marginal coverage $\ge 1-\alpha$, on one assumption: the calibration points and the test point are **exchangeable**. The problem is what survives when an adversary breaks that assumption on purpose.

Three variants, with different difficulty:

- **Measurement.** Given a deployed conformal predictor and an adversary with a stated budget, *estimate* the realized coverage. Hard because the adversary's optimal attack is not the one you tested, so measured coverage upper-bounds nothing.
- **Method.** Construct a set predictor $\hat{C}$ whose coverage is $\ge 1-\alpha$ for **every** adversary in a budget class, with prediction sets small enough to be decision-relevant (not the trivial full label set).
- **Theory.** Characterize the exact coverage/efficiency frontier: for a given attack budget, what is the smallest achievable expected set size at guaranteed coverage $1-\alpha$, and is any known method on that frontier?

Three distinct attack surfaces must be kept separate — they are routinely conflated:
1. **Test-time perturbation:** $x_{n+1} \mapsto x_{n+1}+\delta$, $\|\delta\|_2 \le \epsilon$; calibration set clean.
2. **Calibration poisoning:** the adversary edits $k$ of $n$ calibration points (features, labels, or both).
3. **Sequential/strategic shift:** data arrive online and the environment reacts to the predictor.

Solved means: a bound whose slack is a measurable function of the stated budget, plus an attack search strong enough that no better attack is found under a compute budget an order of magnitude larger than the defender's.

## 2. Formal Setting

Data $Z_i=(X_i,Y_i)\in\mathcal{X}\times\mathcal{Y}$. A fitted model gives a nonconformity score $s:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}$ (measured as, e.g., $s(x,y)=1-\hat{p}_y(x)$ for classification, or the APS/RAPS cumulative-mass score). Calibration scores $S_i=s(X_i,Y_i)$, $i=1..n$. With $\hat{q}=\lceil (n+1)(1-\alpha)\rceil$-th smallest $S_i$,
$$\hat{C}(x)=\{y: s(x,y)\le \hat{q}\},\qquad \mathbb{P}(Y_{n+1}\in \hat{C}(X_{n+1}))\ge 1-\alpha .$$

**Measured quantities.**
- Empirical coverage $\widehat{\mathrm{Cov}} = \frac{1}{m}\sum_{j=1}^m \mathbf{1}\{Y_j\in\hat{C}(\tilde{X}_j)\}$ over $m$ held-out test points, $\tilde X_j$ the attacked input. Binomial standard error $\sqrt{\alpha(1-\alpha)/m}$ — at $\alpha=0.1,\ m=10{,}000$ that is $0.3$ points, so any claimed gap below $\approx 1$ point is noise.
- Efficiency: mean set size $\frac{1}{m}\sum_j |\hat{C}(\tilde X_j)|$ (classification) or mean interval width (regression).
- Attack budget: $\epsilon$ in a fixed norm, plus the *search* budget (PGD steps, restarts) — the second is part of the measurement and is usually unreported.

**Robust-coverage predicate.** For adversary class $\mathcal{A}_\epsilon$,
$$\inf_{A\in\mathcal{A}_\epsilon}\ \mathbb{P}\big(Y_{n+1}\in \hat{C}(A(X_{n+1}))\big)\ \ge\ 1-\alpha .$$

**Beyond-exchangeability bound** (Barber, Candès, Ramdas, Tibshirani, *Ann. Statist.* 2023): with weights $w_i\in[0,1]$ and $\sigma_i$ the index-swap of the joint law,
$$\mathbb{P}(Y_{n+1}\in\hat{C}(X_{n+1}))\ \ge\ 1-\alpha-\frac{\sum_{i=1}^n w_i\,d_{\mathrm{TV}}(Z,\,Z^{\sigma_i})}{1+\sum_{i=1}^n w_i}.$$
This is the general slack term. It is *not* computable from data — $d_{\mathrm{TV}}$ is unknown — which is the crux of §6.

**Assumptions known violated in practice.** (i) Exchangeability — violated by construction here. (ii) The score $s$ is fixed independent of calibration data — violated whenever the model is retrained or the score is tuned on the calibration split. (iii) Continuous scores / no ties — violated for discrete classifiers, costing up to $1/(n+1)$ coverage. (iv) The attack budget class is known at design time — violated always; deployed adversaries are not norm-ball constrained.

## 3. State of the Art

**Theory SOTA (established).**
- *Weighted conformal* under known covariate shift with likelihood ratio $w(x)=\frac{dP_{\text{test}}}{dP_{\text{train}}}(x)$: exact $1-\alpha$ coverage (Tibshirani, Barber, Candès, Ramdas, NeurIPS 2019). Requires $w$ known or estimated; estimation error is not absorbed by the theorem.
- *Beyond exchangeability*: the TV-slack bound above; assumption-free, but vacuous when the adversary is unconstrained.
- *f-divergence-ball robust conformal* (Cauchois, Gupta, Ali, Duchi, JASA 2024): coverage over all shifts within $D_f(Q\|P)\le\rho$, by inflating the quantile level via a worst-case-shift map. Valid but conservative; set size grows fast in $\rho$.
- *Randomized-smoothing conformal* (RSCP; Gendler, Weng, Daniel, Romano, ICLR 2022): certified coverage under $\ell_2$ test-time perturbation by bounding the score's change under smoothing.
- *RSCP+* (Yan et al., ICLR 2024, "Provably Robust Conformal Prediction with Improved Efficiency"): showed RSCP's Monte-Carlo smoothing estimate breaks the certificate, and repaired it; adds a post-training transformation to cut set size.

**Online SOTA (established).** Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021) updates $\alpha_t$ by $\alpha_{t+1}=\alpha_t+\gamma(\alpha-\mathrm{err}_t)$ and gives a *deterministic* long-run bound $|\frac{1}{T}\sum_t \mathrm{err}_t-\alpha|\le \frac{\max(\alpha_1,1-\alpha_1)+\gamma}{T\gamma}$, with no distributional assumption at all — so it holds under adversarial sequences. Extensions: DtACI (Gibbs & Candès 2024), Conformal PID (Angelopoulos, Candès, Tibshirani, NeurIPS 2023).

**Claimed but unablated.** (a) Certified-radius numbers for RSCP-family methods are reported on CIFAR-10/ImageNet at small $\epsilon$ only; the comparison arm is usually vanilla conformal under the *same* attack, not an attack tuned against the defense. (b) Claims that conformal prediction is "naturally robust" to label noise (Einbinder, Romano, Sesia, Zhou, NeurIPS 2022 — dispersive noise) are established for the noise model stated and do **not** transfer to adversarial label flips. (c) Poisoning defenses that report coverage restoration on $k/n \le 1\%$ are benchmark numbers, not certificates.

## 4. What Is Known

- Split conformal coverage is $[1-\alpha,\ 1-\alpha+\frac{1}{n+1}]$ exactly, for any $n$, any model, under exchangeability. At $n=1000,\alpha=0.1$ the upper slack is $0.1$ points.
- **Poisoning is cheap.** Coverage is a rank statistic: moving the calibration score at the $\lceil(n+1)(1-\alpha)\rceil$ rank is enough. With $n=1000,\alpha=0.1$, the threshold is rank 901; corrupting the top $k=100$ scores downward ($10\%$ of calibration) shifts $\hat q$ to the clean rank-801 value, dropping coverage to roughly $80\%$ — a 10-point loss for a 10% budget. This scaling ($\approx k/n$ coverage loss) is elementary and reproduced across setups.
- **Test-time attacks destroy coverage.** Gendler et al. (ICLR 2022) report that on CIFAR-10 with $\ell_2$ perturbations of $\epsilon=0.125$, standard conformal coverage falls far below nominal $90\%$ while RSCP restores $\ge 90\%$ at the cost of markedly larger sets (single digits of labels out of 10). Order of magnitude, not exact value, should be taken as the durable claim.
- ACI's long-run bound is a theorem, holds for adversarial sequences, and is tight in $\gamma$ — but it is about *time-averaged* coverage; conditional coverage in any window can be arbitrarily bad.
- Conditional coverage is impossible: no non-trivial finite-length interval achieves $\mathbb{P}(Y\in\hat C(X)\mid X=x)\ge 1-\alpha$ for all $x$ distribution-free (Vovk 2012; Lei & Wasserman 2014; Barber, Candès, Ramdas, Tibshirani 2021). Adversarial robustness inherits this: an adversary that concentrates on the worst $x$-region is fighting an impossibility result, not a weak method.

## 5. What Is Not Known

- **Theoretically open.** The coverage/efficiency frontier under a bounded test-time adversary. No lower bound says "any method with certified $90\%$ coverage at $\epsilon=0.25$ on ImageNet must have expected set size $\ge B$". Without it, nobody knows whether RSCP+'s sets are 2× or 20× from optimal.
- **Theoretically open.** Joint calibration-poisoning *and* test-time attack. Existing results treat one channel at a time; the composed guarantee is not additive in any known way.
- **Empirically open.** Whether adaptive/online conformal (ACI, PID) retains useful coverage against an adversary that explicitly optimizes against the $\alpha_t$ update rule — the update is public, so it is attackable. The experiment is runnable today at modest scale; it has not been run against a strong adaptive attacker.
- **Methodologically blocked.** "Realized coverage under adversarial shift" is not a well-defined measurement without a specified attack class *and* a specified search budget. Reported robust-coverage numbers are attack-dependent upper bounds on the true worst case, with no bound on the gap.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure the thing it names**, compounded by **non-identifiability of the slack term**.

- The general bound's penalty $\sum_i w_i d_{\mathrm{TV}}(Z,Z^{\sigma_i})$ is not estimable from the same data used for calibration — an adversary chooses the shift, and TV distance between an empirical sample and an unknown adversarial law is not identifiable from finitely many samples. So the honest bound is uncomputable and the computable bounds require assuming the budget class.
- Robust coverage is an $\inf$ over attacks; every experiment reports a $\sup$-side estimate from one attack. Adversarial ML has a decade of evidence (obfuscated gradients, adaptive attacks) that this gap is often large and detected only after publication.
- Certification cost: randomized-smoothing certificates need $10^2$–$10^5$ forward passes per test point. On ImageNet at $m=10^4$ test points that is up to $10^9$ inferences — enough that the honest evaluation is rarely run at the scale where set sizes matter.

## 7. Current Research (as of 2026)

- Certified conformal under $\ell_p$ perturbation, tightening RSCP+ set sizes with training-time smoothing and score reshaping — Technion (Romano), UCSD, and collaborators.
- Poisoning and label-attack analyses of conformal calibration, and robustness of conformal on graphs — Bojchevski's group (Zargarbashi, Bojchevski) *(frontier — verify current results)*.
- Online/adaptive conformal against non-stochastic sequences: PID control, decaying-weight ACI, and regret-style guarantees — Berkeley/Stanford (Angelopoulos, Candès, Gibbs, Bates).
- Conformal risk control and distribution-free bounds for non-coverage losses (Bates, Angelopoulos, Lei, Jordan), extended to shifted deployments *(frontier — verify)*.
- Conformal abstention/guardrails for LLMs, where "exchangeability" is broken by prompt distribution drift and by users adapting to the guardrail *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does adaptive conformal inference retain useful coverage against an attacker that knows the update rule?

**Scale.** CIFAR-100 or ImageNet-val, ResNet-50 or a ViT-B/16; $T = 20{,}000$ sequential test points; $\alpha=0.1$; ACI with $\gamma\in\{0.005,0.01,0.05\}$; scores = APS.

**Attack arm.** A white-box scheduler that, at each $t$, sees $\alpha_t$ and chooses whether to submit an easy point (drives $\alpha_t$ down) or a PGD-attacked hard point at $\epsilon=0.25$ ($\ell_2$), budget: attacked fraction $\le 20\%$ of the stream, 100 PGD steps, 5 restarts. The scheduler is optimized to minimize coverage over the *worst 500-point window*, not over the full stream.

**Control arms.** (1) Static split conformal, same stream. (2) ACI with the same $20\%$ attacked points inserted at random positions (non-adaptive adversary). (3) RSCP+ certified sets on the same stream.

**Deciding number.** Minimum coverage over any 500-point sliding window, $\min_t \frac{1}{500}\sum_{u=t}^{t+499}\mathbf{1}\{Y_u\in\hat C_u\}$. Binomial SE at $m=500,\alpha=0.1$ is $1.3$ points; declare a real effect at a $\ge 5$-point drop.
- If arm 1's worst window drops below $\approx 75\%$ while ACI stays $\ge 85\%$, ACI's long-run bound is doing real local work.
- If the adaptive scheduler pushes ACI's worst window to $\le 60\%$ while the random-insertion control stays $\ge 85\%$, the headline result is: **ACI's assumption-free guarantee is time-averaged only, and an adversary who reads $\alpha_t$ can concentrate all the miscoverage into the window that matters.** That is the publishable outcome and it is currently unrun.

Cost estimate: ~$10^6$ forward passes for arms 1–2, plus the PGD search ($\approx 500\times$ per attacked point on $4{,}000$ points $\approx 2\times10^6$); a single 8-GPU node-day.

## 9. Key References

- **[Foundational]** Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer, 2005 (2nd ed. 2022).
- **[Foundational]** Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. *Distribution-Free Predictive Inference for Regression.* JASA, 2018.
- **[Foundational]** Tibshirani, Barber, Candès, Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS, 2019. — arXiv:1904.06019
- **[SOTA — theory]** Barber, Candès, Ramdas, Tibshirani. *Conformal prediction beyond exchangeability.* Annals of Statistics, 2023. — arXiv:2202.13415
- **[SOTA — online]** Gibbs, Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS, 2021. — arXiv:2106.00170
- **[SOTA — online]** Angelopoulos, Candès, Tibshirani. *Conformal PID Control for Time Series Prediction.* NeurIPS, 2023.
- **[SOTA — certified]** Gendler, Weng, Daniel, Romano. *Adversarially Robust Conformal Prediction.* ICLR, 2022.
- **[SOTA — certified]** Yan et al. *Provably Robust Conformal Prediction with Improved Efficiency.* ICLR, 2024.
- **[SOTA — shift ball]** Cauchois, Gupta, Ali, Duchi. *Robust Validation: Confident Predictions Even When Distributions Shift.* JASA, 2024.
- **[Related]** Einbinder, Romano, Sesia, Zhou. *Label Noise Robustness of Conformal Prediction.* (NeurIPS 2022 version: *Conformal Prediction is Robust to Dispersive Label Noise*.)
- **[Impossibility]** Barber, Candès, Ramdas, Tibshirani. *The limits of distribution-free conditional predictive inference.* Information and Inference, 2021.
- **[Survey]** Angelopoulos, Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* FnT in Machine Learning, 2023. — arXiv:2107.07511

## 10. Worked Example

**Setup.** ImageNet, $\alpha=0.1$, $n=1000$ calibration points, softmax score $s(x,y)=1-\hat p_y(x)$. Threshold rank $\lceil 1001\times0.9\rceil = 901$. Suppose the sorted calibration scores near the top are $S_{(896..905)} = (0.72,\,0.74,\,0.75,\,0.77,\,0.79,\,0.82,\,0.85,\,0.88,\,0.91,\,0.95)$, so $\hat q = S_{(901)} = 0.82$.

**Attack 1 — poisoning 20 points ($2\%$).** The adversary replaces the 20 highest-score calibration points with confidently-correct examples ($s\approx0.01$). The rank-901 order statistic of the corrupted set equals the clean rank-881 value; if the clean score distribution has $S_{(881)}=0.63$, then $\hat q$ falls $0.82\to0.63$. Coverage on clean test data drops from $90\%$ to the clean CDF at $0.63$, i.e. about $88\%$ — a $2\%$ budget buys about a $2$-point loss, matching the $k/n$ rule. **The obstruction is not the size of the loss but its invisibility:** the defender's diagnostic (empirical coverage on the *poisoned* calibration set) reads exactly $90\%$, by construction. The rank statistic is self-consistent with its own corrupted sample.

**Attack 2 — test-time.** Take a point with clean score $0.55 < \hat q$, so the true label is covered. A 100-step PGD attack at $\epsilon=0.25$ ($\ell_2$) raises $\hat p$ on a wrong class and drops $\hat p_y$ from $0.45$ to $0.08$, so $s = 0.92 > 0.82$: coverage lost on that point. Repeat over the test set and coverage falls from $90\%$ toward the fraction of points whose score margin exceeds what the attack can move.

**Where the two attacks compose badly.** The defender who certifies against Attack 2 with RSCP+ inflates $\hat q$ using the *calibration* scores. If those scores were poisoned by Attack 1, the inflation is computed off a corrupted base and the certificate inherits the $2$-point deficit — the certified radius is stated in $\ell_2$ units about a threshold that is itself wrong by an amount the certificate does not model. No published bound covers this composition; there is no measurement available to the defender that separates "$\hat q$ is low because the model is good" from "$\hat q$ is low because 20 calibration points were edited". That non-identifiability, not the attack strength, is what keeps the problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*