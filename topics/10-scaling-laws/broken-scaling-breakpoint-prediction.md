---
id: 10-scaling-laws/broken-scaling-breakpoint-prediction
title: "Broken Neural Scaling and Break Point Prediction"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Broken Neural Scaling and Break Point Prediction

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/broken-scaling-breakpoint-prediction` · **Status:** open

## 1. Problem Statement

Neural scaling curves are not globally single power laws. Loss-versus-compute and metric-versus-scale curves show *breaks*: regions where the log-log slope changes, sometimes sharply. The problem is to predict where the next break is **before** training past it.

Three variants, different difficulty:

- **Measurement.** Given a measured curve $\{(x_i, y_i)\}$, decide whether a break exists and locate it. Requires a break to be defined independently of the fitting family, and requires noise on $y$ to be characterized.
- **Method.** Given observations restricted to $x \le X_{\max}$, output a predicted break location $\hat{d}$ and post-break slope $\hat{c}$ with calibrated uncertainty. Solved iff held-out prediction error on $\log_{10} d$ is small (say $< 0.3$ dex) on curves whose breaks were never seen during fitting.
- **Theory.** Derive break locations from properties of the data distribution, architecture, or optimizer — not fit them. Solved iff a mechanistic model predicts $d$ from measurable pre-break quantities (e.g. skill-frequency spectrum, manifold dimension) with no free parameter tuned on post-break data.

The method variant is where the money is: a break at $10^{26}$ FLOP invalidates a compute allocation chosen at $10^{23}$.

## 2. Formal Setting

Let $x$ be the scaling variable — parameters $N$, training tokens $D$, compute $C \approx 6ND$ FLOP, or dataset size — measured as the actual count used, not the nominal budget. Let $y(x)$ be the performance measure. Two families, and they behave differently:

- **Loss:** $y = \mathcal{L}(x) = -\frac{1}{|S|}\sum_{t \in S} \log p_\theta(t \mid t_{<})$ in nats/token on a held-out set $S$ drawn from the *same* distribution as training. Measured at a fixed checkpoint after a fixed LR schedule completes.
- **Downstream metric:** $y = \frac{1}{|T|}\sum_{q\in T} m(q)$, $m$ ∈ {0/1 exact match, Brier score, per-token edit distance, log-likelihood of the gold continuation}. The choice of $m$ changes whether a break exists at all (§4).

The **broken neural scaling law** (BNSL, Caballero et al., ICLR 2023) with $n$ breaks:

$$y(x) = a + b\,x^{-c_0}\prod_{i=1}^{n}\left(1 + \left(\frac{x}{d_i}\right)^{1/f_i}\right)^{-c_i f_i}$$

Here $a$ is the irreducible floor, $c_0$ the pre-break slope, $d_i$ the $i$-th break location, $f_i$ the transition sharpness ($f_i \to 0$ gives a kink, large $f_i$ a slow bend), and the asymptotic slope after $k$ breaks is $c_0 + \sum_{i\le k} c_i$.

Model-free break definition: with $u = \log x$, $v = \log y$ (or $\log(y-a)$), a break is a local extremum of the curvature

$$\kappa(u) = \frac{d^2 v}{du^2},$$

estimated from finite differences on a grid, with $\kappa$ significant relative to seed noise.

Assumptions, and their status:

| Assumption | Status |
|---|---|
| $y$ measured without noise | **Violated.** Seed-to-seed std on LM validation loss is typically 0.2–1% relative; on 0/1 downstream metrics far larger. |
| Compute-optimal frontier traced (each $x$ at its best $N,D$ split) | Usually violated — most curves fix one axis or use a fixed token multiplier. |
| Same data distribution across the ladder | Violated whenever data is repeated or filtered differently at scale. |
| Hyperparameters transfer along the ladder | Approximately true under $\mu$P; otherwise a hyperparameter artifact can masquerade as a break. |
| $n$ known a priori | **Never true.** $n$ is selected by fitting, which is the crux of §6. |

## 3. State of the Art

**Established (empirical, replicated in spirit).** Power-law fits over restricted ranges are reliable: Hestness et al. (2017), Kaplan et al. (2020), Hoffmann et al. (Chinchilla, NeurIPS 2022) — the compute-optimal $N^\star \propto C^{0.5}$, $D^\star \propto C^{0.5}$ result held up under reanalysis by Besiroglu et al. (2024), who corrected the fitted exponents' confidence intervals but not the conclusion. Gadre et al. (2024) established that downstream *average* error over many tasks is predictable from a loss fit using far less compute than the target run.

**Claimed but unablated.** BNSL (Caballero, Gupta, Rish, Krueger, ICLR 2023) is the strongest functional-form result: it fits and extrapolates better than a large set of competing forms across vision, language, RL, arithmetic, and double-descent curves, including the sharp inflection on multi-digit addition. What is *not* ablated: the number of breaks $n$ is chosen per curve, and extrapolation is scored by fitting a prefix of an *already observed* curve. There is no reported protocol in which $n$ and $d$ were committed before the post-break points existed. That is the difference between curve-fitting and prediction.

**Benchmark-number-only results.** Alabdulmohsin, Neyshabur, Zhai (NeurIPS 2022) release M4 and rank functional forms by extrapolation error; the ranking is a leaderboard over a fixed curve corpus, not a mechanism. Hu et al. (ICLR 2024, *PassUntil*) predict task performance of a 2.4B model from smaller ones using an infinite-resolution estimator; the reported deviation is small but for a single target scale and single model family.

**Theory SOTA.** Bahri, Dyer, Kaplan, Lee, Sharma (PNAS 2024) derive power-law exponents from data-manifold dimension in the variance-limited and resolution-limited regimes — but predicts *slopes*, not breaks. Michaud, Liu, Venkatesh, Tegmark (*The Quantization Model of Neural Scaling*, NeurIPS 2023) is the only mechanistic account that *generates* breaks: discrete "quanta" learned in frequency order, Zipf-distributed with exponent $\alpha+1$, give aggregate loss $\propto N^{-\alpha}$ while each individual quantum contributes a step.

## 4. What Is Known

- **Chinchilla scale.** ~400 models, 70M–16B params, 5B–500B tokens; fitted $a \approx 0.34$, $b \approx 0.28$ for $N^\star, D^\star \propto C^{a},C^{b}$. No break detected inside that range on validation loss.
- **Metric-induced breaks are often artifacts.** Schaeffer, Miranda, Koyejo (*Are Emergent Abilities of LLMs a Mirage?*, NeurIPS 2023, best paper) show that on BIG-Bench tasks, swapping exact-match for Brier score or token edit distance removes the discontinuity for the majority of tasks they examined; they also induce apparent emergence in vision models by changing the metric. So a break in a 0/1 metric is not evidence of a break in the model.
- **Breaks that survive metric change exist.** Multi-digit arithmetic and grokking-type curves (Power et al., 2022) show slope changes in continuous measures, not just thresholded ones.
- **Data-repetition break.** Muennighoff et al. (NeurIPS 2023): up to ~4 epochs of repeated data is nearly as good as fresh data; return decays sharply afterwards and is near-zero by ~40 epochs (models up to 9B params, up to 900B tokens). This is a break with a known, measurable cause.
- **Observational proxy.** Ruan, Maddison, Hashimoto (NeurIPS 2024) fit a low-dimensional capability space over ~100 public models; downstream metrics are log-linear in that space, and some "emergent" curves become smooth in it.

## 5. What Is Not Known

- **Theoretically open.** No theorem gives $d_i$ from pre-break observables. The quantization model predicts breaks *given* a quanta spectrum, but the spectrum is not measurable from a pre-break checkpoint.
- **Methodologically blocked.** "Does this curve have a break?" is not a well-posed question until (i) the metric is fixed to a continuous one, (ii) the noise model on $y$ is specified, and (iii) model selection over $n$ is penalized. None of the three is standard practice; break counts in published fits are effectively free parameters.
- **Empirically open.** No published study has run a *pre-registered* break prediction: fit on $x \le X_{\max}$, publish $\hat d$, then train past $\hat d$. The compute exists at frontier labs; the protocol has not been run in public.

## 6. Why It Is Hard

**Non-identifiability, quantitatively.** Before the break, the BNSL factor deviates from a pure power law by a fraction that shrinks as a power of $(x/d)$. Fitting inside a region where that deviation is under the seed noise floor makes $d$ formally unidentifiable — the likelihood is flat in $d$ over orders of magnitude. §10 makes this arithmetic explicit: with 1% noise you must train within roughly a factor of 2 of the break to see it at $3\sigma$. A method that only sees the break at $x \approx d/2$ predicts nothing useful, because $d/2$ is one more doubling, not one more decade.

Compounding: **confounded measurement** (breaks caused by LR schedule, data repetition, tokenizer, or eval saturation are indistinguishable in the curve from breaks caused by the model), **absent ground truth** (no curve has a known true $d$ except synthetic ones), and **evaluation that does not measure what it names** — extrapolation scores computed on already-observed curves reward flexible families for retrodiction.

## 7. Current Research (as of 2026)

- Continuous-resolution downstream estimators (PassUntil-style pass-rate estimation, log-likelihood-of-gold metrics) as the default for scaling curves — now common in open evaluation stacks.
- Observational / capability-space scaling (Stanford, Toronto) extended to more model families *(frontier — verify)*.
- Mechanistic decomposition of loss into skill or circuit components, following the quantization model, aimed at making the quanta spectrum measurable (MIT, and interpretability groups at Anthropic/DeepMind) *(frontier — verify)*.
- Data-constrained and repeat-aware scaling laws, where the break has a known cause and can be predicted — the one setting where break prediction is arguably solved.
- Bayesian model selection over break counts with explicit noise models; scattered, no standard benchmark *(frontier — verify)*.

## 8. Concrete Next Experiment

**Pre-registered break prediction on a public ladder.**

- **Scale.** Train 24 decoder-only models on one fixed corpus, $\mu$P-transferred hyperparameters, Chinchilla-optimal token counts: 8 sizes from 30M to 1B params × 3 seeds. Total compute ≈ $2\times10^{21}$ FLOP — a few thousand A100-hours. Evaluate 12 downstream tasks with *continuous* metrics only (gold-continuation log-likelihood, Brier, edit distance) plus validation loss.
- **Protocol.** Fit BNSL ($n \in \{0,1,2\}$, BIC-penalized) on models up to 300M params only. **Publish $\hat d_i$ with 90% intervals, hashed and timestamped, before training the 600M and 1B runs.** Then train them.
- **Control arm.** A single power law with no break, fitted on the same prefix, plus a permutation null in which curvature is estimated from seed-shuffled data. Both produce point predictions for the 1B loss.
- **Deciding number.** Median absolute error in $\log_{10}$ of the predicted 1B-scale metric, BNSL vs. single power law. If BNSL does not beat the no-break control by $\ge 2\times$ on curves that turned out to have a break, and the 90% intervals on $\hat d$ span more than 1.0 dex, the method variant is unsolved and the extrapolation literature is measuring retrodiction. Secondary: fraction of $\hat d$ intervals containing the post-hoc curvature-estimated break; target $\ge 0.9$ for calibration.

## 9. Key References

- **[Foundational]** J. Hestness et al. *Deep Learning Scaling is Predictable, Empirically.* 2017. — arXiv:1712.00409
- **[Foundational]** J. Kaplan, S. McCandlish et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** J. Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[SOTA]** E. Caballero, K. Gupta, I. Rish, D. Krueger. *Broken Neural Scaling Laws.* ICLR 2023. — arXiv:2210.14891
- **[SOTA]** I. Alabdulmohsin, B. Neyshabur, X. Zhai. *Revisiting Neural Scaling Laws in Language and Vision.* NeurIPS 2022. — arXiv:2209.06640
- **[SOTA]** R. Schaeffer, B. Miranda, S. Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS 2023. — arXiv:2304.15004
- **[SOTA]** E. Michaud, Z. Liu, U. Venkatesh, M. Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS 2023. — arXiv:2303.13506
- **[SOTA]** S. Gadre et al. *Language Models Scale Reliably With Over-Training and on Downstream Tasks.* 2024. — arXiv:2403.08540
- **[SOTA]** Y. Ruan, C. J. Maddison, T. Hashimoto. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS 2024. — arXiv:2405.10938
- **[Theory]** Y. Bahri, E. Dyer, J. Kaplan, J. Lee, U. Sharma. *Explaining Neural Scaling Laws.* PNAS, 2024. — arXiv:2102.06701
- **[Related]** N. Muennighoff et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[Survey]** J. Wei et al. *Emergent Abilities of Large Language Models.* TMLR, 2022. — arXiv:2206.07682

## 10. Worked Example

Take a BNSL with one break, floor $a = 0$:

$$y(x) = x^{-0.1}\left(1 + (x/d)^{2}\right)^{-0.15}, \qquad d = 10^{21}\text{ FLOP},\ c_1 = 0.3,\ f_1 = 0.5.$$

Post-break slope is $0.1 + 0.3 = 0.4$: a fourfold change in return per decade of compute. Enormous consequences for allocation.

Now suppose you have trained a ladder spanning $10^{18}$–$10^{20}$ FLOP — two full decades, a serious ablation. The largest point sits at $x = 10^{20} = d/10$. The relative deviation from a pure $x^{-0.1}$ power law there is

$$1 - \left(1 + 10^{-2}\right)^{-0.15} = 1 - e^{-0.15\ln(1.01)} \approx 1.49\times10^{-3},$$

i.e. **0.15%**. Seed-to-seed relative std on validation loss is typically 0.2–1%. The break is buried under noise. Every $d \ge 10^{21}$ fits the data equally well; the profile likelihood in $\log_{10} d$ is flat to the right, so BIC selects $n=0$ and you report a clean power law.

How close must you get? For a $3\sigma$ signal at $\sigma_{\text{rel}} = 1\%$ you need deviation $\ge 3\%$:

$$0.15\ln(1+u) = 0.03 \Rightarrow u = e^{0.2}-1 = 0.221 \Rightarrow (x/d)^2 = 0.221 \Rightarrow x \approx 0.47\,d.$$

**You must train within a factor of 2.1 of the break to detect it.** With $\sigma_{\text{rel}} = 0.2\%$ (30 seeds averaged, or a much larger eval set) the requirement relaxes to $x \approx 0.21\,d$ — still under one decade of headroom, at 5× the cost.

That is the obstruction, in one number: the warning arrives one doubling before the event, not one decade. Break prediction is not blocked by fitting technology; it is blocked by the ratio between the pre-break signal, which falls as $(x/d)^{1/f}$, and the noise floor, which does not fall at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*