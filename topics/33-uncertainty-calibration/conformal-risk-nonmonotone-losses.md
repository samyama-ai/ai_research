---
id: 33-uncertainty-calibration/conformal-risk-nonmonotone-losses
title: "Conformal Risk Control for Non-Monotone Losses"
topic: 33-uncertainty-calibration
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Conformal Risk Control for Non-Monotone Losses

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/conformal-risk-nonmonotone-losses` · **Status:** open

## 1. Problem Statement

Conformal risk control (CRC) picks a single scalar knob $\lambda$ — a score threshold, an abstention rate, a beam width, a retrieval depth — so that the expected loss of the deployed predictor is at most $\alpha$, with no assumption on the data distribution beyond exchangeability. The theorem behind it needs the per-example loss $L_i(\lambda)$ to be **monotone non-increasing in $\lambda$**. Many losses people actually deploy are not: set-level $F_1$ is unimodal in set size, calibration error is U-shaped in temperature, hallucination-vs-omission trade-offs in RAG are V-shaped in retrieval depth, and any loss combining a coverage term with a cost term has an interior optimum.

- **Input.** Calibration data $(X_i, Y_i)_{i=1}^n$ exchangeable with a test point $(X_{n+1}, Y_{n+1})$; a family of predictors $\{\mathcal{C}_\lambda\}_{\lambda \in \Lambda}$, $\Lambda \subset \mathbb{R}$ compact; a bounded loss $L: \mathcal{Y} \times \mathcal{C} \to [0, B]$.
- **Output.** $\hat\lambda(D_{\text{cal}})$.
- **Predicate (expectation variant).** $\mathbb{E}[L_{n+1}(\hat\lambda)] \le \alpha$, marginally over calibration and test draws.
- **Predicate (PAC variant).** $\mathbb{P}_{D_{\text{cal}}}\big(R(\hat\lambda) \le \alpha\big) \ge 1 - \delta$ where $R(\lambda) = \mathbb{E}[L(\lambda)]$.

Three variants, different difficulty:

| Variant | Question | Status |
|---|---|---|
| **Theory** | Does a finite-sample, distribution-free, expectation-form guarantee exist for arbitrary bounded $L(\lambda)$ without a union bound over $\Lambda$? | Open |
| **Method** | Given that Learn-then-Test already solves the PAC variant by multiple testing, how much of its $O(\sqrt{\log|\Lambda|/n})$ price is removable? | Open, runnable |
| **Measurement** | For a non-monotone risk the feasible set $\{\lambda : R(\lambda) \le \alpha\}$ is a union of intervals; which element should a method return, and what is the right efficiency metric? | Under-defined |

Solving it means: a selection rule with a finite-sample guarantee that degrades gracefully in the *shape complexity* of $R(\cdot)$ (e.g. number of sign changes of $R'$) rather than in $|\Lambda|$.

## 2. Formal Setting

Data $(X_i, Y_i) \sim P$, exchangeable, $i = 1, \dots, n+1$, with $Y_{n+1}$ unobserved. Loss $L_i(\lambda) := L(Y_i, \mathcal{C}_\lambda(X_i)) \in [0, B]$, measured as the sample mean over the calibration split:
$$\hat R_n(\lambda) = \frac{1}{n}\sum_{i=1}^n L_i(\lambda), \qquad R(\lambda) = \mathbb{E}[L_1(\lambda)].$$
$\hat R_n$ is what you compute; $R$ is never observed. **Monotonicity** is the assumption $L_i(\lambda') \le L_i(\lambda)$ for $\lambda' \ge \lambda$, *pathwise for each $i$* — a strictly stronger condition than $R$ being monotone, and the one CRC's proof uses.

CRC (Angelopoulos et al., ICLR 2024) sets
$$\hat\lambda = \inf\Big\{\lambda \in \Lambda : \frac{n}{n+1}\hat R_n(\lambda) + \frac{B}{n+1} \le \alpha\Big\},$$
and proves $\mathbb{E}[L_{n+1}(\hat\lambda)] \le \alpha$ given (i) pathwise monotonicity, (ii) right-continuity in $\lambda$, (iii) $L \le B < \infty$, (iv) $L(\lambda_{\max}) \le \alpha$ almost surely.

Learn-then-Test (LTT; Angelopoulos, Bates, Candès, Jordan, Lei, 2021) drops all monotonicity. It grids $\Lambda$ into $m$ points, forms a $p$-value $p_\lambda$ for $H_\lambda: R(\lambda) > \alpha$ — typically the Hoeffding–Bentkus bound of Bates et al. (JACM 2021), $p_\lambda^{\text{HB}} = \min\{e \cdot \mathbb{P}(\text{Bin}(n, \alpha) \le \lceil n\hat R_n\rceil), e^{-nh_1(\hat R_n, \alpha)}\}$ — and returns $\hat\Lambda = \{\lambda : p_\lambda \le \delta/m\}$ or a fixed-sequence / graphical FWER procedure. The guarantee is PAC, not expectation-form.

**Measured quantities.** $n$: calibration set size after any split used to fit the score. $m = |\Lambda_{\text{grid}}|$: grid resolution, chosen by the practitioner and rarely reported. Efficiency: mean set size, mean abstention rate, or mean retrieval cost at $\hat\lambda$, averaged over the same test split.

**Assumptions violated in practice.**
- *Pathwise monotonicity*: violated by construction here — this is the problem.
- *Exchangeability*: broken under distribution shift and for LLM outputs scored by a judge whose behaviour drifts across model versions.
- *Independence of $\Lambda$ from calibration data*: violated whenever the grid is chosen after looking at a risk curve plotted on the same data. This is common and usually unreported.
- *Bounded $B$ known a priori*: for unbounded losses (log-loss, cost in dollars) $B$ is set by clipping, and the clip point silently enters the bound as $B/(n+1)$.

## 3. State of the Art

**Established (theory).**
- RCPS (Bates, Angelopoulos, Lei, Malik, Jordan, *JACM* 2021): PAC risk control for monotone losses via a pointwise upper confidence bound; proof needs monotonicity.
- CRC (Angelopoulos, Bates, Fisch, Lei, Schuster, *ICLR* 2024): expectation-form control, no grid, no $\delta$; needs pathwise monotonicity.
- LTT (Angelopoulos et al., arXiv:2110.01052, 2021): PAC control for **arbitrary** bounded losses, including non-monotone. This is the correct statement of "the problem is solved for the PAC variant" — the gap is efficiency, not validity.
- Quantile Risk Control (Snell, Zollo, Deng, Pitassi, Zemel, *ICLR* 2023) and dispersion control (Deng, Zollo, Snell, Pitassi, Zemel, *NeurIPS* 2023): control functionals of the loss distribution, not just its mean, via numerical CDF bounds; also non-monotone-tolerant.

**Established (method/efficiency).** Pareto Testing (Laufer-Goldshtein, Fisch, Barzilay, Jaakkola, *ICLR* 2023) orders hypotheses along an estimated Pareto frontier from a held-out split so that fixed-sequence testing spends its power on the few $\lambda$ that matter — the strongest published answer to LTT's multiplicity cost, and it is ablated against Bonferroni LTT.

**Claimed but unablated.** Papers applying CRC to $F_1$-style or composite losses routinely "monotonise" — replace $L_i(\lambda)$ by $\tilde L_i(\lambda) = \inf_{\lambda' \le \lambda} L_i(\lambda')$ or by a running minimum of $\hat R_n$ — and report empirical coverage as evidence. The monotonised loss satisfies CRC's hypotheses and the theorem applies **to $\tilde L$**, but $\tilde L \le L$, so the guarantee transfers in the wrong direction: $\mathbb{E}[\tilde L(\hat\lambda)] \le \alpha$ does not bound $\mathbb{E}[L(\hat\lambda)]$. There is no published characterisation of the resulting slack. Reported "it holds empirically" numbers are benchmark numbers only.

**Absent.** No expectation-form theorem for non-monotone $L$ with a complexity term better than $\log m$.

## 4. What Is Known

- CRC's inflation term is exactly $B/(n+1)$ and is tight; at $n = 1000$, $B = 1$ it costs $0.001$ of risk budget. Measured across the ICLR 2024 paper's tasks (MS-COCO multi-label, ~$5{,}000$ calibration images; hierarchical ImageNet; MIMIC-CXR report generation), empirical risk sits at or just below $\alpha$ with no visible slack.
- The Hoeffding–Bentkus $p$-value at $n = 1000$, $\alpha = 0.1$, $\delta = 0.1$ certifies $\lambda$ when $\hat R_n \lesssim 0.075$ — a $25\%$ relative haircut before any multiplicity correction.
- Bonferroni over $m$ grid points adds $\sqrt{\log(m)/(2n)}$ to the required margin under Hoeffding: at $n = 1000$, going from $m = 1$ to $m = 200$ costs about $0.052$ of extra margin. Fixed-sequence testing removes this **only** when the ordering is informative, which is precisely what Pareto Testing supplies and what non-monotone risk curves make hard.
- For non-monotone $R$, the feasible set is a union of intervals; on the standard MS-COCO $1 - F_1$ curve it is a single bounded interval, not a half-line, so "the smallest valid $\lambda$" and "the largest" differ in mean set size by more than $2\times$ at $\alpha$ near the curve's minimum (measured on 5k held-out images).
- Selection bias is quantified in general: for $m$ grid points with per-point standard error $s$, the risk at the empirical argmin exceeds its estimate by up to $s\sqrt{2\log m}$ in the worst case. This is the mechanism by which naive CRC-on-non-monotone-loss fails; it is a classical winner's-curse bound, not a conformal result.

## 5. What Is Not Known

- **Theoretically open.** Whether any expectation-form guarantee $\mathbb{E}[L(\hat\lambda)] \le \alpha$ is achievable for arbitrary bounded non-monotone $L$ without a $\log m$ (or $\log$ covering-number) penalty. Also open: a lower bound showing the penalty is *necessary*. Neither direction has a proof. Also open: any bound on the slack introduced by pathwise monotonisation $\tilde L = \inf_{\lambda' \le \lambda} L$ in terms of the number of sign changes of $R'$.
- **Empirically open.** How much of the LTT margin is recoverable in practice by shape-aware procedures (unimodality-constrained isotonic-type estimators, or confidence bands for $R(\cdot)$ from Waudby-Smith–Ramdas betting martingales) on realistic $n \in [10^3, 10^4]$. The experiment is cheap; nobody has run it as a head-to-head across a fixed suite.
- **Methodologically blocked.** Which feasible $\lambda$ to return, and how to score a method that returns a *set* of valid $\lambda$. With a half-line feasible set the answer is "the boundary". With a union of intervals, efficiency ($|\mathcal{C}_\lambda|$), cost, and robustness to shift disagree, and no benchmark fixes the objective. Until this is fixed, cross-paper efficiency comparisons on non-monotone losses are not commensurable.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the feasible set from a single calibration split, combined with post-selection bias that monotonicity would otherwise neutralise**.

Under monotonicity the map $\lambda \mapsto \hat R_n(\lambda)$ is itself monotone, so "first crossing of $\alpha$" is a *stopping time* in $\lambda$, and CRC's proof is a leave-one-out exchangeability argument that never unions over $\Lambda$ — the search costs nothing. Remove monotonicity and the crossing is no longer a stopping rule: the selected $\lambda$ can sit in a downward noise excursion anywhere on the grid, and the estimate at the selected point is biased low by order $s\sqrt{2\log m}$. Every known repair pays for the search explicitly (union bound, FWER graph, sample splitting), and each payment is $O(\sqrt{\log m / n})$, which for $n \sim 10^3$ is comparable to $\alpha$ itself.

This is not a compute problem — the grids are small and evaluation is one forward pass per $\lambda$. It is a statistical one: the information needed to certify *which* dip in $\hat R_n$ is real is exactly the information the calibration set does not have.

## 7. Current Research (as of 2026)

- **Ordering-based multiplicity reduction.** Pareto Testing and successors: use a split to rank candidate $\lambda$, then fixed-sequence test. Extension to non-monotone multi-objective frontiers is active *(frontier — verify)*.
- **Betting-based confidence sequences.** Waudby-Smith & Ramdas (*JRSS-B* 2024) estimators are anytime-valid and empirically tighter than Hoeffding–Bentkus at small $n$; plugging them into LTT gives uniform-over-$\lambda$ bands cheaply. Ramdas' group (CMU) and the Berkeley conformal group (Angelopoulos, Bates, Barber) are the natural loci.
- **Structure-restricted risk curves.** Assuming $R$ unimodal or has $\le k$ sign changes should give $\log k$ rather than $\log m$ penalties; the shape-constrained inference literature has the tools, and no conformal paper has applied them at the time of writing *(frontier — verify)*.
- **LLM applications forcing the issue.** Prompt Risk Control (Zollo, Morrill, Deng, Snell, Pitassi, Zemel, *ICLR* 2024) selects prompts under non-monotone risk and uses LTT machinery — the applied driver for making this cheaper.

## 8. Concrete Next Experiment

**Question.** Does shape-constrained calibration beat Bonferroni-LTT on genuinely non-monotone risk, at a fixed validity level?

- **Scale.** MS-COCO multi-label with a fixed backbone; $n = 2000$ calibration, $3000$ test, $m = 200$ thresholds; loss $L = 1 - F_1$ of the predicted label set, $B = 1$, $\alpha = 0.30$, $\delta = 0.10$. Repeat over $500$ random calibration/test splits. One GPU-day total.
- **Arms.** (A) *Control:* Bonferroni-LTT with Hoeffding–Bentkus $p$-values. (B) Naive CRC applied directly to the non-monotone loss (expected to be invalid — this arm measures the size of the violation). (C) Monotonised CRC on $\tilde L = \inf_{\lambda' \le \lambda} L$. (D) Unimodality-constrained upper confidence band for $R(\cdot)$, uniform at level $1 - \delta$.
- **Deciding number.** Mean test $1 - F_1$ at $\hat\lambda$, and the **split-level violation rate** $\widehat{\mathbb{P}}(R(\hat\lambda) > 0.30)$ estimated from the 500 splits against the held-out $R$ computed on the full test pool. Arm D wins iff its violation rate is $\le 0.10$ **and** its mean set size is at least $10\%$ smaller than arm A's. If arm B's violation rate exceeds $0.10$ by more than Monte-Carlo error ($\pm 0.013$), the "just monotonise it" folklore is falsified in print, which alone is worth the run.

## 9. Key References

- **[Foundational]** Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[Foundational]** Bates, Angelopoulos, Lei, Malik, Jordan. *Distribution-Free, Risk-Controlling Prediction Sets.* Journal of the ACM, 2021. — arXiv:2101.02703
- **[SOTA]** Angelopoulos, Bates, Fisch, Lei, Schuster. *Conformal Risk Control.* ICLR, 2024. — arXiv:2208.02814
- **[SOTA]** Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2021. — arXiv:2110.01052
- **[SOTA]** Laufer-Goldshtein, Fisch, Barzilay, Jaakkola. *Efficiently Controlling Multiple Risks with Pareto Testing.* ICLR, 2023.
- **[Related]** Snell, Zollo, Deng, Pitassi, Zemel. *Quantile Risk Control: A Flexible Framework for Bounding the Probability of High-Loss Predictions.* ICLR, 2023.
- **[Related]** Deng, Zollo, Snell, Pitassi, Zemel. *Distribution-Free Statistical Dispersion Control for Societal Applications.* NeurIPS, 2023.
- **[Related]** Zollo, Morrill, Deng, Snell, Pitassi, Zemel. *Prompt Risk Control: A Rigorous Framework for Responsible Deployment of Large Language Models.* ICLR, 2024.
- **[Tooling]** Waudby-Smith, Ramdas. *Estimating means of bounded random variables by betting.* Journal of the Royal Statistical Society Series B, 2024.
- **[Survey]** Angelopoulos, Barber, Bates. *Theoretical Foundations of Conformal Prediction.* Cambridge University Press, 2025. — arXiv:2411.11824
- **[Tooling]** Bentkus. *On Hoeffding's inequalities.* Annals of Probability, 2004.

## 10. Worked Example

Multi-label classification, threshold $\lambda$ on per-class scores, loss $1 - F_1$. Suppose the true risk curve on the grid is

| $\lambda$ | 0.90 | 0.70 | 0.55 | 0.50 | 0.45 | 0.30 | 0.10 |
|---|---|---|---|---|---|---|---|
| $R(\lambda)$ | 0.55 | 0.28 | 0.24 | 0.22 | 0.24 | 0.26 | 0.45 |
| mean set size | 1.1 | 2.4 | 3.1 | 3.5 | 4.0 | 6.2 | 21.7 |

Target $\alpha = 0.25$. The feasible set is the **interval** $\lambda \in [\approx 0.43, \approx 0.58]$ — not a half-line. CRC's assumption (iv), "$L$ at the largest $\lambda$ is below $\alpha$", fails in the direction the algorithm scans.

Take $n = 1000$, per-example loss sd $\approx 0.30$, so $s = 0.0095$. Run CRC anyway: scan $\lambda$ downward, stop at the first grid point with $\hat R_n + 1/(n+1) \le 0.25$.

- At $\lambda = 0.55$, $R = 0.24$: stops correctly on most draws. Fine.
- But on draws where noise pushes $\hat R_n(0.70)$ down, CRC stops at $R = 0.28$ — a $12\%$ overshoot of $\alpha$. Across $m = 200$ grid points the winner's-curse bias is $s\sqrt{2\log m} = 0.0095 \times 3.26 = \mathbf{0.031}$, so the *expected* risk at CRC's stopping point sits near $0.25 + 0.03$. The theorem does not apply, and the failure is not asymptotically negligible: it shrinks as $\sqrt{\log m/n}$, the same rate as the correction CRC was supposed to avoid paying.

Now the valid alternative. Bonferroni-LTT at $\delta = 0.1$, $m = 200$: per-hypothesis level $5 \times 10^{-4}$, Hoeffding margin $\sqrt{\log(1/5{\times}10^{-4})/(2 \cdot 1000)} = \sqrt{7.60/2000} = \mathbf{0.062}$. LTT certifies only $\lambda$ with $\hat R_n \le 0.188$. On this curve the minimum of $R$ is $0.22 > 0.188$, so **LTT returns the empty set** — no threshold is certifiable at $n = 1000$, and the practitioner gets no predictor at all.

That is the obstruction in one table. The invalid method is off by $0.031$; the valid method needs $0.062$ of headroom and here returns nothing. The gap between $0.031$ and $0.062$ — a factor of two in required calibration size, $n$ from $1000$ to $4000$ — is exactly the space a shape-aware procedure would have to occupy, and no published method occupies it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*