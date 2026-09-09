---
id: 10-scaling-laws/irreducible-loss-term-identifiability
title: "Irreducible Loss Term Identifiability"
topic: 10-scaling-laws
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Irreducible Loss Term Identifiability

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/irreducible-loss-term-identifiability` · **Status:** methodologically-blocked

## 1. Problem Statement

Modern neural scaling laws are fitted as an additive decomposition of held-out loss into a floor plus decaying terms:

$$L(N, D) \;=\; E \;+\; \frac{A}{N^{\alpha}} \;+\; \frac{B}{D^{\beta}}$$

with $N$ parameters and $D$ training tokens. The constant $E$ — the "irreducible loss" — is read in practice as the entropy of the data distribution: the loss no amount of compute can remove. Every downstream use of the law depends on it. Compute-optimal $N{:}D$ ratios, the predicted loss at $10\times$ budget, and claims that a model is "near the entropy floor" are all functions of $E$.

The problem: **is $E$ identifiable from the data actually collected, and does the fitted $E$ measure the quantity it is named after?**

Three variants, of increasing difficulty:

- **Measurement.** Given a fixed grid of runs over a range $[N_{\min}, N_{\max}] \times [D_{\min}, D_{\max}]$ and measured losses with known noise, produce an honest confidence interval on $E$. Solving this means: a published $E$ with a calibrated interval, plus a stated range condition under which the interval is finite.
- **Method.** Design a fitting or experimental protocol whose $E$ estimate is stable under changes that should not move it (seed, optimizer schedule, held-out split) and moves correctly under changes that should (corpus, tokenizer).
- **Theory.** Prove or refute that the three-term additive form is asymptotically correct, i.e. that $\lim_{N,D\to\infty} L(N,D)$ exists and equals the conditional entropy rate of the data source under the tokenizer. If the residual is not a pure constant — e.g. $\log N$ corrections or a second power law — then $E$ is a nuisance parameter, not a physical one.

## 2. Formal Setting

Let $\mathcal{D}$ be a distribution over token sequences from vocabulary $\mathcal{V}$ induced by a corpus $C$ and a tokenizer $T$. A model $p_\theta$ with $N$ non-embedding parameters is trained on $D$ tokens.

**Measured loss.** $\hat{L}$ is the mean next-token negative log-likelihood, in nats/token, on a held-out split $S$:

$$\hat{L}(N,D) \;=\; -\frac{1}{|S|}\sum_{(x_{<t},x_t)\in S} \log p_\theta(x_t \mid x_{<t})$$

evaluated at the **end** of a schedule whose cosine (or WSD) decay completes exactly at $D$. Intermediate checkpoints of a longer run are *not* interchangeable with completed shorter runs — using them biases $\beta$ upward and $E$ downward.

**Entropy floor.** The quantity $E$ is intended to be the conditional entropy rate

$$H_T(\mathcal{D}) \;=\; \lim_{t\to\infty} \mathbb{E}\left[-\log \Pr(x_t \mid x_{<t})\right],$$

which depends on $T$: dividing by mean bytes-per-token $\bar{b}$ converts to nats/byte and is the only cross-tokenizer-comparable form.

**Fit.** Estimate $\phi = (E, A, B, \alpha, \beta)$ by minimizing a Huber loss on log-predictions over $K$ runs, as in Hoffmann et al. Identifiability is governed by the Jacobian $J_{ij} = \partial \log L_i / \partial \phi_j$. The pathology is that $\partial L/\partial E = 1$ and $\partial L/\partial A = N^{-\alpha}$ become near-collinear when $N^{-\alpha}$ varies little across the grid — i.e. when $\alpha \log(N_{\max}/N_{\min})$ is small. The Fisher information then has a near-zero eigenvalue along a direction mixing $E$, $A$ and $\alpha$.

**Assumptions, and which are violated.**

| Assumption | Status |
|---|---|
| Loss additively separable in $N$ and $D$ | Violated: interaction terms appear in data-repetition regimes (Muennighoff et al. 2023) |
| Single power law over the whole range | Violated: break points reported in vision and translation (Caballero et al. 2023; Ghorbani et al. 2021) |
| Optimizer/LR tuned at every grid point | Usually violated; Porian et al. 2024 show untuned LR at small $N$ shifts the fitted exponents |
| $E$ independent of $N,D$ | Untestable within range — this *is* the problem |
| Held-out split independent of training data | Violated by web-scale near-duplicates; $H_T$ for a fixed corpus is then not well defined |
| Noise homoscedastic in $\log L$ | Approximately true, but seed variance grows at small $N$ |

## 3. State of the Art

**Empirical SOTA (established).** The three-term form of §1 with Huber-loss fitting is the standard, from Hoffmann et al. (NeurIPS 2022), building on the reducible/irreducible split introduced by Henighan et al. (2020) and the constructive error surface of Rosenfeld et al. (ICLR 2020). Reported Chinchilla values: $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$, over $\sim$400 runs from 70M to 16B parameters.

**Established critique.** Besiroglu et al. (2024) refit Hoffmann's published loss table and obtained $E=1.82$, $A=482$, $B=2085$, $\alpha=0.35$, $\beta=0.37$, and showed the original paper's reported confidence intervals on the exponents were implausibly narrow given the same data. This is a genuine reproduction with released data, not a claim.

**Claimed but unablated.** That $E$ equals a data entropy. No paper anchors a fitted $E$ against an independently known entropy rate. The claim rests entirely on the functional form's name.

**Theory SOTA.** Bahri et al. (PNAS 2024) derive power-law exponents in variance-limited and resolution-limited regimes; Sharma & Kaplan relate $\alpha$ to intrinsic data-manifold dimension; Hutter (2021) gives an exactly solvable learning-curve model. None of these derives $E$ as a separately identifiable constant — in each, the floor is *imposed* as the Bayes risk rather than recovered from a finite-range fit.

**Benchmark-number-only.** Published "bits per byte near the entropy of English" comparisons are single numbers with no interval and no cross-tokenizer normalization stated.

## 4. What Is Known

- Fitted $E$ moves by $0.13$ nats/token ($1.69 \to 1.82$, $\approx 8\%$) under re-estimation on *identical* data with a different optimizer/uncertainty treatment (Besiroglu et al. 2024, 70M–16B scale).
- Kaplan-vs-Chinchilla exponent disagreement ($\alpha$-driven $N\propto C^{0.73}$ vs $C^{0.50}$) is fully explained by three procedural factors: counting embedding parameters, LR-schedule length mismatch, and untuned LR at small scale (Porian et al., NeurIPS 2024, runs up to $\sim$900M).
- Number of grid points needed for a usable fit is small — Choshen et al. (2024), over 485 pretrained models, report that $\sim$5 model sizes suffice to predict a target loss to a few percent relative error. Predicting *loss in range* is easy; recovering $E$ is not the same task.
- Loss curves for a fixed architecture family are smooth and monotone in $N$ and $D$ across four decades of compute (Hestness et al. 2017; Kaplan et al. 2020), i.e. the fit residuals are small — which is exactly why many $(E,A,\alpha)$ triples fit equally well.
- Independent entropy anchors for English exist but are not commensurate: Shannon (1951) estimated $\sim$0.6–1.3 bits/character; Brown et al. (1992) gave an upper bound of 1.75 bits/character on a word-level corpus. Both are human/model-specific upper bounds, not the entropy rate of a web corpus under a BPE tokenizer.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no operational definition of the target. $H_T(\mathcal{D})$ for a real corpus is not measurable by any known procedure, and for a fixed finite corpus with train/test near-duplication it is not even well defined. Without ground truth, no fitting method can be validated — only compared to other fitting methods.
- **Theoretically open.** Whether $L(N,D) - E$ is asymptotically a pure sum of two power laws, or carries log-corrections / additional break points. If the latter, $E$ is not a limit of anything. No proof either way for transformers on natural text.
- **Empirically open.** Whether a pre-registered leave-the-top-decade-out protocol yields an $E$ interval narrow enough to be decision-relevant. The experiment is runnable at $\sim$$10^{21}$ FLOPs (§8); nobody has published it with intervals.
- **Open.** Whether $E$ is a property of the data alone, or partly of the architecture class. Current practice assumes the former; a single experiment fitting $E$ for two architecture families on identical tokenized data would test it and has not been reported with intervals.

## 6. Why It Is Hard

Two obstructions, both specific.

**Non-identifiability over the accessible range.** $E$ and $(A,\alpha)$ trade off along a near-null direction of the Fisher information. The curvature that separates a constant from a slowly decaying power law lives in the *far* tail; over two decades of $N$ the residual signature of a $0.05$-nat error in $E$ is $\sim$0.01 nats at mid-range (§10) — the same order as seed-to-seed spread. Widening the range is the only fix, and range costs compute superlinearly.

**Absent ground truth.** Even a perfectly identified $E$ cannot be checked. There is no corpus for which the conditional entropy rate under a given tokenizer is known, so the naming of $E$ as "entropy" is an interpretation, never a validated measurement. This is what makes the problem methodologically blocked rather than merely expensive.

## 7. Current Research (as of 2026)

- **Estimation hygiene.** Choshen, Zhang & Andreas (MIT); Porian, Carmon et al. (Technion / Stanford / LAION) — protocol standardization, LR tuning per grid point, released fit corpora.
- **Replication and uncertainty.** Epoch AI (Besiroglu, Erdil) — refits with bootstrap intervals on published loss tables.
- **Functional-form search.** Caballero, Krueger et al. — smoothly-broken laws that make $E$ one parameter among many, weakening its interpretation.
- **Mechanistic floors.** Michaud, Tegmark et al. — quantization/discrete-skill models that predict a power law from a discrete task distribution and imply the floor is a truncation artifact *(frontier — verify)*.
- **Synthetic sources with known entropy.** Scattered use of HMM and formal-language sources to validate scaling-law fitting where $H$ is analytic. Not yet done at the scale needed to test $E$ recovery *(frontier — verify)*.

## 8. Concrete Next Experiment

**Recover $E$ where its true value is known, then measure how much range that took.**

- **Scale.** Two arms, both trained to completed cosine schedules with per-point LR tuning. Arm A: synthetic data from an order-$k$ hidden Markov source with analytically computed entropy rate $H^\star$ (choose $H^\star = 1.50$ nats/token), tokenized identically to Arm B. Arm B: a deduplicated web corpus. Grid: $N \in \{3\!\times\!10^7, 10^8, 3\!\times\!10^8, 10^9, 3\!\times\!10^9\}$ (2 decades), $D$ at 5 ratios spanning $10\times$, 3 seeds each — 75 runs. Largest run $\approx 6ND = 1.1\times10^{21}$ FLOPs; total $\approx 3\times10^{21}$ FLOPs, roughly 2,000–3,000 H100-hours.
- **Control arm.** Arm A is the control: $H^\star$ is known in closed form, so the fitted $\hat{E}$ has a ground truth. Additionally hold out the top decade ($N = 10^{10}$, one seed) from fitting and use it only for extrapolation checking.
- **Deciding number.** The width of the 95% bootstrap CI on $\hat{E}$ from the lower two decades, in nats/token, together with $|\hat{E} - H^\star|$ on Arm A. **Decision rule:** if the CI width exceeds $0.15$ nats while the in-range fit RMSE is below $0.01$ nats, $E$ is non-identifiable at two decades and every published point estimate of $E$ should be reported as an interval. If the CI width is below $0.05$ nats *and* covers $H^\star$ on Arm A, the fit is sound and the blockage is only about corpus entropy, not about estimation.

## 9. Key References

- **[Foundational]** J. Hestness, S. Narang, N. Ardalani, et al. *Deep Learning Scaling is Predictable, Empirically.* arXiv, 2017. — arXiv:1712.00409
- **[Foundational]** J. S. Rosenfeld, A. Rosenfeld, Y. Belinkov, N. Shavit. *A Constructive Prediction of the Generalization Error Across Scales.* ICLR, 2020. — arXiv:1909.12673
- **[Foundational]** J. Kaplan, S. McCandlish, T. Henighan, et al. *Scaling Laws for Neural Language Models.* arXiv, 2020. — arXiv:2001.08361
- **[Foundational]** T. Henighan, J. Kaplan, M. Katz, et al. *Scaling Laws for Autoregressive Generative Modeling.* arXiv, 2020. — arXiv:2010.14701
- **[SOTA]** J. Hoffmann, S. Borgeaud, A. Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** T. Besiroglu, E. Erdil, M. Barnett, J. You. *Chinchilla Scaling: A Replication Attempt.* arXiv, 2024. — arXiv:2404.10102
- **[SOTA]** T. Porian, M. Wortsman, J. Jitsev, L. Schmidt, Y. Carmon. *Resolving Discrepancies in Compute-Optimal Scaling of Neural Networks.* NeurIPS, 2024. — arXiv:2406.19146
- **[SOTA]** L. Choshen, Y. Zhang, J. Andreas. *A Hitchhiker's Guide to Scaling Law Estimation.* arXiv, 2024. — arXiv:2410.11840
- **[Theory]** Y. Bahri, E. Dyer, J. Kaplan, J. Lee, U. Sharma. *Explaining Neural Scaling Laws.* PNAS, 2024. — arXiv:2102.06701
- **[Theory]** M. Hutter. *Learning Curve Theory.* arXiv, 2021. — arXiv:2102.04074
- **[Form]** E. Caballero, K. Gupta, I. Rish, D. Krueger. *Broken Neural Scaling Laws.* ICLR, 2023. — arXiv:2210.14891
- **[Form]** I. Alabdulmohsin, B. Neyshabur, X. Zhai. *Revisiting Neural Scaling Laws in Language and Vision.* NeurIPS, 2022. — arXiv:2209.06640
- **[Data regime]** N. Muennighoff, A. M. Rush, B. Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Entropy anchor]** C. E. Shannon. *Prediction and Entropy of Printed English.* Bell System Technical Journal, 1951.
- **[Entropy anchor]** P. F. Brown, S. A. Della Pietra, R. L. Mercer, V. J. Della Pietra, J. C. Lai. *An Estimate of an Upper Bound for the Entropy of English.* Computational Linguistics, 1992.

## 10. Worked Example

Take the parameter-only slice of the Chinchilla fit, $L(N) = 1.69 + 406.4\,N^{-0.34}$, and evaluate over two decades:

```
N        reducible   L(N)
1e8      0.7751      2.4651
1e9      0.3536      2.0436
1e10     0.1613      1.8513
```

Now assume a different floor, $E' = 1.65$, and refit $A', \alpha'$ to pass exactly through the endpoints $N=10^8$ and $N=10^{10}$:

$$\alpha' = \frac{\log(0.8151/0.2013)}{\log 100} = 0.304, \qquad A' = 0.8151 \cdot (10^{8})^{0.304} = 220.1$$

At the midpoint $N=10^9$ this curve gives $L' = 1.65 + 220.1\cdot(10^9)^{-0.304} = 2.0554$, against the true $2.0436$. **Mismatch: 0.0118 nats.** Repeating for $E' = 1.60$ gives a midpoint mismatch of $0.023$ nats; for $E' = 1.40$, $0.050$ nats.

The obstruction is now numeric. Run-to-run spread in end-of-schedule held-out loss at fixed configuration is of order $0.01$ nats (*assumed*, from typical reported ablation variation) and grows at small $N$. So a $\pm 0.04$-nat error in $E$ produces a residual signature at or below the noise floor across two decades of $N$ — the fit cannot distinguish it. Only when the assumed floor is wrong by $\sim$0.3 nats does the misfit clear the noise by $5\times$. That is consistent with the observed $0.13$-nat shift between Hoffmann's $E=1.69$ and Besiroglu's refit $E=1.82$: both are inside the non-identifiable band.

The second half of the obstruction: neither value can be checked. Converting $E=1.69$ nats/token at $\bar{b}\approx4.2$ bytes/token gives $0.402$ nats/byte $=0.58$ bits/byte. Shannon's 1951 estimates for printed English sit near $0.6$–$1.3$ bits/character, and Brown et al.'s 1992 upper bound is $1.75$ bits/character. The fitted floor lands *below* the low end of every independent estimate — plausible, since those are upper bounds on a different corpus with different preprocessing, but the comparison carries no information. There is no measurement that says $0.58$ bits/byte is right and $0.62$ is wrong. $E$ is a fitted nuisance constant with an entropy-sounding name.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*