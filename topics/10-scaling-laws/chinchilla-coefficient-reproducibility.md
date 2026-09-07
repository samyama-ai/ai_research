---
id: 10-scaling-laws/chinchilla-coefficient-reproducibility
title: "Chinchilla Coefficient Reproducibility"
topic: 10-scaling-laws
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Chinchilla Coefficient Reproducibility

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/chinchilla-coefficient-reproducibility` · **Status:** partially-solved

## 1. Problem Statement

Hoffmann et al. (2022) reported that compute-optimal language model training allocates parameters $N$ and tokens $D$ to a fixed FLOP budget $C$ as $N_{\mathrm{opt}} \propto C^{a}$, $D_{\mathrm{opt}} \propto C^{b}$ with $a \approx b \approx 0.5$, i.e. roughly 20 tokens per parameter. That single ratio has driven the token budgets of most public pretraining runs since.

The problem has three separable variants:

- **Measurement.** Do the published Chinchilla coefficients follow from the published data? Concretely: does the parametric fit (their "Approach 3") reproduce $\hat\alpha = 0.34$, $\hat\beta = 0.28$, $\hat E = 1.69$ from the loss values plotted in the paper, and are the reported confidence intervals attainable? This variant is **closed and the answer is no** (Besiroglu et al., 2024).
- **Method.** Are the three estimation procedures Hoffmann et al. used (minimum-over-training-curve, IsoFLOP profile, parametric Huber fit) mutually consistent, and is each identifiable from realistic amounts of data? Partially answered.
- **Theory.** Is there any reason the exponent should be $\approx 1/2$, or is $a = b$ an artifact of the loss surface being locally symmetric in $\log N$ and $\log D$ over the fitted range? Open.

Solving the problem means: an independently run, independently fitted, hyperparameter-controlled replication that reports $\hat a$ with a calibrated interval, and a stated set of conditions (tokenizer, data mixture, LR schedule, optimizer) under which that interval holds.

## 2. Formal Setting

Let a decoder-only transformer have $N$ non-embedding parameters, trained on $D$ tokens from distribution $\mathcal{P}$. Compute is measured by the standard approximation
$$C = 6ND \quad \text{FLOPs},$$
which counts forward + backward matmuls only and omits attention-quadratic terms; it is exact to within a few percent for $N \gtrsim 10^{8}$ at sequence length 2048, and drifts once context length is large relative to model width.

The loss $L$ is **cross-entropy in nats per token, on a held-out sample from the training distribution**, at the final step of a run whose LR schedule was annealed to completion. This last clause matters: an intermediate checkpoint of a cosine-scheduled run has systematically higher loss than a run of that length, so "loss at step $t$" and "loss of a $t$-step run" are different measurements.

The parametric model:
$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$
with $E$ the irreducible entropy, and the $A$-, $B$-terms the capacity and optimization deficits. Fitting minimizes a Huber loss on log-residuals:
$$\min_{a_0,b_0,e,\alpha,\beta}\ \sum_{i} \mathrm{Huber}_{\delta}\!\left(\mathrm{LSE}\big(a_0 - \alpha \log N_i,\ b_0 - \beta \log D_i,\ e\big) - \log L_i\right), \quad \delta = 10^{-3},$$
where $\mathrm{LSE}$ is log-sum-exp and $A = e^{a_0}$, $B = e^{b_0}$, $E = e^{e}$. Minimizing $L$ subject to $6ND = C$ gives
$$a = \frac{\beta}{\alpha+\beta}, \qquad b = \frac{\alpha}{\alpha+\beta}, \qquad a + b = 1 .$$

Assumptions, and their status:

1. **$a+b=1$ exactly.** Forced by the functional form, not measured. Violated if the true surface has a cross term in $N$ and $D$.
2. **Hyperparameters are optimal at every $(N,D)$.** Known violated. Chinchilla's own runs used a cosine schedule whose length was set per-run; learning rate and batch size were tuned on a grid, not per-point.
3. **Loss is a scalar sufficient statistic for model quality.** Violated for downstream tasks; $a$ estimated from loss need not be $a$ for benchmark accuracy.
4. **Single-epoch, fixed data distribution.** Violated at $D \gg$ corpus size (Muennighoff et al., 2023).
5. **$C = 6ND$.** Approximation; embedding parameters excluded, which is one of the documented sources of Kaplan/Chinchilla divergence.

## 3. State of the Art

**Established.**

- Hoffmann et al. (2022, NeurIPS) — over 400 runs, $70$M–$16$B parameters, $5$B–$500$B tokens; three approaches giving $a = 0.50$, $0.49$, $0.46$. The 70B Chinchilla model trained on 1.4T tokens beat the 280B Gopher on a broad benchmark suite, confirming the *direction* of the correction independently of the exponent's exact value.
- Besiroglu, Erdil, Barnett, You (2024), *Chinchilla Scaling: A replication attempt* — extracted the Approach 3 loss values from the paper's Figure 4 via plot digitization and refit. The refit does **not** reproduce the published $(\alpha,\beta,E)$; it yields roughly $\alpha \approx 0.35$, $\beta \approx 0.37$, $E \approx 1.82$, implying $a \approx 0.51$ rather than $0.46$. Separately, the published confidence interval on $a$, $[0.454, 0.455]$, is narrower than any resampling of ~245 noisy points can support; the refit interval is wide enough to contain $0.5$ comfortably. DeepMind acknowledged an error in the reported Approach 3 numbers.
- Porian, Wortsman, Jitsev, Schmidt, Carmon (2024, NeurIPS), *Resolving Discrepancies in Compute-Optimal Scaling of Neural Networks* — identifies three concrete causes for Kaplan et al.'s $a = 0.73$ versus Chinchilla's $0.5$: (i) counting embedding parameters in $N$, (ii) LR warmup too long relative to short small-scale runs, (iii) untuned/miscscaled batch size. Correcting all three moves the estimate to $\approx 0.5$. This is the strongest existing *mechanistic* reproduction result.

**Claimed but unablated.**

- Vendor scaling-law papers (e.g. DeepSeek LLM, 2024) report compute-optimal exponents materially away from $0.5$ on their own data mixtures, and attribute the shift to data quality. The attribution is a claim; no controlled swap of mixtures at fixed everything-else has been published.
- Token-per-parameter ratios of 100–1000 used in current production models are inference-cost arguments (Sardana et al., 2024), not claims that $a \neq 0.5$.

**Benchmark-number-only.** The Chinchilla-vs-Gopher downstream comparison is a single paired result at one scale; it does not constrain $a$.

## 4. What Is Known

- Chinchilla's Approaches 1 and 2 give $a = 0.50$ and $a = 0.49$; only Approach 3 ($0.46$) fails to reproduce. Scale: $70$M–$16$B params, up to $\sim 10^{24}$ FLOP.
- The $\approx 20$ tokens/param rule holds only near $C \sim 10^{23}$ FLOP even under the paper's own fit; because $a$ and $b$ differ by $0.08$ in the published Approach 3, the implied ratio grows with $C$ — an artifact that vanishes when $a = b$.
- Refit optimum from digitized data (Besiroglu et al.): $\approx 20$ tokens/param, but with an interval spanning roughly a factor of two in $D/N$ at $10^{23}$ FLOP.
- Kaplan et al. (2020) measured $a = 0.73$ at $10^{3}$–$10^{9}$ params; the gap to $0.5$ is now attributed to the three artifacts above, not to a real regime change.
- Muennighoff et al. (2023, NeurIPS) — repeated data decays in value; up to ~4 epochs is near-free, and by ~16 epochs added compute is worth almost nothing. Measured at $\le 9$B params, $\le 900$B tokens.
- Hägele et al. (2024) — constant-LR-plus-cooldown schedules make IsoFLOP sweeps far cheaper by reusing a single trunk, and recover Chinchilla-consistent optima at $\le 2$B params.

## 5. What Is Not Known

- **Empirically open.** Whether an *independent* full replication — new runs, new corpus, tuned hyperparameters at every grid point — produces $a$ inside $[0.48, 0.52]$. Nobody has published one at $\ge 10^{23}$ FLOP with per-point tuning. Runnable; costs on the order of $10^{24}$ FLOP total.
- **Empirically open.** How much $a$ moves with data mixture, tokenizer, and optimizer (AdamW vs. Muon/Shampoo-class). Small-scale evidence is contradictory.
- **Methodologically blocked.** There is no agreed estimator or uncertainty model for $a$. Approaches 1–3 are three different estimators of three subtly different targets; the Huber-on-log-loss objective has no derived sampling distribution, and reported intervals are typically neither bootstrapped over runs nor over seeds. Choshen et al. (2024) show fitted exponents shift materially with which points are included.
- **Theoretically open.** No derivation of $\alpha \approx \beta$ from any model of transformer learning. Data-manifold and quantization-of-skills accounts (Sharma & Kaplan, 2022; Michaud et al., 2023) predict power laws but do not pin the ratio.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under measurement noise, compounded by unshared raw data**.

$L(N,D) = E + AN^{-\alpha} + BD^{-\beta}$ has a nearly flat likelihood ridge: $E$ trades against $A$ and $B$ almost freely, because over the fitted range $AN^{-\alpha}$ varies by only $\sim 0.1$–$0.3$ nats while $L \approx 2$–$3$ nats. Small changes in $E$ therefore swing $\alpha$ and $\beta$ by tens of percent while barely changing fit residuals. The Huber $\delta = 10^{-3}$ makes it worse: most residuals fall in the linear (outlier-robust) regime, so the effective sample size is far below the nominal point count, and the objective is non-convex with multiple local optima. Besiroglu et al.'s core finding is exactly this — a fit that looks visually fine yields a very different $a$.

Second obstruction: the ground truth is a *tuned* optimum, but tuning is per-point and expensive. Untuned points bias $\alpha$ upward at small $N$ (Porian et al.), which is precisely how the Kaplan/Chinchilla gap arose. Any replication that does not re-tune reproduces artifacts rather than exponents.

Third: the raw loss table for Chinchilla was never released. Replication had to proceed by digitizing a scatter plot — recovering ~245 points from pixels — which injects noise into a fit already ill-conditioned.

## 7. Current Research (as of 2026)

- **Epoch AI** (Besiroglu, Erdil, Barnett) continues scaling-law audit work, including refits of published laws and uncertainty quantification.
- **Estimator methodology.** Choshen, Zhang, Andreas et al., *A Hitchhiker's Guide to Scaling Law Estimation* (2024), assembles a large corpus of loss curves across model families and asks how many points/scales are needed for a given predictive error — the closest thing to a standard protocol.
- **Cheap sweep design.** Hägele et al. (EPFL/HuggingFace) on constant-LR + cooldown; makes per-point tuning affordable and is being adopted for IsoFLOP work.
- **Optimizer-dependent exponents.** *(frontier — verify)* Reports that second-order/orthogonalized optimizers shift the compute-optimal frontier; whether they shift $a$ or only the multiplicative constant is unresolved.
- **Inference-aware allocation.** Sardana et al. (2024) reframe the objective to include serving cost, which changes the *decision* even if $a = 0.5$ holds.

## 8. Concrete Next Experiment

**Question decided:** is $a = 0.5$ recoverable from a clean, tuned IsoFLOP sweep, and how wide is the honest interval?

- **Scale.** Six IsoFLOP bands at $C \in \{3\times10^{19}, 10^{20}, 3\times10^{20}, 10^{21}, 3\times10^{21}, 10^{22}\}$ FLOP, 8 model sizes per band ($N$ from $10^{7}$ to $3\times10^{9}$), 3 seeds at the two smallest bands to estimate run-to-run noise. Total $\approx 2\times10^{22}$ FLOP — roughly one week on 256 H100s.
- **Treatment arm.** Learning rate and batch size tuned *per grid point* by a short proxy search; constant-LR trunk with a 20%-of-budget cooldown so each $(N,D)$ point is a properly annealed run.
- **Control arm.** Identical grid, hyperparameters set by a single global scaling rule fit at the smallest band and extrapolated (the standard practice, and what Kaplan et al. effectively did). Embedding parameters excluded from $N$ in both arms; a third mini-arm includes them to re-measure that artifact's size.
- **Analysis.** Fit $a$ three ways (IsoFLOP parabola vertex, parametric Huber fit, minimum-over-curve), each with a run-level bootstrap over seeds and a leave-one-band-out check.
- **Deciding number.** The width and location of the bootstrap 90% interval on $a$ in the tuned arm. If it is contained in $[0.47, 0.53]$, the Chinchilla exponent is reproducible and the published Approach 3 value was an arithmetic error, full stop. If the interval width exceeds $0.10$, the exponent is not identifiable at this compute scale and every downstream token-budget claim derived from it is over-precise. The tuned-minus-control difference in $\hat a$ measures how much of any residual disagreement is hyperparameter artifact.

Release the full $(N, D, L)$ table. That single artifact would have prevented the original controversy.

## 9. Key References

- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[SOTA / replication]** Besiroglu, Erdil, Barnett, You. *Chinchilla Scaling: A Replication Attempt.* Epoch AI, 2024. — arXiv:2404.10102
- **[SOTA / reconciliation]** Porian, Wortsman, Jitsev, Schmidt, Carmon. *Resolving Discrepancies in Compute-Optimal Scaling of Neural Networks.* NeurIPS 2024.
- **[Methodology]** Choshen, Zhang, Andreas, et al. *A Hitchhiker's Guide to Scaling Law Estimation.* 2024.
- **[Related]** Muennighoff, Rush, Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[Related]** Hägele, Bakouch, Kosson, et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS 2024.
- **[Related]** Sardana, Portes, Young, Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML 2024.
- **[Theory]** Michaud, Liu, Girit, Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS 2023.

## 10. Worked Example

Take the published Chinchilla Approach 3 parameters and ask what they imply at a budget most labs care about, $C = 10^{25}$ FLOP.

With $\alpha = 0.34$, $\beta = 0.28$:
$$a = \frac{0.28}{0.62} = 0.452, \qquad b = 0.548 .$$
With the Besiroglu refit, $\alpha \approx 0.35$, $\beta \approx 0.37$:
$$a = \frac{0.37}{0.72} = 0.514, \qquad b = 0.486 .$$

Both are anchored near 20 tokens/param at $C_0 = 10^{23}$ FLOP, so write the ratio as
$$\frac{D}{N}(C) = 20 \left(\frac{C}{C_0}\right)^{b-a}.$$

- Published fit: $b - a = 0.096$, and $(10^{2})^{0.096} = 1.55$ → **31 tokens/param**, i.e. at $10^{25}$ FLOP a 230B model on 7.2T tokens.
- Refit: $b - a = -0.028$, $(10^{2})^{-0.028} = 0.88$ → **17.6 tokens/param**, i.e. a 310B model on 5.4T tokens.

Two orders of magnitude of extrapolation from the same experiment turn a 0.06 disagreement in an exponent into a **35% difference in the recommended parameter count** — and the two fits differ in *sign*: one says the token ratio should grow with compute, the other says it should shrink. The obstruction is visible here: the fits are near-indistinguishable on the data they were fit to (residuals differ in the third decimal of nats), and no measurement in the $70$M–$16$B range separates them. The disagreement is not about the data; it is about a ridge in the likelihood that the data never resolved, and which was reported as a $\pm 0.0005$ interval.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*