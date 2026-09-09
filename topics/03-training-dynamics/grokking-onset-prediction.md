---
id: 03-training-dynamics/grokking-onset-prediction
title: "Grokking Onset Time Prediction"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Grokking Onset Time Prediction

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/grokking-onset-prediction` · **Status:** open

## 1. Problem Statement

Grokking is delayed generalization: training accuracy saturates near 100% at step $t_m$, test accuracy stays at chance for a long interval, then rises sharply at step $t_g \gg t_m$ (Power et al., 2022).

The problem: **given only the first $T_{\text{obs}}$ steps of a run, predict $t_g$.**

- **Input:** the run configuration $c$ (architecture, optimizer, weight decay $\lambda$, initialization scale $\alpha$, dataset fraction $p$) and the observed prefix — losses, accuracies, weight norms, gradients — for $t \le T_{\text{obs}}$, where $T_{\text{obs}} \ll t_g$ and typically $T_{\text{obs}} \approx t_m$.
- **Output:** $\hat t_g$, or the decision predicate $\mathbb{1}[t_g < B]$ for a compute budget $B$.
- **Solved** = a predictor whose held-out error on $\log_{10} t_g$ is close to the seed-noise floor, on tasks not in its fitting set.

Three variants, different difficulty:

- **Measurement.** Define $t_g$ so it is a property of the run, not of the eval grid or the threshold. Currently informal.
- **Method.** Fit a predictor from prefix statistics. Runnable today; nobody has run it as a clean forecasting benchmark.
- **Theory.** Derive $t_g$ from $(\lambda, \alpha, p, \text{width})$ for a model class where grokking is provable. Solved only for modular addition on two-layer nets and for linear/lazy-to-rich toy models.

## 2. Formal Setting

Train $f_\theta$ on $S = \{(x_i,y_i)\}_{i=1}^n$ drawn from $\mathcal{D}$, with $n = p\,|\mathcal{X}|$ for algorithmic tasks where $\mathcal{X}$ is the full input set. Let $A_{\text{tr}}(t)$, $A_{\text{te}}(t)$ be train and test accuracy at optimizer step $t$, measured on a fixed eval grid $\mathcal{G} \subset \mathbb{N}$.

Onset times, at thresholds $\tau_m, \tau_g$ (conventionally $0.99$ and $0.90$ for modular arithmetic):

$$t_m = \min\{t \in \mathcal{G} : A_{\text{tr}}(t) \ge \tau_m\}, \qquad t_g = \min\{t \in \mathcal{G} : A_{\text{te}}(t) \ge \tau_g\},$$

and the **grokking gap** $G = \log_{10}(t_g / t_m)$. Grokking is conventionally declared when $G \gtrsim 1$.

The prediction task is a map $F$ with error

$$\varepsilon = \big|\log_{10} \hat t_g - \log_{10} t_g\big|, \qquad \hat t_g = F\big(c,\ \{h(t)\}_{t \le T_{\text{obs}}}\big),$$

where $h(t)$ is any per-step observable: $\|\theta_t\|_2$, $\|\nabla \mathcal{L}\|$, train loss, the Fourier-restricted loss of Nanda et al. (2023), or the last-layer feature-kernel change $\|K_t - K_0\|_F$ used by Kumar et al. (2024).

**How each quantity is actually measured.** $t_g$ is read off a *log-spaced* grid $\mathcal{G}$ in almost every published run, so its resolution is multiplicative: with 20 points per decade, $t_g$ is known to $\pm 5\%$ in log space at best. $A_{\text{te}}$ is a finite-sample estimate with binomial noise; near $\tau_g$ the accuracy curve is steep, which helps, but with $|\mathcal{X}\setminus S| \approx 10^4$ the noise is $\sim 0.3\%$ and negligible. $\lambda$ is measured as the *decoupled* AdamW coefficient, not $L_2$ added to the loss — the two give different $t_g$ scaling.

**Assumptions, and which are violated.**

1. *A single sharp transition exists.* Violated on real data: Humayun et al. (2024) report delayed robustness (grokking in adversarial robustness) long after test accuracy has saturated, so $t_g$ depends on which metric names "generalization."
2. *$t_g$ is a deterministic function of the config.* Violated. Seed-to-seed spread in $t_g$ is routinely a factor of 2–3 on modular arithmetic.
3. *$t_g$ is finite.* Violated at small $p$: below the critical dataset size $D_{\text{crit}}$ of Varma et al. (2023), the generalizing circuit never wins, so $t_g = \infty$ and the regression target is undefined.
4. *The transition is a property of the model, not of numerics.* Prieto et al. (2025) show softmax-collapse floating-point effects drive part of late-phase grokking dynamics.

## 3. State of the Art

**Established.**

- *Mechanistic progress measures.* Nanda et al. (ICLR 2023) define restricted and excluded loss on a 1-layer transformer doing addition mod 113; both move monotonically during the apparently flat "memorization" plateau, showing the plateau is not flat internally. Reproduced independently.
- *Grokking is not specific to algorithmic data.* Liu et al., *Omnigrok* (ICLR 2023): scaling the initialization to a large norm induces grokking on MNIST (1k examples), IMDB, and QM9; shrinking it removes the delay. The control (small init, same data) is run.
- *Circuit-efficiency account.* Varma et al. (2023) predict and observe **ungrokking** (test accuracy falling back after shrinking the dataset below $D_{\text{crit}}$) and **semi-grokking** (a stable intermediate test accuracy near $D_{\text{crit}}$). Both were novel predictions confirmed post hoc.
- *Provable grokking.* Lyu et al. (ICLR 2024) prove, for homogeneous nets with large initialization and small weight decay, an early kernel-regime phase followed by a late margin-maximization phase, giving grokking. Mohamadi et al. (ICML 2024) prove for modular addition with a two-layer network that memorization needs $\Theta(1)$ examples per class while generalization needs a constant fraction, separating the two timescales.

**Claimed but unablated / benchmark-only.**

- Notsawo et al. (2023), *Predicting Grokking Long Before it Happens*, is the only direct attack on this page's problem: oscillations in the early training-loss spectrum correlate with later grokking. Reported on modular arithmetic; there is no held-out-task evaluation, no config-only control arm, and no reported error in dex.
- *Grokfast* (Lee et al., 2024) low-pass-filters gradients and reports up to $\sim 50\times$ reduction in $t_g$. This is a benchmark number on a handful of tasks; it accelerates grokking but does not predict it, and the reported speedup is not decomposed against an equivalent effective-learning-rate control.
- Slingshot effects (Thilak et al., 2022): grokking is claimed to coincide with adaptive-optimizer norm-growth cycles. The correlation is shown; causal ablation is partial.

**Theory SOTA and empirical SOTA disagree.** No theory predicts a *number* $t_g$ for a transformer at the scales where grokking is actually measured; every quantitative $t_g$ in the literature is a measured value, not a predicted one.

## 4. What Is Known

- Power et al. (2022), 2-layer decoder transformer ($\sim$400k params), binary ops mod 97 ($97^2 = 9409$ pairs): at $p \approx 0.3$–$0.5$, $t_m \sim 10^3$ steps and $t_g \sim 10^5$–$10^6$ steps. $t_g$ rises steeply and roughly super-exponentially as $p$ falls toward the task-dependent floor ($\approx$ 30–40% for many ops).
- Nanda et al. (2023), 1-layer transformer, $d_{\text{model}}=128$, addition mod 113, $p = 0.3$: the learned algorithm uses a *small* set of key Fourier frequencies (5 in the reported run); cleanup of memorizing components coincides with the test-accuracy jump.
- Liu et al., Omnigrok (2023): on MNIST with 1000 training points, multiplying the initialization norm by a factor of order $10$ turns a no-delay run into one with a delay of $\sim 10^2$–$10^3\times$; weight decay compresses the delay back.
- Barak et al. (NeurIPS 2022): for $k$-sparse parities on $d$ bits, SGD shows a long flat loss phase with hidden progress, and the transition happens near the $\Theta(d^{k})$-ish computational limit — an explicit case where the plateau length is predictable from a complexity parameter.
- Kumar et al. (ICLR 2024): grokking coincides with the lazy→rich transition; $t_g$ shrinks as the output scale / laziness parameter is reduced, measured on 2-layer nets and small transformers.
- Reliable regularity across all of these: **larger $\lambda$, smaller $\alpha$, larger $p$ each reduce $t_g$**, monotonically over the measured range.

## 5. What Is Not Known

- **Methodologically blocked.** There is no threshold-free, metric-free definition of $t_g$. Choosing $\tau_g = 0.9$ vs $0.5$ moves $t_g$; choosing test accuracy vs adversarial robustness (Humayun et al., 2024) moves it by orders of magnitude. Until $t_g$ is defined as, say, the argmax of $d A_{\text{te}}/d\log t$ with a stated smoothing kernel, cross-paper numbers are not comparable.
- **Empirically open.** No published predictor is evaluated as a *forecast*: fit on one task family, tested on a held-out family, against a config-only baseline, with a seed-noise floor reported. The runs cost GPU-hours, not GPU-years. This is the cheapest open piece.
- **Theoretically open.** No bound of the form $t_g \le g(\lambda, \alpha, p, d, \text{width})$ for transformers. Existing proofs (Lyu et al. 2024; Mohamadi et al. 2024) establish *separation of phases* and asymptotic sample thresholds, not step counts with matching constants.
- **Theoretically open.** Whether $t_g$ is even identifiable from a prefix: is there a pair of runs with identical observable prefixes on $[0,T_{\text{obs}}]$ and $t_g$ differing by more than the seed floor?

## 6. Why It Is Hard

The specific obstruction is **a low signal-to-noise ratio in log space, on a target that is itself grid-quantized and threshold-defined.**

- The predictable dynamic range is small. Across the usual sweep ($p \in [0.3, 0.6]$, $\lambda$ over a decade), $\log_{10} t_g$ spans roughly 1.5–2 dex.
- The irreducible seed spread is $\sim 0.2$–$0.4$ dex. That is 10–25% of the range, before any model error.
- So a predictor must beat a trivial config-only regression by a margin that is comparable to noise, which forces many seeds per cell — and multiplying seeds is exactly what the long tail of $t_g$ makes expensive.

Secondary: **absent ground truth at small $p$.** Runs that would have grokked at $10^7$ steps are censored by the budget and recorded as $t_g = \infty$ or dropped, biasing any fit toward the fast half of the distribution. This is survival analysis being done as regression.

## 7. Current Research (as of 2026)

- **Mechanistic-interpretability progress measures** (Nanda, Conmy, and the Neel Nanda / MATS-adjacent community; DeepMind's Varma, Shah, Kramár, Kumar): turning circuit-formation metrics into online early-warning signals.
- **Phase-transition physics framing** (Rubin, Seroussi, Ringel, ICLR 2024: grokking as a first-order transition in two-layer networks; Tegmark's group at MIT): treating $t_g$ as a nucleation time, which predicts sensitivity to init scale rather than a closed form.
- **Lazy-to-rich theory** (Pehlevan group at Harvard; Lyu, Hu, Lee): the most likely source of a real $t_g$ bound.
- **Numerical-stability accounts** (Prieto, Birdal, Imperial College; arXiv 2025): softmax collapse and floating-point-driven late dynamics.
- **Grokking in LLM reasoning** (Wang et al., NeurIPS 2024, "Grokked Transformers are Implicit Reasoners"): delayed generalization on composition/comparison tasks well past training-loss convergence. Whether pretraining-scale emergence is the same phenomenon is *(frontier — verify)*; the tempting identification with emergent abilities is not established and Schaeffer et al. (2023) show metric choice alone can manufacture apparent emergence.

## 8. Concrete Next Experiment

**Question:** does any prefix observable beat configuration alone at predicting $t_g$?

**Scale.** 1-layer transformer, $d_{\text{model}} = 128$, 4 heads, AdamW. Task family A (fit): addition mod 113. Task family B (held out): $x^2 + xy \bmod 97$ and subtraction mod 97. Grid: $p \in \{0.30,0.35,\dots,0.60\}$ (7) $\times$ $\lambda \in \{0.03,0.1,0.3,1.0\}$ (4) $\times$ 6 seeds = 168 runs per family, budget $3\times10^6$ steps, eval on a 25-points-per-decade log grid. Each run is minutes on one A100; the whole thing is $\sim$300–600 GPU-hours including the censored tail.

**Arms.**

- *Control (config-only):* ridge regression of $\log_{10} t_g$ on $(\log p, \log \lambda, \log \alpha)$ and their pairwise products. No trajectory information.
- *Treatment:* the same regression plus prefix features computed on $t \le T_{\text{obs}} = 2\,t_m$: $\|\theta_t\|_2$ slope, train-loss oscillation power spectrum (Notsawo et al., 2023), $\|K_t - K_0\|_F$, and excluded-loss slope (Nanda et al., 2023).
- Censored runs enter as right-censored observations via a Tobit / accelerated-failure-time fit, not as deletions.

**Deciding number.** Median absolute error $\tilde\varepsilon = \mathrm{median}\,|\log_{10}\hat t_g - \log_{10} t_g|$ on family B, with the seed floor $\sigma_{\text{seed}}$ measured from the 6-seed cells.

- Treatment wins if $\tilde\varepsilon_{\text{treat}} \le \tilde\varepsilon_{\text{ctrl}} - 0.3$ dex (a factor-of-2 improvement) with a bootstrap 95% CI excluding zero.
- If $\tilde\varepsilon_{\text{treat}} \approx \sigma_{\text{seed}}$, the problem is solved at this scale and the next question is transformer-scale transfer.
- If neither arm gets within 0.5 dex of $\sigma_{\text{seed}}$, the evidence points to non-identifiability from the prefix, which is the theoretically interesting outcome.

## 9. Key References

- **[Foundational]** Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* ICLR 2022 MATH-AI Workshop. — arXiv:2201.02177
- **[SOTA — mechanism]** Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt. *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023. — arXiv:2301.05217
- **[SOTA — generality]** Ziming Liu, Eric J. Michaud, Max Tegmark. *Omnigrok: Grokking Beyond Algorithmic Data.* ICLR 2023. — arXiv:2210.01117
- **[SOTA — prediction]** Pascal Junior Tikeng Notsawo, Hattie Zhou, Mohammad Pezeshki, Irina Rish, Guillaume Dumas. *Predicting Grokking Long Before it Happens: A Look into the Loss Landscape of Models Which Grok.* 2023. — arXiv:2306.13253
- **[Theory]** Kaifeng Lyu, Jikai Jin, Zhiyuan Li, Simon S. Du, Jason D. Lee, Wei Hu. *Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking.* ICLR 2024. — arXiv:2311.18817
- **[Theory]** Mohamad Amin Mohamadi, Zhiyuan Li, Lei Wu, Danica J. Sutherland. *Why Do You Grok? A Theoretical Analysis of Grokking Modular Addition.* ICML 2024. — arXiv:2407.12332
- **[Theory]** Tanishq Kumar, Blake Bordelon, Samuel J. Gershman, Cengiz Pehlevan. *Grokking as the Transition from Lazy to Rich Training Dynamics.* ICLR 2024. — arXiv:2310.06110
- **[Mechanism]** Vikrant Varma, Rohin Shah, Zachary Kenton, János Kramár, Ramana Kumar. *Explaining Grokking Through Circuit Efficiency.* 2023. — arXiv:2309.02390
- **[Related]** Boaz Barak, Benjamin L. Edelman, Surbhi Goel, Sham Kakade, Eran Malach, Cyril Zhang. *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit.* NeurIPS 2022. — arXiv:2207.08799
- **[Related]** Ahmed Imtiaz Humayun, Randall Balestriero, Richard Baraniuk. *Deep Networks Always Grok and Here is Why.* ICML 2024. — arXiv:2402.15555
- **[Related]** Boshi Wang, Xiang Yue, Yu Su, Huan Sun. *Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization.* NeurIPS 2024. — arXiv:2405.15071
- **[Caution]** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS 2023. — arXiv:2304.15004

## 10. Worked Example

One cell of the grid above: addition mod 113, $p = 0.35$, $\lambda = 0.1$, six seeds. Suppose the measured $t_g$ values are

$$\{1.2,\ 1.9,\ 2.6,\ 3.4,\ 5.1,\ 8.0\}\times 10^5 \ \text{steps}.$$

In log space: $\{5.08, 5.28, 5.41, 5.53, 5.71, 5.90\}$, mean $5.49$, standard deviation $\sigma_{\text{seed}} \approx 0.29$ dex. The full sweep, $p = 0.30 \to 0.60$, moves the cell median from $\approx 10^{5.9}$ to $\approx 10^{4.0}$ — a signal spread of $\sigma_{\text{sig}} \approx 0.6$ dex once averaged over the $\lambda$ axis.

The ceiling on any predictor's $R^2$ against a target this noisy is

$$R^2_{\max} = \frac{\sigma_{\text{sig}}^2}{\sigma_{\text{sig}}^2 + \sigma_{\text{seed}}^2} = \frac{0.36}{0.36 + 0.084} \approx 0.81.$$

Now the obstruction. The config-only control already captures nearly all of $\sigma_{\text{sig}}$, because $p$ and $\lambda$ *are* the sweep axes — it will land near $R^2 \approx 0.75$ and $\tilde\varepsilon \approx 0.3$ dex. The prefix features have at most $0.06$ dex of residual variance left to explain, and $\sigma_{\text{seed}} = 0.29$ dex sits on top of it. To detect a genuine 0.3-dex improvement at 95% confidence you need roughly $n \gtrsim 2(1.96\,\sigma_{\text{seed}}/0.3)^2 \approx 8$ independent runs per condition — but the seeds that matter most are the slow ones at $t_g > 10^6$, which are precisely the ones the budget censors.

That is the shape of the problem: **the easy variance is already explained by the knobs you turned, and the variance that a trajectory-based predictor would have to explain is the same size as the seed noise, concentrated in the runs you cannot afford to finish.**

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*