---
id: 08-loss-and-heads/optimal-bin-design-distributional-heads
title: "Optimal Bin Design for Distributional Value Heads"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Bin Design for Distributional Value Heads

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/optimal-bin-design-distributional-heads` · **Status:** open

## 1. Problem Statement

A distributional value head replaces a scalar regression output with a categorical distribution over $m$ fixed bins, trained by cross-entropy. This is now the default value head in MuZero, DreamerV3, and classification-based value learning for deep RL. The design choices — number of bins $m$, support range $[v_{\min}, v_{\max}]$, spacing (uniform, log, squashed), label smoothing width $\sigma$, and the decode rule — are set by hand and copied between papers without re-derivation.

Three variants, of different difficulty:

- **Measurement.** Given a task and architecture, which bin design minimises downstream return (not held-out log-likelihood)? Currently unanswered because no standard protocol separates the *distributional* effect from the *label-smoothing regulariser* effect.
- **Method.** Produce a rule mapping observable statistics of the return distribution (scale, tail index, drift over training) to $(m, \text{support}, \sigma)$ that beats a tuned fixed grid, at a cost below one extra training run.
- **Theory.** Bound the excess value error induced by binning as a function of $(m, \sigma, \Delta)$ under function approximation, and prove where the optimum sits.

Solving it means: a design rule, validated on at least two domain families, that recovers within noise of an exhaustive per-task sweep.

## 2. Formal Setting

Returns $G \in \mathbb{R}$ with conditional law $\eta(\cdot \mid s)$. Choose bin centres $z_1 < \dots < z_m$, uniform in a transformed coordinate $h$, so $z_i = h^{-1}(u_i)$, $u_i = u_{\min} + (i-1)\Delta$, $\Delta = (u_{\max}-u_{\min})/(m-1)$. The head emits logits $\ell_\theta(s) \in \mathbb{R}^m$ and $p_\theta = \mathrm{softmax}(\ell_\theta)$.

Two label constructions:

**Two-hot (categorical projection $\Pi_C$).** For target $y$ with $h(y) \in [u_i, u_{i+1}]$,
$$q_i = \frac{u_{i+1}-h(y)}{\Delta}, \quad q_{i+1} = \frac{h(y)-u_i}{\Delta}.$$
Mean-preserving in $h$-space, not in return space.

**HL-Gauss.** $q_i = \Phi\!\big((b_{i+1}-h(y))/\sigma\big) - \Phi\!\big((b_i-h(y))/\sigma\big)$ over bin edges $b_i$. The free parameter is the ratio $\rho = \sigma/\Delta$.

Loss: $\mathcal{L}(\theta) = -\sum_i q_i \log p_{\theta,i}$.

**Decode.** Two inequivalent rules: $\hat v_{\text{ret}} = \sum_i p_i\, h^{-1}(u_i)$ and $\hat v_{\text{tr}} = h^{-1}\big(\sum_i p_i u_i\big)$. They differ by a Jensen gap of order $\tfrac12 (h^{-1})''\!\cdot\!\mathrm{Var}_p[u]$.

Squashing (Pohlen et al., 2018): $h(x)=\mathrm{sign}(x)(\sqrt{|x|+1}-1)+\varepsilon x$, $\varepsilon=10^{-3}$ (MuZero), or $h=\mathrm{symlog}$ (DreamerV3).

**Measured quantities.** $m$, $\Delta$, $\rho$: read off config. Support coverage: empirical fraction of bootstrapped targets with $h(y) \notin [u_{\min},u_{\max}]$, logged per 10k steps. Resolution loss: $\mathbb{E}[(\hat v - y)^2]$ on a held-out target set with $\theta$ frozen at convergence. Decision metric: IQM of normalised return over $\ge 5$ seeds.

**Assumptions, and how they fail.**
1. *Return support is known a priori.* Violated — bootstrapped targets drift as the policy improves; a range fixed at init is wrong by the end.
2. *Bin design is orthogonal to optimisation.* Violated — cross-entropy over $m$ bins changes the loss curvature and effective gradient scale, so $m$ and learning rate interact.
3. *Cross-entropy on the projected target minimises a distributional metric.* Violated — $\Pi_C \mathcal{T}^\pi$ contracts in the Cramér metric, but the KL used in training is not the gradient of that metric (Rowland et al., 2018).
4. *Returns are unimodal.* Violated in sparse-reward and multi-goal tasks, which is the case where the distributional head should help most.

## 3. State of the Art

**Established (ablated, reproduced).**
- C51 (Bellemare et al., ICML 2017) fixed $m=51$, support $[-10,10]$, and ablated $m \in \{5,11,21,51\}$ on Atari: performance rises monotonically and saturates near 51. The support bound was chosen for clipped rewards, not derived.
- HL-Gauss (Farebrother et al., ICML 2024) swept $\rho = \sigma/\Delta$ and reported a broad optimum near $\rho \approx 0.75$, with degradation on both sides. This is the single most direct piece of bin-design evidence in the literature, and it was swept on Atari and re-checked on chess and robot manipulation.
- Imani & White (ICML 2018) established that soft-target cross-entropy ("HL-Gauss") beats squared error on plain regression, i.e. part of the benefit is not distributional at all.

**Claimed but unablated.**
- MuZero's $m=601$, support $[-300,300]$ in $h$-space (Schrittwieser et al., Nature 2020) — no published sweep over $m$ or support.
- DreamerV3's 255 symlog buckets over $[-20,20]$ (Hafner et al., 2023) — reported as part of a package of tricks; the bin count is not isolated.
- Non-uniform / learned spacing: FQF (Yang et al., NeurIPS 2019) learns quantile fractions and reports mean human-normalised score ~1426% on 55 Atari games vs ~1112% for IQN — a benchmark number, obtained with a second proposal network and its own learning rate, never isolated to the adaptivity itself.

**Theory SOTA.** Rowland et al. (AISTATS 2018): $\Pi_C\mathcal{T}^\pi$ is a $\sqrt{\gamma}$-contraction in the supremum Cramér metric $\bar\ell_2$, giving a fixed point and an $O(\Delta)$ projection bias — but the bound is in the *tabular* setting and says nothing about the optimum $m$ under function approximation.

## 4. What Is Known

- **Bins help, then saturate.** C51 on 57 Atari games at 200M frames: $m=5$ is clearly worse than $m=51$; gains between 21 and 51 are small. Scale: DQN-class CNN, ~2M params.
- **The regulariser is real and separable.** Farebrother et al. (2024) show two-hot ($\rho=0$) performs at roughly MSE level, while HL-Gauss at $\rho\approx0.75$ beats both, across Atari 200M, chess, Wordle, and language-agent value learning. Implication: most of the measured "distributional head" gain at these scales is smoothing, not distribution modelling.
- **Distributional heads give no benefit in tabular or linear settings.** Lyle, Bellemare & Castro (AAAI 2019): expected and distributional TD have the same fixed point and comparable behaviour without nonlinear approximation. The effect is an approximation/optimisation phenomenon.
- **Projection bias is bounded by bin width.** Two-hot preserves the mean in $h$-space exactly and inflates variance by at most $\Delta^2/4$ per target.
- **Support clipping is unbounded.** Once $|h(y)| > u_{\max}$, the target is truncated and the induced value error grows without bound in $y$. C51's $[-10,10]$ works only because rewards are clipped to $[-1,1]$ and $\gamma=0.99$ gives $|G|\le 100$ — the bound is comfortably violated in principle and rarely in practice.
- **Cost.** The head costs $m\cdot d$ parameters; at $d=2048$, $m=601$ this is 1.2M parameters — negligible against a backbone, so $m$ is not compute-limited.

## 5. What Is Not Known

- **Theoretically open.** No bound on excess value error as a joint function of $(m,\rho)$ under nonlinear function approximation. The $\sqrt{\gamma}$ Cramér contraction is tabular; it predicts monotone improvement in $m$, which contradicts the observed saturation, so the operative mechanism is unmodelled. Also open: why $\rho\approx0.75$ rather than any other constant — no derivation exists.
- **Empirically open.** Whether adaptive support (rescaled online from running return quantiles) beats a well-tuned fixed grid. Runnable today at Atari 200M or DMC scale; not run as a clean isolated ablation. Also empirically open: whether $m$ should scale with model size or dataset size.
- **Methodologically blocked.** "Which bin design is best" is not well posed until the decode rule is fixed, because $\hat v_{\text{ret}}$ and $\hat v_{\text{tr}}$ disagree and papers do not consistently state which they use. Equally blocked: separating the distributional contribution from the smoothing contribution requires a control that smooths *without* binning, and no standard such control exists.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement across three simultaneous effects with a single knob**. Changing $m$ at fixed support simultaneously changes (i) quantisation resolution, (ii) the effective label-smoothing width in units of return, and (iii) the softmax's gradient scale and loss curvature. Any observed change in return is attributable to all three. Holding $\rho = \sigma/\Delta$ fixed while varying $m$ controls (ii) in *bin* units but not in *return* units; holding $\sigma$ fixed in return units breaks $\rho$. There is no assignment of the knobs that isolates one effect, so every published $m$-sweep measures a composite.

Secondary: **absent ground truth**. The true return distribution $\eta(\cdot\mid s)$ is unobserved off-policy, so calibration of the head cannot be scored directly; papers fall back on downstream return, which is a noisy, seed-dominated $\pm$5–10% IQM signal requiring $\ge$5 seeds per cell — a 4×4 grid over $(m,\rho)$ is 80 Atari-scale runs.

## 7. Current Research (as of 2026)

- Classification-based value learning as default practice, following Farebrother et al. (Google DeepMind / Mila). $\rho=0.75$ HL-Gauss is being copied into new codebases largely unexamined *(frontier — verify whether anyone has re-swept $\rho$ at >1B parameters)*.
- Distributional critics in RLHF/RLVR pipelines for language models, where the return is a bounded reward in $[0,1]$ and the natural bin design is entirely different from the Atari one *(frontier — verify)*.
- Learned/adaptive support: online rescaling of $[v_{\min},v_{\max}]$ from running percentiles of bootstrapped targets, an idea present in FQF-style adaptivity and in reward-normalisation work, not yet isolated *(frontier — verify)*.
- Theory of statistical functionals in distributional RL (Rowland et al., ICML 2019; Bellemare, Dabney & Rowland, MIT Press 2023) — the framework exists to state a bin-design optimality result; the result has not been stated.

## 8. Concrete Next Experiment

**Scale.** Atari-10 subset (or DMC-15), 50M frames, IMPALA-ResNet (~1.5M params), 10 seeds per cell. ~120 runs, feasible on ~2k GPU-hours.

**Grid.** $m \in \{21, 51, 151, 601\}$ crossed with three smoothing conditions: (a) two-hot, $\rho=0$; (b) HL-Gauss, $\rho=0.75$ (constant in bin units); (c) HL-Gauss with $\sigma$ fixed at $0.75 \times \Delta_{m=51}$ in *return* units (constant in return units, so $\rho$ varies with $m$). Support fixed in $h$-space for all cells. Decode rule fixed to $\hat v_{\text{ret}}$ and logged.

**Control arm.** Scalar MSE head with the identical backbone and an $\ell_2$ target-noise regulariser of matched magnitude $\sigma$ — smoothing without binning. This is the arm that isolates the distributional contribution.

**The deciding number.** The IQM human-normalised score gap between condition (b) and condition (c) at $m=601$. If it is within seed noise ($<3$ IQM points), $\rho$ is the operative parameter and $m$ matters only through $\Delta \le \sigma$ — bin count is a nuisance, and the design rule is "pick $\sigma$ from return scale, then any $m$ with $\Delta \lesssim \sigma$". If the gap exceeds 10 IQM points, resolution and smoothing are genuinely separate axes and a two-parameter design rule is required.

## 9. Key References

- **[Foundational]** Bellemare, Dabney, Munos. *A Distributional Perspective on Reinforcement Learning.* ICML, 2017. — arXiv:1707.06887
- **[Foundational]** Rowland, Bellemare, Dabney, Munos, Teh. *An Analysis of Categorical Distributional Reinforcement Learning.* AISTATS, 2018. — arXiv:1802.08163
- **[SOTA]** Farebrother, Orbay, Vuong, Taïga, Chebotar, Xiao, Irpan, Levine, Castro, Faust, Kumar, Agarwal. *Stop Regressing: Training Value Functions via Classification for Scalable Deep RL.* ICML, 2024. — arXiv:2403.03950
- **[Foundational]** Imani, White. *Improving Regression Performance with Distributional Losses.* ICML, 2018.
- **[SOTA]** Dabney, Rowland, Bellemare, Munos. *Distributional Reinforcement Learning with Quantile Regression.* AAAI, 2018. — arXiv:1710.10044
- **[SOTA]** Dabney, Ostrovski, Silver, Munos. *Implicit Quantile Networks for Distributional Reinforcement Learning.* ICML, 2018. — arXiv:1806.06923
- **[SOTA]** Yang, Zhao, Qiu, Chen, Liu, Liu. *Fully Parameterized Quantile Function for Distributional Reinforcement Learning.* NeurIPS, 2019. — arXiv:1911.02140
- Schrittwieser, Antonoglou, Hubert, et al. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.* Nature 588, 2020. — arXiv:1911.08265
- Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Domains through World Models.* 2023. — arXiv:2301.04104
- Pohlen, Piot, Hester, et al. *Observe and Look Further: Achieving Consistent Performance on Atari.* 2018. — arXiv:1805.11593
- Lyle, Bellemare, Castro. *A Comparative Analysis of Expected and Distributional Reinforcement Learning.* AAAI, 2019.
- **[Survey]** Bellemare, Dabney, Rowland. *Distributional Reinforcement Learning.* MIT Press, 2023.

## 10. Worked Example

MuZero-style head: $h(x)=\mathrm{sign}(x)(\sqrt{|x|+1}-1)+10^{-3}x$, bins at integer $h$-values, $\Delta=1$. Take a bootstrapped target $y=30$.

$h(30)=\sqrt{31}-1+0.03 = 4.598$. Two-hot weights: $w_5 = 0.598$ on $u=5$, $w_4=0.402$ on $u=4$.

Bin centres decoded back: $h^{-1}(4)\approx 24.0$, $h^{-1}(5)\approx 35.0$.

Decode 1 (return-space mean): $\hat v_{\text{ret}} = 0.402(24.0) + 0.598(35.0) = 9.65 + 20.93 = \mathbf{30.58}$.
Decode 2 (transform-space mean): $\hat v_{\text{tr}} = h^{-1}(0.402\cdot4 + 0.598\cdot5) = h^{-1}(4.598) = \mathbf{30.00}$.

**The obstruction, visible.** The label is exactly mean-preserving, the network fits it perfectly, and two published decode rules for the same head return values differing by $0.58$ — a $1.9\%$ bias that is *pure convention*, not error. At $y=250$ the same calculation gives $\hat v_{\text{ret}}=250.1$ vs $\hat v_{\text{tr}}=250.0$; at $y=3$, $h(3)=1.003$, weights $(0.997,0.003)$ on $u=1,2$, decoding to $3.02$ vs $3.00$. The bias is not monotone in $y$ — it is largest where $w(1-w)$ is largest, i.e. mid-bin, and it is $O(\Delta^2 (h^{-1})'')$, which for the square-root squash grows linearly in $|y|$.

Now the confound. Halving $\Delta$ (doubling $m$ to 1201) cuts this bias by 4×, but at fixed $\rho=0.75$ it also halves $\sigma$ in return units — so the regulariser strength that Farebrother et al. found to matter changes at the same time. A return improvement from $m{=}601 \to 1201$ cannot be attributed. That is why the deciding experiment in §8 must vary $\sigma$ in return units and in bin units as separate arms; without that split, every bin-count sweep in the literature is uninterpretable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*