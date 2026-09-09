---
id: 03-training-dynamics/implicit-bias-adaptive-optimizers
title: "Implicit Bias of Adaptive Optimizers"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Implicit Bias of Adaptive Optimizers

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/implicit-bias-adaptive-optimizers` · **Status:** open

## 1. Problem Statement

Modern networks are heavily overparameterized: the set of parameters reaching near-zero training loss is a high-dimensional manifold, and which point on it the optimizer lands on determines test error. For gradient descent (GD) on separable linear models this selection rule is a theorem — the iterates converge in direction to the $\ell_2$ max-margin separator. For Adam, RMSProp, Adafactor, Lion, Shampoo and Muon — the optimizers that actually train frontier models — no comparable characterization exists at practical hyperparameters.

Three variants, with different difficulty:

- **Theory variant.** Given an optimizer $\mathcal{A}$, a loss $L$, and an overparameterized model, characterize $\lim_{t\to\infty} \theta_t / \|\theta_t\|$ or the limiting point on the zero-loss manifold as the solution of an explicit constrained problem $\min_\theta R_\mathcal{A}(\theta) \ \text{s.t.}\ L(\theta)=0$. Solved for GD and sign-GD in restricted regimes; open for Adam with $\varepsilon>0$, $\beta_2<1$, momentum, and finite step size.
- **Measurement variant.** Define a bias functional $R_\mathcal{A}$ that is estimable from a real training run and predicts the optimizer's endpoint better than chance. Currently ill-posed: candidate proxies (sharpness, weight norm, margin, rank) disagree with each other.
- **Method variant.** Given a target bias, construct a preconditioner that realizes it. This is what Muon, Lion and weight-decay tuning do heuristically; nobody can currently derive the preconditioner from the desired bias.

Solving it means: an $R_\mathcal{A}$ that (a) is proved to be minimized in the limit under stated assumptions, and (b) predicts, on a held-out architecture, which of two optimizers generalizes better — before running both.

## 2. Formal Setting

Parameters $\theta \in \mathbb{R}^d$, data $S = \{(x_i,y_i)\}_{i=1}^n$, loss $L(\theta) = \frac{1}{n}\sum_i \ell(f(\theta;x_i), y_i)$. Adam with hyperparameters $(\eta, \beta_1, \beta_2, \varepsilon, \lambda)$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2},\quad \theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon} + \lambda\theta_t\right)$$

with $g_t$ the minibatch gradient at batch size $B$ and $\hat\cdot$ the bias-corrected moments.

Quantities as measured:

- **Limiting direction** $\bar\theta = \theta_T/\|\theta_T\|_2$ at the first step $T$ where train loss $< 10^{-3}$ per token (a finite-time stand-in for $t\to\infty$; the asymptotic object is not observable).
- **Normalized margin** $\gamma(\theta) = \min_i y_i f(\theta;x_i) / \|\theta\|_2^{p}$ for a $p$-homogeneous network; measured on the training set, not a validation set.
- **Preconditioned sharpness** $\lambda_{\max}\!\left(P_t^{1/2} \nabla^2 L\, P_t^{1/2}\right)$ where $P_t = \mathrm{diag}((\sqrt{\hat v_t}+\varepsilon)^{-1})$; measured by Lanczos on a fixed 10k-example probe batch, 20 iterations, $\pm 3\%$ reproducibility.
- **Bias functional** $R_\mathcal{A}$: any scalar of $\theta$ claimed to be implicitly minimized. Verified by fixing $L(\theta_T)$ across arms and comparing $R_\mathcal{A}$.

Assumptions the theory rests on, and their status in practice:

| Assumption | Status |
|---|---|
| $\varepsilon = 0$ | **Violated.** $\varepsilon \in [10^{-8},10^{-6}]$ and is not negligible: for LLM training a large fraction of coordinates have $\sqrt{\hat v}$ within an order of magnitude of $\varepsilon$ late in training. |
| $\beta_2 \to 1$ (or $\to 0$) limits | **Violated.** $\beta_2=0.95$–$0.999$ is the operating point; both limits are singular. |
| Full-batch / vanishing noise | **Violated.** $B \ll n$; gradient noise on transformers is heavy-tailed (Zhang et al., 2020). |
| Step size $\eta \to 0$ (gradient-flow / SDE limit) | **Violated.** Training runs at the edge of stability, where the discrete dynamics are not tracked by the flow (Cohen et al., 2021; 2022). |
| Homogeneous network, exponential-tailed loss | **Violated.** LayerNorm, biases and residual connections break homogeneity. |
| Separable data, $t\to\infty$ | **Violated.** LLMs are trained roughly single-epoch and never reach the asymptotic regime. |

Every published characterization needs at least three of these. That is the core problem.

## 3. State of the Art

**Theory SOTA (established).**
- *Steepest descent geometry.* Gunasekar et al. (ICML 2018): steepest descent w.r.t. a norm $\|\cdot\|$ on separable linear data with exponential-tailed loss converges in direction to the max-margin solution for that norm. Sign descent ($\ell_\infty$ steepest descent) $\Rightarrow$ $\ell_1$ max margin — a sparsity bias, opposite to GD's $\ell_2$ bias.
- *AdaGrad breaks it.* Qian & Qian (NeurIPS 2019): AdaGrad on separable linear data converges in direction, but the limit depends on initialization and step size and is **not** the max-margin solution for any fixed norm.
- *Homogeneous nets.* Wang et al. (ICML 2021): Adam and RMSProp on homogeneous networks with exponential loss converge to KKT points of an $\ell_2$ max-margin problem — but only in the $\varepsilon$-large regime where the preconditioner is nearly constant.
- *Backward error analysis.* Cattaneo, Klusowski & Shigida (ICML 2024): the continuous flow that tracks Adam to $O(\eta^2)$ contains an implicit penalty on a gradient norm; its sign and form depend on bias correction and on $\varepsilon$.

**Empirical SOTA (established).** Adam $\approx$ sign-momentum descent on transformers (Kunstner et al., ICLR 2023). Adam's advantage over SGD on language models correlates with heavy-tailed class imbalance in the token distribution (Kunstner et al., NeurIPS 2024). Preconditioned sharpness under Adam stabilizes at $\approx (2+2\beta_1)/(\eta(1-\beta_1)) = 38/\eta$ for $\beta_1=0.9$ (Cohen et al., 2022).

**Claimed but unablated.** (i) That Adam generalizes worse than SGD — Wilson et al. (2017) reported this on CIFAR-scale vision, but Choi et al. (2019) and Schmidt et al. (ICML 2021) show the gap largely disappears under equal tuning budgets; the vision result does not transfer to language. (ii) That Muon's spectral-norm steepest-descent framing (Bernstein & Newhouse, 2024) explains its wall-clock wins — the wins are benchmark numbers (Moonshot AI, 2025) with no bias measurement isolating geometry from step-size effects. (iii) That Adam's bias is "low-rank" or "flat-minima-seeking" — asserted in several papers, never shown with the loss level held fixed.

## 4. What Is Known

- **GD baseline, exact.** Logistic loss on separable data: $\theta_t/\|\theta_t\| \to$ $\ell_2$ max-margin direction at rate $O(1/\log t)$ (Soudry et al., JMLR 2018). Scale: linear models, $d$ up to $10^3$.
- **Sign vs. Adam equivalence.** On a 100M-parameter transformer, sign descent with momentum recovers Adam's loss curve to within noise, while removing the noise (full batch) does not close the SGD–Adam gap (Kunstner et al., 2023). Scale: 100M params, WikiText-103 / PTB.
- **Edge of stability, adaptive.** Across ResNets and transformers up to ~100M params, preconditioned sharpness rises then locks at $38/\eta$ ($\beta_1 = 0.9$) and stays within a few percent for the rest of training (Cohen et al., 2022).
- **Benchmark null result.** Schmidt, Schneider & Hennig (ICML 2021): 15 optimizers × 8 problems × 4 schedules, >50,000 runs. No optimizer dominates; a well-tuned Adam is within the top group everywhere. Scale: CIFAR/SVHN/IMDB/small transformers, $\le$ 30M params.
- **Optimizer near-equivalence at LM scale.** Zhao et al. (2024): Adam, Adafactor, Lion and Signum reach within ~1% of each other's validation loss at 150M–1.2B params under per-optimizer tuning; the differences that survive are hyperparameter *stability*, not endpoint quality.
- **Weight decay is not $\ell_2$ regularization here.** Decoupled decay (AdamW; Loshchilov & Hutter, ICLR 2019) changes the endpoint substantially; at LLM scale its effect is better described as controlling effective learning rate than as norm shrinkage (Andriushchenko et al., NeurIPS 2024).

## 5. What Is Not Known

- **Theoretically open.** No characterization of Adam's limiting direction at the practical operating point ($\varepsilon = 10^{-8}$, $\beta_2 = 0.95$, $\beta_1 = 0.9$, finite $\eta$, minibatch). The two solved regimes ($\varepsilon$ dominant $\to$ SGD-like; $\varepsilon \to 0$, $\beta_2 \to 0 \to$ sign descent) bracket practice but do not interpolate. Whether a single $R_{\text{Adam}}$ even exists — i.e. whether the endpoint is a function of the optimizer rather than of the whole trajectory — is itself unproven.
- **Empirically open.** Whether the sign-descent equivalence and the $38/\eta$ sharpness law hold at $\ge 7$B parameters and $\ge 10^{12}$ tokens. Runnable today, roughly $10^4$–$10^5$ GPU-hours; nobody has published the measurement.
- **Methodologically blocked.** "Implicit bias" has no agreed measurement for non-homogeneous networks trained for one epoch. Margin is undefined without homogeneity; the $t\to\infty$ limit is never approached; and any comparison of two optimizers' endpoints is confounded unless train loss is matched exactly, which almost no published comparison does.

## 6. Why It Is Hard

**Non-identifiability, then confounded measurement.** The endpoint difference between two optimizers is a sum of at least four effects: (1) the preconditioner geometry, (2) the effective step size $\eta/(\sqrt{\hat v}+\varepsilon)$, which differs per coordinate and per step, (3) the implicit noise covariance from minibatching, which is *reshaped* by the preconditioner, and (4) the interaction of decoupled weight decay with (2). Any single scalar comparison — test error, sharpness, weight norm — is a function of all four. Changing $\eta$ alone moves an optimizer across the entire range of reported "bias" differences, so an unmatched-tuning comparison measures tuning, not bias. This is exactly why Wilson et al. (2017) and Choi et al. (2019) reach opposite conclusions from compatible experiments.

Second obstruction: **the asymptotic object is not the measured object.** Every theorem is about $t \to \infty$ on separable data; every frontier run is one epoch, never separates, and stops far from any limit. The theory's predictions are not falsifiable by the runs people actually do.

## 7. Current Research (as of 2026)

- **Norm-geometry optimizer design.** Bernstein & Newhouse's "old optimizer, new norm" program recasts Adam, Shampoo and Muon as steepest descent under different norms (spectral, modular), turning bias into a design choice. Muon's adoption at scale (Moonshot AI, 2025) makes this the most active line. *(frontier — verify: whether the norm framing predicts endpoint properties, as opposed to convergence speed, remains untested.)*
- **Backward error analysis for adaptive methods.** Princeton (Cattaneo, Klusowski) and follow-ups deriving modified losses for Adam/AdamW; open question is validity at edge-of-stability step sizes, where the $O(\eta^2)$ expansion is not obviously convergent.
- **SDE and scaling rules.** Malladi, Lyu, Panigrahi & Arora (NeurIPS 2022) give square-root scaling rules for adaptive methods; extensions to $\mu$P and to the Adam–Muon comparison are ongoing.
- **Class-imbalance / heavy-tail explanations.** Kunstner, Bietti and collaborators (UBC / NYU): Adam's edge comes from progress on rare tokens. *(frontier — verify at $>$1B scale.)*
- **Edge-of-stability theory for preconditioned dynamics.** Cohen and collaborators; still descriptive rather than predictive.

## 8. Concrete Next Experiment

**Question.** Does Adam have an endpoint bias distinct from its step-size schedule, at fixed training loss?

**Scale.** Four 1.3B-parameter decoder-only transformers, 26B tokens each (Chinchilla-optimal $\times 1$), identical data order, identical initialization seed. ~2,000 A100-hours total.

**Arms.**
1. AdamW, $\varepsilon = 10^{-8}$ (reference).
2. AdamW, $\varepsilon = 10^{-3}$ (pushes the preconditioner toward identity; geometry changes, noise structure does not).
3. Signum (sign descent + momentum), tuned $\eta$ (the $\varepsilon\to 0$, $\beta_2\to 0$ corner).
4. **Control arm:** SGD-momentum with a *per-coordinate learning-rate schedule replayed from arm 1's recorded $\eta/(\sqrt{\hat v_t}+\varepsilon)$ trace*. This reproduces Adam's effective step sizes exactly while removing any online feedback between gradient and preconditioner.

Every arm is stopped at the *same* training loss $L^\ast = 2.30$ nats/token, not the same step count. This is the matching that existing comparisons omit.

**Deciding number.** Validation loss gap between arm 1 and arm 4 at $L^\ast$, in nats/token, with 3 seeds. If $|\Delta| < 0.005$ (below seed noise, which is ~0.004 at this scale), Adam's "implicit bias" is fully explained by its effective step-size schedule and the search for $R_{\text{Adam}}$ as a separate geometric object is misdirected. If $|\Delta| > 0.02$, a genuine feedback-driven bias exists, and arms 2–3 localize it along the $\varepsilon$ axis.

## 9. Key References

- **[Foundational]** Kingma, D. P., & Ba, J. *Adam: A Method for Stochastic Optimization.* ICLR, 2015. — arXiv:1412.6980
- **[Foundational]** Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., & Srebro, N. *The Implicit Bias of Gradient Descent on Separable Data.* JMLR 19(70), 2018. — arXiv:1710.10345
- **[Foundational]** Gunasekar, S., Lee, J., Soudry, D., & Srebro, N. *Characterizing Implicit Bias in Terms of Optimization Geometry.* ICML, 2018. — arXiv:1802.08246
- **[Theory SOTA]** Qian, Q., & Qian, X. *The Implicit Bias of AdaGrad on Separable Data.* NeurIPS, 2019. — arXiv:1906.03559
- **[Theory SOTA]** Wang, B., Meng, Q., Chen, W., & Liu, T.-Y. *The Implicit Bias for Adaptive Optimization Algorithms on Homogeneous Neural Networks.* ICML, 2021.
- **[Theory SOTA]** Cattaneo, M. D., Klusowski, J. M., & Shigida, B. *On the Implicit Bias of Adam.* ICML, 2024.
- **[Theory]** Lyu, K., & Li, J. *Gradient Descent Maximizes the Margin of Homogeneous Neural Networks.* ICLR, 2020. — arXiv:1906.05890
- **[Theory]** Malladi, S., Lyu, K., Panigrahi, A., & Arora, S. *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms.* NeurIPS, 2022. — arXiv:2205.10287
- **[Empirical SOTA]** Kunstner, F., Chen, J., Lavington, J. W., & Schmidt, M. *Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be.* ICLR, 2023.
- **[Empirical SOTA]** Kunstner, F., Yadav, R., Milligan, A., Schmidt, M., & Bietti, A. *Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models.* NeurIPS, 2024.
- **[Empirical]** Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR, 2021. — arXiv:2103.00065
- **[Empirical]** Cohen, J. M., et al. *Adaptive Gradient Methods at the Edge of Stability.* 2022. — arXiv:2207.14484
- **[Empirical]** Wilson, A. C., Roelofs, R., Stern, M., Srebro, N., & Recht, B. *The Marginal Value of Adaptive Gradient Methods in Machine Learning.* NeurIPS, 2017. — arXiv:1705.08292
- **[Empirical]** Balles, L., & Hennig, P. *Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients.* ICML, 2018.
- **[Empirical]** Zhang, J., Karimireddy, S. P., Veit, A., Kim, S., Reddi, S. J., Sra, S., & Kumar, S. *Why Are Adaptive Methods Good for Attention Models?* NeurIPS, 2020.
- **[Method]** Loshchilov, I., & Hutter, F. *Decoupled Weight Decay Regularization.* ICLR, 2019. — arXiv:1711.05101
- **[Method]** Chen, X., et al. *Symbolic Discovery of Optimization Algorithms.* NeurIPS, 2023. — arXiv:2302.06675
- **[Method]** Bernstein, J., & Newhouse, L. *Old Optimizer, New Norm: An Anthology.* 2024. — arXiv:2409.20325
- **[Survey/Benchmark]** Schmidt, R. M., Schneider, F., & Hennig, P. *Descending through a Crowded Valley — Benchmarking Deep Learning Optimizers.* ICML, 2021. — arXiv:2007.01547
- **[Survey/Benchmark]** Choi, D., Shallue, C. J., Nado, Z., Lee, J., Maddison, C. J., & Dahl, G. E. *On Empirical Comparisons of Optimizers for Deep Learning.* 2019. — arXiv:1910.05446

## 10. Worked Example

Take the smallest case where theory is complete and watch it fail at the operating point. Separable 2-D logistic regression, $n = 2$:

$$x_1 = (1,\ 0.1),\ y_1 = +1;\qquad x_2 = (-1,\ 0.1),\ y_2 = -1 .$$

Only coordinate 1 separates. Coordinate 2 is common to both classes and carries no signal.

- **GD** ($\ell_2$ max margin): the max-margin separator is $w \propto (1, 0)$. Coordinate 2 must go to zero relative to coordinate 1. Margin $\gamma_{\ell_2} = 1$.
- **Sign descent** ($\ell_1$ max margin, i.e. $\ell_\infty$ steepest descent): the $\ell_1$ max-margin solution is also $w \propto (1,0)$ here, but it is reached in $O(1)$ steps of size $\eta$ per coordinate rather than in $O(1/\log t)$ — a different trajectory, same limit.
- **Adam at the real operating point.** $\varepsilon = 10^{-8}$, $\beta_2 = 0.95$. Coordinate 2's gradient magnitude is $0.1\times$ coordinate 1's, so $\sqrt{\hat v_2} \approx 0.1\sqrt{\hat v_1}$ and Adam's update on coordinate 2 is amplified $10\times$ relative to GD. Adam therefore *grows* the uninformative coordinate faster than GD does, early in training, before the margin term eventually dominates.

Now the obstruction. Run all three to train loss $10^{-3}$ and read off the normalized weight on coordinate 2, $|w_2|/\|w\|$. All three go to $0$ — the limits agree, because in the separable linear case every bias functional is minimized at the same point. The differences that matter are **pre-asymptotic**: at $L = 0.05$ (a realistic stopping loss), $|w_2|/\|w\|$ is roughly $10\times$ larger under Adam than under GD, and that ratio is a function of $\varepsilon$, $\beta_2$ and $\eta$ jointly, with no closed form.

So the tractable object (the limit) is identical across optimizers and irrelevant to practice; the object that differs (the finite-loss iterate) is exactly the one the theory does not describe. Scaling the example up does not help: at 1.3B parameters the limit is never reached at all, and the measured endpoint is a pre-asymptotic point whose position is set by the same three unresolved hyperparameters. That gap — not compute, not data — is what keeps this problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*