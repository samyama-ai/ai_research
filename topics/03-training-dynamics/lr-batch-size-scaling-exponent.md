---
id: 03-training-dynamics/lr-batch-size-scaling-exponent
title: "Learning Rate versus Batch Size Scaling Exponent"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learning Rate versus Batch Size Scaling Exponent

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/lr-batch-size-scaling-exponent` · **Status:** open

## 1. Problem Statement

When batch size $B$ changes, how must the peak learning rate $\eta$ change so training is unharmed? Practice assumes a power law $\eta^\star(B) \propto B^{\alpha}$ and argues over $\alpha$: $1$ (linear), $1/2$ (square-root), or something smaller that decays as $B$ approaches the critical batch size.

Three variants, different difficulty:

- **Measurement.** Given an architecture, optimizer, data distribution, and token budget $D$, estimate $\eta^\star(B) = \arg\min_\eta L(\eta, B)$ over a grid, and fit $\alpha$. Runnable; the dispute is over what is held fixed.
- **Method.** Produce a transfer rule that predicts $\eta^\star$ at a target $(N, B, D)$ from sweeps at $\ge 100\times$ less compute, with final loss within the seed noise band of a direct sweep. Partly solved for width ($\mu$P), not for batch size.
- **Theory.** Prove, for Adam-family optimizers on non-convex objectives, the exponent that keeps the discrete trajectory close to a fixed continuous-time limit, and predict where that rule breaks.

Solving it means: a rule with a stated validity region, an error bar on $\alpha$, and a falsifiable prediction at a scale where nobody has swept.

## 2. Formal Setting

Parameters $\theta \in \mathbb{R}^N$, per-example loss $\ell(\theta, x)$, population loss $L(\theta) = \mathbb{E}_{x \sim \mathcal{D}}[\ell]$. Minibatch gradient $g_B = \frac{1}{B}\sum_{i=1}^{B} \nabla \ell(\theta, x_i)$, with per-example gradient covariance $\Sigma(\theta) = \mathrm{Cov}_{x}[\nabla \ell]$, so $\mathrm{Cov}[g_B] = \Sigma/B$.

**Measured quantities.**

- $\eta^\star(B; N, D)$: the peak LR minimizing final validation loss, on a fixed cosine or WSD schedule with warmup, over a log-spaced grid (typically $\times 2$ or $\times\sqrt{2}$ spacing). Estimated by fitting a quadratic in $\log \eta$ near the minimum; grid spacing bounds resolution.
- The exponent, from a regression over batch sizes: $\log \eta^\star(B) = \alpha \log B + c$, $\alpha$ read off by least squares with heteroscedastic weights from seed variance.
- Gradient noise scale (McCandlish et al., 2018), the batch size at which the noise term matches the curvature term:
$$\mathcal{B}_{\text{noise}} = \frac{\mathrm{tr}(H\Sigma)}{\nabla L^\top H \nabla L},\qquad \mathcal{B}_{\text{simple}} = \frac{\mathrm{tr}(\Sigma)}{\|\nabla L\|^2}$$
measured in practice by comparing gradient norms at two batch sizes, $\hat{\mathcal{B}} = \mathbb{E}\|g\|^2$-based unbiased estimators, not by forming $H$.
- Critical batch size $B_{\mathrm{crit}}$: measured from a steps-to-target-loss curve $S(B)$ fit to $S(B)/S_{\min} = 1 + B_{\mathrm{crit}}/B$; the knee of the Pareto frontier between steps and examples.

**Assumptions, and which are violated.**

1. *$\eta^\star$ is a single well-separated minimum.* Violated: the loss-vs-$\eta$ curve is asymmetric and near-flat over a factor of $2$–$4$ at large $N$, so $\alpha$ is weakly identified.
2. *Fixed $D$, varying $B$, means varying step count $S = D/B$.* The alternative — fixed $S$ — gives a different $\alpha$. Both appear in the literature, often unlabeled.
3. *Constant $\Sigma$ along the trajectory.* Violated: $\mathcal{B}_{\text{noise}}$ grows by an order of magnitude within one run as loss falls.
4. *The SDE approximation holds.* Requires $\eta$ small relative to curvature; violated exactly in the edge-of-stability regime where the largest Hessian eigenvalue sits at $\approx 2/\eta$.
5. *Other hyperparameters are held fixed.* Violated: weight decay, $\beta_2$, and warmup length all interact with $B$; the $\lambda \eta$ product sets an effective timescale.

## 3. State of the Art

**Established (reproduced, ablated).**

- Linear scaling $\eta \propto B$ with warmup for SGD+momentum on ImageNet ResNet-50 up to $B = 8192$ (Goyal et al., 2017); breaks above that.
- The steps-vs-batch curve has three regimes — perfect scaling, diminishing returns, saturation — across 6 model families and 7 datasets (Shallue et al., JMLR 2019). This is the most carefully controlled measurement in the area: every point is a separate metaparameter sweep.
- The optimal $\eta$ does *not* follow a single power law across the whole range for any optimizer studied; SGD and Adam differ in where they break (Zhang et al., NeurIPS 2019).
- $\mu$P transfers $\eta^\star$ across *width* at fixed $B$ (Yang et al., 2022). Width transfer is not batch-size transfer, and the papers say so.

**Claimed, not independently ablated.**

- Square-root scaling for Adam, $\eta \propto \sqrt{B}$, derived from an SDE argument (Malladi et al., ICML 2022). The derivation is sound under its stated noise assumption; the empirical support is a handful of settings, and the assumption (gradient noise dominating the signal) is the regime where the answer matters least.
- Joint LR/batch scaling laws from LLM pretraining: DeepSeek LLM (2024) fits $\eta^\star \propto C^{-0.1250}$, $B^\star \propto C^{0.3271}$ in compute $C$; Li et al. "Step Law" (2025) fits a convex surface over $(N, D)$ and reports $\approx 0.09\%$ deviation from swept optima. Both are single-lab benchmark numbers on one tokenizer, one data mix, one architecture family. Neither has an outside reproduction.
- Bergsma et al. (2025) argue the optimum is better parameterized by tokens-per-parameter and the $\lambda\eta$ product than by $B$ alone — a reframing, not yet independently tested.

## 4. What Is Known

- **Linear scaling holds and then stops.** ResNet-50/ImageNet, $B = 256 \to 8192$: linear $\eta$ plus 5-epoch warmup matches baseline top-1 within $0.1\%$ (Goyal et al., 2017, 256 GPUs, 1 hour). At $B = 16384$ the match degrades.
- **Critical batch size grows with data, not model size.** Zhang et al. (2024), models $85$M–$1.2$B on C4: $B_{\mathrm{crit}}$ scales primarily with $D$; at fixed $D$, varying $N$ over $14\times$ moves $B_{\mathrm{crit}}$ little. Reported $B_{\mathrm{crit}} \approx 2$M tokens at their largest budget.
- **The measured exponent is between the two textbook values.** Granziol, Zohren, Roberts (JMLR 2022) derive from random matrix theory that SGD admits linear scaling up to a noise-dependent threshold while Adam admits square-root, and confirm on CIFAR/ImageNet-scale runs.
- **Exponent estimates disagree by more than their error bars.** Porian et al. (2024), reconciling Kaplan- and Chinchilla-style laws, show that failing to tune $\eta$ jointly with $B$ shifts the fitted compute-optimal exponent enough to explain a large part of the historical discrepancy — models $10$M–$1$B.
- **Warmup is not a nuisance parameter.** At $B \ge 8$k, removing warmup changes $\eta^\star$ by more than the difference between $\alpha = 1/2$ and $\alpha = 1$ over a $4\times$ batch range (Goyal 2017; Shallue 2019).

## 5. What Is Not Known

- **Theoretically open.** No proof of the correct exponent for Adam/AdamW on non-convex objectives outside the small-$\eta$ SDE limit. Malladi et al. prove weak-approximation validity of the SDE and derive $\sqrt{B}$ *under* that limit; there is no theorem for the edge-of-stability regime where LLM pretraining actually runs, and no proof that a single exponent exists there.
- **Empirically open.** Nobody has published a controlled $\eta \times B$ grid — say $8 \times 8$ points with $\ge 3$ seeds — at $\ge 10$B parameters with everything else fixed. Estimated cost is the reason (§6). The exponent above $B_{\mathrm{crit}}$ is therefore an extrapolation everywhere it is quoted.
- **Empirically open.** Whether $\alpha$ depends on data repetition, mixture composition, or sequence length. All published fits use near-single-epoch web text.
- **Methodologically blocked.** "Optimal LR" is not well defined when the loss-vs-$\eta$ curve is flat to within seed noise over a $2$–$4\times$ range: the argmin is a noisy statistic of a nearly flat function, so $\alpha$ inherits variance that is rarely reported. No standard estimator or confidence interval for $\eta^\star$ exists.
- **Methodologically blocked.** $B_{\mathrm{crit}}$ is defined relative to a target loss; different targets on the same runs give different $B_{\mathrm{crit}}$, and papers pick targets differently.

## 6. Why It Is Hard

Three named obstructions.

- **Non-identifiability of the argmin.** Near $\eta^\star$ the loss behaves like $L(\eta) \approx L^\star + \kappa(\log \eta - \log \eta^\star)^2$ with small $\kappa$. If seed noise in final loss is $\sigma$, the resolvable interval is $|\log \eta - \log \eta^\star| \le \sqrt{\sigma/\kappa}$. With $\sigma \approx 0.003$ nats and $\kappa \approx 0.01$ nats per (log-decade)$^2$ — typical for a 1B run — the interval is $\pm 0.55$ decades. Fitting $\alpha$ across a $16\times$ batch range then gives a standard error near $\pm 0.4$: wide enough to contain both $0.5$ and $1.0$.
- **Confounded measurement.** Changing $B$ at fixed token budget changes step count, so it changes schedule shape, warmup fraction, and total weight-decay pressure at once. Any of these can produce an apparent exponent. Few papers report the fixed-$S$ arm alongside the fixed-$D$ arm.
- **Compute cost of the decisive arm.** The question only bites above $B_{\mathrm{crit}}$, and $B_{\mathrm{crit}}$ grows with $D$. Resolving $\alpha$ at 10B parameters over a $16\times$ batch range with seed replication is a multi-million-GPU-hour experiment — one that produces no model anyone ships.

## 7. Current Research (as of 2026)

- **Critical-batch-size scaling laws.** Kakade/Harvard-adjacent work (Zhang et al., 2024) establishing $B_{\mathrm{crit}}$'s dependence on $D$ rather than $N$; follow-ups extend to MoE and to fine-tuning *(frontier — verify)*.
- **Joint hyperparameter surfaces.** StepLaw (Li et al., 2025) and Cerebras' Power Lines (Bergsma et al., 2025) fit $(\eta, B, \lambda)$ jointly rather than one at a time. The shared claim — that the right coordinates are $\lambda\eta$ and tokens-per-parameter, not $\eta$ and $B$ — is the most likely resolution of the exponent dispute, and is untested outside the originating labs.
- **$\mu$P extensions to depth and batch.** Work extending Tensor-Programs-style transfer beyond width; batch-size transfer remains the weakest leg *(frontier — verify)*.
- **Optimizer-dependence.** Muon, Shampoo, and second-order-flavored optimizers report different $B_{\mathrm{crit}}$ and different LR sensitivity; whether they have a *different* $\alpha$ or merely a shifted $B_{\mathrm{crit}}$ is open *(frontier — verify)*.

## 8. Concrete Next Experiment

**Goal.** Estimate $\alpha$ with a reported confidence interval, in the regime where the rules disagree.

**Scale.** Decoder-only transformer, $N = 1.4$B non-embedding parameters, $D = 30$B tokens fixed, sequence length 4096, AdamW ($\beta_2 = 0.95$), WSD schedule with 1% warmup measured in *tokens* (not steps — this removes the schedule confound).

**Grid.** $B \in \{0.25, 0.5, 1, 2, 4, 8\}$M tokens — chosen to straddle the $\approx 2$M-token $B_{\mathrm{crit}}$ reported by Zhang et al. (2024). For each $B$, seven LRs spaced $\sqrt{2}$ apart, centered by a pilot at $B = 0.25$M. Three seeds at the three LRs nearest each per-$B$ optimum, one seed elsewhere. Total $\approx 6 \times (7 + 6) = 78$ runs.

**Control arm.** Same grid at fixed *step count* $S = 30{,}000$ (so $D$ grows with $B$), at three batch sizes $\{0.5, 2, 8\}$M. If $\alpha_{\text{fixed-}D} \ne \alpha_{\text{fixed-}S}$ beyond error, the exponent is a property of the protocol, not the optimizer — which is itself the result.

**The deciding number.** $\hat{\alpha}$ over $B \in [0.25, 2]$M (below $B_{\mathrm{crit}}$) and over $B \in [2, 8]$M (above), each with a bootstrap 95% CI from the seed replicates. The question is settled in one direction if the sub-critical CI excludes $1.0$ and contains $0.5$ while the super-critical CI excludes both and sits below $0.3$ — the prediction of a rule where $\eta^\star$ saturates rather than continuing as a power law. If either CI has width $> 0.4$, the honest conclusion is that $\alpha$ is not identifiable at this scale, and the field should stop quoting point estimates.

## 9. Key References

- **[Foundational]** P. Goyal, P. Dollár, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, K. He. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* 2017. — arXiv:1706.02677
- **[Foundational]** A. Krizhevsky. *One weird trick for parallelizing convolutional neural networks.* 2014. — arXiv:1404.5997
- **[Foundational]** S. McCandlish, J. Kaplan, D. Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch Training.* 2018. — arXiv:1812.06162
- **[Foundational]** S. L. Smith, Q. V. Le. *A Bayesian Perspective on Generalization and Stochastic Gradient Descent.* ICLR, 2018. — arXiv:1710.06451
- **[SOTA]** C. J. Shallue, J. Lee, J. Antognini, J. Sohl-Dickstein, R. Frostig, G. E. Dahl. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019. — arXiv:1811.03600
- **[SOTA]** G. Zhang, L. Li, Z. Nado, J. Martens, S. Sachdeva, G. E. Dahl, C. J. Shallue, R. Grosse. *Which Algorithmic Choices Matter at Which Batch Sizes? Insights from a Noisy Quadratic Model.* NeurIPS, 2019. — arXiv:1907.04164
- **[SOTA]** S. Malladi, K. Lyu, A. Panigrahi, S. Arora. *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms.* NeurIPS, 2022. — arXiv:2205.10287
- **[SOTA]** H. Zhang, D. Morwani, N. Vyas, J. Wu, D. Zou, U. Ghai, D. Foster, S. Kakade. *How Does Critical Batch Size Scale in Pre-training?* ICLR, 2025. — arXiv:2410.21676
- **[SOTA]** H. Li, S. Zheng, J. Wang, et al. *Predictable Scale: Part I — Optimal Hyperparameter Scaling Law in Large Language Model Pretraining.* 2025. — arXiv:2503.04715
- **[SOTA]** S. Bergsma, N. Dey, et al. *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training.* 2025. — arXiv:2505.13738
- **[Theory]** D. Granziol, S. Zohren, S. Roberts. *Learning Rates as a Function of Batch Size: A Random Matrix Theory Approach to Neural Network Training.* JMLR 23(173), 2022.
- **[Related]** G. Yang, E. J. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, J. Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[Related]** T. Porian, M. Wortsman, J. Jitsev, L. Schmidt, Y. Carmon. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models.* NeurIPS, 2024. — arXiv:2406.19146
- **[Related]** DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* 2024. — arXiv:2401.02954

## 10. Worked Example

Take a 400M-parameter model, $D = 8$B tokens, AdamW, and a measured optimum $\eta^\star = 1.0 \times 10^{-3}$ at $B = 0.5$M tokens. Predict $\eta^\star$ at $B = 8$M ($16\times$).

- Linear ($\alpha = 1$): $\eta^\star = 1.6 \times 10^{-2}$.
- Square-root ($\alpha = 1/2$): $\eta^\star = 4.0 \times 10^{-3}$.
- Saturating ($\alpha \approx 0.13$, the DeepSeek-style fit re-expressed in $B$): $\eta^\star \approx 1.4 \times 10^{-3}$.

The three predictions span $11\times$. Now the obstruction. Run all three at $B = 8$M. Suppose final validation losses come out $2.641$, $2.628$, $2.633$ nats, and the seed-to-seed standard deviation at fixed $(\eta, B)$ is $\sigma = 0.006$ nats. The spread between the best and worst is $0.013$ nats — about $2.2\sigma$ from one comparison, but with three arms and one seed each, a two-sided test at $p = 0.05$ does not reject "all equal". Fitting $\alpha$ from these three points gives $\hat{\alpha} = 0.5$ with a bootstrap CI of roughly $[0.1, 1.0]$: the data cannot distinguish an $11\times$ difference in the recommended learning rate.

To shrink the CI to width $0.4$ you need the standard error on $\log \eta^\star$ at each $B$ to drop by about $2.5\times$, which means roughly $6\times$ the seeds — from 3 runs to about 18 at the top batch size alone. That is the whole problem in one line: **the quantity is a noisy argmin of a flat function, and the cost of resolving it grows quadratically in the precision you want, at exactly the scale where each run is most expensive.** Every published exponent is a point estimate from a design that could not have separated these hypotheses.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*