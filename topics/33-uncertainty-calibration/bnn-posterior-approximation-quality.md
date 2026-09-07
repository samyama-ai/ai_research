---
id: 33-uncertainty-calibration/bnn-posterior-approximation-quality
title: "Bayesian Neural Network Posterior Approximation Quality Metrics"
topic: 33-uncertainty-calibration
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Bayesian Neural Network Posterior Approximation Quality Metrics

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/bnn-posterior-approximation-quality` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a neural network with parameters $\theta \in \mathbb{R}^d$, a prior $p(\theta)$, data $\mathcal{D}$, and an approximation $q$ to the posterior $p(\theta \mid \mathcal{D})$, produce a number $D(q, p)$ that (i) is computable without samples from $p$, (ii) is small only when $q$ is a good approximation, and (iii) predicts downstream uncertainty quality. No such number exists at deep-network scale.

Three variants, of very different difficulty:

- **Measurement.** Define and compute a discrepancy between $q$ and an intractable $p$ in $d \sim 10^6$–$10^{10}$ dimensions. This is the blocked variant.
- **Method.** Build approximations that score well. Well-supplied (VI, SG-MCMC, Laplace, ensembles) but unranked, because the ranking depends on the metric in (i).
- **Theory.** Prove approximation-quality bounds — which families can represent the true posterior predictive, and at what error. Partially answered for shallow mean-field networks; open in general.

Solving it means: a diagnostic that flags a bad approximation *before* the downstream task reveals it, and that does not merely re-measure test log-likelihood under a different name.

## 2. Formal Setting

Posterior and predictive:

$$p(\theta \mid \mathcal{D}) \propto p(\theta)\prod_{i=1}^{n} p(y_i \mid x_i, \theta), \qquad p(y \mid x, \mathcal{D}) = \int p(y \mid x,\theta)\, p(\theta \mid \mathcal{D})\, d\theta.$$

Candidate discrepancies, each as actually measured:

- **Weight-space KL.** $\mathrm{KL}(q \Vert p) = \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\theta,\mathcal{D})] + \log p(\mathcal{D})$. Measured only up to the unknown $\log p(\mathcal{D})$, so the ELBO gives a *relative* score within one model, not an absolute one, and not across model classes.
- **Kernel Stein discrepancy (KSD).** $\mathrm{KSD}(q) = \sup_{\|f\|_{\mathcal{H}}\le 1}\mathbb{E}_{\theta\sim q}[\nabla_\theta \log p(\theta,\mathcal{D})^\top f(\theta) + \nabla\cdot f(\theta)]$. Computed from $m$ samples as a $O(m^2)$ U-statistic using only $\nabla \log p$, so no normalizer is needed. Cost is $m^2 d$; kernel choice in $d>10^4$ is unresolved.
- **Function-space discrepancy.** On a held-out probe set $X^\star$ of size $k$, $D_{\mathrm{fn}} = \frac{1}{k}\sum_j \mathrm{JS}\!\left(q\text{-predictive}(\cdot\mid x_j^\star)\,\Vert\, p\text{-predictive}(\cdot\mid x_j^\star)\right)$, with the reference predictive estimated from a long HMC run. Measurable only where gold-standard HMC is affordable.
- **Chain diagnostics.** Split-$\hat R$ and effective sample size (Vehtari et al., 2021), applied per-coordinate in weight space or per-probe-point in function space.

**Assumptions, and their status.**
- *The reference posterior is reachable.* Violated above ~$10^7$ parameters; the only published full-batch HMC reference is CIFAR-scale (Izmailov et al., 2021).
- *Weight-space distances are meaningful.* Violated: permutation and sign symmetries make $p(\theta\mid\mathcal{D})$ multimodal with $\ge h!\,2^h$ equivalent modes per width-$h$ layer, so $q$ can have near-zero function-space error and enormous weight-space error.
- *The likelihood is the generative model.* Violated whenever data augmentation, label smoothing, or curation is used — the effective likelihood is not $\prod_i p(y_i\mid x_i,\theta)$ (Wenzel et al., 2020; Noci et al., 2021).
- *SG-MCMC samples the stated posterior.* Violated: minibatch noise plus finite step size leaves an $O(\epsilon)$ bias that is not quantified for real networks.

## 3. State of the Art

**Reference-posterior SOTA (established).** Izmailov et al., *What Are Bayesian Neural Network Posteriors Really Like?* (ICML 2021): full-batch HMC on CIFAR-10 with ResNet-20-FRN, hundreds of TPU-v3 cores, tens of thousands of leapfrog steps per sample. This is the field's only widely used gold standard, and it is a single architecture at one scale.

**Diagnostics SOTA (established as theory, unablated at scale).** Gorham & Mackey (NeurIPS 2015; ICML 2017) prove that KSD with an inverse multiquadric kernel detects non-convergence, and that Gaussian-kernel KSD does not in high dimension. No published study applies KSD to a $>10^6$-parameter BNN and shows it ranks approximations consistently with function-space error.

**Calibration-of-inference SOTA.** Simulation-based calibration (Talts et al., 2018), extending Cook–Gelman–Rubin (2006): sample $\theta_0\sim p(\theta)$, simulate data, check the rank of $\theta_0$ among posterior draws is uniform. Sound and exactly correct; applied to BNNs only on synthetic small-$d$ models. It tests the *sampler*, not the fit to real data.

**Claimed but unablated.** That a low test-set expected calibration error indicates a good posterior approximation. Ovadia et al. (NeurIPS 2019) showed the opposite in benchmark form: deep ensembles beat single-mode Bayesian methods on shifted data, while being a worse weight-space posterior approximation by construction.

**Benchmark-number-only results.** The NeurIPS 2021 *Evaluating Approximate Inference in Bayesian Deep Learning* competition ranked entrants by agreement with HMC reference predictives on CIFAR-10/IMDB-scale tasks. Rankings there are leaderboard positions on fixed reference runs, not a validated metric.

## 4. What Is Known

- **Mean-field VI has a representational hole.** Foong, Burt, Li & Turner (NeurIPS 2020): for a single-hidden-layer network, mean-field Gaussian VI cannot produce predictive variance in between two well-observed input clusters that exceeds the variance at the clusters. For $\ge 2$ hidden layers, the family is expressive enough — the failure is optimization, not capacity.
- **Depth beats covariance structure empirically.** Farquhar, Smith & Gal (NeurIPS 2020) report that mean-field posteriors match full-covariance/structured ones on deep networks across UCI, MNIST and CIFAR, contradicting the standard "mean-field is too restrictive" story.
- **Cold posteriors.** Wenzel et al. (ICML 2020): tempering the posterior to $T \ll 1$ improves held-out accuracy on ResNet-20/CIFAR-10 and CNN-LSTM/IMDB — i.e. the *correct* Bayes posterior is not the best predictor. Noci et al. (NeurIPS 2021) and Izmailov et al. (2021) attribute most of the effect to data augmentation and curation rather than a defective prior; without augmentation, the effect is small or absent.
- **HMC vs. cheap approximations at CIFAR scale.** Izmailov et al. (2021): HMC ResNet-20-FRN on CIFAR-10 reaches roughly 90% test accuracy, above single-chain SGD and comparable to or above deep ensembles in-distribution, while being *worse* than SGD-based baselines under CIFAR-10-C corruption. Better posterior approximation therefore does not imply better robustness.
- **Weight-space mixing does not happen.** In the same work, weight-space $\hat R$ across HMC chains is far from 1 while function-space predictive $\hat R$ is near 1 for most test points. Chains do not visit each other's modes; predictions still agree.
- **Multimodality is combinatorial.** A single hidden layer of width $h$ with an odd activation yields $\ge h!\,2^h$ symmetric copies of every mode.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted definition of "good posterior approximation" for a BNN that is (a) invariant to weight symmetries, (b) computable without a reference posterior, and (c) demonstrated to predict downstream uncertainty quality. Every current metric fails at least one: KL/ELBO fails (b) across model classes, function-space divergence fails (b), test NLL/ECE fail (c) in the direction of being the downstream task itself.
- **Empirically open.** Whether KSD (IMQ kernel), computed on $10^2$–$10^3$ draws from VI/SG-MCMC/Laplace on a $10^7$-parameter network, orders methods the same way as function-space distance to an HMC reference. Runnable today on the Izmailov reference; unpublished at that scale.
- **Empirically open.** Whether posterior-approximation quality measured in function space predicts OOD detection and selective-prediction performance, or is orthogonal to it. Current evidence (Ovadia 2019; Izmailov 2021) suggests orthogonality but was not designed to test it.
- **Theoretically open.** Non-asymptotic bounds relating a weight-space discrepancy $D(q,p)$ to predictive-distribution error $\sup_x \mathrm{TV}(q\text{-pred}, p\text{-pred})$ for non-Lipschitz-controlled deep networks. Also open: the exact stationary-distribution bias of SG-MCMC with preconditioning and augmentation at practical step sizes.

## 6. Why It Is Hard

Three concrete obstructions, none of which is "importance".

1. **Absent ground truth at scale.** The reference requires HMC; HMC on a ResNet needs $\sim10^4$–$10^5$ full-dataset gradient evaluations per effective sample, i.e. hundreds of accelerator-months for one architecture. No reference exists above roughly $10^7$ parameters, so any metric can only be validated in a regime where it is not needed.
2. **Non-identifiability.** Permutation/scaling symmetry means the target is a distribution over an equivalence class, not a point cloud. Weight-space metrics (KL, Wasserstein, KSD) measure a quantity that is provably large for a perfectly good approximation supported on one mode. Quotienting by the symmetry group is $\\#P$-hard to do exactly.
3. **Evaluation that does not measure what it names.** ECE and test NLL are called "posterior quality" but score the predictive under the *empirical* test distribution. Since tempered ($T\ll1$) posteriors — which are further from $p(\theta\mid\mathcal{D})$ — score better on both, these metrics are anti-correlated with approximation quality in a documented regime.

## 7. Current Research (as of 2026)

- **Reference posteriors beyond CIFAR.** Extending Izmailov-style HMC references to transformer-scale models and to fine-tuning-only posteriors (low-rank subspaces), where $d$ is $10^5$–$10^7$ rather than $10^9$. *(frontier — verify)*
- **Symmetry-aware comparison.** Aligning networks by neuron matching before measuring weight-space distance, borrowing from linear-mode-connectivity work (Ainsworth, Hayase & Srinivasa, ICLR 2023). Whether alignment makes weight-space discrepancies informative is untested for posterior *distributions* rather than point estimates. *(frontier — verify)*
- **Laplace as the practical default.** `laplace-torch` (Daxberger et al., NeurIPS 2021) made last-layer and KFAC Laplace cheap; the open question is which of its predictive approximations (probit, linearized, MC) is closest to the true predictive, not just best-scoring.
- **Position and agenda work.** Papamarkou et al., *Position: Bayesian Deep Learning is Needed in the Age of Large-Scale AI* (ICML 2024), argues explicitly for inference diagnostics as a prerequisite; Papamarkou et al. (*Statistical Science*, 2022) catalogs the MCMC obstacles.

## 8. Concrete Next Experiment

**Question.** Does any reference-free diagnostic order approximation methods the same way as distance to the true posterior predictive?

- **Scale.** ResNet-20-FRN on CIFAR-10 ($\approx 2.7\times10^5$ parameters), using the published full-batch HMC chains as the reference predictive on the 10,000-image test set plus CIFAR-10-C at severity 3.
- **Arms.** Five approximations: mean-field VI, SWAG, KFAC Laplace, cyclical SG-HMC, deep ensemble ($M=10$). Each gives $m=100$ weight draws.
- **Control arm.** Two independent HMC chains from different initializations, treated as an approximation of each other. This sets the noise floor: any diagnostic that scores the HMC–HMC pair worse than a VI–HMC pair is measuring symmetry, not quality.
- **Measurements.** (a) Reference-free: IMQ-kernel KSD on each arm's draws. (b) Reference-based: mean Jensen–Shannon divergence between arm predictive and HMC predictive, per test point.
- **Deciding number.** Spearman rank correlation $\rho$ between (a) and (b) across the five arms plus the control. **$\rho \ge 0.9$** means KSD is usable as a reference-free diagnostic and the problem moves from methodologically blocked to empirically open. **$\rho \le 0.5$**, or the HMC–HMC control scoring worse than any approximation arm, means weight-space Stein diagnostics are dead for BNNs and the field should standardize on function-space references — which requires funding reference runs, not new math.
- **Cost.** Under 500 GPU-hours excluding the already-published HMC chains.

## 9. Key References

- **[Foundational]** Radford M. Neal. *Bayesian Learning for Neural Networks.* PhD thesis, University of Toronto, 1995 (Springer, 1996).
- **[Foundational]** Max Welling, Yee Whye Teh. *Bayesian Learning via Stochastic Gradient Langevin Dynamics.* ICML, 2011.
- **[Foundational]** Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra. *Weight Uncertainty in Neural Networks.* ICML, 2015. — arXiv:1505.05424
- **[SOTA]** Pavel Izmailov, Sharad Vikram, Matthew D. Hoffman, Andrew Gordon Wilson. *What Are Bayesian Neural Network Posteriors Really Like?* ICML, 2021. — arXiv:2104.14421
- **[SOTA]** Florian Wenzel et al. *How Good is the Bayes Posterior in Deep Neural Networks Really?* ICML, 2020. — arXiv:2002.02405
- **[SOTA]** Jackson Gorham, Lester Mackey. *Measuring Sample Quality with Kernels.* ICML, 2017. — arXiv:1703.01717
- **[SOTA]** Andrew Y. K. Foong, David R. Burt, Yingzhen Li, Richard E. Turner. *On the Expressiveness of Approximate Inference in Bayesian Neural Networks.* NeurIPS, 2020.
- **[SOTA]** Sebastian Farquhar, Lewis Smith, Yarin Gal. *Liberty or Depth: Deep Bayesian Neural Nets Do Not Need Complex Weight Posterior Approximations.* NeurIPS, 2020.
- **[SOTA]** Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, Philipp Hennig. *Laplace Redux — Effortless Bayesian Deep Learning.* NeurIPS, 2021.
- **[Method]** Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, Andrew Gelman. *Validating Bayesian Inference Algorithms with Simulation-Based Calibration.* 2018. — arXiv:1804.06788
- **[Method]** Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, Paul-Christian Bürkner. *Rank-Normalization, Folding, and Localization: An Improved $\hat{R}$ for Assessing Convergence of MCMC.* Bayesian Analysis, 2021.
- **[Empirical]** Yaniv Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[Empirical]** Lorenzo Noci, Kevin Roth, Gregor Bachmann, Sebastian Nowozin, Thomas Hofmann. *Disentangling the Roles of Curation, Data-Augmentation and the Prior in the Cold Posterior Effect.* NeurIPS, 2021.
- **[Survey]** Theodore Papamarkou, Jacob Hinkle, M. Todd Young, David Womble. *Challenges in Markov Chain Monte Carlo for Bayesian Neural Networks.* Statistical Science, 2022.
- **[Survey]** Theodore Papamarkou et al. *Position: Bayesian Deep Learning is Needed in the Age of Large-Scale AI.* ICML, 2024.

## 10. Worked Example

Take a 2-layer MLP, 50 hidden units, on a binary task: $d = 50\cdot p + 50 + 50 + 1$ parameters. Hidden-layer permutation and $\tanh$ sign symmetry give

$$50!\cdot 2^{50} \approx 3.04\times10^{64}\cdot 1.13\times10^{15} \approx 3.4\times10^{79}$$

exact copies of every posterior mode. Now run two long HMC chains, $A$ and $B$, from different inits. Both concentrate near one mode each; with probability $\approx 1 - 3\times10^{-80}$ those modes are different symmetric copies.

Measure three ways:

| Comparison | Weight-space 2-Wasserstein (relative) | Function-space mean JS on 1,000 probe points | Test NLL |
|---|---|---|---|
| HMC $A$ vs HMC $B$ | large — modes are far apart | $\approx 0$ | identical |
| MFVI vs HMC $A$ | comparable, sometimes smaller | clearly $>0$ | MFVI worse |

The weight-space metric ranks a *perfect* sample (HMC $B$) as being as bad as, or worse than, a known-deficient one (MFVI). That is the obstruction in one table: the metric measures which symmetric copy you landed on, not how well you approximate the posterior.

Function space fixes the ranking but not the problem — computing column 2 required HMC chain $A$, the very object you cannot obtain at scale. And column 3 does not rescue it either: at $T=10^{-2}$ a tempered chain, which is *further* from $p(\theta\mid\mathcal{D})$ in every sense, typically wins on test NLL. Three candidate metrics, three different orderings, and no reference-free one that is symmetry-invariant. That is why this entry is filed as methodologically blocked rather than merely expensive.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*