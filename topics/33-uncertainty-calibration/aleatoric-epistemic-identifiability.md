---
id: 33-uncertainty-calibration/aleatoric-epistemic-identifiability
title: "Separating Aleatoric from Epistemic Uncertainty Identifiably"
topic: 33-uncertainty-calibration
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Separating Aleatoric from Epistemic Uncertainty Identifiably

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/aleatoric-epistemic-identifiability` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a predictor and an input $x$, split its predictive uncertainty into **aleatoric** (AU — irreducible label noise at $x$) and **epistemic** (EU — reducible ignorance about the predictor) parts, such that the split is *identifiable*: determined by observable quantities rather than by an arbitrary choice of hypothesis class, prior, or ensemble construction.

Three variants, of very different difficulty:

- **Measurement.** Define an estimand $\mathrm{AU}(x)$ and $\mathrm{EU}(x)$ with a ground truth that can be checked against data. Blocked: for EU there is no observable target at all outside synthetic settings.
- **Method.** Produce estimators that behave differently from each other — one tracking label ambiguity, one tracking distribution shift / data scarcity. Empirically open, and currently failing: existing estimators are near-collinear.
- **Theory.** Prove that a loss or elicitation procedure incentivizes a faithful second-order belief. Settled negatively for the standard formulation (§4).

Solving it means: a decomposition rule plus an evaluation protocol under which two estimators disagree on the ground-truth AU/EU targets in a *double dissociation* — the AU estimator beats the EU estimator on the AU task and loses on the EU task — and under which the decomposition is invariant to hypothesis-class relabelings that leave the data distribution unchanged.

## 2. Formal Setting

Data $(x,y) \sim P$, $x \in \mathcal{X}$, $y \in \mathcal{Y}=\{1,\dots,K\}$. The Bayes-optimal conditional is $\eta(x) = P(\cdot \mid x) \in \Delta^{K-1}$.

A **second-order learner** returns a distribution $Q_x$ over first-order predictors $\theta \mapsto p_\theta(\cdot\mid x)$: a Bayesian posterior, an ensemble empirical measure over $M$ members, or a Dirichlet output. The predictive distribution is $\bar p_x = \mathbb{E}_{\theta \sim Q_x}[p_\theta(\cdot \mid x)]$.

The standard information-theoretic decomposition (Depeweg et al., ICML 2018):

$$\underbrace{H(\bar p_x)}_{\text{total}} \;=\; \underbrace{\mathbb{E}_{\theta\sim Q_x}\big[H(p_\theta(\cdot\mid x))\big]}_{\widehat{\mathrm{AU}}(x)} \;+\; \underbrace{I(y;\theta \mid x)}_{\widehat{\mathrm{EU}}(x)}$$

**How each quantity is actually measured.**

- $\widehat{\mathrm{AU}}, \widehat{\mathrm{EU}}$: Monte-Carlo over $M$ ensemble members or MC-dropout samples, typically $M \in \{5,10,20\}$. Both are $O(1/M)$-biased; $\widehat{\mathrm{EU}}$ is upward-biased and $\widehat{\mathrm{AU}}$ downward, so $M$ alone moves the split.
- Ground-truth AU: soft labels from repeated human annotation. $\mathrm{AU}^\star(x) = H(\hat\eta(x))$ with $\hat\eta(x)$ the empirical label frequency over $R$ annotators. Sampling error on entropy is $O(K/R)$ — at $R{=}51$ (CIFAR-10H) and $K{=}10$ the plug-in entropy bias is on the order of $0.1$ bits, comparable to the effect being measured.
- Ground-truth EU: no direct observable. Proxies used are OOD-vs-ID membership, train-set density, or error on a held-out shifted split — none of which is a *quantity* of the predictor.

**Assumptions, and which fail.**

1. *$\eta$ is well defined and the annotator distribution equals it.* Violated: annotator soft labels mix genuine ambiguity with annotator incompetence and label-guideline drift.
2. *The model class contains $\eta$ (well-specification).* Violated for every deep net; under misspecification $\mathbb{E}_\theta H(p_\theta)$ does not converge to $H(\eta)$.
3. *$Q_x$ is a faithful posterior.* Violated: ensembles and dropout are not posteriors, and $\widehat{\mathrm{EU}}$ depends on the initialization distribution, which is a design choice.
4. *AU is invariant to the learner.* Violated by construction — see §10.

## 3. State of the Art

**Theory SOTA (established).** Bengs, Hüllermeier & Waegeman (NeurIPS 2022; ICML 2023) prove impossibility results: no proper scoring rule over second-order predictions gives a learner an incentive to report its epistemic uncertainty faithfully; the loss minimizer collapses toward degenerate (zero-EU) second-order distributions regardless of data quantity. Wimmer et al. (UAI 2023) show the entropy decomposition above violates natural axioms one would demand of an AU/EU split — the "aleatoric" term is not monotone in information gain and depends on the reference measure.

**Empirical SOTA (benchmark numbers only).** Mucsányi, Kirchhof & Oh (NeurIPS 2024 Datasets & Benchmarks) benchmark ~10 disentanglement methods on ImageNet-1k and CIFAR-10 and report that AU and EU estimates from the same method are near-rank-equivalent, with correlations frequently above $0.9$; no method achieves a double dissociation across AU-specific and EU-specific tasks. This is a benchmark result, not a theorem, and it is one benchmark.

**Claimed but unablated.** Evidential/prior networks (Malinin & Gales, NeurIPS 2018; Sensoy et al. 2018) claim a single forward pass yields separated AU/EU. The claim rests on OOD-detection AUROC, which does not test separation; ablations showing the "epistemic" head is not simply a monotone function of max-softmax are largely absent. Credal-set and distance-based second-order measures (Sale et al., ICML 2024; Hofman et al. 2024) fix specific axiom violations but have not been shown to disentangle on real data.

## 4. What Is Known

- **Impossibility for loss-based elicitation.** Second-order proper scoring rules cannot elicit faithful EU (Bengs et al., NeurIPS 2022 / ICML 2023). Established, with proof.
- **Axiom violations of the MI decomposition.** Wimmer et al. (UAI 2023) give explicit counterexamples where $\widehat{\mathrm{AU}}$ increases as the learner becomes more informed.
- **Estimator collinearity at scale.** ImageNet-1k, ResNet-50, $M{=}5$–$10$: reported AU/EU rank correlations above $0.9$ for most methods (Mucsányi et al. 2024). At that correlation the two numbers carry roughly one degree of freedom, not two.
- **Human soft labels exist and are informative.** CIFAR-10H: 511,400 human labels over the 10,000 CIFAR-10 test images (~51 per image; Peterson et al., ICCV 2019). ImageNet-ReaL: 50,000 validation images relabeled with multi-label annotations (Beyer et al., 2020). Training on soft labels improves robustness — evidence that $\hat\eta$ carries real signal about AU.
- **$M$-dependence.** MI-based EU is biased upward at small $M$; the finite-sample bias of plug-in mutual information is $O((M{-}1)(K{-}1)/2M)$ nats in the standard regime, non-negligible at $M{=}5$, $K{=}1000$.
- **Valdenegro-Toro & Mori (CVPRW 2022)** found EU estimates from ensembles/dropout/flipout on CIFAR-10 do not vanish with more data as the definition requires, and vary more with architecture than with dataset size.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no observable ground truth for EU. Every current "EU benchmark" substitutes an OOD label, so a perfect OOD detector scores perfectly without estimating any epistemic quantity. Until EU has a measurable estimand, method comparisons cannot be adjudicated.
- **Theoretically open.** Whether *any* identifiable decomposition exists once the hypothesis class is treated as a free parameter — i.e. whether there is a functional $F(P, \text{observables}) \to (\mathrm{AU},\mathrm{EU})$ invariant to reparameterizations of the learner. No proof either way. Also open: whether the Bengs et al. impossibility survives if the learner is scored on *sequences* of decisions (bandit regret, active-learning gains) rather than one-shot predictions.
- **Empirically open.** Whether collinearity persists when AU ground truth is strong. No study has run the double-dissociation matrix on CIFAR-10H/ImageNet-ReaL with a total-uncertainty control arm at $M \ge 20$.
- **Empirically open.** Whether EU estimates decay at the $O(n^{-1})$ rate the definition implies, measured over four decades of training-set size at fixed architecture.

## 6. Why It Is Hard

**Non-identifiability, plus an evaluation that does not measure what it names.**

The data distribution $P$ constrains only $\eta(x)$, hence only the *total*. The split into $\mathbb{E}_\theta H(p_\theta)$ and $I(y;\theta)$ is a property of $Q_x$, and $Q_x$ is chosen, not observed. Two learners with identical predictive distributions, identical held-out likelihood, and identical calibration can report opposite decompositions (§10). No amount of held-out data distinguishes them, because they agree on every observable.

Compounding this: the standard EU benchmark is OOD detection, which correlates with total uncertainty by construction, so a method can score well while its "epistemic" head is a monotone transform of its "aleatoric" head. That is why collinearity above $0.9$ went unnoticed for years — the evaluation rewarded it.

## 7. Current Research (as of 2026)

- **Axiomatic second-order UQ** — Hüllermeier's group (LMU Munich) and collaborators: credal sets, distance-based measures, and formal desiderata for AU/EU functionals; explicitly motivated by the entropy decomposition's failures.
- **Disentanglement benchmarking** — Kirchhof, Mucsányi, Oh (Tübingen) and Gal's group (Oxford): task-specific evaluation, where each uncertainty is judged on the task it is supposed to serve rather than on OOD AUROC.
- **Impossibility and elicitation** — Bengs/Waegeman line, extending negative results to sequential and decision-theoretic settings *(frontier — verify)*.
- **LLM uncertainty** — semantic-entropy style estimators for free-form generation, where AU/EU separation is even less defined because $\mathcal{Y}$ is unbounded and human "soft labels" are ill-posed *(frontier — verify)*.
- **Human-label-informed AU** — soft-label datasets and annotator models as the only available AU ground truth.

## 8. Concrete Next Experiment

**The double-dissociation matrix with a total-uncertainty control.**

- **Scale.** ImageNet-1k, ResNet-50 and ViT-B/16, deep ensemble $M{=}20$ (bias at $M{=}5$ is too large), plus MC-dropout, evidential head, and SNGP. ~$6\times10^2$ GPU-hours.
- **Targets.** AU target: per-image human soft-label entropy from ImageNet-ReaL (50k images), restricted to images with $\ge 5$ annotations. EU target: reduction in test log-loss at $x$ when the training set grows from $n$ to $10n$, measured by training the same architecture on $n \in \{10^4,10^5,10^6\}$ subsets — an operational EU, defined as realized error reduction rather than an OOD label.
- **Control arm.** A single scalar — predictive entropy $H(\bar p_x)$ — used to predict *both* targets. This is the arm the field has never run, and it is what makes the result interpretable.
- **Deciding number.** $\Delta = \rho_s(\widehat{\mathrm{AU}}, \mathrm{AU}^\star) - \rho_s(H(\bar p_x), \mathrm{AU}^\star)$ and symmetrically for EU. A decomposition is doing work only if **both** gaps exceed $+0.05$ Spearman with bootstrap 95% CI excluding zero, while $\rho_s(\widehat{\mathrm{AU}},\widehat{\mathrm{EU}}) < 0.5$. Current expectation from the 2024 benchmark: $\Delta \approx 0$ and cross-correlation $> 0.9$ — i.e. the two numbers are one number.

## 9. Key References

- **[Foundational]** A. Kendall, Y. Gal. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS, 2017. — arXiv:1703.04977
- **[Foundational]** S. Depeweg, J. M. Hernández-Lobato, F. Doshi-Velez, S. Udluft. *Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning.* ICML, 2018. — arXiv:1710.07283
- **[Survey]** E. Hüllermeier, W. Waegeman. *Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods.* Machine Learning 110(3), 2021. — arXiv:1910.09457
- **[Theory]** V. Bengs, E. Hüllermeier, W. Waegeman. *Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation.* NeurIPS, 2022.
- **[Theory]** V. Bengs, E. Hüllermeier, W. Waegeman. *On Second-Order Scoring Rules for Epistemic Uncertainty Quantification.* ICML, 2023.
- **[Theory]** L. Wimmer, Y. Sale, P. Hofman, B. Bischl, E. Hüllermeier. *Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning: Are Conditional Entropy and Mutual Information Appropriate Measures?* UAI, 2023.
- **[SOTA]** B. Mucsányi, M. Kirchhof, S. J. Oh. *Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks.* NeurIPS Datasets & Benchmarks, 2024.
- **[SOTA]** Y. Sale, M. Caprio, E. Hüllermeier. *Second-Order Uncertainty Quantification: A Distance-Based Approach.* ICML, 2024.
- **[Empirical]** M. Valdenegro-Toro, D. S. Mori. *A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement.* CVPR Workshops, 2022.
- **[Data]** J. C. Peterson, R. M. Battleday, T. L. Griffiths, O. Russakovsky. *Human Uncertainty Makes Classification More Robust.* ICCV, 2019. — arXiv:1908.07086
- **[Data]** L. Beyer, O. J. Hénaff, A. Kolesnikov, X. Zhai, A. van den Oord. *Are We Done with ImageNet?* 2020. — arXiv:2006.07159
- **[Baseline]** B. Lakshminarayanan, A. Pritzel, C. Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS, 2017. — arXiv:1612.01474
- **[Baseline]** A. Malinin, M. Gales. *Predictive Uncertainty Estimation via Prior Networks.* NeurIPS, 2018. — arXiv:1802.10501

## 10. Worked Example

One input $x$, binary labels, true $\eta(x) = (0.7, 0.3)$. Total predictive entropy is fixed by the data:

$$H(0.7) = -0.7\log_2 0.7 - 0.3\log_2 0.3 = 0.881 \text{ bits.}$$

**Learner A — probabilistic, well-specified.** Trained to convergence on $n \to \infty$ samples; every ensemble member outputs $p_\theta = (0.7,0.3)$.

$$\widehat{\mathrm{AU}} = \mathbb{E}_\theta H(p_\theta) = 0.881, \qquad \widehat{\mathrm{EU}} = 0.881 - 0.881 = 0.000.$$

**Learner B — deterministic hypothesis class.** Members are hard classifiers, $p_\theta \in \{(1,0),(0,1)\}$; 70% of members converge to class 1 (each fits its bootstrap majority).

$$\bar p_x = (0.7,0.3), \quad \widehat{\mathrm{AU}} = \mathbb{E}_\theta H(p_\theta) = 0.000, \qquad \widehat{\mathrm{EU}} = 0.881 - 0.000 = 0.881.$$

**The obstruction, visible.** A and B produce the *same* predictive distribution $(0.7,0.3)$. They have the same test log-loss ($0.881$ bits), the same ECE ($0$), the same accuracy ($70\%$), the same Brier score. Every observable agrees. Yet A reports "all noise, no ignorance" and B reports "no noise, all ignorance" — a $0.881$-bit swing, the entire budget, driven purely by the choice of hypothesis class.

Worse, B's EU does not shrink with data: at $n = 10^6$ each member still commits to a hard label and the members still split 70/30, so $\widehat{\mathrm{EU}} = 0.881$ forever. This contradicts the defining property of epistemic uncertainty (reducible by data) while remaining a valid instance of the standard decomposition. That is the identifiability failure: the split is a function of $Q_x$, and $Q_x$ is not identified by $P$.