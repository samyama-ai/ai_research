---
id: 33-uncertainty-calibration/ood-detection-without-reference
title: "Out-of-Distribution Detection Without a Reference Distribution"
topic: 33-uncertainty-calibration
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Out-of-Distribution Detection Without a Reference Distribution

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/ood-detection-without-reference` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a model trained on data from a distribution $P_{\text{in}}$ and a single input $x$, decide whether $x$ came from $P_{\text{in}}$ — using only the training data and the model, with no sample, parametric family, or benchmark split standing in for the "outside".

Three variants, with different difficulty:

- **Measurement.** Define a target quantity that a detector could be scored against when no outlier sample exists. Every current metric (AUROC, FPR@95TPR) is computed against a chosen $P_{\text{out}}$. Without one, there is no label to score against. This variant is the blocked one.
- **Method.** Build a scorer $s: \mathcal{X} \to \mathbb{R}$ whose threshold is set from in-distribution data alone (e.g. a target false-positive rate), not tuned on outliers. Runnable today; rarely done cleanly.
- **Theory.** Characterise when a detector learned from $P_{\text{in}}$ alone generalises to *any* $P_{\text{out}}$. Partly settled, and the answer is largely negative (Fang et al., NeurIPS 2022).

Solving it means: a procedure that, given only $P_{\text{in}}$ samples and a target in-distribution rejection rate $\alpha$, produces a rejection region with a *stated, checkable* guarantee that does not name the alternative.

## 2. Formal Setting

Input space $\mathcal{X}$, label space $\mathcal{Y}=\{1,\dots,K\}$. Training set $D=\{(x_i,y_i)\}_{i=1}^n \sim P_{\text{in}}^n$. Model $f_\theta$ with logits $z(x)\in\mathbb{R}^K$ and penultimate features $\phi(x)\in\mathbb{R}^d$.

A detector is a score $s_\theta$ and threshold $\tau$; reject when $s_\theta(x) < \tau$.

**Calibrated on ID only.** Set $\tau$ as the empirical $\alpha$-quantile of scores on a held-out ID split $D_{\text{cal}}$ of size $m$:
$$\tau_\alpha = \hat{Q}_\alpha\big(\{s_\theta(x)\}_{x\in D_{\text{cal}}}\big).$$
Measured: run $m$ forward passes, sort, take index $\lceil \alpha(m+1)\rceil$. Under exchangeability of $D_{\text{cal}}$ and future ID inputs, this gives a distribution-free ID false-rejection rate $\le \alpha$ (split conformal, Vovk et al.). It says **nothing** about OOD recall.

**Standard scores, as computed.**
- MSP: $s=\max_k \mathrm{softmax}(z(x))_k$.
- Energy: $s=T\log\sum_k e^{z_k(x)/T}$, $T=1$ in practice.
- Mahalanobis: $s=-\min_k(\phi(x)-\mu_k)^\top \hat\Sigma^{-1}(\phi(x)-\mu_k)$, with $\mu_k$ the class means and $\hat\Sigma$ the pooled covariance estimated on $D$.
- kNN: $s=-\|\phi(x)-\phi_{(k)}\|_2$ on $\ell_2$-normalised features, $k=50$ typical.
- Likelihood: $s=\log p_\psi(x)$ from a separately trained density model.

**Evaluation, as actually done.** Pick $P_{\text{out}}$, draw $N$ samples, report
$$\mathrm{AUROC}=\Pr\big[s(X_{\text{in}})>s(X_{\text{out}})\big],\qquad \mathrm{FPR}@95=\Pr\big[s(X_{\text{out}})>\tau_{0.05}\big].$$
Both are functionals of the *pair* $(P_{\text{in}},P_{\text{out}})$. There is no $P_{\text{out}}$-free reduction of either.

**Assumptions, and which break.**
1. *$P_{\text{out}}$ is disjoint in support from $P_{\text{in}}$.* Violated: benchmark OOD splits contain ID objects (§4).
2. *A single scalar score suffices for all alternatives.* Violated by construction — the likelihood-ratio-optimal score depends on $P_{\text{out}}$ (Neyman–Pearson).
3. *Density is a meaningful "insideness" measure.* Violated: $\log p(x)$ is not invariant to a change of variables. For any $x$ and any target density value, a smooth reparameterisation exists making $x$ look typical or atypical (Le Lan & Dinh, *Entropy* 2021).
4. *Calibration data is exchangeable with deployment inputs.* Violated whenever deployment shifts — which is the case OOD detection exists to catch.

## 3. State of the Art

**Empirical SOTA (established).** Post-hoc scores over a strong closed-set classifier. On OpenOOD v1.5 (Zhang et al., 2023), across ~20 post-hoc methods on CIFAR-10/100 and ImageNet-1K: no method wins on both near-OOD and far-OOD; ranking flips between the two regimes. Established by that benchmark's own ablation, and independently anticipated by Tajwar et al. (2021), "No True State-of-the-Art? OOD Detection Methods are Inconsistent across Datasets".

**Established, reproduced:** improving closed-set accuracy improves open-set AUROC. Vaze et al. (ICLR 2022) show a strong positive correlation across architectures and training recipes; a well-trained baseline with MSP matches or beats bespoke open-set methods on standard splits.

**Claimed but unablated:** most "SOTA" gains from methods with an outlier-dependent hyperparameter — ODIN's temperature and input-perturbation magnitude (Liang et al., ICLR 2018) were originally tuned on OOD data; ReAct's activation-clipping percentile and Mahalanobis's layer-ensemble weights are chosen with reference to some outlier set. When tuning is restricted to ID data, reported margins shrink; the size of the shrinkage has not been measured uniformly across methods.

**Benchmark-number-only:** vision-language scores (MCM, Ming et al., NeurIPS 2022) and Outlier Exposure (Hendrycks et al., ICLR 2019) report large gains, but OE explicitly uses an auxiliary outlier corpus — it solves a different problem (the reference *is* given) and its numbers should not be read as evidence on this one.

**Theory SOTA.** Fang et al. (NeurIPS 2022, "Is Out-of-Distribution Detection Learnable?") give PAC-style impossibility: in the total space of distributions, OOD detection is not learnable; learnability holds only under restrictions (finite ID/OOD domain space, or a separate-space condition), with necessary-and-sufficient conditions in those cases.

## 4. What Is Known

- **Density models rank OOD inputs above ID inputs.** A Glow trained on CIFAR-10 assigns SVHN *lower* bits-per-dimension (≈2.4) than CIFAR-10 test (≈3.4) — higher likelihood to the outliers (Nalisnick et al., ICLR 2019). Reproduced for PixelCNN and VAEs, at 32×32 scale.
- **The effect is largely explained by input complexity.** Correcting $\log p(x)$ by a general-purpose compressor's code length removes most of the inversion (Serrà et al., ICLR 2020); Zhang et al. (ICML 2021) trace remaining failures to model misestimation, not just typicality.
- **Standard ImageNet OOD splits are contaminated.** Bitterwolf et al. (ICML 2023, NINCO) manually inspected the usual splits (iNaturalist/SUN/Places subsets, Textures, OpenImage-O) and found a large fraction of images containing ID-class objects; they release a cleaned set of 64 OOD classes (~5.9k images). Method rankings change on the clean set.
- **Near-OOD is hard at ImageNet scale.** OpenOOD v1.5 near-OOD AUROC on ImageNet-1K sits in the mid-to-high 70s for the best post-hoc methods, only a few points above MSP; far-OOD exceeds 90 for several methods. Same backbone, same scale — the gap is the regime, not the model.
- **Reparameterisation invalidates density scores.** Le Lan & Dinh (2021): a perfect density model does not guarantee anomaly detection; the answer depends on the choice of base measure on $\mathcal{X}$.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no accepted definition of the target quantity when $P_{\text{out}}$ is unnamed. AUROC against a chosen split measures "distinguishes $P_{\text{in}}$ from *this* $P_{\text{out}}$", not "detects the outside". No proposed replacement — worst-case AUROC over a distribution class, minimum-volume set coverage, semantic-shift-only recall — is standard, and each smuggles in a class or a metric on $\mathcal{X}$.
- **Theoretically open.** What is the *largest* class $\mathcal{P}$ of alternatives for which a single ID-trained score achieves uniform non-trivial power, for realistic $\mathcal{X}$ and deep feature maps? Fang et al. bound the extremes; the useful middle is uncharacterised.
- **Empirically open.** How much of the reported method-over-MSP margin survives ID-only hyperparameter selection, on clean splits, at ImageNet scale? Runnable now; not run uniformly.
- **Empirically open.** Whether representation scale (a 10B-parameter vision or vision-language encoder) shrinks the near-OOD gap or only moves the decision boundary.

## 6. Why It Is Hard

**Non-identifiability plus an evaluation that does not measure what it names.**

Non-identifiability: by Neyman–Pearson, the optimal test of $P_{\text{in}}$ against $P_{\text{out}}$ is the likelihood ratio $p_{\text{out}}/p_{\text{in}}$. With $P_{\text{out}}$ unspecified, no ordering of $\mathcal{X}$ is optimal for all alternatives; and any ordering that is good for one is provably bad for another (a score that ranks blurry inputs as OOD fails on adversarially sharp OOD). Le Lan & Dinh sharpen this: even the *ideal* density model gives a score whose ordering is an artifact of the coordinate system.

Absent ground truth: "OOD" in practice means "semantically outside the label set", which is a human judgement, not a property of $P_{\text{in}}$. NINCO shows the judgements in existing benchmarks are wrong often enough to reorder methods. So the community measures agreement with a mislabelled proxy and calls it detection rate.

Confounded measurement: reported gains conflate (a) better score, (b) better backbone, (c) hyperparameters chosen using the test-time outliers. Only (a) is the claim. Compute is *not* the obstruction — the whole benchmark suite runs in GPU-hours.

## 7. Current Research (as of 2026)

- **Benchmark hygiene.** OpenOOD (Yang, Zhou, Li et al.) and NINCO (Tübingen: Bitterwolf, Müller, Hein) — clean splits, unified protocols, ID-only tuning rules. Most credible near-term progress.
- **Learnability theory.** Fang, Liu, Han, Sugiyama and collaborators — extending PAC conditions beyond the separate-space case *(frontier — verify)*.
- **Conformal / distribution-free framing.** ID-side guarantees at fixed $\alpha$, with power reported descriptively. Sidesteps the definitional problem rather than solving it.
- **Foundation-model scores.** CLIP/SigLIP-based MCM-style scores and "ask the VLM whether the object is in the label set" — turns the reference distribution into the model's own vocabulary, which is a reference distribution by another name.
- **Task-relative OOD.** Redefining the target as "will the model be wrong here" (failure prediction) rather than "is this outside". Well-posed and measurable; a different problem, and arguably the honest replacement.

## 8. Concrete Next Experiment

**Question:** how much of the post-hoc OOD literature's advantage over MSP survives when no outlier data touches any design decision?

**Scale.** ImageNet-1K, one fixed backbone (ResNet-50 and ViT-B/16, both public checkpoints). 12 post-hoc scores from OpenOOD v1.5 (MSP, ODIN, Energy, Mahalanobis, ReAct, kNN, ViM, GEN, ASH, DICE, MLS, KL-Matching). Evaluate on NINCO (clean near-OOD) plus two far-OOD sets. ~200 GPU-hours total; no training.

**Protocol.** Every hyperparameter — temperature, perturbation size, clipping percentile, $k$, layer weights — selected by a *pre-registered* ID-only rule (e.g. maximise ID-score variance, or minimise leave-one-out ID quantile instability on a held-out ID split). Thresholds set to $\alpha=0.05$ on ID calibration data only.

**Control arm.** The same 12 scores with hyperparameters tuned to maximise AUROC on the *test* OOD set — the implicit protocol of much of the literature. This is the upper bound, not a baseline.

**Deciding number.** $\Delta = \mathrm{AUROC}^{\text{ID-tuned}}_{\text{best method}} - \mathrm{AUROC}^{\text{ID-tuned}}_{\text{MSP}}$ on NINCO. If $\Delta < 2$ AUROC points while the outlier-tuned control shows $>8$ points, the field's progress is mostly a tuning artifact and the reference-free problem is untouched. If $\Delta > 5$, at least one score carries genuine ID-only signal and the method variant is live.

## 9. Key References

- **[Foundational]** Hendrycks, Gimpel. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR 2017. — arXiv:1610.02136
- **[Foundational]** Nalisnick, Matsukawa, Teh, Gorur, Lakshminarayanan. *Do Deep Generative Models Know What They Don't Know?* ICLR 2019. — arXiv:1810.09136
- **[Theory]** Fang, Li, Lu, Gong, Han, Liu. *Is Out-of-Distribution Detection Learnable?* NeurIPS 2022. — arXiv:2210.14707
- **[Theory]** Le Lan, Dinh. *Perfect Density Models Cannot Guarantee Anomaly Detection.* Entropy, 2021. — arXiv:2012.03808
- **[SOTA/Benchmark]** Zhang, Yang, Wang, et al. *OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection.* 2023. — arXiv:2306.09301
- **[SOTA/Benchmark]** Bitterwolf, Müller, Hein. *In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation.* ICML 2023.
- **[SOTA]** Sun, Ming, Zhu, Li. *Out-of-Distribution Detection with Deep Nearest Neighbors.* ICML 2022. — arXiv:2204.06507
- **[SOTA]** Liu, Wang, Owens, Li. *Energy-based Out-of-distribution Detection.* NeurIPS 2020. — arXiv:2010.03759
- **[Analysis]** Serrà, Álvarez, Gómez, Slizovskaia, Núñez, Luque. *Input Complexity and Out-of-Distribution Detection with Likelihood-based Generative Models.* ICLR 2020. — arXiv:1909.11480
- **[Analysis]** Vaze, Han, Vedaldi, Zisserman. *Open-Set Recognition: A Good Closed-Set Classifier is All You Need?* ICLR 2022.
- **[Survey]** Yang, Zhou, Li, Liu. *Generalized Out-of-Distribution Detection: A Survey.* IJCV, 2024. — arXiv:2110.11334

## 10. Worked Example

Take a Glow trained on CIFAR-10. Under it, CIFAR-10 test averages ≈3.4 bits/dim and SVHN ≈2.4 bits/dim. Threshold at the 5th percentile of ID log-likelihood: nearly every SVHN image passes as in-distribution. Detection rate ≈0. Flip the rule to "reject high likelihood" and SVHN is caught — but now CIFAR-10's own smooth, low-complexity images (sky, snow) are rejected, and CelebA, which sits *between* the two in bits/dim, is caught by neither rule.

Now make the obstruction explicit. Apply an invertible, smooth per-pixel map $g$ — say a fixed monotone tone curve applied channel-wise. Densities transform as
$$\log p_{g(X)}(g(x)) = \log p_X(x) - \log|\det J_g(x)|,$$
and $|\det J_g|$ depends on pixel values. Choose $g$ that compresses the bright, low-variance regions SVHN is full of; $-\log|\det J_g|$ then adds several bits/dim to SVHN and near-zero to CIFAR-10 textures. The two orderings swap. Nothing about the data or the model's fit changed — only the coordinates in which "density" was written. The *same* perfect density model gives opposite answers.

That is the block in one line: with no reference distribution, "OOD" is not a property of $x$ and $P_{\text{in}}$ alone; it is a property of $x$, $P_{\text{in}}$, and a choice — of alternative, or of base measure — that the current evaluation makes silently and then scores as if it were ground truth.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*