---
id: 33-uncertainty-calibration/calibration-under-shift-unlabeled
title: "Calibration Under Distribution Shift Without Target Labels"
topic: 33-uncertainty-calibration
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration Under Distribution Shift Without Target Labels

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/calibration-under-shift-unlabeled` · **Status:** open

## 1. Problem Statement

A classifier $f$ is trained and calibrated on a labeled source distribution $P_S$. It is deployed on a target distribution $P_T \neq P_S$. Only **unlabeled** target inputs are available. The question: can the model's confidences be recalibrated on $P_T$, and can the residual miscalibration be *certified*, without ever seeing a target label?

Three variants, of very different difficulty:

- **Measurement.** Given $f$, a source labeled sample, and an unlabeled target sample, output an estimate $\widehat{\mathrm{CE}}_T$ of target calibration error with a usable error bar. This is an estimation problem.
- **Method.** Output a recalibration map $g: [0,1] \to [0,1]$ (or a full post-hoc head) such that $\mathrm{CE}_T(g \circ f) < \mathrm{CE}_T(f)$, reliably, across shift families.
- **Theory.** State assumptions on $(P_S, P_T)$ under which target calibration is *identified* from $(\text{labeled } P_S, \text{unlabeled } P_T)$, and prove a finite-sample bound; or prove that under weaker assumptions no estimator beats the trivial one.

Solving it means: an estimator whose predicted target ECE tracks the true (held-out-label) target ECE within a stated tolerance on shift families it was never tuned on, plus a recalibration method that does not degrade any of them.

## 2. Formal Setting

Inputs $x \in \mathcal{X}$, labels $y \in \{1,\dots,K\}$. Model outputs a confidence vector; write top-label confidence $c(x) = \max_k f_k(x)$ and prediction $\hat{y}(x) = \arg\max_k f_k(x)$.

**Top-label calibration error** on distribution $P$, the $\ell_1$ form:
$$\mathrm{CE}_P = \mathbb{E}_{x \sim P}\big[\,\big|\,\Pr[\hat{y}(x)=y \mid c(x)]-c(x)\,\big|\,\big].$$

**As measured**: with $n$ target points binned into $M$ bins $B_1,\dots,B_M$,
$$\widehat{\mathrm{ECE}} = \sum_{m=1}^{M} \frac{|B_m|}{n}\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|,$$
which requires target labels for $\mathrm{acc}(B_m)$ — exactly what is absent. The unlabeled problem is to replace $\mathrm{acc}(B_m)$ by an estimate.

**Shift models.**
- Covariate shift: $p_T(y\mid x) = p_S(y \mid x)$, $p_T(x) \neq p_S(x)$. Density ratio $w(x) = p_T(x)/p_S(x)$; importance-weighted calibration replaces $\mathbb{E}_S$ by $\mathbb{E}_S[w(x)\,\cdot\,]$, estimated by a source-vs-target domain discriminator $d(x)$ with $\hat w(x) = d(x)/(1-d(x))$.
- Label shift: $p_T(x\mid y)=p_S(x\mid y)$, priors move. Estimable from unlabeled target via BBSE: $\hat{q}_T = \hat{C}^{-1}\hat{\mu}_T$, where $\hat C_{jk}=\hat\Pr_S[\hat y = j, y=k]$ is the source confusion matrix and $\hat\mu_T$ the target predicted-label histogram.

**Assumptions, and their status in practice.**

| Assumption | Needed for | Violated in practice? |
|---|---|---|
| $p_T(y\mid x)=p_S(y\mid x)$ | importance-weighted calibration | Yes — ImageNet-R, WILDS-Camelyon, and most real deployment shifts change the conditional |
| Overlap: $w(x) < B < \infty$ | finite-variance IW estimates | Yes — new-domain images have near-zero source density; $\hat w$ diverges |
| $p_S(x\mid y)=p_T(x\mid y)$ | BBSE / label-shift correction | Yes for corruption and style shift |
| Confusion matrix $\hat C$ invertible and stable | BBSE | Marginal at $K{=}1000$ with few source points per class |
| Binned ECE $\to$ CE as $M,n\to\infty$ | any binned report | Binned ECE is downward-biased at finite $n$; bias depends on $M$ |

## 3. State of the Art

**Established (reproduced, ablated).**
- Temperature scaling (Guo et al., ICML 2017) fixes in-distribution ECE with one parameter and *does not transfer* under shift — reconfirmed by Ovadia et al. (NeurIPS 2019) across MNIST/CIFAR-10/ImageNet plus text.
- Deep ensembles are the most robust calibration baseline under shift in Ovadia et al.; no post-hoc method beat them there.
- Average Thresholded Confidence (ATC; Garg et al., ICLR 2022) estimates *target accuracy* from unlabeled target data by thresholding a source-fit score, and beats prior unlabeled accuracy estimators by 2–4$\times$ in mean absolute error across a large shift suite.
- Weighted conformal prediction (Tibshirani et al., NeurIPS 2019) gives exact marginal coverage under known covariate shift — a certified *set-valued* guarantee, not a certified confidence.

**Claimed but under-ablated.**
- Importance-weighted recalibration under covariate shift (Park et al., AISTATS 2020) — correct under its assumptions; the assumption itself is not checkable from unlabeled data, and results are on shifts constructed to satisfy it.
- Test-time adaptation (TENT, Wang et al., ICLR 2021) changes both accuracy and confidence; entropy minimization *systematically sharpens* confidences, which can lower or raise ECE. Papers usually report accuracy only.
- Claims that ViT/MLP-Mixer families are better calibrated OOD (Minderer et al., NeurIPS 2021) hold for their model set, but architecture is confounded with pretraining data scale.

**Benchmark-number-only.** Most "OOD calibration" leaderboard entries report ECE on ImageNet-C at fixed $M=15$ bins. That is a single estimator at a single binning on one synthetic corruption family — not evidence of a method property.

## 4. What Is Known

- **Shift breaks calibration monotonically with severity.** Ovadia et al. (2019), ImageNet + ImageNet-C, ResNet-50 scale: ECE grows from roughly $0.02$–$0.05$ in-distribution to above $0.1$–$0.2$ at corruption severity 5, for every single-model method tested. Temperature scaling's advantage vanishes by severity 3.
- **Accuracy is predictable, calibration less so.** "Accuracy on the line" (Miller et al., ICML 2021) shows near-linear ID/OOD accuracy correlation across hundreds of models on CIFAR-10.1, ImageNet-V2. "Agreement-on-the-line" (Baek et al., NeurIPS 2022) turns pairwise model agreement — computable unlabeled — into an OOD accuracy estimate with strong correlation on the same suites.
- **ECE estimation is itself biased.** Roelofs et al. (AISTATS 2022) and Nixon et al. (CVPR-W 2019): equal-width binned ECE is downward biased; measured bias is on the order of the effect sizes methods claim (a few ECE points) at $n \sim 10^4$ ImageNet validation scale.
- **Debiased estimation exists in-distribution.** Kumar et al. (NeurIPS 2019) give a scaling-binning calibrator with sample complexity for $\ell_2$ calibration error and a debiased estimator; the guarantee is for the distribution the calibration set is drawn from.
- **Label shift is solvable when it is truly label shift.** BBSE (Lipton et al., ICML 2018) and its unified successors (Garg et al., NeurIPS 2020) recover target priors consistently from unlabeled data; Podkopaev & Ramdas (UAI 2021) extend to distribution-free calibration guarantees under label shift.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the largest assumption class under which target calibration error is *identified* from labeled source + unlabeled target. The negative direction is folklore, not a stated theorem: for a fixed target marginal $p_T(x)$, any $p_T(y\mid x)$ is consistent with the data, so $\mathrm{CE}_T$ ranges over nearly its whole feasible interval. What is missing is a sharp result — the exact width of the identified set as a function of a shift-magnitude constraint (e.g. $\mathrm{TV}(p_S(y|x),p_T(y|x)) \le \epsilon$), and a matching estimator.
- **Empirically open.** Whether the agreement-on-the-line phenomenon extends from *accuracy* to *calibration*: does ID calibration of a model pair predict OOD calibration across a model zoo? Runnable now on existing zoos; not run at scale.
- **Methodologically blocked.** There is no agreed target-side calibration metric that is estimable without labels and that has a decision-theoretic meaning. Reports differ in binning, $\ell_1$ vs $\ell_2$, top-label vs classwise vs canonical calibration — and rank methods differently. Until the estimand is fixed, "improves OOD calibration" is not a falsifiable claim.

## 6. Why It Is Hard

The obstruction is **non-identifiability plus absent ground truth**, in that order.

1. Unlabeled target data constrains only $p_T(x)$. Calibration is a property of $p_T(y \mid x)$. Nothing in the data pins it down. Every working method smuggles in an untestable bridge assumption (covariate shift, label shift, or an anchor-point condition), and no unlabeled diagnostic can test that assumption — testing it would require target labels.
2. Overlap fails exactly where the problem matters. Under real shift $\hat w(x)$ is estimated by a discriminator that reaches near-perfect separation, so $\hat w$ is effectively $\infty$ on target points and the importance-weighted estimator has unbounded variance with a finite-sample value dominated by a handful of points.
3. The evaluation does not measure what it names. Binned ECE with $M=15$ is a biased plug-in for a quantity defined by a conditional expectation; its bias moves with $n$, $M$ and confidence distribution — and shift changes the confidence distribution. So a reported OOD ECE *drop* can be an estimator artifact of a sharpened confidence histogram.

## 7. Current Research (as of 2026)

- **Unlabeled performance estimation** extending ATC/DoC/agreement-on-the-line from accuracy to full reliability curves — CMU (Garg, Balakrishnan, Lipton, Raghunathan), Berkeley/Washington (Miller, Schmidt, Recht lineage). *(frontier — verify current status)*
- **Distribution-free UQ beyond exchangeability**: conformal under non-exchangeable data (Barber, Candès, Ramdas, Tibshirani, *Annals of Statistics* 2023) — coverage guarantees degrade gracefully in a total-variation-like shift term. The active question is converting coverage guarantees into confidence-calibration guarantees.
- **Calibration of LLM verbalized/token confidences under domain shift**, where no source calibration set is even well defined. *(frontier — verify)*
- **Proper-scoring-rule decompositions** as bias-controlled replacements for binned ECE (Gruber & Buettner, NeurIPS 2022).

## 8. Concrete Next Experiment

**Question.** Does ID calibration predict OOD calibration the way ID accuracy predicts OOD accuracy?

- **Scale.** 200+ ImageNet classifiers spanning architecture, pretraining scale, and augmentation (an existing public zoo suffices — `timm` checkpoints). Evaluate on ImageNet-val (ID) and five OOD sets with labels held out for scoring only: ImageNet-V2, ImageNet-R, ImageNet-Sketch, ObjectNet, ImageNet-C severity 3. Compute cost: one forward pass per model per set, roughly $10^3$ GPU-hours.
- **Estimand, fixed in advance.** $\ell_2$ top-label calibration error via the debiased scaling-binning estimator of Kumar et al. (2019), $M=100$ equal-mass bins, bootstrap CI over 1000 resamples. Report binned $\ell_1$ ECE at $M=15$ alongside, purely to quantify the estimator-choice gap.
- **Control arm.** The same regression run on *accuracy* instead of calibration, which is known to yield $R^2 > 0.9$ on these sets. If the calibration regression fails while the accuracy control reproduces, the failure is about calibration, not about the zoo.
- **Deciding number.** $R^2$ of probit-transformed OOD calibration error on ID calibration error, per shift set. $R^2 \ge 0.7$ on at least four of five sets means unlabeled calibration prediction is a solvable regression problem and the field should build estimators on it. $R^2 \le 0.3$ means calibration transfer is model-specific and the search should move to per-model unlabeled diagnostics.

## 9. Key References

- **[Foundational]** Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Foundational]** Ovadia, Fertig, Ren, Nado, Sculley, Nowozin, Dillon, Lakshminarayanan, Snoek. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[Foundational]** Lipton, Wang, Smola. *Detecting and Correcting for Label Shift with Black Box Predictors.* ICML, 2018. — arXiv:1802.03916
- **[SOTA]** Garg, Balakrishnan, Lipton, Neyshabur, Sedghi. *Leveraging Unlabeled Data to Predict Out-of-Distribution Performance.* ICLR, 2022. — arXiv:2201.04234
- **[SOTA]** Baek, Jiang, Raghunathan, Kolter. *Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift.* NeurIPS, 2022. — arXiv:2206.13089
- **[SOTA]** Kumar, Liang, Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[SOTA]** Tibshirani, Barber, Candès, Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS, 2019. — arXiv:1904.06019
- **[Method]** Park, Bastani, Weimer, Lee. *Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation.* AISTATS, 2020.
- **[Method]** Podkopaev, Ramdas. *Distribution-Free Uncertainty Quantification for Classification Under Label Shift.* UAI, 2021.
- **[Method]** Wang, Shelhamer, Liu, Olshausen, Darrell. *Tent: Fully Test-Time Adaptation by Entropy Minimization.* ICLR, 2021. — arXiv:2006.10726
- **[Measurement]** Roelofs, Cain, Shlens, Mozer. *Mitigating Bias in Calibration Error Estimation.* AISTATS, 2022.
- **[Measurement]** Gruber, Buettner. *Better Uncertainty Calibration via Proper Scores for Classification and Beyond.* NeurIPS, 2022.
- **[Empirical]** Minderer, Djolonga, Romijnders, Hubis, Zhai, Houlsby, Tran, Lucic. *Revisiting the Calibration of Modern Neural Networks.* NeurIPS, 2021. — arXiv:2106.07998
- **[Empirical]** Miller, Taori, Raghunathan, Sagawa, Koh, Shankar, Liang, Carmon, Schmidt. *Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization.* ICML, 2021.
- **[Survey]** Silva Filho, Song, Perello-Nieto, Santos-Rodriguez, Kull, Flach. *Classifier Calibration: A Survey on How to Assess and Improve Predicted Class Probabilities.* Machine Learning, 2023.

## 10. Worked Example

ResNet-50, ImageNet-val, temperature-scaled on 25k held-out ID images. Representative numbers at this scale:

| Set | Accuracy | Mean conf. | ECE ($M{=}15$) |
|---|---|---|---|
| ImageNet-val (ID) | 0.76 | 0.77 | 0.02 |
| ImageNet-C, Gaussian noise, sev. 3 | 0.39 | 0.63 | 0.24 |

Now try to recover the $0.24$ without labels.

**Step 1 — importance weighting.** Train a logistic discriminator on penultimate features to separate 25k ID from 25k corrupted images. It reaches AUC $\approx 0.999$. So $d(x) \to 1$ on target and $\hat w(x)=d/(1-d)$ blows up: the effective sample size $\big(\sum_i \hat w_i\big)^2/\sum_i \hat w_i^2$ collapses from 25{,}000 to order 10. The IW estimate of target ECE is then a weighted average over $\sim$10 source points — its bootstrap CI spans nearly $[0, 0.3]$. Useless, and *correctly* so: the estimator is honest that it has no information.

**Step 2 — label shift.** BBSE on the same data returns a target prior estimate. But the true corruption changes $p(x\mid y)$, not $p(y)$; the confusion matrix $\hat C$ estimated on clean data no longer describes the corrupted predictor. BBSE reports a prior shift that does not exist and leaves ECE essentially unchanged at the source value $\approx 0.02$ — a $12\times$ underestimate, delivered with no warning signal.

**Step 3 — ATC.** ATC estimates target accuracy well here (typically within 2–5 points of $0.39$). Convert to a calibration estimate by comparing to mean target confidence $0.63$: $\widehat{\text{gap}} = 0.63 - 0.41 = 0.22$, close to $0.24$.

**The obstruction, visible.** Step 3 works, but it estimates only the *aggregate* confidence–accuracy gap, $\big|\mathbb{E}[c] - \Pr[\hat y = y]\big|$, which lower-bounds $\mathrm{CE}_T$ and equals it only if the error is a uniform offset. A model whose bin-wise errors cancel — overconfident at $c \approx 0.9$, underconfident at $c \approx 0.4$ — has aggregate gap $0$ and large true ECE. No unlabeled method distinguishes those two models, because the difference lives entirely in $p_T(y \mid x)$, which the unlabeled sample does not constrain. Step 1 fails loudly, step 2 fails silently, and step 3 succeeds only on the component of the problem that happens to be identified.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*