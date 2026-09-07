---
id: 08-loss-and-heads/calibration-under-shift-output-heads
title: "Calibration Under Distribution Shift for Deep Heads"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration Under Distribution Shift for Deep Heads

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/calibration-under-shift-output-heads` · **Status:** open

## 1. Problem Statement

A deep network's output head maps features to a probability vector. **Calibration** asks that the number mean what it says: among inputs where the head says $0.8$, the label should be correct $80\%$ of the time. Post-hoc recalibration on held-out in-distribution data solves this well. The open problem is what happens when the test distribution moves.

Given a head trained and recalibrated on source distribution $P$, and a test distribution $Q \neq P$ from which we observe **unlabeled** inputs only, produce confidences that remain calibrated under $Q$ — or produce an honest certificate that they are not.

Three variants, of very different difficulty:

- **Measurement.** Estimate calibration error under $Q$ from finite samples without a labeled $Q$ set, and without the estimator's own bias dominating the effect being measured.
- **Method.** Build a head or recalibration map whose calibration degrades gracefully in the shift magnitude, at no meaningful accuracy cost.
- **Theory.** Characterize the class of shifts $\mathcal{Q}$ for which distribution-free calibration transfer is possible from unlabeled $Q$ data, and prove impossibility outside it.

Solving it means: a stated shift family, a procedure, and a bound on calibration error under $Q$ that holds without $Q$-labels.

## 2. Formal Setting

Input $x \in \mathcal{X}$, label $y \in \{1,\dots,K\}$. Head $f_\theta: \mathcal{X} \to \Delta^{K-1}$, top-label confidence $c(x) = \max_k f_\theta(x)_k$, prediction $\hat{y}(x) = \arg\max_k f_\theta(x)_k$.

**Top-label calibration error** under distribution $R$:

$$\mathrm{CE}_p(R) = \left( \mathbb{E}_{x\sim R} \left| \mathbb{P}_R\!\left[\hat y(x)=y \mid c(x)\right] - c(x) \right|^p \right)^{1/p}$$

**As measured:** the conditional probability is not observable, so practice replaces it with a binned plug-in. With $B$ equal-width bins $\{I_b\}$ and $n$ labeled test points,

$$\widehat{\mathrm{ECE}} = \sum_{b=1}^{B} \frac{|I_b|}{n}\left| \mathrm{acc}(I_b) - \mathrm{conf}(I_b) \right|, \qquad B=15 \text{ by convention (Guo et al., 2017).}$$

This estimator is **biased downward** — binning averages away within-bin miscalibration — and the bias scales with $1/B$ while variance scales with $B/n$ (Kumar, Liang & Ma, 2019; Nixon et al., 2019). No unbiased plug-in exists.

**Shift model.** Write $Q$'s deviation from $P$ as a density ratio $w(x) = dQ/dP(x)$. Three standard restrictions:

- *Covariate shift:* $Q(y\mid x) = P(y\mid x)$, $Q(x)\neq P(x)$.
- *Label shift:* $Q(x\mid y) = P(x\mid y)$, $Q(y)\neq P(y)$.
- *Bounded non-exchangeability:* $d_{\mathrm{TV}}(P,Q)\le \epsilon$ with no structural assumption.

**Assumptions and their status in practice:**

| Assumption | Status |
|---|---|
| $\mathrm{supp}(Q)\subseteq\mathrm{supp}(P)$, $w$ bounded | Violated. ImageNet-C severity 5, new domains, and post-cutoff text sit outside source support; $w$ is effectively infinite. |
| Covariate shift ($P(y\mid x)$ fixed) | Violated whenever labeling policy or annotator pool changes; untestable without $Q$-labels. |
| Head is the only shifted component | Violated — features shift too; recalibrating the head cannot repair a representation collapse. |
| Calibration set is exchangeable with test | Violated by construction; this is the problem. |

## 3. State of the Art

**Established (reproduced, ablated).**
- Temperature scaling — one scalar $T$ fit by NLL on a source validation split — reduces in-distribution ECE of large image classifiers from double digits to about $1\%$ and does not change accuracy (Guo et al., ICML 2017). Reproduced hundreds of times.
- That same $T$ **does not transfer**: ECE grows monotonically with corruption severity for every method tested, temperature scaling included (Ovadia et al., NeurIPS 2019). Deep ensembles degrade most slowly. This is the single most-replicated result in the area.
- Conformal prediction gives finite-sample coverage under *known* covariate shift by weighting calibration scores by $w$ (Tibshirani, Barber, Candès & Ramdas, NeurIPS 2019), and coverage degrades by at most a computable function of the non-exchangeability when $w$ is unknown (Barber, Candès, Ramdas & Tibshirani, *Annals of Statistics*, 2023). This is coverage of sets, not calibration of scalars — a weaker, and different, guarantee.

**Claimed but incompletely ablated.**
- Focal loss yields better-calibrated heads than cross-entropy, including under shift (Mukhoti et al., NeurIPS 2020). The in-distribution effect replicates; the shift claim is confounded with focal loss's implicit temperature change and is rarely ablated against temperature scaling at matched accuracy.
- Modern architectures (ViT, MLP-Mixer, BiT) are better calibrated in-distribution *and* under ImageNet-C than ResNets, breaking the "bigger is worse" narrative (Minderer et al., NeurIPS 2021). Established as a benchmark observation; the causal driver (architecture vs. pretraining data scale) is not isolated.
- Multi-domain temperature scaling — fit temperatures across several source domains and extrapolate (Yu, Bates, Ma & Jordan, NeurIPS 2022) — improves shift-time ECE where multiple labeled source domains exist. Reported at moderate scale; a single-source method it is not.

**Benchmark-number-only.** Most reported gains for label-smoothing, mixup, and self-supervised auxiliary losses under shift exist as one ECE column on CIFAR-10-C/ImageNet-C, at one bin count, one seed regime, without an accuracy-matched control.

## 4. What Is Known

- **Magnitude.** On ImageNet-C at severity 5, top-1 accuracy of a standard ResNet-50 falls to roughly $20\%$ while confidence remains high; ECE rises to the tens of percent. Every method in Ovadia et al. (2019) degrades; ensembles of 5–10 members degrade least. Scale: ImageNet, ResNet/CNN family, 2019.
- **Direction of error is one-sided.** Under corruption shift, heads are overconfident, not underconfident, in essentially all reported settings. A single temperature refit on labeled target data usually removes most of the gap — the failure is *estimating* the temperature without labels, not the model class.
- **Distribution-free calibration is impossible without discretization.** Gupta & Ramdas (NeurIPS 2020) show that a distribution-free calibrated predictor with continuous output cannot exist under finite samples; only binned/discretized predictors admit guarantees. Any shift-time claim must inherit this.
- **Binning bias is comparable to the effect.** The plug-in $\widehat{\mathrm{ECE}}$ with $B=15$ can under-report calibration error by an amount of the same order as the differences between competing methods (Nixon et al., CVPR-W 2019; Vaicenavicius et al., AISTATS 2019). Debiased and spline estimators exist (Kumar et al., 2019; Gupta et al., ICLR 2021) but are not the field default.
- **Unsupervised domain adaptation has an information-theoretic floor.** Ben-David et al. (2010) bound target risk by source risk plus an $\mathcal{H}\Delta\mathcal{H}$-divergence term; the same non-identifiability blocks label-free calibration transfer.
- **LLM heads.** Pretrained base models are near-calibrated on multiple-choice tasks; RLHF-tuned models are markedly overconfident (Kadavath et al., 2022; OpenAI GPT-4 technical report, 2023, which shows post-RLHF calibration curves visibly worse than pre-RLHF on MMLU).

## 5. What Is Not Known

- **Theoretically open.** No characterization of the maximal shift family $\mathcal{Q}$ for which nontrivial calibration bounds under $Q$ are achievable from source labels plus unlabeled $Q$ samples. Both a positive result (a family broader than bounded-$w$ covariate shift) and a matching impossibility theorem are absent.
- **Empirically open.** Whether the shift-robustness ranking of heads and losses (focal, ensembles, ViT, mixup) survives accuracy-matching, debiased estimators, and scale. The experiment is runnable today; nobody has run the full factorial at ImageNet/LLM scale with a fixed accuracy control.
- **Empirically open.** Whether a single global temperature is the right functional form under shift at all, versus an input-conditional map $T(x)$ — and whether $T(x)$ can be fit from unlabeled target data alone without collapsing to the identity.
- **Methodologically blocked.** Calibration for generative and open-ended outputs. "Correctness" is not a $0/1$ event, so $\mathrm{CE}_p$ is undefined until a correctness judge is fixed — and the judge's own error under shift is unquantified. Reported LLM calibration numbers are judge-relative, not model properties.
- **Methodologically blocked.** No agreed scalar for *shift magnitude* against which calibration degradation can be plotted. ImageNet-C severity is an ordinal artifact, not a distance.

## 6. Why It Is Hard

Two specific obstructions.

**Non-identifiability.** Without $Q$-labels, the label-shift and concept-shift explanations of the same unlabeled $Q$ are observationally equivalent. Two worlds — the model is right and the class prior moved; the model is wrong and the inputs moved — produce identical unlabeled data and demand opposite temperature corrections. Label shift is estimable via BBSE-style confusion-matrix inversion (Lipton, Wang & Smola, ICML 2018) only under the label-shift assumption itself, which is untestable from the same data.

**The metric does not measure what it names.** $\widehat{\mathrm{ECE}}$ is a biased estimator of a quantity that is itself not proper: a head can achieve $\mathrm{ECE}=0$ by outputting the marginal base rate for every input. So a "shift-robust calibration" method can win the headline metric by discarding the sharpness that made the head useful. Papers that report ECE under shift without jointly reporting accuracy and a proper score (Brier, NLL) are measuring the wrong object.

Compute is not the binding constraint here — a full ImageNet-C sweep is a few thousand GPU-hours. The blockers are epistemic.

## 7. Current Research (as of 2026)

- **Beyond-exchangeability conformal.** Weighted and adaptive conformal methods (Gibbs & Candès; Barber et al.) are being pushed from set-coverage toward scalar calibration; Stanford/Berkeley/Chicago groups. Established for coverage, open for calibration.
- **Unlabeled target recalibration.** Fitting $T$ on target data using agreement-based or entropy-based surrogates for accuracy — including "agreement-on-the-line" style predictors of target accuracy from source/target agreement. Reported gains hold on ImageNet-C-like shifts and fail on natural shifts *(frontier — verify)*.
- **Verbalized and sampled confidence in LLMs.** Self-consistency spread, token-logit aggregation, and trained probes as calibration signals for generation. Anthropic, OpenAI, DeepMind and academic groups; blocked on the correctness-judge problem above *(frontier — verify)*.
- **Proper-score decompositions.** Recasting evaluation as calibration–sharpness decompositions of Brier/NLL rather than ECE alone, to close the degenerate-solution loophole.

## 8. Concrete Next Experiment

**Question.** Under shift, does any of the popular interventions beat *source temperature scaling* once accuracy is held fixed and the estimator is debiased?

**Scale.** ImageNet-1k, four architectures (ResNet-50, ConvNeXt-T, ViT-B/16, DeiT-S), three seeds each. Test on ImageNet-C (15 corruptions × 5 severities), ImageNet-R, ImageNet-Sketch, ObjectNet. About 36 training runs plus inference — order $10^3$ A100-hours.

**Arms.** (1) Cross-entropy + source temperature scaling — **the control**. (2) Focal loss ($\gamma=3$) + source TS. (3) 5-member deep ensemble + source TS. (4) Unlabeled-target temperature fit by entropy matching. (5) Oracle: temperature refit on labeled target — the achievable ceiling.

**Controls.** Match top-1 accuracy across arms to within $0.5$ points by early-stopping/checkpoint selection before any calibration comparison. Report the debiased-binned and spline estimators, not $B=15$ plug-in alone, plus Brier and its calibration–sharpness decomposition.

**Deciding number.** The **oracle gap closure**: $\displaystyle g = 1 - \frac{\mathrm{ECE}_{\text{arm}}(Q) - \mathrm{ECE}_{\text{oracle}}(Q)}{\mathrm{ECE}_{\text{control}}(Q) - \mathrm{ECE}_{\text{oracle}}(Q)}$, averaged over the shift suite. An arm is a real advance iff $g > 0.3$ with a bootstrap 95% CI excluding $0$, at matched accuracy. Current expectation, from Ovadia et al., is that only the ensemble arm clears it — and that the arms which look best on raw ECE lose sharpness.

## 9. Key References

- **[Foundational]** Guo, Pleiss, Sun & Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Foundational]** Ovadia, Fertig, Ren, Nado, Sculley, Nowozin, Dillon, Lakshminarayanan & Snoek. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[Foundational]** Ben-David, Blitzer, Crammer, Kulesza, Pereira & Vaughan. *A Theory of Learning from Different Domains.* Machine Learning, 79(1–2), 2010.
- **[SOTA/Theory]** Barber, Candès, Ramdas & Tibshirani. *Conformal Prediction Beyond Exchangeability.* Annals of Statistics, 2023.
- **[SOTA/Theory]** Tibshirani, Foygel Barber, Candès & Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS, 2019.
- **[Theory]** Gupta & Ramdas. *Distribution-free Binary Classification: Prediction Sets, Confidence Intervals and Calibration.* NeurIPS, 2020.
- **[Theory]** Kumar, Liang & Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019.
- **[Measurement]** Nixon, Dusenberry, Zhang, Jerfel & Tran. *Measuring Calibration in Deep Learning.* CVPR Workshops, 2019.
- **[Measurement]** Vaicenavicius, Widmann, Andersson, Lindsten, Roll & Schön. *Evaluating Model Calibration in Classification.* AISTATS, 2019.
- **[SOTA/Empirical]** Minderer, Djolonga, Romijnders, Hubis, Zhai, Houlsby, Tran & Lucic. *Revisiting the Calibration of Modern Neural Networks.* NeurIPS, 2021.
- **[Method]** Mukhoti, Kulharia, Sanyal, Golodetz, Torr & Dokania. *Calibrating Deep Neural Networks using Focal Loss.* NeurIPS, 2020.
- **[Method]** Yu, Bates, Ma & Jordan. *Robust Calibration with Multi-domain Temperature Scaling.* NeurIPS, 2022.
- **[Method]** Lipton, Wang & Smola. *Detecting and Correcting for Label Shift with Black Box Predictors.* ICML, 2018.
- **[LLM]** Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Survey]** Gawlikowski et al. *A Survey of Uncertainty in Deep Neural Networks.* Artificial Intelligence Review, 2023.

## 10. Worked Example

A CIFAR-10 ResNet-18 head, source validation split, $n = 5{,}000$.

- **In-distribution.** Accuracy $94.8\%$, mean confidence $97.9\%$, $\widehat{\mathrm{ECE}}_{15} = 3.1\%$. Fit $T^\star = 1.6$ on this split: $\widehat{\mathrm{ECE}}_{15} \to 0.8\%$, accuracy unchanged. Textbook success.
- **Shift.** CIFAR-10-C, Gaussian noise, severity 3, $n = 10{,}000$. Accuracy drops to $71\%$. With $T^\star=1.6$ applied, mean confidence is $89\%$ — so $\widehat{\mathrm{ECE}}_{15} \approx 18\%$, all overconfidence.
- **The oracle.** Refitting $T$ on *labeled* corrupted data gives $T_Q \approx 3.4$ and $\widehat{\mathrm{ECE}}_{15} \approx 2\%$. The functional form is fine. The parameter is $2.1\times$ off, and nothing in the unlabeled corrupted images tells you so.

**Where the obstruction becomes visible.** Try to recover $T_Q$ from unlabeled target data by matching mean confidence to an accuracy estimate. Any such method needs an accuracy estimate, which is the thing you lack. Now construct a second distribution $Q'$: clean CIFAR-10 images, but the class prior reweighted so that the head's *predicted* confidence histogram matches the corrupted one almost exactly. Under $Q'$ accuracy is still $\approx 94\%$ and the correct temperature is still $1.6$. The unlabeled input marginals differ, but every statistic the recalibrator actually consumes — the confidence histogram, the predicted-label distribution, the entropy — agrees to within noise. The two worlds demand $T=3.4$ and $T=1.6$.

That is the non-identifiability, in one pair of datasets: no label-free procedure that reads only the head's output distribution can distinguish them, so no such procedure can be correct on both. Progress requires either a label budget on $Q$, a testable structural restriction on the shift family, or a statistic that reads the features rather than the head.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*