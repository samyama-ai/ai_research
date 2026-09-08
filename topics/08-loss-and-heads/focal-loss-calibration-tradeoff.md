---
id: 08-loss-and-heads/focal-loss-calibration-tradeoff
title: "Focal Loss and Calibration Trade-off"
topic: 08-loss-and-heads
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Focal Loss and Calibration Trade-off

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/focal-loss-calibration-tradeoff` · **Status:** partially-solved

## 1. Problem Statement

Focal loss down-weights well-classified examples by $(1-p_t)^\gamma$. It was introduced to fix foreground/background imbalance in dense object detection (Lin et al., ICCV 2017), and later re-purposed as a *calibration* method: networks trained with it report lower expected calibration error than cross-entropy (Mukhoti et al., NeurIPS 2020). But focal loss is **not a proper scoring rule** — its population minimizer under the softmax link is not the true class posterior (Charoenphakdee et al., CVPR 2021). So the method appears to improve the estimate of a quantity it provably biases.

Three variants of the problem, with different difficulty:

- **Measurement.** Is the reported ECE reduction a real reduction in distance-from-calibration, or an artifact of binned ECE being a biased, non-monotone functional that rewards the underconfidence focal loss induces? Blocked on estimator choice.
- **Method.** After a fair post-hoc recalibration control (temperature scaling fit on held-out data), does any $\gamma$ schedule beat cross-entropy on a proper score at matched accuracy? Empirically open at ImageNet scale and above.
- **Theory.** Focal loss is classification-calibrated but not strictly proper. Its minimizer is a known, invertible distortion of the posterior. Is the empirical calibration benefit *explained* by that distortion (i.e. a fixed, data-independent bias that happens to cancel the optimization-induced overconfidence), or by the implicit entropy regularization and gradient reweighting during training? Open.

Solving it means: a statement of the form "focal loss with $\gamma \in [a,b]$ reduces [named proper-score-based calibration functional] by $\delta$ relative to cross-entropy *after* temperature scaling, at matched top-1, for model class $\mathcal{M}$" — plus a mechanism that predicts $\delta$.

## 2. Formal Setting

Inputs $x \in \mathcal{X}$, labels $y \in \{1,\dots,K\}$, joint $\mathcal{D}$. Model $f_\theta$ emits logits $z \in \mathbb{R}^K$; $\hat p = \mathrm{softmax}(z)$, $\hat p_t = \hat p_y$, confidence $c(x) = \max_k \hat p_k$, prediction $\hat y = \arg\max_k \hat p_k$.

**Focal loss.**
$$\mathcal{L}_\gamma(\hat p, y) = -(1-\hat p_y)^\gamma \log \hat p_y, \qquad \gamma \ge 0,$$
with $\gamma = 0$ recovering cross-entropy. Sample-dependent variants set $\gamma = \gamma(\hat p_y)$ (FLSD-53: $\gamma=5$ for $\hat p_y \in [0,0.2)$, $\gamma=3$ otherwise).

**Calibration.** Perfect calibration: $\Pr[y = \hat y \mid c(x) = v] = v$ for all $v$. Measured as binned top-label ECE with $M$ bins $B_m$:
$$\widehat{\mathrm{ECE}}_M = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\big|\,\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\,\big|.$$
As actually measured this depends on: $M$ (usually 15), binning scheme (equal-width vs equal-mass), top-label vs classwise vs full multiclass, and whether the debiased estimator of Kumar et al. (NeurIPS 2019) is used. $\widehat{\mathrm{ECE}}_M$ is a **biased-downward, non-monotone** estimator of $\mathrm{ECE} = \mathbb{E}|\Pr[y=\hat y \mid c] - c|$; bias grows with $M$ and shrinks with $n$.

**Proper alternatives.** NLL $= -\frac1n\sum \log \hat p_{y_i}$ and Brier $= \frac1n \sum \|\hat p_i - e_{y_i}\|_2^2$ are strictly proper; both decompose (Murphy) into calibration + refinement, and the calibration term can be estimated with a proper-score-based estimator (Gruber & Büttner, NeurIPS 2022).

**Impropriety.** For fixed $x$ with true posterior $q$, the minimizer $\hat p^\star$ of $\mathbb{E}_{y \sim q}[\mathcal{L}_\gamma]$ satisfies $\hat p^\star \ne q$ for $\gamma > 0$; the map $q \mapsto \hat p^\star$ is strictly monotone in each coordinate and invertible, so the Bayes-optimal *decision* is preserved (classification-calibrated) while the *probability* is not (Charoenphakdee et al., CVPR 2021).

**Entropy bound.** Mukhoti et al. give
$$\mathcal{L}_\gamma \;\ge\; \mathrm{KL}(q \,\|\, \hat p) \;-\; \gamma\, \mathbb{H}[\hat p],$$
i.e. focal loss upper-bounds a cross-entropy objective with an entropy bonus of weight $\gamma$ — the stated mechanism for reduced overconfidence.

**Assumptions known to be violated in practice.** (i) The test distribution equals the training distribution — violated for every OOD calibration claim. (ii) The network reaches the population minimizer — never true; the observed benefit is an optimization-trajectory effect, not a minimizer property, and the two explanations are routinely conflated. (iii) $n$ per bin is large enough for $\widehat{\mathrm{ECE}}$ to be near-unbiased — violated at $n = 10{,}000$ (CIFAR test set) with 15 bins, where nearly all mass concentrates in the top bin.

## 3. State of the Art

**Empirical SOTA (established).** Mukhoti et al., *Calibrating Deep Neural Networks using Focal Loss*, NeurIPS 2020: focal loss with sample-dependent $\gamma$ (FLSD-53) lowers pre-temperature-scaling ECE by large margins on CIFAR-10/100, Tiny-ImageNet, and 20 Newsgroups, at comparable test error. Reported ResNet-50/CIFAR-100: ECE ≈ 17.5% (cross-entropy) → ≈ 4.5% (FLSD-53); ResNet-50/CIFAR-10: ≈ 4.35% → ≈ 1.55%. Independently reproduced in several follow-ups. The *pre*-scaling gap is established; the post-scaling gap is small and inconsistent.

**Theory SOTA (established).** Charoenphakdee, Vongkulbhisal, Chairatanakul, Sugiyama, *On Focal Loss for Class-Posterior Probability Estimation: A Theoretical Perspective*, CVPR 2021: focal loss is classification-calibrated but not strictly proper; they derive the link correction that recovers a consistent posterior estimate from a focal-trained score.

**Refinements (claimed, partially ablated).** AdaFocal (Ghosh et al., NeurIPS 2022) adapts $\gamma$ per bin from validation calibration statistics; Dual Focal Loss (Tao et al., ICML 2023) targets the gap between the true-class and highest competing logit. Both report ECE gains; neither has been ablated against a strong post-hoc recalibration control at ImageNet scale in independent work.

**Benchmark-number-only results.** Most focal-loss detection calibration claims (RetinaNet-family) exist only as table entries. Detection ECE is itself ill-posed — see Pathiraja et al., CVPR 2023.

**Countervailing.** Minderer et al., *Revisiting the Calibration of Modern Neural Networks*, NeurIPS 2021: modern non-convolutional architectures (ViT, MLP-Mixer, BiT) trained with plain cross-entropy are already well calibrated and calibration improves with in-distribution accuracy — undercutting the premise that a loss-side fix is needed at scale.

## 4. What Is Known

- Focal loss is classification-calibrated for $\gamma \ge 0$ but not strictly proper (CVPR 2021). Theorem, not measurement.
- $\mathcal{L}_\gamma \ge \mathrm{KL}(q\|\hat p) - \gamma \mathbb{H}[\hat p]$ (NeurIPS 2020). The regularizer is on the *predicted* entropy, so its effect is confounded with label smoothing and with weight decay's effect on logit norm.
- Pre-scaling ECE reductions of 3–13 percentage points on CIFAR-10/100 and Tiny-ImageNet, ResNet-50/ResNet-110/Wide-ResNet-26-10/DenseNet-121, $n_{\text{test}} = 10^4$, 15 equal-width bins.
- Temperature scaling on a held-out split closes most of the cross-entropy gap: Guo et al., ICML 2017, report post-scaling ECE typically under 2% for the same architectures. Focal loss frequently *needs* $T < 1$ (it is underconfident), which is the diagnostic signature of over-correction.
- Binned ECE underestimates true calibration error and the bias is estimator-dependent (Kumar, Liang, Ma, NeurIPS 2019; Nixon et al., CVPR-W 2019). Debiased and equal-mass variants change method rankings on the same predictions.
- Distance-from-calibration has a consistent theory: Błasiok, Gopalan, Hu, Nakkiran, STOC 2023 give a family of polynomially-related, robust calibration distances; binned ECE is not in that family for all binning choices.

## 5. What Is Not Known

- **Methodologically blocked.** Whether the focal-loss "calibration gain" survives a *fair* metric. Almost every claim is stated in binned top-label ECE, which rewards underconfidence asymmetrically at the metric's own bias scale. No large study reports focal vs cross-entropy under the STOC-2023 smooth calibration distance or a debiased proper-score decomposition.
- **Empirically open.** ImageNet-1k and larger, modern architectures (ViT-B/16, ConvNeXt), 3+ seeds, with temperature scaling applied to *both* arms and accuracy matched: is post-scaling calibration better, worse, or equal? Runnable today; the compute is ~10 GPU-days per arm; nobody has published the full matrix.
- **Theoretically open.** Whether the empirical benefit is the impropriety bias (a fixed distortion) or the training-dynamics effect (gradient reweighting slows logit-norm growth on easy examples). Applying the CVPR-2021 link correction should *remove* the bias explanation; if calibration stays good after correction, the mechanism is dynamical. This decisive test has not been reported.
- **Theoretically open.** Any excess-calibration-risk bound for $\mathcal{L}_\gamma$: no known rate relating focal excess risk to calibration distance, in contrast to standard proper-loss bounds.

## 6. Why It Is Hard

**Confounded measurement, plus an evaluation that does not measure what it names.** Binned ECE is a plug-in estimate of a discontinuous functional. Its bias is downward and depends on bin occupancy, so a method that shifts mass out of the saturated top bin — exactly what focal loss does — reduces the *estimator* even when the underlying calibration distance is unchanged. Second, ECE is invariant to which examples are wrong: a model can be perfectly calibrated and useless. Third, the comparison is not held fixed on the axis that matters: focal loss changes both the induced posterior (impropriety) and the optimization path (gradient reweighting), and no published ablation separates them, so the effect is **non-identifiable** from the reported data. Fourth, the natural control — temperature scaling — is a one-parameter fix that closes most of the cross-entropy gap for free, so the practically interesting effect size is small (sub-percentage-point) and is below the seed-to-seed noise of a single CIFAR run.

## 7. Current Research (as of 2026)

- Adaptive-$\gamma$ methods (AdaFocal lineage; Ghosh, Schaaf, Gormley) and margin-based variants (Dual Focal Loss).
- Calibration-metric foundations: Błasiok/Nakkiran-style distance-from-calibration, and proper-score decompositions (Gruber & Büttner) being adopted as replacements for binned ECE in evaluation code. *(frontier — verify: adoption rate in loss-function papers is still low.)*
- Calibration of LLM token heads and of RLHF reward models, where focal-style reweighting is being tried against selective-prediction objectives. *(frontier — verify.)*
- Detection- and segmentation-specific calibration where focal loss is the default classification loss (Pathiraja et al., CVPR 2023; Küppers et al., CVPR-W 2020) — here the trade-off is unavoidable, since removing focal loss changes the imbalance handling too.

## 8. Concrete Next Experiment

**Question.** Does focal loss improve calibration beyond what one temperature parameter buys?

**Scale.** ImageNet-1k, two architectures (ResNet-50, ViT-B/16), identical recipes, 3 seeds each. Arms: $\gamma \in \{0, 1, 2, 3, 5\}$, FLSD-53, plus label smoothing $\varepsilon = 0.05$ as a second regularization control. 21 runs per architecture; ~2 GPU-days per ResNet-50 run on 8×A100.

**Control arm.** $\gamma = 0$ (cross-entropy), **both arms temperature-scaled** on a held-out 25k split carved from train, evaluated on the 50k val set. Second control: focal arm with the Charoenphakdee link correction applied at inference, no temperature scaling.

**Deciding number.** Post-temperature-scaling **debiased equal-mass top-label ECE (15 bins, Kumar et al. estimator)**, reported with a bootstrap 95% CI, at accuracy matched to within 0.2 pp.

- If $\mathrm{ECE}_{\text{focal}} - \mathrm{ECE}_{\text{CE}} > -0.3$ pp (i.e. no meaningful improvement), focal loss's calibration benefit is subsumed by temperature scaling and the method claim collapses to a pre-scaling artifact.
- If the gap is $\le -0.3$ pp and *persists* after the link correction, the mechanism is dynamical, not the impropriety bias.

Report NLL and Brier alongside; a calibration gain paid for with worse NLL is a refinement loss, not a win.

## 9. Key References

- **[Foundational]** Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. *Focal Loss for Dense Object Detection.* ICCV, 2017. — arXiv:1708.02002
- **[Foundational]** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[SOTA]** Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip H.S. Torr, Puneet K. Dokania. *Calibrating Deep Neural Networks using Focal Loss.* NeurIPS, 2020. — arXiv:2002.09437
- **[Theory]** Nontawat Charoenphakdee, Jayakorn Vongkulbhisal, Nuttapong Chairatanakul, Masashi Sugiyama. *On Focal Loss for Class-Posterior Probability Estimation: A Theoretical Perspective.* CVPR, 2021. — arXiv:2011.09172
- **[SOTA]** Arindam Ghosh, Thomas Schaaf, Matthew R. Gormley. *AdaFocal: Calibration-aware Adaptive Focal Loss.* NeurIPS, 2022.
- **[SOTA]** Linwei Tao, Minjing Dong, Chang Xu. *Dual Focal Loss for Calibration.* ICML, 2023.
- **[Measurement]** Ananya Kumar, Percy Liang, Tengyu Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[Measurement]** Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC, 2023. — arXiv:2211.16886
- **[Measurement]** Sebastian G. Gruber, Florian Büttner. *Better Uncertainty Calibration via Proper Scores for Classification and Beyond.* NeurIPS, 2022.
- **[Counter-evidence]** Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, Mario Lucic. *Revisiting the Calibration of Modern Neural Networks.* NeurIPS, 2021. — arXiv:2106.07998
- **[Survey]** Jeremy Nixon, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel, Dustin Tran. *Measuring Calibration in Deep Learning.* CVPR Workshops, 2019. — arXiv:1904.01685

## 10. Worked Example

Take a two-class problem where the true posterior at some input is $q = (0.7, 0.3)$, and ask what a *perfectly optimized* focal model predicts. Minimize $\mathbb{E}_{y\sim q}[-(1-\hat p_y)^\gamma \log \hat p_y]$ over $\hat p = (u, 1-u)$.

For $\gamma = 0$: $u^\star = 0.7$ (cross-entropy is proper). For $\gamma = 3$, solving the stationarity condition numerically gives $u^\star \approx 0.64$ — the model is *underconfident by about 6 points* at the population optimum, by construction, not by accident.

Now the obstruction. Suppose the empirical network trained with cross-entropy is overconfident: it reports $0.85$ where the truth is $0.7$ (a 15-point error, typical of ResNets on CIFAR-100). The focal-trained network reports something near $0.64$–$0.70$ — a 0–6 point error. Binned ECE rewards this heavily: the CE model's mass sits in the $[0.8,0.9]$ bin with 70% accuracy, contributing $\approx 0.15$; the focal model contributes $\approx 0.03$. Reported: "focal loss cuts ECE by 5×."

But apply one temperature $T$ to the CE logits. A single $T \approx 1.4$ maps $0.85 \to \approx 0.70$ and closes the gap to near zero, *without* touching the ranking, so accuracy is unchanged. The focal model needs $T < 1$ to *undo* its own bias. The two arms end at the same place, and the headline number measured a pre-scaling difference that one scalar erases.

The deeper problem: the two errors have different signs but the metric takes an absolute value, so $|{-0.06}| < |{+0.15}|$ is read as "better probabilities" when the population-optimal focal predictor is *provably wrong* and the population-optimal CE predictor is *provably right*. Only the empirical CE model was wrong, for optimization reasons that a scalar fixes. This is what makes the mechanism non-identifiable from the published tables: the reported improvement is consistent with both "focal loss regularizes the training trajectory" and "one bias happened to cancel another."

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*