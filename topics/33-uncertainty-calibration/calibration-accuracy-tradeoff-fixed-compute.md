---
id: 33-uncertainty-calibration/calibration-accuracy-tradeoff-fixed-compute
title: "Calibration Versus Accuracy Tradeoff Under Fixed Compute"
topic: 33-uncertainty-calibration
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration Versus Accuracy Tradeoff Under Fixed Compute

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/calibration-accuracy-tradeoff-fixed-compute` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed training-plus-inference compute budget $C$ (FLOPs), you can spend it on a bigger model, more tokens, an ensemble, a calibration-aware loss, or a held-out recalibration stage. Each allocation lands somewhere in the plane of (task accuracy, calibration error). The question: **is there a genuine Pareto frontier — an allocation at which buying calibration costs accuracy — or is the observed tradeoff an artifact of poor allocation and biased estimators?**

Three variants, with very different difficulty:

- **Measurement.** Define a calibration error whose plug-in estimator is not dominated by binning bias at the sample sizes available. Currently the weak link.
- **Method.** Find the compute allocation that minimizes calibration error subject to an accuracy floor. Runnable today; not run as a controlled sweep.
- **Theory.** Prove or refute the existence of a nontrivial tradeoff: for a fixed hypothesis class and budget, does reducing calibration error below $\epsilon$ force excess risk $\ge f(\epsilon) > 0$?

Solving it means producing a measured iso-FLOP frontier with confidence intervals, plus a statement of which side of the frontier post-hoc recalibration moves you to.

## 2. Formal Setting

Inputs $x \in \mathcal{X}$, labels $y \in \mathcal{Y}$, $|\mathcal{Y}| = K$. A model $f_\theta$ outputs $p_\theta(\cdot\mid x) \in \Delta^{K-1}$. Confidence $\hat{p}(x) = \max_k p_\theta(k \mid x)$, prediction $\hat{y}(x) = \arg\max_k p_\theta(k\mid x)$.

**Accuracy**, measured as the empirical mean on a held-out set of $n$ points:
$$\mathrm{Acc} = \tfrac{1}{n}\sum_{i=1}^n \mathbb{1}[\hat{y}(x_i) = y_i].$$

**Top-label calibration error**:
$$\mathrm{CE}_q = \left(\mathbb{E}_x\left|\mathbb{P}(\hat{y}=y \mid \hat{p}(x)) - \hat{p}(x)\right|^q\right)^{1/q}.$$
Measured as binned ECE with $M$ equal-mass bins $B_m$:
$$\widehat{\mathrm{ECE}} = \sum_{m=1}^M \frac{|B_m|}{n}\left|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\right|.$$
This estimator is **biased downward**: binning can only shrink the deviation, and the bias does not vanish at fixed $M$ (Kumar, Liang & Ma 2019; Vaicenavicius et al. 2019). Report equal-mass, not equal-width, bins and a bootstrap CI, or the number is not comparable across runs.

**Proper-score decomposition.** With NLL $\mathcal{L} = -\frac1n\sum_i \log p_\theta(y_i\mid x_i)$, the population version decomposes (DeGroot & Fienberg 1983; Bröcker 2009) as
$$\mathbb{E}[\mathcal{L}] = \underbrace{\mathrm{CAL}}_{\ge 0} - \underbrace{\mathrm{REF}}_{\text{refinement}} + \underbrace{H(Y)}_{\text{irreducible}},$$
so NLL alone cannot separate the two axes — this is why the tradeoff must be measured in two dimensions.

**Compute.** Training FLOPs $C_{\text{train}} \approx 6ND$ for $N$ parameters and $D$ tokens; inference $\approx 2N$ per token per ensemble member $E$. Total budget $C = 6ND + 2N E T_{\text{eval}}$. The decision object is
$$\min_{(N,D,E,\lambda)\,:\, C(N,D,E) \le C} \ \mathrm{CE}_2 \quad \text{s.t.}\quad \mathrm{Acc} \ge \alpha,$$
where $\lambda$ indexes the loss (label smoothing strength, focal $\gamma$, temperature).

**Assumptions, and which fail.** (i) i.i.d. test data — violated under any distribution shift, and calibration degrades far faster than accuracy under shift (Ovadia et al. 2019). (ii) Top-label calibration suffices — violated when downstream decisions use the full distribution; multiclass/class-wise CE is a strictly harder target. (iii) Labels are ground truth — violated wherever annotator disagreement is real, in which case a "miscalibrated" model may be tracking aleatoric noise the single-label test set cannot express. (iv) $6ND$ is the right compute accounting — ignores attention cost at long context and data-loading overhead.

## 3. State of the Art

**Established (reproduced independently).**
- Temperature scaling: one scalar $T$ fit on a validation split, essentially accuracy-preserving because $\arg\max$ is invariant to $T$ (Guo et al., ICML 2017). This is the strongest fact in the area: *post-hoc scaling gives calibration at zero accuracy cost*, so any claimed tradeoff must be measured **after** temperature scaling or it is measuring nothing.
- Deep ensembles improve both accuracy and calibration, at $E\times$ inference cost (Lakshminarayanan et al., NeurIPS 2017; confirmed by Ovadia et al., NeurIPS 2019).
- Modern non-convolutional image models (ViT, MLP-Mixer) are better calibrated *and* more accurate than the ResNets of Guo et al. (Minderer et al., NeurIPS 2021) — evidence against a universal tradeoff.

**Claimed but unablated at fixed compute.**
- Focal loss "improves calibration" (Mukhoti et al., NeurIPS 2020) and label smoothing "improves calibration" (Müller et al., NeurIPS 2019) — both are reported at fixed *epochs*, not fixed FLOPs, and largely without a temperature-scaled control arm.
- Bayesian/variational alternatives to ensembles: benchmark numbers exist; the iso-FLOP comparison against "just train longer, then temperature-scale" is largely missing.
- RLHF degrades calibration: reported for GPT-4 as a figure, not an ablation with matched compute (OpenAI, *GPT-4 Technical Report*, 2023).

**Theory SOTA.** Błasiok, Gopalan, Hu & Nakkiran give a consistent *distance to calibration* (STOC 2023) and show when optimizing a proper loss implies calibration (NeurIPS 2023). No theorem states a compute-constrained tradeoff.

## 4. What Is Known

- **Guo et al. 2017**, CIFAR-100/ResNet-110 (~1.7M params): ECE $\approx 16.5\%$ pre-scaling, $\approx 1.3\%$ after temperature scaling, accuracy unchanged to the last digit.
- **Minderer et al. 2021**, ImageNet: ViT-L/16 and MLP-Mixer reach ECE $\approx 0.02$–$0.03$ at higher top-1 than ResNets with ECE $\approx 0.05$+. Within-family, larger and more accurate does not imply worse calibrated.
- **Ovadia et al. 2019**, ImageNet-C: under corruption severity 5, ECE rises roughly $3$–$5\times$ for single models; ensembles of 5 remain the best-calibrated method at every shift level.
- **OpenAI 2023**, MMLU: pre-trained GPT-4 ECE $\approx 0.007$; after RLHF post-training, $\approx 0.074$ — about a $10\times$ degradation with accuracy *up*. This is the single most-cited data point suggesting a real tradeoff, and it is an alignment-procedure effect, not a compute-allocation effect.
- **Kumar, Liang & Ma 2019**: binned ECE underestimates true CE; scaling-binning gives a calibrated estimate with $O(1/\sqrt{n})$ guarantees. Sample complexity for a trustworthy per-bin estimate is $\Omega(M/\epsilon^2)$ — with $M=15$ and $\epsilon=0.01$, tens of thousands of held-out points.
- **Kadavath et al. 2021** (Anthropic): self-evaluation ("P(True)") calibration improves monotonically with model size on multiple-choice tasks, up to 52B.

## 5. What Is Not Known

- **Empirically open.** No published iso-FLOP sweep in which $(N, D, E)$ are varied under a fixed budget and both accuracy and post-temperature-scaling $\mathrm{CE}_2$ are reported with CIs. The experiment is runnable on a $10^{20}$–$10^{21}$ FLOP budget; nobody has published it. This is the core gap.
- **Empirically open.** Whether RLHF's calibration cost is recoverable by post-hoc scaling on the RLHF'd model, at what fraction of a percent of accuracy.
- **Theoretically open.** No lower bound of the form "excess risk $\ge f(\epsilon)$ when $\mathrm{CE}_2 \le \epsilon$" for a fixed class and budget. Existing theory says calibration is achievable, not that it is expensive.
- **Methodologically blocked.** Calibration error for open-ended generation. There is no agreed measurable analogue of $\hat p$ when the output is a sequence; sequence log-probability is length-confounded and verbalized confidence is not a probability. Until this is defined, the tradeoff cannot be measured at all on the tasks frontier models are used for.

## 6. Why It Is Hard

**Confounded measurement plus a biased estimator, in the same number.** $\widehat{\mathrm{ECE}}$ has bias that scales with $M$ and shrinks with $n$, so a "calibration improvement" of 0.01 is inside the estimator's own error bar at $n = 10{,}000$. Two papers using equal-width vs. equal-mass bins are not comparing the same quantity.

**Non-identifiability against label noise.** With single-label test sets you cannot distinguish a model that is overconfident from a model correctly reporting $0.7$ on an item where 30% of humans would answer differently. Absent per-item human distributions, the "true" target of calibration is unobserved.

**Compute cost of the control arm.** The honest comparison needs a matched-FLOP baseline for every treatment. A 4-point grid over $(N, D)$ crossed with $E \in \{1,2,4\}$ and three losses is ~36 training runs at the target scale — the reason the sweep does not exist.

## 7. Current Research (as of 2026)

- **Post-hoc calibration for LLMs**: temperature and Platt-style scaling fitted per-task on multiple-choice benchmarks; conformal prediction as a distribution-free alternative that sidesteps ECE entirely (Angelopoulos & Bates tutorial line of work).
- **Calibration theory**: Błasiok/Gopalan/Nakkiran and collaborators on consistent calibration measures and calibration from proper losses. Active, and the most likely source of the missing lower bound.
- **Post-training effects on uncertainty**: whether RLHF/DPO mode-collapse is the mechanism of confidence inflation *(frontier — verify)*.
- **Efficient ensembling** (BatchEnsemble, snapshot ensembles, LoRA-ensembles) explicitly framed as buying calibration per FLOP *(frontier — verify for the newest LoRA variants)*.

## 8. Concrete Next Experiment

**Scale.** Fix $C = 3\times10^{20}$ training FLOPs (~1 A100-month-scale, reproducible in academia). Train decoder-only LMs on a Chinchilla-style grid: $N \in \{150\text{M}, 400\text{M}, 1.1\text{B}\}$ with $D$ set so $6ND = C$ in each cell; cross with $E \in \{1, 2, 4\}$ (splitting the same $C$ across members) and loss $\lambda \in \{\text{CE}, \text{label smoothing } 0.1, \text{focal } \gamma{=}3\}$. 27 runs.

**Control arm.** Every cell evaluated twice: raw, and after a single temperature fitted on 5,000 held-out items. The control question is whether any treatment beats *the cheapest model plus one scalar*.

**Evaluation.** Multiple-choice suites (MMLU, ARC, HellaSwag), $n \ge 30{,}000$ items pooled, $\mathrm{CE}_2$ estimated with scaling-binning (Kumar et al. 2019), 1,000-bootstrap CIs.

**The deciding number.** $\Delta = \mathrm{CE}_2^{\text{best treatment, temp-scaled}} - \mathrm{CE}_2^{\text{best accuracy cell, temp-scaled}}$, reported alongside the accuracy gap. **If the 95% CI on $\Delta$ contains 0 while the accuracy-optimal cell is also within noise on calibration, the tradeoff is an artifact and the correct advice is "maximize accuracy, then scale."** If $\Delta < -0.01$ with the accuracy-optimal cell losing $\ge 1$ point of accuracy, the frontier is real and its slope is measured.

## 9. Key References

- **[Foundational]** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Foundational]** Morris H. DeGroot, Stephen E. Fienberg. *The Comparison and Evaluation of Forecasters.* The Statistician, 1983.
- **[Foundational]** Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS, 2017. — arXiv:1612.01474
- **[SOTA]** Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, Mario Lucic. *Revisiting the Calibration of Modern Neural Networks.* NeurIPS, 2021. — arXiv:2106.07998
- **[SOTA]** Ananya Kumar, Percy Liang, Tengyu Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[SOTA]** Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC, 2023. — arXiv:2211.16886
- **[Empirical]** Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[Empirical]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Empirical]** Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, Thomas B. Schön. *Evaluating Model Calibration in Classification.* AISTATS, 2019. — arXiv:1902.06977
- **[Scaling]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Survey]** Jeremy Nixon, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel, Dustin Tran. *Measuring Calibration in Deep Learning.* CVPR Workshops, 2019. — arXiv:1904.01685

## 10. Worked Example

Two candidate allocations of the same $3\times10^{20}$ FLOPs, evaluated on 30,000 multiple-choice items ($K=4$):

| Arm | $N$ | $D$ | $E$ | Acc | raw $\widehat{\mathrm{ECE}}$ | temp-scaled |
|---|---|---|---|---|---|---|
| A: one big model | 1.1B | 45B | 1 | 0.612 | 0.081 | 0.014 |
| B: four small models | 275M | 45B | 4 | 0.598 | 0.026 | 0.011 |

Read raw, B looks like a 3× calibration win for 1.4 accuracy points — the classic "tradeoff." After the control arm (one scalar temperature, fit on 5,000 held-out items, zero accuracy change by $\arg\max$ invariance) the gap is $0.014$ vs $0.011$, i.e. $0.003$.

Now the obstruction. With $M=15$ equal-mass bins and $n=30{,}000$, each bin holds 2,000 points; the binomial standard error on a bin's accuracy is $\sqrt{0.6\cdot0.4/2000} \approx 0.011$. Aggregating, the bootstrap CI on $\widehat{\mathrm{ECE}}$ is roughly $\pm 0.004$. **The measured difference $0.003$ is smaller than the error bar on either endpoint**, and both plug-in estimates are biased downward by an unknown amount that scales with $M$.

So the tradeoff that looked like a factor of 3 collapses into estimator noise once you (i) apply the free control and (ii) attach a CI. To resolve $0.003$ at 95% confidence you need roughly $n \gtrsim 10^5$ held-out items per arm — more than most benchmark suites contain. That, not the training compute, is why the frontier has not been measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*