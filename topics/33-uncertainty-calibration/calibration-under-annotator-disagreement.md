---
id: 33-uncertainty-calibration/calibration-under-annotator-disagreement
title: "Calibrated Uncertainty Under Label Noise and Disagreement"
topic: 33-uncertainty-calibration
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibrated Uncertainty Under Label Noise and Disagreement

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/calibration-under-annotator-disagreement` · **Status:** open

## 1. Problem Statement

Standard calibration assumes each input $x$ has one correct label $y$. Real annotation does not deliver that. On natural language inference, image classification of ambiguous images, toxicity, and medical grading, competent annotators disagree systematically and reproducibly. Two distinct sources are mixed together in the observed labels:

- **Aleatoric disagreement** — the item genuinely admits more than one reading, so the human population's response distribution $h(x)$ is non-degenerate.
- **Label noise** — annotator error, inattention, task misunderstanding, guideline drift, which corrupts $h(x)$ toward something not worth predicting.

Three variants, different difficulty:

- **Measurement.** Given a model $\hat p(\cdot\mid x)$ and $m$ human labels per item, define and estimate an error that is zero exactly when the model's predictive distribution matches the *quantity we want it to match*. Currently unresolved which quantity that is.
- **Method.** Train or post-hoc recalibrate so the model matches human response distributions on ambiguous items without inflating uncertainty on unambiguous ones.
- **Theory.** Identify conditions under which the aleatoric component is separable from annotator noise given $m$ labels per item, and give the sample complexity in $(m, n)$.

Solving it means: a calibration error estimator with known finite-$m$ bias, a target distribution justified by the downstream decision, and a method that improves that error without degrading accuracy on the majority label.

## 2. Formal Setting

Inputs $x\in\mathcal X$, label set $\mathcal Y$, $|\mathcal Y|=K$. An annotator population $\mathcal A$ with sampling distribution $\pi$. The **human opinion distribution** is

$$h(x)_y \;=\; \Pr_{a\sim\pi}\big[\,Y_a(x)=y\,\big],$$

*measured* by drawing $m$ annotators i.i.d. and forming the plug-in $\hat h(x)_y = c_y/m$ with counts $c_y$. The **majority label** is $y^\ast(x)=\arg\max_y h(x)_y$; the *observed* majority uses $\hat h$ and is itself a random variable.

Model output $\hat p(\cdot\mid x)\in\Delta^{K-1}$, confidence $\hat c(x)=\max_y \hat p(y\mid x)$.

Three inequivalent calibration targets:

1. **Majority-label (hard) calibration.** $\mathbb E\!\left[\mathbf 1\{\hat y(x)=y^\ast(x)\}\mid \hat c(x)=v\right]=v$, estimated by binned ECE, $\widehat{\mathrm{ECE}}=\sum_{b}\frac{n_b}{n}\lvert \mathrm{acc}_b-\mathrm{conf}_b\rvert$.
2. **Single-draw calibration.** Same predicate with $y\sim h(x)$, one random annotator's label. This is what a dataset with $m=1$ actually measures.
3. **Distribution calibration** (human calibration error, Baan et al. 2022). A divergence to the opinion distribution, e.g. $\mathbb E_x\,\mathrm{JSD}(\hat p(\cdot\mid x)\,\|\,h(x))$ or $\mathbb E_x\!\left[\lvert \hat c(x)-\max_y h(x)_y\rvert\right]$.

Targets 1 and 3 disagree by construction: a model matching $h$ exactly is *under*confident under target 1 whenever $\max_y h(x)_y > $ its own max.

Noise model. Write $\tilde h(x) = (1-\varepsilon)\,h^\star(x) + \varepsilon\, q(x)$ where $h^\star$ is the "signal" opinion distribution, $q$ the noise distribution, $\varepsilon$ the noise rate. Only $\tilde h$ is observed.

Assumptions, with those known violated in practice marked:

- **A1.** Annotators i.i.d. from $\pi$. *Violated:* crowd pools are non-random, small, and correlated within worker; per-item annotator sets differ.
- **A2.** Annotator responses conditionally independent given $x$. *Violated:* shared guidelines and priming induce correlation; Dawid–Skene-style models assume it explicitly.
- **A3.** $h^\star$ and $q$ separable. *Violated / not identified* without extra structure — a systematically biased subgroup is indistinguishable from genuine ambiguity.
- **A4.** $\hat h$ is a good estimate of $h$. *Violated:* most benchmarks use $m\in\{1,3,5\}$; plug-in entropy of $\hat h$ carries bias $\approx -(K-1)/(2m)$ nats.

## 3. State of the Art

**Established.**
- Temperature scaling (Guo et al., ICML 2017) reduces top-label ECE on hard labels at essentially no accuracy cost; reproduced widely.
- Dirichlet calibration (Kull et al., NeurIPS 2019) and scaling-binning with verified guarantees (Kumar et al., NeurIPS 2019) extend this to classwise calibration with sample-complexity bounds.
- Binned ECE is a biased, binning-dependent estimator of true calibration error (Vaicenavicius et al., AISTATS 2019; Nixon et al., CVPR-W 2019; Gruber & Buettner, NeurIPS 2022). This is a theorem plus reproduced measurement, not a conjecture.
- Human disagreement on NLI is reproducible, not annotator sloppiness (Pavlick & Kwiatkowski, TACL 2019; Nie et al., EMNLP 2020).

**Claimed but unablated.**
- Training on soft human labels improves calibration (Peterson et al., ICCV 2019 on CIFAR-10H; Collins et al., HCOMP 2022). The improvement is reported on the same distribution the soft labels came from; transfer of the gain to a *new* annotator pool is largely unablated.
- Ensembles/deep ensembles produce "better uncertainty" (Lakshminarayanan et al., NeurIPS 2017; Ovadia et al., NeurIPS 2019). Established against distribution shift with hard labels; not established against human opinion distributions.

**Benchmark-number-only.** ChaosNLI reports large JSD/KL between strong NLI models' predictive distributions and 100-annotator human distributions despite high majority-label accuracy. That is a number on one benchmark; no ablation isolates whether the gap is model miscalibration, task-format artifact, or annotator-pool composition.

## 4. What Is Known

- **CIFAR-10H** (Peterson et al., ICCV 2019): 511,400 human categorizations from 2,571 annotators over the 10,000 CIFAR-10 test images, ~50 labels/image. Training on the soft label distribution improved generalization and robustness relative to hard labels at CIFAR scale (ResNet/VGG-class models, $\sim10^7$ params).
- **ChaosNLI** (Nie et al., EMNLP 2020): 100 annotations each on 4,645 items (1,514 SNLI, 1,514 MNLI, 1,532 $\alpha$NLI). Models with majority-label accuracy in the high 80s–90s still show large distributional divergence from $h$; on a substantial fraction of items the original single gold label is not the 100-annotator majority.
- **Pavlick & Kwiatkowski** (TACL 2019): on RTE-style inference, disagreement persists under replication and richer instructions — the multi-modal response pattern reproduces across annotator batches.
- **Baan et al.** (EMNLP 2022): with human distributions available, standard ECE and human-calibration error rank models differently; temperature scaling tuned for ECE need not improve the human-distribution metric.
- **Estimator bias.** Plug-in Shannon entropy from $m$ samples over $K$ categories is biased downward by $\approx (K-1)/(2m)$ nats (Miller–Madow). At $K=3, m=5$ that is $0.2$ nats against a maximum of $\ln 3 = 1.10$ — 18% of full range.

## 5. What Is Not Known

- **Methodologically blocked (primary).** Which target — majority label, single draw, or opinion distribution — a calibration number should name is not settled, and papers report "ECE" for all three. Worse, there is no accepted decomposition of observed $\tilde h$ into ambiguity and noise, so "calibrated to human uncertainty" has no agreed referent.
- **Theoretically open.** Identifiability: under what conditions on $(\varepsilon, q, m,$ annotator covariates$)$ is $h^\star$ recoverable from $\tilde h$? Dawid–Skene (1979) identifies annotator confusion matrices under A2 and a single true label — exactly the assumption that fails when disagreement is aleatoric. No analogue exists for the case where the latent object is a *distribution*, not a label. No lower bound on samples-per-item needed for a distribution-calibration estimate with bias below a target $\delta$.
- **Empirically open.** Does soft-label training or distribution-matching recalibration transfer across annotator pools? Runnable today with two disjoint pools on the same items; not run at scale. Also open: whether frontier LLMs' verbalized or token-level probabilities track $h$ better or worse than fine-tuned encoders at matched majority accuracy.

## 6. Why It Is Hard

Three named obstructions, in order of bite.

1. **Non-identifiability.** $\tilde h = (1-\varepsilon)h^\star + \varepsilon q$ has more free parameters than observables. A 70/30 split can be one ambiguous item or two confident subgroups; no amount of *unlabelled-annotator* data separates them. Only annotator covariates or per-item elicited rationales break the tie, and those are extra measurement, not extra inference.
2. **Absent ground truth compounded by small $m$.** The target itself is estimated. At $m\le 5$ the plug-in estimate of $h$ has entropy bias of order $(K-1)/2m$, which is the same magnitude as the model-vs-human gaps being reported.
3. **An evaluation that does not measure what it names.** Binned ECE is not a consistent estimator of calibration error and its bias depends on binning; layering an estimated target on top gives an error metric with two uncontrolled biases whose signs need not agree.

Compute is *not* the obstruction. Annotation cost is: 100 labels/item over a 5,000-item eval set is $5\times10^5$ judgments.

## 7. Current Research (as of 2026)

- **Human label variation as a first-class object** — Plank's line (EMNLP 2022 position paper) and the LeWiDi shared tasks on learning with disagreement (Uma, Poesio, and collaborators): treat the distribution as the target, not the aggregate.
- **Soft-label elicitation** — Collins, Bhatt, Weller (HCOMP 2022) elicit per-annotator soft labels rather than inferring distributions from hard votes; this directly attacks obstruction 1 by adding within-annotator information.
- **Proper-scoring-rule-based calibration estimation** — Gruber & Buettner (NeurIPS 2022) and kernel calibration tests (Widmann et al., NeurIPS 2019) as replacements for binned ECE.
- **LLM uncertainty vs human distributions** *(frontier — verify)*: measuring whether instruction-tuned models' output distributions on ambiguous NLI/annotation tasks approach $h$, and whether RLHF flattens or sharpens them. Reported informally; no clean matched-accuracy ablation known to us.

## 8. Concrete Next Experiment

**Question.** Does distribution-matching recalibration transfer to a *different* annotator pool, or does it fit pool-specific idiosyncrasy?

**Scale.** 2,000 items sampled from ChaosNLI (so $K=3$ and 100 existing labels exist as a high-$m$ reference). Collect $m=30$ fresh labels per item from **two disjoint pools**, A and B, recruited from different platforms/demographic frames: $2 \times 60{,}000 = 120{,}000$ judgments, roughly $10$–$15$k USD at typical crowd rates. Model: one fixed DeBERTa-v3-large NLI checkpoint, frozen features.

**Arms.**
- *Treatment:* fit a distribution-matching recalibration head (Dirichlet or temperature+prior) on pool A's $\hat h_A$ over a 1,000-item split.
- *Control 1:* temperature scaling fit to pool A's **majority labels** only (hard-label calibration).
- *Control 2:* identity (uncalibrated).

**Deciding number.** $\Delta = \mathbb E_x\big[\mathrm{JSD}(\hat p\|\hat h_B)\big]_{\text{control 1}} - \mathbb E_x\big[\mathrm{JSD}(\hat p\|\hat h_B)\big]_{\text{treatment}}$ on the held-out 1,000 items, evaluated against **pool B**, with bootstrap CI and a matched-$m$ noise floor $\mathbb E_x[\mathrm{JSD}(\hat h_A\|\hat h_B)]$ reported alongside.

Decision rule: soft-label calibration transfers iff $\Delta > 0$ with the 95% CI excluding 0 **and** $\Delta$ exceeds 20% of the A-vs-B floor. If $\Delta \le 0$ or sits inside the floor, the reported gains in the literature are pool-fitting, and the field should report the A-vs-B floor as a mandatory denominator.

## 9. Key References

- **[Foundational]** A. P. Dawid, A. M. Skene. *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* Journal of the Royal Statistical Society C, 1979.
- **[Foundational]** E. Pavlick, T. Kwiatkowski. *Inherent Disagreements in Human Textual Inferences.* TACL, 2019.
- **[Foundational]** C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[SOTA]** J. C. Peterson, R. M. Battleday, T. L. Griffiths, O. Russakovsky. *Human Uncertainty Makes Classification More Robust.* ICCV, 2019. — arXiv:1908.07086
- **[SOTA]** Y. Nie, X. Zhou, M. Bansal. *What Can We Learn from Collective Human Opinions on Natural Language Inference Data?* EMNLP, 2020. — arXiv:2010.03532
- **[SOTA]** J. Baan, W. Aziz, B. Plank, R. Fernández. *Stop Measuring Calibration When Humans Disagree.* EMNLP, 2022. — arXiv:2210.16133
- **[SOTA]** M. Kull, M. Perelló-Nieto, M. Kängsepp, T. Silva Filho, H. Song, P. Flach. *Beyond Temperature Scaling: Obtaining Well-Calibrated Multiclass Probabilities with Dirichlet Calibration.* NeurIPS, 2019.
- **[SOTA]** A. Kumar, P. Liang, T. Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[SOTA]** S. G. Gruber, F. Buettner. *Better Uncertainty Calibration via Proper Scores for Classification and Beyond.* NeurIPS, 2022.
- **[SOTA]** K. M. Collins, U. Bhatt, A. Weller. *Eliciting and Learning with Soft Labels from Every Annotator.* AAAI HCOMP, 2022. — arXiv:2207.00810
- **[Survey]** A. Uma, T. Fornaciari, D. Hovy, S. Paun, B. Plank, M. Poesio. *Learning from Disagreement: A Survey.* JAIR, 2021.
- **[Survey]** B. Plank. *The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation.* EMNLP, 2022. — arXiv:2211.02570
- **[Context]** J. Vaicenavicius, D. Widmann, C. Andersson, F. Lindsten, J. Roll, T. B. Schön. *Evaluating Model Calibration in Classification.* AISTATS, 2019.
- **[Context]** L. Aroyo, C. Welty. *Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation.* AI Magazine, 2015.

## 10. Worked Example

One NLI item, $K=3$ (entailment / neutral / contradiction). Suppose the true opinion distribution is $h=(0.50,\,0.30,\,0.20)$, with entropy

$$H(h) = -\!\sum_y h_y\ln h_y = 0.347 + 0.361 + 0.322 = 1.030 \text{ nats.}$$

The model outputs $\hat p=(0.55,\,0.28,\,0.17)$, $H(\hat p)=0.996$ nats. Truth: the model is slightly **over**confident relative to humans, by $0.034$ nats.

Now measure $h$ the way benchmarks do. With $m=5$ annotators, the plug-in entropy has bias $\approx -(K-1)/(2m) = -2/10 = -0.20$ nats, so $\mathbb E[H(\hat h)]\approx 0.83$ nats. The measured comparison reads $H(\hat p)=0.996 > H(\hat h)=0.83$: the model appears **under**confident by $0.17$ nats. The sign of the finding flipped, and the magnitude grew fivefold, purely from the estimator.

Discreteness makes it worse. With $m=5$ and $K=3$ the plug-in $\hat h$ can only take values in multiples of $0.2$; $h=(0.50,0.30,0.20)$ is not representable at all. The modal draw, $(0.4,0.4,0.2)$ or $(0.6,0.2,0.2)$, either erases or exaggerates the entailment–neutral gap, and the *observed majority label* is neutral with probability roughly $0.2$ — so a fifth of "gold" labels for items like this are the wrong majority.

To get the entropy bias under $0.02$ nats you need $m \gtrsim (K-1)/(2\cdot 0.02) = 50$ annotators per item. That is the obstruction in one number: **detecting a 0.03-nat model-vs-human gap requires ~50 labels per item, and the datasets people report these gaps on have 1 to 5.** ChaosNLI's $m=100$ is the exception that makes the rest visible, and it covers 4,645 items in one task family.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*