---
id: 29-distillation/distilling-calibration-uncertainty
title: "Distilling Calibration and Uncertainty"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distilling Calibration and Uncertainty

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/distilling-calibration-uncertainty` · **Status:** open

## 1. Problem Statement

A teacher — an ensemble, a Bayesian posterior approximation, or a large calibrated model — carries two things a single small student is asked to inherit: **calibrated marginal probabilities** and a **decomposition of uncertainty** into aleatoric (data noise) and epistemic (model ignorance) parts. Standard knowledge distillation transfers only the mean predictive distribution, which destroys the decomposition by construction: averaging an ensemble collapses disagreement into a flatter single distribution, and a student fit to that mean cannot recover how much of the flatness was disagreement.

Three variants, different difficulty:

- **Measurement.** Given a student $s$ and teacher $t$, is there an estimator of "the student inherited the teacher's uncertainty" that is not a proxy for accuracy and is not dominated by estimator bias? Currently the field uses ECE, OOD AUROC, and selective-prediction risk-coverage, all of which conflate calibration with discrimination.
- **Method.** Build a student with $\le 1/K$ of the teacher's inference cost that matches the teacher's epistemic uncertainty on in-distribution, shifted, and out-of-distribution inputs.
- **Theory.** Characterize what a single forward pass of a fixed-capacity student can represent about a distribution over distributions, and whether the loss of epistemic signal under mean-matching is information-theoretic or an artifact of the loss.

Solved would mean: a student at $\le 1/K$ cost whose OOD/shift uncertainty rankings and calibration error match the $K$-member ensemble within measurement noise, with an ablation showing the gain is not just temperature.

## 2. Formal Setting

Inputs $x \in \mathcal{X}$, labels $y \in \{1,\dots,C\}$. The teacher is an ensemble $\{\pi^{(m)}\}_{m=1}^{M}$, each $\pi^{(m)}(y\mid x) \in \Delta^{C-1}$, inducing an empirical distribution over the simplex $q_x = \frac{1}{M}\sum_m \delta_{\pi^{(m)}}$.

**Uncertainty decomposition** (measured by Monte Carlo over the $M$ members):

$$\underbrace{\mathcal{H}\!\left[\tfrac{1}{M}\textstyle\sum_m \pi^{(m)}(\cdot\mid x)\right]}_{\text{total}} = \underbrace{\tfrac{1}{M}\textstyle\sum_m \mathcal{H}\!\left[\pi^{(m)}(\cdot\mid x)\right]}_{\text{aleatoric}} + \underbrace{\mathcal{I}(y;m\mid x)}_{\text{epistemic (mutual information)}}$$

With $M$ finite, $\mathcal{I}$ is a biased estimator: entropy of the mean is over-estimated less than the mean entropy, so $\hat{\mathcal{I}}$ carries an $O(1/M)$ bias — at $M=5$ this is not negligible relative to the effect sizes reported.

**Student.** Mean-matching distillation fits $p_\theta(y\mid x)$ to $\bar\pi(y|x)$; the epistemic term is unrepresentable. Distribution distillation fits $p_\theta(\pi \mid x) = \mathrm{Dir}(\pi \mid \alpha_\theta(x))$, $\alpha_\theta(x) \in \mathbb{R}^C_{>0}$, by
$$\mathcal{L}(\theta) = -\mathbb{E}_{x\sim\mathcal{D}_{\text{transfer}}}\ \tfrac{1}{M}\textstyle\sum_m \log \mathrm{Dir}(\pi^{(m)}(\cdot\mid x)\mid \alpha_\theta(x)),$$
recovering $\mathcal{I}$ in closed form from $\alpha_0 = \sum_c \alpha_c$.

**Measured quantities.**
- Calibration: binned ECE $\widehat{\mathrm{ECE}} = \sum_{b=1}^{B}\frac{|B_b|}{N}\left|\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\right|$. This estimator is biased upward and $B$-dependent (Vaicenavicius et al., AISTATS 2019; Nixon et al., CVPRW 2019). Report equal-mass binning, $B$, and a bootstrap CI, or the number is uninterpretable.
- Epistemic fidelity: Spearman $\rho$ between student and teacher per-example $\mathcal{I}$ on a held-out mixture of in-distribution, shifted, and OOD inputs — *not* AUROC, which only needs a monotone transform to be right.
- Compute: teacher $M$ forward passes vs. student 1.

**Assumptions known to be violated.** (i) The transfer set covers the region where epistemic uncertainty matters — false; ensembles disagree exactly where transfer data is absent. (ii) The Dirichlet family contains the ensemble's simplex distribution — false at large $C$; ensemble members are near-vertex and near-deterministic, so the ML Dirichlet fit is degenerate (Ryabinin et al., NeurIPS 2021). (iii) Members are exchangeable samples from a posterior — false; they are SGD modes with shared data and architecture.

## 3. State of the Art

**Method SOTA (established).** *Ensemble Distribution Distillation* (Malinin, Mlodozeniec, Gales, ICLR 2020) — student predicts a Dirichlet, recovers total/aleatoric/epistemic on CIFAR-10/100 with $M=5$–10. Its failure at $C=1000$ is documented, not hidden. *Proxy-target EnD²* (Ryabinin, Malinin, Gales, NeurIPS 2021) fixes the degeneracy by fitting a proxy Dirichlet to teacher moments and is the first method to run on ImageNet-scale class counts. *Hydra* (Tran et al., 2020) keeps $M$ lightweight heads on a shared trunk — cheaper than an ensemble but not a single-pass student.

**Baseline that keeps winning (established).** Deep ensembles (Lakshminarayanan et al., NIPS 2017), evaluated under shift by Ovadia et al. (NeurIPS 2019): ensembles dominate every single-model uncertainty method on shifted data, and the gap widens with shift severity. No distilled student has been shown to close it.

**Claimed but unablated.** Most "our distilled student is better calibrated" results report post-hoc ECE without a temperature-scaled control; temperature scaling (Guo et al., ICML 2017) is a one-parameter fix that recovers most in-distribution ECE, so any calibration claim without it is uninformative. Claims that distillation transfers *epistemic* uncertainty are usually evidenced by OOD AUROC on CIFAR-10 vs. SVHN — a benchmark number, and one that a well-tuned max-softmax baseline already gets $\approx 0.90$ AUROC on.

**LLM regime.** Semantic entropy (Kuhn, Gal, Farquhar, ICLR 2023; Farquhar et al., *Nature* 2024) is the strongest sequence-level uncertainty signal, but it costs $\sim$10 samples per query. Distilling it into a single-pass probe is the open frontier; published probe results exist but without an equal-compute control.

## 4. What Is Known

- **Modern nets are miscalibrated and the fix is cheap in-distribution.** ResNet-110 on CIFAR-100: ECE $\approx 16.5\%$, reduced to $\approx 1$–$2\%$ by a single temperature (Guo et al., ICML 2017; scale $\sim$1.7M–70M params).
- **Calibration under shift does not survive.** Ovadia et al. (NeurIPS 2019): temperature scaling fit on the in-distribution validation set degrades monotonically with corruption severity on CIFAR-10-C/ImageNet-C; ensembles of $M=5$ degrade least. Measured at ResNet-20/ResNet-50 scale.
- **Students do not match teachers even where they could.** Stanton et al. (NeurIPS 2021): on CIFAR-100 and ImageNet, self-distilled students often agree with the teacher on well under 90% of *training* points despite matching test accuracy — distillation fidelity is an optimization failure, not a capacity failure.
- **Deviations are systematic, not noise.** Nagarajan et al. (NeurIPS 2023) show students consistently *exaggerate* teacher confidence on points the teacher is confident about — a directional bias that predicts worse calibration than the teacher.
- **Dirichlet distillation fails as $C$ grows.** Ryabinin et al. (NeurIPS 2021) report the ML Dirichlet objective becoming ill-conditioned well before $C=1000$; proxy targets restore ImageNet-scale training.
- **RLHF destroys calibration in LLMs.** GPT-4 technical report (OpenAI, 2023): MMLU ECE $\approx 0.007$ pre-RLHF, $\approx 0.073$ post-RLHF — a $10\times$ degradation from a post-training step, on the same base model.
- **Label smoothing erases the structure distillation needs** (Müller, Kornblith, Hinton, NeurIPS 2019) — smoothed teachers are better calibrated but worse teachers.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted estimator of "epistemic uncertainty transfer" that is invariant to accuracy. ECE is biased and binning-dependent; OOD AUROC is rank-only and saturated on the standard pairs; risk-coverage confounds discrimination with calibration. Worse, the teacher's own $\mathcal{I}$ at $M=5$ is a biased estimate of a quantity with no ground truth — there is no reference epistemic uncertainty to distill *toward*.
- **Theoretically open.** No result characterizes the capacity a single deterministic network needs to represent an $M$-member ensemble's simplex distribution to accuracy $\epsilon$ over a domain, nor whether the loss under mean-matching is information-theoretically forced. No lower bound rules out a single-pass student matching an ensemble's shift calibration.
- **Empirically open.** Nobody has run distribution distillation at LLM scale (7B student, $M=5$ 70B teachers) with an equal-inference-compute control arm. Also unrun: whether distilling *semantic* entropy into a single pass survives distribution shift.

## 6. Why It Is Hard

The obstruction is **absent ground truth compounded by confounded measurement**. Epistemic uncertainty is defined relative to a posterior nobody has; the $M=5$ ensemble estimate is itself the target *and* the only reference, so "the student matched the teacher" and "the student is right" are not separable. On top of that, the standard scoreboard does not measure what it names: ECE's plug-in estimator has bias of order $B/N$ and can be driven to near-zero by a model that is uninformative but marginally correct, and OOD AUROC rewards any monotone reindexing of confidence — a student can score identically to its teacher while its absolute uncertainty values are wrong by a factor of three. Add the transfer-set problem: the signal lives exactly where the transfer data is not.

## 7. Current Research (as of 2026)

- **Proxy-moment and moment-matching distillation** at large $C$, continuing the Cambridge/Yandex line (Malinin, Gales, Ryabinin). Established through NeurIPS 2021; extensions to sequence models are *(frontier — verify)*.
- **Distilling sampling-based LLM uncertainty into single-pass probes** — semantic-entropy regression heads, following OATML (Gal, Farquhar, Kuhn). Active; equal-compute controls mostly missing *(frontier — verify)*.
- **Conformal prediction as the transfer target** — distill a set-valued predictor with a finite-sample coverage guarantee, sidestepping the missing ground truth. Coverage is measurable; efficiency (set size) is the objective. Growing, and the most likely route around the measurement block.
- **Proper-score decompositions replacing ECE** (Gruber & Buettner, NeurIPS 2022) as the reporting standard.

## 8. Concrete Next Experiment

**Question:** does distribution distillation transfer anything beyond a learned temperature?

- **Scale.** CIFAR-100 and ImageNet-1k. Teacher: $M=10$ independently seeded ResNet-50s. Students: single ResNet-50 (equal per-example inference cost), three seeds each.
- **Arms.** (1) Mean-matching KD. (2) Proxy-Dirichlet EnD². (3) **Control arm: mean-matching KD plus a per-example temperature head $\tau_\theta(x)$**, trained to match the teacher's total entropy — one extra scalar output, no distribution-over-distributions machinery.
- **Evaluation.** Held-out mixture: clean, CIFAR-100-C / ImageNet-C at severities 1–5, and one far-OOD set. Report Spearman $\rho$ between student and teacher per-example mutual information $\mathcal{I}$, with a bootstrap 95% CI over examples and seeds.
- **Deciding number.** $\Delta\rho = \rho(\text{EnD}^2) - \rho(\text{control})$ on severity-5 shift. If the CI for $\Delta\rho$ includes 0, distribution distillation buys nothing an input-dependent temperature does not, and the field's method line is chasing a reparameterization. A pre-registered effect size worth caring about: $\Delta\rho \ge 0.10$.

## 9. Key References

- **[Foundational]** Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop, 2014. — arXiv:1503.02531
- **[Foundational]** Lakshminarayanan, Pritzel, Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NIPS, 2017. — arXiv:1612.01474
- **[Foundational]** Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[SOTA]** Malinin, Mlodozeniec, Gales. *Ensemble Distribution Distillation.* ICLR, 2020. — arXiv:1905.00076
- **[SOTA]** Ryabinin, Malinin, Gales. *Scaling Ensemble Distribution Distillation to Many Classes with Proxy Targets.* NeurIPS, 2021. — arXiv:2105.06987
- **[SOTA]** Malinin, Gales. *Predictive Uncertainty Estimation via Prior Networks.* NeurIPS, 2018. — arXiv:1802.10501
- **[Empirical]** Ovadia, Fertig, Ren, Nado, Sculley, Nowozin, Dillon, Lakshminarayanan, Snoek. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[Empirical]** Stanton, Izmailov, Kirichenko, Alemi, Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- **[Empirical]** Nagarajan, Menon, Bhojanapalli, Rawat, Kumar. *On Student-Teacher Deviations in Distillation: Does It Pay to Disobey?* NeurIPS, 2023.
- **[Measurement]** Vaicenavicius, Widmann, Andersson, Lindsten, Roll, Schön. *Evaluating Model Calibration in Classification.* AISTATS, 2019. — arXiv:1902.06977
- **[Measurement]** Nixon, Dusenberry, Zhang, Jerfel, Tran. *Measuring Calibration in Deep Learning.* CVPR Workshops, 2019.
- **[Measurement]** Gruber, Buettner. *Better Uncertainty Calibration via Proper Scores for Classification and Beyond.* NeurIPS, 2022.
- **[LLM]** Kuhn, Gal, Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR, 2023. — arXiv:2302.09664
- **[LLM]** Farquhar, Kossen, Kuhn, Gal. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature 630, 2024.
- **[LLM]** Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Related]** Müller, Kornblith, Hinton. *When Does Label Smoothing Help?* NeurIPS, 2019. — arXiv:1906.02629

## 10. Worked Example

Take one CIFAR-100-C image at corruption severity 5, teacher = 10 ResNet-50s. Suppose 6 members put mass $0.9$ on class A and 4 put $0.9$ on class B (remaining mass spread uniformly).

- Mean predictive: $\bar\pi \approx (0.54, 0.36, \text{rest} \approx 0.001)$. Total entropy $\mathcal{H}[\bar\pi] \approx 1.05$ nats.
- Mean member entropy $\approx 0.55$ nats (each member is confident).
- Epistemic $\hat{\mathcal{I}} \approx 0.50$ nats — nearly half the total. This is the number that says "the ensemble is split, do not trust this."

Now a mean-matching student fit to $\bar\pi$. At convergence it outputs something near $(0.54, 0.36, \dots)$ and reports $\mathcal{I} = 0$ by construction: a single softmax has no disagreement to report. Its *total* entropy, though, is $\approx 1.05$ — matching the teacher exactly. So on any metric built from total uncertainty (ECE, max-softmax OOD AUROC, risk-coverage), this student is indistinguishable from the ensemble on this example, while having discarded 100% of the epistemic signal.

The obstruction is now visible twice over. First, the standard scoreboard cannot see the loss. Second, the "correct" answer is unavailable: the $\hat{\mathcal{I}} = 0.50$ nats is itself an $M=10$ estimate — resampling 10 fresh seeds gives a different split (7/3 gives $\approx 0.42$ nats, 5/5 gives $\approx 0.55$), a spread of roughly $\pm 25\%$ that is comparable to the improvements distribution-distillation papers report. There is no larger-$M$ reference computed in these papers to check against, because $M=100$ ResNet-50 trainings is the experiment nobody runs. Until someone does, "the student matched the teacher's epistemic uncertainty" is a claim about a target whose own error bar is unmeasured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*