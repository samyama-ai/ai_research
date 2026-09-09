---
id: 29-distillation/distillation-under-distribution-shift
title: "Distillation Under Distribution Shift"
topic: 29-distillation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distillation Under Distribution Shift

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/distillation-under-distribution-shift` · **Status:** empirically-open

## 1. Problem Statement

Knowledge distillation fits a student to a teacher's outputs on a **transfer set**. Deployment happens on a distribution that differs from that transfer set. The problem: characterise and control how much of a teacher's *out-of-distribution* behaviour — accuracy, calibration, robustness, refusal behaviour — survives distillation, when the transfer set contains no samples from the shifted distribution.

Three variants, with different difficulty:

- **Measurement.** Given teacher $f_t$, student $f_s$, transfer distribution $\mathcal{D}_{\mathrm{tr}}$ and target $\mathcal{D}_{\mathrm{te}}$, define a quantity that isolates *distillation-induced* robustness loss from the robustness the student's architecture and data would have had anyway. Currently ill-posed: most papers report student OOD accuracy with no matched-ID-accuracy control.
- **Method.** Choose the transfer set, temperature, loss and augmentation so that $f_s$ matches $f_t$ on $\mathcal{D}_{\mathrm{te}}$, not just on $\mathcal{D}_{\mathrm{tr}}$.
- **Theory.** Bound $\varepsilon_{\mathrm{te}}(f_s) - \varepsilon_{\mathrm{te}}(f_t)$ in terms of a divergence between $\mathcal{D}_{\mathrm{tr}}$ and $\mathcal{D}_{\mathrm{te}}$, the student hypothesis class, and the teacher's off-support behaviour.

Solved would mean: a recipe that, at fixed student capacity and fixed ID accuracy, provably or reliably preserves the teacher's *effective robustness* to within 1 probit point across a standard shift suite.

## 2. Formal Setting

Inputs $x \in \mathcal{X}$, labels $y \in [K]$. Teacher $f_t: \mathcal{X} \to \Delta^{K-1}$ is fixed. Student $f_s(\cdot;\theta)$ is trained on $n$ samples from the transfer distribution $\mathcal{D}_{\mathrm{tr}}$ by

$$\hat\theta = \arg\min_\theta \; \frac1n \sum_{i=1}^n \big[(1-\alpha)\,\ell_{\mathrm{CE}}(f_s(x_i), y_i) + \alpha\, T^2\, \mathrm{KL}\big(f_t^T(x_i)\,\|\,f_s^T(x_i)\big)\big],$$

with temperature $T$ and mixing weight $\alpha$; $f^T$ denotes softmax at temperature $T$.

Measured quantities:

- **Risk** $\varepsilon_{\mathcal{D}}(f) = \Pr_{(x,y)\sim\mathcal{D}}[\arg\max f(x) \neq y]$, estimated on a held-out test split; report a binomial CI, since shift suites are small ($10^3$–$5\times10^4$ images).
- **Fidelity** (Stanton et al. 2021): top-1 agreement $A_{\mathcal{D}} = \Pr_{x\sim\mathcal{D}}[\arg\max f_s(x) = \arg\max f_t(x)]$ and predictive KL $\kappa_{\mathcal{D}} = \mathbb{E}_x\,\mathrm{KL}(f_t(x)\|f_s(x))$. Fidelity needs no labels, so it is measurable on unlabelled shifted data.
- **Distillation shift gap**, the object of interest:
$$\Delta = \big[\varepsilon_{\mathrm{te}}(f_s) - \varepsilon_{\mathrm{te}}(f_t)\big] - \big[\varepsilon_{\mathrm{tr}}(f_s) - \varepsilon_{\mathrm{tr}}(f_t)\big].$$
$\Delta > 0$ means the student loses *more* to the teacher off-distribution than on it.
- **Effective robustness** (Taori et al. 2020): $\rho(f) = \mathrm{acc}_{\mathrm{te}}(f) - \beta\big(\mathrm{acc}_{\mathrm{tr}}(f)\big)$, where $\beta$ is a linear fit in probit space over a reference model population. This is the only widely used estimator that controls for ID accuracy; it is only as good as the reference population.
- **Budget:** student FLOPs, transfer-set size $n$, and teacher inference cost $n \cdot C_t$ per epoch (dominant for LLM teachers).

Assumptions and their violations:

| Assumption | Status in practice |
|---|---|
| Covariate shift: $p_{\mathrm{te}}(y\mid x) = p_{\mathrm{tr}}(y\mid x)$ | Approximately holds for ImageNet-V2/-R/-Sketch; fails for label-shifted or adversarial suites |
| Bounded density ratio $\sup_x p_{\mathrm{te}}(x)/p_{\mathrm{tr}}(x) < \infty$ | **Violated.** Rendition and sketch shifts lie largely outside the transfer support; importance-weighting bounds are vacuous |
| Teacher probabilities approximate $p(y\mid x)$ off-support | **Violated.** Teachers are confidently wrong under shift (Ovadia et al. 2019); the student then matches a wrong target |
| Student can realise the teacher (capacity) | Violated by design in compression; also fails *optimisation*-wise even when capacity is sufficient (Stanton et al. 2021) |

## 3. State of the Art

**Established.** Function matching — long schedules, *consistent* teacher/student input views, aggressive mixup — is the strongest known image-distillation recipe: Beyer et al. (CVPR 2022) reach **82.8%** ImageNet top-1 with a ResNet-50 student distilled from a BiT-M ResNet-152x2, at ~9,600 epochs. The paper's ablation isolates view consistency and schedule length as the causal factors, and shows the recipe transfers across the five datasets tested. Adversarially Robust Distillation (Goldblum et al., AAAI 2020) shows robustness *can* be transferred when the transfer objective itself contains the shift (adversarial examples generated during training).

**Claimed but unablated.** That distillation "transfers robustness" as a general property. Most reports compare a distilled student to a same-architecture baseline at *different* ID accuracy, so the OOD gain is confounded with the ID gain; effective robustness is rarely reported. Ojha et al. (NeurIPS 2023) directly probe which teacher properties transfer and find the picture is property-dependent, not uniform.

**Benchmark-number-only.** Nearly all CLIP-distillation results (TinyCLIP, ICCV 2023, and successors) report zero-shot ImageNet plus a shift suite as a table, with no matched-ID control and no transfer-set ablation. They establish that a small student *can* be robust; they do not establish that distillation is what made it robust.

**Theory SOTA.** Ben-David et al. (Machine Learning, 2010) gives the target-risk bound $\varepsilon_{\mathrm{te}}(h) \le \varepsilon_{\mathrm{tr}}(h) + \tfrac12 d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_{\mathrm{tr}},\mathcal{D}_{\mathrm{te}}) + \lambda$, but for a *hypothesis*, not for a teacher–student pair. Phuong & Lampert (ICML 2019) and Ji & Zhu (NeurIPS 2020) give transfer-risk bounds for linear/wide-network students — both in-distribution only.

## 4. What Is Known

- **Fidelity is low even without shift.** Stanton et al. (NeurIPS 2021) show that on CIFAR-100 and ImageNet, students fail to match the teacher even on the *training* points, including in self-distillation where capacity is not the constraint. Optimisation, not capacity, is the binding limit. Generalisation improves while fidelity stays poor — the two are decoupled.
- **Robustness follows data, not objective.** Fang et al. (ICML 2022) show CLIP's effective robustness is caused by its pre-training image distribution: matching the training set makes the contrastive and supervised objectives equally (non-)robust. This is the strongest available evidence that a transfer set drawn from ImageNet cannot teach ImageNet-R behaviour whatever the loss.
- **ID accuracy predicts OOD accuracy on a tight line.** Taori et al. (NeurIPS 2020), over 200+ ImageNet models: effective robustness is ≈0 for essentially all models trained on standard data; only different/larger training data moves off the line. Recht et al. (ICML 2019) measured the ImageNet-V2 drop at 11–14 top-1 points.
- **Language distillation does not transfer held-out capability.** Gudibande et al. (arXiv:2305.15717, 2023): imitation models trained on outputs of a stronger model close the human-preference gap on the imitation distribution while showing no improvement on Natural Questions, HumanEval or MMLU. Style transfers; capability under task shift does not.
- **Calibration degrades under shift for all methods tested.** Ovadia et al. (NeurIPS 2019), across image/text/genomics at ImageNet scale: every uncertainty method's calibration worsens monotonically with shift severity. Distilled students inherit an already-miscalibrated target.

## 5. What Is Not Known

- **Empirically open.** Whether, at *matched ID accuracy and matched transfer-set distribution*, a distilled student's effective robustness $\rho(f_s)$ equals its teacher's $\rho(f_t)$. Every ingredient exists — CLIP teachers, standard shift suites, the probit-fit protocol — but no published run varies only the transfer distribution with the teacher held fixed. This is the load-bearing gap.
- **Empirically open.** Whether high on-transfer fidelity ($\kappa_{\mathrm{tr}}\to 0$) implies off-transfer fidelity. Since fidelity is label-free, this can be measured directly on unlabelled shifted data.
- **Theoretically open.** No bound on $\Delta$ (§2) that is non-vacuous when the density ratio is unbounded. Existing distillation-risk bounds assume a single distribution; existing domain-adaptation bounds do not model a teacher.
- **Methodologically blocked.** "Robustness transferred by distillation" has no agreed estimator. Effective robustness is defined against a *reference model population*; a student and its teacher may not belong to the same population (different data, architecture family, and pre-training), so the fitted $\beta$ is not shared and $\rho(f_s)-\rho(f_t)$ is not obviously a valid comparison.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent off-support ground truth**.

1. Under shift, the teacher is itself wrong at an unknown, input-dependent rate. Matching it perfectly would be *bad*. So "student ≠ teacher on OOD data" cannot be scored as a failure without knowing whether the disagreement was on points where the teacher was right — which requires labels the shift suites give only in aggregate.
2. Any measured OOD gain is confounded with ID accuracy through the Taori line. Removing the confound requires matching students at equal ID accuracy, which means training multiple students per condition and interpolating — roughly a 3–5× compute multiplier.
3. Distillation is not identifiable from the loss: two transfer sets can produce students with identical ID accuracy and identical transfer-set fidelity but different OOD behaviour, because the loss constrains $f_s$ only on $\mathrm{supp}(\mathcal{D}_{\mathrm{tr}})$. Off-support the student is determined by inductive bias, which no term in the objective controls.

## 7. Current Research (as of 2026)

- **Data-centric distillation for VLMs.** Transfer sets drawn from web-scale pools rather than curated ImageNet, motivated by Fang et al. Groups: UW/LAION-adjacent open-source efforts, Google DeepMind's function-matching line *(frontier — verify current results)*.
- **Fidelity-first evaluation.** Following Stanton et al., reporting agreement and KL alongside accuracy; still a minority practice.
- **Synthetic transfer sets and generator-augmented distillation** to cover the shifted region without collecting shifted data. Interacts with model-collapse results (Shumailov et al., *Nature*, 2024) showing recursive synthetic training loses distribution tails — precisely the region shift lives in *(frontier — verify)*.
- **LLM reasoning distillation under task shift**, post *Distilling Step-by-Step* (Hsieh et al., ACL Findings 2023): whether rationale supervision generalises off the rationale distribution is actively contested *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** At matched ID accuracy, does the transfer distribution — not the distillation loss — determine the student's effective robustness?

**Scale.** One fixed teacher: CLIP ViT-L/14. Students: ViT-B/16 (86M params), 32 epochs over $n = 1.28$M images. Three transfer sets, each of exactly $n$ images: (A) ImageNet-1k train images; (B) an equal-size LAION subset; (C) 50/50 mix. Two losses per set: soft-label KD ($T=2$, $\alpha=1$) and hard-label CE on teacher's argmax. Three seeds, plus 3 checkpoints per run to interpolate to matched ID accuracy. 18 runs × ~100 A100-hours ≈ 1,800 A100-hours (~$4k spot).

**Control arm.** The hard-label CE students on the *same* transfer sets. If soft labels carry OOD information beyond the images themselves, KD must beat CE **within** each transfer set.

**Deciding number.** $\rho(f_s) - \rho(f_t)$ in probit points on the suite {ImageNet-V2, -R, -A, -Sketch, ObjectNet}, evaluated at ID accuracy matched to $\pm 0.3$ points via checkpoint interpolation. Decision rule: if $|\rho_{\mathrm{KD}} - \rho_{\mathrm{CE}}| < 1$ probit point within every transfer set while $\rho$ varies by $>3$ points *across* transfer sets, the loss is irrelevant and the problem is a data-coverage problem. The reverse pattern would establish that soft labels carry off-support signal — currently unestablished.

## 9. Key References

- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NIPS 2014 Deep Learning Workshop. — arXiv:1503.02531
- **[Foundational]** Cristian Buciluă, Rich Caruana, Alexandru Niculescu-Mizil. *Model Compression.* KDD, 2006.
- **[Foundational]** Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, Jennifer Wortman Vaughan. *A Theory of Learning from Different Domains.* Machine Learning 79(1–2), 2010.
- **[SOTA]** Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, Andrew Gordon Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- **[SOTA]** Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov. *Knowledge Distillation: A Good Teacher Is Patient and Consistent.* CVPR, 2022. — arXiv:2106.05237
- **[SOTA]** Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, Ludwig Schmidt. *Data Determines Distributional Robustness in Contrastive Language Image Pre-training (CLIP).* ICML, 2022.
- **[Measurement]** Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, Ludwig Schmidt. *Measuring Robustness to Natural Distribution Shifts in Image Classification.* NeurIPS, 2020. — arXiv:2007.00644
- **[Measurement]** Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, Vaishaal Shankar. *Do ImageNet Classifiers Generalize to ImageNet?* ICML, 2019. — arXiv:1902.10811
- **[Theory]** Mary Phuong, Christoph H. Lampert. *Towards Understanding Knowledge Distillation.* ICML, 2019.
- **[Theory]** Guangda Ji, Zhanxing Zhu. *Knowledge Distillation in Wide Neural Networks: Risk Bound, Data Efficiency and Imperfect Teacher.* NeurIPS, 2020.
- **[Empirical]** Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, Dawn Song. *The False Promise of Imitating Proprietary LLMs.* ICLR, 2024. — arXiv:2305.15717
- **[Empirical]** Yaxin Ovadia et al. (Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek). *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019.
- **[Empirical]** Micah Goldblum, Liam Fowl, Soheil Feizi, Tom Goldstein. *Adversarially Robust Distillation.* AAAI, 2020. — arXiv:1905.09747
- **[Survey]** Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao. *Knowledge Distillation: A Survey.* International Journal of Computer Vision, 2021.

## 10. Worked Example

**Setup.** Teacher: CLIP ViT-L/14 zero-shot. As reported in Radford et al. (ICML 2021), it scores roughly 76% ImageNet top-1 and *far above the ImageNet-trained trend* on shift suites — ImageNet-R near 89%, ImageNet-Sketch near 60%. The matched-ID-accuracy supervised comparison in the same paper, a ResNet-101 at ≈76% ImageNet, scores ≈38% on ImageNet-R and ≈25% on ImageNet-Sketch. The teacher's effective robustness on ImageNet-R is therefore about **+51 points** over the ImageNet-trained trend.

**Distil it.** Take the teacher, generate soft labels on the 1.28M ImageNet train images, train a ResNet-50 student. Suppose the student reaches 76% ImageNet top-1 — full ID fidelity to the teacher's headline number.

**Now bound the fidelity you can possibly observe on ImageNet-R.** Agreement is constrained by the two accuracies:

$$A_{\mathrm{R}} \le 1 - \big|\mathrm{acc}_{\mathrm{R}}(f_t) - \mathrm{acc}_{\mathrm{R}}(f_s)\big|.$$

If the student lands on the ImageNet-trained trend, $\mathrm{acc}_{\mathrm{R}}(f_s) \approx 0.38$, so $A_{\mathrm{R}} \le 1 - (0.89 - 0.38) = \mathbf{0.51}$. Against an in-distribution agreement typically above 0.80, the student that *looks* like a faithful copy on ImageNet val is provably disagreeing with its teacher on at least half of ImageNet-R. The distillation shift gap is $\Delta \approx (0.62 - 0.11) - (0.24 - 0.24) = +0.51$.

**Where the obstruction becomes visible.** Two explanations produce the identical table: (i) distillation destroys robustness — the KD objective fails to transmit the teacher's off-support function; (ii) the transfer set never sampled the region where teacher and student differ, so the loss placed no constraint there at all. The single-arm experiment cannot separate them, because both predict $A_{\mathrm{R}} \approx 0.5$ with $A_{\mathrm{val}} \approx 0.85$. Only the transfer-set arm of §8 — same teacher, same loss, LAION images instead of ImageNet images — distinguishes them, and that arm has not been run at matched ID accuracy.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*