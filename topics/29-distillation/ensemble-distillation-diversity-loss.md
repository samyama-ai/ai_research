---
id: 29-distillation/ensemble-distillation-diversity-loss
title: "Ensemble Distillation Diversity Loss"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Ensemble Distillation Diversity Loss

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/ensemble-distillation-diversity-loss` · **Status:** open

## 1. Problem Statement

Distilling a deep ensemble into a single student preserves the ensemble's *mean* prediction but discards the *spread* across members. That spread is the ensemble's estimate of epistemic uncertainty — the part of uncertainty that shrinks with more data, as opposed to aleatoric label noise. A student trained on the averaged teacher probabilities has, by construction, zero member-to-member disagreement: its epistemic uncertainty estimate is exactly $0$.

Three variants, of different difficulty:

- **Measurement.** Given a teacher ensemble and a student, quantify how much of the ensemble's disagreement signal survives distillation. Is the right target the *value* of the mutual information $\mathbb{I}[y;\theta \mid x]$ per input, its *ranking* across inputs (what OOD detection actually uses), or its *calibration* against held-out error?
- **Method.** Build a single-forward-pass student whose epistemic uncertainty matches the ensemble's on in-distribution, shifted, and out-of-distribution inputs, at inference cost $O(1)$ instead of $O(M)$ members.
- **Theory.** Characterize what a single parametric predictive-distribution-over-distributions can and cannot represent about an $M$-member ensemble, and whether the observed gap is an identifiability limit or an optimization failure.

Solved would mean: a student at $\le 1.2\times$ single-model inference FLOPs that reaches $\ge 95\%$ of the ensemble's OOD-detection AUROC *and* $\le 1.5\times$ the ensemble's epistemic-uncertainty calibration error, on ImageNet-scale data, reproduced independently.

## 2. Formal Setting

Inputs $x \in \mathcal{X}$, labels $y \in \{1,\dots,K\}$. A teacher ensemble is $M$ parameter draws $\theta_1,\dots,\theta_M$ from a training-induced distribution $q(\theta)$ (in practice: independent SGD runs with different seeds and data order). Member predictive: $p_m(y\mid x) = p(y \mid x, \theta_m) \in \Delta^{K-1}$.

**Measured quantities.** Ensemble mean $\bar p(y\mid x) = \frac1M \sum_m p_m(y \mid x)$. The uncertainty decomposition (Depeweg et al., ICML 2018) is

$$\underbrace{\mathcal{H}[\bar p(y\mid x)]}_{\text{total}} \;=\; \underbrace{\tfrac1M\textstyle\sum_m \mathcal{H}[p_m(y\mid x)]}_{\text{expected data uncertainty}} \;+\; \underbrace{\mathbb{I}_M[y;\theta\mid x]}_{\text{knowledge uncertainty}},$$

with $\mathbb{I}_M[y;\theta\mid x] = \frac1M\sum_m \mathrm{KL}\!\left(p_m(\cdot\mid x)\,\|\,\bar p(\cdot\mid x)\right)$ — computed exactly from the $M$ stored member outputs, in nats. This is the *diversity* the problem is about. Note $\mathbb{I}_M$ is a biased-low estimate of the $M\to\infty$ quantity; the bias is $O(1/M)$ and is rarely reported.

**Student.** Standard distillation (Hinton et al., 2015) fits $p_\phi$ by $\min_\phi \mathbb{E}_x\,\mathrm{KL}(\bar p \,\|\, p_\phi)$ at temperature $\tau$. Then $\mathbb{I}[y;\theta\mid x] \equiv 0$ for the student: total and aleatoric uncertainty collapse onto each other. Ensemble *distribution* distillation (EnD², Malinin et al., ICLR 2020) instead fits a Dirichlet $\mathrm{Dir}(\pi \mid \alpha_\phi(x))$ over the simplex by maximum likelihood on member outputs:

$$\mathcal{L}(\phi) = -\,\mathbb{E}_x\,\tfrac1M\textstyle\sum_m \log \mathrm{Dir}\!\left(p_m(\cdot\mid x)\,\middle|\,\alpha_\phi(x)\right),$$

giving a closed-form student knowledge uncertainty $\hat{\mathbb{I}}_\phi(x) = \mathcal{H}[\bar\pi] - \mathbb{E}_{\pi}\mathcal{H}[\pi]$ with $\bar\pi = \alpha_\phi/\alpha_0$, $\alpha_0 = \sum_k \alpha_{\phi,k}$.

**Diversity retention** as measured: $R_\rho = \mathrm{corr}_\rho\!\left(\hat{\mathbb{I}}_\phi(x),\, \mathbb{I}_M(x)\right)$ (Spearman, over a held-out input set), plus the ratio of OOD AUROC obtained by ranking with $\hat{\mathbb{I}}_\phi$ versus $\mathbb{I}_M$.

**Assumptions, and which are violated.** (i) *Dirichlet adequacy* — that member outputs are Dirichlet-distributed on the simplex. Violated: real member outputs are multi-modal (different members commit to different wrong classes), and the Dirichlet is unimodal. (ii) *$M$ samples suffice to define the target* — violated at $M=5$–$10$, the usual budget. (iii) *Distillation transfer set covers the region where diversity matters* — violated: disagreement is concentrated off the training manifold, which the transfer set by definition under-samples. (iv) *Numerical tractability of the Dirichlet NLL at large $K$* — violated at $K=1000$; near-zero member probabilities make $\log \mathrm{Dir}$ diverge, which is the explicit motivation for proxy targets (Ryabinin et al., NeurIPS 2021).

## 3. State of the Art

**Empirical SOTA (established).**
- *EnD² (Malinin, Mlodozeniec, Gales, ICLR 2020)* — Dirichlet student recovers a usable knowledge-uncertainty signal on CIFAR-10/100 with $K\le 100$. Established: the mean-distillation baseline has no epistemic signal at all; EnD² beats it on OOD ranking. Not established: that it matches the ensemble.
- *Proxy-target EnD² (Ryabinin, Malinin, Gales, NeurIPS 2021)* — replaces the exact Dirichlet NLL with proxy targets to make training stable at $K=1000$ (ImageNet). This is the first ensemble-distribution distillation to run at ImageNet scale.
- *Hydra (Tran et al., 2020)* — one shared body, $M$ lightweight heads, each matched to one member. Retains member-level diversity by construction at $\approx M$ head-cost, not $O(1)$.
- *Diversity Matters When Learning From Ensembles (Nam, Yoon, Lee, Lee, NeurIPS 2021)* — perturbs distillation inputs (output-diversified sampling) to expose regions of member disagreement; reported gains in student calibration/OOD on CIFAR and ImageNet.

**Claimed but unablated.** Most papers report OOD-detection AUROC and ECE, not $R_\rho$ against the teacher's per-input $\mathbb{I}_M$. So "diversity is retained" is inferred from a downstream benchmark number rather than measured directly. Almost no paper ablates $M$ while holding the transfer set fixed, so the reported retention is confounded with teacher quality.

**Theory SOTA.** Allen-Zhu & Li (*Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning*, ICLR 2023) prove, in a multi-view data model, that ensembles gain by covering disjoint feature views and that distillation transfers those views to a single student — an accuracy-side result. There is no corresponding theorem for uncertainty: nothing bounds how much of $\mathbb{I}_M$ a single network of given capacity can represent.

## 4. What Is Known

- **Ensembles are the strongest baseline under shift.** Ovadia et al. (NeurIPS 2019), across MNIST/CIFAR-10/ImageNet and text, found deep ensembles ($M=5$–$10$) best in calibration under distribution shift, beating temperature scaling, dropout, and SVI. Gains largely saturate by $M \approx 5$.
- **Mean distillation destroys epistemic uncertainty exactly, not approximately.** $\mathbb{I} = 0$ is an identity, not an empirical finding.
- **Students do not match teachers even on the mean.** Stanton et al. (*Does Knowledge Distillation Really Work?*, NeurIPS 2021) show a persistent top-1 agreement gap between student and teacher on the *training* set — an optimization failure, not a capacity or generalization one, at CIFAR-100/ImageNet with ResNet students. This bounds any diversity-retention claim from below.
- **Diversity comes from mode separation.** Fort, Hu, Lakshminarayanan (2019) show independent SGD runs land in distinct loss-landscape basins with low function-space similarity; within-basin methods (dropout, subspace) give much less diversity. So the signal being distilled is genuinely multi-modal.
- **Cheap ensembles cut cost but not to $O(1)$.** BatchEnsemble (Wen et al., ICLR 2020) and MIMO (Havasi et al., ICLR 2021) recover much of a deep ensemble's accuracy/calibration at a fraction of parameters or one forward pass with $M$ input/output slots — MIMO reported near-ensemble CIFAR/ImageNet performance at roughly single-model cost. These are alternatives to distillation, not solutions to it.

## 5. What Is Not Known

- **Theoretically open.** No bound on the representable epistemic uncertainty of a single network with a parametric simplex distribution head. Specifically: is there a capacity-independent obstruction to a unimodal Dirichlet reproducing $\mathbb{I}_M$ when members are multi-modal? No proof either way.
- **Theoretically open.** Non-identifiability: infinitely many $\alpha_\phi(x)$ give the same $\bar\pi$ and nearly the same $\hat{\mathbb{I}}$, so the fitted concentration $\alpha_0$ is only weakly determined by $M=5$ samples. No result says how large $M$ must be for $\alpha_0$ to be identified to a given tolerance.
- **Empirically open.** Nobody has published $R_\rho$ — the per-input rank correlation between student and ensemble knowledge uncertainty — at ImageNet scale with $M$ swept over $\{2,5,10,20\}$. The experiment is runnable today for well under $10^4$ GPU-hours.
- **Methodologically blocked.** There is no ground-truth epistemic uncertainty. OOD-AUROC is a proxy that rewards any input-density score, including ones with no epistemic content, so a student can score well by learning "is this ImageNet-like?" while retaining none of the teacher's disagreement structure. The measurement names one thing and scores another.

## 6. Why It Is Hard

The core obstruction is **confounded measurement compounded by absent ground truth**. The field's headline metric (OOD AUROC) is monotone in a nuisance variable — input typicality — that correlates with, but is not, epistemic uncertainty. Two students with $R_\rho = 0.3$ and $R_\rho = 0.8$ against the same teacher can post identical AUROC. Until retention is scored against the teacher's own per-input $\mathbb{I}_M$, "diversity preserved" is unfalsifiable.

Second, **non-identifiability**: the Dirichlet NLL at $M=5$ leaves $\alpha_0$ under-determined, and $\alpha_0$ is precisely the quantity that sets $\hat{\mathbb{I}}$. The loss is nearly flat in the direction that matters.

Third, **the diversity lives where the transfer set is not**. Members agree on-manifold and diverge off it; a distillation set drawn from the training distribution carries almost no signal about the target.

## 7. Current Research (as of 2026)

- Proxy-target and reparameterized distribution distillation at large $K$, following Ryabinin/Malinin (Yandex Research, Cambridge Engineering).
- Diversity-seeking transfer-set construction: adversarial or output-diversified inputs chosen to maximize teacher disagreement (Nam et al. line, KAIST/AITRICS).
- Efficient-ensemble substitutes rather than distillation — BatchEnsemble, MIMO, and Plex-style reliability suites (Google Brain / DeepMind uncertainty group, Tran et al. 2022).
- Distilling ensembles of LLMs, where members are decoding-level or LoRA-level variants and the "diversity" target is sequence-level; retention metrics here are essentially undefined *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** ImageNet-1k. Teacher: $M=20$ independently trained ResNet-50s (seeds and data order differ), $\approx 20 \times 90$ epochs — under 3,000 A100-hours, or reuse a published checkpoint set. Cache all $20 \times 1.28\mathrm{M}$ member logit vectors once.

**Arms.** (1) Mean distillation (KL to $\bar p$) — the control, whose $\hat{\mathbb{I}} \equiv 0$ fixes the floor. (2) EnD² with proxy targets. (3) Hydra, $M$ heads — the near-ceiling, since it keeps members explicitly. (4) EnD² trained on a disagreement-weighted transfer set (importance weight $\propto \mathbb{I}_M(x)$).

**Sweep.** $M \in \{2,5,10,20\}$ with the transfer set held fixed, so retention is decoupled from teacher quality.

**Deciding number.** Spearman $R_\rho$ between $\hat{\mathbb{I}}_\phi(x)$ and the teacher's $\mathbb{I}_{20}(x)$ on 50k held-out ImageNet-val images *plus* 50k shifted images (ImageNet-C, severity 3), reported as a single pooled value. **Verdict rule:** $R_\rho \ge 0.8$ for arm (2) at $M=20$ ⇒ the diversity loss is an engineering problem, closed by better targets. $R_\rho \le 0.5$ while arm (3) exceeds $0.8$ ⇒ the loss is a representational limit of the single-head parametric student, and the theory question in §5 is the real problem. Secondary readout: does $R_\rho$ rise with $M$? If it is flat in $M$, the bottleneck is the student, not the sample count.

## 9. Key References

- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS, 2017. — arXiv:1612.01474
- **[Foundational]** Cristian Buciluă, Rich Caruana, Alexandru Niculescu-Mizil. *Model Compression.* KDD, 2006.
- **[SOTA]** Andrey Malinin, Bruno Mlodozeniec, Mark Gales. *Ensemble Distribution Distillation.* ICLR, 2020. — arXiv:1905.00076
- **[SOTA]** Max Ryabinin, Andrey Malinin, Mark Gales. *Scaling Ensemble Distribution Distillation to Many Classes with Proxy Targets.* NeurIPS, 2021.
- **[SOTA]** Giung Nam, Jongmin Yoon, Yoonho Lee, Juho Lee. *Diversity Matters When Learning From Ensembles.* NeurIPS, 2021.
- Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, Andrew Gordon Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, Steffen Udluft. *Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning.* ICML, 2018.
- Zeyuan Allen-Zhu, Yuanzhi Li. *Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning.* ICLR, 2023. — arXiv:2012.09816
- Marton Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper Snoek, Balaji Lakshminarayanan, Andrew M. Dai, Dustin Tran. *Training Independent Subnetworks for Robust Prediction.* ICLR, 2021. — arXiv:2010.06610
- **[Survey]** Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao. *Knowledge Distillation: A Survey.* International Journal of Computer Vision, 2021. — arXiv:2006.05525

## 10. Worked Example

Take $K=3$, one input $x$, and $M=4$ members that split two–two on a genuinely ambiguous image:

| member | $p_m$ |
|---|---|
| 1 | $(0.80,\ 0.15,\ 0.05)$ |
| 2 | $(0.75,\ 0.20,\ 0.05)$ |
| 3 | $(0.15,\ 0.80,\ 0.05)$ |
| 4 | $(0.20,\ 0.75,\ 0.05)$ |

Mean: $\bar p = (0.475,\ 0.475,\ 0.05)$, $\mathcal{H}[\bar p] = 0.842$ nats. Mean member entropy: each member has $\mathcal{H}\approx 0.62$ nats, so expected data uncertainty $\approx 0.62$. Knowledge uncertainty $\mathbb{I}_4 \approx 0.22$ nats — a fifth of the total.

Now compare two students, both fit perfectly on their own objective.

- **Mean-distilled student.** Reproduces $(0.475, 0.475, 0.05)$ exactly. $\hat{\mathbb{I}} = 0$. Its total entropy is right; it attributes all $0.842$ nats to aleatoric noise. It cannot tell "the label is genuinely ambiguous" from "my ensemble is split."
- **Dirichlet student.** Fitting a Dirichlet to those four simplex points by maximum likelihood must place mass on the *arc* between the two modes. A symmetric fit near $\alpha = (a, a, b)$ with $\bar\pi = \bar p$ spreads probability over the whole $\{1,2\}$ ridge — including the midpoint $(0.475, 0.475, 0.05)$, which no member ever produced. It can be tuned to reproduce $\hat{\mathbb{I}} = 0.22$, but so can a Dirichlet fit to four members *clustered* around the midpoint with small isotropic scatter — an ensemble with the same $\mathbb{I}$ and completely different structure.

That last sentence is the obstruction. $\mathbb{I}$ is one scalar summary of a $4\times 3$ table; matching it does not mean matching the disagreement. Two ensembles with identical $(\mathcal{H}[\bar p], \mathbb{I})$ — one bimodal, one unimodal — are indistinguishable to every metric in current use, and a downstream OOD-AUROC number will not separate them either. Any honest claim of diversity retention has to be scored per input against $\mathbb{I}_M$ at minimum, and ideally against a two-sample test on the member cloud itself.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*