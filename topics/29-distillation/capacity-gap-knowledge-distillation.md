---
id: 29-distillation/capacity-gap-knowledge-distillation
title: "Capacity Gap in Knowledge Distillation"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Capacity Gap in Knowledge Distillation

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/capacity-gap-knowledge-distillation` · **Status:** open

## 1. Problem Statement

Knowledge distillation trains a student $f_S$ to match a teacher $f_T$. Folk practice says a better teacher makes a better student. It does not hold monotonically: past some teacher size, student test accuracy *falls* as the teacher grows, even though the teacher itself keeps improving. This non-monotonicity is the **capacity gap**.

Three variants, routinely conflated:

- **Measurement.** Is the effect a property of teacher–student capacity ratio at all, or an artifact of a fixed optimization budget (temperature, epochs, augmentation) that was tuned for one teacher and reused for another? Define a protocol under which "capacity gap" is identifiable.
- **Method.** Given a fixed student architecture and compute budget, choose the teacher (or a teacher sequence, or a loss) maximizing student generalization. Solving it means an algorithm that never does worse than distilling from the *best small* teacher, and beats it when a large teacher exists.
- **Theory.** Prove or refute: for a natural function class and student capacity $c$, student risk as a function of teacher capacity $C$ is non-monotone, with the turning point a function of $c$ and sample size $n$ — not of optimization failure.

Solved = a predictive rule mapping $(c, C, n, \text{data})$ to the sign of the transfer benefit, validated out of sample.

## 2. Formal Setting

Data $(x,y) \sim \mathcal{D}$ over $\mathcal{X} \times [K]$, sample $S = \{(x_i,y_i)\}_{i=1}^n$. Teacher $p_T(\cdot\mid x) = \mathrm{softmax}(z_T(x)/\tau)$, student $p_S$ likewise. The standard objective (Hinton et al., 2015):

$$\mathcal{L}(\theta_S) = (1-\alpha)\,\mathrm{CE}\!\left(y, p_S^{(1)}\right) + \alpha\,\tau^2\,\mathrm{KL}\!\left(p_T^{(\tau)} \,\|\, p_S^{(\tau)}\right).$$

Measured quantities:

- **Capacity** $c$: not parameter count alone. Measure as *matched-budget* accuracy — top-1 of that architecture trained on $S$ with the same tuned recipe. Parameter ratio $C/c$ is a proxy that fails across families (ViT vs. ResNet).
- **Gap** $\Delta(C) = \mathrm{Acc}(f_S \mid \text{teacher } C) - \mathrm{Acc}(f_S \mid \text{no teacher})$, both arms tuned independently over $(\alpha,\tau,\text{lr},\text{epochs})$ by the same search budget. Non-monotonicity is the claim $\exists\, C_1 < C_2$ with $\Delta(C_1) > \Delta(C_2)$ while teacher accuracy rises.
- **Fidelity** (Stanton et al., 2021): top-1 agreement $\Pr[\arg\max p_S = \arg\max p_T]$ on train and test, and mean $\mathrm{KL}(p_T\|p_S)$ on train points. Fidelity is the *optimization* diagnostic; accuracy is the *generalization* one. They dissociate.
- **Supervision complexity** (Harutyunyan et al., 2023): $\|p_T\|_{K^{-1}}$-type quantity measuring how well the teacher's soft labels align with the student's kernel — the operative "learnability by this student" measure.

Assumptions and their violations: (i) the student *can* represent the teacher's function — false whenever $C \gg c$; (ii) distillation loss is optimized to convergence — false, Stanton et al. show train-set fidelity gaps remain large even with heavy optimization; (iii) temperature is a nuisance parameter transferable across teachers — false in practice, and this is the main confounder; (iv) teacher softmax entropy is comparable across scales — false, larger teachers are typically more confident (lower-entropy, more nearly one-hot), so the "dark knowledge" signal shrinks as $C$ grows.

## 3. State of the Art

**Established (ablated, reproduced independently).**
- Cho & Hariharan (ICCV 2019) showed the effect exists and that *early-stopping the teacher* recovers much of the loss — an ablation directly against teacher quality, not just architecture.
- Stanton et al. (NeurIPS 2021) established that the failure is largely *optimization*, not representational: students with enough capacity to match the teacher still fail to, and better fidelity does not imply better accuracy.
- Beyer et al. (CVPR 2022) established that treating distillation as *function matching* — consistent teacher/student input views, aggressive mixup, very long schedules — removes much of the apparent gap. This is the strongest empirical rebuttal to the capacity-gap-as-fundamental view.

**Claimed but under-ablated.**
- Teacher-assistant chains (TAKD, Mirzadeh et al., AAAI 2020) report gains, but the control arm is usually a short-schedule direct distillation, not a compute-matched one; the assistant chain also multiplies the training budget.
- Feature/relational methods (FitNets 2015; RKD, CVPR 2019; CRD, ICLR 2020) claim robustness to large gaps; Tian et al.'s own ablation shows much of the CRD margin survives, but the *specific* capacity-gap claim rests on benchmark tables, not a controlled sweep of $C$ at fixed everything else.
- **Benchmark-number-only:** most reported "large teacher hurts" results are single CIFAR-100 or ImageNet cells with one seed and one $(\alpha,\tau)$. The dependence on seed and on temperature search budget is rarely reported.

**Theory SOTA.** Phuong & Lampert (ICML 2019) give generalization bounds for linear students under distillation; Menon et al. (ICML 2021) frame distillation as variance reduction in the risk estimate — predicting benefit scales with how well $p_T$ approximates the Bayes class-probability, which *decays* if a big teacher overfits toward one-hot. Harutyunyan et al. (ICLR 2023) give the sharpest mechanism: a bound in which the teacher must be simple *relative to the student's kernel*, giving a principled reason a stronger teacher can be a worse target. None of these yields a computable turning point $C^\ast(c,n)$.

## 4. What Is Known

- **CIFAR-100, ~1–40M params:** ResNet/WRN students distilled from progressively deeper teachers show the student peaking at an intermediate teacher; TAKD reports assistant chains recovering roughly 1–2 top-1 points versus direct distillation from the largest teacher (Mirzadeh et al., AAAI 2020).
- **ImageNet-1k, ResNet-18 student:** the ResNet-34 teacher is a stronger *distillation* teacher than substantially larger ones under fixed short recipes, despite lower teacher accuracy (Cho & Hariharan, ICCV 2019). Same paper: early-stopped (lower-accuracy) teachers can beat fully trained ones.
- **The gap is schedule-dependent.** Beyer et al. (CVPR 2022) distill BiT-M-R152x2 into ResNet-50 and reach **82.8%** ImageNet top-1 — with schedules of order $10^3$–$10^4$ epochs. At 300 epochs the same recipe is materially worse. So "large teacher hurts" and "large teacher helps enormously" are both true at different budgets: the ordering is not budget-invariant.
- **Fidelity ≠ accuracy.** Stanton et al. (NeurIPS 2021): student–teacher top-1 agreement stays well below 100% even in self-distillation where the student *is* the teacher's architecture; increasing agreement past a point does not track test accuracy.
- **Scaling side.** Busbridge et al. (arXiv:2502.08606, 2025) fit a distillation scaling law for LLMs: student loss as a function of student size, teacher loss, and distillation tokens, with a regime where a stronger teacher yields *worse* student loss at fixed student size — the LLM analogue of the capacity gap, fit over a large sweep. This is the closest thing to a quantitative turning-point model.

## 5. What Is Not Known

- **Theoretically open.** No theorem gives $C^\ast(c,n)$ — the teacher capacity maximizing student risk reduction — for any non-toy class beyond linear/NTK settings. Whether non-monotonicity survives exact optimization of the distillation objective is unproven either way.
- **Empirically open.** Nobody has run the decisive sweep: teacher capacity varied over $\ge 5$ scales with *per-cell* independent tuning of $(\alpha,\tau,\text{epochs},\text{augmentation})$ under a compute-matched budget, at ImageNet scale or LLM scale, with $\ge 3$ seeds. Every existing sweep fixes at least one of these. Cost is the only blocker.
- **Methodologically blocked.** "Capacity" has no measurement that transfers across architecture families. Parameter count, FLOPs, and matched-budget accuracy disagree, so the *x-axis of the phenomenon* is undefined. Until it is, "gap too large" is not falsifiable.

## 6. Why It Is Hard

**Confounded measurement, with a compute wall behind it.** The reported effect is the composition of at least four things: teacher confidence rising with scale (soft-label entropy falls), an optimization objective that is harder to fit as the target sharpens, hyperparameters tuned once and reused, and finite schedule length. Beyer et al. show that fixing the fourth alone flips the sign of the conclusion in some cells. Disentangling requires per-cell tuning, and a properly tuned ImageNet distillation sweep at 5 teacher scales × 3 seeds × a $(\alpha,\tau)$ grid × long schedules is $10^2$–$10^3$ ImageNet trainings. That is why the definitive experiment has not been run — not because it is conceptually subtle.

Second obstruction: **non-identifiability of capacity**. A ViT-B and a ResNet-152 with similar accuracy behave differently as teachers. Without a capacity metric that predicts teacher-role behavior, the phenomenon has no well-posed independent variable.

## 7. Current Research (as of 2026)

- **Scaling-law framing.** Fitting student loss as a joint function of teacher loss, student size, and distillation tokens (Apple, Busbridge et al. 2025) — the direction most likely to yield the predictive rule. Extensions to instruction-tuned and reasoning-trace distillation are active *(frontier — verify)*.
- **On-policy / sequence-level KD for LLMs.** GKD (Agarwal et al., ICLR 2024) trains on student-sampled sequences, which removes part of the distribution-mismatch component of the gap. Whether it removes the *capacity* component is untested.
- **Supervision-complexity-guided teacher selection** and curriculum over teacher checkpoints (Google Research line following Harutyunyan et al., ICLR 2023).
- **Reasoning distillation.** Widespread reports that small students trained on large-model chain-of-thought traces gain more than logit distillation predicts, with a floor below which students fail to use traces at all *(frontier — verify)*; the floor is the capacity gap in a new modality and is characterized only anecdotally.

## 8. Concrete Next Experiment

**Question:** does non-monotonicity in $\Delta(C)$ survive per-cell hyperparameter tuning and a long schedule?

- **Scale.** ImageNet-1k. Student fixed: ResNet-50 (25.6M). Teachers: 5 scales spanning ~2× to ~30× student compute within one family (BiT-R50x1, R50x3, R101x1, R101x3, R152x2), plus one cross-family control (ViT-L/16) to test capacity identifiability.
- **Per cell.** Independent search over $\alpha \in \{0.5,1.0\}$, $\tau \in \{1,2,4,8\}$, schedule $\in \{300, 3000\}$ epochs, consistent-view function-matching recipe (Beyer et al.). 3 seeds at the selected point.
- **Control arm.** ResNet-50 trained from labels only, tuned with the *same* search budget and the same 3000-epoch schedule. Second control: distillation from the *smallest* teacher, identically tuned.
- **Deciding number.** $\Delta(C_{\max}) - \max_{C} \Delta(C)$ in ImageNet top-1 points, with a 95% CI over seeds. If this is $\le -0.5$ pt with a CI excluding 0 at 3000 epochs, the capacity gap is a real capacity phenomenon, not a schedule artifact. If it is $\ge -0.2$ pt at 3000 epochs while clearly negative at 300 epochs, the gap is a budget artifact and the field should stop modelling it as capacity.

Cost estimate: ~90 ResNet-50-equivalent runs at 3000 epochs — roughly $10^5$ TPU-v3-core-hours. This is the cheapest version that is not confounded.

## 9. Key References

- **[Foundational]** Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Romero, Ballas, Kahou, Chassang, Gatta, Bengio. *FitNets: Hints for Thin Deep Nets.* ICLR, 2015. — arXiv:1412.6550
- **[Foundational]** Cho, Hariharan. *On the Efficacy of Knowledge Distillation.* ICCV, 2019. — arXiv:1910.01348
- **[Foundational]** Mirzadeh, Farajtabar, Li, Levine, Matsukawa, Ghasemzadeh. *Improved Knowledge Distillation via Teacher Assistant.* AAAI, 2020. — arXiv:1902.03393
- **[SOTA]** Beyer, Zhai, Royer, Markeeva, Anil, Kolesnikov. *Knowledge Distillation: A Good Teacher Is Patient and Consistent.* CVPR, 2022. — arXiv:2106.05237
- **[SOTA]** Stanton, Izmailov, Kirichenko, Alemi, Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- **[SOTA]** Busbridge, Shidani, Weers, Ramapuram, Littwin, Webb. *Distillation Scaling Laws.* 2025. — arXiv:2502.08606
- **[Theory]** Phuong, Lampert. *Towards Understanding Knowledge Distillation.* ICML, 2019.
- **[Theory]** Menon, Rawat, Reddi, Kim, Kumar. *A Statistical Perspective on Distillation.* ICML, 2021.
- **[Theory]** Harutyunyan, Rawat, Menon, Kim, Kumar. *Supervision Complexity and Its Role in Knowledge Distillation.* ICLR, 2023. — arXiv:2301.12245
- **[Method]** Tian, Krishnan, Isola. *Contrastive Representation Distillation.* ICLR, 2020. — arXiv:1910.10699
- **[Method]** Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist, Bachem. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD).* ICLR, 2024. — arXiv:2306.13649
- **[Survey]** Gou, Yu, Maybank, Tao. *Knowledge Distillation: A Survey.* International Journal of Computer Vision, 2021. — arXiv:2006.05525

## 10. Worked Example

Take a CIFAR-100 student, ResNet-8 (~0.08M params), and two teachers: ResNet-20 (~0.28M, ~69% top-1) and ResNet-110 (~1.7M, ~74% top-1). Standard finding: the ResNet-110-taught student lands *below* the ResNet-20-taught one, often by ~1 point, despite the teacher being 5 points better.

Now look at the soft labels, which is where the obstruction becomes visible. At $\tau = 1$ the mean max-probability on training points is roughly 0.95 for ResNet-20 and roughly 0.99 for ResNet-110 (a well-fit deeper net memorizes the train set harder). The KL term's informative content — the non-argmax mass — is therefore about $0.05$ vs. $0.01$: a **5× reduction in dark-knowledge signal** purely from teacher confidence.

Compensate by raising temperature. At $\tau = 4$, softening logits $z/\tau$ redistributes that mass; the effective entropy of the ResNet-110 target rises to roughly what ResNet-20 delivered at $\tau=1$. Papers that report the ResNet-110 loss almost always used the $\tau$ tuned for the mid-size teacher.

The obstruction: the experiment as normally run cannot distinguish

1. *the student lacks capacity to represent the deep teacher's function*, from
2. *the target was mis-scaled by a nuisance parameter*, from
3. *the student needed more steps to fit a sharper target*.

All three predict the same single number — one lower accuracy cell. The gradient of the KL term scales as $\tau^{-1}$ times the probability difference, so a 5× confidence change is a first-order change in the learning signal, comparable in magnitude to the reported 1-point effect. Only per-cell tuning over $(\alpha, \tau, \text{epochs})$ separates them, and that is exactly what Section 8 costs $10^5$ core-hours to do at a scale anyone will believe.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*