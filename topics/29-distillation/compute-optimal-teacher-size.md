---
id: 29-distillation/compute-optimal-teacher-size
title: "Compute-Optimal Teacher Size"
topic: 29-distillation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Teacher Size

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/compute-optimal-teacher-size` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed student architecture and size, and a fixed total compute budget that must cover *both* producing a teacher and training the student from it, what teacher size minimizes final student loss?

Three variants, with different difficulty:

- **Measurement variant.** For a fixed student and fixed total FLOPs, is student loss unimodal in teacher size $N_T$, and where is the minimum? Runnable today; the obstruction is cost and accounting discipline.
- **Method variant.** Given a budget, produce the best student — free to choose teacher size, teacher token count, teacher checkpoint (early-stopped or converged), distillation loss, and how many tokens carry teacher supervision. Optimization over a large discrete-continuous space.
- **Theory variant.** Prove that student loss is non-monotone in teacher capacity, and derive the optimal teacher size from properties of the data distribution and the student's hypothesis class. Open; no proof either way for realistic models.

Solving it means: a predictive rule $N_T^\star = f(N_S, C, \text{amortization } K)$ that transfers across scales and is validated by holding out a scale it was not fit on.

## 2. Formal Setting

Let $N_S, N_T$ be student and teacher non-embedding parameter counts, $D_T$ teacher training tokens, $D_S$ distillation tokens. Measured quantities:

- **Teacher pretraining cost:** $C_T = 6 N_T D_T$ FLOPs (forward+backward, the standard Kaplan approximation).
- **Teacher inference cost** for producing logits over the distillation set: $C_I = 2 N_T D_S$, if logits are computed online; zero marginal cost if a cached teacher already exists and its logits are reused.
- **Student cost:** $C_S = 6 N_S D_S$.
- **Total:** $C = \frac{C_T}{K} + C_I + C_S$, where $K \ge 1$ is the number of distinct students amortizing one teacher. $K=1$ is the "build a teacher for this student" regime; $K \to \infty$ is "the teacher already exists".

Student objective, measured as token-level cross-entropy on a held-out corpus:
$$L_S = \mathbb{E}_{x}\big[-\log p_S(x_t \mid x_{<t})\big].$$

Training loss mixes ground truth and teacher:
$$\mathcal{L} = (1-\lambda)\,\mathrm{CE}(y, p_S) + \lambda\, \tau^2\, \mathrm{KL}\!\left(p_T^{(\tau)} \,\|\, p_S^{(\tau)}\right),$$
with temperature $\tau$ and mixing weight $\lambda$. The decision predicate is
$$N_T^\star(N_S, C, K) = \arg\min_{N_T} \; L_S \quad \text{s.t.} \quad \tfrac{6N_TD_T}{K} + 2N_TD_S + 6N_SD_S \le C .$$

**Assumptions, and which are violated.**

1. *FLOP proxies are faithful.* $6ND$ ignores attention quadratic terms and memory-bound logit transfer; at long context the error exceeds 10%. Violated mildly.
2. *Teacher quality is summarized by $N_T$.* False in practice — data mixture, tokenizer, and instruction tuning move teacher loss more than a $2\times$ parameter change. Any experiment must hold the teacher's data pipeline fixed.
3. *One scalar loss orders students.* Violated: distillation improves held-out cross-entropy while degrading calibration and out-of-distribution agreement (Stanton et al., 2021).
4. *Teacher logits are available.* Violated for API teachers, which return top-$k$ logprobs or samples only; this changes the estimator, not just the constant.
5. *$D_T$ set by Chinchilla.* Modern teachers are heavily over-trained relative to Chinchilla, so $C_T$ in deployed pipelines is far above $6N_T \cdot 20 N_T$.

## 3. State of the Art

**Established.**
- Bigger teachers are not monotonically better students. Cho & Hariharan (*On the Efficacy of Knowledge Distillation*, ICCV 2019) show ImageNet/CIFAR student accuracy falling as teacher capacity grows past a point, and that early-stopping the teacher recovers much of the loss. Reproduced repeatedly.
- Mirzadeh et al. (*Improved Knowledge Distillation via Teacher Assistant*, AAAI 2020) show an intermediate-size TA network beats direct distillation from the largest teacher — the "capacity gap".
- Busbridge et al. (*Distillation Scaling Laws*, 2025, arXiv:2502.08606) fit a parametric law for student cross-entropy as a joint function of teacher loss, student size, and distillation tokens, over students and teachers spanning roughly $10^8$–$10^{10}$ parameters and up to hundreds of billions of tokens. It is the first work to state the compute-allocation question in the form of §2 and answer it with a fitted surface rather than a point ablation.

**Claimed but unablated.**
- That the optimum depends on teacher *loss* rather than teacher *size* — i.e. that $N_T$ enters only through $L_T$. Plausible and convenient, but not tested against teachers matched in loss and mismatched in size (e.g. an over-trained small model versus a Chinchilla-optimal large one).
- Production reports (Gemma 2, arXiv:2408.00118; Minitron, Muralidharan et al., NeurIPS 2024) state that distillation from a larger sibling beat from-scratch training at fixed student size. These are benchmark numbers under $K \gg 1$ with the teacher's cost excluded; they do not bear on $N_T^\star$ at $K=1$.

**Theory SOTA** is weaker than empirical. Menon et al. (*A Statistical Perspective on Distillation*, ICML 2021) show the teacher helps by reducing variance of the label estimate, with benefit governed by how close $p_T$ is to the Bayes-optimal conditional — which predicts monotone improvement, not an interior optimum. Harutyunyan et al. (*Supervision Complexity and its Role in Knowledge Distillation*, ICLR 2023) give the complementary half: a bound trading off teacher accuracy against the complexity of the teacher's targets relative to the student's kernel. Neither yields a computable $N_T^\star$.

## 4. What Is Known

- **Interior optimum exists in vision.** Cho & Hariharan report CIFAR-100 WRN students degrading by 1–2 accuracy points when the teacher is scaled up past a mid-size point, at student scales of $10^6$–$10^7$ parameters.
- **The gap is fixable by intervention, not only by teacher choice.** Beyer et al. (*Knowledge distillation: a good teacher is patient and consistent*, CVPR 2022) reach 82.8% ImageNet top-1 with a ResNet-50 student by using consistent input views and very long schedules (up to $10^6$ epochs-equivalent of patient training), showing much of the apparent capacity gap is under-training of the distillation objective.
- **Distillation is not free at $K=1$.** Busbridge et al. report that distillation beats supervised pretraining at equal *total* compute only below a student-size/token threshold, and that once the teacher's training cost is charged once and only once, supervised training wins for sufficiently large distillation budgets. This is the single most decision-relevant established number in the area.
- **Students do not match teachers even in-distribution.** Stanton et al. (*Does Knowledge Distillation Really Work?*, NeurIPS 2021) measure top-1 teacher-student agreement well below what the student's own accuracy allows — optimization, not capacity, is the binding constraint at CIFAR/ImageNet scale.
- **On-policy supervision changes the tradeoff.** GKD (Agarwal et al., ICLR 2024) and MiniLLM (Gu et al., ICLR 2024) both improve over forward-KL offline distillation at 0.3B–7B student scale, but neither varies $N_T$ systematically.

## 5. What Is Not Known

- **Empirically open.** The full surface $L_S(N_T \mid N_S, C, K)$ for LLMs at $N_S \ge 7$B with modern data mixtures. One fitted law exists; it has not been independently reproduced, and no group has tested its extrapolation to a scale outside the fit. The experiment is runnable — it costs roughly $10^{22}$–$10^{23}$ FLOPs.
- **Empirically open.** Whether $N_T$ enters only through $L_T$ (the loss-sufficiency hypothesis). Decidable with loss-matched, size-mismatched teachers.
- **Theoretically open.** No proof that student loss must be non-monotone in teacher capacity for any nontrivial function class. The capacity-gap phenomenon has no theorem; the two available frameworks (variance reduction, supervision complexity) each explain one direction.
- **Methodologically blocked.** "Teacher size" is not a well-defined axis once teachers differ in training tokens, data mixture, and post-training. Until the community fixes a canonical teacher family varying only in $N_T$, cross-paper comparison of $N_T^\star$ is uninterpretable.

## 6. Why It Is Hard

The specific obstruction is **confounded compute accounting**, compounded by cost.

Every published comparison chooses a value of $K$ implicitly and never states it. A lab distilling a 9B model from an existing 27B model correctly excludes the 27B training cost ($K$ effectively large); a reader planning a single student cannot use that result, because at $K=1$ the teacher dominates the budget by an order of magnitude (see §10). The same experiment supports opposite conclusions under the two accountings, and the accounting is rarely reported.

Second: the sweep is genuinely expensive. Resolving $N_T^\star$ to within a factor of 2 requires at least four teachers per student size and three student sizes to test transfer — twelve teacher pretraining runs at frontier-ish scale. This is why one paper has the surface and nobody has checked it.

Third: **non-identifiability** between capacity gap and under-optimization. Beyer et al. show extending the distillation schedule removes much of the apparent gap. So an observed interior optimum at fixed $D_S$ may be an artifact of the student not being trained long enough to fit a hard teacher, not a property of teacher capacity at all.

## 7. Current Research (as of 2026)

- **Scaling-law fitting for distillation.** Apple's distillation scaling law (Busbridge et al.) is the anchor; follow-up work extending it to on-policy and sequence-level objectives is active *(frontier — verify)*.
- **Prune-then-distill.** NVIDIA's Minitron line derives the student from the teacher by structured pruning, which collapses the teacher-choice question into a pruning-ratio question and sidesteps $K$ entirely.
- **Teacher ensembles and teacher assistants at LLM scale.** Reported to help; no compute-matched ablation published *(frontier — verify)*.
- **Inference-aware allocation.** Sardana et al. (*Beyond Chinchilla-Optimal*, ICML 2024) recast scaling laws to include serving cost; the natural composition with distillation — charging teacher, student, and lifetime serving in one budget — is not yet published *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question decided:** does $N_T^\star$ at $K=1$ sit at $N_T \approx N_S$ or at $N_T \gg N_S$?

**Scale.** Student fixed at $N_S = 1$B non-embedding parameters, $D_S = 100$B tokens ($C_S = 6\times10^{20}$ FLOPs). Four teachers from one family, identical data and tokenizer, Chinchilla-trained at $D_T = 20N_T$: $N_T \in \{1, 3, 8, 27\}$B. Distillation: $\lambda = 1$, $\tau = 1$, full-vocabulary logits, identical student schedule across arms.

**Control arms.** (a) Student trained on ground-truth tokens only for $D_S = 100$B — no teacher. (b) *Compute-matched* control: student trained on ground truth for the total FLOPs each distillation arm consumed including teacher cost, i.e. $D_S^{\text{ctrl}} = C_{\text{arm}}/(6N_S)$, capped by data availability. Arm (b) is the one that matters; arm (a) is the one everyone reports.

**Deciding number.** Held-out token cross-entropy $L_S$ (nats) on a fixed 100M-token validation set, plotted against $N_T$ at matched total FLOPs. The decision: the sign of $L_S(27\text{B teacher}) - L_S(3\text{B teacher})$, and whether $\min_{N_T} L_S$ is below control arm (b). A gap of $>0.01$ nats is resolvable above seed noise (measure seed variance with 3 seeds on the 3B arm; typical is $\sim 0.003$ nats at this scale).

**Secondary, cheap, high-value:** add one loss-matched pair — an over-trained 3B teacher and a Chinchilla 8B teacher with equal $L_T$ — to test loss-sufficiency. If their students differ by more than seed noise, teacher loss is not a sufficient statistic and every fitted law parameterized by $L_T$ needs a size term.

## 9. Key References

- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NIPS 2014 Deep Learning Workshop. — arXiv:1503.02531
- **[Foundational]** Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, Hassan Ghasemzadeh. *Improved Knowledge Distillation via Teacher Assistant.* AAAI 2020. — arXiv:1902.03393
- **[Foundational]** Jang Hyun Cho, Bharath Hariharan. *On the Efficacy of Knowledge Distillation.* ICCV 2019. — arXiv:1910.01348
- **[SOTA]** Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, Russ Webb. *Distillation Scaling Laws.* 2025. — arXiv:2502.08606
- **[SOTA]** Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov. *Knowledge distillation: A good teacher is patient and consistent.* CVPR 2022. — arXiv:2106.05237
- **[Theory]** Aditya Krishna Menon, Ankit Singh Rawat, Sashank Reddi, Seungyeon Kim, Sanjiv Kumar. *A Statistical Perspective on Distillation.* ICML 2021.
- **[Theory]** Hrayr Harutyunyan, Ankit Singh Rawat, Aditya Krishna Menon, Seungyeon Kim, Sanjiv Kumar. *Supervision Complexity and its Role in Knowledge Distillation.* ICLR 2023. — arXiv:2301.12245
- **[Empirical]** Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, Andrew Gordon Wilson. *Does Knowledge Distillation Really Work?* NeurIPS 2021. — arXiv:2106.05945
- **[Context]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Context]** Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML 2024. — arXiv:2401.00448
- **[Systems]** Saurav Muralidharan et al. *Compact Language Models via Pruning and Knowledge Distillation.* NeurIPS 2024. — arXiv:2407.14679
- **[Systems]** Gemma Team. *Gemma 2: Improving Open Language Models at a Practical Size.* 2024. — arXiv:2408.00118

## 10. Worked Example

Take the §8 setup, $N_S = 1$B, $D_S = 100$B tokens, and price each arm.

| Arm | $C_T = 6N_TD_T$ | $C_I = 2N_TD_S$ | $C_S$ | Total at $K=1$ | Total at $K=32$ |
|---|---|---|---|---|---|
| 1B teacher | $1.2\times10^{20}$ | $2.0\times10^{20}$ | $6.0\times10^{20}$ | $9.2\times10^{20}$ | $8.0\times10^{20}$ |
| 3B teacher | $1.1\times10^{21}$ | $6.0\times10^{20}$ | $6.0\times10^{20}$ | $2.3\times10^{21}$ | $1.2\times10^{21}$ |
| 8B teacher | $7.7\times10^{21}$ | $1.6\times10^{21}$ | $6.0\times10^{20}$ | $9.9\times10^{21}$ | $2.4\times10^{21}$ |
| 27B teacher | $8.7\times10^{22}$ | $5.4\times10^{21}$ | $6.0\times10^{20}$ | $9.3\times10^{22}$ | $8.7\times10^{21}$ |

The student's own training is $6\times10^{20}$ FLOPs in every row. At $K=1$ the 27B arm costs **155× the student**; the teacher is the experiment and the student is a rounding error.

Now apply the standard reporting convention — quote only $C_S$, because "the teacher already existed". All four arms then cost the same $6\times10^{20}$, and the ranking is by student loss alone. Suppose the measured losses are $2.41, 2.36, 2.34, 2.35$ nats for the 1B/3B/8B/27B teachers. Under the convention, the 8B teacher wins and the headline is "distill from an $8\times$ larger teacher".

Charge the teacher at $K=1$ and compare against control arm (b): the 8B arm's $9.9\times10^{21}$ FLOPs would buy the student $D_S^{\text{ctrl}} = 9.9\times10^{21}/(6\times10^9) \approx 1.65$T ground-truth tokens instead of 100B. A 1B model on 1.65T tokens sits near $2.1$–$2.2$ nats on the same validation set by Chinchilla-family extrapolation — comfortably better than $2.34$. The distillation arm loses outright.

The obstruction is now visible: the *same* four measurements say "use an 8B teacher" and "use no teacher at all", and which one is correct depends entirely on $K$ — a number that appears in no paper's results table. Nothing in the experiment is wrong; the accounting is unstated, and $N_T^\star$ is undefined until it is stated.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*