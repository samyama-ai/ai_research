---
id: 29-distillation/distillation-scaling-laws
title: "Scaling Laws for Distillation"
topic: 29-distillation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws for Distillation

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/distillation-scaling-laws` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed total compute budget $C$, predict — before training — whether spending part of it on a teacher and distilling into a student beats spending all of it on supervised pretraining of the same student, and if so, how to split the budget.

Three variants, different difficulty:

- **Measurement.** Fit a function $L_S = f(N_S, D_S, L_T)$ mapping student parameters, distillation tokens, and teacher quality to student loss, accurate enough to extrapolate one order of magnitude in compute. Runnable today; expensive.
- **Method.** Choose the teacher size, teacher token budget, distillation loss, and token allocation that minimize student loss at fixed $C$. Depends on the measurement being right.
- **Theory.** Explain *why* the exponents take the values they do — why the teacher's soft targets act as a variance-reduced estimate of the data distribution, and why that advantage is capped. No derivation exists that predicts the fitted exponents from first principles.

A solution to the measurement variant is a law with held-out extrapolation error below the run-to-run seed noise (roughly $\pm 0.005$ nats at 1B scale) on models an order of magnitude larger than the fitting grid.

## 2. Formal Setting

Teacher $p_T$ and student $q_\theta$ over vocabulary $\mathcal{V}$. Student has $N_S$ non-embedding parameters, trained on $D_S$ distillation tokens; the teacher has $N_T$ parameters trained on $D_T$ tokens.

Measured quantities:

- **Student loss** $L_S$: next-token cross-entropy against ground-truth tokens on a held-out corpus, in nats/token. Not the training objective — this is the reported number.
- **Teacher cross-entropy** $L_T$: the teacher's own held-out cross-entropy on the same corpus. This, not $N_T$, is the covariate that carries teacher quality.
- **Distillation objective**, mixing weight $\lambda \in [0,1]$, temperature $\tau$:

$$\mathcal{L}(\theta) = (1-\lambda)\,\mathbb{E}\big[-\log q_\theta(y_t \mid x_{<t})\big] + \lambda\,\tau^2\,\mathbb{E}\big[\mathrm{KL}\big(p_T^{(\tau)}(\cdot \mid x_{<t})\,\|\,q_\theta^{(\tau)}(\cdot \mid x_{<t})\big)\big]$$

- **Compute.** Supervised baseline $C_{\text{sup}} \approx 6 N_S D_S$. Distillation adds teacher training $6 N_T D_T$ (amortized over $k$ students) plus teacher inference $\approx 2 N_T D_S$ per student:

$$C_{\text{distill}} \approx 6 N_S D_S + 2 N_T D_S + \tfrac{1}{k}\,6 N_T D_T$$

- **Fitted law**, the structure reported by Busbridge et al. (2025) in the notation used here:

$$L_S(N_S, D_S, L_T) = L_T + \left(\frac{1}{L_T^{c_0}}\right)\!\left(1 + \left(\frac{L_T}{\tilde{L}_S\, d_1}\right)^{1/f_1}\right)^{-c_1 f_1}\!\left(\frac{A}{N_S^{\alpha}} + \frac{B}{D_S^{\beta}}\right)$$

with $\tilde{L}_S$ the Chinchilla-style supervised loss the student would reach at $(N_S, D_S)$. The point of the broken-power-law factor is a regime switch, not the algebra.

**Assumptions, and which are violated:**
- *Teacher quality is a scalar.* Violated: two teachers with equal $L_T$ can differ in calibration and top-$k$ entropy, which changes what transfers.
- *Amortization $k$ is known in advance.* Violated: teacher reuse count is a deployment decision made after training.
- *Fixed $\lambda, \tau$ across scale.* Violated: the optimal $\lambda$ drifts toward pure distillation ($\lambda \to 1$) as $D_S$ grows in some reported sweeps.
- *Loss is the target.* Violated where it matters most: downstream task accuracy and reasoning benchmarks are not monotone in $L_S$ across different data mixtures.
- *Same distribution for teacher and student training data.* Violated in practice — teachers are usually post-trained; students often see different mixtures.

## 3. State of the Art

**Empirical SOTA.** *Distillation Scaling Laws*, Busbridge, Shidani, Weers, Ramapuram, Littwin, Webb (Apple, ICML 2025, arXiv:2502.08606) is the only systematic law. Teachers and students spanning roughly 143M–12.6B parameters and up to hundreds of billions of tokens, with an IsoFLOP-style grid crossed over teacher/student pairs. **Established** by their grid: the capacity gap is a function of $L_T$ relative to the student's achievable loss, not of $N_T$; distillation beats supervised training only below a student-size-dependent compute threshold; above it, supervised wins. **Claimed but unablated** outside their setup: that the fitted exponents transfer to other architectures, tokenizers, or data mixtures — the law is fitted on one family.

**Systems SOTA.** Gemma 2 (Google DeepMind, 2024) trained the 9B and 2.6B models with token-level distillation from a larger teacher, reporting that this let them train well past the compute-optimal token count for those sizes. Llama 3.2 1B/3B (Meta, 2024) used logit distillation from Llama 3.1 8B/70B. Both are **benchmark numbers only** — neither ships a supervised control arm at matched compute, so neither establishes the size of the distillation gain.

**Theory SOTA.** Phuong & Lampert (*Towards Understanding Knowledge Distillation*, ICML 2019) prove a transfer-risk bound for distillation of linear/deep-linear students showing $O(1/n)$-style fast rates from soft labels. Menon et al. (*A Statistical Perspective on Distillation*, ICML 2021) show the teacher's soft labels act as a lower-variance estimate of the Bayes class-probability, giving a bias–variance decomposition of the gain. Neither predicts an exponent.

## 4. What Is Known

- **Soft labels reduce label variance.** Menon et al. (ICML 2021): distillation risk decomposes into teacher bias plus a variance term scaled by teacher fidelity to $p^*(y|x)$; the gain is largest when data is scarce relative to model capacity. Measured on CIFAR-scale and ImageNet classifiers.
- **Bigger teachers can hurt.** Cho & Hariharan (*On the Efficacy of Knowledge Distillation*, ICCV 2019): on CIFAR-100/ImageNet, distilling a WRN-28-4 student from progressively larger teachers is non-monotone — accuracy peaks then falls. Mirzadeh et al. (AAAI 2020) patch this with an intermediate teacher assistant.
- **The capacity gap is a loss gap.** Busbridge et al. (2025) reproduce the non-monotonicity and show it collapses onto $L_T$: a stronger teacher helps until $L_T$ falls far enough below the student's attainable loss, after which the student loses.
- **Distillation is compute-bounded, not free.** Same paper: at large $D_S$ the distilled student's loss curve flattens above the supervised curve. Distillation is preferable only when the teacher already exists or serves $k \gtrsim$ several students.
- **Patience beats architecture.** Beyer et al. (*Knowledge Distillation: A Good Teacher Is Patient and Consistent*, CVPR 2022): consistent input views plus ~9600 epochs takes a ResNet-50 to 82.8% ImageNet top-1 — the schedule length, not the loss, carried the result.
- **Fidelity ≠ generalization.** Stanton et al. (*Does Knowledge Distillation Really Work?*, NeurIPS 2021): students improve in accuracy while top-1 agreement with the teacher stays well below 100% on CIFAR-100/ImageNet; the optimization does not find the teacher-matching solution even when the student has the capacity.
- **Sequence-level KD works for generation.** Kim & Rush (EMNLP 2016) on WMT; DistilBERT (Sanh et al., 2019) retains ~97% of BERT-base GLUE at 40% fewer parameters.

## 5. What Is Not Known

- **Empirically open.** Whether the Busbridge law extrapolates past ~10B students to 70B–400B, and whether it holds for mixture-of-experts students. The grid exists; the compute to extend it (roughly $10^{23}$–$10^{24}$ FLOPs for one clean IsoFLOP sheet) has not been spent outside frontier labs, and those labs do not publish control arms.
- **Empirically open.** Whether the law survives when the teacher is post-trained (RLHF'd) rather than a base model — the standard industrial case.
- **Theoretically open.** No derivation of $\alpha, \beta$, or the breakpoint from a data or optimization model. The variance-reduction argument gives the sign of the effect, not its exponent.
- **Theoretically open.** Whether the capacity-gap crossover is an optimization artifact (student cannot reach the teacher-matching basin) or a representational one. Stanton et al. is evidence for the former; there is no proof.
- **Methodologically blocked.** "Teacher quality" as a scalar. Two teachers at equal $L_T$ but different predictive entropy transfer differently, and no accepted second statistic exists. Also blocked: downstream capability scaling — no law predicts distilled MMLU or code-generation accuracy from $(N_S, D_S, L_T)$.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement under a two-dimensional budget**. Any distillation experiment moves at least four knobs — $N_T$, $D_T$, $N_S$, $D_S$ — plus $\lambda$ and $\tau$, and the answer depends on the amortization count $k$, which is not a property of the experiment. Holding total FLOPs fixed while sweeping this space is a product grid, not a line: a single clean IsoFLOP sheet at the 10B scale is a multi-million-GPU-hour object.

Second: **non-identifiability of teacher quality**. $L_T$ and $N_T$ are correlated across any realistic teacher family, so a fit that attributes the capacity gap to $L_T$ rather than $N_T$ is only separable if you deliberately train undertrained large teachers and overtrained small ones — which is exactly the expensive off-diagonal part of the grid.

Third: **the evaluation does not measure what it names**. The law predicts cross-entropy; deployments care about instruction-following and reasoning. Distillation from a post-trained teacher changes downstream accuracy far more than it changes $L_S$, so a validated loss law can be simultaneously correct and useless for the decision it is invoked to make.

## 7. Current Research (as of 2026)

- **Apple ML Research** (Busbridge, Ramapuram, Webb): extending the distillation law to on-policy and self-distillation regimes *(frontier — verify)*.
- **Google DeepMind**: production token-level distillation across the Gemma line; scaling ablations are not published.
- **On-policy / reverse-KL distillation**: MiniLLM (Gu et al., ICLR 2024) and GKD (Agarwal et al., ICLR 2024) show that sampling from the student and scoring with the teacher beats forward-KL on generation tasks. Whether these change the *exponents* or only the constants is unmeasured *(frontier — verify)*.
- **Reasoning-trace distillation**: DeepSeek-R1 distilled models (2025) transfer long chain-of-thought traces into 1.5B–70B students. Reported as benchmark numbers with no matched-compute supervised control.
- **Theory**: statistical-perspective line (Menon, Rawat, Kumar and collaborators at Google Research) continuing on variance reduction and teacher-label smoothing.

## 8. Concrete Next Experiment

**Question.** Does the teacher-quality covariate $L_T$ subsume $N_T$, at a scale where the two can be decorrelated?

**Scale.** Students at $N_S \in \{0.4\text{B}, 1.4\text{B}\}$, each trained on $D_S \in \{20, 80, 320\}$ tokens-per-parameter. Nine teachers forming a 3×3 grid: $N_T \in \{1\text{B}, 3\text{B}, 8\text{B}\}$ crossed with $D_T \in \{5, 20, 80\}$ tokens/param. This grid deliberately contains pairs with matched $L_T$ but $8\times$ different $N_T$ (undertrained 8B vs. overtrained 1B). Total: 9 teachers + 54 student runs ≈ $4\times10^{22}$ FLOPs, about 60k H100-hours.

**Control arm.** For each $(N_S, D_S)$, a supervised run on identical data, seed, and schedule — $\lambda = 0$, everything else fixed.

**Deciding number.** Fit $L_S = g(N_S, D_S, L_T)$ ignoring $N_T$, then test whether adding $\log N_T$ as a covariate reduces held-out residual RMSE. **If the RMSE improvement is below 0.005 nats/token (the seed noise floor), $L_T$ subsumes teacher size and the law is well-posed in one teacher variable. If it exceeds 0.02 nats/token, the scalar-teacher assumption fails and every published distillation law is misspecified.**

Secondary readout: the crossover $D_S^\ast$ at which the supervised control overtakes the distilled student, reported per $N_S$. If $D_S^\ast$ scales sublinearly in $N_S$, distillation's usable window shrinks as models grow.

## 9. Key References

- **[SOTA]** Busbridge, D., Shidani, A., Weers, F., Ramapuram, J., Littwin, E., Webb, R. *Distillation Scaling Laws.* ICML, 2025. — arXiv:2502.08606
- **[Foundational]** Hinton, G., Vinyals, O., Dean, J. *Distilling the Knowledge in a Neural Network.* NeurIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Hoffmann, J., et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Kaplan, J., et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Theory]** Menon, A. K., Rawat, A. S., Reddi, S., Kim, S., Kumar, S. *A Statistical Perspective on Distillation.* ICML, 2021.
- **[Theory]** Phuong, M., Lampert, C. *Towards Understanding Knowledge Distillation.* ICML, 2019.
- **[Empirical]** Cho, J. H., Hariharan, B. *On the Efficacy of Knowledge Distillation.* ICCV, 2019. — arXiv:1910.01348
- **[Empirical]** Stanton, S., Izmailov, P., Kirichenko, P., Alemi, A., Wilson, A. G. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- **[Empirical]** Beyer, L., Zhai, X., Royer, A., Markeeva, L., Anil, R., Kolesnikov, A. *Knowledge Distillation: A Good Teacher Is Patient and Consistent.* CVPR, 2022. — arXiv:2106.05237
- **[Empirical]** Mirzadeh, S. I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., Ghasemzadeh, H. *Improved Knowledge Distillation via Teacher Assistant.* AAAI, 2020. — arXiv:1902.03393
- **[Method]** Kim, Y., Rush, A. M. *Sequence-Level Knowledge Distillation.* EMNLP, 2016. — arXiv:1606.07947
- **[Method]** Gu, Y., Dong, L., Wei, F., Huang, M. *MiniLLM: Knowledge Distillation of Large Language Models.* ICLR, 2024. — arXiv:2306.08543
- **[Method]** Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Ramos, S., Geist, M., Bachem, O. *On-Policy Distillation of Language Models (GKD).* ICLR, 2024. — arXiv:2306.13649
- **[Systems]** Gemma Team, Google DeepMind. *Gemma 2: Improving Open Language Models at a Practical Size.* Technical report, 2024. — arXiv:2408.00118
- **[Survey]** Gou, J., Yu, B., Maybank, S., Tao, D. *Knowledge Distillation: A Survey.* IJCV, 2021. — arXiv:2006.05525

## 10. Worked Example

Budget: $C = 10^{22}$ FLOPs. Target: a 1B-parameter serving model.

**Arm A — supervised.** $6 N_S D_S = 10^{22}$ with $N_S = 10^9$ gives $D_S \approx 1.7\text{T}$ tokens, about 1700 tokens/param — roughly 85× past Chinchilla-optimal. Deep in the flat tail of the data term $B/D_S^\beta$.

**Arm B — distill from an 8B teacher.** Train the teacher Chinchilla-optimally: $6 \times 8\times10^9 \times 1.6\times10^{11} \approx 7.7\times10^{21}$ FLOPs. That leaves $2.3\times10^{21}$. Student training plus teacher inference costs $(6 N_S + 2 N_T) D_S = (6\times10^9 + 1.6\times10^{10}) D_S = 2.2\times10^{10} D_S$, so $D_S \approx 105$B tokens — **16× fewer tokens than Arm A**, because the teacher forward pass alone costs $2 N_T = 1.6\times10^{10}$ FLOPs/token, 2.7× the student's own training cost.

**Where it flips.** If the teacher is amortized over $k = 10$ students, teacher training drops to $7.7\times10^{20}$ and $D_S$ rises to ~420B. If the teacher already exists ($k \to \infty$), $D_S \approx 455$B — still 3.7× fewer tokens than the supervised arm, purely from inference overhead.

**The obstruction, visible.** The verdict flips on $k$, which is not measured by any experiment in the grid. At $k=1$ Arm A almost certainly wins on tokens alone. At $k=\infty$ Arm B wins if and only if the per-token advantage of soft targets exceeds the 3.7× token deficit — which requires knowing $\beta$ and the crossover $D_S^\ast$ to better than the ~0.01 nats separating the two curves at 1B scale. Published laws are fitted on grids whose seed noise is of that same order, so the decision sits inside the error bar of the instrument meant to make it. That is the empirical gap: not a missing idea, a missing measurement precise enough to be actionable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*