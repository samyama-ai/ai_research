---
id: 29-distillation/distillation-loss-optimality
title: "Distillation Loss Function Optimality"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distillation Loss Function Optimality

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/distillation-loss-optimality` · **Status:** open

## 1. Problem Statement

Given a fixed teacher $p_T$, a fixed student architecture $q_\theta$, a fixed data budget, and a fixed compute budget, **which divergence between teacher and student should be minimized?** The candidate set includes forward KL (the original soft-target loss), reverse KL, symmetric and skewed mixtures, general $f$-divergences, total variation, logit MSE, and feature/representation matching terms — each optionally combined with the hard-label cross-entropy and each with a temperature and mixing weight.

Three variants, routinely conflated:

- **Measurement.** Is the difference between losses detectable at all, once temperature, weight, learning rate, and epoch budget are tuned *per loss*? Most published comparisons tune one arm.
- **Method.** Is there a rule mapping the setting — capacity gap, on- vs. off-policy data, teacher calibration, task metric (accuracy vs. sequence likelihood vs. pass@k) — to the loss that maximizes downstream metric per unit of student compute?
- **Theory.** Is there a divergence that is provably optimal, or a proof that no fixed divergence dominates and the choice is intrinsically setting-dependent?

Solved would mean: a decision procedure taking $(\text{capacity gap}, \text{data regime}, \text{target metric})$ to a loss, validated to beat per-loss-tuned baselines out-of-sample at more than one scale.

## 2. Formal Setting

Input $x \sim \mathcal{D}$, label space $\mathcal{Y}$ with $|\mathcal{Y}| = K$ (a vocabulary for LMs). Teacher logits $z_T(x) \in \mathbb{R}^K$, student logits $z_S(x;\theta)$. Tempered distributions:

$$p_T^{(\tau)}(y \mid x) = \mathrm{softmax}(z_T(x)/\tau)_y, \qquad q_\theta^{(\tau)}(y \mid x) = \mathrm{softmax}(z_S(x;\theta)/\tau)_y.$$

The general objective:

$$\mathcal{L}(\theta) = (1-\lambda)\,\mathbb{E}_{x,y}\big[-\log q_\theta(y\mid x)\big] \;+\; \lambda \tau^2 \, \mathbb{E}_{x \sim \mu}\big[ D_f\big(p_T^{(\tau)}(\cdot\mid x) \,\|\, q_\theta^{(\tau)}(\cdot\mid x)\big)\big],$$

with $D_f(p\|q) = \sum_y q(y) f(p(y)/q(y))$. Choices: $f(t) = t\log t$ gives forward KL; $f(t) = -\log t$ gives reverse KL; the skew-KL of DistiLLM uses $D_{\mathrm{KL}}(p \,\|\, \alpha p + (1-\alpha) q)$. The $\tau^2$ factor restores gradient scale (Hinton et al., 2015).

**Measured quantities.**
- $\lambda \in [0,1]$, $\tau > 0$: hyperparameters; must be swept per loss or the comparison is invalid.
- $\mu$: the sampling distribution. Off-policy $\mu = \mathcal{D}$; on-policy $\mu = q_\theta$ (student samples, scored by the teacher); mixed $\mu = \beta \mathcal{D} + (1-\beta) q_\theta$.
- **Capacity gap** $g = \log(N_T/N_S)$ in non-embedding parameters, the standard proxy; the honest measurand is teacher-minus-student loss on a held-out set at matched data.
- **Fidelity** — how well the student reproduces the teacher — measured as top-1 agreement $\mathbb{E}_x[\mathbf{1}\{\arg\max q = \arg\max p_T\}]$ and mean $D_{\mathrm{KL}}(p_T\|q)$ on held-out $x$.
- **Downstream metric** $M$: accuracy, ROUGE, pass@1, or win-rate. Report $M$ per student inference FLOP, not per training step.

**Assumptions, and which fail.**
1. *Teacher is a good posterior estimate.* Violated: large LMs are overconfident after RLHF and post-training; the soft targets are not calibrated posteriors.
2. *Fixed loss, fixed optimum.* Violated: the optimal $\lambda,\tau$ drift during training (TAID, ICLR 2025, is built on this).
3. *Token-level factorization.* For sequences, $D_f$ is applied per token under teacher forcing, which is not the divergence between the induced sequence distributions. Known violated — this is the train/inference mismatch that on-policy methods attack.
4. *Full teacher distribution available.* Violated in practice: top-$k$ truncation (often $k \le 128$ of a 128k vocabulary) is used for storage, which changes the estimated divergence, and by different amounts for forward vs. reverse KL.

## 3. State of the Art

**Established (ablated, reproduced).**
- Forward-KL soft targets with $\tau \in [2,5]$ beat hard labels for small vision students (Hinton, Vinyals, Dean, 2015; replicated widely).
- "Function matching" — same augmentation for teacher and student, very long schedules — is the strongest supervised-vision recipe: Beyer et al. (CVPR 2022) reach 82.8% ImageNet top-1 with a distilled ResNet-50, and show the gain requires consistency plus ~1000-epoch budgets.
- Sequence-level KD (Kim & Rush, EMNLP 2016) — train on teacher-generated outputs — beats token-level word-KD for NMT; still the default for many production LM distillations.
- On-policy data matters. GKD (Agarwal et al., ICLR 2024) shows that sampling from the student and scoring with the teacher beats fixed-dataset distillation across summarization, translation, and reasoning; ablations separate the data-source effect from the divergence effect.

**Claimed but not cleanly ablated.**
- *Reverse KL is right for generative LMs because it is mode-seeking.* MiniLLM (Gu et al., ICLR 2024) reports gains, but reverse KL is bundled with policy-gradient training, single-step decomposition, and length normalization. GKD's own sweep finds the best divergence varies by task — forward KL sometimes wins — which undercuts the general claim.
- *Skewed KL is uniformly better.* DistiLLM (Ko et al., ICML 2024) reports quality and speed gains, but the loss change ships with adaptive off-policy replay.
- *Logit MSE beats KL.* Kim et al. (IJCAI 2021) argue KD with $\tau \to \infty$ reduces to logit matching and that direct MSE on logits performs better; the effect is small and scale-limited.

**Benchmark-number-only.** Most frontier-model distillation claims (Gemma 2's use of teacher distillation for the 2B/9B models, and similar reports for small proprietary models) state that distillation helped but do not publish loss-function ablations at that scale. Treat as unablated.

## 4. What Is Known

- **Distillation is not free at scale.** Busbridge et al., *Distillation Scaling Laws* (2025, arXiv:2502.08606), fit a law over students from ~140M to ~12.6B parameters and teachers up to ~12.6B. Result: distillation beats supervised pretraining only below a data/compute threshold; past it, supervised training wins, and a stronger teacher can *hurt* (a capacity-gap effect visible in the fitted law).
- **Fidelity ≠ generalization.** Stanton et al. (NeurIPS 2021) show self-distilled students on CIFAR-100 fail to match teacher predictions even on the *training* set — student-teacher top-1 agreement stays well below 100% while test accuracy improves. Optimization, not capacity, is the binding constraint.
- **Capacity gap is real and non-monotone.** Cho & Hariharan (ICCV 2019): on ImageNet/CIFAR, larger teachers can lower student accuracy; early-stopping the teacher recovers much of the loss.
- **Theory covers only easy cases.** Phuong & Lampert (ICML 2019) prove transfer-risk bounds for linear students under distillation. Menon et al. (ICML 2021) show the variance-reduction benefit scales with how close $p_T$ is to the Bayes class-probability — a statement about teacher quality, not about which divergence.
- **$f$-divergence choice is task-dependent.** Wen et al. (ACL 2023) sweep forward KL, reverse KL, JS, and TV for sequence-level KD and find total variation distance competitive, with no single winner across tasks.

## 5. What Is Not Known

- **Theoretically open.** No theorem gives the optimal $D_f$ as a function of capacity gap and teacher calibration, nor a no-free-lunch result forbidding one. Existing bounds are for linear or convex-optimized students and do not distinguish divergences.
- **Empirically open.** No published study sweeps $\{$forward KL, reverse KL, skew-KL, JS, TV, logit MSE$\}$ × $\{\tau,\lambda\}$ × $\{$on-, off-, mixed-policy$\}$ with per-arm tuning at more than one student scale on the same teacher. Runnable today; nobody has run it. Cost is the reason, not difficulty.
- **Methodologically blocked.** "Which loss is better" has no agreed measurand. Fidelity, downstream accuracy, and calibration rank losses differently (Stanton et al.), and reverse-KL students are systematically lower-entropy, which inflates greedy-decode metrics and deflates pass@k. Until the target metric is fixed and the decoding temperature is held constant across arms, the comparison is not well posed.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement with a shared hyperparameter surface**. Divergence, temperature, mixing weight, and sampling distribution interact: reverse KL at $\tau=1$ resembles forward KL at high $\tau$ in gradient direction; changing $\lambda$ shifts the effective learning rate on the distillation term. A comparison that fixes $(\tau, \lambda)$ across arms measures the tuning, not the loss. Doing it properly costs $|\text{losses}| \times |\text{grid}| \times |\text{scales}|$ full student trainings — order 100 runs.

Second obstruction: **absent ground truth.** There is no reference "correct" student distribution to score against, so the criterion falls back to the downstream benchmark, and the loss that wins depends on which benchmark and which decoding rule is chosen.

## 7. Current Research (as of 2026)

- **On-policy and interpolated objectives.** GKD (Google DeepMind), DistiLLM (KAIST/NVIDIA), TAID (Sakana AI, ICLR 2025), which anneals an interpolated intermediate teacher over training. The trend is away from a fixed divergence toward a schedule *(frontier — verify current variants)*.
- **Scaling-law framing.** Apple's distillation scaling law (2025) recast the question as compute allocation between teacher and student. Extending it to divergence choice is the obvious next step and is not yet published *(frontier — verify)*.
- **Reasoning distillation.** Chain-of-thought and RL-trace distillation (DeepSeek-R1's distilled Qwen/Llama students, 2025) uses plain SFT on teacher traces — no divergence term at all — and is competitive. This is evidence that the sampling distribution dominates the divergence.
- **Top-$k$ logit storage.** Practical work on truncated teacher distributions for pipeline efficiency; the interaction with divergence choice is unstudied.

## 8. Concrete Next Experiment

**Scale.** One frozen teacher, ~7B parameters, on a fixed 50B-token corpus. Students at two sizes: 0.5B and 1.5B (two scales is the minimum to detect capacity-gap dependence). Each student trained on 20B tokens.

**Arms.** Six losses — forward KL, reverse KL, skew-KL ($\alpha=0.1$), JS, TV, logit MSE — crossed with off-policy and 50/50 mixed-policy data. Each arm gets an independent 12-point random search over $(\tau \in [1,6], \lambda \in [0.3,1.0], \text{lr})$ at 1/10 token budget; the best config is trained to full budget. Total: 12 full runs plus 144 short runs.

**Control arm.** Hard-label training at matched tokens and matched tuning budget, plus sequence-level KD on teacher samples with no divergence term.

**Deciding number.** The spread in held-out downstream score (average of MMLU, GSM8K pass@1 at $T=0$, and HumanEval pass@10 at $T=0.8$) across the six divergences *within* each (scale, data-policy) cell, reported against the seed-to-seed standard deviation (3 seeds on one arm). If the max-minus-min divergence spread is under $2\sigma_{\text{seed}}$ at both scales, divergence choice is measurement noise once tuned, and the field should stop optimizing it. If the spread exceeds $2\sigma_{\text{seed}}$ *and the ranking changes between 0.5B and 1.5B*, the capacity-gap-dependent rule is real and worth fitting.

## 9. Key References

- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Yoon Kim, Alexander M. Rush. *Sequence-Level Knowledge Distillation.* EMNLP, 2016. — arXiv:1606.07947
- **[Theory]** Mary Phuong, Christoph Lampert. *Towards Understanding Knowledge Distillation.* ICML, 2019.
- **[Theory]** Aditya Krishna Menon, Ankit Singh Rawat, Sashank Reddi, Seungyeon Kim, Sanjiv Kumar. *A Statistical Perspective on Distillation.* ICML, 2021.
- **[Empirical]** Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander Alemi, Andrew Gordon Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- **[Empirical]** Jang Hyun Cho, Bharath Hariharan. *On the Efficacy of Knowledge Distillation.* ICCV, 2019. — arXiv:1910.01348
- **[SOTA]** Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD).* ICLR, 2024. — arXiv:2306.13649
- **[SOTA]** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang. *MiniLLM: Knowledge Distillation of Large Language Models.* ICLR, 2024. — arXiv:2306.08543
- **[SOTA]** Jongwoo Ko, Sungnyun Kim, Tianyi Chen, Se-Young Yun. *DistiLLM: Towards Streamlined Distillation for Large Language Models.* ICML, 2024. — arXiv:2402.03898
- **[SOTA]** Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, Russ Webb. *Distillation Scaling Laws.* 2025. — arXiv:2502.08606
- **[Empirical]** Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov. *Knowledge Distillation: A Good Teacher Is Patient and Consistent.* CVPR, 2022. — arXiv:2106.05237
- **[Comparison]** Yuqiao Wen, Zichao Li, Wenyu Du, Lili Mou. *f-Divergence Minimization for Sequence-Level Knowledge Distillation.* ACL, 2023. — arXiv:2307.15190
- **[Comparison]** Taehyeon Kim, Jaehoon Oh, NakYil Kim, Sangwook Cho, Se-Young Yun. *Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation.* IJCAI, 2021. — arXiv:2105.08919
- **[Schedule]** Makoto Shing et al. *TAID: Temporally Adaptive Interpolated Distillation for Efficient Language Model Knowledge Distillation.* ICLR, 2025.

## 10. Worked Example

Two-token toy that makes the confound visible. Teacher on input $x$: $p_T = (0.60, 0.30, 0.10)$ over $K=3$. Student is capacity-limited and can only realize distributions of the form $q_\beta = \mathrm{softmax}(\beta \cdot (1, 0, 0))$ — one degree of freedom, so it cannot represent $p_T$.

$q_\beta = \left(\frac{e^\beta}{e^\beta+2}, \frac{1}{e^\beta+2}, \frac{1}{e^\beta+2}\right)$.

- **Forward KL**, $\sum_y p_T(y)\log\frac{p_T(y)}{q_\beta(y)}$, is minimized where the student's mass on $y_1$ matches the teacher's: $\frac{e^\beta}{e^\beta+2} = 0.60 \Rightarrow e^\beta = 3.0$, $\beta^\star \approx 1.10$. Residual $D_{\mathrm{KL}}(p_T\|q) = 0.6\ln\frac{0.6}{0.6} + 0.3\ln\frac{0.3}{0.2} + 0.1\ln\frac{0.1}{0.2} \approx 0 + 0.1216 - 0.0693 = 0.052$ nats.
- **Reverse KL**, $\sum_y q_\beta(y)\log\frac{q_\beta(y)}{p_T(y)}$, penalizes the student for putting mass where the teacher has little. Because $y_2$ and $y_3$ are tied in $q$ but not in $p_T$, the mode-seeking solution pushes $\beta$ higher: numerically $\beta^\star \approx 1.6$, i.e. $q(y_1) \approx 0.71$.

Both are "correct" minimizers of their own objective. Now the confound: run forward KL at temperature $\tau = 2$ instead. Tempering flattens $p_T$ toward $(0.44, 0.31, 0.25)$, and the tempered forward-KL optimum lands near $q(y_1) \approx 0.44$ in tempered space — which, un-tempered, corresponds to a *lower* $\beta$ than reverse KL by more than the forward/reverse gap itself. **The temperature moves the solution further than the choice of divergence does.** Any experiment that fixes $\tau$ across divergence arms is reporting an artifact of that fixed $\tau$.

Scale this up: on a 128k vocabulary with top-$k$ truncation at $k=128$, the discarded tail carries ~1–3% of teacher mass. Forward KL is nearly insensitive to it (the teacher weights the terms). Reverse KL is not — the student's mass on truncated tokens is scored against a renormalized teacher that assigns them zero, so the loss is biased by the storage format. Two labs running "reverse KL vs. forward KL" with different $k$ can get opposite rankings without either being wrong.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*