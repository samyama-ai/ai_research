---
id: 10-scaling-laws/distillation-compute-allocation
title: "Distillation Compute Allocation"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distillation Compute Allocation

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/distillation-compute-allocation` · **Status:** open

## 1. Problem Statement

Given a total training budget $C$ FLOPs and a target deployment size $N_S$ parameters, how should $C$ be split between (a) training a teacher, (b) running teacher inference to produce logits or samples, and (c) training the student on those targets versus on ground-truth text?

Three variants, with different difficulty:

- **Measurement.** Given a fixed protocol, measure the student loss surface $L_S(N_T, D_T, N_S, D_S)$ over teacher size, teacher tokens, student size, student tokens. Runnable; expensive; partly done.
- **Method.** Produce an allocation rule that beats compute-matched supervised pretraining at the frontier, i.e. a map $C \mapsto (N_T^\*, D_T^\*, N_S, D_S^\*)$. Open at frontier scale.
- **Theory.** Explain *why* a fixed student improves under a mid-size teacher but degrades under a much larger one (the capacity gap), from properties of the target distribution rather than post-hoc curve fits. Theoretically open.

Solving it means: a closed-form allocation with fitted constants that extrapolates at least one order of magnitude in $C$ and holds across two model families, with a stated compute threshold above which distillation stops paying.

## 2. Formal Setting

Objects. Teacher $p_T$ with $N_T$ non-embedding parameters trained on $D_T$ tokens; student $q_S$ with $N_S$ parameters trained on $D_S$ tokens; data distribution $p^\*$.

Loss, as measured: mean next-token cross-entropy in nats on a held-out corpus disjoint from all training data,
$$L(q)=-\frac{1}{|H|}\sum_{(x_{<t},x_t)\in H}\log q(x_t\mid x_{<t}).$$
Report $L$, not perplexity, so that mixing coefficients are linear.

Student objective, the standard interpolation:
$$\mathcal{L}_S=(1-\lambda)\,\mathrm{CE}\!\left(x_t, q_S\right)+\lambda\,\tau^2\,\mathrm{KL}\!\left(p_T^{(\tau)}(\cdot\mid x_{<t})\,\middle\|\,q_S^{(\tau)}(\cdot\mid x_{<t})\right),$$
with temperature $\tau$ and mixing weight $\lambda\in[0,1]$. Top-$k$ truncation of $p_T$ (typical $k\in\{8,\dots,128\}$) is a measured, not incidental, choice: it changes the target.

Compute accounting, as it would actually be billed:
$$C=\underbrace{6N_TD_T}_{\text{teacher train}}+\underbrace{2N_TD_S}_{\text{teacher inference}}+\underbrace{6N_SD_S}_{\text{student train}},$$
using the standard $6ND$ / $2ND$ approximations. Teacher inference is amortised across $M$ students, so the effective term is $2N_TD_S$ per student but $6N_TD_T/M$ for the teacher's training. **The value of $M$ is the whole argument** — at $M=1$ distillation must beat supervised training including the teacher's cost; at $M\gg1$ the teacher is free.

Two regimes must be named separately:
- **Teacher-exists**: teacher cost is sunk, $C=2N_TD_S+6N_SD_S$.
- **Teacher-paid**: full $C$ above.

Assumptions, and which are violated:
- *Power-law-in-$(N,D)$ loss with additive irreducible term* (Chinchilla form). Violated near data repetition and for post-trained teachers.
- *Teacher quality is a scalar summarised by $L_T$*. Violated: two teachers at equal $L_T$ but different $(N_T,D_T)$ do not transfer equally in reported ablations; calibration and entropy differ.
- *Fixed distillation recipe across scales* ($\lambda,\tau,k$ constant). Violated in practice — optimal $\lambda$ drifts with $N_S/N_T$, which confounds every scaling fit that holds it fixed.
- *On-policy/off-policy equivalence*. Violated: sequence-level and on-policy variants (GKD, MiniLLM) change the exponent, not just the constant.

## 3. State of the Art

**Empirical SOTA (established).** Busbridge et al., *Distillation Scaling Laws* (Apple, ICML 2025, arXiv:2502.08606) fit a law for student cross-entropy as a function of $(L_T,N_S,D_S)$ over a large sweep (students to ~7B parameters, teachers larger, up to hundreds of billions of distillation tokens). Two findings are ablated: the capacity gap is a *function of teacher loss and student capability*, not of the parameter ratio alone; and there exists a student-compute threshold beyond which supervised pretraining overtakes distillation. Below the threshold, and when the teacher already exists or serves many students, distillation is compute-optimal.

**Systems SOTA (claimed, partly unablated).** Gemma 2 (Google DeepMind, 2024) trained 2B/9B students on teacher distributions and reported large gains over compute-matched supervised training; there is no released compute-matched control including teacher cost. Minitron (Muralidharan et al., NeurIPS 2024, arXiv:2407.14679) reports up to $40\times$ fewer training tokens for a pruned-and-distilled 8B/4B model versus training from scratch — a benchmark number under a "teacher exists" assumption, not an allocation law. Llama 3.2 1B/3B are described as pruned + distilled; no allocation ablation published.

**Theory SOTA.** Menon et al., *A Statistical Perspective on Distillation* (ICML 2021): the teacher's soft labels act as a lower-variance estimate of the Bayes class-probabilities, and the gain scales with the variance reduction — this predicts a benefit but not a capacity gap. Mirzadeh et al. (TAKD, AAAI 2020) give the teaching-assistant construction as an empirical remedy, no bound. No theory predicts the observed compute threshold.

## 4. What Is Known

- **Fidelity is not the mechanism.** Stanton et al. (NeurIPS 2021) show students often fail to match teacher predictions even on the distillation data — measured agreement stays well below what generalisation gains would imply — so distillation's benefit is optimisation/regularisation, not function matching. Measured at CIFAR/ImageNet scale with ResNets.
- **Capacity gap is real.** Across image classification (ResNet/ViT) and LMs, holding the student fixed and increasing $N_T$ gives a non-monotone student loss: improvement then degradation. Reported repeatedly; the turning point is not predicted a priori.
- **Consistency and duration dominate the recipe.** Beyer et al. (CVPR 2022) show that with identical augmentation views for teacher and student and very long schedules ($\sim$1M epochs-equivalent on small sets), a ResNet-50 student reaches 82.8% ImageNet top-1 — a gain that comes from schedule length, not loss design.
- **Distillation has a compute ceiling.** In the Apple fits, distillation's advantage vanishes once student compute exceeds a scale-dependent threshold; past it, supervised training at the same $C$ gives lower $L_S$.
- **Inference-aware allocation shifts the optimum.** Sardana et al. (ICML 2024, arXiv:2401.00448) show that accounting for serving cost pushes optimal models far smaller and more over-trained than Chinchilla — the same accounting that makes distillation attractive at all.

## 5. What Is Not Known

- **Empirically open.** Whether a single allocation law fitted below $10^{22}$ FLOPs extrapolates to $10^{24}$–$10^{25}$ FLOPs frontier runs. The experiment is well defined and simply unrun outside a handful of labs; published frontier distillation runs report no compute-matched supervised control.
- **Empirically open.** The correct $M$ (students per teacher) at which distillation dominates, and whether the law is invariant to the student *family* (dense vs MoE, different tokenizers). No public cross-family fit exists.
- **Theoretically open.** Any bound predicting the capacity-gap turning point from properties of $p_T$ (entropy, calibration, top-$k$ mass) rather than from a fitted curve. No proof either way that a gap must exist for expressive students.
- **Methodologically blocked.** "Teacher quality" as an input to the law. Summarising a teacher by its cross-entropy is known to be lossy, but no measured alternative statistic (per-token entropy profile, top-$k$ mass concentration, calibration error) has been shown to be the sufficient one. Until that is fixed, the law's independent variable is mis-specified.
- **Methodologically blocked.** Downstream transfer. Allocation laws are fitted on cross-entropy; whether the optimum for $L_S$ is the optimum for reasoning benchmarks is unmeasured, and distilled models are known to shift the loss-to-benchmark mapping.

## 6. Why It Is Hard

The controlling obstruction is **confounded measurement plus non-identifiability of teacher quality**. The fit has at least five free axes ($N_T, D_T, N_S, D_S, \lambda$, plus $\tau,k$), and the recipe hyperparameters interact with the scaling axes: the optimal $\lambda$ moves with $N_S/N_T$, so a sweep at fixed $\lambda$ measures a slice of the surface and attributes recipe suboptimality to the capacity gap. Distinguishing the two requires re-tuning $\lambda$ at every grid point, which multiplies an already ~$10^2$-run sweep.

Second, **teacher cost accounting is unfalsifiable without $M$**. Nearly every published win is reported in the teacher-exists regime, where the expensive term is deleted from the denominator. A method that "beats supervised training" at $M=\infty$ and loses at $M=1$ is not wrong, but the two claims are routinely printed as one.

Third, **compute cost at the deciding scale**. The disagreement between distillation and supervised training is small below $10^{21}$ FLOPs and only opens where a single arm costs seven figures — so the regime that would settle it is the regime nobody ablates.

## 7. Current Research (as of 2026)

- Extending distillation scaling laws to on-policy targets, where the student's own samples are scored by the teacher (GKD, Agarwal et al., ICLR 2024; MiniLLM, Gu et al., ICLR 2024). Whether on-policy changes the *exponent* or only the constant is the live question *(frontier — verify)*.
- Prune-then-distill pipelines as an allocation primitive: Minitron-style width/depth pruning of an existing teacher into its own student, NVIDIA and Meta *(frontier — verify)*.
- Small-model-assisted large-model training — a weak model providing soft targets or curriculum for a stronger one (Rawat et al., *A Little Help Goes a Long Way*, Google Research, 2024) — which inverts the usual direction and is unexplained by capacity-gap arguments.
- Reasoning-trace distillation (distilling chain-of-thought from a reasoning teacher) as a distinct allocation problem where teacher inference cost per token is much larger than $2N_T$ *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** At $M=1$ (teacher paid in full), does any allocation of $C$ through a teacher beat supervised training at the same $C$?

**Scale.** $C = 3\times10^{20}$ FLOPs total per arm; student fixed at $N_S = 400$M non-embedding parameters. Roughly 8 GPU-days per arm on H100s; ~20 arms.

**Arms.** Teachers at $N_T \in \{0.4, 1, 3\}$B, each trained to Chinchilla-optimal $D_T$, then distilled into the 400M student on whatever $D_S$ the residual budget allows after charging $6N_TD_T + 2N_TD_S$. At each grid point re-tune $\lambda\in\{0.3,0.5,0.7,1.0\}$ — do not hold it fixed.

**Control arm.** A 400M model trained supervised on $D_S = C/(6N_S) \approx 1.25\times10^{11}$ tokens from the identical corpus, same tokenizer, same optimiser, same seed count.

**Deciding number.** $\Delta L = L_S^{\text{distill,best}} - L_S^{\text{supervised}}$ in nats on held-out data, with seed variance estimated from 3 seeds per arm. If $\Delta L < -0.01$ nats (about $1\%$ perplexity, comfortably above the $\sim0.003$ nat seed spread at this scale), distillation pays at $M=1$ and the compute threshold sits above $3\times10^{20}$. If $\Delta L \geq 0$ for every $(N_T,\lambda)$, the teacher-paid regime is settled negatively at this scale and the frontier claim must be argued from $M$, not from the loss.

**Second reading, free.** Plot $\Delta L$ against $L_T$ and against $N_T/N_S$. Whichever collapses the curves is the sufficient teacher statistic — this is the methodologically blocked item in §5, and this sweep is the cheapest test of it.

## 9. Key References

- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, Russ Webb. *Distillation Scaling Laws.* ICML, 2025. — arXiv:2502.08606
- **[SOTA]** Saurabh Muralidharan et al. *Compact Language Models via Pruning and Knowledge Distillation.* NeurIPS, 2024. — arXiv:2407.14679
- **[Evidence]** Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, Andrew Gordon Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- **[Evidence]** Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov. *Knowledge Distillation: A Good Teacher Is Patient and Consistent.* CVPR, 2022. — arXiv:2106.05237
- **[Theory]** Aditya Krishna Menon, Ankit Singh Rawat, Sashank Reddi, Seungyeon Kim, Sanjiv Kumar. *A Statistical Perspective on Distillation.* ICML, 2021.
- **[Method]** Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem. *On-Policy Distillation of Language Models (GKD).* ICLR, 2024. — arXiv:2306.13649
- **[Method]** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang. *MiniLLM: Knowledge Distillation of Large Language Models.* ICLR, 2024. — arXiv:2306.08543
- **[Related]** Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML, 2024. — arXiv:2401.00448
- **[Survey]** Xiaohan Xu et al. *A Survey on Knowledge Distillation of Large Language Models.* 2024. — arXiv:2402.13116

## 10. Worked Example

Budget $C=10^{21}$ FLOPs, student fixed at $N_S=1$B.

*Supervised control.* $D_S = C/(6N_S) = 1.67\times10^{11}$ tokens. A 1B model on 167B tokens is roughly Chinchilla-optimal ($\approx$20 tokens/param would be 20B; this is 167× over-trained, so squarely in the over-trained regime where the Chinchilla fit still holds). Take the resulting held-out loss as $L_{\text{sup}}$.

*Distillation arm, teacher paid.* Choose $N_T=7$B at Chinchilla-optimal $D_T=1.4\times10^{11}$ tokens. Teacher training costs $6\times7\times10^9\times1.4\times10^{11}=5.9\times10^{21}$ FLOPs — **six times the entire budget**. The arm is infeasible before a single student token is seen.

Shrink the teacher to $N_T=1.5$B, $D_T=3\times10^{10}$: teacher cost $2.7\times10^{20}$, leaving $7.3\times10^{20}$. Student tokens then satisfy $(6N_S+2N_T)D_S = 7.3\times10^{20}$, i.e. $(6\times10^9+3\times10^9)D_S$, giving $D_S=8.1\times10^{10}$ tokens.

The obstruction is now visible as arithmetic: the student sees **8.1e10 tokens against the control's 1.67e11** — less than half — and its teacher is only $1.5\times$ its own size, comfortably inside the regime where soft targets add little because $L_T$ is barely below what the student could reach alone. To buy a teacher good enough to be worth learning from, you must spend the budget that would have trained the student. Teacher inference alone ($2N_TD_S$) eats a further 33% of student throughput.

At $M=8$ students, the teacher-training term drops to $3.4\times10^{19}$ and $D_S$ rises to $1.07\times10^{11}$ — still short of the control, but now the loss gap from fewer tokens (roughly $0.03$ nats at this scale, from a Chinchilla-form fit with exponent $\approx0.28$ on $D$) is plausibly repaid by the soft targets. **The entire conclusion turns on $M$, a deployment fact, not a learning fact** — and no published frontier result states its $M$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*