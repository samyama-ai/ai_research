---
id: 12-quantization-compression/distillation-capacity-gap
title: "Knowledge Distillation Capacity Gap"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Knowledge Distillation Capacity Gap

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/distillation-capacity-gap` · **Status:** open

## 1. Problem Statement

Knowledge distillation trains a small **student** on the outputs of a large **teacher**. The empirical regularity that defines this problem: *making the teacher stronger does not monotonically make the student stronger*. Past some teacher size, student accuracy flattens or drops, even though the teacher's own accuracy keeps rising. This is the **capacity gap**.

Three variants, routinely conflated:

- **Measurement.** Given a student architecture $S$ and a family of teachers $\{T_c\}$ indexed by compute $c$, is there a well-defined $c^\star(S)$ that maximizes student quality, and is it identifiable from cheap observables (teacher loss, teacher–student agreement, gradient noise) without training every student?
- **Method.** Given a fixed student budget and unlimited access to teachers, can a procedure recover the accuracy achievable by the *best* teacher without a search over teachers? Teacher assistants, early-stopped teachers, temperature/loss-weight tuning, and data augmentation are all proposed answers.
- **Theory.** Is the non-monotonicity a property of the *representational* capacity of $S$ (the teacher's function is not in the student's hypothesis class), or of *optimization* (it is in the class, but SGD from a soft-label objective does not find it)? These predict different interventions and no experiment cleanly separates them at scale.

Solving it means: a predictive rule for $c^\star(S)$ accurate to within one model-size step, plus a mechanism claim that survives an ablation distinguishing representation from optimization.

## 2. Formal Setting

Data $(x,y) \sim \mathcal{D}$ over $\mathcal{X} \times [K]$. Teacher $p_T(\cdot\mid x) \in \Delta^{K-1}$, student $p_S(\cdot \mid x;\theta)$, $\theta \in \mathbb{R}^{d_S}$. Temperature-$\tau$ softened distributions $p^\tau \propto p^{1/\tau}$. The standard objective:

$$\mathcal{L}(\theta) = (1-\alpha)\,\mathbb{E}\big[\ell_{\mathrm{CE}}(y, p_S(\cdot\mid x;\theta))\big] \;+\; \alpha\,\tau^2\,\mathbb{E}\big[\mathrm{KL}\big(p_T^\tau(\cdot\mid x)\,\|\,p_S^\tau(\cdot\mid x;\theta)\big)\big].$$

Quantities, **as measured**:

- **Teacher compute** $c_T$ = parameters $\times$ training tokens (or FLOPs), the axis used by distillation scaling laws.
- **Student quality** $A_S$ = top-1 accuracy, or held-out cross-entropy in nats/token for LMs. Report cross-entropy: accuracy saturates and hides the gap.
- **Fidelity** $F$ = top-1 agreement $\mathbb{E}_x[\mathbf{1}\{\arg\max p_S = \arg\max p_T\}]$, measured **on the training set**, and predictive KL $\mathbb{E}_x[\mathrm{KL}(p_T\|p_S)]$. Fidelity and $A_S$ are distinct objectives and are empirically decoupled.
- **Capacity gap** $\Gamma(S,T) = \mathbb{E}_x[\mathrm{KL}(p_T\|p_S^\star)]$ where $p_S^\star$ is the *distillation-optimal* student. Not directly measurable: $p_S^\star$ requires solving a non-convex problem, so every reported $\Gamma$ is an upper bound contaminated by optimization error.
- **Optimum** $c^\star(S) = \arg\max_{c} A_S(c)$, estimated by a sweep over teachers; each point costs one full student run.

Assumptions and their status:

| Assumption | Status |
|---|---|
| Teacher probabilities are calibrated estimates of $P(y\mid x)$ | **Violated** — large nets are overconfident; the "dark knowledge" signal is partly miscalibration |
| Student can be optimized to its distillation optimum | **Violated** — Stanton et al. show large train-set fidelity gaps that persist under longer training |
| One scalar $c_T$ orders teachers | **Violated** — a 1B model trained on 10T tokens and a 10B model on 1T tokens are not interchangeable teachers |
| Distillation set $\approx \mathcal{D}$ | Often violated; transfer-set choice changes conclusions |

## 3. State of the Art

**Established (reproduced, ablated):**
- Non-monotonicity itself. Cho & Hariharan (*On the Efficacy of Knowledge Distillation*, ICCV 2019) show larger teachers can hurt on CIFAR-100/ImageNet, and that **early-stopping the teacher** recovers much of the loss — an ablation that separates teacher accuracy from teacher usefulness.
- Fidelity $\neq$ generalization. Stanton et al. (*Does Knowledge Distillation Really Work?*, NeurIPS 2021) show students that generalize better than the baseline still fail to match the teacher on training data, and that improved optimization raises fidelity without raising accuracy.
- Long-schedule consistent distillation works. Beyer et al. (*Knowledge distillation: a good teacher is patient and consistent*, CVPR 2022) reach ResNet-50 82.8% ImageNet top-1 by treating distillation as function matching with shared augmentations over $\sim$10k epochs.
- Distillation scaling laws. Busbridge et al. (*Distillation Scaling Laws*, 2025) fit student loss as a function of teacher and student compute across a wide LM sweep and find a regime boundary: distillation beats supervised pretraining only below a student-compute threshold, and student loss is non-monotone in teacher capability at fixed student size.

**Claimed but unablated:**
- **Teacher assistant** (Mirzadeh et al., AAAI 2020): intermediate-size TA closes part of the gap on CIFAR-10/100. The confound — TA training adds compute and a second soft-label pass — is not isolated against a compute-matched single-teacher control.
- Feature/attention-matching methods (FitNets, TinyBERT-style layer mapping) reported to help across gaps; gains are usually entangled with extra augmentation or longer schedules.
- Frontier-LM claims (Gemma 2/3 technical reports) that distillation from a much larger teacher improves small models. These are **benchmark numbers only**: no teacher-size sweep at fixed student, no public control arm.

**Theory SOTA** is weaker than empirical. Menon et al. (*A Statistical Perspective on Distillation*, ICML 2021) show soft labels lower the variance of the risk estimate when they approximate $P(y\mid x)$ — which predicts *better* students from better teachers, i.e. it does not explain the gap. Harutyunyan et al. (*Supervision Complexity and its Role in Knowledge Distillation*, ICLR 2023) give a bound where the student's generalization depends on the *complexity of the teacher's labels relative to the student's kernel*, producing a genuine trade-off between teacher accuracy and teacher learnability. This is the closest thing to a theory of the gap; it is proved in a kernel/NTK regime.

## 4. What Is Known

- CIFAR-100, WRN/ResNet students: distillation from progressively larger teachers peaks and then declines; TA-based staging reported to recover roughly **0.5–2 points** top-1 (Mirzadeh et al., AAAI 2020) at CIFAR scale — small models, small data.
- ImageNet, ResNet-18/ResNet-50 students: naive distillation from very large teachers underperforms distillation from mid-size teachers unless the teacher is early-stopped or the schedule is very long (Cho & Hariharan 2019; Beyer et al. 2022, 82.8% for ResNet-50).
- Train-set top-1 teacher–student agreement stays well short of 100% even when the student has enough capacity to represent the teacher, and self-distillation students do not match their own teacher (Stanton et al. 2021, CIFAR-100 and ImageNet). This localizes a large part of the gap in **optimization, not representation**.
- LM scale: DistilBERT (Sanh et al. 2019) retains ~97% of BERT-base GLUE score at 40% fewer parameters — a single teacher–student pair, not a sweep. Turc et al. (*Well-Read Students Learn Better*, 2019) show pre-training the student before distillation matters more than teacher choice for small BERTs.
- Distillation scaling laws (Busbridge et al. 2025): at fixed student size and data, student cross-entropy is U-shaped in teacher cross-entropy — the strongest published quantitative evidence that the effect survives to LM pretraining scale.

## 5. What Is Not Known

- **Theoretically open.** No proof that non-monotonicity must occur for any natural function class and optimizer, and no proof it cannot. Harutyunyan et al. give a mechanism under kernel assumptions; no extension to feature-learning regimes.
- **Theoretically open.** Whether $\Gamma(S,T)$ is representationally lower-bounded away from zero for realistic $(S,T)$ pairs, or is entirely an optimization artifact. Stanton's evidence favors optimization; there is no separation theorem.
- **Empirically open.** A full teacher-size $\times$ student-size grid at LM pretraining scale with *compute-matched control arms* and every knob (temperature, $\alpha$, transfer-set size, schedule length) tuned per cell. Runnable today; the cost is the blocker.
- **Empirically open.** Whether the teacher-assistant gain survives compute matching. Nobody has published TA vs. single-teacher-with-equal-total-FLOPs.
- **Methodologically blocked.** "Capacity gap" has no agreed measurement. Papers report accuracy deltas, KL, agreement, and rank correlation interchangeably; these give different orderings of the same teachers. Until $\Gamma$ is defined operationally with an optimization-error control, cross-paper comparison is not meaningful.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus a quadratic sweep cost**.

- Every observation of $c^\star(S)$ requires training a student per teacher. A $6 \times 6$ teacher-student grid at 1B-parameter scale is 36 pretraining runs; with per-cell hyperparameter tuning ($\tau$, $\alpha$, schedule) it is several hundred. That is why the literature reports single pairs.
- Under-tuning is not neutral — it biases *toward* finding a gap. A large teacher needs a different $\tau$ and a longer schedule (Beyer et al. 2022 needed ~10k epochs). A sweep with a shared recipe will report non-monotonicity that a per-cell-tuned sweep might not.
- $\Gamma$ is not identifiable: measured $\mathrm{KL}(p_T\|p_S)$ = representation error + optimization error, and there is no estimator that separates them without knowing $p_S^\star$.
- The evaluation often does not measure what it names. Downstream accuracy is what is reported; fidelity is what distillation optimizes. Their decoupling means an "improved distillation method" may simply be a better regularizer.

## 7. Current Research (as of 2026)

- **Distillation scaling laws** (Apple, Busbridge et al.) — fitting closed forms for student loss over teacher/student compute, and deriving when to distill vs. pretrain. The main quantitative program.
- **Frontier small-model distillation** — Google DeepMind (Gemma line), Meta (Llama small models), Microsoft (Phi line) train small models on large-teacher logits or synthetic teacher data at production scale. Teacher-size ablations are not published *(frontier — verify)*.
- **On-policy / sequence-level distillation** — GKD-style methods where the student's own samples are scored by the teacher, reducing train/inference distribution mismatch. Reported to reduce the gap for generative tasks *(frontier — verify whether the gain is gap-closing or just exposure-bias fixing)*.
- **Supervision-complexity theory** (Google Research; Harutyunyan, Menon, Kumar and collaborators) — extending kernel-regime bounds and designing teacher-side regularizers that make labels learnable rather than accurate.
- **Distillation interacting with quantization/pruning** — whether QAT-plus-distillation shifts $c^\star(S)$, since a quantized student has lower effective capacity than its parameter count implies. Largely unmeasured.

## 8. Concrete Next Experiment

**Question:** is the capacity gap representational or optimizational, at a scale where the answer matters?

**Scale.** Language modeling, single corpus (e.g. a fixed 100B-token slice). Students: 150M and 400M parameters. Teachers: 0.5B, 1.5B, 4B, 12B, all trained on the same corpus to the same token count. Students trained on 20B tokens of teacher logits (top-64 sparse logits, stored once per teacher). 8 distillation cells + controls; ~$10^{22}$ FLOPs total, weeks on a mid-size cluster.

**Per-cell tuning (mandatory).** Sweep $\tau \in \{1,2,4\}$ and $\alpha \in \{0.5, 1.0\}$ at 1/10 token budget; carry the best forward. Without this the result is uninterpretable.

**Control arms.**
1. **Supervised control** — same student, same 20B tokens, hard labels only. Establishes whether distillation helps at all.
2. **Optimization control** — for the best and worst teacher, rerun the student with $3\times$ tokens and cosine-restart schedule. If the U-shape flattens, the gap is optimization.
3. **Capacity control** — a student with the *same* architecture as the teacher, distilled from that teacher (self-distillation). Representational error is zero by construction; any residual KL is pure optimization error, giving the subtraction baseline for $\Gamma$.

**Deciding number.** Student held-out cross-entropy, in nats/token, as a function of teacher size. Define $\Delta = \mathrm{CE}_S(\text{12B teacher}) - \min_c \mathrm{CE}_S(c)$. Decision rule at 3 seeds, seed noise expected $\approx 0.005$ nats:

- $\Delta > 0.02$ nats and it **persists** under the optimization control → the gap is representational; teacher selection is a real hyperparameter and $c^\star(S)$ must be predicted.
- $\Delta > 0.02$ nats but **shrinks below 0.01** under $3\times$ tokens → the gap is an optimization artifact; the correct fix is schedule length, not teacher assistants.
- $\Delta < 0.01$ nats → non-monotonicity does not survive per-cell tuning at LM scale, and the CIFAR-era result does not transfer.

## 9. Key References

- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Jimmy Ba, Rich Caruana. *Do Deep Nets Really Need to be Deep?* NeurIPS, 2014. — arXiv:1312.6184
- **[Foundational]** Jang Hyun Cho, Bharath Hariharan. *On the Efficacy of Knowledge Distillation.* ICCV, 2019. — arXiv:1910.01348
- **[Foundational]** Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, Hassan Ghasemzadeh. *Improved Knowledge Distillation via Teacher Assistant.* AAAI, 2020. — arXiv:1902.03393
- **[SOTA]** Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander Alemi, Andrew Gordon Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- **[SOTA]** Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov. *Knowledge distillation: A good teacher is patient and consistent.* CVPR, 2022. — arXiv:2106.05237
- **[SOTA]** Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, Russ Webb. *Distillation Scaling Laws.* 2025. — arXiv:2502.08606
- **[Theory]** Aditya Krishna Menon, Ankit Singh Rawat, Sashank Reddi, Seungyeon Kim, Sanjiv Kumar. *A Statistical Perspective on Distillation.* ICML, 2021.
- **[Theory]** Hrayr Harutyunyan, Ankit Singh Rawat, Aditya Krishna Menon, Seungyeon Kim, Sanjiv Kumar. *Supervision Complexity and its Role in Knowledge Distillation.* ICLR, 2023. — arXiv:2301.12245
- **[Theory]** David Lopez-Paz, Léon Bottou, Bernhard Schölkopf, Vladimir Vapnik. *Unifying distillation and privileged information.* ICLR, 2016. — arXiv:1511.03643
- **[Applied]** Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf. *DistilBERT, a distilled version of BERT.* NeurIPS EMC^2 Workshop, 2019. — arXiv:1910.01108
- **[Applied]** Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. *Well-Read Students Learn Better: On the Importance of Pre-training Compact Models.* 2019. — arXiv:1908.08962
- **[Survey]** Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao. *Knowledge Distillation: A Survey.* International Journal of Computer Vision, 2021. — arXiv:2006.05525

## 10. Worked Example

Take a ResNet-8-style student on CIFAR-100 and three teachers: ResNet-14, ResNet-32, ResNet-110. The canonical shape of results in this setup (Mirzadeh et al., AAAI 2020, Table 1 region):

| Teacher | Teacher top-1 | Student top-1 (distilled) |
|---|---|---|
| — (hard labels) | — | ~61% |
| ResNet-14 | ~68% | ~62% |
| ResNet-32 | ~71% | ~62% |
| ResNet-110 | ~74% | ~61.8% |

Teacher accuracy rises 6 points; student accuracy moves by a few tenths, non-monotonically. Now make the obstruction visible with a calculation.

Assume distillation with $\alpha=1$: the student minimizes $\mathbb{E}[\mathrm{KL}(p_T\|p_S)]$. Decompose the achieved KL:

$$\underbrace{\mathbb{E}[\mathrm{KL}(p_T\|p_{\hat S})]}_{\text{measured}} = \underbrace{\mathbb{E}[\mathrm{KL}(p_T\|p_{S^\star})]}_{\Gamma:\ \text{representational}} + \underbrace{\mathbb{E}[\mathrm{KL}(p_T\|p_{\hat S})] - \mathbb{E}[\mathrm{KL}(p_T\|p_{S^\star})]}_{\text{optimization error}}.$$

Suppose the measured train-set KL against ResNet-110 is 0.30 nats and against ResNet-32 is 0.18 nats. The tempting reading: the big teacher is 0.12 nats "beyond the student's capacity". But run the capacity control — distill ResNet-110 into a *second ResNet-110*, where $\Gamma = 0$ exactly. Stanton et al. find this self-distillation run still leaves a substantial train-set KL and top-1 agreement far below 100%. If that residual is, say, 0.15 nats, then at most $0.30 - 0.15 = 0.15$ nats of the ResNet-110 number is representational — and that subtraction is only valid if optimization error is transferable across architectures, which nobody has shown.

So the observed quantity is a sum of two terms, only one of which the name "capacity gap" refers to, and there is no measurement that isolates it. The 0.2-point drop from ResNet-32 to ResNet-110 is within seed noise on CIFAR-100 ($\pm$0.3 points typical over 3 seeds) — meaning the headline effect at the scale where it is most cited is not individually significant, and the effect is only credible in aggregate across the sweep. That is the obstruction: an unidentifiable quantity, measured near the noise floor, at a scale chosen because the full sweep is affordable there and nowhere else.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*