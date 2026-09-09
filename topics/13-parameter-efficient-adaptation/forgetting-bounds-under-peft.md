---
id: 13-parameter-efficient-adaptation/forgetting-bounds-under-peft
title: "Catastrophic Forgetting Bounds Under PEFT"
topic: 13-parameter-efficient-adaptation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Catastrophic Forgetting Bounds Under PEFT

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/forgetting-bounds-under-peft` · **Status:** open

## 1. Problem Statement

Parameter-efficient fine-tuning (PEFT) restricts an update to a low-dimensional subspace — LoRA's rank-$r$ factors, an adapter bottleneck, a soft prompt. The folk claim is that this restriction *bounds* forgetting: a smaller update can damage less. The problem is to convert that claim into a statement with a proof or a falsification.

Three variants, with different difficulty:

- **Theory.** Given a pretrained $\theta_0$, an adaptation family $\mathcal{A}_p$ of intrinsic dimension $p$, and a target task, produce a non-vacuous upper bound on degradation of the pretraining distribution as a function of $p$ (or of rank $r$, or of $\|\Delta\theta\|$) — a bound that is tighter than the trivial smoothness bound and that separates PEFT from full fine-tuning at matched target-task loss.
- **Method.** Find a PEFT parameterization with a *certificate*: a computable quantity, available before or during adaptation, that predicts post-hoc forgetting to within a stated tolerance without evaluating the retained distribution.
- **Measurement.** Define forgetting so that it is not confounded with format drift, prompt sensitivity, or decoding changes. This variant is the one currently blocking the other two.

Solving it means: given $(\theta_0, r, \text{target loss } \varepsilon)$, predicting the retained-capability drop to within a factor of 2, without running the retained evaluation.

## 2. Formal Setting

Let $\theta_0 \in \mathbb{R}^d$ be pretrained weights, $\mathcal{D}_0$ the pretraining (or "retained") distribution, $\mathcal{D}_1$ the target. PEFT defines a map $\phi \mapsto \theta_0 + \Delta(\phi)$, $\phi \in \mathbb{R}^p$, $p \ll d$. For LoRA on matrix $W_\ell \in \mathbb{R}^{m\times n}$: $\Delta W_\ell = \tfrac{\alpha}{r} B_\ell A_\ell$ with $B_\ell \in \mathbb{R}^{m\times r}$, $A_\ell \in \mathbb{R}^{r\times n}$.

**Forgetting, as measured.** Two operational definitions, not equivalent:

$$F_{\text{nll}} = \mathbb{E}_{x\sim\mathcal{D}_0}\big[-\log p_{\theta_0+\Delta}(x)\big] - \mathbb{E}_{x\sim\mathcal{D}_0}\big[-\log p_{\theta_0}(x)\big] \quad \text{(nats/token)}$$

$$F_{\text{task}} = \tfrac{1}{K}\sum_{k=1}^K \big(\mathrm{acc}_k(\theta_0) - \mathrm{acc}_k(\theta_0+\Delta)\big) \quad \text{(accuracy points)}$$

$F_{\text{nll}}$ is measured on a held-out shard of a pretraining-like corpus (in practice: RedPajama/Dolma/The Pile slices, never the true pretraining mixture — it is not public for any frontier model). $F_{\text{task}}$ is measured on a fixed benchmark set $\{k\}$ (typically HellaSwag, ARC-c, WinoGrande, MMLU) with a *frozen* prompt template and decoding config.

**The only rigorous bound currently available.** If $L_0(\theta) = \mathbb{E}_{\mathcal{D}_0}[\ell]$ is $H$-smooth with $\|\nabla L_0(\theta_0)\| \le G$ on the segment,

$$F_{\text{nll}} \le G\|\Delta\theta\|_2 + \tfrac{H}{2}\|\Delta\theta\|_2^2 .$$

This is what PEFT-bounds-forgetting arguments reduce to. Section 10 shows it is ~$10^2$–$10^3\times$ vacuous at 7B scale.

**Assumptions, and their violation status.**
- *Smoothness with a small, measurable $H$ along the update path.* Violated: transformer loss landscapes are non-smooth, and $H$ is estimated only by finite-difference probes.
- *$\mathcal{D}_0$ is samplable.* Violated for every closed and most open-weight models; the retained set is a proxy.
- *Forgetting is a function of $\|\Delta\theta\|$.* Violated: Panigrahi et al. (ICML 2023) show sparse, highly localized updates — tiny norm — can move task behavior a long way, and the converse (large-norm, behaviorally inert directions) also holds.
- *Rank $r$ controls update magnitude.* False as stated: $\|\Delta W\|_F$ is set by training steps and learning rate, not by $r$. Rank constrains the *subspace*, not the *distance*.

## 3. State of the Art

**Theory SOTA.** Exact forgetting analyses exist only for linear/NTK regimes. Evron et al. (COLT 2022) give tight forgetting rates for sequential linear regression, including a $\Theta(1)$ worst case under adversarial task ordering and $O(1/T)$ under random ordering. Doan et al. (AISTATS 2021) bound forgetting by the spectrum of an NTK task-overlap matrix. Lin, Ju, Liang & Shroff (ICML 2023) give a closed-form forgetting/generalization decomposition for overparameterized linear models. **None of these is stated for a rank-constrained update**; adapting them to LoRA's factored, non-convex parameterization is unpublished as of 2026. Zeng & Lee (ICLR 2024) characterize LoRA's *expressive* power (rank needed to represent a target function) — an approximation result, not a forgetting result, and it is regularly miscited as one.

**Empirical SOTA.** Biderman et al. (TMLR 2024), *LoRA Learns Less and Forgets Less*, is the reference measurement: Llama-2-7B/13B, code and math domains, continued pretraining and instruction tuning, ranks 16–256. LoRA forgets less than full fine-tuning at matched target gain — **established** and independently consistent with several follow-ups. What is **claimed but unablated** in the wider literature: that rank monotonically controls forgetting. Biderman's own sweep shows the rank→forgetting curve is weak and non-monotone once target-task loss is matched.

**Benchmark-number-only results.** O-LoRA (Wang et al., EMNLP Findings 2023) reports low forgetting on a 15-task text-classification stream by constraining new LoRA subspaces orthogonal to old ones. The number is real; there is no ablation isolating orthogonality from the implied capacity reduction, and it has not been run on generative pretraining-scale retention.

## 4. What Is Known

- **Full FT forgets more than LoRA at matched target performance.** Biderman et al. 2024: Llama-2-7B, ~20B tokens of Starcoder-Python. On the average of HellaSwag/ARC-challenge/WinoGrande, full fine-tuning drops several accuracy points where LoRA at $r=16$–$256$ stays within ~1 point. Scale: 7B and 13B, single-domain.
- **Forgetting is monotone in fine-tuning tokens, not in rank.** Same study: the token axis dominates; ranks 16 through 256 are close together on retention.
- **Scale reduces forgetting.** Ramasesh, Dyer & Raghu (ICLR 2022) show pretrained-model forgetting on sequential task streams falls sharply with model size across T5 scales — a robust regularity, measured on classification streams, not on open-ended generation.
- **Some "forgetting" is recoverable and therefore not destruction.** Kotha, Springer & Raghunathan (ICLR 2024) show fine-tuned LMs' lost capabilities can be substantially restored by conditioning changes alone, i.e. the model implicitly re-infers the task. This directly implies $F_{\text{task}}$ overstates parameter-level damage.
- **LoRA updates are not a low-norm version of full FT.** Shuttleworth et al. (2024) find "intruder dimensions": singular vectors in the LoRA-merged weights with no counterpart in the full-FT solution, and these correlate with worse out-of-distribution retention at equal target accuracy. Scale: RoBERTa and Llama-class models.
- **Alignment tax is a forgetting instance.** Ouyang et al. (2022) report measurable public-NLP regressions from RLHF on GPT-3-class models, mitigated by mixing pretraining gradients — evidence that rehearsal, not update size, is the effective lever.

## 5. What Is Not Known

- **Theoretically open.** No non-vacuous forgetting bound for rank-constrained updates on non-linear networks. Specifically: no theorem of the form $F \le f(r, \|\Delta\|, \text{task overlap})$ that beats the smoothness bound. Nor any separation theorem proving PEFT *must* forget less than full FT at matched target loss — the empirical gap may be an optimization artifact, not a capacity property.
- **Empirically open.** The rank-vs-forgetting sweep at matched target loss, above 13B and on a genuinely pretraining-like retained set, has not been run publicly. Also unrun: whether the LoRA advantage survives when full FT is given equal-norm early stopping or weight decay to the pretrained point.
- **Methodologically blocked.** $F_{\text{task}}$ conflates parameter damage with prompt-format drift (Kotha et al.); $F_{\text{nll}}$ requires a pretraining distribution nobody outside the lab can sample. Until a retention metric is defined that is invariant to recoverable conditioning effects, "forgetting" numbers are not comparable across papers.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability of the retained distribution**.

1. *The evaluation does not measure what it names.* A drop in ARC-challenge accuracy after instruction tuning can come from the model answering in a different format. Kotha et al. show a large fraction of it reverses under re-conditioning with no parameter change. So $F_{\text{task}}$ is an upper bound on damage of unknown tightness.
2. *No ground truth for $\mathcal{D}_0$.* $F_{\text{nll}}$ on a proxy corpus estimates forgetting on a distribution the model was not trained on. Two papers using Pile vs. Dolma slices are not measuring the same quantity, and the difference is not reported.
3. *Non-identifiability of "the same target performance."* Comparing LoRA to full FT requires matching target loss; there are many $(\text{lr}, \text{steps}, \text{schedule})$ settings at the same target loss with different $\|\Delta\theta\|$. The comparison has a free parameter that dominates the effect being measured.
4. *Cost.* A clean sweep — 5 ranks × 3 token budgets × 2 methods × 3 seeds at 7B — is ~90 runs of tens of billions of tokens. That is the reason it is unrun, but it is the least fundamental of the four.

## 7. Current Research (as of 2026)

- **Subspace-orthogonality methods** (O-LoRA lineage; groups at Alibaba DAMO, ICT/CAS) — constrain new adapters orthogonal to previously used subspaces. Open question: whether orthogonality in parameter space implies non-interference in function space. It does not, in general.
- **Spectral diagnostics of merged adapters** — the intruder-dimension line (MIT CSAIL and collaborators) is the most promising *certificate* candidate: a quantity computed from $\Delta W$ alone that correlates with retention. Whether it predicts, rather than post-hoc correlates, is untested *(frontier — verify)*.
- **Rehearsal-free retention via gradient projection onto a Fisher-estimated null space** — descendants of EWC and Orthogonal Gradient Descent (Farajtabar et al., AISTATS 2020) applied inside the LoRA subspace *(frontier — verify)*.
- **Model-merging as an implicit forgetting control** — treating $\theta_0$ interpolation (e.g. weight averaging with the base model) as a post-hoc knob; widely used industrially, thinly characterized theoretically.

## 8. Concrete Next Experiment

**Question:** at matched target-task loss, does rank $r$ predict forgetting at all, once update norm is controlled?

**Scale.** Llama-3.1-8B (open weights, open-ish data lineage). Target: 2B tokens of a single domain (Python from The Stack v2). Arms: $r \in \{4, 16, 64, 256\}$, plus full FT. Three seeds. ~15 runs × 2B tokens ≈ 5k A100-hours — feasible for one academic cluster-month.

**Control arm.** Full fine-tuning with an explicit $\ell_2$ trust region to $\theta_0$, tuned so that $\|\Delta\theta\|_2$ matches the $r=16$ LoRA run to within 5%. This is the arm missing from every published comparison: it separates "low rank" from "small update."

**Retention set.** 500M held-out tokens of Dolma-v1.7 (a documented public approximation to the Llama pretraining mixture), reported as $F_{\text{nll}}$ in nats/token, plus $F_{\text{task}}$ reported *twice* — once with the base model's few-shot template and once with a per-arm best-of-4 template, to bound the format-drift confound.

**The deciding number.** Spearman correlation $\rho(r, F_{\text{nll}})$ across arms at matched target loss (target loss matched to $\pm 0.01$ nats by early stopping). If $|\rho| < 0.3$ with the norm-matched control inside the LoRA range, the rank-bounds-forgetting hypothesis is dead and the field should reparameterize the question around $\|\Delta\theta\|$ and update geometry. If $\rho > 0.7$, there is a real rank effect worth trying to prove a bound for.

## 9. Key References

- **[Foundational]** McCloskey & Cohen. *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* Psychology of Learning and Motivation, 1989.
- **[Foundational]** Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017. — arXiv:1612.00796
- **[Foundational]** Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[SOTA]** Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[SOTA]** Shuttleworth et al. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* 2024. — arXiv:2410.21228
- **[Theory]** Evron et al. *How catastrophic can catastrophic forgetting be in linear regression?* COLT, 2022. — arXiv:2205.09588
- **[Theory]** Doan et al. *A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix.* AISTATS, 2021. — arXiv:2010.04003
- **[Theory]** Lin, Ju, Liang & Shroff. *Theory on Forgetting and Generalization of Continual Learning.* ICML, 2023. — arXiv:2302.05836
- **[Theory]** Zeng & Lee. *The Expressive Power of Low-Rank Adaptation.* ICLR, 2024. — arXiv:2310.17513
- **[Measurement]** Kotha, Springer & Raghunathan. *Understanding Catastrophic Forgetting in Language Models via Implicit Inference.* ICLR, 2024. — arXiv:2309.10105
- **[Measurement]** Ramasesh, Lewkowycz & Dyer. *Effect of scale on catastrophic forgetting in neural networks.* ICLR, 2022.
- **[Measurement]** Panigrahi et al. *Task-Specific Skill Localization in Fine-tuned Language Models.* ICML, 2023. — arXiv:2302.06600
- **[Survey]** Wang et al. *A Comprehensive Survey of Continual Learning: Theory, Method and Application.* IEEE TPAMI, 2024. — arXiv:2302.00487
- **[Survey]** Han et al. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR, 2024. — arXiv:2403.14608

## 10. Worked Example

**Setup.** Llama-2-7B, hidden size $n = 4096$, 32 layers, LoRA $r=16$ on $W_q, W_v$. Trainable parameters: $32 \times 2 \times (2 \cdot 4096 \cdot 16) = 8.39$M, i.e. $p/d \approx 1.2\times10^{-3}$.

**Measured quantities.** A pretrained $4096\times4096$ attention matrix has entries of std $\approx 0.02$, so $\|W\|_F \approx 4096 \times 0.02 \approx 82$. A converged LoRA update on a code corpus has $\|\Delta W_\ell\|_F \approx 1.5$ (about 2% relative). Across all 64 adapted matrices:

$$\|\Delta\theta\|_2 = \sqrt{64 \times 1.5^2} = 12.0 .$$

**Plug into the smoothness bound.** With $G = \|\nabla L_0(\theta_0)\|_2 \approx 0.5$ (finite-difference estimate on a held-out Pile shard; order-1 is typical) and even the generous $H \approx 0$:

$$F_{\text{nll}} \le 0.5 \times 12.0 = 6.0 \ \text{nats/token}.$$

**Compare to the actual measurement.** Retained-corpus loss moves from $\approx 1.85$ to $\approx 1.87$ nats/token — $F_{\text{nll}} \approx 0.02$. The bound is **300× loose**. It is also vacuous in absolute terms: 6 nats/token is worse than a uniform distribution over a 50k vocabulary would be at 10.8 nats, but far worse than any usable model.

**Where the obstruction shows.** The bound fails not because $G$ or $H$ are badly estimated, but because it assumes the update direction is aligned with $\nabla L_0$. It is not: the measured cosine between $\Delta\theta$ and $\nabla L_0(\theta_0)$ is near zero ($\sim10^{-2}$), so the first-order term nearly cancels. The real forgetting is a second-order, curvature-weighted quantity $\tfrac12 \Delta\theta^\top \nabla^2 L_0 \, \Delta\theta$ restricted to the LoRA subspace — and *that* requires the Hessian spectrum of the pretraining loss restricted to a rank-16 subspace, which nobody has measured at 7B. Rank enters the true expression only through which subspace is selected, not through the norm. That is exactly why matching $\|\Delta\theta\|$ (Section 8's control arm) is the experiment that decides the question.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*