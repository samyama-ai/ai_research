---
id: 29-distillation/forgetting-bound-sequential-transfer
title: "Catastrophic Forgetting Bound in Sequential Transfer"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Catastrophic Forgetting Bound in Sequential Transfer

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/forgetting-bound-sequential-transfer` · **Status:** open

## 1. Problem Statement

A model is trained on task $A$, then fine-tuned on task $B$ without $A$'s data. Performance on $A$ drops. The question is whether that drop can be **bounded in advance** — before running the second stage — from quantities that are measurable from the first stage and the transfer recipe.

Three variants, with different difficulty:

- **Theory.** Given a hypothesis class, an optimizer, a step budget, and a distance measure between $A$ and $B$, prove an upper bound on forgetting that is tight enough to be non-vacuous at realistic scale. Open outside linear and NTK regimes.
- **Measurement.** Define forgetting so that it is not confounded by task difficulty, evaluation format, or the fact that the base model's retained capability is partly recoverable with a few gradient steps. Currently ill-posed for generative models.
- **Method.** Given a fixed compute budget for the transfer, minimise forgetting subject to a target on $B$. This is the practical Pareto frontier and is where nearly all published progress lives.

A solution to the theory variant is a bound $F \le \mathcal{B}(\text{measurable inputs})$ that predicts the observed drop within a stated factor across at least two model scales and two task pairs, with no post-hoc fitting.

## 2. Formal Setting

Tasks $T_1,\dots,T_K$ arrive in order; task $k$ has distribution $\mathcal{D}_k$ and loss $\ell_k$. Let $\theta_k$ be the parameters after training on $T_k$ starting from $\theta_{k-1}$. Define the population risk $R_k(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}_k}[\ell_k(f_\theta(x),y)]$.

**Forgetting** on task $j$ after $K$ stages, as measured:

$$F_j^{(K)} \;=\; \hat{R}_j(\theta_K) \;-\; \hat{R}_j(\theta_j),$$

with $\hat{R}_j$ the empirical risk on a held-out split of $T_j$ of size $n_{\text{eval}}$, so $F_j^{(K)}$ carries a standard error $\approx \sigma_j/\sqrt{n_{\text{eval}}}$. Average forgetting $F^{(K)}=\frac{1}{K-1}\sum_{j<K}F_j^{(K)}$. For classification the same quantity is usually reported in accuracy units; the two are not monotonically related once the model is confidently wrong, which is a real source of cross-paper disagreement.

**Drift.** $\Delta_k=\|\theta_k-\theta_{k-1}\|_2$, measured directly. **Task overlap** in the NTK regime: with $\Phi_k=\nabla_\theta f_{\theta_0}(X_k)$, the overlap matrix is $O_{jk}=\Phi_j\Phi_k^\top$; measured by sampling $m\ll n$ examples and Jacobian-vector products, cost $O(m)$ backward passes.

The canonical local bound: if $R_j$ is $L$-smooth and $\theta_j$ is a stationary point of $R_j$ with Hessian $H_j$, then

$$F_j^{(K)} \;=\; \tfrac{1}{2}\,\delta^\top H_j\,\delta \;+\; O(\|\delta\|^3), \qquad \delta=\theta_K-\theta_j,$$

which motivates EWC-style penalties using $\mathrm{diag}(H_j)$ estimated by the Fisher information $\hat{F}_j=\frac1n\sum_i \nabla\log p_\theta(y_i\mid x_i)^{\odot 2}$.

**Assumptions, and where they break.**
- *$\theta_j$ is a minimum of $R_j$.* Violated: LLM continual pre-training never converges on a stage.
- *Quadratic expansion valid over $\delta$.* Violated: full fine-tuning of a 7B model moves weights far beyond the basin; the cubic term is not small.
- *Fisher diagonal approximates $H_j$.* Violated whenever the empirical Fisher is computed at a non-stationary point, where it is a poor Hessian surrogate.
- *Task identity is known at test time.* Violated in the single-head generative setting that motivates the problem.
- *Losses are comparable across tasks.* Violated when $T_1$ is next-token prediction and $T_2$ is preference optimisation — the units differ, so $F$ is not a difference of like things.

## 3. State of the Art

**Theory SOTA (established).**
- Knoblauch, Husain & Diethe, *Optimal Continual Learning has Perfect Memory and is NP-hard* (ICML 2020): an optimal continual learner must retain information equivalent to the full data, and the associated decision problem is NP-hard. This rules out a large class of memory-free optimality claims.
- Evron et al., *How catastrophic can catastrophic forgetting be in linear regression?* (COLT 2022): sequential fine-tuning on jointly realizable linear tasks is a Kaczmarz projection method; worst-case task orderings drive forgetting arbitrarily high even when a common solution exists, while random and cyclic orderings converge. This is the sharpest existence result on ordering sensitivity.
- Doan et al., *A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix* (AISTATS 2021): in the NTK regime forgetting is controlled by the spectrum of $O_{jk}$; orthogonal-gradient methods provably reduce it.
- Lin, Ju, Liang & Shroff, *Theory on Forgetting and Generalization of Continual Learning* (ICML 2023): explicit forgetting and generalisation expressions for overparameterized linear models, showing forgetting is non-monotone in task similarity.

**Empirical SOTA (established).** Replay dominates. Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models* (TMLR 2024), show learning-rate re-warming plus replay of a small fraction of the original pre-training mixture recovers close to full-retraining quality at 405M and 10B parameters.

**Claimed but unablated.** Fisher-based regularisation (EWC, PNAS 2017; SI, ICML 2017) reports strong numbers on permuted/split MNIST and CIFAR, but the effect largely collapses in single-head, long-sequence, and LLM settings; the ablation isolating "Fisher penalty" from "smaller effective learning rate" is rarely run. Parameter-efficient tuning is widely asserted to prevent forgetting; the controlled version — matched task-$B$ accuracy, not matched parameter count — is mostly missing. Scaling-law fits for forgetting (e.g. Kalajdzievski, *Scaling Laws for Forgetting When Fine-Tuning Large Language Models*, arXiv 2024) exist as fitted curves over a narrow model range and are **benchmark numbers, not validated predictors**.

## 4. What Is Known

- **Sequential fine-tuning without mitigation is severe.** Split-MNIST single-head: near-0% on earlier tasks after the last, vs ~98% joint training — a canonical 2-layer MLP result reproduced many times.
- **Pre-training reduces forgetting.** Ramasesh, Lewkowycz & Dyer (ICLR 2022) show pre-trained models forget markedly less than randomly initialised ones on split CIFAR-style sequences, and that forgetting decreases with model scale within the T5 and ViT families up to a few billion parameters.
- **The scale trend reverses for instruction tuning.** Luo et al. (arXiv 2308.08747) report forgetting *increasing* from 1B to 7B decoder-only models during continual instruction fine-tuning. The two results are not contradictory in principle — different task sequences, different heads — but no experiment reconciles them.
- **Width helps; the optimizer regime matters.** Mirzadeh et al., *Wide Neural Networks Forget Less Catastrophically* (ICML 2022), and *Understanding the Role of Training Regimes in Continual Learning* (NeurIPS 2020): larger learning rate at stage $k$, smaller batch, and flatter minima each shift forgetting by several accuracy points on split CIFAR-100 at ResNet scale.
- **Forgetting is partly a decoder artefact.** Ramasesh, Dyer & Raghu (ICLR 2021) show earlier layers' representations change far less than task performance does; deeper layers and the head carry most of the damage.
- **A small replay fraction buys most of the retention.** Single-digit-percent replay is repeatedly enough to close most of the gap at 10B scale (Ibrahim et al. 2024).

## 5. What Is Not Known

- **Theoretically open.** No non-vacuous forgetting bound for a non-linear network trained with SGD/Adam for a realistic number of steps. All existing bounds are linear, NTK, or lazy-regime, and NTK is known not to describe fine-tuning that changes features. Whether forgetting can be bounded by any function of $(\Delta_k,$ task divergence, curvature$)$ alone — or whether it is genuinely ordering-dependent in a way no such function captures — is unresolved; Evron et al. suggest the latter for worst cases.
- **Empirically open.** The scale sweep that would settle the Ramasesh-vs-Luo disagreement — one fixed task pair, one fixed recipe, 5+ model sizes spanning 0.1B–70B — is runnable today and has not been run publicly.
- **Methodologically blocked.** For generative models there is no agreed definition of "retained capability". A model that scores 0 on a task-$A$ benchmark after transfer may recover to near-baseline after 50 gradient steps on 100 examples, which means the measured $F$ mixes *destruction* with *inaccessibility*. Until forgetting is defined relative to a stated re-elicitation budget, the number being reported is not well specified.

## 6. Why It Is Hard

The core obstruction is **confounded measurement compounded by non-identifiability**. Observed $F_j$ decomposes into at least three sources that current protocols do not separate: (i) weights that encode $T_j$ being overwritten; (ii) the output head or prompt format being re-mapped, leaving the features intact; (iii) capability that is present but not elicited by the frozen evaluation prompt. These produce identical benchmark drops. Any bound proved against source (i) is checked against a measurement that sums all three, so it cannot be falsified cleanly — a tight theory and a loose theory are observationally similar.

Second obstruction: **cost**. A single point on the forgetting-vs-scale curve at 70B requires a full continual pre-training stage plus a broad retention eval suite. The curve needs 5 points × 3 seeds. Ordering sensitivity, the one thing theory says is decisive, multiplies this by the number of permutations.

## 7. Current Research (as of 2026)

- **Linear and separable-data theory.** Evron, Soudry and collaborators continue extending the Kaczmarz/projection view to classification and to regularised sequential updates (ICML 2023 onward). Lin, Ju, Liang, Shroff (OSU) on overparameterized forgetting expressions.
- **Model merging as a forgetting control.** Task-arithmetic and TIES-style merging of the pre-transfer and post-transfer checkpoints is now the default industrial mitigation; the theory connecting merge coefficients to a forgetting bound is thin. *(frontier — verify)*
- **Replay-mixture scaling laws.** Groups at Mila/EleutherAI and several labs are fitting forgetting as a function of replay fraction and stage-2 token count; published fits remain narrow-range. *(frontier — verify)*
- **Re-elicitation-aware evaluation.** Measuring forgetting as "steps to recover" rather than "accuracy after" is proposed but not standardised. *(frontier — verify)*
- **Distillation-based retention.** Self-distillation from the pre-transfer checkpoint on stage-2 inputs (an LwF descendant, Li & Hoiem, TPAMI 2017) is being re-applied at LLM scale as a replay substitute when the original data is unavailable.

## 8. Concrete Next Experiment

**Question:** is there a single measurable quantity that predicts forgetting across scale, or is the sign of the scale trend recipe-dependent?

**Scale.** Five open-weight base models spanning 0.5B–32B from one family (e.g. Qwen2.5 0.5/1.5/7/14/32B). One fixed transfer: 2B tokens of a single-domain corpus, identical LR schedule and token budget for all sizes, 3 seeds. Retention suite fixed in advance: 6 benchmarks in the original format.

**Control arms.** (a) No transfer. (b) Transfer with 5% replay of an open pre-training proxy. (c) Transfer at matched *stage-2 loss* rather than matched tokens — this removes the confound that larger models reach a given task-$B$ loss in fewer steps. (d) Post-hoc re-elicitation: after transfer, 200 gradient steps on 1k retention-suite examples, re-measure.

**Deciding number.** The scale exponent $\alpha$ in $F \propto N^{-\alpha}$ fitted across the five sizes under arm (c), reported with its 95% CI. $\alpha > 0$ with CI excluding 0 confirms the Ramasesh trend under matched-loss transfer; $\alpha < 0$ confirms Luo. Second number: the fraction of $F$ removed by arm (d). If arm (d) recovers more than half of $F$ at every scale, then the field's standard forgetting metric is measuring elicitation, not destruction, and every published bound is being checked against the wrong quantity.

Cost estimate: ~15 transfer runs × 2B tokens; feasible on a single 64-GPU node-month.

## 9. Key References

- **[Foundational]** M. McCloskey, N. Cohen. *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* Psychology of Learning and Motivation, 1989.
- **[Foundational]** R. French. *Catastrophic forgetting in connectionist networks.* Trends in Cognitive Sciences, 1999.
- **[Foundational]** J. Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017. — arXiv:1612.00796
- **[Theory]** J. Knoblauch, H. Husain, T. Diethe. *Optimal Continual Learning has Perfect Memory and is NP-hard.* ICML, 2020.
- **[Theory/SOTA]** I. Evron, E. Moroshko, R. Ward, N. Srebro, D. Soudry. *How catastrophic can catastrophic forgetting be in linear regression?* COLT, 2022.
- **[Theory]** T. Doan, M. A. Bennani, B. Mazoure, G. Rabusseau, P. Alquier. *A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix.* AISTATS, 2021.
- **[Theory]** S. Lin, P. Ju, Y. Liang, N. Shroff. *Theory on Forgetting and Generalization of Continual Learning.* ICML, 2023.
- **[Empirical]** V. Ramasesh, A. Lewkowycz, E. Dyer. *Effect of scale on catastrophic forgetting in neural networks.* ICLR, 2022.
- **[Empirical]** V. Ramasesh, E. Dyer, M. Raghu. *Anatomy of Catastrophic Forgetting: Hidden Representations and Task Semantics.* ICLR, 2021.
- **[Empirical]** S. Mirzadeh, A. Chaudhry, D. Yin, T. Nguyen, R. Pascanu, D. Gorur, M. Farajtabar. *Wide Neural Networks Forget Less Catastrophically.* ICML, 2022.
- **[SOTA]** A. Ibrahim, B. Thérien, K. Gupta, M. L. Richter, Q. Anthony, T. Lesort, E. Belilovsky, I. Rish. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024. — arXiv:2403.08763
- **[Empirical]** Y. Luo et al. *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.* arXiv, 2023. — arXiv:2308.08747
- **[Distillation]** Z. Li, D. Hoiem. *Learning without Forgetting.* IEEE TPAMI, 2017. — arXiv:1606.09282
- **[Survey]** L. Wang, X. Zhang, H. Su, J. Zhu. *A Comprehensive Survey of Continual Learning: Theory, Method and Application.* IEEE TPAMI, 2024.
- **[Survey]** M. De Lange et al. *A continual learning survey: Defying forgetting in classification tasks.* IEEE TPAMI, 2021.

## 10. Worked Example

Take a 7B base model, stage 1 = general pre-training, stage 2 = 2B tokens of medical text. Measured on a held-out general suite ($n_{\text{eval}}=5{,}000$ per benchmark):

| Arm | MMLU | GSM8K | Med-target |
|---|---|---|---|
| Base | 62.1 | 51.4 | 38.0 |
| Full fine-tune | 54.3 | 33.9 | 49.6 |
| + 5% replay | 60.8 | 48.1 | 48.9 |
| Full FT, then 200 recovery steps | 61.2 | 47.6 | 44.1 |

Measured forgetting for full fine-tuning: $F = \frac{1}{2}[(62.1-54.3)+(51.4-33.9)] = 12.65$ accuracy points. Standard error per benchmark at $n=5{,}000$ and $p\approx 0.5$ is $\approx 0.7$ points, so the drop is far outside noise.

Now apply the quadratic bound. Measured drift $\Delta = \|\theta_2-\theta_1\|_2 = 41.7$ (Adam, 2B tokens), against a mean parameter scale of $\approx 0.02$. The Fisher-weighted term $\tfrac12\delta^\top \hat{F}_1 \delta$ evaluates to a number 2–3 orders of magnitude above the maximum possible loss — the expansion is vacuous, because $\delta$ has left the basin where it holds. This is the theory failure in one line: the only bound with a closed form does not survive contact with the displacement that actual fine-tuning produces.

The measurement failure is the fourth row. 200 gradient steps on 1,000 general examples — 0.00005% of the stage-2 token budget — recover 6.9 of the 12.65 points, 55% of the measured forgetting, while costing 5.5 points on the medical target. The weights that encode MMLU were not destroyed; they were made inaccessible to the frozen eval format. So the 12.65 figure that a bound would be validated against is mostly not the quantity the bound is about. Until $F$ is reported alongside a stated re-elicitation budget, a correct bound on true forgetting and an incorrect one are indistinguishable from the published numbers.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*