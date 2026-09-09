---
id: 29-distillation/finetuning-delta-composability
title: "Fine-Tuning Delta Composability"
topic: 29-distillation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Fine-Tuning Delta Composability

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/finetuning-delta-composability` · **Status:** partially-solved

## 1. Problem Statement

A fine-tuning delta (task vector) is the parameter difference $\tau_t = \theta_t - \theta_0$ between a model fine-tuned on task $t$ and its pre-trained initialization. **Composability** asks: when does adding several deltas to one base model produce a model that is simultaneously good at all their tasks, with no access to the original training data?

Three variants, of different difficulty:

- **Measurement.** Given $\{\tau_t\}_{t=1}^T$, define a scalar that predicts composition loss *before* the merged model is evaluated. Currently there is no accepted such scalar; cosine similarity between deltas is used and is a weak predictor.
- **Method.** Find $f(\tau_1,\dots,\tau_T)$ — a merge operator — maximizing average held-out task accuracy under a fixed inference budget of one model. Partially solved: several operators beat naive summation reliably.
- **Theory.** Prove conditions on $\theta_0$, the task losses, and the fine-tuning trajectory under which $L_t(\theta_0 + \sum_s \lambda_s \tau_s)$ is within $\epsilon$ of $L_t(\theta_0 + \tau_t)$ for all $t$. Open beyond the linearized (NTK) regime.

Solving it means: a merge operator with a *predictive* guarantee, not a benchmark average.

## 2. Formal Setting

Base $\theta_0 \in \mathbb{R}^d$. Task $t$ has data distribution $\mathcal{D}_t$ and loss $L_t(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_t}[\ell(f_\theta(x),y)]$. Fine-tuning yields $\theta_t$; $\tau_t = \theta_t - \theta_0$.

**Merged model.** $\theta_\lambda = \theta_0 + \sum_{t=1}^T \lambda_t \tau_t$, $\lambda \in \mathbb{R}^T$ (or per-layer $\lambda \in \mathbb{R}^{T\times L}$).

**Normalized composition gap** — the quantity actually reported, measured on held-out test splits with the task-specific classification head or prompt held fixed:

$$G(\lambda) = \frac{1}{T}\sum_{t=1}^T \frac{A_t(\theta_t) - A_t(\theta_\lambda)}{A_t(\theta_t) - A_t(\theta_0)}$$

where $A_t$ is task accuracy. $G=0$ means composition is free; $G=1$ means the merge retained nothing. Reported "normalized accuracy" in the merging literature is $1-G$ up to sign conventions.

**Weight disentanglement** (Ortiz-Jimenez et al., 2023), the closest thing to a measurement primitive:

$$f(x;\theta_0 + \textstyle\sum_t \lambda_t \tau_t) = \sum_t g_t(x;\lambda_t \tau_t) \quad \text{for } x \in \mathcal{D}_t$$

i.e. off-task deltas must not change the function on $\mathcal{D}_t$'s support. Measured as a disentanglement error $\xi(\lambda_1,\lambda_2) = \sum_t \mathbb{E}_{x\sim\mathcal{D}_t}[\mathbb{1}\{f(x;\theta_0+\lambda_t\tau_t) \neq f(x;\theta_\lambda)\}]$ — a prediction-disagreement rate, so it needs unlabeled data from each $\mathcal{D}_t$.

**Interference** decomposes into two measured parts (Yadav et al., 2023): *sign conflict*, the fraction of coordinates $i$ where $\mathrm{sign}(\tau_{t,i})$ differs across $t$ among coordinates with $|\tau_{t,i}|$ above the $k$-th percentile; and *magnitude swamping*, $\|\tau_s\|_2 / \|\tau_t\|_2$ for $s \neq t$.

**Assumptions, and which fail.**
1. *Shared basin*: all $\theta_t$ lie in one linearly connected loss basin. Holds empirically for fine-tunes of the same $\theta_0$ (Neyshabur et al., 2020; Frankle et al., 2020); **fails** across different pre-training seeds, where permutation alignment is needed first.
2. *Local linearity*: $L_t$ is well approximated to first order along $\tau$. **Known violated** — nonlinear fine-tuning outperforms its own linearization on single tasks while composing worse, so the regime that composes is not the regime that performs.
3. *Task-vector additivity of representations*. **Violated** for tasks with overlapping input support; disentanglement error grows with support overlap.
4. *Identical tokenizer, architecture and $\theta_0$*. Violated whenever deltas come from independently released checkpoints — the common practical case.

## 3. State of the Art

**Established (reproduced across labs and architectures):**
- **Task arithmetic** (Ilharco et al., ICLR 2023). $\theta_0 + \lambda \sum_t \tau_t$ with one global $\lambda$ tuned on validation. Editing by negation ($-\lambda\tau$) reduces target-behavior metrics with small control-task damage. This is the baseline everything else is measured against.
- **Model soups / uniform averaging** (Wortsman et al., ICML 2022) — averaging same-task fine-tunes improves accuracy and OOD robustness at no inference cost. Reproduced widely; distinct from multi-task composition.
- **TIES-Merging** (Yadav et al., NeurIPS 2023): trim to top-$k$ magnitude, elect a sign, mean only the agreeing coordinates. Beats task arithmetic across T5, ViT, and (IA)$^3$ settings.
- **DARE** (Yu et al., ICML 2024): drop a random $p$ fraction of delta coordinates and rescale by $1/(1-p)$. At $p \le 0.9$ for SFT deltas of 7B–13B LLMs, single-task performance is largely preserved, and merging after DARE reduces interference.
- **Linearized fine-tuning** (Ortiz-Jimenez et al., NeurIPS 2023): fine-tuning in the tangent space raises weight disentanglement and improves multi-task merging accuracy on 8-task CLIP ViT benchmarks.

**Claimed but under-ablated:**
- **AdaMerging** (Yang et al., ICLR 2024) learns per-layer $\lambda$ by test-time entropy minimization on unlabeled test data. Large reported gains, but the comparison to a task-arithmetic baseline given the *same* unlabeled budget is thin, and entropy minimization is a known accuracy-inflating objective on in-distribution test sets.
- **LoRA composition** (LoraHub, Huang et al., COLM 2024; Zhang et al., NeurIPS 2023 for PEM arithmetic). Reported as competitive with few-shot ICL on BBH. Results are benchmark numbers on one instruction-tuned backbone; sensitivity to LoRA rank, initialization, and adapter placement is not systematically ablated.
- **Cross-checkpoint merging** of independently released community models. Widely practiced (mergekit); essentially no controlled evidence, since the base checkpoints differ.

## 4. What Is Known

- **Numbers, 8-task CLIP ViT-B/32 benchmark** (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN), the field's standard scale: individual fine-tuned average ≈ 90%; pre-trained zero-shot ≈ 48%; uniform weight averaging ≈ 65%; task arithmetic ≈ 69–70%; TIES ≈ 72–73%; AdaMerging (test-time, unlabeled) reported ≈ 80%. Composition gap $G$ therefore falls from about 0.6 (averaging) to about 0.25 (best reported) but does not approach 0.
- **Scaling helps.** The same merge operator gives a smaller gap on ViT-L/14 than ViT-B/32 — larger, more overparameterized bases compose better. Consistent across task-arithmetic and TIES papers.
- **Delta sparsity.** SFT deltas of 7B–13B LLMs tolerate 90%+ random coordinate removal with rescaling at near-zero single-task loss (DARE) — deltas are highly redundant. Continued-pretraining deltas do not tolerate this.
- **Sign conflict is real and load-bearing.** Removing sign-conflicting coordinates is the single largest contributor to TIES's gain over task arithmetic in its own ablation.
- **Linear mode connectivity of fine-tunes** from a shared $\theta_0$ holds; from different initializations it does not without permutation alignment (Git Re-Basin, Ainsworth et al., ICLR 2023).
- **Negation works asymmetrically.** Unlearning by delta negation degrades the target capability faster than it degrades control tasks, but the effect is often shallow — it suppresses outputs more than it removes the underlying computation.

## 5. What Is Not Known

- **Theoretically open.** No non-vacuous bound on $G(\lambda)$ outside the NTK/linearized regime. There is no theorem stating conditions on $(\theta_0, \{L_t\})$ under which $T$ deltas compose with gap $\le \epsilon$, and no known scaling law for $G$ in $T$ (is the gap linear in $T$, or does it saturate?).
- **Empirically open.** Whether the ViT-B/32 8-task findings transfer to $T \ge 20$ heterogeneous deltas on a 70B-parameter base. Runnable — costs GPU-days, not GPU-years — but no controlled study exists at that scale. Also open: whether merging beats a routed mixture of the same deltas at matched *memory*, not matched parameter count.
- **Methodologically blocked.** The measurement. Weight disentanglement requires unlabeled data from every $\mathcal{D}_t$ — exactly what the data-free premise excludes. There is no delta-only statistic validated as predicting $G$. Cosine similarity between deltas is reported to be near-orthogonal for almost all task pairs, so it does not discriminate between pairs that merge well and pairs that do not.

## 6. Why It Is Hard

**Non-identifiability of the delta.** The map from "capability acquired" to "delta" is many-to-one: optimizer state, LR schedule, seed and data order all change $\tau_t$ while leaving $A_t(\theta_t)$ fixed. Two deltas for the same task from different runs can have near-zero cosine similarity. So any statistic computed on $\tau$ alone is a statistic of a nuisance-heavy representation, and a merge operator can only see $\tau$.

**The benchmark measures the wrong thing.** Average accuracy over 8 image-classification tasks with disjoint label spaces and separate heads mostly measures whether the shared trunk stayed sane. It does not test the case people care about — composing *interfering* capabilities (a code delta and a safety delta) in one shared output space. A high number on the standard benchmark does not license a claim about composability in general.

**Compute cost of the negative result.** Establishing that a merge operator fails requires the full-fine-tune control arm for each of $T$ tasks, so cost is $O(T)$ full fine-tunes per data point — which is why $T$ stays at 8.

## 7. Current Research (as of 2026)

- **Delta compression as infrastructure**: sparsification plus low-bit quantization of deltas for multi-tenant serving (BitDelta-line work, MIT/Together). Composability becomes an operational question about the serving stack rather than only a merging one.
- **Learned merge coefficients** — per-layer, per-neuron, and localize-then-stitch approaches that solve a small optimization over a validation or unlabeled set. Active at UNC (Yadav/Bansal), Tsinghua, and Sakana AI (evolutionary model merging).
- **Tangent-space and Fisher-geometry merging**: making fine-tuning *produce* composable deltas rather than repairing them afterwards. EPFL/Google line following Ortiz-Jimenez.
- **Merging as a safety surface**: whether merging a benign delta into an aligned base restores unsafe behavior. Early results say yes, sometimes. *(frontier — verify)*
- **Delta interference scaling laws** — fitting $G(T)$ across model sizes. *(frontier — verify; no published fit at 2026-09)*

## 8. Concrete Next Experiment

**Question.** Does a delta-only statistic predict the composition gap, without any task data?

**Scale.** One base: Llama-3.1-8B. $T=12$ SFT deltas trained by the *same* recipe (identical LR, schedule, steps, seed) on 12 disjoint domains — code, math, medical QA, legal, 3 languages, tool-use, summarization, safety refusal, roleplay, SQL. ~12 × 4 GPU-hours on 8×H100. Then all 66 pairs and 20 random triples merged by task arithmetic at $\lambda$ tuned per-merge on 200 held-out examples.

**Control arms.** (a) Each single-task fine-tune, giving the $A_t(\theta_t)$ denominator of $G$. (b) A seed-replication arm: 3 additional deltas for one task under different seeds, merged pairwise — these must give $G \approx 0$, and their pairwise cosine similarity establishes the nuisance floor.

**Deciding number.** Spearman correlation $\rho$ between a candidate delta-only statistic (top-$k$ sign-conflict rate at $k=1\%$) and measured $G$, across the 66 pairs. $\rho \ge 0.7$ makes the measurement variant solved for same-recipe deltas and turns merge-operator design into an optimization problem. $\rho \le 0.3$ says the statistic is dominated by the seed nuisance identified in arm (b) and that no delta-only predictor of this family will work — a genuinely useful negative.

## 9. Key References

- **[Foundational]** Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi. *Editing Models with Task Arithmetic.* ICLR 2023. — arXiv:2212.04089
- **[Foundational]** Wortsman et al. *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy Without Increasing Inference Time.* ICML 2022. — arXiv:2203.05482
- **[SOTA]** Yadav, Tam, Choshen, Raffel, Bansal. *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS 2023. — arXiv:2306.01708
- **[SOTA]** Yu, Yu, Shen, Wang, Li, Tao. *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE).* ICML 2024. — arXiv:2311.03099
- **[Theory]** Ortiz-Jimenez, Favero, Frossard. *Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models.* NeurIPS 2023 (oral). — arXiv:2305.12827
- **[SOTA]** Yang, Wang, Shen, Liu, Guo, Wang, Tao. *AdaMerging: Adaptive Model Merging for Multi-Task Learning.* ICLR 2024. — arXiv:2310.02575
- **[Foundational]** Matena, Raffel. *Merging Models with Fisher-Weighted Averaging.* NeurIPS 2022. — arXiv:2111.09832
- **[Foundational]** Ainsworth, Hayase, Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR 2023. — arXiv:2209.04836
- **[Foundational]** Neyshabur, Sedghi, Zhang. *What is Being Transferred in Transfer Learning?* NeurIPS 2020. — arXiv:2008.11687
- **[SOTA]** Huang, Liu, Lin, Pang, Du, Lin. *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.* COLM 2024. — arXiv:2307.13269
- **[Survey]** Yadav, Choshen, Raffel, Bansal, et al. *Model merging surveys* (see "Model Merging in LLMs, MLLMs, and Beyond", 2024) — treat coverage as of 2024.

## 10. Worked Example

Two CLIP ViT-B/32 deltas: MNIST ($\tau_1$) and SVHN ($\tau_2$) — both digit recognition, overlapping semantics, disjoint input statistics.

Fine-tuned accuracies: MNIST 99.7%, SVHN 97.5%. Zero-shot: MNIST 48.3%, SVHN 31.6%.

Task arithmetic at $\lambda = 0.4$ gives roughly MNIST 98%, SVHN 92% (order-of-magnitude typical of published 2-task merges at this scale). Composition gap:

$$G = \tfrac{1}{2}\left[\frac{99.7-98}{99.7-48.3} + \frac{97.5-92}{97.5-31.6}\right] = \tfrac{1}{2}\left[0.033 + 0.083\right] \approx 0.058$$

Small. Now the obstruction. Measure $\cos(\tau_1,\tau_2)$: for CLIP ViT-B/32 fine-tunes it lands near 0.02–0.06 — indistinguishable from the value for MNIST vs. SUN397, a pair with *no* semantic overlap and a similarly small gap. And re-run the MNIST fine-tune with a different seed to get $\tau_1'$: $\cos(\tau_1,\tau_1')$ is also small, well under 0.2, even though $\tau_1$ and $\tau_1'$ encode the *same* function.

So the geometry says: same task, different seed ≈ different tasks ≈ orthogonal. The one statistic available without task data cannot separate "these compose because they are compatible" from "these compose because the base is overparameterized enough that almost anything composes at $T=2$". Push to $T=8$ and $G$ rises to ~0.25 while all pairwise cosines stay near zero — the statistic did not move, the outcome did. That is the measurement gap, made visible in four numbers.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*