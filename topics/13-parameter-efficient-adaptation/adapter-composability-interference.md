---
id: 13-parameter-efficient-adaptation/adapter-composability-interference
title: "Composability of Independently Trained Adapters"
topic: 13-parameter-efficient-adaptation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Composability of Independently Trained Adapters

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/adapter-composability-interference` · **Status:** open

## 1. Problem Statement

Given a frozen base model and $n$ adapters (LoRA, bottleneck adapters, prefix vectors) each trained **independently** on its own task, produce a single artifact that performs all $n$ tasks at close to each adapter's solo accuracy — without joint retraining and without paying $n\times$ inference cost.

Three variants, routinely conflated:

- **Measurement.** Define an interference score that is comparable across tasks, adapter ranks, and base models, and that separates *representational* conflict (the tasks want incompatible features) from *parameterization* conflict (the adapters collide in weight space but the tasks do not).
- **Method.** Find a composition operator $C(\Delta_1,\dots,\Delta_n)$ — sum, merge, route, or fuse — that maximizes worst-task retention under a fixed parameter and latency budget.
- **Theory.** State conditions on $(\theta_0, \Delta_i, \mathcal{D}_i)$ under which additive composition is *provably* lossless to first order, and prove they are or are not satisfiable by independent training.

Solved would mean: for $n=10$ non-trivial tasks on a $\geq$7B base, a training-free composition retains $\geq 95\%$ of each adapter's solo score on **every** task, with the failure cases predicted in advance by a cheap statistic computed from the adapters alone.

## 2. Formal Setting

Base parameters $\theta_0 \in \mathbb{R}^d$, frozen. Adapter $i$ is a low-rank update to selected weight matrices: for matrix $W_0 \in \mathbb{R}^{m\times k}$, $\Delta W_i = \tfrac{\alpha}{r} B_i A_i$ with $B_i\in\mathbb{R}^{m\times r}$, $A_i\in\mathbb{R}^{r\times k}$, $r \ll \min(m,k)$. Flattened, $\Delta_i \in \mathbb{R}^d$.

Task $i$ has distribution $\mathcal{D}_i$ and metric $s_i(\theta) \in [0,1]$ (accuracy, exact match, pass@1) — *measured on a held-out split of $\ge 1000$ examples, fixed decoding, three seeds*.

**Solo score** $s_i^\star = s_i(\theta_0 + \Delta_i)$. **Base score** $s_i^0 = s_i(\theta_0)$.

**Normalized retention** under composition $C$:
$$R_i(C) = \frac{s_i(C) - s_i^0}{s_i^\star - s_i^0}$$
Reported only for tasks where $s_i^\star - s_i^0 \geq 0.05$; below that the ratio is numerically unstable and is the main source of inflated merging results in the literature.

**Interference** is the retention deficit $I_i(C) = 1 - R_i(C)$, and the headline number is the worst case $I_{\max} = \max_i I_i(C)$, not the mean — means hide the task that broke.

Composition families:
- additive / task arithmetic: $C = \theta_0 + \lambda\sum_i \Delta_i$;
- sparsified-and-signed (TIES, DARE): $C = \theta_0 + \lambda\sum_i \gamma_i \odot \Delta_i$ with $\gamma_i$ a trim/sign mask;
- learned scalars (LoraHub): $\lambda \in \mathbb{R}^n$ fit by gradient-free search on a few-shot set;
- routed (AdapterFusion, MoLE): per-token gates $g_i(x)$, so no single weight vector exists.

**Weight disentanglement** (Ortiz-Jimenez et al., 2023): the model is disentangled if $f(x;\theta_0+\sum_i\lambda_i\Delta_i) = f(x;\theta_0+\lambda_i\Delta_i)$ for $x \in \mathcal{D}_i$. Measured by the disentanglement error on a $\lambda$ grid.

**Subspace overlap**, the cheap predictor candidate: for layer $\ell$, principal angles between $\mathrm{col}(B_i)$ and $\mathrm{col}(B_j)$, summarized as $\rho_{ij}^{(\ell)} = \tfrac{1}{r}\|U_i^\top U_j\|_F^2 \in [0,1]$ with $U$ orthonormal bases.

Assumptions and their status:
- *Linearity of the loss in $\Delta$ near $\theta_0$* — holds only in the NTK/tangent regime; **violated** for full nonlinear LoRA fine-tuning, which is why linearized fine-tuning merges better.
- *Adapters occupy near-orthogonal subspaces* — **violated**: independently trained LoRAs concentrate on the same top singular directions of $W_0$.
- *Solo scores are the right ceiling* — **violated** when tasks are positively transferring; retention can exceed 1.
- *Scale invariance of $\lambda$* — **violated**: the optimal $\lambda$ shrinks roughly as $1/n$, so a single fixed coefficient is not a fair control.

## 3. State of the Art

**Established (ablated, reproduced):**
- Task arithmetic on CLIP ViT-B/32 and ViT-L/14 across 8 image tasks: a single scaled sum improves over base substantially but loses accuracy relative to individual fine-tunes; the gap widens with $n$ (Ilharco et al., ICLR 2023).
- TIES-Merging (trim to top-20% magnitude, elect sign, disjoint mean) beats plain averaging and task arithmetic across PEFT and full fine-tuning settings (Yadav et al., NeurIPS 2023).
- DARE: dropping 90–99% of delta parameters and rescaling by $1/(1-p)$ leaves single-task performance nearly unchanged for supervised fine-tuning deltas, and improves merging (Yu et al., ICML 2024). The rescaling identity is exact in expectation; the empirical claim is that variance does not matter at these scales.
- Linearized (tangent-space) fine-tuning merges better than nonlinear fine-tuning, and weight disentanglement — not low task-gradient similarity per se — is the mechanism (Ortiz-Jimenez et al., NeurIPS 2023).

**Claimed but unablated / benchmark-only:**
- LoraHub's few-shot composition of 20 upstream LoRAs matching few-shot ICL on BBH (Huang et al., COLM 2024) is a benchmark number on one base (Flan-T5-large) and one suite; the ablation separating "composition works" from "any 20-way coefficient search overfits the 5-shot set" is not reported at scale.
- Routed methods (AdapterFusion, EACL 2021; mixtures of LoRA experts) report gains but pay $O(n)$ parameter storage and added latency, so they answer a different question than merging.
- DARE's near-lossless drop rates degrade sharply for adapters trained from scratch on distant tasks; the boundary is stated qualitatively, not as a rule.

## 4. What Is Known

- **Scaling of interference with $n$.** On the standard 8-task CLIP ViT-B/32 benchmark, merged-model average accuracy sits roughly 10–15 points below the average of individually fine-tuned models, and the deficit grows monotonically as tasks are added. Scale: ViT-B/32 and ViT-L/14, 8 tasks.
- **Sign conflict is a real and large effect.** TIES attributes most of the merging loss to (i) redundant low-magnitude parameters and (ii) opposite-sign updates on shared coordinates; removing both recovers several points across 7–11 task suites at T5-base/large and ViT scale.
- **Deltas are extremely redundant.** 90% (often 99%) of SFT delta entries can be zeroed with rescaling at little cost, at 7B–13B (WizardLM/WizardMath-class checkpoints). Redundancy is necessary for merging to work but not sufficient.
- **LoRA solutions are not equivalent to full fine-tuning solutions.** LoRA introduces "intruder dimensions" — singular vectors near-orthogonal to the pretrained spectrum — that full fine-tuning does not produce, and these correlate with worse behavior outside the target task (Shuttleworth et al., 2024). Scale: RoBERTa and LLaMA-family, ranks 1–64.
- **LoRA forgets less than full fine-tuning** at matched target performance (Biderman et al., TMLR 2024, LLaMA-2-7B/13B, code and math). This is why composition is attractive and also why LoRA deltas are lower-signal to merge.
- **Rank matters non-monotonically.** Higher rank raises solo score and raises overlap $\rho_{ij}$; there is no published rank-vs-interference curve at fixed solo score.

## 5. What Is Not Known

- **Theoretically open.** No necessary-and-sufficient condition on $(\Delta_i)$ for additive composition to be lossless beyond the tangent-space regime. Whether independent training can be *constrained* (orthogonality penalties, fixed random $A$) to guarantee bounded $I_{\max}$ without seeing the other tasks is unproven either way.
- **Empirically open.** No systematic sweep of $I_{\max}$ against $n \in \{2,\dots,32\}$, rank, and base scale (1B → 70B) with matched solo scores. Every merging paper fixes $n$ and varies the method. The runs are affordable; nobody has published them.
- **Methodologically blocked.** Interference is not separated from (a) the $\lambda$ shrinkage confound, (b) format/prompt drift after merging, (c) tasks with tiny $s_i^\star - s_i^0$. Without that separation, "merging works" and "the benchmark rewards base-model competence" are indistinguishable. No accepted cheap predictor of pairwise interference exists — $\rho_{ij}$ is the obvious candidate and its correlation with $I_{ij}$ has not been reported.

## 6. Why It Is Hard

**Non-identifiability of the low-rank factorization.** $B_iA_i$ is invariant to $B \mapsto BM$, $A \mapsto M^{-1}A$ for any invertible $M \in \mathbb{R}^{r\times r}$. Two adapters representing the *same* function can present arbitrarily different factors, so any composition operator defined on $(A,B)$ rather than on the product is measuring gauge, not content. Concatenation-style composition inherits this directly.

**Confounded measurement.** The merging coefficient $\lambda$ is tuned on validation data in most papers; a merged model with tuned $\lambda$ is compared against solo adapters with untuned $\lambda$. Part of the reported gap — sign unknown — is tuning budget, not interference.

**Absent ground truth.** There is no reference "correct" multi-task solution to compare against: the jointly trained multi-task adapter is itself subject to task-weighting choices, so the ceiling $s_i^\star$ is a proxy, and it moves.

## 7. Current Research (as of 2026)

- Sparsify-then-merge lines (TIES, DARE, and successors) remain the strongest training-free family; work continues on learned masks rather than magnitude heuristics.
- LoRA libraries with retrieval-based routing — building a bank of task LoRAs and selecting per input (Ostapenko et al., ICML 2024) — trade storage for interference.
- Tangent-space and linearization approaches, pushing weight disentanglement as the design target rather than a post-hoc diagnostic.
- Orthogonality-constrained PEFT: training adapters in mutually orthogonal subspaces by construction so sums commute. *(frontier — verify: results are mostly at ViT and <3B scale.)*
- Subspace-geometry diagnostics of merged models, following the intruder-dimension result. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Does pairwise subspace overlap $\rho_{ij}$ predict pairwise interference $I_{ij}$?

**Scale.** One base: Llama-3.1-8B. $n = 12$ tasks with clean, disjoint evaluations (GSM8K, HumanEval, a legal QA set, a clinical QA set, three languages of translation, two classification sets, summarization, SQL, function calling). Train 12 LoRAs at $r = 16$ on attention + MLP projections, each to within 1 point of its own converged solo score. Cost: 12 LoRA runs $\approx$ 200–400 A100-hours total; 66 pairwise merges + 12 solo evals $\approx$ 100 GPU-hours of inference.

**Arms.**
1. Solo adapter (ceiling).
2. **Control:** merge task $i$ with a *rank-matched random* $\Delta_j$ of identical Frobenius norm and layer placement. This is the arm the literature omits, and it separates "another task interfered" from "any perturbation of that magnitude degrades task $i$".
3. Additive merge, $\lambda$ swept over $\{0.3,\dots,1.0\}$ per pair, best-$\lambda$ reported — same tuning budget as the solo arm.

**Deciding number.** Spearman correlation between $\bar\rho_{ij}$ (overlap averaged over layers, weighted by $\|\Delta\|_F$) and $I_{ij}$ across the 66 pairs. $|\rho_s| \geq 0.6$ makes overlap a usable a priori interference predictor and turns adapter-library routing into a solved retrieval problem. $|\rho_s| \le 0.3$ falsifies the weight-space-collision story and forces interference to be defined on function outputs, not parameters.

## 9. Key References

- **[Foundational]** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. — arXiv:2106.09685
- **[Foundational]** Houlsby, Giurgiu, Jastrzebski, Morrone, de Laroussilhe, Gesmundo, Attariyan, Gelly. *Parameter-Efficient Transfer Learning for NLP.* ICML 2019. — arXiv:1902.00751
- **[Foundational]** Pfeiffer, Kamath, Rücklé, Cho, Gurevych. *AdapterFusion: Non-Destructive Task Composition for Transfer Learning.* EACL 2021. — arXiv:2005.00247
- **[SOTA]** Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi. *Editing Models with Task Arithmetic.* ICLR 2023. — arXiv:2212.04089
- **[SOTA]** Yadav, Tam, Choshen, Raffel, Bansal. *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS 2023. — arXiv:2306.01708
- **[SOTA]** Yu, Yu, Yu, Huang, Li. *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch.* ICML 2024. — arXiv:2311.03099
- **[SOTA]** Ortiz-Jimenez, Favero, Frossard. *Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models.* NeurIPS 2023. — arXiv:2305.12827
- **[SOTA]** Huang, Liu, Lin, Pang, Du, Lin. *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.* COLM 2024. — arXiv:2307.13269
- **[SOTA]** Wortsman, Ilharco, Gadre, Roelofs, Gontijo-Lopes, Morcos, Namkoong, Farhadi, Carmon, Kornblith, Schmidt. *Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time.* ICML 2022. — arXiv:2203.05482
- **[SOTA]** Zhang, Chen, Liu, Yang, Zhu, Sun, Yang, Wang. *Composing Parameter-Efficient Modules with Arithmetic Operations.* NeurIPS 2023. — arXiv:2306.14870
- **[SOTA]** Ostapenko, Su, Ponti, Charlin, Le Roux, Caccia, Sordoni. *Towards Modular LLMs by Building and Reusing a Library of LoRAs.* ICML 2024. — arXiv:2405.11157
- **[Analysis]** Shuttleworth, Andreas, Torralba, Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* 2024. — arXiv:2410.21228
- **[Analysis]** Biderman, Portes, Ortiz, Paul, Greengard, Jennings, King, Havens, Chiley, Frankle, Blakeney, Cunningham. *LoRA Learns Less and Forgets Less.* TMLR 2024. — arXiv:2405.09673
- **[Survey]** Lialin, Deshpande, Rumshisky. *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning.* 2023. — arXiv:2303.15647

## 10. Worked Example

Two LoRAs on Llama-3.1-8B, $r=16$, $\alpha=32$, applied to $W_q, W_v, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$.

- Adapter A: GSM8K math. $s_A^0 = 0.52$, $s_A^\star = 0.71$.
- Adapter B: SQL generation (Spider execution match). $s_B^0 = 0.41$, $s_B^\star = 0.68$.

Merge $\theta_0 + \lambda(\Delta_A + \Delta_B)$, $\lambda = 0.7$. Illustrative observed outcome, consistent with the reported 8-task CLIP pattern scaled to two tasks: $s_A = 0.66$, $s_B = 0.55$.

$$R_A = \frac{0.66-0.52}{0.71-0.52} = 0.74, \qquad R_B = \frac{0.55-0.41}{0.68-0.41} = 0.52$$

So $I_{\max} = 0.48$ — nearly half of SQL's gain destroyed — while the *mean* accuracy across the two tasks, $0.605$, is comfortably above the base mean $0.465$ and would be reported as "merging works".

Now the obstruction. Run the control: replace $\Delta_B$ with a random rank-16 delta of the same per-layer Frobenius norm. Suppose $s_A$ drops to $0.67$. Then $R_A = 0.79$ under a *content-free* perturbation, versus $0.74$ under the real adapter. Only $0.05$ of the $0.26$ deficit is attributable to task conflict; the remaining $0.21$ is the model's sensitivity to a perturbation of that magnitude, which the correct $\lambda$ would partly absorb.

Meanwhile the diagnostic: measure $\bar\rho_{AB}$. Because both adapters are trained from the same $\theta_0$ and gradients concentrate on the same dominant singular directions of $W_0$, $\bar\rho_{AB}$ typically lands well above the $\approx r/\min(m,k) \approx 16/4096 \approx 0.004$ expected for random 16-dimensional subspaces — but there is no published curve mapping that number onto $I_{\max}$. The experiment in §8 is precisely the missing curve: without it, every merging result is a benchmark number whose failure mode cannot be predicted before the merge is run and evaluated.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*