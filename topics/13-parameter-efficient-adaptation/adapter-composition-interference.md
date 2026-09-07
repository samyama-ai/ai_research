---
id: 13-parameter-efficient-adaptation/adapter-composition-interference
title: "Adapter Composition Without Interference"
topic: 13-parameter-efficient-adaptation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adapter Composition Without Interference

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/adapter-composition-interference` · **Status:** open

## 1. Problem Statement

Given a frozen base model and $n$ independently trained parameter-efficient adapters (LoRA, IA³, prefix, bottleneck adapters), each competent on its own task, produce a single artifact that is simultaneously competent on all $n$ tasks — without joint retraining, without access to the original training data, and at inference cost close to that of one adapter.

Three variants, routinely conflated:

- **Measurement variant.** Define *interference*: a quantity attributable to the composition operator rather than to task conflict inherent in the data. No agreed definition exists; most papers report only end-task accuracy drop, which confounds the two.
- **Method variant.** Find a composition operator $C$ whose per-task loss matches the individual adapters within a stated tolerance for $n$ in the tens or hundreds. Empirically open.
- **Theory variant.** Characterize when adapters *can* compose losslessly — what property of $\{\Delta_i\}$ and of the base model's local geometry makes additive or routed composition exact. Theoretically open; the only partial answer is weight disentanglement in the linearized (NTK) regime.

Solving it means: a stated operator, a stated condition on the adapter set under which the operator is provably (or reproducibly) lossless, and a measurement separating operator-induced loss from task conflict.

## 2. Formal Setting

Base parameters $\theta_0 \in \mathbb{R}^d$. Adapter $i$ is a low-rank delta
$$\Delta_i = \frac{\alpha_i}{r_i} B_i A_i, \qquad B_i \in \mathbb{R}^{d_{\text{out}} \times r_i},\; A_i \in \mathbb{R}^{r_i \times d_{\text{in}}},$$
trained on $\mathcal{D}_i$ from a shared init. Composition operator $C: 2^{\{\Delta_i\}} \to \Theta$, e.g. linear $C_\lambda = \theta_0 + \sum_i \lambda_i \Delta_i$, or input-conditional $C(x) = \theta_0 + \sum_i g_i(x)\Delta_i$ with router $g$.

**Measured quantities.**

- *Single-adapter reference:* $a_i = \mathbb{E}_{(x,y)\sim \mathcal{D}_i^{\text{test}}}[m_i(f_{\theta_0+\Delta_i}(x), y)]$, $m_i$ the task metric. Measured on a held-out split of $\ge 1{,}000$ examples; report seed variance over $\ge 3$ adapter training seeds.
- *Normalized retention:* $R(C) = \frac{1}{n}\sum_i \frac{a_i(C) - b_i}{a_i - b_i}$, where $b_i$ is the zero-shot base score. $R=1$ means lossless; $R<1$ is the composition penalty. Retention normalizes away metric scale, which raw accuracy-drop averages do not.
- *Interference proper:* separate the operator from the data by comparing against a **joint-training oracle** $\theta^\star = \arg\min \sum_i \mathcal{L}_i$ under the same parameter budget:
$$\mathrm{Int}(C) = R(\theta^\star) - R(C).$$
$\mathrm{Int}>0$ is attributable to the operator; $R(\theta^\star)<1$ is genuine task conflict.
- *Weight disentanglement error* (Ortiz-Jimenez et al., 2023): for $x$ in the support of task $i$,
$$\xi(\lambda) = \sum_i \mathbb{E}_{x\sim\mathcal{D}_i}\big[\mathrm{dist}\big(f_{\theta_0+\sum_j \lambda_j\Delta_j}(x),\; f_{\theta_0+\lambda_i\Delta_i}(x)\big)\big].$$
Measured directly on logits; $\xi=0$ is exact linear composability.
- *Subspace overlap:* $O_{ij} = \|\mathrm{col}(B_iA_i)^\top \mathrm{col}(B_jA_j)\|_F^2 / \min(r_i,r_j)$, from SVD of the deltas. Cheap and computable without data.

**Assumptions, and their violation status.**

1. *Additivity:* $f_{\theta_0+\Delta_i+\Delta_j} \approx f_{\theta_0+\Delta_i} + f_{\theta_0+\Delta_j} - f_{\theta_0}$. Holds to first order only; **known violated** — cross terms $B_iA_iB_jA_j$ appear in every stacked layer, and the residual grows with depth and with $\|\Delta\|$.
2. *Low-rank sum is low-rank:* false by construction — $\mathrm{rank}(\sum_i \Delta_i) \le \sum_i r_i$, so merging $n$ rank-16 adapters needs rank $16n$ to be exact.
3. *Shared basis:* adapters trained from the same $\theta_0$ with the same architecture. **Violated** whenever adapters come from a public hub with different ranks, target modules, $\alpha$ scaling, or a differently-quantized base.
4. *Task supports disjoint:* assumed by routing methods. **Violated** for overlapping domains (two code adapters, two Spanish adapters).

## 3. State of the Art

**Established (ablated, reproduced independently).**

- **Task arithmetic** (Ilharco et al., ICLR 2023): $\theta_0 + \lambda\sum_i \tau_i$ with a single global $\lambda$ tuned on validation. The baseline everyone reports against; effect reproduced across CLIP/ViT and T5.
- **TIES-Merging** (Yadav et al., NeurIPS 2023): trim low-magnitude delta entries, elect a sign per coordinate, average only sign-agreeing entries. Beats task arithmetic consistently across 7–11 task suites; the trim and sign-election steps are separately ablated in the paper.
- **DARE** (Yu et al., ICML 2024): drop a fraction $p$ of delta entries and rescale by $1/(1-p)$ before merging. Ablated over $p$; robust for SFT-scale deltas, degrades for large-delta continued pretraining.
- **Linearized / tangent-space fine-tuning** (Ortiz-Jimenez et al., NeurIPS 2023): weight disentanglement, not linearity per se, is what makes task arithmetic work; fine-tuning in the tangent space raises disentanglement and merged accuracy.

**Claimed but under-ablated.**

- **LoraHub** (Huang et al., COLM 2024): gradient-free (Shiwa/CMA-ES) search over mixing coefficients for ~20 LoRAs on few-shot BBH. Strong headline numbers, but coefficient search uses the few-shot examples — the comparison to few-shot ICL is not budget-matched in an obvious way, and $n$ beyond a few dozen is untested.
- **Arrow / LoRA-library routing** (Ostapenko et al., ICML 2024): zero-shot routing using the top right-singular vector of each LoRA as a prototype. Promising at $n\approx 250$ adapters, but reported mainly as benchmark aggregate; per-task interference is not decomposed.
- **AdapterFusion** (Pfeiffer et al., EACL 2021): learned attention over adapter outputs. Requires a fusion training stage and adds $O(n)$ forward cost — it sidesteps rather than solves interference.
- **Orthogonality constraints** (O-LoRA, Wang et al., EMNLP Findings 2023): train each new adapter in the null space of previous ones. Works for sequential continual learning; needs the earlier adapters at training time, so it does not apply to post-hoc composition of independently trained modules.

Several results exist **only as benchmark numbers** — merged-model scores on GLUE/BBH/ViT-8-task suites, with no per-task retention curve as $n$ grows and no joint-training oracle. That is the main evidential gap in the literature.

## 4. What Is Known

- **Merging degrades with $n$.** On the 8-task ViT-B/32 CLIP benchmark, task arithmetic recovers roughly 70% of individual fine-tuned accuracy; TIES improves this by about 2–4 points absolute at the same $n$. Both fall further as $n$ goes from 8 to 20+ (Yadav et al., NeurIPS 2023).
- **Sign conflict is a real mechanism.** In TIES's own analysis, a large fraction of delta coordinates have conflicting signs across tasks, and resolving sign before averaging is worth more than trimming alone — the two ablations separate cleanly.
- **Deltas are highly redundant.** DARE drops 90% (and, for some SFT models, up to 99%) of delta parameters with near-zero single-task loss, at 7B scale on LM-family models. Redundancy is a property of the delta, not of the merge.
- **Rank is a hard ceiling.** Merging $n$ rank-$r$ LoRAs into one rank-$r$ LoRA is a rank-$r$ approximation of a rank-$\le nr$ matrix; the truncation error is the tail singular mass and is measurable in closed form.
- **Concatenation is lossless for two adapters at inference if you keep both.** Stacking $[B_1\,B_2]$, $[A_1;A_2]$ reproduces the sum exactly — but only the *sum*, which still is not either individual model, and cost grows linearly in $n$.
- **Weight disentanglement, not small $\|\Delta\|$, predicts merge success** (Ortiz-Jimenez et al., NeurIPS 2023), measured on CLIP ViT-B/32 and ViT-L/14.

## 5. What Is Not Known

- **Theoretically open.** No necessary-and-sufficient condition on $\{\Delta_i\}$ for $\xi(\lambda)=0$ outside the linearized regime. No bound relating subspace overlap $O_{ij}$ (data-free) to retention loss $1-R$. No theory of how interference scales with $n$ — is $1-R$ linear in $n$, or in $\sum_{i<j}O_{ij}$, or does it saturate?
- **Empirically open.** Nobody has published retention curves for $n \in \{2,4,8,16,32,64,128\}$ with a joint-training oracle at each $n$, at 7B+ scale, with adapter seed variance. The experiment is runnable on ~8 A100s; it has not been run at the right scale.
- **Methodologically blocked.** "Interference" is unmeasured as defined here. Papers report merged accuracy, which contains task conflict, router error, coefficient-search overfitting, and operator loss, all summed. Without the oracle arm, an improvement in merged accuracy cannot be attributed to a better operator.

## 6. Why It Is Hard

**Confounded measurement is the primary obstruction.** Merged accuracy is a sum of at least four terms and no standard protocol separates them. A method that improves merged accuracy by tuning $\lambda$ on the evaluation tasks has improved coefficient search, not composability, and the papers rarely hold the search budget fixed.

**Non-identifiability is the second.** LoRA factors are invariant to $B \to BM$, $A \to M^{-1}A$ for any invertible $M$. Every method that operates on $A$ or $B$ separately — prototype routing, factor-wise averaging, per-factor orthogonalization — is measuring a gauge-dependent quantity. Only functions of the product $BA$ are well defined, and comparisons across adapters trained with different initializations are not obviously meaningful at the factor level.

**Absent ground truth.** There is no reference "correct composition" of two adapters. Joint training gives an achievable target but not an upper bound, and it costs $n$ times the training compute — which is exactly why the oracle arm gets skipped.

## 7. Current Research (as of 2026)

- **MoErging / post-hoc routing over adapter libraries** — routing to expert adapters at token or request granularity; surveyed by Yadav et al., *A Survey on Model MoErging* (TMLR, 2024/2025). Active at Mila, UNC, IBM Research.
- **Data-free merging with second-order information** — Fisher-weighted averaging (Matena & Raffel, NeurIPS 2022) and RegMean (Jin et al., ICLR 2023) extended to low-rank deltas *(frontier — verify)*.
- **Subspace-aware merging** — allocating disjoint singular directions per task before merging, rather than resolving conflicts after *(frontier — verify)*.
- **Serving systems** — S-LoRA and Punica style multi-adapter batching sidestep composition entirely by keeping adapters separate; this is the operationally deployed answer and a fair systems baseline for any merging claim.
- **Diffusion-side composition** — ZipLoRA (Shah et al., 2023) for style+subject; the two-adapter case with a visual, human-judgeable ground truth.

## 8. Concrete Next Experiment

**Question.** Does composition loss scale with the number of adapters $n$, or with pairwise subspace overlap $\sum_{i<j}O_{ij}$?

**Scale.** Llama-3.1-8B base, frozen. Train 64 LoRA adapters, rank 16, $\alpha=32$, on attention + MLP projections — 64 distinct tasks drawn from FLAN/Super-NaturalInstructions clusters, chosen so half the pairs are same-cluster (high expected overlap) and half cross-cluster. 3 seeds per adapter. Cost: ~64 × 3 × 2 GPU-hours ≈ 400 A100-hours for adapters.

**Arms.**
1. *Operators:* task arithmetic, TIES, DARE+TIES, Arrow routing, and the rank-$16n$ exact concatenation (upper reference).
2. *Control arm (the one usually missing):* joint LoRA training on the union of the $n$ task datasets at rank $16n$ — matched parameter budget, matched data. This gives $R(\theta^\star)$ at every $n$.
3. Sweep $n \in \{2,4,8,16,32,64\}$, 5 random subsets per $n$, stratified by mean $O_{ij}$ into low/high-overlap subsets.

**Deciding number.** Fit $1 - R(C) = \beta_1 n + \beta_2 \bar{O} + \epsilon$ per operator, over the ~150 (subset, operator) points. The decision is the ratio of standardized coefficients $\hat\beta_2/\hat\beta_1$ with a bootstrap CI. If $\hat\beta_2/\hat\beta_1 > 2$ with the CI excluding 1, interference is overlap-driven — subspace allocation is the right research direction and a data-free overlap statistic predicts merge success in advance. If $<0.5$, interference is capacity-driven, merging into fixed rank is bounded by truncation, and routing (not merging) is the answer. Report $\mathrm{Int}(C) = R(\theta^\star) - R(C)$ at $n=64$ alongside; if $\mathrm{Int}$ is under 2 points while $1-R$ is 20 points, the field has been optimizing the wrong term.

## 9. Key References

- **[Foundational]** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Pfeiffer, Kamath, Rücklé, Cho, Gurevych. *AdapterFusion: Non-Destructive Task Composition for Transfer Learning.* EACL, 2021. — arXiv:2005.00247
- **[Foundational]** Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi. *Editing Models with Task Arithmetic.* ICLR, 2023. — arXiv:2212.04089
- **[SOTA]** Yadav, Tam, Choshen, Raffel, Bansal. *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS, 2023. — arXiv:2306.01708
- **[SOTA]** Yu, Yu, Yu, Huang, Li. *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE).* ICML, 2024. — arXiv:2311.03099
- **[SOTA]** Ortiz-Jimenez, Favero, Frossard. *Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models.* NeurIPS, 2023. — arXiv:2305.12827
- **[SOTA]** Huang, Liu, Lin, Pang, Du, Lin. *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.* COLM, 2024. — arXiv:2307.13269
- **[SOTA]** Ostapenko, Su, Ponti, Charlin, Le Roux, Caccia, Sordoni. *Towards Modular LLMs by Building and Reusing a Library of LoRAs.* ICML, 2024. — arXiv:2405.11157
- **[Method]** Matena, Raffel. *Merging Models with Fisher-Weighted Averaging.* NeurIPS, 2022. — arXiv:2111.09832
- **[Method]** Jin, Ren, Preotiuc-Pietro, Cheng. *Dataless Knowledge Fusion by Merging Weights of Language Models (RegMean).* ICLR, 2023. — arXiv:2212.09849
- **[Method]** Wang, Chen, Chen, Ma, Xu, Zhao, Qi, Zhang. *Orthogonal Subspace Learning for Language Model Continual Learning (O-LoRA).* Findings of EMNLP, 2023. — arXiv:2310.14152
- **[Survey]** Yadav, Raffel, Muqeeth, Caccia, Liu, Chen, Bansal, Fajcik et al. *A Survey on Model MoErging: Recycling and Routing Among Specialized Experts for Collaborative Learning.* TMLR, 2024/2025. — arXiv:2408.07057

## 10. Worked Example

Two rank-16 LoRAs on one $4096\times4096$ attention projection of an 8B model. $\Delta_1$ (SQL generation) and $\Delta_2$ (medical QA), both $\|\Delta_i\|_F = 1.0$ after normalization.

**Step 1 — rank arithmetic.** $\Delta_1+\Delta_2$ has rank up to 32. Truncating the SVD back to rank 16 keeps only the top-16 singular mass. For a typical overlap $O_{12}\approx 0.15$, the sum's singular spectrum splits roughly as: 16 directions carrying ~78% of the energy, 16 carrying ~22%. Truncation error:
$$\frac{\|\Delta_{1+2} - \mathrm{SVD}_{16}(\Delta_{1+2})\|_F^2}{\|\Delta_{1+2}\|_F^2} \approx 0.22.$$
Discarding 22% of the delta energy is not a rounding error — DARE's tolerance for pruning applies to *random* coordinate dropping with rescaling, not to removing the tail of the spectrum, which is where the task-specific directions of the *weaker-scaled* adapter live.

**Step 2 — the cross term.** With adapters on two consecutive layers, the merged forward pass contains $B_1A_1 B_2A_2 x$ — a term present in neither single-adapter model. Its magnitude relative to the linear terms is $\approx \|\Delta_1\|\|\Delta_2\|/\|W_0\|$ per layer pair, and it compounds over 32 layers. This is why $\xi(\lambda)>0$ even when subspaces are perfectly orthogonal: **orthogonality kills the first-order conflict but not the second-order composition term.**

**Step 3 — where the measurement fails.** Suppose merged SQL accuracy drops from 71% to 58% and medical QA from 64% to 55%. Retention $R \approx 0.63$ (against base scores 12% and 31%). A paper would report this as 11 points of "interference". But the joint-rank-32 oracle, trained on both datasets, scores 69% and 62% — $R(\theta^\star)=0.93$. So $\mathrm{Int}(C) = 0.30$: about 30% of the achievable performance is lost by the *operator*, and 7% by genuine task conflict. Without the oracle arm, those two numbers are indistinguishable — and every method comparison in the literature that omits it is comparing sums, not the term it claims to improve.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*