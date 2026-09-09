---
id: 29-distillation/linear-mode-connectivity-transfer
title: "Linear Mode Connectivity Across Transfer Tasks"
topic: 29-distillation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Linear Mode Connectivity Across Transfer Tasks

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/linear-mode-connectivity-transfer` · **Status:** partially-solved

## 1. Problem Statement

Two networks fine-tuned from the same pretrained checkpoint $\theta_0$ on two different downstream tasks land at $\theta_A$ and $\theta_B$. The **linear path** $\theta(\alpha) = (1-\alpha)\theta_A + \alpha\theta_B$ either stays low-loss on each task or it does not. The problem: predict, before training, whether it will — and say what property of the pretraining checkpoint and the task pair controls it.

Three variants, different difficulty:

- **Measurement.** Define a barrier that is comparable across task pairs with different loss scales, different label spaces, and different head geometry. Currently not standardized.
- **Method.** Given $\theta_0$ and tasks $A, B$, produce fine-tuned endpoints that are provably (or reliably) barrier-free, so that weight averaging, task arithmetic, and merged multi-task models work. Partially solved: model soups and task vectors work on CLIP/ViT fine-tunes and fail on from-scratch runs.
- **Theory.** Prove a sufficient condition on $\theta_0$ — a basin-radius, a curvature, or a linearization regime — under which cross-task LMC holds. Open.

Solving it means: a computable predictor $\hat{B}$ of the cross-task barrier from $\theta_0$ and cheap task statistics, correlating with the measured barrier at $r > 0.8$ across at least 50 task pairs, plus a theorem specifying when the barrier is $o(1)$.

## 2. Formal Setting

Let $f_\theta: \mathcal{X} \to \mathbb{R}^k$ with $\theta \in \mathbb{R}^d$. Task $T$ has data distribution $\mathcal{D}_T$ and loss $\mathcal{L}_T(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_T}[\ell(f_\theta(x), y)]$, measured as the mean cross-entropy on a held-out split of $n \geq 10^4$ examples (train-loss barriers and test-loss barriers differ; report both).

**Barrier.** For endpoints $\theta_A, \theta_B$ and evaluation task $T$:

$$B_T(\theta_A, \theta_B) = \max_{\alpha \in [0,1]} \mathcal{L}_T\big((1-\alpha)\theta_A + \alpha\theta_B\big) - \big[(1-\alpha)\mathcal{L}_T(\theta_A) + \alpha \mathcal{L}_T(\theta_B)\big].$$

Measured on a grid $\alpha \in \{0, 0.05, \dots, 1\}$ (21 forward passes per task). The **cross-task barrier** is $\bar{B} = \frac{1}{2}(B_A + B_B)$ where $B_A$ is evaluated on task $A$'s loss and $B_B$ on task $B$'s. LMC is declared at threshold $\bar{B} < \varepsilon$; the field uses $\varepsilon = 0.02$ in accuracy units or "within run-to-run noise," which is not the same thing.

**Practical measurement caveats that are usually violated:**

1. **Shared parameterization.** $\theta_A$ and $\theta_B$ must live in the same coordinate frame. Different label spaces mean different heads; the standard fix is to interpolate the backbone only and evaluate with each task's own frozen head. This makes $\bar{B}$ a statement about the trunk, not about $f_\theta$.
2. **Batch-norm statistics are not interpolated.** Running means/variances at $\theta(\alpha)$ are stale. Recomputing them (a "BN reset" pass) removes a large fraction of the apparent barrier on ResNets. Papers that do not reset are not measuring the same quantity as papers that do.
3. **Permutation symmetry.** $\bar{B}$ is not invariant to the $\prod_l m_l!$ hidden-unit permutations. Cross-task LMC from a shared $\theta_0$ is usually studied *without* re-alignment, on the assumption that fine-tuning does not permute — plausible at small learning rates, unverified at large ones.
4. **Weight decay and LR schedule** change $\|\theta_A - \theta_0\|$ by a factor of several; barrier scales with that distance, so barriers across papers with different recipes are not comparable.

Define the **fine-tuning radius** $\rho_T = \|\theta_T - \theta_0\|_2 / \|\theta_0\|_2$. The working hypothesis in the field is that cross-task LMC holds when $\rho \ll 1$ and the pretrained model is in a near-linearized (NTK-like) regime, i.e. $f_\theta \approx f_{\theta_0} + \nabla_\theta f_{\theta_0}^\top(\theta - \theta_0)$ over the fine-tuning ball. This assumption is known to be violated: fine-tuning does change features, and full-model fine-tuning outperforms its own linearization on most tasks.

## 3. State of the Art

**Established (reproduced, ablated).**

- *Instability analysis* (Frankle, Dziugaite, Roy, Carbin, ICML 2020): two runs from the same initialization with different SGD noise are barrier-free only after a short "stability" period of training. Independently reproduced many times.
- *Same-checkpoint fine-tunes on the same task are often linearly connected* (Neyshabur, Sedghi, Zhang, NeurIPS 2020): fine-tunes from a shared ImageNet checkpoint on CheXpert/DomainNet show essentially no barrier, while from-scratch pairs show a large one. This is the cleanest positive result and it is *within*-task.
- *Model soups* (Wortsman et al., ICML 2022): averaging many CLIP fine-tunes with different hyperparameters improves accuracy, which requires a low barrier along those segments. Heavily ablated.
- *Task arithmetic* (Ilharco et al., ICLR 2023): $\theta_0 + \sum_i \lambda_i (\theta_i - \theta_0)$ produces usable multi-task models — an operational consequence of approximate cross-task flatness.

**Claimed but under-ablated.**

- The claim that cross-task LMC follows from *weight disentanglement* rather than from the NTK regime (Ortiz-Jimenez, Favero, Frossard, NeurIPS 2023). The paper shows linearized fine-tuning improves task arithmetic, but disentanglement is diagnosed on the same models it explains.
- Layerwise linear feature connectivity (Zhou et al., NeurIPS 2023) as a *cause* of LMC rather than a correlate.

**Benchmark-number-only.** Most reported cross-task merging results are single accuracy figures on the 8-task CLIP/ViT suite (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN). Barriers themselves are rarely reported; merged accuracy is reported instead. These are not interchangeable — a merge can lose accuracy from head/scale mismatch with zero trunk barrier.

## 4. What Is Known

- **Shared pretraining is close to necessary.** From-scratch ResNet-50 pairs on ImageNet show barriers of order 1–2 nats; pairs sharing an ImageNet checkpoint show barriers indistinguishable from zero on DomainNet transfer tasks (Neyshabur et al., 2020; ResNet-50 scale).
- **Stability onset is early.** ResNet-20/CIFAR-10 becomes stable to SGD noise within roughly the first 1–3% of training; ResNet-50/ImageNet at roughly epoch 18 of 90 (Frankle et al., ICML 2020).
- **Permutation alignment collapses most barriers between independent runs.** Git Re-Basin (Ainsworth, Hayase, Srinivasa, ICLR 2023) drives the CIFAR-10 VGG/ResNet barrier to near zero after weight matching for sufficiently wide models; the effect weakens sharply at standard width. Entezari et al. (ICLR 2022) conjecture that with permutations accounted for, most SGD solutions are in one basin — a conjecture, not a theorem, with proofs only for one-hidden-layer models at large width.
- **Same task, same checkpoint, still multiple basins.** BERT fine-tuned on MNLI from one pretrained checkpoint splits into distinct linearly-connected clusters; cluster identity predicts HANS out-of-distribution behavior (Juneja et al., ICLR 2023). Barrier is therefore not purely a function of $\theta_0$ and the task.
- **Continual/multi-task connectivity.** Mirzadeh et al. (ICLR 2021) show multitask and sequentially-trained solutions are often linearly connected with low barrier, and exploit this (MC-SGD) to reduce forgetting on Rotated/Permuted MNIST and CIFAR-100 splits.
- **Scale of the effect in practice.** Greedy model soups on ViT-G/14 CLIP reach 90.94% ImageNet top-1 (Wortsman et al., 2022) — a working existence proof of low barriers among fine-tunes at 1.8B parameters.

## 5. What Is Not Known

- **Theoretically open.** No proof of a sufficient condition on $(\theta_0, \mathcal{D}_A, \mathcal{D}_B)$ for $\bar{B} = o(1)$ outside the linearized/infinite-width limit. Entezari's permutation conjecture is unproven at finite width and finite depth. Whether cross-task LMC requires the NTK regime or merely weight disentanglement is unresolved — the two make different predictions for large-$\rho$ fine-tuning and nobody has separated them.
- **Empirically open.** The scaling law of $\bar{B}$ in model width, pretraining-data volume, and fine-tuning radius $\rho$ has not been measured on a single controlled grid. Runnable today: 5 model sizes × 20 task pairs × 3 radii is ~300 fine-tunes, feasible at ViT-B scale. Nobody has run it.
- **Methodologically blocked.** There is no accepted normalization making barriers comparable across tasks with different loss scales, and no agreement on whether BN statistics get reset. A single "barrier" number in the literature can mean at least four different quantities.

## 6. Why It Is Hard

**Confounded measurement, compounded by non-identifiability.** The measured barrier mixes at least four causes that current protocols do not separate: (i) genuine loss-landscape structure, (ii) stale normalization statistics at the midpoint, (iii) head/logit-scale mismatch, and (iv) permutation misalignment. A near-zero barrier can be manufactured by shrinking the learning rate until $\rho \to 0$, at which point both endpoints are nearly $\theta_0$ and connectivity is trivial. Conversely a large barrier can be an artifact of BN drift. Because the confound (ii) can be removed by a cheap reset pass and (iv) by re-alignment, published barrier values are only comparable within a paper. Compute cost is real but secondary — the deeper problem is that $\bar{B}$ as measured is not a function of the landscape alone, so no amount of GPU time on the current protocol settles the theory question.

## 7. Current Research (as of 2026)

- **Merging and model soups at LLM scale** — TIES-merging, DARE, task-vector variants; the practical arm, dominated by industrial labs and open-weight communities. Barriers largely go unmeasured; merged benchmark scores stand in for them.
- **Permutation and symmetry theory** — Git Re-Basin lineage, simultaneous connectivity of more than two models modulo permutation (Sharma et al., ECML/PKDD 2024). *(frontier — verify)* Extensions to attention-head and residual-stream symmetries in transformers remain thin.
- **Mechanistic connectivity** — Lubana et al. (ICML 2023) argue barrier-free paths imply shared mechanisms, giving a use for LMC as a diagnostic rather than a merging tool. Active at Michigan/Mila/Cambridge-adjacent groups.
- **Linearized fine-tuning** — tangent-space task arithmetic (EPFL) as both an explanation and a method; the open question is whether the accuracy cost of linearizing is worth the disentanglement gain at LLM scale. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does the cross-task barrier depend on the landscape, or only on fine-tuning radius $\rho$?

- **Scale.** ViT-B/16, CLIP-pretrained. 8 tasks from the standard suite → 28 ordered pairs. For each task, fine-tune at three learning rates chosen post hoc to hit $\rho \in \{0.005, 0.02, 0.08\}$, 3 seeds each = 72 fine-tunes, ~1 GPU-day each on an A100 → ~72 GPU-days. Measure $\bar{B}$ on a 21-point grid with LayerNorm statistics untouched (ViT has no BN, removing confound (ii) by construction) and per-task frozen heads.
- **Control arm.** Same 72 runs but from a *randomly initialized* ViT-B trained to matched downstream accuracy, plus a within-task same-seed-different-noise pair at each $\rho$. The within-task pair gives the noise floor; the from-scratch pair gives the ceiling.
- **Deciding number.** The partial correlation between $\bar{B}$ and $\rho$ with task-pair identity held fixed, versus the partial correlation between $\bar{B}$ and task-pair identity with $\rho$ held fixed. If the first exceeds $0.7$ and the second falls below $0.2$, cross-task LMC is a radius phenomenon and the "shared basin" language should be retired. If the second exceeds $0.4$ at fixed $\rho$, there is genuine task-pair structure, and the next job is to predict it from task-gradient alignment $\cos(\nabla\mathcal{L}_A(\theta_0), \nabla\mathcal{L}_B(\theta_0))$.

## 9. Key References

- **[Foundational]** Garipov, Izmailov, Podoprikhin, Vetrov, Wilson. *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs.* NeurIPS 2018. — arXiv:1802.10026
- **[Foundational]** Frankle, Dziugaite, Roy, Carbin. *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* ICML 2020. — arXiv:1912.05671
- **[Foundational]** Neyshabur, Sedghi, Zhang. *What is being transferred in transfer learning?* NeurIPS 2020. — arXiv:2008.11687
- **[SOTA]** Entezari, Sedghi, Saukh, Neyshabur. *The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks.* ICLR 2022. — arXiv:2110.06296
- **[SOTA]** Ainsworth, Hayase, Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR 2023. — arXiv:2209.04836
- **[SOTA]** Wortsman et al. *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time.* ICML 2022. — arXiv:2203.05482
- **[SOTA]** Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi. *Editing Models with Task Arithmetic.* ICLR 2023. — arXiv:2212.04089
- **[SOTA]** Ortiz-Jimenez, Favero, Frossard. *Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models.* NeurIPS 2023. — arXiv:2305.12827
- Mirzadeh, Farajtabar, Görür, Pascanu, Ghasemzadeh. *Linear Mode Connectivity in Multitask and Continual Learning.* ICLR 2021. — arXiv:2010.04495
- Juneja, Bansal, Cho, Sedoc, Saphra. *Linear Connectivity Reveals Generalization Strategies.* ICLR 2023. — arXiv:2205.12411
- Lubana, Bigelow, Dick, Krueger, Tanaka. *Mechanistic Mode Connectivity.* ICML 2023. — arXiv:2211.08422
- Zhou et al. *Going Beyond Linear Mode Connectivity: The Layerwise Linear Feature Connectivity.* NeurIPS 2023. — arXiv:2307.08286
- **[Survey]** Yang et al. *Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities.* 2024. — arXiv:2408.07666

## 10. Worked Example

Take CLIP ViT-B/32, $\theta_0$ the pretrained trunk ($d \approx 8.8 \times 10^7$). Fine-tune on **EuroSAT** ($\theta_A$) and **SVHN** ($\theta_B$) with the standard recipe: 2000 steps, lr $10^{-5}$, cosine decay, AdamW. Measured radii come out near $\rho_A \approx 0.011$, $\rho_B \approx 0.014$ (typical for this recipe; the exact value is recipe-specific and must be recorded).

Now interpolate the trunk, keeping each task's own head frozen. Two facts fall out.

- On the EuroSAT head, accuracy at $\alpha = 0$ is roughly 99%, at $\alpha = 1$ it drops to the zero-shot-ish level, and along the way the curve is *monotone and concave-down but has no bump*: $B_A$ is small, on the order of a few hundredths of a nat.
- Read that as "same basin" and you have overclaimed. The endpoints are $\|\theta_A - \theta_B\|_2 \approx 0.018\|\theta_0\|_2$ apart. The midpoint is inside a ball of radius $\sim 1\%$ of the weight norm around the pretrained model — a region over which the network is close to its own first-order Taylor expansion, so *any* pair of points in it is linearly connected almost by construction. The barrier is small because the segment is short, not because the landscape is benign.

Make the obstruction visible: rerun SVHN at lr $10^{-4}$ for the same 2000 steps, reaching $\rho_B \approx 0.06$, with matched final accuracy. Now the same protocol yields a visible bump — barrier on the order of tenths of a nat. Nothing about the tasks changed. The reported barrier moved by an order of magnitude because a hyperparameter moved.

That is the whole difficulty in one instance: $\bar{B}$ is reported as a property of the task pair, but it is at least as strongly a property of the optimizer's step-size schedule. Until barriers are reported alongside $\rho$ — and ideally normalized by it, e.g. $\bar{B}/\rho^2$, which is the natural scale if the leading term is quadratic curvature along the segment — cross-paper comparison of "cross-task linear mode connectivity" is comparing different numbers with the same name.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*