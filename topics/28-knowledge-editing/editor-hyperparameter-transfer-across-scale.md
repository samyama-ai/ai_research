---
id: 28-knowledge-editing/editor-hyperparameter-transfer-across-scale
title: "Hyperparameter Transfer of Editors Across Scale"
topic: 28-knowledge-editing
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hyperparameter Transfer of Editors Across Scale

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/editor-hyperparameter-transfer-across-scale` · **Status:** empirically-open

## 1. Problem Statement

Every parametric knowledge editor ships a per-model hyperparameter file: which layers to write, how strongly to regularize, how many optimization steps to take for the target activation, how much second-moment mass to mix into the covariance preconditioner. These files are tuned by hand, per checkpoint, against a validation split of the same benchmark they are later scored on. The question:

**Do editor hyperparameters admit a scaling rule?** Given a tuned configuration $\theta^\star(M_s)$ for a small model $M_s$, is there a map $\mathcal{T}$ such that $\mathcal{T}(\theta^\star(M_s), M_s, M_l)$ is within tolerance of $\theta^\star(M_l)$ for a large model $M_l$ — the model-editing analogue of $\mu$P zero-shot transfer for pretraining (Yang et al., NeurIPS 2021)?

Three variants, of different difficulty:

- **Measurement.** Define the loss surface over $\theta$ in a way that is comparable across scales at all. Edit success and locality damage are measured on different scales in different models; a single scalar "editor quality" is not agreed upon.
- **Method.** Find $\mathcal{T}$ empirically — fit optimal $\theta^\star$ at 3–4 scales, check whether the optima lie on a smooth curve in the right coordinates (fractional depth, width-normalized $\lambda$).
- **Theory.** Prove that under a stated parameterization, the argmax of the edit objective is scale-invariant in transformed coordinates, as $\mu$P does for the optimal learning rate.

Solved = a published rule that, transferred from a $\le 1.5$B proxy, matches per-scale tuning at $\ge 70$B on both efficacy and locality, with the tuning-cost saving quantified.

## 2. Formal Setting

Let $M$ be a decoder-only transformer with depth $L$, hidden width $d$, MLP width $d_m$. An editor $E_\theta$ maps $(M, \mathcal{R}) \mapsto M'$, where $\mathcal{R} = \{(s_i, r_i, o_i^\star)\}_{i=1}^n$ are requested edits.

For ROME/MEMIT-family editors the write is a rank-$n$ update to one MLP down-projection $W \in \mathbb{R}^{d \times d_m}$:

$$\hat{W} = \arg\min_{\tilde W}\ \|\tilde W K_0 - V_0\|_F^2 + \|\tilde W K_1 - V_1\|_F^2,$$

with $K_0,V_0$ the preserved key/value set and $K_1,V_1$ the edited set. The closed form uses $C = \lambda\,\mathbb{E}_{x\sim\mathcal{D}}[k(x)k(x)^\top]$, an uncentered second moment over a Wikipedia sample. The hyperparameter vector is

$$\theta = (\underbrace{\mathcal{L}}_{\text{layer set}},\ \underbrace{\lambda}_{\texttt{mom2\_update\_weight}},\ \underbrace{\eta_v, T_v}_{\text{target-activation lr, steps}},\ \underbrace{\beta_{\mathrm{KL}}}_{\text{KL factor}},\ \underbrace{c}_{\text{clamp norm factor}}).$$

Measured quantities, all on a held-out edit set:

- **Efficacy** $\mathrm{ES} = \frac{1}{n}\sum_i \mathbb{1}[\,p_{M'}(o_i^\star \mid s_i, r_i) > p_{M'}(o_i^{c} \mid s_i, r_i)\,]$, $o^c$ the pre-edit object.
- **Generalization** $\mathrm{PS}$: same predicate on paraphrases.
- **Locality** $\mathrm{NS}$: same predicate on neighbourhood prompts that must not change.
- **Collateral damage** $\Delta\mathrm{PPL} = \mathrm{PPL}_{M'}(\mathcal{D}_{\text{wiki}}) / \mathrm{PPL}_M(\mathcal{D}_{\text{wiki}}) - 1$, and downstream accuracy drop on a fixed suite.

Scalarize by a constrained objective, not a harmonic mean:

$$\theta^\star(M) = \arg\max_\theta\ \mathrm{ES}(\theta)\quad \text{s.t.}\quad \Delta\mathrm{PPL}(\theta) \le \epsilon,\ \ \mathrm{NS}(\theta) \ge \nu.$$

Transfer gap for a candidate rule $\mathcal{T}$:

$$G(\mathcal{T}) = \mathrm{ES}\big(\theta^\star(M_l)\big) - \mathrm{ES}\big(\mathcal{T}(\theta^\star(M_s))\big),$$

evaluated with both arms satisfying the same constraints.

**Assumptions, with the violated ones flagged.**
1. *Unimodality of $\mathrm{ES}$ in $\theta$.* Not established; layer choice is known to be non-monotone and multi-peaked.
2. *Fractional depth $\ell/L$ is the right depth coordinate.* Violated in practice — see §10.
3. *$\lambda$ is scale-free.* Violated: $C$ has entries growing with $d_m$ and with activation norm, which itself grows with depth in pre-LN models, yet shipped $\lambda$ is a constant across widths.
4. *The same benchmark distribution at every scale.* Approximately holds (CounterFact, zsRE), but base-model accuracy on the neighbourhood prompts differs by scale, so $\mathrm{NS}$ is not a fixed-difficulty quantity.
5. *Independence of edits.* Violated for $n>1$ sequential editing (Gupta et al., 2024).

## 3. State of the Art

**Established.** No published scaling rule for editor hyperparameters exists. The empirical SOTA is per-checkpoint manual tuning, shipped as config files in the MEMIT repository and in EasyEdit (Wang et al., ACL 2024 demo), covering GPT-2 XL, GPT-J 6B, GPT-NeoX 20B, Llama-2-7B/13B, Mistral-7B, Qwen. Editors themselves — ROME (Meng et al., NeurIPS 2022), MEMIT (Meng et al., ICLR 2023), PMET (Li et al., AAAI 2024), AlphaEdit (Fang et al., ICLR 2025) — each introduce new hyperparameters and each inherit the tuning problem.

**Claimed but unablated.** Papers routinely report that a method "works across model sizes"; what is actually shown is a table with a different config per row. No paper in this family reports a sensitivity sweep of $\lambda$ or $\mathcal{L}$ at more than two scales with a common protocol, so the claim "these hyperparameters transfer" has never been tested — it has been assumed by construction.

**Benchmark-number-only results.** All headline ES/PS/NS figures for 20B+ models are single-config point estimates without a tuning-budget disclosure. They are not comparable across methods, because the tuning effort per row is unreported.

**Adjacent theory SOTA.** $\mu$P (Yang et al., NeurIPS 2021) proves zero-shot transfer of optimal learning rate across width for *training*; Tensor Programs VI extends it to depth for residual networks. Everett et al. (ICML 2024) show the transferred optimum is parameterization- and optimizer-dependent and that alignment assumptions can fail. None of this has been carried over to a one-shot rank-one write with a preconditioner estimated from data.

## 4. What Is Known

- **The tuned layer is not at constant fractional depth.** ROME's shipped config edits layer 17 of GPT-2 XL ($L{=}48$, $\ell/L = 0.354$) and layer 5 of GPT-J ($L{=}28$, $0.179$). MEMIT edits layers 3–8 of GPT-J and 4–8 of Llama-2-7B ($L{=}32$, centre $\approx 0.19$). Measured from the released configs; the ratio differs by roughly $2\times$ between GPT-2 XL and everything larger.
- **$\lambda$ is held constant across three orders of magnitude of parameters.** $\lambda = 15000$ appears for GPT-J 6B, GPT-NeoX 20B and Llama-2-7B; $\lambda \approx 20000$ for GPT-2 XL 1.5B. It was not re-tuned with width. Derived observation: this is dimensionally inconsistent with $C \propto \mathbb{E}[kk^\top]$.
- **Causal-tracing peaks do not predict the best edit layer.** Hase et al. (NeurIPS 2023) show editing at layers far from the tracing peak succeeds equally well — the localization signal that motivated the layer hyperparameter does not determine its optimum.
- **Sensitivity is real and sharp in the sequential regime.** Gupta et al. (ACL Findings 2024; and *Rebuilding ROME*, EMNLP 2024) show ROME/MEMIT collapse after $10^3$–$10^4$ sequential edits on GPT-2 XL and GPT-J, with the collapse point depending on the same hyperparameters that are never re-tuned.
- **Editing degrades general ability at fixed edit-success.** Gu et al. (EMNLP 2024) report substantial downstream drops on Llama-2-7B / GPT-2 XL at configurations reported as successful under ES/PS/NS.
- **Specificity metrics overstate locality.** Hoelscher-Obermaier et al. (ACL Findings 2023) show CounterFact neighbourhood scores miss unrelated-fact damage, so the constraint $\mathrm{NS} \ge \nu$ used in tuning is loose.

## 5. What Is Not Known

- **Empirically open.** Whether $\theta^\star$ traces a smooth curve in $(\ell/L, \lambda/d_m, \eta_v)$ across $\{0.1, 1, 7, 70\}$B. The experiment is a sweep — expensive, not conceptually blocked. Nobody has published it.
- **Empirically open.** Whether the transfer gap $G$ is dominated by layer choice or by $\lambda$. Untested; a two-factor ablation at two scales would answer it.
- **Theoretically open.** Whether any parameterization makes the edit-objective optimum width-invariant. There is no theorem, either way, connecting $\mu$P-style feature-learning limits to a closed-form rank-one write with a data-estimated preconditioner.
- **Methodologically blocked.** The scalar being optimized. $\mathrm{ES}$, $\mathrm{NS}$ and $\Delta\mathrm{PPL}$ have different base rates per model, so "the optimum moved with scale" and "the metric moved with scale" are not separable under current protocols. Until locality is measured against a scale-matched control (§8), $\theta^\star$ is not well defined.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under a confounded objective**, compounded by cost.

Non-identifiability: $\lambda$ and $\eta_v T_v$ trade off. Raising $\lambda$ inflates $C$, shrinking the effective update; raising the target-activation optimization budget pushes $v_\star$ further from $Wk_\star$, enlarging it. Many $(\lambda, \eta_v, T_v)$ triples give the same $\|\hat W - W\|_F$ and near-identical ES. Fitting a scaling law to an argmax that is a flat ridge yields an exponent with a confidence interval wider than the effect.

Confounded measurement: a larger model has better base neighbourhood accuracy, so an identical amount of collateral damage produces a *larger* measured $\mathrm{NS}$ drop. The metric's scale-dependence is aliased into any measured drift of $\theta^\star$.

Cost: one point of the sweep is not one edit. Estimating $C$ requires a forward pass over $\sim 10^5$ Wikipedia samples per candidate layer, cached per model; the sweep is $|\mathcal{L}| \times |\lambda| \times |\eta_v T_v|$ configurations, each scored on $\ge 1000$ edits plus a downstream suite. At 70B this is thousands of GPU-hours — enough to deter, not enough to excuse.

## 7. Current Research (as of 2026)

- **Constraint-based editors that reduce hyperparameter count.** AlphaEdit (Fang et al., ICLR 2025) projects the update into the null space of preserved keys, removing the preservation/edit trade-off weight. Whether this shrinks the transfer gap is untested *(frontier — verify)*.
- **Sequential-editing stabilizers** — PRUNE (Ma et al.), Rebuilding ROME (Gupta et al.) — treat drift as a norm-control problem, which is a hyperparameter question restated.
- **Unified formulations.** EMMET (Gupta et al., EMNLP Findings 2024) puts ROME and MEMIT under one objective, making a shared hyperparameter coordinate system possible for the first time.
- **Tooling.** EasyEdit / KnowEdit (Zhang et al., 2024) is the only infrastructure where a cross-scale sweep is cheap to script; the configs there remain hand-tuned.
- Groups active in the area: Northeastern (Bau), Zhejiang NLP (EasyEdit), UC Berkeley (Anumanchipalli), MIT-IBM (Hartvigsen).

## 8. Concrete Next Experiment

**Scale.** Four checkpoints from one family with shared tokenizer and pretraining recipe — Pythia 410M / 1.4B / 6.9B / 12B, or Llama-3 1B / 3B / 8B / 70B. One family only; cross-family comparison reintroduces the depth/width confound.

**Sweep.** MEMIT, grid over layer-set centre $\ell/L \in \{0.10,0.15,\dots,0.45\}$ and $\lambda \in \{3.75, 7.5, 15, 30, 60\}\times 10^3$, with $\eta_v, T_v, \beta_{\mathrm{KL}}, c$ fixed at repo defaults. 35 configs $\times$ 4 scales, 1000 CounterFact edits each, batched $n=100$.

**Control arm.** Two controls, both required. (i) *Per-scale tuned*: $\theta^\star(M_l)$ from the full grid at the largest scale. (ii) *Scale-matched locality*: for each model, an unedited copy scored on the same neighbourhood prompts, so $\mathrm{NS}$ is reported as a delta from that model's own base rate — this is what removes the §6 confound.

**Deciding number.** The transfer gap

$$G = \mathrm{ES}(\theta^\star(M_{70\text{B}})) - \mathrm{ES}\big(\mathcal{T}(\theta^\star(M_{1\text{B}}))\big),$$

with $\mathcal{T}$ = constant fractional depth and $\lambda \propto d_m$, both arms held to $\Delta\mathrm{PPL} \le 0.02$ and base-rate-corrected $\mathrm{NS}$ drop $\le 2$ points. **$G \le 2$ points $\Rightarrow$ transfer works and per-scale tuning is waste. $G \ge 10$ points $\Rightarrow$ no naive rule exists and the coordinates are wrong.** Report the 95% CI over edit-set bootstrap; the grid must be dense enough that the CI is under 2 points, or the result is uninformative.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[Foundational]** Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS 2021. — arXiv:2203.03466
- **[SOTA]** Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Xiang Wang, Xiangnan He, Tat-Seng Chua. *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025. — arXiv:2410.02355
- **[SOTA]** Akshat Gupta, Dev Sajnani, Gopala Anumanchipalli. *A Unified Framework for Model Editing.* Findings of EMNLP 2024.
- **[Evidence]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023. — arXiv:2301.04213
- **[Evidence]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL 2024.
- **[Evidence]** Jia-Chen Gu, Hao-Xiang Xu, Jun-Yu Ma, Pan Lu, Zhen-Hua Ling, Kai-Wei Chang, Nanyun Peng. *Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue.* EMNLP 2024.
- **[Evidence]** Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas, Fazl Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* Findings of ACL 2023.
- **[Tooling]** Peng Wang, Ningyu Zhang, et al. *EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models.* ACL 2024 (System Demonstrations).
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. — arXiv:2305.13172
- **[Survey]** Ningyu Zhang et al. *A Comprehensive Study of Knowledge Editing for Large Language Models.* 2024. — arXiv:2401.01286

## 10. Worked Example

Take the two most-used shipped ROME configs and ask whether a fractional-depth rule reproduces them.

| Model | $L$ | $d_m$ | shipped edit layer $\ell$ | $\ell/L$ | shipped $\lambda$ |
|---|---|---|---|---|---|
| GPT-2 XL (1.5B) | 48 | 6400 | 17 | 0.354 | $2.0\times10^4$ |
| GPT-J (6B) | 28 | 16384 | 5 | 0.179 | $1.5\times10^4$ |

Apply the obvious rule $\mathcal{T}$: constant fractional depth, from GPT-2 XL to GPT-J.

$$\ell_{\text{pred}} = 0.354 \times 28 = 9.9 \approx 10 \quad\text{vs. shipped } 5.$$

The prediction misses by 5 layers out of 28 — 18% of the depth, and outside the $[3,8]$ band MEMIT uses for this model at all. A rule fitted the other way (constant absolute layer) predicts $\ell = 17$ for a 28-layer model, which is past the mid-stack and empirically a poor edit site. Neither coordinate works.

Now $\lambda$. If $\lambda$ should scale with $d_m$ (so that $\lambda C^{-1}$ has fixed effective rank contribution), then from GPT-2 XL:

$$\lambda_{\text{pred}} = 2.0\times10^4 \times \frac{16384}{6400} = 5.1\times10^4 \quad\text{vs. shipped } 1.5\times10^4.$$

A $3.4\times$ miss, in the direction *opposite* to the hand-tuned value — the tuner reduced $\lambda$ as width grew.

**What this makes visible.** Two plausible transfer rules disagree with the hand-tuned configs in opposite directions, and there is no published sweep that says which of the three is right, because ES was never measured off the shipped point. The confound in §6 is the reason: a sweep would have to score locality against a scale-matched base rate to tell "the optimum moved" from "the metric moved", and no released result does. The 5-layer and $3.4\times$ discrepancies are not evidence that transfer fails — they are evidence that nobody has measured whether it does.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*