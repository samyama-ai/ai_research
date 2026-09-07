---
id: 13-parameter-efficient-adaptation/adapter-capacity-metric
title: "Adapter Capacity Metric Definition"
topic: 13-parameter-efficient-adaptation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adapter Capacity Metric Definition

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/adapter-capacity-metric` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a frozen pretrained model $f_{\theta_0}$ and a parameter-efficient adaptation (PEFT) module family $\mathcal{A}$ — LoRA of rank $r$, a bottleneck adapter of width $b$, a prefix of length $p$, BitFit's bias set — we want a scalar **capacity** $C(\mathcal{A}, \theta_0)$ that predicts how much adaptation the family can absorb, and that is comparable *across* families.

Three variants, routinely conflated:

- **Measurement.** Define $C$ so that two configurations with equal $C$ reach equal downstream loss on held-out tasks, up to optimization noise. This is the blocked variant: no accepted definition exists.
- **Method.** Given a budget $B$ (trainable parameters, or peak memory), choose the family and hyperparameters maximizing downstream performance. Currently solved by grid search per task.
- **Theory.** Prove expressivity statements: which target functions are reachable from $\theta_0$ by $\mathcal{A}$. Partly solved for linear and ReLU networks under exact-fit criteria that do not bind in practice.

Solving it means: a computable $C$ that (i) is monotone in the module's ability to fit, (ii) is invariant to reparameterizations that leave the induced function class unchanged, and (iii) predicts cross-family transfer — "LoRA $r{=}8$ on attention $\approx$ adapter $b{=}32$ on FFN" — better than trainable-parameter count does.

## 2. Formal Setting

Let $\theta_0 \in \mathbb{R}^N$ be frozen weights, $\phi \in \mathbb{R}^m$ the adapter parameters, and $g: \mathbb{R}^m \to \mathbb{R}^N$ (or a function-space map) the adaptation. Trained on task distribution $\mathcal{D}$ with loss $\ell$:

$$\phi^\star = \arg\min_{\phi \in \mathbb{R}^m}\ \mathbb{E}_{(x,y)\sim\mathcal{D}}\ \ell\big(f_{\theta_0 + g(\phi)}(x),\, y\big).$$

Candidate capacity measures, each with its measurement protocol:

- **Trainable count.** $C_{\text{param}} = m$. Measured by counting tensors with `requires_grad=True`. For LoRA, $m = r(d_{\text{in}}+d_{\text{out}})$ per adapted matrix.
- **Reachable-manifold dimension.** $C_{\dim} = \operatorname{rank} \, J_g(\phi)$, $J_g = \partial g/\partial \phi$. Measured by SVD of the Jacobian at $\phi^\star$; for LoRA $g(\phi)=BA$ the map is bilinear and $C_{\dim} \le r(d_{\text{in}}+d_{\text{out}}) - r^2$ — strictly below $C_{\text{param}}$ because of the $GL(r)$ gauge $B \mapsto BM,\ A \mapsto M^{-1}A$.
- **Effective dimensionality** (Maddox et al., NeurIPS 2020). With $H$ the loss Hessian in $\phi$ at $\phi^\star$ and eigenvalues $\lambda_i$, $$C_{\text{eff}}(z) = \sum_{i=1}^{m} \frac{\lambda_i}{\lambda_i + z}.$$ Measured by Lanczos on Hessian-vector products; $z$ is a prior-precision knob with no task-independent setting.
- **Intrinsic-dimension threshold** (Li et al., 2018; Aghajanyan et al., ACL 2021). $d_{90}$ = smallest $d$ such that training in a random $d$-dimensional affine subspace reaches $90\%$ of full fine-tuning performance. Measured by sweeping $d$ with a fixed random projection. This is a property of the *task*, not of the adapter family — the usual category error.
- **Tangent-kernel capacity.** $C_{\text{NTK}} = \operatorname{rank}$ of $K(x,x') = \nabla_\phi f(x)^\top \nabla_\phi f(x')$ over a probe set, or its effective rank $\exp(H(\{\hat\lambda_i\}))$. Measured on $n$ probe inputs; cost $O(n^2 m)$.

Assumptions, with the ones known to fail in practice marked:

1. $\phi^\star$ is a well-defined minimizer. **Violated** — training is stochastic and stopped early; $C_{\text{eff}}$ and $C_{\dim}$ are evaluated at non-stationary points.
2. Capacity is optimization-independent. **Violated** — LoRA's usable capacity depends on the $\alpha/r$ scaling (Kalajdzievski, 2023) and on per-matrix learning rates (LoRA+, Hayou et al., ICML 2024); the same $r$ gives different losses.
3. Equal function class ⇒ equal downstream loss. **Violated** — full fine-tuning subsumes every LoRA class yet does not dominate it on every axis (Biderman et al., TMLR 2024: LoRA forgets less).
4. Capacity is task-independent. **Violated** by construction for $d_{90}$.

## 3. State of the Art

**Theory SOTA.** Zeng & Lee, *The Expressive Power of Low-Rank Adaptation* (ICLR 2024): for linear models, LoRA rank $\approx \frac{1}{2}$ of the width suffices to represent any target of comparable depth exactly; for ReLU networks, the required rank scales with (target depth / frozen depth) $\times$ width. Established as a theorem; the ranks it demands are one to two orders of magnitude above the $r \in \{4,\dots,64\}$ used in practice, so it does not explain observed behavior. Jang et al. (ICLR 2024) analyze LoRA in the NTK regime and show rank $\Theta(\sqrt{n})$ suffices to fit $n$ points — again a fitting criterion, not a generalization one.

**Empirical SOTA.** No accepted metric. The de facto standard is $C_{\text{param}}$ reported as "% of trainable parameters": Houlsby et al. (ICML 2019) 3.6%, BitFit (Ben Zaken et al., ACL 2022) ~0.08%, LoRA (Hu et al., ICLR 2022) 0.01% of GPT-3 175B. Rank-allocation methods implicitly define a capacity proxy: AdaLoRA (Zhang et al., ICLR 2023) uses a singular-value sensitivity score to redistribute rank; SoRA and IncreLoRA use similar heuristics. These are **claimed but unablated as capacity metrics** — they are validated only by end-task score at fixed budget, never by predicting a *different* family's performance.

**Benchmark-number-only results.** DoRA (Liu et al., ICML 2024) beats LoRA at matched $C_{\text{param}}$ on commonsense reasoning suites. This is a benchmark delta, not evidence about capacity: it shows $C_{\text{param}}$ is not sufficient, and offers no replacement.

## 4. What Is Known

- **Rank 1–2 can match full fine-tuning at 175B scale.** Hu et al. (ICLR 2022) report GPT-3 175B LoRA with $r=1$ on $W_q,W_v$ matching or beating full fine-tuning on WikiSQL and MNLI-m; increasing $r$ to 64 gives no reliable gain. Established, reproduced widely.
- **Intrinsic dimension shrinks with pretraining scale.** Aghajanyan et al. (ACL 2021): RoBERTa-scale models reach $90\%$ of full fine-tuning on MRPC within a few hundred random subspace dimensions, and $d_{90}$ falls as pretraining proceeds. Measured at 125M–355M parameters.
- **The gap opens on hard, high-token adaptation.** Biderman et al. (TMLR 2024), Llama-2 7B/13B, ~20B tokens of code continued-pretraining: LoRA underperforms full fine-tuning on target-domain accuracy while retaining more source-domain performance. Same $C_{\text{param}}$ ordering, opposite conclusion from the GLUE-scale result — capacity is regime-dependent.
- **Placement dominates count.** He et al., *Towards a Unified View of Parameter-Efficient Transfer Learning* (ICLR 2022): modifying FFN beats attention at matched budget in the large-budget regime; prefix tuning and adapters are two instances of one formal family. Established via controlled ablation on XSum/MT at BART-large scale.
- **Prompt tuning's capacity is scale-gated.** Lester et al. (EMNLP 2021): prompt tuning matches full fine-tuning at 11B (T5-XXL) but loses by >10 SuperGLUE points at 220M with the same prompt length. $C_{\text{param}}$ is constant across those points; performance is not.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no definition of adapter capacity that is invariant to the $GL(r)$ gauge, to $\alpha/r$ rescaling, and to learning-rate choice, and that is measurable without training on the target task. Every current candidate fails at least one.
- **Empirically open.** Whether any existing candidate ($C_{\text{eff}}$, NTK effective rank, $C_{\dim}$) predicts *cross-family* iso-performance. The experiment is a few thousand GPU-hours at 7B; nobody has published it.
- **Theoretically open.** Whether a generalization bound for the frozen-backbone class $\{f_{\theta_0+g(\phi)}\}$ exists that is tighter than the $\tilde{O}(\sqrt{m/n})$ parameter-counting bound and that recovers the observed $r$-insensitivity.

## 6. Why It Is Hard

Three named obstructions.

1. **Non-identifiability.** LoRA's $(A,B)$ parameterization has an $r^2$-dimensional continuous symmetry. Any capacity read off the parameterization — including $m$ and any Hessian spectrum, which is singular along gauge directions by construction — measures the chart, not the function class. $C_{\text{eff}}$ inherits $r^2$ exact zero eigenvalues.
2. **Confounded measurement.** Capacity and optimizability are measured by the same instrument: downstream loss after training. rsLoRA (Kalajdzievski, 2023) changes only the scaling constant $\alpha/r \to \alpha/\sqrt{r}$ and unlocks gains at large $r$ that were previously read as "extra rank does not help." The same experiment therefore supports either "capacity saturates" or "the optimizer stalls."
3. **Absent ground truth.** There is no task with a known required capacity. Synthetic tasks with a planted rank-$k$ weight delta give ground truth but do not reproduce the pretrained-feature geometry that makes low rank sufficient in the first place.

## 7. Current Research (as of 2026)

- **Rank allocation as implicit capacity estimation** — AdaLoRA-descendants, SoRA, sparse/structured variants. Mature but still task-coupled.
- **Spectral and geometric proxies** — DoRA's magnitude/direction decomposition suggests capacity should be split into a norm budget and a directional budget. *(frontier — verify)* Whether the direction component alone predicts iso-performance is unresolved.
- **Optimizer-corrected comparison** — LoRA+, rsLoRA, and PiSSA-style spectral initialization aim to remove obstruction (2) so that a capacity claim can be made at the optimizer's ceiling rather than its default.
- **Loss-landscape capacity** — effective-dimensionality and Hessian-spectrum tooling applied to adapters. *(frontier — verify)* No published cross-family validation known to us.
- Groups active in PEFT theory/measurement include Microsoft Research (LoRA lineage), NVIDIA (DoRA), Princeton/EPFL (LoRA optimization theory), and Databricks (LoRA-vs-full-FT at scale).

## 8. Concrete Next Experiment

**Question.** Does any candidate capacity metric predict cross-family iso-performance better than trainable-parameter count?

**Scale.** Llama-3.1 8B, frozen. Five adaptation families: LoRA (attention only), LoRA (FFN only), bottleneck adapter, prefix tuning, BitFit-plus-LayerNorm. Six budgets per family spanning $10^5$–$10^8$ trainable parameters. Four target tasks spanning regimes: GSM8K (reasoning, ~10M tokens), a 2B-token code corpus (domain shift), a summarization task, and a low-resource translation pair. $5 \times 6 \times 4 = 120$ runs, each with rsLoRA-style scaling and a per-family learning-rate sweep of 3 points to defuse the optimizability confound. Roughly 3,000 A100-hours.

**Control arm.** $C_{\text{param}}$ as the predictor, plus full fine-tuning at each task as the loss floor.

**Deciding number.** For each candidate metric $C$, fit held-out loss $L \approx h(C)$ with a single monotone $h$ shared across all five families, and report the **cross-family leave-one-family-out $R^2$** on held-out validation loss. A metric passes only if it exceeds $C_{\text{param}}$'s $R^2$ by $\ge 0.15$ absolute, on the same 120 runs, with the left-out family never seen during the fit. Below that, the metric is a within-family reparameterization and the problem stays blocked.

## 9. Key References

- **[Foundational]** Neil Houlsby, Andrei Giurgiu, Stanisław Jastrzębski, et al. *Parameter-Efficient Transfer Learning for NLP.* ICML, 2019. — arXiv:1902.00751
- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Chunyuan Li, Heerad Farkhoor, Rosanne Liu, Jason Yosinski. *Measuring the Intrinsic Dimension of Objective Landscapes.* ICLR, 2018. — arXiv:1804.08838
- **[Foundational]** Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL, 2021. — arXiv:2012.13255
- **[SOTA — theory]** Yuchen Zeng, Kangwook Lee. *The Expressive Power of Low-Rank Adaptation.* ICLR, 2024. — arXiv:2310.17513
- **[SOTA — empirical]** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[SOTA]** Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, et al. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML, 2024. — arXiv:2402.09353
- **[SOTA]** Qingru Zhang, Minshuo Chen, Alexander Bukharin, et al. *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR, 2023. — arXiv:2303.10512
- **[Unifying]** Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig. *Towards a Unified View of Parameter-Efficient Transfer Learning.* ICLR, 2022. — arXiv:2110.04366
- **[Measure]** Wesley Maddox, Gregory Benton, Andrew Gordon Wilson. *Rethinking Parameter Counting in Deep Models: Effective Dimensionality Revisited.* NeurIPS, 2020. — arXiv:2003.02139
- **[Method]** Damjan Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* Preprint, 2023. — arXiv:2312.03732
- **[Method]** Soufiane Hayou, Nikhil Ghosh, Bin Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML, 2024. — arXiv:2402.12354
- **[Survey]** Ning Ding, Yujia Qin, Guang Yang, et al. *Parameter-efficient fine-tuning of large-scale pre-trained language models.* Nature Machine Intelligence, 2023.
- **[Survey]** Vladislav Lialin, Vijeta Deshpande, Anna Rumshisky. *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning.* Preprint, 2023. — arXiv:2303.15647

## 10. Worked Example

Take one $4096 \times 4096$ attention projection in an 8B model and match two configurations by trainable-parameter count.

- **LoRA, $r = 16$:** $m = 16 \times (4096 + 4096) = 131{,}072$.
- **Bottleneck adapter, $b = 16$** (down-project, ReLU, up-project, plus biases): $m = 2 \times 16 \times 4096 + 16 + 4096 = 135{,}184$.

$C_{\text{param}}$ calls these equal to within 3%. Now compute the other candidates.

**Gauge correction.** LoRA's $GL(16)$ symmetry removes $r^2 = 256$ directions: $C_{\dim} = 131{,}072 - 256 = 130{,}816$, a 0.2% correction. The adapter's ReLU breaks the symmetry down to positive diagonal rescaling, removing only $b = 16$. So the identifiability fix changes nothing at the resolution that matters — it is a 0.2% edit to a quantity that is off by an unknown factor.

**Rank of the induced weight update.** LoRA's $\Delta W = BA$ has rank $\le 16$ out of 4096. The adapter's update is not a weight delta at all: it is a residual function $x + U\sigma(Dx)$ that is nonlinear, so it has no rank. The two configurations are not comparable on this axis — the metric is undefined for one arm.

**Scaling changes the answer without changing any count.** With the standard $\alpha/r$ scaling and $\alpha = 16$, the LoRA update is multiplied by $1.0$; under rsLoRA's $\alpha/\sqrt{r}$ it is multiplied by $4.0$. Identical $m$, identical $C_{\dim}$, identical rank — and in the regimes Kalajdzievski (2023) reports, materially different loss. In a published GPT-3 setting the difference between $r=1$ and $r=64$ is under a point on MNLI-m (Hu et al., ICLR 2022), i.e. smaller than what a scaling constant can move.

**The obstruction, visible.** Two arms matched to 3% on the only metric anyone reports; one candidate replacement moves the number by 0.2%, a second is undefined on half the comparison, and a change to a single scalar that no metric sees moves the outcome more than a 64-fold rank change does. Any $C$ blind to the optimizer's effective step is not measuring capacity — which is why Section 8 sweeps learning rate inside each arm before fitting $h$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*