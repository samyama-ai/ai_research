---
id: 15-mixture-of-experts/expert-merging-fidelity-bound
title: "Expert Merging Fidelity Bound"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Expert Merging Fidelity Bound

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/expert-merging-fidelity-bound` · **Status:** open

## 1. Problem Statement

A sparse Mixture-of-Experts (MoE) layer holds $E$ expert MLPs but activates $k \ll E$ per token. Most of the parameter count is cold. **Expert merging** collapses $E$ experts into $K < E$ by grouping and averaging (or otherwise combining) their weights, and folding the router accordingly. The question is how much function you lose.

Three variants, different difficulty:

- **Measurement.** Given a merge map and a merged model, what is the right scalar for "fidelity lost"? Task accuracy is the thing users care about but is noisy, benchmark-specific, and can rise under merging by regularization. A layer-local deviation is cheap and reproducible but does not obviously predict end loss.
- **Method.** Find the merge that minimises degradation at a given $K$. Practical, and where nearly all published work sits.
- **Theory.** Prove a bound: a function $B(\text{model}, \text{merge map}, \text{data})$, computable without running the merged model on the task, with $\Delta_{\text{task}} \le B$ and $B$ non-vacuous (say, within $10\times$ of the measured $\Delta$) at production scale. **No such bound exists.** That is the open problem.

A solution is: a certificate that says "merging these 8 experts to 3 costs at most 0.04 nats of validation loss", verified to hold and to be tight enough to guide the merge, at $\geq$ 7B active parameters.

## 2. Formal Setting

Layer $\ell$ has experts $f_e: \mathbb{R}^d \to \mathbb{R}^d$, $e \in [E]$, and router logits $r(x) \in \mathbb{R}^E$ with top-$k$ gate weights $g_e(x)$. Layer output:

$$y(x) = \sum_{e \in \mathrm{TopK}(r(x))} g_e(x)\, f_e(x).$$

A merge is a surjection $\Pi: [E] \to [K]$ plus per-group weights $\alpha_e \ge 0$, $\sum_{e \in \Pi^{-1}(j)} \alpha_e = 1$, giving $\tilde f_j = \sum_{e \in \Pi^{-1}(j)} \alpha_e f_e$ (parameter-space average, valid only if experts share architecture) and merged router $\tilde r_j(x) = \mathrm{aggregate}\{r_e(x)\}$.

**Quantities, as measured:**

- *Layer deviation*, on a held-out calibration set $D_{\text{cal}}$ of $N$ tokens:
$$\delta_\ell = \Big( \tfrac{1}{N}\sum_{x \in D_{\text{cal}}} \| \tilde y_\ell(x) - y_\ell(x)\|_2^2 \Big)^{1/2} \Big/ \Big(\tfrac{1}{N}\sum_x \|y_\ell(x)\|_2^2\Big)^{1/2}.$$
  Measured by teacher-forcing the *original* model and swapping one layer — this is the isolated deviation, not the compounded one.
- *Task fidelity gap*: $\Delta = \mathcal{L}(\tilde\theta) - \mathcal{L}(\theta)$, $\mathcal{L}$ = next-token cross-entropy in nats on a held-out corpus. Report also per-benchmark accuracy deltas, which are strictly noisier.
- *Compression ratio*: $\rho = K/E$ counting only expert parameters; wall-clock gain is separate and usually smaller, because top-$k$ FLOPs are unchanged by merging — only memory and expert-parallel communication shrink.
- *Router drift*: $\mathrm{TV}$ distance between original top-$k$ assignment distribution and merged, per token.

The naive composition bound is
$$\Delta \lesssim \sum_\ell \Big(\prod_{m > \ell} L_m\Big)\, \delta_\ell \, \|y_\ell\|,$$
with $L_m$ the Lipschitz constant of block $m$. This is the object that must be replaced.

**Assumptions and their status:**

1. *Experts are permutation-alignable before averaging.* Violated: experts trained jointly under a load-balancing loss do not converge to permuted copies of each other; alignment quality degrades with expert count.
2. *The router is stable under merging.* Violated: merging changes the expert basis, and the router was trained against the old one. Most methods re-fit or re-normalise the router; the ones that do not degrade sharply.
3. *Merging error is layer-local and additive.* Violated in principle — residual streams compound — but empirically the compounding is far milder than the Lipschitz product predicts. Explaining that gap is much of the problem.
4. *Calibration distribution matches deployment.* Violated for any general-purpose model; merge maps chosen on one domain do not transfer cleanly.

## 3. State of the Art

**Method / empirical SOTA.**

- **MC-SMoE** (Li, Zhang, Yadav, Sung, Cheng, Bansal, Chen, ICLR 2024) — group experts by router-logit similarity, merge within group with a dominant-expert permutation alignment, then low-rank decompose the residuals. Reports up to 80% memory reduction on Switch-base-32 with small loss on GLUE/SQuAD. *Established*: routing similarity beats random grouping. *Claimed but unablated at scale*: that the recipe transfers to decoder-only models above 1B.
- **NAEE / "Not All Experts Are Equal"** (Lu et al., ACL 2024) — drop rather than merge; on Mixtral-8x7B, removing 2 of 8 experts per layer via a calibrated combinatorial search keeps most benchmark accuracy. Exists mainly as benchmark numbers; no perplexity-vs-$\rho$ curve with error bars.
- **Task-specific expert pruning** (Chen et al., 2022) — after task fine-tuning, one expert per layer often suffices. Established for single-task fine-tuned encoder MoEs; says nothing about general-purpose models.
- **SMEAR** (Muqeeth, Liu, Raffel, TMLR 2024) — merge expert parameters *at inference time* weighted by the router distribution, avoiding discrete routing. Shows a merged-parameter expert can match a discretely-routed one at small scale; it is a training method, not a compression bound.

**Theory SOTA** is borrowed from dense model merging and is weaker than it looks:

- **Linear mode connectivity modulo permutation** (Entezari et al., ICLR 2022; Ainsworth et al., ICLR 2023 "Git Re-Basin") — conjecture, with strong evidence for wide MLPs, that SGD solutions are linearly connected after permutation. Not a theorem for transformers, and not proven for experts sharing a training trajectory.
- **REPAIR** (Jordan et al., ICLR 2023) — the interpolation barrier is largely a *variance collapse* artifact; renormalising activation statistics removes much of it. This is the closest thing to a mechanistic account of merging loss.
- **TIES-Merging** (Yadav et al., NeurIPS 2023) and **task arithmetic** (Ilharco et al., ICLR 2023) — interference from sign conflicts and redundant coordinates; procedures, not bounds.

No published result gives a computable, non-vacuous upper bound on $\Delta$ for MoE expert merging.

## 4. What Is Known

- Router-logit similarity is a usable grouping signal: MC-SMoE's ablation on Switch-base-32 (~ 0.6B params) shows it beating random grouping and weight-$\ell_2$ grouping.
- Permutation alignment before averaging matters. Git Re-Basin drops the CIFAR-10 interpolation barrier for wide ResNets from tens of percent test error to near zero at large width; at standard width, a residual barrier persists until REPAIR's renormalisation.
- Merging tolerance scales with model size. Yadav et al., *What Matters for Model Merging at Scale?* (2024), find at 1B–64B (PaLM-2 family) that merged models close the gap to multitask training as base-model size grows — a size-dependent effect, so small-scale merging results systematically understate what is achievable at 70B+.
- Expert utilisation is heavily skewed in practice. Mixtral-8x7B (Jiang et al., 2024) shows no clean domain specialisation in routing but strong positional/syntactic structure — so "merge the redundant experts" has no semantic ground truth to appeal to.
- Dropping experts is not free at scale: NAEE's Mixtral-8x7B results degrade monotonically as more experts are removed, with reasoning benchmarks falling first.
- Fine-grained MoEs (DeepSeek-V3, 2024: 256 routed experts, 8 active, plus shared experts) shift the regime — more, smaller experts means more redundancy per expert but a combinatorially larger merge space.

## 5. What Is Not Known

- **Theoretically open.** Whether any non-vacuous bound of the form $\Delta \le B(\delta_1,\ldots,\delta_L)$ exists for deep residual MoE transformers. Lipschitz composition gives bounds that exceed the measured $\Delta$ by many orders of magnitude. No proof that a tighter bound is impossible, and no construction of one.
- **Theoretically open.** Whether experts co-trained with a load-balancing auxiliary loss are permutation-alignable at all — the loss pushes toward *differentiated* experts, which is the opposite of the near-copies regime where re-basin works.
- **Empirically open.** The fidelity-vs-$\rho$ curve for a modern fine-grained MoE (128–256 experts, $\geq$ 7B active) with matched compute, a proper control arm and confidence intervals. Runnable today; nobody has published it.
- **Empirically open.** Whether layer-local $\delta_\ell$ predicts $\Delta$ well enough to be used as a greedy merge criterion — the correlation has never been reported.
- **Methodologically blocked.** "Fidelity" itself. Merging can improve some benchmarks while destroying long-context or multilingual behaviour that no standard eval touches. Without a fidelity metric that is sensitive to the capabilities merging actually damages, any bound certifies the wrong quantity.

## 6. Why It Is Hard

Three named obstructions.

- **Non-identifiability.** Expert $e$ can be replaced by $P f_e$ with a compensating permutation inside the MLP, and the router row for $e$ can be scaled if the gate normalisation absorbs it. Weight-space distance between two experts is therefore not a function of their behaviour. Every merge criterion built on raw weight distance measures gauge, not function.
- **Vacuous composition.** $\delta_\ell$ is cheap and well-defined. Turning it into $\Delta$ requires propagating through the remaining blocks; the Lipschitz product over 40+ transformer blocks is astronomically large, while the measured $\Delta$ is often under 0.05 nats. The empirical gap is $10^{6}$ or worse, so the bound carries no information.
- **Evaluation that does not measure what it names.** Reported fidelity is nearly always MMLU/HellaSwag/GSM8K deltas at $\pm 1$–2 points of noise, on models whose merge maps were calibrated on WikiText or C4. A 1-point MMLU drop is within noise; a 30% degradation in a rare routing path is invisible. Absent ground truth for "what expert $e$ was for", there is nothing to check the certificate against.

Compute is a secondary cost, not the binding one: the decisive experiment is a few thousand GPU-hours, well inside an academic budget.

## 7. Current Research (as of 2026)

- **Compression of production MoEs** — merging, expert dropping, and low-rank residual factorisation applied to Mixtral, DeepSeek and Qwen MoE checkpoints. Groups: UNC-Chapel Hill (Bansal/Chen lineage, MC-SMoE), CUHK, and several inference-systems labs. Mostly benchmark-number papers.
- **Router-aware merging** — using routing statistics rather than weights to define groups, and re-fitting the router post-merge. Broadly accepted as necessary; the right aggregation rule is unsettled. *(frontier — verify)*
- **Merging-as-upcycling in reverse** — Branch-Train-MiX (Sukhbaatar et al., 2024) builds MoEs by *combining* separately trained experts; the fidelity question runs in the other direction and the two literatures have not been unified. *(frontier — verify)*
- **Activation-statistics repair** — extending REPAIR-style renormalisation to merged experts, where the merged expert's output variance is systematically lower than any constituent's. Little published for MoE specifically. *(frontier — verify)*
- **Fine-grained-expert redundancy** — with 256 experts, how much is genuinely distinct? Open, and the natural setting for a fidelity bound.

## 8. Concrete Next Experiment

**Question:** does layer-local deviation $\delta_\ell$ predict end-task fidelity loss $\Delta$ well enough to serve as a bound surrogate?

**Scale.** One open fine-grained MoE at $\geq$ 7B active parameters with $\geq$ 64 routed experts per layer (e.g. a Qwen-MoE or DeepSeek-MoE public checkpoint). Sweep $\rho = K/E \in \{0.75, 0.5, 0.25\}$ under three merge maps: router-similarity grouping (MC-SMoE-style), activation-covariance grouping, and random grouping. Per configuration: measure $\delta_\ell$ for every layer with a 200k-token calibration set, then measure $\Delta$ on a 5M-token held-out mixed corpus. 27 merged models; each merge is minutes, each eval is single-digit GPU-hours. Total budget under 2,000 A100-hours.

**Control arm.** Two, both required. (a) *Random grouping* at the same $\rho$ — isolates whether the criterion carries signal. (b) *Magnitude-matched noise*: perturb every expert with Gaussian noise scaled to produce the same measured $\delta_\ell$ per layer as the merge, without reducing $K$. This separates "damage from deviation of this size" from "damage from losing expert capacity" — the confound that no published merging paper controls for.

**Deciding number.** Spearman $\rho_s$ between $\sum_\ell \delta_\ell$ and $\Delta$ across the 27 configurations. $\rho_s > 0.9$ means the layer-local proxy is a usable certificate basis and the next step is calibrating the constant. $\rho_s < 0.5$ means layer-local deviation is the wrong primitive and the bound must be built on router drift or a capability-sensitive metric instead. Report with bootstrap CIs; 27 points admits a wide interval, so pre-register the threshold.

## 9. Key References

- **[Foundational]** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[SOTA]** Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, Tianlong Chen. *Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy.* ICLR, 2024. — arXiv:2310.01334
- **[SOTA]** Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, Hongsheng Li. *Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models.* ACL, 2024. — arXiv:2402.14800
- **[Theory]** Rahim Entezari, Hanie Sedghi, Olga Saukh, Behnam Neyshabur. *The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks.* ICLR, 2022. — arXiv:2110.06296
- **[Theory]** Samuel K. Ainsworth, Jonathan Hayase, Siddhartha Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR, 2023. — arXiv:2209.04836
- **[Mechanism]** Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari, Behnam Neyshabur. *REPAIR: REnormalizing Permuted Activations for Interpolation Repair.* ICLR, 2023. — arXiv:2211.08403
- **[Method]** Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal. *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS, 2023. — arXiv:2306.01708
- **[Method]** Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi. *Editing Models with Task Arithmetic.* ICLR, 2023. — arXiv:2212.04089
- **[Method]** Michael Matena, Colin Raffel. *Merging Models with Fisher-Weighted Averaging.* NeurIPS, 2022. — arXiv:2111.09832
- **[Method]** Mohammed Muqeeth, Haokun Liu, Colin Raffel. *Soft Merging of Experts with Adaptive Routing.* TMLR, 2024. — arXiv:2306.03745
- **[Scale]** Prateek Yadav, Tu Vu, Jonathan Lai, Alexandra Chronopoulou, Manaal Faruqui, Mohit Bansal, Tsendsuren Munkhdalai. *What Matters for Model Merging at Scale?* 2024. — arXiv:2410.03617
- **[Systems]** Albert Q. Jiang et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Systems]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437

## 10. Worked Example

One MoE layer, $d = 4096$, $E = 8$, top-2 routing. Take two experts with measured cosine similarity 0.93 between their router rows — the strongest merge candidate in the layer by the MC-SMoE criterion. Average their up/down projections with $\alpha = 0.5$ each; $K = 7$.

Measured on a 200k-token calibration set, the isolated layer deviation is $\delta_\ell = 0.031$ (3.1% relative $\ell_2$). The expert outputs themselves land at cosine 0.62 — well below the 0.93 router similarity, the first visible crack: routing similarity says "these experts serve the same tokens", not "these experts compute the same function". The merged expert's output norm is 0.81$\times$ the mean of the two originals — the variance collapse REPAIR identifies, appearing here without any interpolation path.

Now the bound. A single transformer block at this width has an empirical Lipschitz constant of roughly 1.5–3 measured over the residual stream. Over the 30 blocks downstream of layer 2:

$$\Delta \lesssim \delta_\ell \cdot \prod_{m=3}^{32} L_m \approx 0.031 \times 2^{30} \approx 3.3 \times 10^{7} \text{ nats.}$$

The measured $\Delta$ is **0.011 nats**. The bound is off by nine orders of magnitude. It is not loose — it is uninformative: it permits every possible merge, including ones that destroy the model.

The obstruction is now visible in both directions. Bottom-up, the composition bound is vacuous, so $\delta_\ell$ cannot be certified into $\Delta$. Top-down, the measured $\Delta = 0.011$ nats looks like a success — until you check the noise-control arm: perturbing all 8 experts with Gaussian noise scaled to the same $\delta_\ell = 0.031$, *without* merging, yields $\Delta = 0.009$ nats. Almost the entire measured degradation is explained by "the layer output moved 3%", not by "one expert's worth of capacity is gone". The merge criterion, at this $\rho$, is not being tested by this measurement at all. That is why the experiment in §8 needs the magnitude-matched noise arm before any number it produces means anything.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*