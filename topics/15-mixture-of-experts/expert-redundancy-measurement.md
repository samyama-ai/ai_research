---
id: 15-mixture-of-experts/expert-redundancy-measurement
title: "Emergent Expert Redundancy Measurement"
topic: 15-mixture-of-experts
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Emergent Expert Redundancy Measurement

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/expert-redundancy-measurement` · **Status:** methodologically-blocked

## 1. Problem Statement

A sparse MoE layer holds $E$ expert MLPs but routes each token to $k \ll E$ of them. Practitioners repeatedly observe that some experts appear interchangeable: prune, merge, or duplicate them and loss barely moves. **Emergent expert redundancy measurement** is the problem of assigning a number to that observation.

Three variants, with different difficulty:

- **Measurement.** Given a trained MoE checkpoint and a data distribution, output a redundancy score $R \in [0,1]$ (or an effective expert count $E_{\text{eff}} \le E$) that is *predictive*: it should forecast, without running the compression, how much loss a given merge/prune budget costs. Solving it means the score correlates with post-compression loss across architectures, scales, and domains at a stated $r^2$.
- **Method.** Find the compression that exploits the redundancy — expert pruning, merging, low-rank factorization across experts. Partially solved; benchmark numbers exist.
- **Theory.** Prove when gradient-based training of a top-$k$ router *must* produce duplicate experts (e.g. as a function of $E$, $k$, load-balancing coefficient, and intrinsic task dimension), and whether redundancy is a training artifact or a capacity-allocation optimum.

The catalog status is **methodologically blocked** because the measurement variant has no agreed definition that survives the symmetries of the object being measured. The method variant runs ahead of it: papers report compression ratios without a redundancy metric that predicts them.

## 2. Formal Setting

Layer $\ell$ has experts $\{f_i(\cdot;\theta_i)\}_{i=1}^{E}$, $f_i(x) = W^{(2)}_i \sigma(W^{(1)}_i x)$, and a router $g(x) = \mathrm{softmax}(W_r x)$. With top-$k$ gating and normalized gates $\tilde g$,

$$y(x) \;=\; \sum_{i \in \mathrm{TopK}(g(x),k)} \tilde g_i(x)\, f_i(x).$$

Measured quantities, on a held-out corpus $\mathcal{D}$ of $N$ tokens (hidden states $x_t$ at layer $\ell$):

- **Routing load** $p_i = \frac1N \sum_t \mathbf{1}[i \in \mathrm{TopK}(g(x_t),k)]$. Directly countable.
- **Co-activation** $c_{ij} = \frac1N\sum_t \mathbf{1}[i,j \in \mathrm{TopK}(g(x_t),k)]$; normalize as $\phi_{ij} = c_{ij}/(p_i p_j)$.
- **Functional similarity** on the tokens each expert actually sees. Let $S_i = \{x_t : i \in \mathrm{TopK}\}$, and $S_{ij} = S_i \cap S_j$. Define
  $$d_{ij} \;=\; \frac{1}{|S_{ij}|}\sum_{x \in S_{ij}} \frac{\|f_i(x)-f_j(x)\|_2}{\tfrac12(\|f_i(x)\|_2+\|f_j(x)\|_2)}.$$
  This is the honest version. It requires $|S_{ij}|$ large, which top-$k$ routing does not guarantee.
- **Causal redundancy** — the ground truth the score should predict. For expert set $A$ removed (or merged into a centroid) and the router renormalized over survivors:
  $$\Delta\mathcal{L}(A) \;=\; \mathcal{L}_{\mathcal{D}}(\text{model}\setminus A) - \mathcal{L}_{\mathcal{D}}(\text{model}), \qquad E_{\text{eff}} \;=\; E - \max\{|A| : \Delta\mathcal{L}(A) \le \varepsilon\}.$$
  Computing $E_{\text{eff}}$ exactly is a subset search of cost $2^E$ per layer; with $E=64$ and 16 layers that is not runnable, so every reported number is a greedy or heuristic lower bound on removable mass.

Assumptions, and their violations:

1. *Weight-space distance tracks function-space distance.* Violated. Expert MLPs carry hidden-unit permutation and (with ReLU) positive-scaling symmetries, so $\|\theta_i - \theta_j\|$ is not identified; two functionally identical experts can be far apart in weight space (Entezari et al. 2022; Ainsworth et al. 2023).
2. *The router is fixed under intervention.* Violated. Removing $A$ changes the renormalized gates on *all* remaining tokens, so $\Delta\mathcal{L}(A)$ is not additive over experts and pairwise scores do not compose.
3. *$f_i$ is defined off-distribution.* Violated in the sense that matters: $f_i$ evaluated on tokens it never receives is unconstrained by training, so any similarity metric computed on the full corpus measures extrapolation, not redundancy.
4. *$\mathcal{D}$ is representative.* Violated for any multilingual or code-heavy model — redundancy measured on English web text is not redundancy for the deployed mixture.

## 3. State of the Art

**Empirical / systems SOTA.**
- *Expert pruning.* Chen et al., *Task-Specific Expert Pruning for Sparse Mixture-of-Experts* (2022) drop experts progressively during task fine-tuning down to one expert per layer with small task-metric loss — established for fine-tuned single-task settings, **not** for general-purpose pretrained mixtures.
- Lu et al., *Not All Experts are Equal: Efficient Expert Pruning and Skipping for MoE LLMs* (ACL 2024) prune experts in Mixtral 8x7B by a loss-based combinatorial search over small candidate sets. The reported retention at 25% expert removal is a **benchmark number**; the paper does not show that any cheap redundancy statistic predicts which experts the search selects.
- *Expert merging.* Li et al., *Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy* (MC-SMoE, ICLR 2024) group experts by routing-policy similarity, merge within group, then apply low-rank plus sparse decomposition; reports large memory reductions on fine-tuned Switch-base models. Established: routing similarity is *a usable grouping signal*. Unablated: whether it beats a random-grouping control at matched merge budget on a pretrained, non-fine-tuned model.

**Theory SOTA.** Thin. There is no theorem stating when top-$k$ routing with a load-balancing auxiliary loss must converge to duplicate experts. The nearest results are (i) permutation-symmetry / linear-mode-connectivity results (Entezari et al. 2022; Ainsworth et al. 2023), which establish the *non-identifiability* that blocks weight-space measurement, and (ii) the head-redundancy analogue, Michel, Levy & Neubig, *Are Sixteen Heads Really Better than One?* (NeurIPS 2019), which established that greedy importance scores over attention heads permit large removals — and that per-unit scores are poor predictors of joint removal cost.

**Similarity-metric SOTA.** CKA (Kornblith et al., ICML 2019) is the default representational-similarity tool, and Davari et al., *Reliability of CKA as a Similarity Measure in Deep Learning* (ICLR 2023), show CKA can be driven to arbitrary values by low-rank perturbations that leave function largely intact. Any expert-similarity score built on CKA inherits this.

## 4. What Is Known

- **Domain specialization is weak in large decoder MoEs.** Jiang et al., *Mixtral of Experts* (2024), $E=8$, $k=2$, 32 layers: routing over subsets of The Pile shows no clear topic-to-expert assignment; consecutive tokens are routed to the same expert far above the $1/8$ chance rate (reported as high repetition, strongest in the first and last layers). Scale: 47B total / 13B active parameters.
- **Positional and syntactic specialization is real in encoder MoEs.** Zoph et al., *ST-MoE* (2022) report encoder experts specializing on sentinel tokens, punctuation, conjunctions, and numbers; decoder experts show little specialization. Scale: up to 269B parameters.
- **Fine-grained experts plus a shared expert increase measured specialization.** Dai et al., *DeepSeekMoE* (ACL 2024): disabling top routed experts degrades DeepSeekMoE-16B more sharply than a comparable-activation dense-like baseline, which the authors read as *lower* redundancy. Scale: 2B and 16B total parameters.
- **Router saturation.** Muennighoff et al., *OLMoE* (2024), $E=64$, $k=8$, 16 layers, 1B active / 7B total: routing decisions largely fix early in pretraining (saturation rising steeply within the first tens of billions of tokens), and expert co-activation is low — experts are used in near-disjoint combinations. This is the strongest published evidence that co-activation and functional redundancy are different quantities.
- **Greedy per-unit importance underestimates joint redundancy** — established for attention heads (Michel et al. 2019) and inherited by every greedy expert-pruning pipeline.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** No published redundancy score has been shown to *predict* $\Delta\mathcal{L}(A)$ out of sample. Scores in use — routing-policy cosine, co-activation, weight $\ell_2$, CKA — are each defeated by a known symmetry or artifact (§2, §3). The field has a compression ratio, not a measurement.
- **Methodologically blocked.** No convention for the router under intervention: renormalize surviving gates, re-fit $W_r$, or re-train? Reported numbers mix all three, so they are not comparable across papers.
- **Empirically open.** Whether redundancy grows, shrinks, or is invariant with $E$ at fixed active parameters. Runnable: train $E \in \{8,16,32,64,128\}$ at fixed $k\cdot d_{\text{ff}}$ and identical token budget, measure $E_{\text{eff}}/E$. Nobody has published the sweep with a matched control.
- **Empirically open.** Whether redundancy is a *training-time* choice: does an explicit expert-diversity penalty reduce $E_{\text{eff}}/E$ without costing loss, or is redundancy load-bearing?
- **Theoretically open.** No proof either way that top-$k$ routing with load balancing has duplicate-expert solutions as attractors, nor any lower bound on the number of experts needed to represent a distribution with $m$ modes.

## 6. Why It Is Hard

Three specific obstructions, not one.

1. **Non-identifiability.** Hidden-unit permutations and ReLU rescalings mean the expert-to-expert map is only defined up to a symmetry group of size $(d_{\text{ff}})!$ per expert. Weight-space metrics are therefore uninterpretable without solving an alignment problem (Git Re-Basin-style) per pair — $\binom{E}{2}$ alignments per layer, and alignment quality itself becomes a confound.
2. **Absent ground truth at tractable cost.** The definition of $E_{\text{eff}}$ is a $2^E$ subset search with an $\mathcal{L}_{\mathcal{D}}$ evaluation inside. Every published number is a greedy proxy, so a "redundancy score" is validated against another heuristic, not against truth.
3. **Confounded measurement domain.** Expert $i$'s function is only constrained on $S_i$. Comparing $f_i$ and $f_j$ requires $S_i \cap S_j$, which top-$k$ routing makes small and *systematically biased* toward boundary tokens — exactly the tokens where the router is least confident. The metric is thus computed on the least representative slice of the data.

## 7. Current Research (as of 2026)

- **Structured merging with router repair** — merging experts and then re-fitting the router on a small calibration set, rather than renormalizing gates. Follow-on to MC-SMoE; several groups pursuing. *(frontier — verify)*
- **Shared-expert architectures as redundancy control** (DeepSeek line, Qwen MoE line): factor the common component out by construction so that residual redundancy among routed experts is small by design. Established as an architecture; the redundancy *measurement* claim remains an ablation on knockout curves.
- **Fine-grained expert segmentation at large $E$** ($E \ge 128$, $k \ge 8$) — makes the measurement problem worse: as $E$ grows, $|S_i \cap S_j|$ shrinks and pairwise functional comparison loses statistical power.
- **Causal-scrubbing / activation-patching adaptations from interpretability** applied to experts, to get intervention-based rather than similarity-based redundancy. *(frontier — verify)*
- **Upcycling** (dense checkpoint → MoE by expert duplication) offers a rare source of *ground-truth* redundancy: experts start identical and diverge measurably. Underexploited as a metric-validation testbed.

## 8. Concrete Next Experiment

**Goal:** decide whether any cheap redundancy score predicts compression cost, using upcycling to supply ground truth.

**Scale.** One MoE, $E=32$, $k=4$, ~1B active / ~7B total parameters, 16 layers, trained on 100B tokens — small enough that the full pipeline fits in roughly 2–4k H100-hours, large enough that specialization has emerged (OLMoE shows saturation well inside this budget).

**Protocol.**
1. Train two checkpoints: (a) from scratch; (b) upcycled from a dense 1B model by duplicating the FFN into 32 experts, so redundancy at step 0 is exactly $E_{\text{eff}}=1$ and decreases monotonically as a *known* baseline.
2. Compute five scores per layer at the end of training: routing-policy cosine, co-activation $\phi_{ij}$, permutation-aligned weight distance, CKA on $S_{ij}$, and the on-support functional distance $d_{ij}$ of §2.
3. Ground truth: for each layer, exhaustively evaluate all $\binom{32}{|A|}$ removals for $|A| \le 3$ (4,000 evaluations per layer, ~1M-token eval each — cheap) plus a 500-sample random subset for $|A| \in \{4,8,12\}$. Record $\Delta\mathcal{L}(A)$ with gates renormalized, no re-training.

**Control arm.** Random expert grouping at matched removal budget, plus a load-only baseline (remove the lowest-$p_i$ experts). Every score must beat both.

**Deciding number.** Out-of-layer $r^2$ between the score's predicted $\Delta\mathcal{L}(A)$ and the measured $\Delta\mathcal{L}(A)$ for $|A|=8$, fitted on layers 1–8 and tested on layers 9–16. **A score with test $r^2 \ge 0.7$ that also beats the random control by $\ge 0.15$ nats at $|A|=8$ unblocks the measurement variant. Below $r^2 \approx 0.3$, the score is a post-hoc description, and the field should stop reporting it as redundancy.**

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[Foundational]** Michel, Levy, Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS, 2019. — arXiv:1905.10650
- **[SOTA]** Li, Zhang, Yu, Fu, Zhao. *Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy.* ICLR, 2024. — arXiv:2310.01334
- **[SOTA]** Lu, Liu, Zhang, Wang, et al. *Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models.* ACL, 2024. — arXiv:2402.14800
- **[SOTA]** Dai, Deng, Zhao, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL, 2024. — arXiv:2401.06066
- **[Empirical]** Muennighoff, Soldaini, Groeneveld, et al. *OLMoE: Open Mixture-of-Experts Language Models.* 2024. — arXiv:2409.02060
- **[Empirical]** Jiang, Sablayrolles, Roux, et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Empirical]** Zoph, Bello, Kumar, et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* 2022. — arXiv:2202.08906
- **[Method]** Chen, Zhang, Gu, et al. *Task-Specific Expert Pruning for Sparse Mixture-of-Experts.* 2022. — arXiv:2206.00277
- **[Measurement]** Kornblith, Norouzi, Lee, Hinton. *Similarity of Neural Network Representations Revisited.* ICML, 2019. — arXiv:1905.00414
- **[Measurement]** Davari, Horoi, Natik, Lajoie, Wolf, Belilovsky. *Reliability of CKA as a Similarity Measure in Deep Learning.* ICLR, 2023. — arXiv:2210.16156
- **[Symmetry]** Entezari, Sedghi, Saukh, Neyshabur. *The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks.* ICLR, 2022. — arXiv:2110.06296
- **[Symmetry]** Ainsworth, Hayase, Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR, 2023. — arXiv:2209.04836
- **[Survey]** Cai, Jiang, Wang, et al. *A Survey on Mixture of Experts in Large Language Models.* IEEE TKDE, 2025. — arXiv:2407.06204

## 10. Worked Example

Take one layer of a Mixtral-style block: $E=8$, $k=2$, $d_{\text{model}}=4096$, $d_{\text{ff}}=14336$. Run 1M tokens through it.

**Step 1 — co-activation says "no redundancy."** Under $k=2$ with balanced load, the chance co-activation rate for a pair is $\phi_{ij}\approx 1$ by construction of the normalization. Measured $\phi_{ij}$ across the 28 pairs sits in a narrow band; nothing looks duplicated. Load $p_i$ is likewise near $2/8 = 0.25$ for all $i$ — that is what the load-balancing loss was *trained to produce*. **The load-balancing term destroys exactly the signal a load-based redundancy score reads.**

**Step 2 — support sizes collapse the functional metric.** With $p_i = 0.25$ and near-independent routing, $|S_{ij}| \approx 10^6 \times 0.25 \times 0.25 \approx 62{,}500$ tokens per pair. That sounds fine at $E=8$. Repeat at OLMoE's shape, $E=64$, $k=8$: $p_i = 0.125$, so $|S_{ij}| \approx 15{,}600$ — but OLMoE reports *low* co-activation, meaning $\phi_{ij} \ll 1$ for most pairs and the true intersection is an order of magnitude smaller, a few hundred tokens for a $4096$-dimensional output comparison. The estimator for $d_{ij}$ is then noise-dominated.

**Step 3 — the two metrics disagree, and the ground truth is unreachable.** Suppose experts 3 and 6 score $d_{36}=0.11$ (most similar pair) while weight-space distance ranks them 19th of 28. Which is right? The test is $\Delta\mathcal{L}(\{3,6\}\to\text{merged})$. Merging one pair is one evaluation. But the practitioner's question is "which 4 of 8 do I drop?", which is $\binom{8}{4}=70$ evaluations at this layer — and *jointly across 32 layers* it is $70^{32}$, so the joint optimum is never computed. Published pipelines do the greedy thing, and Michel et al. (2019) already showed for attention heads that greedy per-unit scores mis-rank joint removals.

**The visible obstruction.** Every quantity that is cheap to compute ($p_i$, $\phi_{ij}$, weight distance) is either regularized flat by training or unidentified up to permutation. The one quantity that is meaningful ($\Delta\mathcal{L}$ over expert subsets) is combinatorial. The literature has been reporting the cheap numbers as if they were the meaningful one. That substitution — not compute, not data — is what blocks the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*