---
id: 15-mixture-of-experts/expert-pruning-without-retraining
title: "Expert Pruning Without Retraining"
topic: 15-mixture-of-experts
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Expert Pruning Without Retraining

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/expert-pruning-without-retraining` · **Status:** partially-solved

## 1. Problem Statement

Given a trained sparse Mixture-of-Experts (MoE) transformer, delete a fraction of the experts and the router rows that address them, **without any gradient step afterwards**, so that the resulting model keeps its general capability. Input: model weights, a small unlabeled calibration set, a target sparsity. Output: a per-layer subset of experts to keep, plus a repaired router. No fine-tuning, no distillation, no weight update beyond closed-form repair.

Three variants that are routinely conflated:

- **Measurement.** Given a candidate pruned model, decide whether it is "as good as" the parent. Perplexity on a held-out corpus is the standard proxy and it is a poor one — an MoE can hold perplexity within 2% while losing 15 points on a multi-step reasoning benchmark, because rare experts carry rare capabilities.
- **Method.** Find the keep-set. This is a combinatorial subset-selection problem, $\binom{E}{k}$ per layer, coupled across layers by routing.
- **Theory.** Prove a bound on the loss increase from removing an expert as a function of an observable statistic (load, gate mass, activation norm), under stated assumptions on the router.

The measurement and method variants are empirically open at frontier scale; the theory variant is open outright. "Solved" would mean: a calibration-only procedure that removes $\geq 50\%$ of expert parameters from a modern fine-grained MoE with $\leq 1$ point average drop across a held-out suite of 10+ tasks, reproduced by a third party.

## 2. Formal Setting

An MoE layer $\ell$ has experts $\{f^{\ell}_e\}_{e=1}^{E}$ with parameters $\theta^{\ell}_e$, a router $W^{\ell}_r \in \mathbb{R}^{d\times E}$, and top-$k$ gating. For token representation $x \in \mathbb{R}^d$:

$$g^{\ell}(x) = \mathrm{softmax}\big(\mathrm{top}_k(W_r^{\ell\top} x)\big), \qquad y = \sum_{e \in \mathcal{T}_k(x)} g^{\ell}_e(x)\, f^{\ell}_e(x).$$

A pruning decision is a mask $m^{\ell} \in \{0,1\}^{E}$ with $\sum_e m^\ell_e = k'$. The pruned layer renormalizes over survivors: $\tilde g^\ell(x) = \mathrm{softmax}(\mathrm{top}_k(W_r^{\ell\top}x + \log m^\ell))$, with $\log 0 = -\infty$. **Renormalization is itself a design choice** and is the single largest unablated confound in the literature: masking before vs. after softmax changes the surviving gate magnitudes and therefore the residual-stream scale.

Measured quantities, on a calibration set $\mathcal{D}_{\mathrm{cal}}$ of $N$ tokens:

- **Gate mass** $\mu^\ell_e = \frac{1}{N}\sum_{x} g^\ell_e(x)$ — the router's own credit assignment, computed in one forward pass.
- **Routing load** $p^\ell_e = \frac{1}{N}\sum_x \mathbf{1}[e \in \mathcal{T}_k(x)]$ — fraction of tokens touching $e$. Note $\mu$ and $p$ are not monotone in each other.
- **Output-contribution norm** $c^\ell_e = \frac{1}{N}\sum_x g^\ell_e(x)\|f^\ell_e(x)\|_2$ — requires running the expert, so $O(E)$ times the cost of $\mu$.
- **Layerwise reconstruction error** $\varepsilon^\ell(m) = \mathbb{E}_{x\sim\mathcal{D}_{\mathrm{cal}}}\|y^\ell(x) - \tilde y^\ell(x;m)\|_2^2$, the surrogate every greedy method actually minimizes.
- **End objective** $\Delta\mathcal{L} = \mathcal{L}_{\mathrm{pruned}} - \mathcal{L}_{\mathrm{parent}}$, cross-entropy nats/token on held-out text, plus a task-suite average $\Delta A$.

Standing assumptions, and their status:

1. *Layer separability* — $\varepsilon^\ell$ can be minimized independently per layer. **Violated:** routing at layer $\ell+1$ depends on $\ell$'s output, so errors compound multiplicatively down the stack.
2. *Calibration coverage* — $\mathcal{D}_{\mathrm{cal}}$ (typically 128–2048 sequences of C4 or WikiText) exercises every capability. **Violated by construction:** an expert specialized to code or a low-resource language receives near-zero load on English web text and is pruned first.
3. *Small-error linearity* — $\Delta\mathcal{L} \approx \sum_\ell \alpha_\ell \varepsilon^\ell$. Only defensible in the small-$\varepsilon$ regime; at $\geq 50\%$ sparsity the models leave it.
4. *Router calibration* — $g_e$ reflects marginal usefulness. Load-balancing auxiliary losses deliberately distort $g$ toward uniformity, so gate mass is partly an artifact of the training objective, not of expert value.

## 3. State of the Art

**Established (method side).**

- **Enumeration on layerwise reconstruction.** Lu et al., *Not All Experts are Equal* (ACL 2024), enumerate expert subsets per layer against $\varepsilon^\ell$ on a small calibration set for Mixtral-8x7B. Dropping 2 of 8 experts per layer is retained with modest degradation on commonsense QA; at 4 of 8 the drop is large. Ablated against random and load-based baselines — this is the cleanest apples-to-apples comparison in the area.
- **Domain-conditional pruning works far better than task-agnostic pruning.** Koishekenov et al. (ACL 2023) prune language-specific experts from NLLB-200 (54.5B, 128 experts/layer) and report removing up to ~80% of experts for a fixed translation direction with small quality loss. The result is real but *conditional*: the pruned model is no longer the general model.
- **Merging beats deleting at matched sparsity.** Li et al., MC-SMoE (ICLR 2024), group experts by routing-policy similarity and merge before compressing; reported large memory reductions with better retention than dropping. Merging is a weight update, so it sits at the boundary of "without retraining" — no gradients, but the parent weights change.

**Claimed but unablated.** Most 2024–2025 papers report a single sparsity point on one model family (usually Mixtral-8x7B) with a fixed calibration set, and do not report seed variance, calibration-set sensitivity, or the masking-vs-renormalization choice. SEER-MoE (Muzio et al., 2024) and EEP (Liu et al., 2024) both report strong numbers; the ablation over calibration corpus is missing in most of this line, so the reported gap to a load-based baseline is not clearly attributable to the scoring function.

**Benchmark-number-only.** Nearly all headline retention figures are reported on 4–6 zero-shot multiple-choice tasks (ARC, HellaSwag, PIQA, WinoGrande, MMLU). These saturate near chance-plus and are insensitive to exactly the long-tail capability loss pruning induces. Treat them as a floor test, not evidence of preserved capability.

**Theory SOTA.** There is no expert-level analogue of the optimal-brain-surgeon-style closed form used by SparseGPT (Frantar & Alistarh, ICML 2023) for dense weights. Expert removal is a discrete, non-smooth change to a routed function; the Hessian-based local quadratic model does not apply across a top-$k$ discontinuity.

## 4. What Is Known

- **Expert utilization is heavily skewed.** In Mixtral-8x7B, routing is closer to token-syntactic than to topic-semantic, and expert specialization is weak in middle layers and stronger at the extremes (Lo et al., 2024; Jiang et al., 2024, the Mixtral report, which found no obvious topic-level specialization on The Pile).
- **Depth matters more than width of pruning.** Across reported results, early and final MoE layers tolerate expert removal far less than middle layers — consistent with the observation that middle-layer experts are more redundant.
- **Random pruning is a surprisingly strong baseline at low sparsity.** At 1-of-8 removal on Mixtral-scale models, the gap between random and best-scored selection is small relative to seed noise; the gap opens at $\geq 37.5\%$.
- **Fine-grained MoEs are harder to prune per-expert.** DeepSeekMoE (Dai et al., 2024) uses many small experts plus always-on shared experts; with 64+ routed experts per layer, per-expert redundancy is lower and the combinatorial search space is larger.
- **Dense-pruning transfer is partial.** Wanda-style scores (Sun et al., ICLR 2024) applied *within* experts (unstructured intra-expert sparsity) compose reasonably with expert-count reduction, but the two are usually reported separately, not jointly.
- **Retraining recovers most of the loss.** Task-specific pruning down to one expert per layer is achievable *with* fine-tuning (Chen et al., 2022). This bounds the value of the no-retraining constraint: the gap between pruned-and-retrained and pruned-only is the quantity of interest, and it is rarely reported.

## 5. What Is Not Known

- **Theoretically open.** No bound of the form $\Delta\mathcal{L} \leq \Phi(\mu, p, c)$ for any observable statistic $\Phi$, under any non-vacuous router assumption. Nobody has shown that optimal expert subset selection is NP-hard for a stated MoE family either — the hardness is assumed, not proved.
- **Empirically open.** Whether the ranking of scoring functions (gate mass vs. reconstruction error vs. contribution norm) is stable across (i) model scale from 8×7B to 400B+, (ii) coarse vs. fine-grained expert granularity, (iii) calibration corpus. Every ingredient exists; the 3-way sweep has not been run and published at frontier scale.
- **Methodologically blocked.** "Capability retained" has no agreed measurement. Perplexity on English web text is insensitive to the failure mode; the standard zero-shot suite is insensitive too. Until there is an evaluation that is *provably sensitive* to long-tail expert removal — e.g. a stratified suite covering low-resource languages, rare code paths, and rare formats — a reported "1% drop" cannot be distinguished from "large drop on 1% of the input distribution."

## 6. Why It Is Hard

The controlling obstruction is **confounded measurement compounded by calibration-set non-coverage**. The selection signal and the evaluation are drawn from the same narrow distribution, so the procedure is self-confirming: experts that see no calibration tokens are pruned, and their absence is then invisible to an evaluation drawn from the same corpus. This is not a compute problem — it is a design flaw that more compute reproduces faithfully at larger scale.

Second: **non-identifiability of expert value under load balancing.** The auxiliary balancing loss pushes $\mu_e$ toward $1/E$ during training, so gate mass measures the training objective's pressure as much as the expert's marginal contribution. Two models with identical capability can have very different $\mu$ profiles depending on the balancing coefficient.

Third: **coupling.** The top-$k$ operator makes pruning at layer $\ell$ change the routing *distribution* at layer $\ell+1$, not just its input. Greedy layerwise selection is therefore optimizing a surrogate whose gradient with respect to the true objective is not sign-guaranteed.

## 7. Current Research (as of 2026)

- **Calibration-free / activation-statistic methods** that estimate expert importance from weights alone, avoiding the coverage problem by not sampling data at all. Promising in principle; weak in practice so far because weight norms do not encode routing frequency. *(frontier — verify)*
- **Merge-then-prune pipelines** (successors to MC-SMoE) that treat deletion as the degenerate case of merging with zero weight. Active at NUS/UT Austin-adjacent groups.
- **Router repair as a closed-form problem**: after masking, re-solving $W_r$ rows by least squares against the parent's gate outputs, rather than renormalizing. This is the most likely near-term source of real gains and is under-explored. *(frontier — verify)*
- **Serving-side alternatives** — expert offloading and caching (keeping all experts, paging cold ones to CPU/NVMe) — which sidestep the accuracy question entirely and are, for many deployments, the correct answer. Any pruning paper that does not compare against offloading at matched GPU memory is answering the wrong question.

## 8. Concrete Next Experiment

**Question:** is the reported advantage of reconstruction-error scoring over gate-mass scoring a property of the method, or of the calibration corpus?

**Scale.** Mixtral-8x7B (32 layers × 8 experts, top-2) and one fine-grained model (DeepSeek-family, 64 routed experts). Sparsity: 25% and 50% of experts removed. Cost: roughly 200–400 A100-hours, dominated by evaluation, not selection.

**Design.** Cross the scoring function $\{$gate mass $\mu$, reconstruction error $\varepsilon$, contribution norm $c$, random$\}$ with the calibration corpus $\{$C4-English, StarCoder-subset, multilingual (FLORES-200 dev)$\}$, 3 seeds each. Fix the renormalization rule and report the alternative as a separate arm.

**Control arm.** Random expert selection at matched sparsity, same seeds, same renormalization. This is the arm most papers omit, and it is the only one that calibrates seed noise.

**Evaluation.** A *stratified* suite: 30% standard zero-shot QA, 30% code (HumanEval + MBPP), 30% multilingual (FLORES-200, 20 directions including 5 low-resource), 10% long-context retrieval. Report per-stratum, never only the mean.

**The deciding number.** $\rho$ — the Kendall rank correlation between the four methods' per-stratum score rankings under C4 calibration and under the mismatched calibration corpora. If $\rho \geq 0.8$, the scoring function is a genuine property and the field's comparisons are valid. If $\rho \leq 0.4$, every published expert-pruning ranking is a statement about C4, not about pruning, and the sub-field needs re-basing on stratified calibration.

## 9. Key References

- **[Foundational]** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[SOTA]** Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, Hongsheng Li. *Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models.* ACL, 2024. — arXiv:2402.14800
- **[SOTA]** Yeskendir Koishekenov, Alexandre Berard, Vassilina Nikoulina. *Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model.* ACL, 2023. — arXiv:2212.09811
- **[SOTA]** Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, Tianlong Chen. *Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy.* ICLR, 2024. — arXiv:2310.01334
- **[Method]** Tianyu Chen, Shaohan Huang, Yuan Xie, Baosong Yang, Daxin Jiang, Linjun Yang, Rangan Majumder, Furu Wei. *Task-Specific Expert Pruning for Sparse Mixture-of-Experts.* 2022. — arXiv:2206.00277
- **[Baseline]** Elias Frantar, Dan Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML, 2023. — arXiv:2301.00774
- **[Baseline]** Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter. *A Simple and Effective Pruning Approach for Large Language Models.* ICLR, 2024. — arXiv:2306.11695
- **[Analysis]** Albert Q. Jiang et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Analysis]** Damai Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL, 2024. — arXiv:2401.06066
- **[Survey]** Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, Jiayi Huang. *A Survey on Mixture of Experts in Large Language Models.* IEEE TKDE / preprint, 2024. — arXiv:2407.06204

## 10. Worked Example

Take Mixtral-8x7B: 32 MoE layers, $E=8$, top-2, ~45B total parameters of which ~96% sit in expert FFNs. Target: remove 2 experts per layer, a 25% cut, ~11B parameters, enough to move an 8-bit deployment from two 80 GB GPUs to one.

**Selection.** Run 128 sequences of 2048 C4 tokens — 262k tokens — and record $\mu^\ell_e$. With top-2 over 8 experts and a balancing loss, the ideal load is $p_e = 0.25$. Measured spreads in this family are roughly $p_e \in [0.18, 0.32]$: skewed, but not by much. Rank by $\mu$, drop the bottom two per layer.

**The check that exposes the problem.** Recompute $\mu^\ell_e$ on 128 sequences of Python from a code corpus. For a given layer, suppose the C4 ranking is $e_3 > e_7 > e_1 > \dots > e_5 > e_2$ (drop $e_5, e_2$) but the code ranking promotes $e_2$ from 8th to 2nd — a plausible outcome given that Mixtral's routing correlates with token syntax, and code has a distinct token distribution. The C4-selected mask deletes an expert that carries a third of the gate mass on code tokens.

**What the standard evaluation reports.** WikiText perplexity moves from 3.84 to roughly 4.0 — about 4%, comfortably inside the "negligible" band papers use. The 5-task zero-shot average moves ~1 point, within seed noise. Both pass.

**What a stratified evaluation would report.** HumanEval pass@1 on the same checkpoint falls disproportionately, because the deleted expert was never exercised by the selection signal *or* by the acceptance test. The parent's ~40% pass@1 does not degrade by 4%; it degrades by an amount nobody has measured, because the measurement is not standard practice.

**The obstruction, made concrete.** The selection statistic and the acceptance statistic are both expectations under $\mathcal{D}_{\mathrm{cal}}$. Any capability whose support has near-zero mass under $\mathcal{D}_{\mathrm{cal}}$ is simultaneously the *most likely to be pruned* and the *least likely to be detected*. The two errors are perfectly correlated by construction — which is why the field can report 25% pruning at negligible cost and still not know what was lost.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*