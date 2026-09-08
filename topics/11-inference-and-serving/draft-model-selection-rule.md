---
id: 11-inference-and-serving/draft-model-selection-rule
title: "Draft Model Selection Rule for Speculative Decoding"
topic: 11-inference-and-serving
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Draft Model Selection Rule for Speculative Decoding

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/draft-model-selection-rule` · **Status:** empirically-open

## 1. Problem Statement

Speculative decoding pairs a large target model $M_t$ with a cheap draft model $M_d$ that proposes $\gamma$ tokens per step; a rejection-sampling verification pass keeps the output distribution exactly equal to $M_t$'s. Given a target model, a serving configuration (hardware, tensor-parallel degree, batch size, context length), and a workload distribution, **which draft should you pick?**

Three variants, different difficulty:

- **Measurement.** Given a candidate pair $(M_t, M_d)$ and a workload, estimate the end-to-end speedup without running full serving benchmarks. Requires a statistic of $M_d$ that predicts acceptance under the *deployed* sampling parameters.
- **Method.** Given a family of candidate drafts (model-zoo checkpoints, distilled students, self-speculative layer-skips, EAGLE/Medusa-style feature heads), output the argmax of goodput under a fixed training + serving budget. No published rule does this; practice is grid search.
- **Theory.** Is there a decision rule — a function of draft size, draft–target divergence, and hardware cost ratio — that provably selects the optimal draft up to a bounded regret, without evaluating every candidate?

A solution is a rule that, on a held-out set of (target, hardware, workload) triples, picks a draft within a stated percentage of the best-of-grid measured speedup, and whose inputs cost materially less than the grid it replaces.

## 2. Formal Setting

Let $p(\cdot \mid x)$ be the target's next-token distribution and $q(\cdot \mid x)$ the draft's, both after the deployed temperature/top-$p$ transform (this matters: acceptance is defined on the *sampled* distributions, not the raw logits).

**Acceptance rate.** The per-token accept probability under standard speculative sampling is
$$\alpha(x) = 1 - \tfrac{1}{2}\lVert p(\cdot\mid x) - q(\cdot\mid x)\rVert_1 = \sum_{v}\min\big(p(v\mid x), q(v\mid x)\big).$$
*As measured:* run the deployed sampler on $N$ prompts, log accept/reject per drafted position, and report $\hat\alpha = (\text{accepted})/(\text{drafted})$. Not the same as top-1 agreement rate, which is what many papers actually report.

**Cost ratio.** $c = T_d / T_t$, where $T_d$ and $T_t$ are wall-clock latencies per forward pass of draft and target *in the deployed configuration*. Measured, not derived from parameter counts.

**Speedup.** With i.i.d. acceptance and draft length $\gamma$, the expected number of tokens per verification block is $(1-\alpha^{\gamma+1})/(1-\alpha)$, and the block costs $c\gamma + 1$ target-forwards, giving Leviathan et al.'s (ICML 2023) formula
$$S(\alpha, \gamma, c) = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(c\gamma+1)}, \qquad \gamma^\star = \arg\max_\gamma S.$$

**Selection objective.** Over candidate drafts $\mathcal{D}$, maximize goodput $G$ (tokens/s/GPU) at a latency SLO, subject to $M_d$'s memory sharing the device with $M_t$ and its KV cache:
$$M_d^\star = \arg\max_{M_d \in \mathcal{D}} \ \max_{\gamma} \ G(M_d, \gamma \mid \text{batch } B, \text{context } L).$$

**Assumptions known violated in practice:**
1. *Acceptance is i.i.d. across positions* — false; rejections cluster at high-entropy positions, so realized block length has heavier-than-geometric variance.
2. *$c$ is constant* — false; at $B \gtrsim 16$ the target becomes compute-bound, verification of $\gamma$ tokens is no longer free, and effective $c$ rises.
3. *Draft cost is proportional to parameters* — badly false; a 1B draft on an H100 is memory-latency-bound and costs far more than $1/70$ of a 70B target.
4. *One $\alpha$ per pair* — false; $\alpha$ varies 0.2+ across Spec-Bench subtasks (translation vs. summarization vs. code) for the same pair.

## 3. State of the Art

**Theory SOTA (established).** The $S(\alpha,\gamma,c)$ formula and the losslessness proof (Leviathan et al., ICML 2023; Chen et al., 2023) are correct and reproduced. SpecTr (Sun et al., NeurIPS 2023) frames multi-draft verification as optimal transport and gives an optimal-acceptance draft-selection rule *given* the draft. Optimality is over verification, not over which draft to use.

**Systems SOTA (established).** Tree-structured drafting — SpecInfer (ASPLOS 2024), Sequoia (NeurIPS 2024), EAGLE-2 (EMNLP 2024) — beats linear drafting at equal draft cost. Sequoia is the closest thing to a selection rule: it contains a hardware-aware solver that picks tree shape and size from measured $(\alpha, c)$ on the target device. It selects *tree topology*, not *which draft model*.

**Claimed but unablated.** Reported speedups (EAGLE 2.7–3.5×, Medusa 2.2–2.8×, EAGLE-3 higher) are near-universally at batch size 1 with a single draft choice; the counterfactual "same compute spent on a different draft" is not run. Papers that compare drafts do so at a *shared* $\gamma$, which is not the comparison the deployment cares about (see §10). Scaling-law-style claims that "draft quality should scale with target size" exist as folklore, not as a fitted curve.

**Benchmark-number-only results.** Spec-Bench (Xia et al., ACL Findings 2024) provides the one common harness — Vicuna-7B/13B/33B, six subtasks, single A100 — but its leaderboard is a ranking of *methods*, not a dataset from which a draft-selection rule could be fit.

## 4. What Is Known

- $S$ is concave in $\gamma$ with an interior optimum; $\gamma^\star$ falls as $c$ rises. Established analytically and confirmed empirically.
- Distillation raises $\alpha$: DistillSpec (ICLR 2024) reports 10–45% additional speedup over an undistilled draft on T5 and GPT-like targets, with the gain largest when the base draft is weakest.
- Original results: T5-XXL 11B with a T5-small draft, 2–3× (Leviathan et al.); Chinchilla 70B with a 4B draft, 2–2.5× (Chen et al.). Both batch 1.
- Feature-level drafting beats token-level at equal draft parameter count: EAGLE (ICML 2024) on Vicuna/LLaMA-2-Chat 7B–70B, batch 1, single A100/H100.
- Batch-size collapse: independent reports (SmartSpec, 2024; vLLM production measurements) show naive speculative decoding falling *below* 1× at large batch, because verification competes with real requests for the same compute. MagicDec (2024) shows the sign flips back positive at long context ($\gtrsim$ 32k), where the target is KV-bandwidth-bound.
- Self-speculation works without a separate model: Draft & Verify (ACL 2024) reaches ~1.3–1.6× on LLaMA-2 by skipping layers — a lower ceiling but zero extra memory.

## 5. What Is Not Known

- **Empirically open (the core gap).** Nobody has published a factorial sweep of draft candidates × target sizes × batch sizes × context lengths on one harness with measured $\alpha$ and $c$. The experiment is entirely runnable on existing hardware; it has not been run at the scale needed to *fit* a rule. This is why the page status is `empirically-open`.
- **Theoretically open.** No regret bound for any draft-selection procedure. No proof of whether $\alpha$ is a monotone function of draft capacity at fixed architecture family, or whether it saturates. No characterization of when a cheap proxy for $\alpha$ (e.g., draft perplexity under target-generated text) is sufficient.
- **Methodologically blocked.** "Draft quality" has no agreed measurement. Top-1 agreement, $\hat\alpha$ under the deployed sampler, and mean accepted length are all called "acceptance rate" in the literature and are numerically different. Until the field fixes one, cross-paper numbers are not comparable.

## 6. Why It Is Hard

**Confounded measurement plus a non-transferable constant.** $\alpha$ and $c$ are both properties of the deployment, not of the model pair. $\alpha$ moves with temperature, top-$p$, prompt distribution, and even the position within a response. $c$ moves with GPU, tensor-parallel degree, kernel implementation, and batch size — and the parameter ratio is a poor predictor because small drafts sit in the memory-bound regime where latency barely falls with size. So a measured speedup on an A100 at batch 1 does not transfer to an H100 at batch 32, and the ranking of drafts — not just the magnitude — can invert.

Compounding this: the search space is a product of *which draft*, *how it was trained*, *tree shape*, and *$\gamma$*, and the objective is not separable across them. Comparing two drafts at a shared $\gamma$ answers a question nobody asked.

## 7. Current Research (as of 2026)

- **Adaptive draft length.** SpecDec++ (2024) and AdaEDL-style entropy-triggered stopping learn when to halt drafting per step, removing $\gamma$ from the search space. Extension to *switching between drafts* mid-request is active *(frontier — verify)*.
- **Goodput-aware serving.** SmartSpec (Liu et al., 2024) and the vLLM/SGLang teams schedule speculation as a function of queue depth. This turns the selection question into an online control problem.
- **Online/continual drafts.** Online Speculative Decoding (ICML 2024) trains the draft on live query distributions using idle serving compute, which makes "which draft" partly moot but adds a training-budget axis.
- **EAGLE-3 and successors** (2025) argue that draft heads follow a scaling law in training data — the first published claim resembling a selection rule *(frontier — verify)*.
- Groups with sustained output: Google (Leviathan, SpecTr), CMU/Catalyst (SpecInfer, Sequoia, MagicDec), Peking University (EAGLE line), Princeton (Medusa), Berkeley Sky Computing (SmartSpec, vLLM).

## 8. Concrete Next Experiment

**Fit the rule, then test it out of distribution.**

- **Scale.** One target, Llama-3.1-70B-Instruct, on 4×H100. Six drafts: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, a distilled 1B, an EAGLE-style head, and self-speculative layer-skip. Grid over $\gamma \in \{1,\dots,10\}$, batch $B \in \{1, 8, 32\}$, context $L \in \{2\text{k}, 32\text{k}\}$, on all six Spec-Bench subtasks. That is 6 × 10 × 3 × 2 × 6 ≈ 2,160 configurations; each is minutes of serving, so ~2 GPU-days on one node.
- **Log per config:** $\hat\alpha$ under the deployed sampler, $T_d$, $T_t$, realized goodput.
- **Control arm.** Predicted $S(\hat\alpha, \gamma, c)$ from batch-1, 2k-context measurements only — i.e. exactly what current papers report — extrapolated to every other cell.
- **The deciding number.** The **rank correlation (Kendall's $\tau$) between the control arm's predicted draft ranking and the measured goodput ranking, within each $(B, L)$ cell.** If $\tau \geq 0.8$ everywhere, batch-1 measurement is a valid selection rule and the problem is closed cheaply. If $\tau$ drops below ~0.4 in the $(B{=}32, L{=}32\text{k})$ cell — the prediction from §2's violated assumptions — then draft selection is irreducibly per-deployment, and the field's headline speedup numbers do not license any deployment choice.

## 9. Key References

- **[Foundational]** Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- **[Foundational]** C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, J. Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. — arXiv:2302.01318
- **[SOTA]** X. Miao et al. *SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.* ASPLOS, 2024. — arXiv:2305.09781
- **[SOTA]** Y. Li, F. Wei, C. Zhang, H. Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML, 2024. — arXiv:2401.15077
- **[SOTA]** Z. Chen et al. *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding.* NeurIPS, 2024. — arXiv:2402.12374
- **[SOTA]** T. Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML, 2024. — arXiv:2401.10774
- **[Method]** Y. Zhou et al. *DistillSpec: Improving Speculative Decoding via Knowledge Distillation.* ICLR, 2024. — arXiv:2310.08461
- **[Method]** Z. Sun et al. *SpecTr: Fast Speculative Decoding via Optimal Transport.* NeurIPS, 2023.
- **[Method]** J. Zhang et al. *Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding.* ACL, 2024. — arXiv:2309.08168
- **[Systems]** X. Liu et al. *Optimizing Speculative Decoding for Serving Large Language Models Using Goodput.* 2024.
- **[Systems]** J. Sadhukhan et al. *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding.* 2024.
- **[Survey]** H. Xia et al. *Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding.* Findings of ACL, 2024. — arXiv:2401.07851

## 10. Worked Example

Target: 70B at TP=4, $T_t = 12$ ms/token. Two drafts, both memory-bound, so their latencies are far above the parameter ratio:

| Draft | $T_d$ | $c = T_d/T_t$ | $\hat\alpha$ |
|---|---|---|---|
| A: 1B | 3.0 ms | 0.250 | 0.72 |
| B: 3B | 4.5 ms | 0.375 | 0.82 |

Note first that $c_A = 0.25$, not $1/70 = 0.014$ — an 18× error if you select on parameter count.

Now apply $S(\alpha,\gamma,c)$:

| $\gamma$ | $S_A$ | $S_B$ | winner |
|---|---|---|---|
| 2 | **1.49** | 1.42 | A |
| 3 | **1.49** | 1.43 | A |
| 5 | 1.37 | 1.35 | A |
| 8 | 1.13 | **1.16** | B |

At each draft's own optimum ($\gamma^\star_A \approx 3$, $\gamma^\star_B \approx 3$), A wins: 1.49 vs 1.43. At the fixed $\gamma = 8$ that several papers use as a default, B wins: 1.16 vs 1.13. **The ranking inverts on a hyperparameter that is not part of the question being asked.** A comparison at shared $\gamma$ is not a comparison of drafts.

Now raise batch to 32. Verification of $\gamma$ tokens no longer rides free on a memory-bound target; suppose effective $c$ doubles for both. At $c_A = 0.5$, $\gamma = 2$: $S_A = (1-0.72^3)/(0.28 \times 2) = 1.12$. Add the ~10% goodput lost to the draft's own KV cache and scheduler overhead and A is at parity with no speculation at all — while its batch-1 number still reads 1.49 in the paper. That gap between the reported number and the deployed number is the obstruction.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*