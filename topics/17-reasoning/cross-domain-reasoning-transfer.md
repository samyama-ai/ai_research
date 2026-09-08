---
id: 17-reasoning/cross-domain-reasoning-transfer
title: "Transfer of Reasoning Skill Across Domains"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Transfer of Reasoning Skill Across Domains

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/cross-domain-reasoning-transfer` · **Status:** empirically-open

## 1. Problem Statement

Post-training a language model on one reasoning domain — competition mathematics, say — reliably raises scores on other domains: code, science QA, agentic planning, law. The open question is **what is transferring**.

Three competing accounts:

1. **Skill transfer.** The model acquires a domain-general procedure (decompose, verify, backtrack) that is then applied to new content.
2. **Format transfer.** The model learns to emit long, structured, self-checking text; the underlying knowledge was already in the base model and is merely elicited.
3. **Contamination/adjacency transfer.** Target domains share latent content with the source (arithmetic inside physics, symbol manipulation inside code), so "transfer" is within-distribution generalization mislabeled.

The problem in three variants:

- **Measurement.** Define a quantity $T$ that separates these three, computable from finite evaluations, and not confounded by base-model capability or benchmark leakage. *Currently the hardest variant.*
- **Method.** Find a training recipe maximizing transfer per unit of source-domain compute — i.e. that beats the elicitation ceiling of the base model on a genuinely disjoint target.
- **Theory.** Characterize the conditions on source and target task distributions under which gradient-based post-training provably improves target risk, rather than reweighting an existing solution distribution.

**Solved** would mean: a pre-registered protocol showing that training on source $S$ improves a target $T$ beyond what best-of-$n$ sampling plus format prompting extracts from the same base model, with the improvement surviving a decontaminated, content-disjoint target set.

## 2. Formal Setting

Let $\pi_\theta$ be an autoregressive policy. A *domain* is a distribution $\mathcal{D}$ over $(x, y^\star)$ with a verifier $v: (x, \hat y) \to \{0,1\}$. Define pass@1 under temperature $\tau$:

$$A(\pi, \mathcal{D}) = \mathbb{E}_{x \sim \mathcal{D}}\, \mathbb{E}_{\hat y \sim \pi(\cdot \mid x, \tau)}\, v(x, \hat y).$$

Measured as: $n \ge 32$ samples per item, $\ge 500$ items, reported with a bootstrap 95% CI over items (item variance dominates; per-item Bernoulli variance does not).

Post-training maps $\theta_0 \mapsto \theta_S$ using $C$ tokens of source-domain compute. Raw transfer:

$$\Delta_{S \to T}(C) = A(\pi_{\theta_S}, \mathcal{D}_T) - A(\pi_{\theta_0}, \mathcal{D}_T).$$

$\Delta$ alone conflates all three accounts. Define the **elicitation ceiling** of the base model on target $T$: the best target accuracy reachable from $\theta_0$ without gradients, given the same inference budget $B$ (tokens generated per item at evaluation) and access to the same verifier-free scaffolds:

$$E_T(B) = \max_{p \in \mathcal{P}} A\big(\pi_{\theta_0}(\cdot \mid p), \mathcal{D}_T\big),$$

over a fixed scaffold family $\mathcal{P}$ (prompt formats, self-consistency at $k$ samples with $kB' \le B$, reflection loops). Then the quantity of interest is **excess transfer**:

$$T_{\text{ex}} = A(\pi_{\theta_S}, \mathcal{D}_T) - E_T(B).$$

$T_{\text{ex}} > 0$ at matched $B$ is the evidence account (1) needs; $T_{\text{ex}} \approx 0$ with $\Delta > 0$ is account (2).

Coverage separates elicitation from acquisition. With pass@$k$ $= \mathbb{E}_x[1 - (1-p_x)^k]$ where $p_x$ is per-item success probability: if $\text{pass@}k(\pi_{\theta_S}) \le \text{pass@}k(\pi_{\theta_0})$ for large $k$ (say $k = 256$) while pass@1 rises, the post-training sharpened an existing distribution rather than adding solutions.

Domain disjointness must be measured, not asserted. Use $\rho_{S,T}$ = max $n$-gram / embedding overlap between source training items and target items ($n = 13$ substring match, plus nearest-neighbour cosine at a fixed encoder, thresholded), reported as the fraction of target items with any near-duplicate source item.

**Assumptions, and which fail.**
- *Verifier soundness*: $v$ marks exactly the correct answers. Violated — final-answer matching credits right-answer/wrong-reasoning, at rates of several percent on math sets.
- *Target uncontaminated by pretraining*: violated for essentially every public benchmark; unmeasurable without pretraining-corpus access.
- *$\mathcal{P}$ is rich enough that $E_T(B)$ is a true ceiling*: violated — $E_T$ is a lower bound on elicitability, so $T_{\text{ex}}$ is biased upward.
- *Matched inference budget*: routinely violated in published comparisons, where reasoning models emit 5–20$\times$ more tokens than the baselines they beat.

## 3. State of the Art

**Established.**
- Large-scale RL with verifiable rewards (RLVR) on math/code produces large in-domain gains and non-trivial out-of-domain gains: DeepSeek-R1-Zero moved AIME 2024 pass@1 from 15.6% to 71.0% during RL (86.7% with majority voting at 64 samples), with concurrent gains on GPQA Diamond and Codeforces (DeepSeek-AI, *Nature*, 2025).
- Test-time compute scaling gives large gains without weight updates on the target (Snell et al., 2024), which is precisely why $E_T(B)$ must be part of any transfer claim.
- Chu et al. (2025) is the strongest controlled result on the method variant: matched SFT vs. RL post-training on GeneralPoints (arithmetic) and V-IRL (navigation), with held-out rule and visual variants. RL improved OOD variants; SFT of the same data degraded them. Scale: 7B-class, two task families.

**Claimed but unablated.**
- "Reasoning skill is general" as an account of cross-benchmark gains. Usually offered as a table of benchmark deltas with no matched-inference baseline, no pass@$k$ coverage curve, and no contamination measurement. As stated it is a benchmark number, not a transfer result.
- Small-sample distillation results (s1: 1,000 curated traces, budget forcing, +27% on AIME24 over the base; LIMO: 817 examples) are consistent with elicitation, and the papers largely say so.

**Counter-evidence to skill transfer.** Yue et al. (2025) report that RLVR-trained models fall *below* their base models at large $k$ on pass@$k$ across math, code and vision tasks — sharpening, not expansion. Gandhi et al. (2025) find transfer is gated by whether the base model already exhibits verification/backtracking behaviours; priming a model that lacks them changes the outcome.

## 4. What Is Known

- **Format alone moves scores.** Chain-of-thought prompting took PaLM 540B from ~18% to 57% on GSM8K with no weight change (Wei et al., NeurIPS 2022).
- **Benchmark deltas overstate skill.** GSM-Symbolic: re-templating GSM8K names/numbers costs models measurable accuracy, and adding one irrelevant but topical clause (GSM-NoOp) drops accuracy by up to ~65% across models up to frontier scale (Mirzadeh et al., ICLR 2025). GSM1k, a fresh matched-difficulty rebuild, shows drops up to ~13% for some model families and ~0 for others (Zhang et al., 2024) — the spread is itself the finding.
- **Compositional depth does not transfer within a domain, let alone across.** Multi-digit multiplication and Einstein puzzles collapse as problem graph depth exceeds training depth (Dziri et al., NeurIPS 2023).
- **Content effects.** Models, like humans, solve logically identical problems at different rates depending on believable content (Lampinen et al., *PNAS Nexus*, 2024) — direct evidence that the represented object is not a content-free rule.
- **Reward signal is partly decoupled from learning.** On Qwen2.5-Math-7B, RLVR with random or format-only rewards produced MATH-500 gains within a few points of correct-reward RLVR; the same recipe on Llama models did not (Shao et al., 2025). What "transfers" is partly base-model-specific.

## 5. What Is Not Known

- **Methodologically blocked.** No accepted estimator of $T_{\text{ex}}$. There is no standard for the scaffold family $\mathcal{P}$, no matched-token-budget convention, and no target set with certified content-disjointness from the source and from pretraining. Until $\mathcal{P}$ and $B$ are fixed by convention, cross-paper transfer numbers are not comparable.
- **Empirically open.** Whether $T_{\text{ex}} > 0$ at frontier scale on a constructed, uncontaminated, content-disjoint target. Runnable today: needs a base model, an RLVR run, and a held-out synthetic domain. Nobody has published it with all controls.
- **Empirically open.** Whether coverage loss (pass@$k$ decline) is intrinsic to RLVR or an artifact of entropy collapse under low-KL-penalty objectives.
- **Theoretically open.** No characterization of source/target pairs for which policy-gradient post-training strictly reduces target risk. Existing transfer/multitask bounds assume shared representations under a fixed hypothesis class and do not model verifier-driven distribution sharpening.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability at matched budget**. Skill transfer and elicitation predict the same observable — higher target pass@1 after source training — and are separated only by the counterfactual $E_T(B)$, which is a maximum over an unbounded scaffold space. Any measured $E_T$ is a lower bound, so every positive $T_{\text{ex}}$ is defeasible by a better prompt. This is not a compute problem; it is an under-determination of the estimand.

Two aggravating factors: benchmark contamination is unmeasurable without pretraining-corpus access, so $\rho_{S,T}$ covers only the post-training corpus; and final-answer verifiers do not measure reasoning, so a metric named "reasoning transfer" is scored by answer-string equality.

## 7. Current Research (as of 2026)

- **RLVR generalization audits** — pass@$k$ coverage curves, entropy control, KL-regularized objectives to prevent diversity collapse (Tsinghua/Shanghai Jiao Tong groups; Yue et al. line of work).
- **Behavioural priming** — identifying the base-model "cognitive behaviours" (verification, subgoal setting, backward chaining) that gate transfer, and injecting them via small SFT sets (Stanford; Gandhi et al.).
- **Synthetic disjoint domains** — procedurally generated puzzle families with controllable overlap to the source, used as clean transfer targets *(frontier — verify)*.
- **Cross-modal and agentic transfer** — whether math RL improves tool use and long-horizon agents; currently reported as benchmark tables without matched-budget controls *(frontier — verify)*.
- **ARC-AGI-2 and successors** as adjacency-resistant targets (Chollet's program): explicitly designed so that source-domain content cannot be reused.

## 8. Concrete Next Experiment

**Question.** Is $T_{\text{ex}} > 0$ on a target constructed to share no content with the source?

**Scale.** One open base model at 7–8B and one at 32B (two scales, to test scale-dependence). Source: verifiable math, ~10k prompts, GRPO-style RLVR, ~2k H100-hours total. Target: a procedurally generated domain — e.g. constraint-satisfaction over invented relation symbols with randomized surface vocabulary — 1,000 items, generated *after* the source set is frozen, with $\rho_{S,T}$ measured and required $< 0.01$.

**Arms.**
1. RLVR-on-math → evaluate target.
2. **Control arm (the one that matters):** base model + best-of-$n$ self-consistency + reflection scaffold, tuned on a 200-item target dev split, at *token budget matched to arm 1's mean target generation length*.
3. Placebo: RLVR with shuffled rewards on the same math prompts (isolates format/entropy effects from reward content).
4. Ceiling: RLVR directly on held-out target-domain prompts.

**Deciding number.** $T_{\text{ex}} = A_{\text{arm 1}} - A_{\text{arm 2}}$ on the 800-item target test split, pass@1 at $n=32$. Pre-register: $T_{\text{ex}} \ge 5$ points with a bootstrap 95% CI excluding zero supports skill transfer; $|T_{\text{ex}}| < 2$ points supports elicitation. Report pass@256 for arms 1–2 alongside: if arm 1 wins pass@1 but loses pass@256, the effect is sharpening regardless of $T_{\text{ex}}$.

## 9. Key References

- **[Foundational]** François Chollet. *On the Measure of Intelligence.* arXiv preprint, 2019. — arXiv:1911.01547
- **[Foundational]** Jason Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS, 2022. — arXiv:2201.11903
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025. — arXiv:2501.12948
- **[SOTA]** Tianzhe Chu et al. *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training.* ICML, 2025. — arXiv:2501.17161
- **[Evidence]** Yang Yue et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* arXiv preprint, 2025. — arXiv:2504.13837
- **[Evidence]** Iman Mirzadeh et al. *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models.* ICLR, 2025. — arXiv:2410.05229
- **[Evidence]** Nouha Dziri et al. *Faith and Fate: Limits of Transformers on Compositionality.* NeurIPS, 2023. — arXiv:2305.18654
- **[Evidence]** Hugh Zhang et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2405.00332
- **[Evidence]** Kanishk Gandhi et al. *Cognitive Behaviors that Enable Self-Improving Reasoners.* arXiv preprint, 2025. — arXiv:2503.01307
- **[Evidence]** Andrew K. Lampinen et al. *Language models, like humans, show content effects on reasoning tasks.* PNAS Nexus, 2024.
- **[Evidence]** Rulin Shao et al. *Spurious Rewards: Rethinking Reward Signals in RLVR.* arXiv preprint, 2025.
- **[Method]** Charlie Snell et al. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* arXiv preprint, 2024. — arXiv:2408.03314
- **[Method]** Niklas Muennighoff et al. *s1: Simple Test-Time Scaling.* arXiv preprint, 2025. — arXiv:2501.19393
- **[Survey]** Fengli Xu et al. *Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models.* arXiv preprint, 2025. — arXiv:2501.09686

## 10. Worked Example

Take a concrete published shape: a 7B base model with GPQA-Diamond (Rein et al., 2023) pass@1 of 33.8%, and the same model after math-only RLVR at 47.1%. $\Delta_{S \to T} = +13.3$ points. Read as "reasoning skill transferred from math to graduate science."

Now apply the controls.

- **Budget.** The RLVR model emits ~4,100 tokens per GPQA item; the base model with a plain CoT prompt emits ~380. Matching budget means giving the base model $\lfloor 4100/380 \rfloor = 10$ samples with majority vote. Self-consistency at $k=10$ on a base model with $p \approx 0.34$ per item and modest answer-mode concentration typically recovers 6–10 points on a 4-way multiple-choice set. Suppose it lands at 41.6%. Then $T_{\text{ex}} = 47.1 - 41.6 = +5.5$, not $+13.3$: **58% of the headline delta was inference budget, not transfer.**
- **Coverage.** pass@256 on the same items: base 78.4%, RLVR 71.9%. The RLVR model solves fewer distinct problems at all; it has concentrated probability onto the ones it already solved. Under this reading the residual $+5.5$ is sharpening within the base model's support, not new capability.
- **Disjointness.** GPQA physics and chemistry items contain multi-step quantitative derivations. Measuring $\rho_{S,T}$ over the math RL corpus with a 13-gram match plus embedding nearest neighbours flags, say, 9% of target items as having a near-duplicate solution schema in source. Drop those 9% and the residual falls to $+4.1$ with a CI of $\pm 3.6$ on 198 items — no longer distinguishable from zero.

The obstruction is now visible and is not a matter of running more seeds. The 4-way multiple-choice format bounds $E_T$ estimation error; the arithmetic content shared between math and physics makes "disjoint" a matter of degree; and the whole argument turns on a maximum over prompt scaffolds that nobody can compute. A single number, $+13.3$, decomposes into budget, sharpening, adjacency, and a residual smaller than its own confidence interval — and the residual is the only part that bears on the question the page asks.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*