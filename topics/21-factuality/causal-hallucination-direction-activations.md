---
id: 21-factuality/causal-hallucination-direction-activations
title: "Causal Test for Hallucination Directions in Activation Space"
topic: 21-factuality
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Causal Test for Hallucination Directions in Activation Space

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/causal-hallucination-direction-activations` · **Status:** empirically-open

## 1. Problem Statement

Probes read out "is this claim true?" from a language model's residual stream at high accuracy. The open question is whether the direction they read is **causally responsible** for the model emitting a false claim, or merely **correlated** with it.

- **Input:** a decoder-only LM $M$ with residual stream activations, a prompt distribution where ground-truth answers are known, and a candidate direction $v \in \mathbb{R}^{d}$ at layer $\ell$ obtained by any method (linear probe, mass-mean difference, sparse autoencoder feature, contrastive activation).
- **Output:** a decision on whether $v$ is a *hallucination-mediating* direction — one whose manipulation changes hallucination rate through the mechanism the probe names, not through a general degradation of generation quality.
- **Solved** = a protocol that, given $(M, v, \ell)$, returns a calibrated verdict which (a) survives a matched-norm random-direction control, (b) transfers across held-out prompt distributions and model families, and (c) rules out the subspace-patching illusion of Makelov et al.

Three variants, different difficulty:

| Variant | Question | Difficulty |
|---|---|---|
| **Measurement** | What does "hallucination rate changed *because of* $v$" mean operationally? | Blocked — no agreed estimand |
| **Method** | Can we find such a $v$ if one exists? | Empirically open |
| **Theory** | Must a linearly-decodable truth feature be linearly steerable? | Theoretically open |

## 2. Formal Setting

Let $M$ have hidden states $h^{(\ell)}(x) \in \mathbb{R}^d$ at layer $\ell$ for prompt $x$. A **hallucination labeler** $H: (x, y) \to \{0,1\}$ maps a prompt and generated continuation to $1$ if $y$ contains a claim unsupported by the reference (measured, in practice, by exact-match against a gold set, an NLI entailment model, or an LLM judge — each with its own error rate).

**Intervention.** For direction $v$, $\|v\|_2 = 1$, define the two standard edits:

$$\text{Add}_\alpha: h^{(\ell)} \mapsto h^{(\ell)} + \alpha v, \qquad \text{Ablate}: h^{(\ell)} \mapsto h^{(\ell)} - (v^\top h^{(\ell)})\,v .$$

Applied at token positions $P$ and layers $L$. **Measured as:** hook the forward pass, apply the edit at every position in $P$ for every decoding step, decode greedily, label with $H$.

**Causal effect estimand.**

$$\tau(v,\alpha) \;=\; \mathbb{E}_{x\sim D}\big[H(x, y_{\text{Add}_\alpha}(x))\big] \;-\; \mathbb{E}_{x\sim D}\big[H(x, y_{\text{base}}(x))\big].$$

**Specificity control.** The claim "$v$ mediates hallucination" requires $\tau$ to exceed what a null direction achieves at the same perturbation magnitude. Draw $v_\perp$ uniformly from the unit sphere (or from the orthogonal complement of the probe subspace) and define the **excess causal effect**

$$\Delta(v,\alpha) \;=\; \tau(v,\alpha) - \mathbb{E}_{v_\perp}[\tau(v_\perp,\alpha)],$$

with fluency held fixed: report $\tau$ only at $\alpha$ where perplexity on held-out text satisfies $\mathrm{PPL}_\alpha / \mathrm{PPL}_0 \le 1.05$. Without this constraint $\tau$ is trivially achievable by breaking the model.

**Faithfulness of the subspace.** Following causal-abstraction practice, $v$ is a faithful mediator if patching $v^\top h$ from a counterfactual run $x'$ into run $x$ moves the output toward $x'$'s answer **and** $v$ carries information the model actually uses on the clean run — the second conjunct is what Makelov et al. show can fail.

**Assumptions, and which break.**
1. *Linearity* — the feature is a direction, not a manifold. Violated for features with magnitude-dependent readout; SAE evidence suggests many features are near-orthogonal but not globally linear.
2. *Single-layer sufficiency* — violated: effects typically require multi-layer application.
3. *Label validity* — $H$ is assumed correct. LLM-judge agreement with human labels on open-ended factuality runs ~80–90%, so $|\tau| < 0.10$ is inside labeler noise.
4. *Stationarity across decoding* — a fixed $\alpha$ applied at step 1 and step 200 is assumed equivalent. Not tested.

## 3. State of the Art

**Established (with ablations).**
- **Inference-Time Intervention** (Li et al., NeurIPS 2023): shifting attention-head activations along a truth direction raises Alpaca-7B TruthfulQA true×informative from 32.5% to 65.1%, with a $K$-head sparsity ablation and a random-direction comparison.
- **Refusal direction** (Arditi et al., NeurIPS 2024): a single difference-in-means direction, ablated across all layers, disables refusal on 13 open chat models up to 72B — the strongest existing demonstration that a *behavioral* direction can be causal, and the methodological template this problem asks to replicate for hallucination.
- **Interpretability illusion in subspace patching** (Makelov et al., ICLR 2024): activation patching along a subspace can produce the target behavior via dormant, off-distribution pathways the model does not use on clean inputs. This is the specific reason positive steering results do not settle the question.

**Claimed but unablated, or benchmark-only.**
- Probe accuracy figures (Azaria & Mitchell, EMNLP Findings 2023; Marks & Tegmark, COLM 2024; Orgad et al., ICLR 2025) are *readout* results. High AUROC is reported; causal follow-through on free-form generation usually is not.
- Contrastive Activation Addition (Rimsky et al., ACL 2024) reports steering across several behaviors; hallucination-specific effect sizes under a matched-norm null are not systematically reported.
- SAE "truthfulness features" are largely qualitative in published form — feature-level causal effects on generation-time hallucination rate, with controls, remain thin *(frontier — verify)*.

## 4. What Is Known

- **Readout is strong and layer-localized.** Linear probes on Llama-2-13B truth/falsehood statement sets reach ~90%+ accuracy, peaking in middle layers (~40–60% depth), and generalize across topic datasets (Marks & Tegmark, COLM 2024, 7B/13B scale).
- **Errors are predictable from internals before decoding.** Orgad et al. (ICLR 2025) show probes on exact-answer token positions predict error type on Mistral-7B and Llama-3-8B substantially above the model's own verbalized confidence — but the signal is task-specific and transfers poorly across datasets.
- **Uncertainty-based detection is a strong non-mechanistic baseline.** Semantic entropy (Farquhar et al., *Nature* 2024) detects confabulations across 6 datasets and models to 70B, requiring ~10 samples per prompt — no activation access needed.
- **Steering works for at least one behavior.** The refusal result (13 models, ≤72B) shows a single direction can be both necessary and sufficient for a behavior.
- **Steering strength trades against fluency.** Across ITI and CAA, effects grow with $\alpha$ and so does perplexity; published curves rarely report the effect at fixed-perplexity operating points.
- **Negative result on generality:** Levinstein & Herrmann (*Philosophical Studies*, 2024) show CCS-style and probe-based "lie detectors" fail to generalize under simple distribution shifts (e.g., negation), at 6B–13B scale.

## 5. What Is Not Known

- **Methodologically blocked:** the estimand. There is no agreed definition of "hallucination caused by direction $v$" separating (i) suppressing a false claim, (ii) suppressing claim-making entirely, and (iii) degrading generation. Abstention inflates $-\tau$ without improving factuality.
- **Empirically open:** whether *any* published truth direction yields $\Delta(v,\alpha) > 0.10$ on open-ended long-form generation at fixed perplexity, at ≥70B scale. The experiment is runnable on 8×H100 in days. Nobody has published it with the full control arm.
- **Empirically open:** whether a hallucination direction found on model $A$ transfers to model $B$ after activation-space alignment, as the refusal direction does within families.
- **Theoretically open:** whether linear decodability implies linear steerability. No theorem either way; the question is when a linear readout coincides with a causally used component under a nonlinear readout head.
- **Theoretically open:** identifiability. Many directions can encode the same feature after a change of basis; which member of the equivalence class is causal is undetermined by probe training alone.

## 6. Why It Is Hard

**Non-identifiability plus confounded measurement, compounding.**

1. **The probe direction is not unique.** Probes trained on the same data with different regularizers recover directions with cosine similarity well below 1 while achieving near-identical accuracy. Probe fit does not select a causal representative.
2. **Truth is entangled with format, topic and stance.** A direction separating true from false statements also separates the *distribution* they were drawn from. Steering along it can change hedging or verbosity, which changes $H$ without touching factuality.
3. **The intervention is off-distribution by construction.** $h + \alpha v$ lands where the model was never trained. Downstream layers may respond through dormant pathways (Makelov et al.) — the effect is real, the mechanism is not the claimed one.
4. **Ground truth for long-form output is expensive and noisy.** Per-claim decomposition and verification costs on the order of $10^0$–$10^1$ USD per response with LLM judges, and judge–human agreement caps the resolvable effect size near 0.10.
5. **Fluency is the free parameter.** Any $\tau$ can be manufactured by raising $\alpha$. Without a fixed-perplexity constraint the reported number is uninterpretable.

## 7. Current Research (as of 2026)

- **Representation engineering / steering:** groups around the CAIS RepE line and Anthropic's SAE-steering work continue on directional control; the open sub-thread is fixed-fluency effect reporting *(frontier — verify)*.
- **Causal abstraction and DAS** (Geiger, Wu, Potts, Stanford): distributed alignment search searches for causally aligned subspaces rather than fitting probes — the natural formal frame for this problem, so far applied mainly to algorithmic tasks (IOI, arithmetic) rather than open-ended factuality.
- **Ignorance vs. error decomposition:** work separating hallucinations where the model lacks the fact from those where it has it but fails to emit it (Simhi et al., 2024–2025) — relevant because only the second class is plausibly steerable.
- **SAE feature-level causal evaluation:** moving from "this feature fires on falsehoods" to feature-clamping effects on generation-time factuality *(frontier — verify)*.
- **Detection baselines strengthening:** semantic-entropy variants and self-consistency remain the comparison any mechanistic method must beat on cost-adjusted terms.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-70B-Instruct and Qwen2.5-72B-Instruct. Prompt set: 2,000 long-form biography/entity prompts with gold reference facts (FActScore-style decomposition), split 1,000 fit / 1,000 held-out, plus 500 held-out prompts from a different domain (scientific claims) for transfer.

**Directions under test (4 arms).**
1. Mass-mean truth direction from a statement corpus, layer chosen by probe peak.
2. ITI-style per-head direction, top-$K$ heads.
3. Top SAE feature by activation difference on hallucinated vs. grounded spans.
4. **Control:** matched-norm random directions, $n = 20$ draws, sampled in the orthogonal complement of the top-10 probe subspace.

**Protocol.** For each arm, sweep $\alpha$; retain only operating points with $\mathrm{PPL}_\alpha/\mathrm{PPL}_0 \le 1.05$ on 100k held-out tokens. Score with FActScore-style per-claim verification. Report abstention rate and mean claim count per response separately — a drop in claims is not a factuality gain.

**The deciding number.**

$$\Delta^\star = \max_{\alpha:\ \mathrm{PPL}_\alpha/\mathrm{PPL}_0 \le 1.05} \Big( \tau(v,\alpha) - \mathbb{E}_{v_\perp}[\tau(v_\perp,\alpha)] \Big) \quad \text{on held-out prompts, at fixed claim count } \pm 10\%.$$

$\Delta^\star \le -0.10$ (a ≥10-point absolute reduction in hallucinated-claim rate over the random-direction null) with transfer to the out-of-domain split: the direction is causal. $|\Delta^\star| < 0.05$: the probe is a readout artifact, and the field should stop reporting probe AUROC as evidence of mechanism. Cost estimate: ~2,000 GPU-hours plus ~$3–5k of verification.

## 9. Key References

- **[Foundational]** Alain, G., Bengio, Y. *Understanding intermediate layers using linear classifier probes.* ICLR Workshop, 2017. — arXiv:1610.01644
- **[Foundational]** Burns, C., Ye, H., Klein, D., Steinhardt, J. *Discovering Latent Knowledge in Language Models Without Supervision.* ICLR, 2023. — arXiv:2212.03827
- **[Foundational]** Azaria, A., Mitchell, T. *The Internal State of an LLM Knows When It's Lying.* Findings of EMNLP, 2023. — arXiv:2304.13734
- **[SOTA]** Li, K., Patel, O., Viégas, F., Pfister, H., Wattenberg, M. *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.* NeurIPS, 2023. — arXiv:2306.03341
- **[SOTA]** Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS, 2024. — arXiv:2406.11717
- **[SOTA]** Marks, S., Tegmark, M. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.* COLM, 2024. — arXiv:2310.06824
- **[SOTA]** Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H., Belinkov, Y. *LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations.* ICLR, 2025. — arXiv:2410.02707
- **[Critical]** Makelov, A., Lange, G., Nanda, N. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR, 2024. — arXiv:2311.17030
- **[Critical]** Levinstein, B. A., Herrmann, D. A. *Still No Lie Detector for Language Models: Probing Empirical and Conceptual Roadblocks.* Philosophical Studies, 2024. — arXiv:2307.00175
- **[Method]** Geiger, A., Wu, Z., Potts, C., Icard, T., Goodman, N. *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations.* CLeaR, 2024. — arXiv:2303.02536
- **[Method]** Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., Turner, A. *Steering Llama 2 via Contrastive Activation Addition.* ACL, 2024. — arXiv:2312.06681
- **[Baseline]** Farquhar, S., Kossen, J., Kuhn, L., Gal, Y. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature 630, 2024.
- **[Eval]** Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., Hajishirzi, H. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Survey]** Zou, A., Phan, L., Chen, S., et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* 2023. — arXiv:2310.01405

## 10. Worked Example

Take Llama-3.1-8B-Instruct, layer 14 of 32, and fit a mass-mean direction $v$ on 1,000 true/false factual statements. Probe accuracy on held-out statements: ~91% (consistent with the 7B/13B figures in Marks & Tegmark). This is the number usually reported as evidence.

Now run the causal test on 200 biography prompts, verified per claim.

| Arm | $\alpha$ | PPL ratio | Hallucinated-claim rate | Claims/response |
|---|---|---|---|---|
| Base | 0 | 1.00 | 0.41 | 12.3 |
| $v$ (truth dir.) | 4 | 1.03 | 0.36 | 11.8 |
| $v$ | 12 | 1.31 | 0.22 | 5.1 |
| Random $v_\perp$ (mean of 20) | 12 | 1.29 | 0.25 | 5.6 |

Read the rows in order. At the fluency-preserving point ($\alpha=4$), the effect is $\tau = -0.05$ — inside per-claim verifier noise, and once the random-direction null at the same norm is subtracted, $\Delta \approx -0.02$. At $\alpha = 12$ the headline improvement looks large, $-0.19$, but perplexity rose 31% and the model emitted 5.1 claims instead of 12.3: it stopped asserting things. The random direction at matched norm bought $-0.16$ of that, so the excess attributable to $v$ is $-0.03$.

The obstruction is visible in the last two rows. Most of the apparent causal effect is **suppression of claim-making**, reproducible with a direction that encodes nothing about truth. A 91% probe and a 19-point steering gain coexist with an excess effect near zero. Until $\Delta^\star$ is reported at fixed perplexity and fixed claim count against a matched-norm null, "we found the hallucination direction" is a statement about readout, not mechanism.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*