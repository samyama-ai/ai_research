---
id: 20-interpretability/internals-based-safety-verification
title: "Scalable Verification of Safety Properties From Internals"
topic: 20-interpretability
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scalable Verification of Safety Properties From Internals

> **Topic:** Interpretability · **ID:** `20-interpretability/internals-based-safety-verification` · **Status:** solved-but-impractical

## 1. Problem Statement

Given a trained network's weights and activations, produce a **guarantee** — not an estimate — that it does not do some specific bad thing on a specified input set. Input: parameters $\theta$, a property $P$, an input region $\mathcal{X}$. Output: a certificate $\pi$ that a checker accepts iff $P$ holds on all of $\mathcal{X}$, plus the checker's runtime and the bound's tightness.

Solving it means: a property of safety interest (not norm-ball robustness) certified on a frontier-scale model, with certification cost sub-quadratic in parameter count and a bound tight enough to be decision-relevant.

Three variants, different difficulty:

- **Theory.** Is exact verification of a nontrivial safety property on a transformer tractable at all? Largely settled negatively — ReLU network reachability is NP-complete (Katz et al., CAV 2017). The open theory question is whether *interpretability structure* (sparsity, low-rank circuits) buys an exponential.
- **Method.** Can mechanistic understanding be converted into machine-checkable proofs? Demonstrated at toy scale (Gross et al. 2024), never at scale. This is the "solved-but-impractical" core.
- **Measurement.** Can the safety property itself be written down formally? For "does not deceive", "does not pursue a hidden goal", nobody has a specification $P$ that is both formal and faithful. This variant is **methodologically blocked**.

## 2. Formal Setting

Let $f_\theta:\mathcal{V}^n\to\Delta(\mathcal{V})$ be an autoregressive model, $\theta\in\mathbb{R}^d$. A safety property is a predicate

$$P(\theta) \;=\; \forall x\in\mathcal{X}\;:\; \phi\big(f_\theta(x),\, h_\theta(x)\big)\le \tau,$$

where $h_\theta(x)\in\mathbb{R}^{L\times n\times d_{\text{model}}}$ are internal activations and $\phi$ scores violation.

**Quantities as measured.**

- **Certificate soundness.** $\pi$ is sound if the checker's acceptance implies $P$. Measured by re-running an independent checker (e.g. exporting to Lean/Coq or a MILP solver) — not by agreement with the prover.
- **Bound gap.** $\gamma = \hat{p}_{\text{cert}} - p_{\text{true}}$, where $\hat p_{\text{cert}}$ is the certified violation upper bound and $p_{\text{true}}$ is the empirical violation rate under the strongest available attack. A certificate with $\hat p_{\text{cert}} = 1$ is sound and useless. $\gamma$ is measurable only as $\hat p_{\text{cert}} - p_{\text{attack}}$, an *upper* bound on the real gap.
- **Proof cost.** Wall-clock checker seconds and certificate length in bytes, as a function of $d$. Report the exponent $\alpha$ in $\text{cost}\propto d^{\alpha}$ fitted across a model-size sweep.
- **Faithfulness of the circuit the proof relies on.** Causal scrubbing (Chan et al., Redwood 2022) or activation-patching loss recovery: fraction of the loss gap restored when the hypothesised mechanism is resampled. Reported as recovered-loss $\in[0,1]$.
- **Input region.** $\mathcal{X}$ must be finite or symbolically representable. In practice $\mathcal{X}$ = suffix-token sets, embedding $\ell_p$-balls, or a template grammar.

**Assumptions, and which are violated.**

1. *Linear representation* — features are directions, superposed (Elhage et al. 2022). **Violated**: circular and multi-dimensional features exist (Engels et al. 2024).
2. *Discrete input region is enumerable/symbolic*. **Violated** for natural language: the semantically meaningful $\mathcal{X}$ ("all jailbreaks") has no formal description.
3. *Circuit sparsity* — the mechanism is a small subgraph. **Partly violated**: attribution methods recover heavy-tailed, not sparse, dependency structure at scale.
4. *Property is a function of the output distribution*. **Violated** for goal-directedness claims, which quantify over deployment trajectories, not single forward passes.

## 3. State of the Art

**Theory / formal-methods SOTA.** Branch-and-bound with bound propagation: $\alpha,\!\beta$-CROWN (Wang et al., NeurIPS 2021), Marabou (Katz et al., CAV 2019). VNN-COMP 2023 (Brix et al.) benchmarks top out at CIFAR-scale convnets and small ResNets — $10^5$–$10^7$ parameters — and almost exclusively on $\ell_\infty$ robustness. **Established**: these tools are sound and complete on their fragment. **Not established**: any extension to attention with softmax at transformer scale that terminates.

**Interpretability-as-proof SOTA.** *Compact Proofs of Model Performance via Mechanistic Interpretability* (Gross, Chughtai, et al. 2024) is the only work that closes the loop: hand-derived mechanistic understanding of 1-layer attention-only transformers trained on max-of-$K$ is compiled into formal accuracy lower bounds, over ~150 model seeds. **Established finding**: shorter proofs require stronger mechanistic assumptions and yield *looser* bounds — an explicit compression/tightness frontier. Scale: $\sim10^4$ parameters.

**Feature-based auditing SOTA.** Sparse autoencoders — Cunningham et al. 2023; *Scaling Monosemanticity* (Templeton et al., Anthropic 2024) on Claude 3 Sonnet; Gemma Scope (Lieberum et al. 2024), 400+ SAEs over Gemma 2. These give *evidence*, not certificates. **Claimed but unablated for verification purposes**: that SAE features constitute a basis in which safety properties become checkable. **Benchmark-number-only**: most SAE quality is reported as reconstruction-loss/sparsity Pareto curves plus autointerp scores, which do not bound anything about behaviour.

**Auditing games.** *Auditing Language Models for Hidden Objectives* (Marks et al., Anthropic 2025): teams found a deliberately implanted hidden objective; interpretability tools helped. **Established**: the property is *detectable* under favourable conditions. **Not established**: any completeness claim — the game measures recall against a known plant, not absence.

## 4. What Is Known

- **Complexity.** Verifying a ReLU network property is NP-complete (Katz et al., CAV 2017). Softmax attention adds transcendental constraints; no complete decision procedure at scale exists.
- **Empirical verification ceiling.** VNN-COMP entries handle $10^5$–$10^7$ parameters on $\ell_\infty$ balls. Frontier LMs are $10^{11}$–$10^{12}$. That is 4–7 orders of magnitude, and the gap is not just constant-factor: branch-and-bound is exponential in unstable-neuron count.
- **Proof–tightness tradeoff at toy scale.** Gross et al. 2024: across ~150 seeds of max-of-$K$ models ($\sim10^4$ params), proof length and bound tightness trade off monotonically; the tightest bounds require compute comparable to brute-force over the input set, which for max-of-4 over 64 tokens is $64^4\approx1.7\times10^7$ — enumerable, unlike anything real.
- **Single-direction mediation.** Refusal in 13 open chat models up to 72B is mediated by one direction; ablating it removes refusal, adding it induces refusal (Arditi et al., NeurIPS 2024). Strong *causal* evidence; not a bound — the direction can be bypassed by attacks it was not tested against.
- **Interpretability illusions are real and measurable.** Subspace activation patching can produce causally "working" interventions on dormant, behaviourally irrelevant subspaces (Makelov, Lange, Nanda, ICLR 2024).
- **SAEs are not automatically the right basis.** SAEs trained on randomly-initialised transformers produce comparably "interpretable" features (Heap et al. 2025); SAE probes did not beat baseline linear probes across most of a sparse-probing suite (Kantamneni et al. 2025).

## 5. What Is Not Known

- **Theoretically open.** Whether any structural property of trained transformers (approximate low-rank circuits, feature sparsity) admits a verification algorithm with cost polynomial in $d$ for a nontrivial $P$. No hardness result rules it out *for the structured subclass*; no algorithm achieves it.
- **Theoretically open.** Whether sound *probabilistic* certificates ("violation rate $\le\epsilon$ under distribution $\mathcal{D}$") derived from internals can beat black-box sampling bounds. Sampling gives $O(1/\epsilon)$ queries with no internals at all; internals must beat that to be worth anything.
- **Empirically open.** Whether the Gross et al. proof pipeline survives 2 orders of magnitude of scale-up — e.g. a 2-layer model on modular arithmetic or IOI-style tasks at $10^7$ params. Runnable today on one node. Nobody has published it.
- **Methodologically blocked.** Formalising the properties that matter. "The model has no hidden goal" has no $\phi$ and no $\mathcal{X}$. Until a specification exists, the verification question is not well-posed; auditing games substitute recall-against-a-plant for it.

## 6. Why It Is Hard

Three distinct obstructions, only the first of which is compute.

1. **Exponential in unstable neurons, not in parameters.** Branch-and-bound splits on neurons whose sign is undetermined over $\mathcal{X}$. In a frontier LM over any linguistically interesting $\mathcal{X}$, essentially every neuron is unstable. The search does not merely get slow; it does not terminate.
2. **Absent ground truth for the property.** For robustness, $P$ is given by the metric. For "not deceptive", the ground truth is what the specification was supposed to define. Auditing games manufacture ground truth by implanting it, which measures detection of *known* plants — an evaluation that does not measure the thing it names (absence of unknown ones).
3. **Non-identifiability of the mechanism.** Multiple mechanistic accounts fit the same activations; patching-based validation admits dormant-subspace illusions (Makelov et al. 2024). A proof built on the wrong decomposition is sound about the decomposition and silent about the model.

## 7. Current Research (as of 2026)

- **Guaranteed Safe AI** (Dalrymple, Bengio, Russell, Tegmark, Tenenbaum et al. 2024): world-model + specification + verifier architecture; positions internals-verification as one leg. Programmatic, not yet empirical.
- **Compact proofs / proof-length-vs-bound frontier**: Gross, Chughtai and collaborators; extension to multi-layer models *(frontier — verify)*.
- **Parameter-space decomposition** as a substitute for activation-space SAEs — attribution-based parameter decomposition, Apollo Research 2025 — motivated explicitly by the non-identifiability problem *(frontier — verify)*.
- **Anthropic / GDM interpretability teams**: auditing games, SAE-based feature steering, Gemma Scope releases. Framing has shifted from "certify" to "audit with evidence".
- **Formal-methods side**: VNN-COMP continues; incremental attention-verification work exists but at $10^6$-parameter scale.
- **Open Problems in Mechanistic Interpretability** (Sharkey et al. 2025) names verification-from-internals as an explicit open goal.

## 8. Concrete Next Experiment

**Question.** Does a mechanistic proof pipeline degrade gracefully with scale, or does the certified bound go vacuous?

- **Scale.** Four model sizes: $10^4$, $10^5$, $10^6$, $10^7$ parameters, 2-layer attention-only transformers, all trained to $\ge99\%$ on the same task: *induction-with-distractor* over a 128-token vocabulary, where $P$ = "never copies a token from the blacklisted subset $B$ ($|B|=8$)". $\mathcal{X}$ = all sequences of length 32 over the vocabulary (symbolic, not enumerated). Budget: ~200 A100-hours total.
- **Method arm.** Derive the copying circuit, compile to interval/linear bounds over the OV and QK matrices, emit a certificate checked by an independent MILP solver.
- **Control arm.** Same $P$, same models, off-the-shelf $\alpha,\!\beta$-CROWN with no mechanistic input, same wall-clock budget per model.
- **Deciding number.** The certified violation upper bound $\hat p_{\text{cert}}$ at $10^7$ parameters. If the mechanistic arm returns $\hat p_{\text{cert}}\le 10^{-3}$ where the control returns $\hat p_{\text{cert}}=1$ (vacuous), interpretability provably buys verification headroom. If both are vacuous at $10^6$, the pipeline is toy-only and the field should say so. Secondary readout: fitted exponent $\alpha$ in checker-seconds $\propto d^\alpha$; $\alpha>2$ means it never reaches $10^{11}$.

## 9. Key References

- **[Foundational]** Katz, Barrett, Dill, Julian, Kochenderfer. *Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks.* CAV, 2017. — arXiv:1702.01135
- **[Foundational]** Elhage, Nanda, Olsson et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021.
- **[SOTA]** Gross, Chughtai, et al. *Compact Proofs of Model Performance via Mechanistic Interpretability.* 2024. — arXiv:2406.11779
- **[SOTA]** Wang, Zhang, Xu, Lin, Jana, Hsieh, Kolter. *Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification.* NeurIPS, 2021.
- **[SOTA]** Templeton, Conerly, Marcus et al. *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Transformer Circuits Thread, Anthropic, 2024.
- **[Method]** Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS, 2024. — arXiv:2406.11717
- **[Caution]** Makelov, Lange, Nanda. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR, 2024. — arXiv:2311.17030
- **[Caution]** Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda. *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing.* 2025. — arXiv:2502.16681
- **[Auditing]** Marks, Treutlein, et al. *Auditing Language Models for Hidden Objectives.* Anthropic, 2025. — arXiv:2503.10965
- **[Programme]** Dalrymple, Skalse, Bengio, Russell, Tegmark, Seshia et al. *Towards Guaranteed Safe AI: A Framework for Ensuring Robust and Reliable AI Systems.* 2024. — arXiv:2405.06624
- **[Survey]** Sharkey, Chughtai, Batson, Lindsey et al. *Open Problems in Mechanistic Interpretability.* 2025. — arXiv:2501.16496
- **[Survey]** Brix, Müller, Bak, Johnson, Liu. *First Three Years of the International Verification of Neural Networks Competition (VNN-COMP).* STTT, 2023.

## 10. Worked Example

Take the refusal direction $r\in\mathbb{R}^{4096}$ found in Llama-2-7B-chat (Arditi et al. 2024). Suppose the safety property is $P$: "for every prompt in the harmful set, the refusal score $s(x)=\langle h_\ell(x), r\rangle$ exceeds $\tau$."

Empirically the direction works: ablating $r$ collapses refusal across the harmful test set; adding it induces refusal on harmless prompts. That is a strong causal result.

Now try to certify it. To bound $\min_{x\in\mathcal{X}} \langle h_\ell(x), r\rangle$ over $\mathcal{X}$ = "all 20-token adversarial suffixes appended to a fixed harmful prompt", the input set has $32000^{20}\approx10^{90}$ elements. Symbolic relaxation instead: propagate interval bounds through 16 layers. Each layer's attention softmax and MLP GELU widens the interval; with $d_{\text{model}}=4096$ and 16 layers, the relaxed bound on $s(x)$ after ~4 layers already spans both signs for essentially every coordinate. The certified statement becomes $\hat p_{\text{cert}}=1$: "the model may refuse or may not." Sound, vacuous.

Branch-and-bound would tighten this by case-splitting unstable neurons. Count them: with $11008$ MLP neurons per layer $\times$ 32 layers, and empirically almost none sign-stable over a 20-token free suffix, the split tree has depth $\sim10^5$. At $2^{10^5}$ leaves this is not a compute-budget problem.

And the illusion risk is live even for the empirical result: the same paper's method family is exactly what Makelov et al. show can latch onto dormant subspaces. So the strongest thing available is: **a causally validated direction with no bound, and a bound procedure that returns "anything may happen."** That is the shape of the whole problem — the toy case is genuinely solved, and every route from there to a real model either goes vacuous or goes exponential.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*