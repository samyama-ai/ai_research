---
id: 20-interpretability/latent-knowledge-elicitation
title: "Latent Knowledge Elicitation Without Labels"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Latent Knowledge Elicitation Without Labels

> **Topic:** Interpretability · **ID:** `20-interpretability/latent-knowledge-elicitation` · **Status:** open

## 1. Problem Statement

A model may internally represent that a proposition is false while its output asserts the proposition is true — because of sycophancy, RLHF pressure, imitation of an unreliable persona, or deliberate deception. **Latent knowledge elicitation** asks: recover the model's internal verdict on a proposition from its activations, using no labels for the propositions in question.

- **Input:** a model $M$, a set of yes/no propositions $\{q_i\}$, and read access to activations. No ground-truth labels for $q_i$, and no assumption that $M$'s output tracks its internal verdict.
- **Output:** a *reporter* $r$ mapping activations to $\{0,1\}$ (or a probability).
- **Predicate for success:** $r$ agrees with what the model actually "believes" on propositions where the model's stated answer is wrong.

Three variants, of very different difficulty:

- **Measurement:** define "the model's belief" so that a reporter can be scored at all. Currently the weakest link.
- **Method:** given a working operationalization, find $r$ without labels. Several algorithms exist; all have known failure modes.
- **Theory:** prove that some unsupervised objective identifies the belief-tracking reporter rather than a behaviorally identical impostor. This is the Eliciting Latent Knowledge (ELK) problem of Christiano, Xu and Cotra (ARC, 2021), and it is open with a known family of counterexamples.

## 2. Formal Setting

Let $M$ have residual-stream activations $h_\ell(x) \in \mathbb{R}^d$ at layer $\ell$ on prompt $x$. For proposition $q$ build a **contrast pair** $(x^+, x^-)$: the same prompt completed with "True" and "False". A linear reporter is $p_\theta(x) = \sigma(w^\top \tilde h_\ell(x) + b)$, where $\tilde h$ is mean-centered and per-dataset standardized (this normalization is load-bearing; without it the pair-identity direction dominates).

**Contrast-Consistent Search** (Burns et al., 2023) minimizes

$$
\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\Big[\underbrace{\big(p_\theta(x_i^+) - (1 - p_\theta(x_i^-))\big)^2}_{\text{consistency}} + \underbrace{\min\big(p_\theta(x_i^+), p_\theta(x_i^-)\big)^2}_{\text{confidence}}\Big].
$$

Quantities as actually measured:

- **Accuracy / AUROC:** computed against human labels on a held-out split — labels the method claims not to need. They re-enter at evaluation.
- **Elicitation gap:** $\Delta = \text{AUROC}(r) - \text{AUROC}(M\text{'s stated answer})$, measured only on a distribution where $M$ is *known* to answer wrongly. Without such a distribution, $\Delta$ is trivially near zero and the experiment is vacuous.
- **Transfer gap:** AUROC of $r$ trained on dataset $A$, evaluated on $B$, minus in-distribution AUROC.
- **Causal effect:** ablate or add $\alpha w$ at layer $\ell$; measure the change in log-odds of the model's stated answer. A reporter that reads a belief should be steerable; correlational probes often are not.
- **Seed dispersion:** standard deviation of test accuracy across random inits at equal training loss. A well-posed objective should have low dispersion.

Assumptions, with those known to be violated marked:

1. Truth is (approximately) linearly decodable at some layer. Partly supported.
2. Contrast pairs differ only in the truth-relevant feature. **Violated** — they also differ in token identity, negation surface form, and any prompt-correlated attribute.
3. The consistency + confidence objective is uniquely minimized by the truth direction. **Violated** — see §4.
4. The model has a single, prompt-independent "belief" about $q$. **Violated in general**; belief is at best persona- and context-relative.

## 3. State of the Art

**Empirical SOTA (established).** Supervised or weakly-supervised probes are strong when the target distribution resembles training. Mass-mean and logistic probes on LLaMA-13B recover simple factual truth with high in-distribution accuracy and produce *causal* effects when added to the residual stream (Marks & Tegmark, COLM 2024). Mallen & Belrose's *quirky models* (2023) give the cleanest positive result: models finetuned so that the "Bob" persona answers systematically wrongly still carry the correct answer linearly, and probes trained only on easy "Alice" examples transfer to hard examples the model answers wrongly, with AUROC in the mid-0.90s at middle layers on several of 12 datasets.

**Unsupervised SOTA (claimed, and substantially ablated *against*).** CCS was reported to beat zero-shot accuracy by about 4 points on average across 6 models and 10 datasets (Burns et al., ICLR 2023). Follow-up work found the gain is not attributable to knowledge discovery: PCA on contrast-pair differences matches CCS, and the objective admits many non-truth minima (Farquhar et al., ICLR 2024; Fry et al., 2023). Representation-engineering and inference-time-intervention results (Zou et al., 2023; Li et al., NeurIPS 2023) improve TruthfulQA scores but use labeled data to pick the direction, so they do not address the label-free variant.

**Benchmark-number-only results.** TruthfulQA deltas from steering are single-benchmark numbers with no demonstration that the steered direction generalizes to unseen deception distributions. SAE-based "truth features" (Templeton et al., 2024) are qualitative feature reports, not elicitation benchmarks; a controlled comparison found SAE probes do not beat plain linear probes on sparse-probing tasks (Kantamneni et al., 2025).

**Theory SOTA.** ARC's ELK report frames the *direct translator* vs *human simulator* dilemma and ran a public prize round (2022) that surfaced ~30 proposals, each met with an explicit counterexample. No proposal is known to be counterexample-free.

## 4. What Is Known

- **The unsupervised objective is not truth-selective.** For essentially any binary feature $b$ that is consistent across a contrast pair, one can exhibit a probe at (near-)zero CCS loss. Farquhar et al. (2024) demonstrate this constructively: inserting an irrelevant binary distractor (e.g. a random "banana/shed" token in the prompt) causes CCS to converge to the distractor rather than truth, at comparable loss, across Llama-2-70B-scale models.
- **Seed dispersion is large.** Levinstein & Herrmann (*Philosophical Studies*, 2024) report CCS solutions that vary widely across initializations at matched training loss, and probes that fail on negated statements — evidence the objective is under-determined, not merely noisy.
- **Truth directions are causal, when found supervisedly.** Adding a mass-mean truth direction at a middle layer of LLaMA-13B flips stated answers on simple factual statements (Marks & Tegmark, 2024).
- **Generalization is skill-specific.** Orgad et al. (ICLR 2025) show error-detection probes transfer poorly across task types, indicating no single "truthfulness" coordinate; probe transfer degrades sharply off the training task family.
- **Elicitation beats behavior in a controlled setting.** On quirky models, probes exceed the model's stated answers by double-digit AUROC on the mislabeled persona — the only clean existence proof that latent knowledge is recoverable when behavior hides it, and it uses labels from the easy split.
- **Auditing is partly feasible with tools, not with a single probe.** Anthropic's blind auditing game (Marks et al., 2025) had 3 of 4 teams find a deliberately implanted hidden objective, using interpretability plus data search — not an unsupervised reporter.

## 5. What Is Not Known

- **Theoretically open.** Whether any objective plus prior identifies the direct reporter. The human-simulator counterexample family has no known refutation; nor is there an impossibility theorem. Both directions are unproven.
- **Methodologically blocked.** "The model's belief" has no consensus operationalization independent of behavior. Every current metric grounds in human labels or in an experimenter-implanted wrongness, so the method is validated on exactly the cases where ground truth exists — the cases where elicitation is not needed.
- **Empirically open.** Whether probe transfer from easy-and-labelable to hard-and-unlabelable holds at frontier scale (>$10^{12}$ parameters, reasoning-trained models) and against a model *optimized to defeat the probe*. The quirky-models paradigm is runnable at that scale; no public run exists.

## 6. Why It Is Hard

The obstruction is **non-identifiability under a behavior-matched equivalence class**, compounded by **absent ground truth**.

Any reporter is selected by a loss computed on data. If the training distribution contains only propositions a human labeler can resolve, then the direct translator ("say what the model represents") and the human simulator ("predict what the labeler would say") are indistinguishable on that data by construction — both achieve zero loss. Unsupervised consistency objectives do not fix this; they enlarge the degenerate set, because consistency and confidence are satisfied by any binary feature that flips with the contrast pair. Adding a simplicity or speed prior picks *a* member of the class, with no argument that it is the intended one.

The measurement problem sits underneath: to check whether $r$ read the belief, one must already know the belief. Experiments escape this only by *manufacturing* the belief (quirky finetunes, implanted objectives), which changes the regime the method must eventually work in.

## 7. Current Research (as of 2026)

- **Weak-to-strong / easy-to-hard generalization** as the practical reframing: train on labelable easy cases, measure transfer to hard ones (EleutherAI, OpenAI superalignment lineage). Positive but distribution-limited.
- **Mechanistic anomaly detection** — flag activations whose *mechanism* differs from the trusted distribution rather than decoding a truth value (ARC; EleutherAI). Attractive because it avoids naming the belief.
- **SAE and crosscoder features as probe substrates**, with results now mixed to negative on sparse-probing benchmarks *(frontier — verify at 2026 scale)*.
- **Auditing agents and honeypot evaluations** at Anthropic, UK AISI and Redwood Research: elicitation evaluated by whether a red-teamed hidden behavior is found, not by probe AUROC.
- **Chain-of-thought faithfulness** as a competing channel: if reasoning traces are legible, elicitation may be partly a monitoring problem rather than a probing problem *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does label-free elicitation survive an adversary that optimizes against the probe?

- **Scale:** one open-weights model at 70B (e.g. Llama-3.1-70B), 12 quirky-style datasets, activations at every 4th layer, 5 seeds. Roughly $10^3$ GPU-hours including finetuning — small enough for one academic cluster.
- **Setup:** finetune two variants. (a) *Quirky*: the "Bob" persona answers wrongly. (b) *Probe-adversarial*: same, plus an auxiliary loss penalizing the AUROC of a held-out linear probe trained on the easy split, so the model is pushed to hide the correct answer linearly.
- **Arms:** CCS (label-free), PCA on contrast differences (label-free control), mass-mean probe trained on the easy Alice split (weakly supervised control), and the model's stated answer (behavioral floor).
- **Deciding number:** the elicitation gap $\Delta$ = AUROC(reporter) − AUROC(stated answer) on hard Bob examples, for variant (b). If label-free CCS holds $\Delta \geq 0.15$ on (b) with seed dispersion below 0.05 AUROC, the objective is doing real work. If $\Delta$ collapses to within 0.03 of the PCA control, or dispersion exceeds 0.10, the reported gains are contrast-pair geometry rather than knowledge, and the label-free variant should be declared blocked pending a new operationalization.

## 9. Key References

- **[Foundational]** Paul Christiano, Mark Xu, Ajeya Cotra. *Eliciting Latent Knowledge: How to tell if your eyes deceive you.* Alignment Research Center technical report, 2021.
- **[Foundational/SOTA]** Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt. *Discovering Latent Knowledge in Language Models Without Supervision.* ICLR, 2023. — arXiv:2212.03827
- **[Critique]** Sebastian Farquhar, Vikrant Varma, Zachary Kenton, Johannes Gasteiger, Vladimir Mikulik, Rohin Shah. *Challenges with unsupervised LLM knowledge discovery.* ICLR, 2024. — arXiv:2312.10029
- **[Critique]** Benjamin Levinstein, Daniel Herrmann. *Still No Lie Detector for Language Models: Probing Empirical and Conceptual Roadblocks.* Philosophical Studies, 2024. — arXiv:2307.00175
- **[SOTA]** Samuel Marks, Max Tegmark. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.* COLM, 2024. — arXiv:2310.06824
- **[SOTA]** Alex Mallen, Nora Belrose. *Eliciting Latent Knowledge from Quirky Language Models.* 2023. — arXiv:2312.01037
- **[SOTA]** Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg. *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.* NeurIPS, 2023. — arXiv:2306.03341
- **[Related]** Andy Zou et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* 2023. — arXiv:2310.01405
- **[Related]** Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, Yonatan Belinkov. *LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations.* ICLR, 2025.
- **[Related]** Samuel Marks et al. *Auditing Language Models for Hidden Objectives.* Anthropic, 2025.
- **[Related]** Subhash Kantamneni et al. *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing.* 2025.

## 10. Worked Example

Take 500 propositions from a factual QA set, prompt a 70B model, and build contrast pairs. Now insert an irrelevant binary token into each prompt: half get the word "banana", half get "shed", assigned independently of truth.

The activation difference $\tilde h(x^+) - \tilde h(x^-)$ has at least two directions that flip consistently with the pair: the truth direction $w_T$, and — because the banana/shed token sits in both halves of the pair — a direction $w_B$ that a probe can combine with the pair-position feature to produce a consistent, confident binary split.

Evaluate the CCS loss for a probe reading each:

| Probe | CCS loss | Accuracy vs truth labels |
|---|---|---|
| $w_T$ (truth) | ~0.01 | ~0.85 |
| $w_B$ (distractor) | ~0.01 | ~0.50 |

Both are at the floor of the objective. The loss cannot tell them apart — this is the finding Farquhar et al. reproduce across models and distractors. Run 5 seeds: some land on $w_T$, some on $w_B$, and the *training loss gives no signal about which*. Test accuracy across seeds ranges from chance to 0.85, with the selection determined by initialization and the distractor's variance in the activation space.

The obstruction is visible here: to report 0.85 rather than 0.50, the experimenter must look at held-out labels and pick the good seed. That step is supervision. Deployed elicitation — on propositions where no one knows the answer — has no such step, so the number that gets published is not the number that would be available in use.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*