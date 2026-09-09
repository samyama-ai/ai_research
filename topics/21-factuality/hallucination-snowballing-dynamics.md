---
id: 21-factuality/hallucination-snowballing-dynamics
title: "Hallucination Snowballing in Autoregressive Decoding"
topic: 21-factuality
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hallucination Snowballing in Autoregressive Decoding

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/hallucination-snowballing-dynamics` · **Status:** open

## 1. Problem Statement

A language model emits a false claim early in a generation. The claim then enters its own context. The question is whether that self-generated falsehood *causally raises* the probability of later falsehoods — and if so, by how much, and through what mechanism.

Three variants, with different difficulty:

- **Measurement.** Given a decoded sequence with per-claim veracity labels, estimate the causal effect of an early error on the hazard of later errors, holding topic, length, and difficulty fixed. Solved-in-principle by prefix intervention; unsolved in practice because "same prompt, one claim flipped" changes more than one thing.
- **Method.** Build a decoder whose error hazard does not increase after a self-generated error — detect-and-revise, backtrack, or resample — at bounded token cost and without degrading tasks where committing early is correct.
- **Theory.** Characterize when the compounding is (a) exposure bias (train/test distribution mismatch on prefixes), (b) commitment/consistency pressure (the model's LM objective rewards coherence with its own prior tokens even when false), or (c) neither — merely a shared latent cause, i.e. hard questions producing errors everywhere in the sequence. These predict different fixes and are currently confounded.

A solution to the measurement variant is an estimator with a stated identification argument and a confidence interval on the hazard ratio. A solution to the method variant is a decoder with a measured reduction in that ratio, at matched compute.

## 2. Formal Setting

Let $p_\theta$ be an autoregressive model over tokens, decoded to $y_{1:T}$ under prompt $x$. Segment $y$ into atomic claims $c_1,\dots,c_n$ (FActScore-style decomposition; Min et al., EMNLP 2023). Each claim has a veracity label $E_i \in \{0,1\}$, $E_i=1$ meaning unsupported by the reference source $K$. Measured by: an entailment or retrieval judge $J(c_i, K)$, whose own error rate $\epsilon_J$ must be reported — typically $\epsilon_J \approx 0.05$–$0.15$ on biography-style claims, and higher on reasoning chains.

**Snowball hazard ratio.** Define
$$\lambda_i \;=\; \frac{\Pr[E_i = 1 \mid \exists j<i:\, E_j=1,\; Z]}{\Pr[E_i = 1 \mid \forall j<i:\, E_j=0,\; Z]}$$
with $Z$ the covariates held fixed (question, claim index $i$, claim type, prefix length in tokens). $\lambda_i > 1$ is the observational association. It is *not* the causal quantity.

**Causal quantity (prefix surgery).** For a fixed continuation point $t$, construct two prefixes that differ only in the veracity of one claim: $\tilde y_{<t}^{\,\mathrm{err}}$ containing a false claim $c$, and $\tilde y_{<t}^{\,\mathrm{ok}}$ containing its corrected counterpart $c'$, matched on token length and surface form. Then
$$\Delta \;=\; \mathbb{E}\big[E_{>t} \mid \mathrm{do}(\tilde y^{\,\mathrm{err}}_{<t})\big] \;-\; \mathbb{E}\big[E_{>t} \mid \mathrm{do}(\tilde y^{\,\mathrm{ok}}_{<t})\big],$$
where $E_{>t}$ is the unsupported-claim rate in the continuation. $\Delta>0$ is snowballing proper.

**Self-recognition gap.** Let $q_\theta(c)$ be the model's probability of judging $c$ false when $c$ is presented *out of context* as a standalone verification query. The commitment effect is
$$G(c) \;=\; q_\theta(c) \;-\; \Pr\nolimits_\theta[\text{model retracts } c \mid \tilde y_{<t}^{\,\mathrm{err}}].$$
$G>0$ means the model knows better than it acts — the load-bearing observation in the snowballing literature.

**Assumptions, and which fail.**
1. *Claims are atomic and independently checkable.* Violated: entailment between claims means flipping one changes the truth value of others.
2. *Prefix surgery is minimal.* Violated: a corrected claim shifts the model's estimate of the question's difficulty and of its own competence, so $\mathrm{do}(\cdot)$ perturbs more than veracity. This is the central identification problem.
3. *The judge is independent of the generator.* Violated whenever the judge is the same family (self-preference bias, measured at 5–10 points on pairwise judging).
4. *Stationary hazard in $i$.* Violated: hazards rise with position for length reasons alone.

## 3. State of the Art

**Established.** Zhang, Press, Merrill, Liu, Smith, *How Language Model Hallucinations Can Snowball* (ICML 2024, arXiv:2305.13534) is the reference result. Three adversarially constructed datasets (primality testing, US-senator search, graph connectivity) elicit a wrong yes/no commitment, after which the model produces an incorrect supporting justification. The load-bearing finding is the self-recognition gap: the models identify a large majority of their own snowballed claims as false when the claims are re-presented in isolation — reported around 87% for GPT-4 and 57% for ChatGPT. That gap is established and has been reproduced qualitatively on open models.

**Claimed but unablated.** That snowballing is *exposure bias* — i.e. that the training/inference prefix mismatch is the mechanism. Arora, El Asri, Bahdanau (*Why Exposure Bias Matters*, Findings of ACL 2022) give an imitation-learning account of error accumulation, but the link to factual snowballing in large pretrained models is asserted, not measured.

**Benchmark-number-only.** Inference-time mitigations report end-task deltas, not hazard ratios: DoLa (Chuang et al., ICLR 2024) contrasting layer logits; Lookback Lens (Chuang et al., EMNLP 2024) using attention mass on context as a hallucination signal; SelfCheckGPT (Manakul, Liusie, Gales, EMNLP 2023) sampling-consistency detection; LM-vs-LM cross-examination (Cohen et al., EMNLP 2023). Each moves FActScore/TruthfulQA-style aggregates. None reports $\Delta$ or $\lambda$.

**Theory SOTA.** Kalai & Vempala, *Calibrated Language Models Must Hallucinate* (STOC 2024) lower-bounds the hallucination rate of a calibrated model by roughly the fraction of facts appearing once in training ("monofacts"). Kalai, Nachum, Vempala, Zhang (2025) reduce hallucination to binary misclassification and argue evaluation scoring sustains it. Neither theory has a temporal component — they bound the *rate*, not the *dependence structure across a sequence*.

## 4. What Is Known

- **Self-recognition gap is real and large.** ~87% (GPT-4) and ~57% (ChatGPT) of self-generated snowballed claims flagged false on isolated re-presentation (Zhang et al., ICML 2024). Scale: proprietary frontier models, ~500 questions per dataset.
- **Self-correction without external feedback does not reliably fix it.** Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* (ICLR 2024): intrinsic self-correction *degrades* GSM8K/CommonSenseQA accuracy when the oracle label is withheld — the widely-cited gains came from label leakage. Scale: GPT-3.5/GPT-4, thousand-item benchmarks.
- **Error compounds superlinearly with composition depth.** Dziri et al., *Faith and Fate* (NeurIPS 2023): on multi-digit multiplication and dynamic-programming tasks, exact-match accuracy collapses as the computation graph deepens even when each single step is learned — the clean mechanistic analogue of snowballing under ground truth.
- **Stated reasoning need not be the cause of the answer.** Turpin et al., *Language Models Don't Always Say What They Think* (NeurIPS 2023): biasing features flip answers up to ~36 points while the verbalized chain rationalizes post hoc. So "the error propagated through the reasoning" cannot be read off the text.
- **Greedy decoding is not uniformly safer.** Sampling raises per-claim error but lowers commitment lock-in; no published crossover point.

## 5. What Is Not Known

- **Theoretically open.** No result bounds $\Delta$ in terms of model properties. Specifically: does calibration on the *training* distribution imply a positive snowball effect on self-generated prefixes? Kalai–Vempala bound marginal rates; the conditional-on-own-error case is unproven either way.
- **Empirically open.** $\Delta$ has never been estimated with a matched prefix-surgery control on open-weight models across a scale ladder (1B → 70B → 400B). Runnable today; nobody has run it.
- **Empirically open.** Whether RLHF/RLVR post-training increases or decreases $\lambda$ relative to the base model. Plausible both ways: consistency-rewarding preference data should raise it; verifier-reward training should lower it.
- **Methodologically blocked.** "Minimal prefix edit" is undefined. Any correction of a false claim also changes fluency, token count, and the implied difficulty of the question. Without a defensible minimality criterion, $\Delta$ is not identified — this is why the field reports $\lambda$ (association) and calls it snowballing.

## 6. Why It Is Hard

**Confounded measurement plus non-identifiability.** The observational $\lambda>1$ has a fully innocent explanation: hard questions produce errors at every position, so early and late errors share a latent cause. Removing that confound needs an intervention, and the intervention is not well defined — flipping a claim's truth value inside a prefix necessarily alters other things the model conditions on. Compare the clean case: in Dziri et al.'s arithmetic tasks, ground truth per step exists and the causal graph is known, and there error propagation is unambiguous. In open-domain generation neither holds. Secondary obstruction: the judge $J$ has $\epsilon_J \approx 0.1$, and since $\Delta$ is a difference of two rates that may themselves be 0.1–0.3, judge noise is the same order as the effect.

## 7. Current Research (as of 2026)

- **Attention-based propagation signals.** Extending Lookback Lens–style features to predict *future* hallucination from the presence of an earlier self-generated error rather than to classify the current span. *(frontier — verify)*
- **Backtracking decoders.** Training models to emit an explicit retraction/rollback token so error recovery is in-distribution rather than an out-of-distribution prefix. Reported in safety-backtracking work; factuality transfer unmeasured. *(frontier — verify)*
- **Process supervision.** Step-level reward models (Lightman et al., *Let's Verify Step by Step*, ICLR 2024) reduce compounding in math; whether the gain is recovery or better first steps is not decomposed.
- **Calibration-theoretic extensions.** Groups around Kalai/Vempala and the OpenAI evaluation-incentives line pushing toward sequential/conditional bounds.
- **Long-horizon agent traces.** Snowballing over tool-call transcripts, where an early wrong retrieval poisons dozens of steps; largely anecdotal, no hazard-ratio measurement published.

## 8. Concrete Next Experiment

**Twin-prefix surgery on an open-weight ladder.**

- **Scale.** Llama-3.1 8B / 70B and Qwen-2.5 7B / 72B, base *and* instruct checkpoints (8 arms). 2,000 biography-style prompts from the FActScore Wikipedia pool, ~10 atomic claims each. Roughly 200 GPU-hours on 8×H100.
- **Procedure.** Generate greedily. Find the first claim $c$ judged false at position $t$. Build $\tilde y^{\,\mathrm{err}}_{<t}$ (unchanged) and $\tilde y^{\,\mathrm{ok}}_{<t}$ (that claim replaced by a supported claim of the same type, matched to within ±2 tokens, verified by a second judge). Continue decoding from both with identical seeds. Score continuations with retrieval-grounded NLI; hold out 300 claims for human labels to estimate $\epsilon_J$.
- **Control arm.** The critical one, and the reason this design differs from prior work: a **placebo edit** — replace $c$ with a *different false claim* of the same type and length. If $\Delta$ against the placebo is ~0 while $\Delta$ against the corrected prefix is large, the effect is truth-conditional. If both are large, the effect is a text-edit artifact, not snowballing.
- **Deciding number.** $\Delta$, the difference in downstream unsupported-claim rate between the erroneous and corrected prefixes, with a bootstrap 95% CI. **$\Delta \ge 0.05$ with the CI excluding zero, and a placebo $\Delta$ within $\pm 0.02$ of zero, establishes causal snowballing.** $\Delta$'s CI containing zero at 70B scale would mean the published effect is confounding plus adversarial prompt construction.

## 9. Key References

- **[Foundational]** Muru Zhang, Ofir Press, William Merrill, Alisa Liu, Noah A. Smith. *How Language Model Hallucinations Can Snowball.* ICML, 2024. — arXiv:2305.13534
- **[Foundational]** Kushal Arora, Layla El Asri, Dzmitry Bahdanau, et al. *Why Exposure Bias Matters: An Imitation Learning Perspective of Error Accumulation in Language Generation.* Findings of ACL, 2022.
- **[Theory]** Adam Tauman Kalai, Santosh S. Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Theory]** Nouha Dziri, Ximing Lu, Melanie Sclar, et al. *Faith and Fate: Limits of Transformers on Compositionality.* NeurIPS, 2023.
- **[SOTA]** Jie Huang, Xinyun Chen, Swaroop Mishra, et al. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR, 2024.
- **[SOTA]** Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, Pengcheng He. *DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models.* ICLR, 2024.
- **[Measurement]** Sewon Min, Kalpesh Krishna, Xinxi Lyu, et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023.
- **[Measurement]** Potsawee Manakul, Adian Liusie, Mark Gales. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP, 2023.
- **[Related]** Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS, 2023.
- **[Survey]** Lei Huang, Weijiang Yu, Weitao Ma, et al. *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* ACM TOIS, 2025.

## 10. Worked Example

Prompt: *"Is 10,007 prime? Answer Yes or No, then justify."*

10,007 is prime. Suppose the model commits "No." The justification must now produce a factor. A typical continuation: *"10,007 = 7 × 1,429."* Check: $7 \times 1{,}429 = 10{,}003 \ne 10{,}007$. Second claim, caused by the first.

Now the diagnostic. Ask separately, fresh context: *"Does 7 divide 10,007?"* Frontier models answer correctly at high rate — Zhang et al. report ~87% self-recognition for GPT-4 on exactly this construction. So $q_\theta \approx 0.87$, retraction-in-context $\approx 0.05$, giving $G \approx 0.82$. The model has the fact and does not use it.

**Where the obstruction appears.** Try to measure $\Delta$ here. Build the corrected prefix: *"Yes. 10,007 is prime."* Continue. Downstream errors drop to near zero — apparently $\Delta \approx 0.9$, huge. But the edit did two things: it removed the false claim, and it removed the *obligation to produce a factor*. The erroneous prefix opens a subgoal with no satisfying answer; the corrected one opens no subgoal. The measured $\Delta$ mixes error propagation with task change.

The placebo arm exposes this. Substitute a different false commitment of matched form — *"No. 10,007 = 11 × 910."* ($11 \times 910 = 10{,}010$.) Downstream error stays high, so placebo $\Delta \approx 0$ against the original: the effect survives a truth-preserving-but-different edit. Fine. But now substitute a *true* prefix that still opens a factoring subgoal — *"No. 10,001 = 73 × 137."* (True: $73 \times 137 = 10{,}001$.) Continuations here are also largely correct.

Two of the three contrasts move together, and the design cannot say whether the driver is the false token in context or the unsatisfiable subgoal it created. That is the identification failure in §5, in eight lines of arithmetic — and it is why the placebo arm in §8 is the load-bearing part of the experiment, not a formality.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*