---
id: 19-evaluation/answer-extraction-robust-metrics
title: "Evaluation Metrics Robust to Answer Extraction Failures"
topic: 19-evaluation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Evaluation Metrics Robust to Answer Extraction Failures

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/answer-extraction-robust-metrics` · **Status:** empirically-open

## 1. Problem Statement

Free-form benchmark scoring has two stages: the model emits a response, and a parser pulls a candidate answer out of it and compares it to gold. Reported accuracy is the composition of model capability with parser recall. A response that solves the problem but writes `x = 12` instead of `\boxed{12}` scores zero under the standard MATH harness; a response that guesses randomly but happens to emit `(B)` in a restatement of the question can score one.

- **Measurement variant.** Given a benchmark, a model, and an extractor, estimate what fraction of the observed score gap between two models is attributable to extraction rather than capability. Solved when the estimate has a stated confidence interval that does not require re-labelling every response.
- **Method variant.** Design a scoring rule whose value is (near-)invariant to the choice of extractor within a reasonable family, without inflating scores for lucky spurious matches. Solved when two independently written extractors yield scores within a pre-registered tolerance (say 0.5 points) on the same generations, for all models on a leaderboard.
- **Theory variant.** Under what assumptions is latent capability identifiable from observed scores plus extractor metadata alone? Solved by an identification theorem or an impossibility result.

The three differ sharply in difficulty. The measurement variant is runnable today at modest cost. The theory variant is likely non-identifiable without a labelled subsample.

## 2. Formal Setting

Benchmark $D = \{(x_i, a_i^\ast)\}_{i=1}^N$ with prompts $x_i$ and gold answers $a_i^\ast \in \mathcal{A}$. Model $M$ induces $r_i \sim M(\cdot \mid x_i)$, a token string. An **extractor** is a map $E: \Sigma^\ast \to \mathcal{A} \cup \{\bot\}$, where $\bot$ is "no answer found". An **equivalence checker** is $\equiv\, \subseteq \mathcal{A} \times \mathcal{A}$ (string match, SymPy normalisation, or an LLM judge). Observed accuracy, as actually computed by every harness in use:

$$\hat A(E) = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[E(r_i) \neq \bot \;\wedge\; E(r_i) \equiv a_i^\ast\right].$$

Define the **latent answer** $L(r_i) \in \mathcal{A}\cup\{\bot\}$: the answer a careful human reader would say the response commits to, $\bot$ if it commits to none (truncation, refusal, hedging). Latent accuracy $A^{\mathrm{lat}} = \frac1N \sum_i \mathbb{1}[L(r_i) \equiv a_i^\ast]$. Then

$$\hat A(E) = A^{\mathrm{lat}} - \underbrace{\varepsilon_{\text{miss}}(E)}_{L \text{ correct}, E \text{ misses or mis-parses}} + \underbrace{\varepsilon_{\text{spur}}(E)}_{L \text{ wrong or } \bot, E \text{ outputs gold}}.$$

Measurable without human labels: the **abstention rate** $\beta(E) = \frac1N\sum_i \mathbb{1}[E(r_i)=\bot]$, and, over an extractor family $\mathcal{E}=\{E_1,\dots,E_K\}$, the **bracket**

$$\hat A_\wedge = \frac1N\sum_i \min_k s_{ik}, \qquad \hat A_\vee = \frac1N\sum_i \max_k s_{ik}, \qquad w = \hat A_\vee - \hat A_\wedge,$$

with $s_{ik} = \mathbb{1}[E_k(r_i)\equiv a_i^\ast]$. The bracket width $w$ is the extraction-attributable uncertainty *observable from the generations alone*. $A^{\mathrm{lat}}$ is not in general contained in $[\hat A_\wedge, \hat A_\vee]$: every extractor in $\mathcal{E}$ can share a blind spot.

Assumptions the standard practice rests on, and their status:

1. **$\varepsilon_{\text{miss}} \approx \varepsilon_{\text{spur}} \approx 0$.** Violated — reasoning models truncated at the token cap produce $E(r)=\bot$ on 1–10% of long-form math items.
2. **Extraction failures are model-independent, so rankings survive.** Violated — format compliance is itself trained (RLHF/instruction tuning), so $\varepsilon_{\text{miss}}$ correlates with model family.
3. **$\equiv$ is an equivalence relation.** Violated — LLM judges are not transitive and not symmetric under answer-order swap.
4. **One sample per item suffices.** Violated for $T>0$ decoding; $\hat A$ is itself a random variable whose variance is rarely reported.

## 3. State of the Art

**Established (reproduced, ablated).**
- Format sensitivity is real and large. Sclar et al. (*FormatSpread*, ICLR 2024) show spreads of up to 76 accuracy points across semantically equivalent prompt formats for LLaMA-2-13B on few-shot tasks. This confounds prompt formatting and extraction jointly, but the mechanism is the same surface-form dependence.
- Leaderboard rank order is not stable to superficial scoring choices. Alzahrani et al. (ACL 2024) change MMLU answer symbols and option order and reorder the top of the leaderboard.
- Metric discontinuity manufactures apparent capability jumps. Schaeffer et al. (NeurIPS 2023) show that exact-match-style metrics produce "emergence" where token-level continuous metrics show smooth improvement — an existence proof that scoring-rule choice can dominate the reported phenomenon.
- Log-likelihood scoring of fixed options removes extraction entirely but changes the task. Robinson & Wingate (ICLR 2023) show multiple-choice-prompt scoring and cloze log-likelihood scoring rank models differently.

**Claimed but unablated.**
- That modern multi-regex cascades (OpenAI `simple-evals`, `lm-evaluation-harness` filter chains, HuggingFace `math-verify`) have driven $\varepsilon_{\text{miss}}$ below the noise floor. No paper reports a human-adjudicated $\varepsilon_{\text{miss}}$ for these libraries at scale.
- That LLM-judge extraction is a strict improvement. Reported as benchmark deltas on leaderboards, not as measured recall against human labels.

**Benchmark-number-only.** Most published MMLU-Pro, MATH-500, AIME and GPQA numbers exist as a single score under one extractor with no bracket, no $\beta$, and no seed variance.

## 4. What Is Known

- Prompt/format spread up to **76 points** (LLaMA-2-13B, 1-shot, 53 tasks; Sclar et al. 2024).
- **Ranking instability** from answer-symbol and option-order changes on MMLU across ~10 open models at 7B–70B scale (Alzahrani et al. 2024; Zheng et al., ICLR 2024, on selection bias — models over-select specific option IDs, and debiasing moves accuracy by several points).
- The `lm-evaluation-harness` maintainers document that harness version, few-shot formatting and normalisation choices move published scores enough to explain contradictory literature claims (Biderman et al. 2024) — measured across the harness's own task suite at 7B–70B.
- MMLU-Pro (Wang et al., NeurIPS 2024 D&B) explicitly reports needing multi-stage regex plus random-guess fallback, i.e. its own designers found single-pattern extraction insufficient; CoT vs direct-answer scoring swings the benchmark by tens of points.
- $\beta$ is nonzero and model-dependent for long-CoT models under token caps; budget-forcing work (Muennighoff et al., *s1*, 2025) shows that forcing a terminal answer changes measured accuracy at fixed compute.

No published number exists for $\varepsilon_{\text{miss}}$ or $\varepsilon_{\text{spur}}$ against human labels on a current frontier model.

## 5. What Is Not Known

- **Methodologically blocked.** $A^{\mathrm{lat}}$ has no operational definition for responses that hedge, self-contradict, or state two candidate answers. "The answer the response commits to" is a judgement call with unmeasured inter-annotator agreement. Until that agreement is measured, "extraction failure rate" is not a well-posed quantity.
- **Empirically open.** The decomposition of leaderboard gaps into capability and extraction. Cost: a few thousand GPU-hours plus a few thousand human labels. Nobody has published it.
- **Theoretically open.** Whether $A^{\mathrm{lat}}$ is identifiable from $\{\hat A(E_k)\}_{k=1}^K$ and $\beta(E_k)$ alone under any nontrivial assumption on the extractor family. Conjecture: not identifiable, because correlated blind spots are unconstrained; a formal impossibility result along the lines of the anti-concentration arguments used for unsupervised evaluation would settle it.

## 6. Why It Is Hard

**Absent ground truth, plus non-identifiability.** $\varepsilon_{\text{miss}}$ and $\varepsilon_{\text{spur}}$ enter $\hat A$ with opposite signs, so they partially cancel; a benchmark can have 4% miss and 4% spurious and look perfectly calibrated. Recovering either term requires labelling the latent answer, and that label is exactly what the extractor was built to produce — the measurement instrument and the ground truth are the same artifact.

**Confounded measurement.** Extraction compliance is trained. A model post-trained to emit `\boxed{}` scores higher under a `\boxed{}` extractor at fixed reasoning ability. So "extraction robustness" and "instruction following" are not separable by observation; separating them needs an intervention (re-prompting, forced completion) that itself perturbs the distribution being measured.

**Cost is not the obstruction** — a decisive study is well under \$50k. The obstruction is that nobody's leaderboard rank improves by publishing it.

## 7. Current Research (as of 2026)

- **Verifier libraries.** HuggingFace `math-verify` and OpenAI `simple-evals` converged on cascade extraction plus symbolic equivalence. Engineering, not measurement — recall against human labels is still unreported.
- **Structured decoding as extraction elimination.** Constrained generation / tool-call-shaped answers make $E$ trivial, at the cost of a distribution shift whose size is disputed *(frontier — verify)*.
- **Bracket reporting.** Some evaluation groups now report min/max over extractor variants alongside the point score; not yet standard on any major leaderboard *(frontier — verify)*.
- **Judge reliability.** MT-Bench/Chatbot Arena line of work (Zheng et al., NeurIPS 2023 D&B) quantifies judge–human agreement for preference, not for answer extraction; transferring those agreement numbers to extraction is unvalidated.
- **Item response theory for benchmarks.** Latent-trait models that treat extraction as an item-level guessing/slip parameter are the most promising identification route *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** 6 models spanning a leaderboard band of ~4 points (mix of open-weight 8B–70B and two API models), 3 benchmarks (MATH-500, GSM8K, MMLU-Pro), 500 items each, 4 samples per item at $T=0.7$ → 36,000 generations. Generations are produced **once** and frozen.

**Extractor family.** $K=4$: (a) `\boxed{}`-only regex; (b) `lm-evaluation-harness` default filter chain; (c) `math-verify` cascade; (d) an LLM judge given the response and asked only "what final answer does this response commit to?" (never shown the gold).

**Control arm.** A stratified human-adjudicated subsample: 600 responses, over-sampled 3:1 on items where the $K$ extractors disagree, double-annotated, with Cohen's $\kappa$ reported. This is the only arm that estimates $A^{\mathrm{lat}}$, $\varepsilon_{\text{miss}}$, $\varepsilon_{\text{spur}}$. Second control: forced-answer re-prompt ("state your final answer only") on the same items, to separate extraction failure from non-commitment.

**Deciding number.** The ratio

$$R = \frac{\text{median pairwise } |\varepsilon_{\text{miss}}(M) - \varepsilon_{\text{miss}}(M')|}{\text{median pairwise } |A^{\mathrm{lat}}(M) - A^{\mathrm{lat}}(M')|}$$

over adjacent model pairs. $R \ge 0.5$: extraction is a first-order nuisance and leaderboards must report brackets. $R \le 0.1$: current practice is defensible and the problem downgrades to a documentation issue. Secondary readout: does $\kappa < 0.8$ on the human arm? If so, the problem is methodologically blocked rather than empirically open, and the definition of $L(r)$ must be fixed first.

## 9. Key References

- **[Foundational]** Hendrycks, Burns, Kadavath, Arora, Basart, Tang, Song, Steinhardt. *Measuring Mathematical Problem Solving With the MATH Dataset.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2103.03874
- **[Foundational]** Cobbe et al. *Training Verifiers to Solve Math Word Problems.* 2021. — arXiv:2110.14168
- **[SOTA]** Sclar, Choi, Tsvetkov, Suhr. *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.* ICLR, 2024. — arXiv:2310.11324
- **[SOTA]** Alzahrani et al. *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards.* ACL, 2024. — arXiv:2402.01781
- **[SOTA]** Zheng, Zhou, Meng, Zhou, Huang. *Large Language Models Are Not Robust Multiple Choice Selectors.* ICLR, 2024. — arXiv:2309.03882
- **[SOTA]** Wang et al. *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.01574
- **[Analysis]** Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004
- **[Analysis]** Robinson, Wingate. *Leveraging Large Language Models for Multiple Choice Question Answering.* ICLR, 2023. — arXiv:2210.12353
- **[Survey]** Biderman et al. *Lessons from the Trenches on Reproducible Evaluation of Language Models.* 2024. — arXiv:2405.14782
- **[Survey]** Liang et al. *Holistic Evaluation of Language Models.* TMLR, 2023. — arXiv:2211.09110
- **[Judges]** Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.05685

## 10. Worked Example

Constructed instance (numbers illustrative, arithmetic exact), 500 MATH-500 items, two models one leaderboard slot apart.

| | Model A | Model B |
|---|---|---|
| $\hat A$, `\boxed{}`-only | 61.2% | 58.4% |
| $\hat A$, `math-verify` cascade | 63.0% | 62.6% |
| $\beta$ (no answer found, `\boxed{}`) | 3.4% | 8.2% |
| $\beta$ (cascade) | 1.0% | 1.4% |

Under the `\boxed{}` extractor A beats B by **2.8 points**. Under the cascade the gap is **0.4 points** — inside the seed noise of 4 samples at $T=0.7$ (binomial s.e. $\approx \sqrt{0.6\cdot0.4/500} = 2.2$ points on a single sample set). The entire published ordering rests on 4.8 points of differential abstention: B is a long-CoT model that ends with "so the answer is $3/4$" and never emits `\boxed{}`.

Now the obstruction. Human adjudication of 120 disagreement cases gives, for B: 39 of 41 `\boxed{}`-missed responses were latently correct ($\varepsilon_{\text{miss}} \approx 39/500 = 7.8$ points) — but also 6 cases where the cascade grabbed a gold-matching number from an intermediate step of a wrong derivation ($\varepsilon_{\text{spur}} \approx 1.2$ points). So

$$A^{\mathrm{lat}}(B) \approx 62.6 + \text{(cascade misses)} - 1.2,$$

and the cascade's own miss rate is unknown because the adjudicators only reviewed cases where extractors *disagreed*. Correlated blind spots — every extractor in $\mathcal{E}$ ignores answers stated only in a final natural-language sentence — are invisible to the bracket by construction: $w = \hat A_\vee - \hat A_\wedge$ can be 0.2 points while $\varepsilon_{\text{miss}}$ is 5 points for all $k$. That is the non-identifiability, made concrete: the disagreement-stratified sample, which is the cheap protocol everyone reaches for, cannot see the errors the family shares.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*