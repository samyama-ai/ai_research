---
id: 19-evaluation/cross-harness-benchmark-reproducibility
title: "Reproducibility of Reported Benchmark Numbers Across Harnesses"
topic: 19-evaluation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Reproducibility of Reported Benchmark Numbers Across Harnesses

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/cross-harness-benchmark-reproducibility` · **Status:** empirically-open

## 1. Problem Statement

A model card says "MMLU 68.2". Another lab runs the same weights on the same dataset and gets 61.4. Neither party is lying. The gap comes from the **harness** — the code that turns a dataset row into a prompt, samples from the model, and maps the output to a score.

- **Input:** model weights $\theta$, a benchmark dataset $D$, a claimed score $\hat{s}$, and the harness $H$ that produced it (often unspecified).
- **Output:** a reproduction $s' = \text{Eval}(\theta, D, H')$ under an independent harness $H'$, plus an accounting of $|\hat{s} - s'|$ attributed to specific harness degrees of freedom.
- **Decision predicate:** is $|\hat{s} - s'|$ small relative to the score differences the number is used to justify (model selection, safety thresholds, release gating)?

Three variants, different difficulty:

- **Measurement:** define a score that is invariant to harness choices, or report the score's distribution over them. Partly blocked — the invariance class is not agreed on.
- **Method:** build harnesses that are bit-reproducible and version-pinned given a full specification. Largely an engineering problem; mostly solved in principle, unevenly practiced.
- **Theory:** characterize when a ranking over models is stable under a family of prompt/scoring perturbations. Open, and closely tied to the fact that the perturbation family has no canonical measure.

Solving it means: given $(\theta, D)$ and a published specification, two independent groups produce scores agreeing within a stated tolerance, and rankings that do not invert.

## 2. Formal Setting

A harness is a tuple
$$H = (\pi, \sigma, \rho, \kappa, \nu, \beta)$$
where $\pi: x \mapsto p$ is the prompt renderer (template, separators, option labels, instruction, whitespace), $\sigma$ the few-shot selector (number $k$, pool, order, seed), $\rho$ the decoding rule (greedy / temperature $T$ / log-likelihood ranking over answer strings), $\kappa$ the answer extractor (regex, first-token logit, constrained decoding), $\nu$ the normalization (byte-length or token-length normalized log-likelihood, unconditional-probability calibration), and $\beta$ the execution stack (tokenizer version, dtype, kernel, batch size, tensor-parallel degree, KV-cache implementation).

The score on dataset $D=\{(x_i,y_i)\}_{i=1}^n$ is
$$s(\theta,D,H) \;=\; \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\!\left[\kappa\big(\rho(\theta,\pi(x_i);\beta)\big) = y_i\right].$$

Measured quantities:

- **Cross-harness gap:** $\Delta(\theta,D) = \max_{H\in\mathcal{H}} s - \min_{H\in\mathcal{H}} s$ over a declared harness family $\mathcal{H}$. Measured by actually running $|\mathcal{H}|$ configurations, not estimated.
- **Sampling error:** for accuracy, the binomial standard error $\mathrm{SE} = \sqrt{s(1-s)/n}$; with clustered items, the cluster-robust estimate. At $n=14{,}042$ (MMLU) and $s=0.65$, $\mathrm{SE}\approx 0.40$ pp, so a 5 pp gap is ~12 SE — harness variance dominates sampling variance by an order of magnitude.
- **Rank instability:** for a model set $M$, $\tau(H,H')$ = Kendall's $\tau$ between rankings induced by two harnesses; and the **inversion rate** $\Pr[\,\mathrm{sign}(s_a - s_b)\ \text{flips}\,]$ over model pairs.
- **Attribution:** ANOVA-style variance decomposition $\mathrm{Var}(s) = \sum_c \mathrm{Var}_c + \text{interactions}$ over components $c \in \{\pi,\sigma,\rho,\kappa,\nu,\beta\}$, estimated from a factorial sweep.

Assumptions, with the ones known to be violated marked:

1. $D$ is disjoint from pretraining data. **Violated** — contamination is documented and unmeasurable without training-corpus access.
2. $\rho$ with $T=0$ is deterministic. **Violated** — floating-point non-associativity makes batched GPU inference batch-size-dependent, so greedy decoding is not reproducible across serving configurations.
3. Component effects are additive. **Violated** — prompt format interacts strongly with few-shot count and with model size.
4. $\mathcal{H}$ has a natural measure so "average over harnesses" is well defined. **Not established** — this is the methodological block in §5.

## 3. State of the Art

**Established (reproduced, ablated).**

- Format sensitivity is real and large. Sclar et al., *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design* (ICLR 2024, arXiv:2310.11324) sweep semantically equivalent formats — separator, casing, spacing — and report accuracy spreads up to ~76 points on a task for LLaMA-2-13B, with no format transferring reliably across models.
- Ranking instability under benign perturbation. Alzahrani et al., *When Benchmarks are Targets* (ACL 2024, arXiv:2402.01781) show MMLU leaderboard order changes under option-ID relabeling and answer-order permutation.
- Multi-prompt reporting changes conclusions. Mizrahi et al., *State of What Art?* (TACL 2024, arXiv:2401.00595) evaluate over many instruction paraphrases per task and find single-prompt rankings are not robust.
- Harness standardization works where it was attempted. `sacreBLEU` (Post, WMT 2018) removed tokenization-induced BLEU differences of several points by fixing $\kappa,\nu$ in a versioned string.

**Claimed but unablated / benchmark-number-only.**

- Most model-card scores. Published MMLU/GSM8K/HumanEval numbers typically report $s$ without $H$, without $n$-adjusted error bars, and without a seed. The number exists; the ablation does not.
- "Our harness matches the original implementation." Common claim, rarely accompanied by a per-component diff.
- Vendor-run internal harnesses (OpenAI `simple-evals`, Anthropic and Google internal eval stacks). Code is partly open; the exact configuration behind a given headline number usually is not.

**Systems SOTA:** `lm-evaluation-harness` (EleutherAI; Gao et al., 2021–; Biderman et al., arXiv:2405.14782), HELM (Liang et al., TMLR 2023, arXiv:2211.09110), UK AISI `Inspect` (2024), OpenAI `simple-evals` (2024). All are version-pinned and scriptable. None is a shared standard, and they disagree.

## 4. What Is Known

- **The canonical demonstration.** Hugging Face's June 2023 analysis of MMLU discrepancies on the Open LLM Leaderboard found three implementations of the *same* benchmark on the *same* LLaMA-65B weights giving roughly 63% (original Hendrycks code), ~64% (HELM), and ~49% (lm-evaluation-harness at the time). The ~15 pp gap came from scoring mechanics — whether the model is scored on the letter token or the full answer string, and whether log-likelihoods are length-normalized — not from the model. Scale: one 65B model, 14,042 MMLU items.
- **Sampling noise is not the explanation.** At MMLU's $n=14{,}042$, 95% CI half-width is ~0.8 pp. Reported cross-harness gaps of 5–15 pp are far outside it (Miller, *Adding Error Bars to Evals*, arXiv:2411.00640).
- **Non-determinism at $T=0$ is measurable.** Batch-size and kernel-dependent floating-point reduction order changes outputs; Thinking Machines Lab's *Defeating Nondeterminism in LLM Inference* (2025) shows batch-invariant kernels remove it, at a throughput cost. Atil et al. (arXiv:2408.04667) report non-trivial output variation across repeated identical greedy runs.
- **Few-shot count and example order matter.** Long-established for GPT-3-era models; order effects of several accuracy points on classification tasks are routine.
- **Cost is the reason nobody sweeps.** Perlitz et al., *Efficient Benchmarking* (NAACL 2024, arXiv:2308.11696) show HELM-scale evaluation is dominated by a few expensive scenarios and can be subsampled with bounded rank error — evidence that full factorial sweeps are affordable only if you first cut $n$.

## 5. What Is Not Known

- **Methodologically blocked:** what the invariance class $\mathcal{H}$ *is*. "Semantically equivalent prompt" has no formal definition, so $\Delta$, $\mathrm{Var}(s)$ and any "harness-marginalized score" are defined only relative to a hand-picked family. Until $\mathcal{H}$ is specified, cross-harness reproducibility is not a measurable quantity, only a demonstrable failure.
- **Empirically open:** the variance decomposition. Nobody has published a full factorial sweep over $(\pi,\sigma,\rho,\kappa,\nu,\beta)$ on a fixed model set and benchmark set, so the fraction of $\Delta$ attributable to scoring mechanics ($\kappa,\nu$) versus prompt surface ($\pi$) versus execution stack ($\beta$) is unknown. The experiment is runnable today.
- **Empirically open:** whether harness sensitivity shrinks with scale, instruction tuning, or RLHF. Anecdotally yes for format; not measured on a controlled model ladder.
- **Theoretically open:** conditions under which a ranking is stable. No result of the form "if the score gap exceeds $g(\mathcal{H}, n)$ then no harness in $\mathcal{H}$ inverts the ordering."

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the estimand**. $s(\theta, D, H)$ is a property of the *pair* $(\theta, H)$, but it is reported and consumed as a property of $\theta$ alone. There is no ground-truth $H^\star$ to reproduce against and no measure on $\mathcal{H}$ to average over, so the target quantity does not exist independently of a convention nobody has agreed to.

Two aggravating factors, both concrete:

- **Confounded measurement.** Harness effects are entangled with contamination. A harness that scores the letter token rewards models that memorized the answer key format; one cannot separate "better harness fit" from "more leakage" without training-data access.
- **Compute cost of the honest version.** A $2^6$ factorial over six components, 10 models, 5 benchmarks, at ~$10^4$ items each, is $\sim 3\times 10^7$ inference calls — feasible for a lab, out of reach for the individual reviewer who is asked to check a claim.

## 7. Current Research (as of 2026)

- **Harness standardization.** EleutherAI (`lm-evaluation-harness` task versioning and result hashes), UK AISI (`Inspect` logs full transcripts and configs), HELM (fixed scenario/adapter specification). Direction: make $H$ a citable, versioned object like a `sacreBLEU` signature.
- **Distributional reporting.** Multi-prompt / multi-seed scores with intervals rather than a point estimate (Mizrahi et al.; Miller 2024). Uptake in model cards is still low. *(frontier — verify)*
- **Deterministic inference.** Batch-invariant kernels to remove $\beta$-induced variance (Thinking Machines Lab, 2025); adoption in vLLM/SGLang is in progress. *(frontier — verify)*
- **Contamination-aware evaluation.** Held-out and continuously refreshed benchmarks (LiveBench, LiveCodeBench and successors) to decouple the reproducibility question from the leakage question. *(frontier — verify)*
- **Agentic evals inherit the problem in worse form.** Tool sandboxes, timeouts and retry policy add components to $H$ with no reporting convention at all. *(frontier — verify)*

## 8. Concrete Next Experiment

**"Harness ANOVA": measure where the variance lives.**

- **Scale:** 8 open-weight models spanning 1B–70B and base/instruct pairs; 4 benchmarks (MMLU, GSM8K, ARC-Challenge, HellaSwag), subsampled to $n=2{,}000$ items each (SE $\le 1.1$ pp). Factorial over 6 binary factors: $\pi$ (original vs. alternate template), $\sigma$ ($k{=}0$ vs. $k{=}5$), $\rho$ (log-likelihood ranking vs. generate-and-extract), $\kappa$ (letter token vs. full answer string), $\nu$ (raw vs. length-normalized), $\beta$ (batch 1 vs. batch 64). $2^6 \times 8 \times 4 \times 2000 \approx 4.1\times10^6$ scored items — order $10^3$ A100-hours.
- **Control arm:** the same cell re-run with 3 independent few-shot seeds and 3 repeat runs, giving a within-cell variance floor. Any between-cell effect must exceed this floor to count.
- **Deciding number:** the fraction of total score variance attributable to **scoring mechanics** $(\kappa,\nu)$, $R^2_{\kappa\nu} = \mathrm{Var}_{\kappa\nu}/\mathrm{Var}_{\text{total}}$. If $R^2_{\kappa\nu} > 0.5$, cross-harness reproducibility is an engineering problem: standardize $\kappa,\nu$ in a versioned signature and most of the gap closes. If $R^2_{\kappa\nu} < 0.2$ and the residual sits in $\pi$ and its interactions, the problem is inherent model brittleness, and single-number reporting must be replaced by distributional reporting. Secondary readout: pairwise inversion rate across cells — if $>10\%$ of model pairs invert, no single-harness leaderboard is defensible.

## 9. Key References

- **[Foundational]** Matt Post. *A Call for Clarity in Reporting BLEU Scores.* WMT 2018. — arXiv:1804.08771
- **[Foundational]** Jesse Dodge, Suchin Gururangan, Dallas Card, Roy Schwartz, Noah A. Smith. *Show Your Work: Improved Reporting of Experimental Results.* EMNLP 2019. — arXiv:1909.03004
- **[Foundational]** Joelle Pineau et al. *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program).* JMLR, 2021.
- **[SOTA]** Melanie Sclar, Yejin Choi, Yulia Tsvetkov, Alane Suhr. *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.* ICLR 2024. — arXiv:2310.11324
- **[SOTA]** Norah Alzahrani et al. *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards.* ACL 2024. — arXiv:2402.01781
- **[SOTA]** Moran Mizrahi et al. *State of What Art? A Call for Multi-Prompt LLM Evaluation.* TACL, 2024. — arXiv:2401.00595
- **[SOTA]** Stella Biderman et al. *Lessons from the Trenches on Reproducible Evaluation of Language Models.* 2024. — arXiv:2405.14782
- **[SOTA]** Evan Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* 2024. — arXiv:2411.00640
- **[SOTA]** Yotam Perlitz et al. *Efficient Benchmarking (of Language Models).* NAACL 2024. — arXiv:2308.11696
- **[Survey]** Percy Liang et al. *Holistic Evaluation of Language Models.* TMLR, 2023. — arXiv:2211.09110
- **[Survey]** Samuel R. Bowman, George E. Dahl. *What Will it Take to Fix Benchmarking in Natural Language Understanding?* NAACL 2021. — arXiv:2104.02145
- **[Context]** Benjamin Marie, Atsushi Fujita, Raphael Rubino. *Scientific Credibility of Machine Translation Research: A Meta-Evaluation of 769 Papers.* ACL 2021.
- **[Context]** Clémentine Fourrier et al. *What's going on with the Open LLM Leaderboard?* Hugging Face blog, June 2023. (MMLU implementation discrepancy analysis.)
- **[Context]** Horace He et al. *Defeating Nondeterminism in LLM Inference.* Thinking Machines Lab, 2025.

## 10. Worked Example

One MMLU item, one model, two harnesses.

Item: *"What is the capital of Australia?"* Options: A) Sydney B) Melbourne C) Canberra D) Perth. Gold: C.

**Harness $H_1$ (letter-token scoring, HELM-style).** Prompt ends `Answer:`. Score the next-token logits restricted to `{" A"," B"," C"," D"}`. Suppose softmax over those four gives $(0.11, 0.14, 0.62, 0.13)$ → predicts C → correct.

**Harness $H_2$ (length-normalized continuation likelihood, early lm-eval style).** Score each full continuation. Suppose token-level log-probs sum to:

| Option | $\log p$ | tokens | $\log p/\text{tok}$ |
|---|---|---|---|
| Sydney | $-6.1$ | 2 | $-3.05$ |
| Melbourne | $-7.4$ | 3 | $-2.47$ |
| Canberra | $-9.0$ | 3 | $-3.00$ |
| Perth | $-6.4$ | 2 | $-3.20$ |

Unnormalized argmax picks Sydney ($-6.1$). Length-normalized argmax picks Melbourne ($-2.47$). Both are wrong; the two *variants of the same harness* are wrong in different ways. $H_1$ is right.

**Where the obstruction becomes visible.** The model's internal knowledge did not change between the three scorings — only $\kappa$ and $\nu$ did. Repeat this across 14,042 items and the per-item disagreements aggregate: the Hugging Face analysis found exactly this mechanism producing a ~15 pp swing on LLaMA-65B (~63% vs. ~49%). Now ask the reproducibility question: which of the three is the "true" MMLU score of the model? There is no answer that does not first fix a convention. The rarer, sharper failure is that the convention is not neutral across models — a model whose tokenizer splits `Canberra` into two tokens rather than three gets a different length-normalized score for the *same* belief, so $\nu$ silently rewards tokenizer geometry. That is the non-identifiability of §6 in one table: the reported number is a joint property of model and harness, reported as a property of the model.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*