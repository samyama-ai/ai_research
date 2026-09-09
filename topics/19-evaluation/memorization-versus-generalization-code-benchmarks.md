---
id: 19-evaluation/memorization-versus-generalization-code-benchmarks
title: "Distinguishing Memorization from Generalization on Code Benchmarks"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distinguishing Memorization from Generalization on Code Benchmarks

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/memorization-versus-generalization-code-benchmarks` · **Status:** open

## 1. Problem Statement

A model scores $p$ on a code benchmark (HumanEval, MBPP, SWE-bench, LiveCodeBench). Decide how much of $p$ is attributable to the benchmark items, or near-duplicates of them, having been in the training corpus, versus a capability that transfers to items drawn from the same task distribution but absent from training.

Three variants, with different difficulty:

- **Measurement.** Given a model, a benchmark, and (optionally) the training corpus, estimate the *contamination-attributable score inflation* $\delta$. Solved would mean: a procedure that returns $\delta$ with a stated confidence interval and is validated against a ground-truth setting where $\delta$ is known by construction.
- **Method.** Build benchmarks or scoring rules whose measured score is provably insensitive to corpus overlap — e.g. continuously refreshed item pools, or per-item transformations that preserve difficulty while destroying surface form.
- **Theory.** Characterize when a copied solution is even distinguishable from a generalizing one. For code, the target output is often the unique short correct program; memorization and generalization can be observationally identical on the output, and differ only in the mechanism.

The measurement variant is the bottleneck: contamination *detection* (was item $x$ seen?) is routinely conflated with contamination *effect* (how much did seeing $x$ raise the score?), and these are different quantities.

## 2. Formal Setting

Let $\mathcal{D}$ be a task distribution over coding problems $x=(\text{spec}, \mathcal{T})$ with $\mathcal{T}$ a test suite. A benchmark is a finite sample $B=\{x_i\}_{i=1}^n \sim \mathcal{D}$. Model $M_\theta$ trained on corpus $C$ produces programs $y \sim M_\theta(\cdot\mid x)$.

**Measured score.** $\text{pass@}k$ is estimated unbiasedly from $N \ge k$ samples with $c$ passing:
$$\widehat{\text{pass@}k}(x) = 1 - \binom{N-c}{k}\binom{N}{k}^{-1}, \qquad p = \tfrac{1}{n}\sum_i \widehat{\text{pass@}k}(x_i).$$
"Pass" is decided by executing $y$ against $\mathcal{T}$ — so $p$ is defined relative to the test suite, not to correctness.

**Contamination-attributable inflation.** The causal quantity is
$$\delta = \mathbb{E}_{x\sim\mathcal{D}}\big[\text{pass}(M_{\theta(C)}, x)\big] - \mathbb{E}_{x\sim\mathcal{D}}\big[\text{pass}(M_{\theta(C \setminus S(x))}, x)\big],$$
where $S(x)\subseteq C$ is the set of documents "about" $x$. $\delta$ requires a counterfactual retrain per item and is never measured directly at frontier scale.

**Overlap proxies (what is actually computed).**
- Surface overlap: $\text{sim}(x, C) = \max_{d \in C} J_k(x, d)$, $k$-gram Jaccard or MinHash/LSH similarity, thresholded (typically $J_{13} > 0.5$ or an $n$-gram-containment rule).
- Semantic overlap: cosine similarity of embeddings of the reference solution against retrieved corpus code.
- Likelihood-based membership: Min-K% Prob $= \frac{1}{|K|}\sum_{t \in K}\log P_\theta(x_t \mid x_{<t})$ over the $K\%$ lowest-probability tokens; Min-K%++ normalizes by the per-position mean and variance of the next-token log-probability.
- Exchangeability test (Oren et al.): under no contamination, $\log P_\theta(B)$ is invariant to the ordering of items; a one-sided permutation test on canonical vs. shuffled order gives a valid $p$-value for verbatim inclusion of the benchmark file.
- Temporal split: $\Delta_{\text{time}} = p(\text{items released} < \text{cutoff}) - p(\text{items released} > \text{cutoff})$.

**Assumptions, and which are violated.**
1. *$C$ is inspectable.* Violated for every frontier model; also violated for "open" models with undisclosed mid-training and RL data.
2. *Contamination is verbatim.* Violated — rephrased and translated variants of benchmark items evade $n$-gram detection while retaining the score benefit.
3. *Pre- and post-cutoff items are exchangeable.* Violated — LeetCode/Codeforces problem difficulty and topic distributions drift over time, so $\Delta_{\text{time}}$ mixes contamination with distribution shift.
4. *Passing the test suite equals correctness.* Violated — HumanEval ships a median of a handful of tests per problem; EvalPlus shows the gap is large.
5. *Membership inference has power at this scale.* Largely violated for single-epoch, near-deduplicated pretraining.

## 3. State of the Art

**Established.**
- *Detection of verbatim benchmark inclusion* is solved for the black-box case where the benchmark has a canonical order: Oren et al. (ICLR 2024) give a permutation test with a valid $p$-value and no false-positive inflation.
- *Extractable memorization exists and scales.* Carlini et al. (ICLR 2023) show memorization grows log-linearly in model size, in the number of duplicates of a string, and in prompt context length, measured on GPT-Neo/Pythia up to 6B and 13B.
- *Deduplication changes outcomes.* Lee et al. (ACL 2022) show near-duplicate removal reduces emitted memorized text by roughly 10× with no loss in perplexity.
- *Test suites under-constrain.* EvalPlus (NeurIPS 2023) grows HumanEval's tests by ~80× and drops reported pass@1 by up to ~19% relative for then-SOTA models.

**Claimed but unablated.**
- Perplexity- and Min-K%-based contamination scores for individual code problems. Duan et al. (COLM 2024) show membership inference on LLM pretraining performs near chance (AUC $\approx$ 0.5–0.6) once member/non-member sets are matched for time and topic; most contamination claims built on these detectors inherit that weakness.
- "Model X is contaminated on HumanEval by $q\%$" numbers from embedding-similarity retrieval (Riddell, Ni & Cohan, ACL 2024; Matton et al., EMNLP Findings 2024). These establish that overlap exists and correlates with per-item pass rates; they do not establish $\delta$, because no counterfactual model is trained.

**Benchmark-number-only results.** LiveCodeBench (ICLR 2025) and LiveBench (ICLR 2025) report post-cutoff score drops for several model families. These are single measurements without a difficulty-matched control arm, so the drop is a *composite* of contamination and temporal difficulty drift.

## 4. What Is Known

- HumanEval is 164 problems; the pass@1 standard error at $p=0.8$ is $\sqrt{0.8\cdot 0.2/164}\approx 3.1$ points. Differences below ~6 points between models are not resolvable on this benchmark regardless of contamination.
- GSM1k (Zhang et al., NeurIPS 2024), a held-out reconstruction of GSM8K's distribution with 1250 items, found accuracy drops up to ~13 points for some open model families and near-zero gaps for the strongest frontier models — the closest existing thing to a clean $\delta$ estimate, in math rather than code.
- Rephrased benchmark items (Yang et al., 2023) recover most of the contamination benefit while passing standard $n$-gram decontamination — measured on 13B-scale models fine-tuned on rephrased test sets.
- Counterfactual-task perturbation (Wu et al., NAACL 2024) — e.g. base-9 arithmetic, 1-indexed Python — produces large, consistent accuracy drops relative to the default-condition task, at GPT-4-era scale. This bounds how much of the default-condition score is condition-specific rather than procedural.
- SWE-bench-family audits (Aleithan et al., 2024) find solution leakage in issue reports and weak test suites for a meaningful fraction of instances; OpenAI's SWE-bench Verified retains 500 of 2294 instances after human filtering, most removals for under-specification or broken tests rather than for contamination.
- Memorization is partly predictable: Biderman et al. (NeurIPS 2023) show low-cost small-model runs predict *which sequences* a larger model will memorize with useful but far-from-perfect precision, on the Pythia suite.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed operational definition of $\delta$ that is computable without retraining. Every deployed metric measures overlap or membership, then asserts a score effect. The blocking issue is definitional, not computational: "the model saw a solution to a similar problem" has no non-arbitrary threshold, and for short canonical programs the memorization/generalization distinction may not be identifiable from behaviour at all.
- **Empirically open.** The dose-response curve of duplication count $\to$ code-benchmark pass@1 has not been measured at 7B+ scale with controlled injection. Pieces exist (Carlini's duplication scaling for *extraction*), but extraction $\neq$ downstream task score.
- **Empirically open.** Whether post-training (SFT/RL on verified solutions) creates a distinct and larger contamination channel than pretraining. RL-on-benchmark-adjacent-tasks is now standard and almost entirely unaudited.
- **Theoretically open.** No bound relates corpus overlap statistics to worst-case score inflation. Nothing forbids $\delta$ being large while all overlap proxies read zero (rephrasing) or $\delta \approx 0$ while proxies read high (the problem is genuinely easy and also common).

## 6. Why It Is Hard

The obstruction is **non-identifiability under an absent counterfactual**. $\delta$ is defined by a retrain-without-$S(x)$ arm that costs a full pretraining run per ablation — $10^{23}$–$10^{25}$ FLOPs — so it is never run. Every practical substitute is confounded:

- Overlap proxies condition on the corpus, which is undisclosed for exactly the models whose scores matter.
- Membership inference is near-chance on single-epoch pretraining once time and topic are matched, so per-item verdicts carry almost no information.
- Temporal splits confound contamination with difficulty drift; there is no difficulty-matched control arm.
- The target is degenerate: for `def has_close_elements(numbers, threshold)`, the correct program is short and near-unique. A generalizing model and a copying model emit the same tokens. The label is unrecoverable from the output.

## 7. Current Research (as of 2026)

- **Live/refreshing benchmarks.** LiveCodeBench (UC Berkeley/UIUC), LiveBench (Abacus/NYU/Nvidia), Codeforces-tracking leaderboards. Direction: make $|C \cap B| = 0$ by construction. Cost: difficulty non-stationarity, and the pool is exhausted as it is published.
- **Programmatic item generation.** Functional/parametric benchmarks (Srivastava et al., 2024) that emit fresh instances from a template, giving an in-distribution control arm. *(frontier — verify current code-specific instantiations.)*
- **Open-corpus provenance.** AI2 (OLMo/Dolma) and EleutherAI (Pythia) offer full corpora plus checkpoints, which is where controlled injection studies are actually runnable.
- **Detector improvement.** Min-K%++ (ICLR 2025) and distributional divergence tests (CDD/TED, ACL 2024). Progress is real but bounded by the Duan et al. near-chance result.
- **Post-training contamination audits.** *(frontier — verify.)* Little published work isolates RL/SFT data leakage on SWE-bench-style agentic benchmarks.

## 8. Concrete Next Experiment

**Controlled duplication dose-response at 7B, with a difficulty-matched control arm.**

- **Scale.** OLMo 2 7B, continued pretraining for 50B tokens (~$2\times10^{22}$ FLOPs, order 4k A100-hours per arm; 1 arm suffices if injection counts are crossed within it).
- **Items.** Construct 400 fresh Python problems in HumanEval/MBPP style with 30+ tests each, verified absent from Dolma by MinHash ($J_{13} < 0.3$) and embedding retrieval. Split into 5 groups of 80.
- **Treatment.** Inject solution-bearing documents at duplication counts $m \in \{0, 1, 4, 16, 64\}$, one count per group, randomized over groups. Add a sixth group of 80: $m=16$ but GPT-rephrased solutions in a different variable-naming and comment style, to test whether surface-form decontamination is protective.
- **Control arm.** The $m=0$ group, difficulty-matched by construction (same generator, same test density, randomized assignment). This is what temporal splits lack.
- **Deciding number.** The duplication elasticity
$$\hat{\eta} = \frac{\partial\, \text{pass@}1}{\partial \log_2 m},$$
estimated by regression across groups, with a bootstrap CI over items. If $\hat{\eta} \le 1$ point per doubling and the $m=64$ vs $m=0$ gap has a 95% CI excluding 5 points, then realistic incidental contamination (a handful of copies) cannot explain double-digit benchmark gaps, and the field's contamination narrative is wrong about magnitude. If $\hat{\eta} \ge 3$ points per doubling, single-digit duplicate counts are sufficient to move leaderboards and per-item overlap auditing becomes mandatory.
- **Secondary readout.** Min-K%++ AUC for member vs non-member items at each $m$ — quantifies how many duplicates a detector needs before it beats chance. Expect near-chance at $m \le 4$.

## 9. Key References

- **[Foundational]** Chen, M. et al. *Evaluating Large Language Models Trained on Code.* 2021 — arXiv:2107.03374
- **[Foundational]** Austin, J. et al. *Program Synthesis with Large Language Models.* 2021 — arXiv:2108.07732
- **[Foundational]** Carlini, N. et al. *Quantifying Memorization Across Neural Language Models.* ICLR 2023 — arXiv:2202.07646
- **[Foundational]** Lee, K. et al. *Deduplicating Training Data Makes Language Models Better.* ACL 2022 — arXiv:2107.06499
- **[SOTA]** Oren, Y., Meister, N., Chatterji, N., Ladhak, F., Hashimoto, T. *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024 — arXiv:2310.17623
- **[SOTA]** Jain, N. et al. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR 2025 — arXiv:2403.07974
- **[SOTA]** Liu, J., Xia, C. S., Wang, Y., Zhang, L. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.* NeurIPS 2023 — arXiv:2305.01210
- **[SOTA]** Zhang, H. et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS 2024 — arXiv:2405.00332
- **[SOTA]** Zhang, J. et al. *Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models.* ICLR 2025 — arXiv:2404.02936
- **[Analysis]** Duan, M. et al. *Do Membership Inference Attacks Work on Large Language Models?* COLM 2024 — arXiv:2402.07841
- **[Analysis]** Riddell, M., Ni, A., Cohan, A. *Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models.* ACL 2024 — arXiv:2403.04811
- **[Analysis]** Matton, A. et al. *On Leakage of Code Generation Evaluation Datasets.* Findings of EMNLP 2024 — arXiv:2407.07565
- **[Analysis]** Yang, S. et al. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023 — arXiv:2311.04850
- **[Analysis]** Wu, Z. et al. *Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks.* NAACL 2024 — arXiv:2307.02477
- **[Analysis]** Biderman, S. et al. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS 2023 — arXiv:2304.11158
- **[Benchmark]** Jimenez, C. et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024 — arXiv:2310.06770
- **[Benchmark]** Aleithan, R. et al. *SWE-Bench+: Enhanced Coding Benchmark for LLMs.* 2024 — arXiv:2410.06992
- **[Survey]** Deng, C. et al. *Investigating Data Contamination in Modern Benchmarks for Large Language Models.* NAACL 2024 — arXiv:2311.09783

## 10. Worked Example

Take HumanEval/0, `has_close_elements(numbers, threshold)`: return whether any two numbers are closer than `threshold`. The canonical solution is a double loop with an `abs()` comparison — about 6 lines.

**Step 1 — detection.** MinHash the reference solution against The Stack. Thousands of GitHub files contain a near-identical double-loop proximity check. $J_{13} > 0.5$ against many documents. Verdict: "contaminated."

**Step 2 — the confound.** The same evidence supports the opposite reading. The pattern is common *because it is the obvious solution*. A model that never saw HumanEval but saw 10,000 double loops would emit the same program. The detector fires on both hypotheses with equal strength.

**Step 3 — the counterfactual is unavailable.** $\delta$ for this item requires a model trained on The Stack minus every proximity-check file. That is a full pretraining run to resolve one of 164 items.

**Step 4 — what a proxy would say.** Suppose contaminated items score 92% and clean items 78%, a 14-point gap over $n=164$ split roughly 60/104. The standard error of the difference is about $\sqrt{0.92\cdot0.08/60 + 0.78\cdot0.22/104} \approx 4.9$ points, so the gap is nominally significant. But the split is not randomized: contaminated items are contaminated *because* they are common, and common problems are easier. Difficulty is a confounder with the same sign as the treatment, and no post-hoc adjustment identifies it.

**Step 5 — the injection design breaks the tie.** In §8, assignment of $m$ is randomized over items generated by one process. Difficulty is balanced in expectation, so the $m$-to-score slope is causal. That is the entire content of the experiment: the observational study cannot separate "seen" from "easy," and one randomized 50B-token run can.

**Obstruction made visible:** the strongest available evidence for contamination on HumanEval/0 — high corpus overlap plus a high pass rate — is exactly what a fully generalizing model on an easy problem also produces.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*