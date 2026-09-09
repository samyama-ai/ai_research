---
id: 19-evaluation/leaderboard-overfitting-detection
title: "Detecting Benchmark-Specific Overfitting from Public Leaderboards"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Detecting Benchmark-Specific Overfitting from Public Leaderboards

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/leaderboard-overfitting-detection` · **Status:** open

## 1. Problem Statement

A public leaderboard reports scores $\hat{R}_S(f_i)$ for models $f_1,\dots,f_k$ on a fixed held-out set $S$. The scores are the only artifact most readers see. The question: **from the leaderboard alone — scores, submission order, timestamps, submitter identity — can you decide whether a given model's rank is inflated by adaptation to $S$ rather than by improvement on the underlying distribution?**

Three variants, with different difficulty:

- **Measurement.** Given a fresh sample $S'$ from the same distribution, estimate each model's overfitting gap $\Delta_i$. This is solvable but expensive: it requires re-collecting the benchmark. Solved in principle, not in practice.
- **Method.** Estimate $\Delta_i$ *without* $S'$, using only leaderboard-visible signals plus black-box model access. Open.
- **Theory.** Prove a detection guarantee: bounds on the power of any test that distinguishes "adapted to $S$" from "genuinely better", as a function of $n=|S|$, $k$, and the adaptivity of the submission process. Largely open; only the *prevention* side has tight results.

Solving it means: a procedure that outputs, per model, a calibrated interval on $\Delta_i$ with coverage validated against held-out replications on at least three benchmarks, and that flags known-contaminated models at a rate materially above chance.

## 2. Formal Setting

Let $\mathcal{D}$ be the task distribution, $S=\{z_1,\dots,z_n\}\sim\mathcal{D}^n$ the public test set, $\ell$ a bounded loss (accuracy: $\ell\in\{0,1\}$). Measured score and target:

$$\hat{R}_S(f)=\frac{1}{n}\sum_{j=1}^{n}\ell(f,z_j),\qquad R(f)=\mathbb{E}_{z\sim\mathcal{D}}[\ell(f,z)],\qquad \Delta(f)=R(f)-\hat{R}_S(f).$$

$\hat R_S$ is directly observed. $R$ is never observed; it is estimated by $\hat R_{S'}$ on a replication set $S'$, which is the first place the setting leaks.

**Adaptive process.** Submission $i$ is a function of all prior public scores: $f_i=\mathcal{A}_i(\hat{R}_S(f_1),\dots,\hat{R}_S(f_{i-1}),\,W)$, with $W$ the submitter's private state (training data, unreported internal runs). The *reported* $k$ is a lower bound on the true query count $k^\star=k+m$, where $m$ counts unreported private evaluations. $m$ is unobserved and is the dominant confound.

**Detection predicate.** A detector $T$ maps the visible transcript $\tau=\{(\hat{R}_S(f_i),t_i,\mathrm{id}_i)\}_{i\le k}$ to $\hat\Delta_i$ or to a binary flag. Power is measured against ground truth $\Delta_i$ from a replication.

**Decomposition of the gap.** For a replication $S'$ collected by a different pipeline,

$$\hat{R}_{S'}(f)-\hat{R}_S(f)=\underbrace{\Delta_{\text{adapt}}(f)}_{\text{selection on }S}+\underbrace{\Delta_{\text{contam}}(f)}_{S\subseteq\text{ training data}}+\underbrace{b(S,S')}_{\text{distribution shift between collections}}+\underbrace{\varepsilon}_{O(n^{-1/2})}.$$

Only the sum is measurable. Separating the three signal terms is the core identification problem.

**Assumptions, and how they break.**

1. *$S'\sim\mathcal{D}$, same $\mathcal{D}$ as $S$.* Violated. ImageNet-v2 replication reproduced the original protocol and still shifted difficulty; the constant $b$ was large enough to move all models 11–14 points.
2. *Submissions are independent draws given the transcript.* Violated: models share pretraining corpora and architectures, so errors correlate, which mechanically reduces effective adaptivity (Mania et al., 2019).
3. *$k^\star=k$.* Violated by construction — private evaluation before public submission is standard practice.
4. *The test set is not in training data.* Violated at unknown rate for web-scraped LLM pretraining corpora.
5. *Scores are i.i.d. samples of a fixed model.* Violated for prompt-sensitive LLM evals, where reported score varies by several points with formatting alone.

## 3. State of the Art

**Established (prevention, theory).** The Ladder mechanism (Blum & Hardt, ICML 2015) releases a score only when it improves by more than a threshold, and guarantees leaderboard error $O\!\big((\log k)^{1/3} n^{-1/3}\big)$ uniformly over *all* $k$ adaptively chosen submissions — no dependence on $k$ beyond a log. Differential-privacy-based reusable holdout (Dwork et al., *Science* 2015; STOC 2015) gives $\tilde O(\sqrt{k}/n)$-type accuracy for $k$ adaptive queries. Matching hardness: Hardt & Ullman (FOCS 2014) and Steinke & Ullman (COLT 2015) show that with $\approx n^2$ adaptive queries an attacker can force any efficient mechanism to give false answers. These are theorems about *preventing* overfitting, not detecting it after the fact.

**Established (measurement, empirical).** Replication studies: Recht et al. on CIFAR-10 (2018) and ImageNet (ICML 2019); Yadav & Bottou's recovery of the lost MNIST test digits (NeurIPS 2019); Roelofs et al.'s meta-analysis of 120+ Kaggle competitions (NeurIPS 2019). All are *measurements* of $\Delta$, obtained by paying for a second test set.

**Claimed but unablated (detection).** Contamination detectors: Oren et al.'s exchangeability test (ICLR 2024) has a genuine $p$-value under a stated null — the strongest guarantee in this family — but tests whether the *benchmark ordering* was seen, not whether the score is inflated. Min-K% Prob (Shi et al., ICLR 2024) reports strong AUC on WikiMIA, but Duan et al. (COLM 2024) show much of that signal is temporal shift between member and non-member sets, and that membership inference on LLM pretraining data is near-chance once the split is properly controlled. ConStat (Dekoninck et al., NeurIPS 2024) compares performance on a benchmark against a reference benchmark and a reference model set, which is the closest existing thing to a leaderboard-only detector; its calibration outside the tested model families is unablated. Time-Travel-in-LLMs (Golchin & Surdeanu, ICLR 2024) relies on GPT-4-as-judge and has no null distribution.

**Benchmark-number-only results.** GSM1k (Zhang et al., 2024), LiveBench (White et al., ICLR 2025), and "Are We Done with MMLU?" (Gema et al., NAACL 2025) each report gap magnitudes. These are single measurements on single benchmark pairs, not validated detectors.

## 4. What Is Known

- **ImageNet-v2** (Recht et al., ICML 2019): 11–14 point top-1 accuracy drop across ~70 models spanning 2013–2018, but rank order almost fully preserved; the drop is a near-linear function of original accuracy. Interpretation: the gap was dominated by $b(S,S')$, not by $\Delta_{\text{adapt}}$.
- **CIFAR-10** (Recht et al., 2018): 3–15 point drops over 30 models, again monotone in original accuracy.
- **Kaggle meta-analysis** (Roelofs et al., NeurIPS 2019): across 120+ competitions with thousands of submissions each, essentially no evidence of adaptive overfitting on public→private splits when the test set is large; overfitting appears mainly with small private sets (order $10^3$).
- **GSM1k** (Zhang et al., 2024): re-collected 1,250 grade-school math problems matched to GSM8K. Some model families dropped up to ~13 points; several frontier families dropped ~0–2. Accuracy on GSM8K correlated with probability of generating GSM8K examples verbatim.
- **Chatbot Arena** (Singh et al., 2025, *The Leaderboard Illusion*): one provider tested 27 private model variants before a public release; two providers accounted for ~19.2% and ~20.4% of all Arena battle data, against ~29.7% shared across 83 open-weight models. This is direct evidence that $k^\star \gg k$ and is submitter-dependent.
- **Prompt sensitivity** (Alzahrani et al., ACL 2024): single-choice-order or formatting changes on MMLU move models by several points and reorder the top of the leaderboard — the noise floor a detector must beat.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted operational definition of $\Delta_{\text{adapt}}$ separable from $b(S,S')$. Every replication conflates them, and no replication protocol has been shown to drive $b$ to zero. Until $b$ is bounded, no detector can be validated — the ground-truth label does not exist.
- **Theoretically open.** No lower bound on the power of any detector that sees only $\{\hat R_S(f_i)\}$ and black-box model access. The prevention-side hardness results (Steinke–Ullman) do not transfer: they bound the adversary's ability to *cause* overfitting, not the auditor's ability to *see* it.
- **Theoretically open.** Identifiability: is $\Delta_{\text{adapt}}$ identifiable from the score transcript under any nontrivial model of $\mathcal{A}_i$? Score sequences from honest incremental progress and from hill-climbing on $S$ are, absent further assumptions, observationally equivalent.
- **Empirically open.** No study has paired a *pre-registered* replication set with a *known* contamination intervention (deliberately train model families on the test set at controlled dose) at frontier scale, so no detector has a calibrated ROC curve against ground truth.
- **Empirically open.** Whether the Mania et al. model-similarity effect, measured on CIFAR/ImageNet, still holds for LLM leaderboards where models share pretraining corpora far more heavily.

## 6. Why It Is Hard

**Non-identifiability under unobserved query count.** The detector's estimand is $\Delta_{\text{adapt}}$, which depends on $k^\star = k+m$. $m$ — private evaluations before public submission — is unobserved and correlated with submitter resources. A well-funded lab with $m=10^3$ and an honest transcript is indistinguishable, on score data alone, from a lab with $m=0$ that is genuinely better. This is not a compute problem; more data on the public transcript does not resolve it.

**Absent ground truth, compounded.** Validating a detector needs true $\Delta_i$, which needs a replication, which introduces $b(S,S')$ of the same magnitude as the effect being detected: ImageNet-v2's 11–14 points dwarfed any plausible adaptation term. The measurement instrument is noisier than the signal.

**The evaluation does not measure what it names.** "Contamination detection" as practiced measures *memorization of strings*; the leaderboard question is *inflation of rank*. A model can memorize the test set and still be ranked correctly if all competitors did too; a model can be uncontaminated and still rank-inflated through architecture search on public scores.

## 7. Current Research (as of 2026)

- **Continuously refreshed benchmarks** as an engineering workaround: LiveBench (White et al., ICLR 2025, NYU/Abacus.AI), LiveCodeBench (Jain et al., UC Berkeley), SWE-bench-Verified-style curated refreshes. These reduce $\Delta_{\text{adapt}}$ but do not detect it in existing leaderboards.
- **Statistical contamination tests with valid nulls**: Oren et al. (Stanford, Hashimoto group); ConStat (Dekoninck, Vechev, ETH SRI). Direction: reference-benchmark comparison to normalize away $b$. *(frontier — verify: extensions to open-ended and preference leaderboards.)*
- **Arena hygiene and submission accounting**: Cohere Labs / MIT / Stanford work following *The Leaderboard Illusion*, plus LMArena's policy responses on private-variant limits and data-share disclosure. *(frontier — verify current policy state.)*
- **Adaptive-data-analysis revival for eval infrastructure**: Ladder-style thresholded score release and DP holdouts applied to LLM eval harnesses. Discussed more than deployed.
- **Membership-inference skepticism**: Duan et al. (UW/AI2) and follow-ups argue most reported MIA success on LLM pretraining is temporal-shift artifact — the main negative result constraining this space.

## 8. Concrete Next Experiment

**Question:** can any leaderboard-only detector separate adaptation from distribution shift, at a rate better than chance, against known ground truth?

**Scale.** Build a synthetic leaderboard with planted labels. Take a 7B open-weight base model. Produce 60 fine-tuned variants: 20 **clean** (instruction data only), 20 **contaminated** at controlled dose (the MMLU test split injected at $10^{-5}$, $10^{-4}$, $10^{-3}$ of tokens; 5 variants each at three doses, plus 5 at dose 0 within this arm), 20 **adapted-not-contaminated** (hill-climb hyperparameters and prompt template against the public MMLU score, 200 private evaluations each, no test text ever in training). Record the full transcript: public score, private query count $m$, submission order. Total cost is dominated by 60 fine-tunes at 7B — roughly $10^3$–$10^4$ GPU-hours, feasible on one node-month.

**Control arm.** A pre-registered replication set: 3,000 fresh MMLU-style items written to the original protocol by annotators with no access to the models, plus a *paraphrase-only* arm (existing MMLU items rewritten, same semantics) that holds $\mathcal{D}$ fixed and therefore estimates $b(S,S')$ directly. The paraphrase arm is the control that no prior replication study has had.

**The deciding number.** Run each candidate detector — ConStat, Min-K% Prob, Oren exchangeability, and a score-trajectory baseline — over the transcript. Report **AUROC for separating the 20 adapted-not-contaminated variants from the 20 clean variants**, after regressing out $\hat b$ measured on the paraphrase arm. Threshold: **AUROC $\ge 0.80$ with a bootstrap 95% CI excluding 0.65** means leaderboard-only detection of adaptation is viable. AUROC $\le 0.65$ means the method variant is empirically closed for this class of detector, and the field should redirect to prevention (Ladder-style release) rather than detection.

Secondary number: contaminated-vs-clean AUROC at each dose, which measures how much of any apparent success is memorization detection rather than adaptation detection.

## 9. Key References

- **[Foundational]** Avrim Blum, Moritz Hardt. *The Ladder: A Reliable Leaderboard for Machine Learning Competitions.* ICML, 2015. — arXiv:1502.04585
- **[Foundational]** Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, Aaron Roth. *The reusable holdout: Preserving validity in adaptive data analysis.* Science 349(6248), 2015.
- **[Foundational]** Moritz Hardt, Jonathan Ullman. *Preventing False Discovery in Interactive Data Analysis is Hard.* FOCS, 2014. — arXiv:1408.1655
- **[Foundational]** Thomas Steinke, Jonathan Ullman. *Interactive Fingerprinting Codes and the Hardness of Preventing False Discovery.* COLT, 2015.
- **[Foundational]** Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, Vaishaal Shankar. *Do ImageNet Classifiers Generalize to ImageNet?* ICML, 2019. — arXiv:1902.10811
- **[Foundational]** Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, Vaishaal Shankar. *Do CIFAR-10 Classifiers Generalize to CIFAR-10?* 2018. — arXiv:1806.00451
- **[Foundational]** Rebecca Roelofs, Vaishaal Shankar, Benjamin Recht, Sara Fridovich-Keil, Moritz Hardt, John Miller, Ludwig Schmidt. *A Meta-Analysis of Overfitting in Machine Learning.* NeurIPS, 2019.
- **[SOTA]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR, 2024. — arXiv:2310.17623
- **[SOTA]** Jasper Dekoninck, Mark Niklas Müller, Maximilian Baader, Marc Fischer, Martin Vechev. *ConStat: Performance-Based Contamination Detection in Large Language Models.* NeurIPS, 2024. — arXiv:2405.16281
- **[SOTA]** Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer. *Detecting Pretraining Data from Large Language Models.* ICLR, 2024. — arXiv:2310.16789
- **[SOTA]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[SOTA]** Hongyi Zhang et al. (Scale AI). *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS, 2024. — arXiv:2405.00332
- **[SOTA]** Colin White et al. *LiveBench: A Challenging, Contamination-Limited LLM Benchmark.* ICLR, 2025. — arXiv:2406.19314
- **[SOTA]** Shivalika Singh et al. *The Leaderboard Illusion.* 2025. — arXiv:2504.20879
- **[Supporting]** Horia Mania, John Miller, Ludwig Schmidt, Moritz Hardt, Benjamin Recht. *Model Similarity Mitigates Test Set Overuse.* NeurIPS, 2019.
- **[Supporting]** Chhavi Yadav, Léon Bottou. *Cold Case: The Lost MNIST Digits.* NeurIPS, 2019. — arXiv:1905.10498
- **[Supporting]** Norah Alzahrani et al. *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards.* ACL, 2024. — arXiv:2402.01781
- **[Survey]** Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, Eneko Agirre. *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark.* Findings of EMNLP, 2023.
- **[Survey]** Samuel R. Bowman, George E. Dahl. *What Will it Take to Fix Benchmarking in Natural Language Understanding?* NAACL, 2021. — arXiv:2104.02145

## 10. Worked Example

**Setting.** MMLU, $n=14{,}042$ items. Two models on the same public leaderboard:

| | Model A | Model B |
|---|---|---|
| Reported MMLU | 78.4 | 79.1 |
| Public submissions $k$ | 1 | 1 |
| Private evals $m$ | 4 (claimed) | 240 (undisclosed) |

**Sampling noise.** Binomial standard error at $p=0.79$, $n=14{,}042$: $\sqrt{0.79\cdot0.21/14042}=0.34$ points. The 0.7-point gap is $\approx 2.1\sigma$ — nominally significant under the i.i.d. null.

**What adaptation buys.** Under a max-of-$m$ selection model with per-eval score noise $\sigma_{\text{eval}}$, selecting the best of $m$ runs inflates the expected reported score by about $\sigma_{\text{eval}}\sqrt{2\ln m}$. Prompt-formatting sensitivity on MMLU is roughly $\sigma_{\text{eval}}\approx 1.0$ point (Alzahrani et al., ACL 2024) — an order of magnitude above the binomial 0.34, because the variation is over prompts and decoding, not over items. Then:

- Model A, $m=4$: expected inflation $1.0\cdot\sqrt{2\ln 4}=1.66$ points.
- Model B, $m=240$: expected inflation $1.0\cdot\sqrt{2\ln 240}=3.31$ points.

**The bite.** The *differential* inflation, $3.31-1.66=1.65$ points, is more than twice the observed 0.7-point gap. Model B's lead is fully explained by having run 240 private evaluations. But $m=240$ is not on the leaderboard; the transcript shows one submission each.

**Why replication does not rescue it.** Re-collect MMLU-style items and both models drop. ImageNet-v2 measured a 11–14 point common drop; a plausible MMLU replication shift is 3–8 points. The quantity you want, $\Delta_{\text{adapt},B}-\Delta_{\text{adapt},A}\approx 1.65$, sits inside a common shift $b$ that is 2–5× larger and that you cannot subtract off without assuming it is model-independent — an assumption ImageNet-v2 supported only because the drop happened to be a smooth function of original accuracy.

**The obstruction, made visible.** Two unobserved scalars — $m_A$, $m_B$ — and one unidentified nuisance term $b$ jointly determine the only difference the leaderboard reports. The detector has three unknowns and one equation. Fixing this requires either disclosure of $m$ (a governance change) or a thresholded score-release mechanism that makes $m$ irrelevant (Ladder). No amount of statistics on the published numbers closes it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*