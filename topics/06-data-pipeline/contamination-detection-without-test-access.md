---
id: 06-data-pipeline/contamination-detection-without-test-access
title: "Benchmark Contamination Detection Without Test-Set Access"
topic: 06-data-pipeline
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Benchmark Contamination Detection Without Test-Set Access

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/contamination-detection-without-test-access` · **Status:** methodologically-blocked

## 1. Problem Statement

A benchmark holder wants to know whether a model's score on their benchmark is inflated because the benchmark's items entered the model's training data. Every deployed detector needs one of two things the auditor usually does not have:

1. **the training corpus** (for $n$-gram or substring overlap), or
2. **the test items in the clear** (for per-example likelihood tests, perplexity gaps, canonical-order permutation tests).

Requirement 2 is self-defeating: to run the test you must hand the test set to the model provider, which contaminates it. Requirement 1 is unavailable for every frontier model.

**The problem.** Given only (a) black-box sampling or logprob access to a model $M$, (b) aggregate score statistics on a private benchmark $B$, and (c) public artifacts (a public split, item metadata, release dates), decide whether $B$ was contaminated in $M$'s training data, with calibrated false-positive control.

Three variants, different difficulty:

- **Measurement variant.** Define a contamination quantity that is identifiable from the available signals. *This is the blocked one.*
- **Method variant.** Given a working definition, build a detector with a controlled false-positive rate. Partially solved when the auditor holds the test set (Oren et al. 2024); unsolved without it.
- **Theory variant.** Characterize when contamination is information-theoretically detectable from $k$ black-box queries under a memorization model. Open.

**Solved** would mean: a procedure that, on a benchmark whose contamination status is known by construction, achieves TPR $\geq 0.8$ at FPR $\leq 0.05$ without the auditor ever revealing a test item, and holds that operating point when the contaminated and clean sets differ in difficulty.

## 2. Formal Setting

Let $B=\{(x_i,y_i)\}_{i=1}^{n}$ be the private benchmark, $D$ the training corpus of $M_\theta$, and $\mathcal{A}$ the auditor's access set.

**Contamination indicator.** For a similarity kernel $s$ and threshold $\tau$,
$$C_\tau(B,D) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\!\left[\max_{d\in D} s\big((x_i,y_i),d\big) \ge \tau\right].$$
As measured: $s$ is 13-gram containment (GPT-3, NeurIPS 2020), 8-gram Jaccard, or an embedding cosine. $\tau$ is chosen by hand; **no principled value exists**, and $C_\tau$ is not monotone in reported inflation.

**Effect quantity (what actually matters).** Let $A(M,B)$ be accuracy, and $M^{-B}$ the counterfactual model trained on $D\setminus\{d: s(\cdot,d)\ge\tau\}$. The inflation is
$$\Delta(B) = \mathbb{E}\big[A(M,B)\big] - \mathbb{E}\big[A(M^{-B},B)\big].$$
$\Delta$ is the target, and it is **not measurable** without a retrain — the counterfactual costs one full pretraining run per benchmark.

**Detector.** $T:\mathcal{A}\to\{0,1\}$ with size $\alpha$: $\Pr[T=1 \mid \text{clean}] \le \alpha$.

**Access sets.**
- $\mathcal{A}_{\text{full}} = \{B, \text{logprobs}\}$: enables the exchangeability test — under a clean model, the log-likelihood of the canonical item order is exchangeable with a random permutation $\pi$, giving an exact $p$-value $\Pr_\pi[\log p_\theta(B_\pi) \ge \log p_\theta(B)]$ (Oren et al., ICLR 2024).
- $\mathcal{A}_{\text{blind}} = \{\text{samples from } M, \; A(M,B), \; \text{public split } B_{\text{pub}}, \; \text{dates}\}$: the setting of this problem. The auditor may query $M$ with items it constructs, but never with items of $B$.

**Assumptions, and which are violated.**
- *IID clean reference set exists.* Violated: "clean" sets are built post-cutoff, so they differ in topic, style and difficulty from $B$ (Duan et al., COLM 2024; Das et al. 2024).
- *Contaminated items were seen verbatim.* Violated: rephrased and translated variants evade 13-gram matching while transferring the score gain (Yang et al. 2023).
- *One exposure regime.* Violated: duplication counts span $1$ to $10^4$; memorization scales log-linearly in duplicates (Carlini et al., ICLR 2023).
- *Logprobs available.* Violated for most frontier APIs.

## 3. State of the Art

**With test-set access (established).** Oren et al., *Proving Test Set Contamination in Black Box Language Models* (ICLR 2024): exchangeability test, exact finite-sample $p$-value, no training-data access; detects contamination injected at 2 duplications in a 1.4B-parameter model, and flags real benchmarks in open models. This is the strongest *established* result in the area — it is a proof-style guarantee, not a heuristic. It requires $B$ in the clear and an ordered benchmark file, and it detects only *order-preserving* exposure.

**Canary/hash mitigation (established as engineering, not as detection).** Jacovi et al. (EMNLP 2023) — encrypt test sets, add canary strings, refuse derivative uploads. Prevents, does not detect.

**Behavioral probes (claimed, largely unablated).** Golchin & Surdeanu, *Time Travel in LLMs* (ICLR 2024): guided-instruction completion plus GPT-4 judging, reported 92–100% accuracy in identifying contaminated splits — but the evaluation uses splits whose contamination status is inferred, not constructed, and the judge is itself a contaminated model. Deng et al.'s TS-Guessing (NAACL 2024) reports GPT-4 recovering masked MMLU options at 57%; that is a benchmark number, with no clean-model control at matched difficulty.

**Membership inference (established negative).** Min-K% Prob (Shi et al., ICLR 2024) and Min-K%++ (Zhang et al., ICLR 2025) report AUC $\approx$ 0.70–0.85 on WikiMIA. Duan et al. (COLM 2024) show near-chance AUC (0.5–0.55) on MIMIR across Pythia 160M–12B when member/non-member sets are $n$-gram-decontaminated and temporally matched. Das, Zhang & Tramèr (2024) show blind baselines — classifiers that never see the model — match or beat published MIAs on several benchmarks, i.e. the reported AUC measured distribution shift, not membership.

**Held-out reconstruction (systems SOTA).** GSM1k (Zhang et al., NeurIPS 2024 D&B): 1,250 new grade-school problems matched to GSM8k by human authoring; some model families drop up to ~13 accuracy points, others ~0. LiveBench (White et al., ICLR 2025): monthly-refreshed items. Both are *avoidance*, not detection, and their control for difficulty is human judgment.

## 4. What Is Known

- **Memorization scales predictably.** Verbatim emission grows log-linearly with model size, duplicate count, and prompt-prefix length (Carlini et al., ICLR 2023; 125M–12B Pythia/GPT-Neo).
- **Contamination is widespread and undocumented.** Sainz et al. (EMNLP Findings 2023) catalog contaminated evaluations across mainstream benchmarks; Dodge et al. (EMNLP 2021) found benchmark test items inside C4.
- **Rephrasing defeats $n$-gram detection.** A 13-gram-clean rephrased GSM8k/MMLU training set still lifts test scores to near-saturation (Yang et al. 2023).
- **MIA at LLM scale is near-chance under matched controls.** AUC 0.5–0.55, Pythia 160M–12B, MIMIR (Duan et al. 2024); confirmed by the SoK of Meeus et al. (SaTML 2025), which shows most reported gains vanish under randomized member/non-member splits.
- **Exact tests exist when you hold the data.** Order-exchangeability gives valid $p$-values with zero distributional assumptions on the text (Oren et al. 2024).
- **The counterfactual is affordable only at small scale.** A clean-vs-injected pretraining pair is routine at $\leq$1.4B parameters; at frontier scale nobody has published one.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no agreed measurand. $C_\tau$ (overlap) is not what anyone cares about; $\Delta$ (inflation) requires a counterfactual retrain. Every blind detector therefore reports a statistic whose relationship to $\Delta$ is unestablished, and "contaminated" has no operational definition an auditor and a provider would both accept.
- **Theoretically open.** No lower bound on the number of black-box queries needed to distinguish a model trained on $B$ from one trained on an equal-difficulty $B'$, under any memorization model. No impossibility theorem either — the non-identifiability is folklore, not proved.
- **Empirically open.** Nobody has published a controlled injection study at $\geq$7B parameters with a matched-difficulty clean twin benchmark, sweeping duplication count $\{0,1,4,16,64\}$ and paraphrase level, evaluating blind detectors. All ingredients exist; the run costs one pretrain per arm.

## 6. Why It Is Hard

**Non-identifiability under difficulty confounding.** A blind detector sees $A(M,B)$ high and $A(M,B')$ lower for a supposedly-matched control $B'$. Two explanations produce identical data: (i) $B$ leaked; (ii) $B'$ is harder. Without a joint model of item difficulty, these are not separable from scores alone — which is exactly the failure Das et al. and Duan et al. demonstrated for MIA, where the "signal" was temporal distribution shift.

**Absent ground truth.** Contamination labels for frontier models do not exist. Detectors are validated on proxies (pre/post-cutoff splits) that are themselves confounded, so a reported AUC of 0.85 may be an AUC of 0.5 on membership plus 0.35 on recency.

**Compute cost of the only clean measurement.** $\Delta$ needs $M^{-B}$. At 7B/1T tokens that is $\sim$10$^{23}$ FLOP per arm; a five-arm duplication sweep is a research budget, not an audit.

## 7. Current Research (as of 2026)

- **Private-set audit protocols.** Encrypted/hashed benchmark serving with rate-limited scoring; provider-side canaries. Descendant of Jacovi et al. 2023; adopted by SWE-bench-style private splits and LiveBench's refresh cycle. *(frontier — verify current deployments.)*
- **Matched-twin benchmark construction.** GSM1k's methodology extended to code and multi-hop QA, with item-response-theory difficulty matching rather than authorial judgment. *(frontier — verify.)*
- **Dataset inference over sets rather than items.** Maini et al., *LLM Dataset Inference* (NeurIPS 2024): aggregate many weak per-item statistics and test at the set level, which recovers power the per-item MIA loses. The natural bridge to the blind setting, but still needs the items.
- **Cryptographic commitments.** Publishing hashes of test items pre-release so post-hoc leakage is provable; no adopted standard.
- **Surveys.** Xu et al., *Benchmark Data Contamination of Large Language Models: A Survey* (2024) is the current map.

## 8. Concrete Next Experiment

**Question:** can any blind detector beat a difficulty-matched control arm?

**Scale.** Pretrain three 1.4B-parameter models on 300B tokens (Pythia recipe, identical seeds and data order except for injection). Inject a 1,000-item benchmark $B$ at duplication counts $\{0, 4, 64\}$; add a fourth arm injecting only *paraphrases* of $B$ (13-gram-clean) at 64×. Cost: ~4 × $10^{21}$ FLOP, days on 64 A100s.

**Control arm.** $B'$: 1,000 items authored to the same specification as $B$, never injected in any arm, with per-item difficulty matched to $B$ by 2-parameter IRT fitted on 20 held-out models. The clean-model gap $A(M_{0\times},B) - A(M_{0\times},B')$ measures residual difficulty mismatch and is the null.

**Blind detectors under test** (auditor never sees $B$): score-gap on the public split; TS-Guessing on public items only; sampled-continuation self-consistency; distributional drift of generated answer formats.

**The deciding number.** Detector AUC for separating $\{4\times, 64\times\}$ from $\{0\times\}$, with the decision threshold calibrated so that FPR on the clean-model $B$-vs-$B'$ comparison is $0.05$. **AUC $\geq 0.80$ at 64× duplication means blind detection is viable; AUC $\leq 0.60$ means the measurand, not the detector, is the problem, and the field should standardize on private-set protocols instead.** Report the paraphrase arm separately: if its AUC drops below the verbatim arm by $>0.15$, blind detection is measuring lexical echo, not contamination.

## 9. Key References

- **[Foundational]** Brown et al. *Language Models are Few-Shot Learners.* NeurIPS, 2020. — arXiv:2005.14165 (§4, 13-gram contamination analysis)
- **[Foundational]** Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA]** Oren, Meister, Chatterji, Ladhak, Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR, 2024. — arXiv:2310.17623
- **[SOTA]** Zhang, da Costa, Marion, et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2405.00332
- **[SOTA]** White et al. *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* ICLR, 2025. — arXiv:2406.19314
- **[Negative result]** Duan, Suri, Mireshghallah, Min, Shi, Zettlemoyer, Tsvetkov, Choi, Evans, Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Negative result]** Das, Zhang, Tramèr. *Blind Baselines Beat Membership Inference Attacks for Foundation Models.* 2024. — arXiv:2406.16201
- **[Method]** Shi, Ajith, Xia, Huang, Liu, Blevins, Chen, Zettlemoyer. *Detecting Pretraining Data from Large Language Models.* ICLR, 2024. — arXiv:2310.16789
- **[Method]** Maini, Jia, Papernot, Dziedzic. *LLM Dataset Inference: Did you train on my dataset?* NeurIPS, 2024. — arXiv:2406.06443
- **[Method]** Golchin, Surdeanu. *Time Travel in LLMs: Tracing Data Contamination in Large Language Models.* ICLR, 2024. — arXiv:2308.08493
- **[Mitigation]** Jacovi, Caciularu, Goldman, Goldberg. *Stop Uploading Test Data in Plain Text.* EMNLP, 2023. — arXiv:2305.10160
- **[Evasion]** Yang, Chiang, Zheng, Gonzalez, Stoica. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023. — arXiv:2311.04850
- **[Survey]** Xu, Wang, Poon, et al. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244
- **[Survey]** Sainz, Campos, García-Ferrero, Etxaniz, de Lacalle, Agirre. *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark.* Findings of EMNLP, 2023. — arXiv:2310.18018

## 10. Worked Example

**Setup.** Auditor holds a private 1,250-item benchmark $B$ (GSM1k's size). Model $M$ scores $A(M, B_{\text{pub}}) = 0.86$ on the public GSM8k-style split and $A(M,B)=0.73$ on the private one. Gap: 13 points. Is that contamination?

**Statistical resolution.** Binomial standard error at $p=0.8$, $n=1250$:
$$\mathrm{se} = \sqrt{\tfrac{0.8 \times 0.2}{1250}} = 0.0113.$$
The two-sample gap has $\mathrm{se}\approx 0.016$, so 13 points is $\approx 8\sigma$. Statistically, the gap is unambiguous.

**Where it breaks.** Zhang et al. (2024) measured this exact comparison across many families. Some models drop ~13 points; Gemini, GPT-4 and Claude families of the time dropped near 0. A model with a 13-point drop is consistent with two hypotheses:

| Hypothesis | Predicted gap |
|---|---|
| $B$ leaked at high duplication | 13 pts |
| $B$ is 13 pts harder than $B_{\text{pub}}$ | 13 pts |

The auditor cannot query $M$ on $B$'s items to disambiguate — doing so hands the private set to the provider. So the available discriminator is the difficulty match, and its residual is unbounded: two independently authored 1,250-item arithmetic sets, matched by human judgment, can differ by several points with no way to bound the difference *a priori*. GSM1k's authors handled this by having the same annotator pool solve both, and still could only argue difficulty parity, not certify it.

**The obstruction, made numeric.** To claim contamination at $\alpha=0.05$, the auditor needs the difficulty-mismatch term bounded below the observed gap. With 1,250 items the *statistical* resolution is 1.6 points; the *difficulty-calibration* resolution — the part that requires knowing $\Delta$, which requires the counterfactual retrain — is unquantified. The binding constraint is not sample size and not detector power. It is that the estimand has no estimator under the available access. That is what "methodologically blocked" means here.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*