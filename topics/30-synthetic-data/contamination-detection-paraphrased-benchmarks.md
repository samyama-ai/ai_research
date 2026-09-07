---
id: 30-synthetic-data/contamination-detection-paraphrased-benchmarks
title: "Contamination Detection for Synthetically Paraphrased Benchmarks"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Contamination Detection for Synthetically Paraphrased Benchmarks

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/contamination-detection-paraphrased-benchmarks` · **Status:** open

## 1. Problem Statement

A benchmark item can enter a pretraining corpus in disguise. An LLM rewrites the question, swaps names and numbers, translates it to another language, or restructures the multiple-choice options; the surface form shares no long $n$-gram with the original, but the model that trains on it gains the benchmark's answer. Surface decontamination (13-gram / 50-char substring filters, as used for GPT-3, Llama, and most open corpora) does not remove these items.

**Input.** A benchmark $D = \{x_i\}_{i=1}^n$, a model $M$ with weights or at least log-probabilities, and optionally a candidate corpus $C$.

**Output.** A decision or a calibrated $p$-value for: *did $M$'s training data contain a semantic paraphrase of $D$?*

Three variants, with different difficulty:

- **Measurement.** Define "paraphrase contamination" so that two labs computing it on the same $(M, D)$ agree. Currently unsolved: there is no agreed equivalence relation on items, so the ground-truth label the detector is scored against is itself contested.
- **Method.** Given a fixed definition, build a detector with high TPR at FPR $\le 1\%$. Runnable today; nobody has a detector that survives an adaptive paraphraser.
- **Theory.** Is paraphrase contamination *identifiable* from black-box access — i.e., do there exist $(M_{\text{clean}}, M_{\text{contaminated}})$ pairs that are indistinguishable under any query budget while differing in benchmark accuracy? Open.

Solving it means: a test that, at declared FPR $1\%$, detects contamination injected by an adversary who is allowed to see the detector, at injection rates low enough to move a leaderboard ($\ge 5$ accuracy points).

## 2. Formal Setting

Let $\mathcal{X}$ be the space of benchmark items. A **paraphrase channel** is a conditional distribution $T(\tilde{x} \mid x)$ realized by an LLM rewriter with prompt $\pi$ and temperature $\tau$. Define the semantic class $[x]_T = \{\tilde{x} : T(\tilde{x}\mid x) > 0\}$. Contamination at level $T$:

$$ \mathrm{Cont}_T(M, D) = \mathbb{1}\left[\, \exists\, i,\ \mathcal{D}_{\text{train}}(M) \cap [x_i]_T \neq \emptyset \,\right]. $$

**As measured**, each quantity is a proxy:

- $\mathcal{D}_{\text{train}}(M)$ is never observable for frontier models. Measured instead by injection: fine-tune or continue-pretraining a known-clean base on $k$ paraphrased copies, giving a *constructed* ground truth.
- $[x]_T$ is measured by an embedding threshold, $[x]_\theta = \{\tilde{x} : \cos(e(x), e(\tilde{x})) \ge \theta\}$, or by an LLM judge $J(x, \tilde{x}) \in \{0,1\}$. Neither is transitive, so $[\cdot]$ is not an equivalence class — a chain of ten $\theta = 0.85$ rewrites reaches an item the judge calls unrelated.
- The detector is a score $s(x, M) \in \mathbb{R}$ with decision $s > \gamma$. Report **TPR at FPR $= 1\%$**, not AUC; AUC hides the low-FPR regime that matters when screening $10^4$ items.
- Likelihood-based scores: $\mathrm{Min\text{-}K\%}(x) = \frac{1}{|K|}\sum_{t \in K} \log p(x_t \mid x_{<t})$ over the $K\%$ lowest-probability tokens (Shi et al., ICLR 2024); the neighbourhood score $s_{\text{nbr}}(x) = \log p(x) - \frac{1}{m}\sum_{j} \log p(x^{(j)})$ over $m$ perturbations (Mattern et al., ACL Findings 2023).
- Performance-based scores: the **reasoning gap** $\Delta = \mathrm{Acc}(M, D) - \mathrm{Acc}(M, T(D))$, measured with matched difficulty and paired bootstrap CIs.
- Order-based: Oren et al. (ICLR 2024) test exchangeability, $H_0$: $\log p_M$ is invariant to permutation of $D$'s canonical order, giving an exact $p$-value by permutation.

**Assumptions, and which fail.**
1. *Benchmark items are exchangeable under $H_0$* — holds for shuffled test sets, **fails** when items are curriculum-ordered or difficulty-sorted.
2. *A clean reference model exists* — **fails** for post-2023 models; every plausible reference has read the web.
3. *The paraphrase preserves difficulty* — **fails** measurably; GSM1k (Zhang et al., NeurIPS 2024) needed human authoring plus difficulty matching precisely because naive rewrites shift solve rates.
4. *Members and non-members are drawn i.i.d. from one distribution* — **fails** in nearly all MIA benchmarks; Duan et al. (COLM 2024) show apparent MIA success on WikiMIA is largely temporal distribution shift.

## 3. State of the Art

**Established.**
- **Surface decontamination fails on rewrites.** Yang et al. (arXiv:2311.04850) fine-tuned a 13B Llama-2 on rephrased MMLU/GSM8K/HumanEval test sets; it reached GPT-4-level scores on those benchmarks while passing $n$-gram and standard embedding decontamination. Their LLM-judge "decontaminator" catches these rewrites in-distribution — this is the strongest positive result and it is **claimed but only lightly ablated** against an adaptive rewriter.
- **Exchangeability test gives real $p$-values.** Oren et al. (ICLR 2024) prove a valid test under $H_0$ (no dependence on item order), detecting contamination on 1.4B models trained with test sets duplicated as few as 2 times, and on 7B open models. It detects *verbatim ordered* inclusion; a paraphrased corpus in shuffled order breaks the premise.
- **MIA on pretraining scale is near chance.** Duan et al. (COLM 2024): AUC $\approx 0.5$–$0.55$ for Loss, Ref, Min-K% on Pythia 160M–12B against the Pile, once member/non-member sets are temporally matched.

**Claimed, unablated, or benchmark-number-only.**
- Min-K%++ (Zhang et al., ICLR 2025) reports large AUC gains on WikiMIA (up to ~$+10$ points over Min-K%); WikiMIA's shift confound means the number does not transfer to paraphrase detection.
- ConStat (Dekoninck et al., NeurIPS 2024) tests whether a model's benchmark performance is anomalous relative to a reference-model population and a harder reference benchmark; reported detections on public models are **benchmark numbers without injected ground truth** at frontier scale.
- Golchin & Surdeanu's "Time Travel in LLMs" (ICLR 2024) guided-instruction test and the Data Contamination Quiz report high accuracy for GPT-4 on verbatim contamination; neither is evaluated against a rewriter tuned to evade them.

There is **no** published detector with a declared FPR that has been run against an adversary optimizing paraphrases to evade it.

## 4. What Is Known

- 13B model + rephrased test sets $\to$ GPT-4-level MMLU/GSM8K/HumanEval scores, undetected by $n$-gram filters (Yang et al., 2023, 13B scale).
- GSM1k vs GSM8K: up to **13 accuracy points** drop for the worst model families, near-zero for others; drop correlates with the probability the model emits GSM8K examples verbatim (Zhang et al., NeurIPS 2024; models from ~7B to frontier API scale).
- Functional-variant "reasoning gap" of **58–80%** relative on MATH for several models when items are re-instantiated with new numbers (Srivastava et al., 2024).
- Memorization scales log-linearly in model size, duplication count, and prompt-context length (Carlini et al., ICLR 2023, 345M–12B GPT-Neo).
- Exchangeability test detects duplication factor 2 at 1.4B (Oren et al., ICLR 2024).
- Repeated-exposure contamination is detectable through loss on the *original* item even when only paraphrases were trained on — but the effect size shrinks with paraphrase distance; no cross-lab reproduction of a quantitative decay curve exists.

## 5. What Is Not Known

- **Methodologically blocked.** The equivalence relation defining "same item". Every reported TPR is relative to one rewriter and one judge. Until $[x]_T$ is fixed by an operational protocol (a published rewriter checkpoint, prompt, and judge), detector numbers are not comparable across papers. This is the binding blocker.
- **Theoretically open.** Identifiability. No proof exists that paraphrase contamination is distinguishable from legitimate training on same-domain data by *any* black-box test. Nor is there an impossibility result. The natural conjecture — that for any detector there is a paraphrase channel $T$ preserving answer-transfer while driving TPR to the FPR level — is unproven in both directions.
- **Empirically open.** The dose–response curve: TPR at FPR $1\%$ as a function of (paraphrase distance, duplication count $k$, model scale) for a model *pretrained*, not fine-tuned, on injected paraphrases at 7B+ with $\ge 100$B tokens. Fine-tuning injection is a poor proxy: it concentrates gradient on the injected set in a way pretraining does not.
- **Empirically open.** Whether performance-based tests (ConStat, GSM1k-style) and likelihood-based tests (Min-K%++) are correlated or complementary on the same injected ground truth.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth at the scale that matters.** Contamination is a property of a training set nobody publishes. Detectors are validated on injected small models; the deployment target is a frontier model whose corpus is secret. The validation distribution and the deployment distribution differ in exactly the variable being estimated.
2. **Confounded measurement.** The two obvious signals — low loss on the item and high accuracy on the benchmark — are also what a genuinely capable, honestly-trained model produces. Duan et al. show the standard MIA benchmarks conflate membership with temporal/topical shift; the same confound reappears whenever a "clean" control benchmark is newer than the contaminated one.
3. **Non-identifiability under an adaptive channel.** The rewriter is an LLM and can be optimized against a published detector at low cost. Detection is a game with a moving defender score; every published number is a static-adversary number.

Compute is *not* the primary obstruction: the decisive injection experiment is a few thousand GPU-hours.

## 7. Current Research (as of 2026)

- **Held-out and rolling benchmarks.** LiveBench (White et al., ICLR 2025) refreshes items monthly from post-cutoff sources; sidesteps detection rather than solving it. Jacovi et al. (EMNLP 2023) push encrypted test sets and canary strings — adopted by BIG-bench and a few successors, but not by most benchmark releases.
- **Population-based statistical tests.** ConStat and dataset-inference approaches (Maini et al., NeurIPS 2024) move from per-example membership (near-chance) to per-*dataset* aggregation, where signal aggregates over $n$ items. This is currently the most promising direction *(frontier — verify: whether dataset inference survives paraphrase-only contamination has not been reported)*.
- **Adversarial paraphrase benchmarks.** Several groups are constructing injection testbeds with graded paraphrase distance; no standard has emerged *(frontier — verify)*.
- **Surveys.** Xu et al., *Benchmark Data Contamination of Large Language Models: A Survey* (2024), catalogs ~50 detection methods and notes that almost none report FPR.

## 8. Concrete Next Experiment

**Question.** At what paraphrase distance does the best available detector fall to chance, and does dataset-level aggregation extend that range?

**Scale.** Continue-pretrain Pythia-1.4B and Pythia-6.9B on 10B fresh Pile tokens. Into that stream inject paraphrases of MMLU-test and GSM8K-test at duplication $k \in \{1, 4, 16\}$, at five paraphrase levels: L0 verbatim; L1 lexical swap; L2 sentence-level LLM rewrite; L3 rewrite + numeric/entity resubstitution; L4 round-trip translation through two languages plus rewrite. That is $2 \times 3 \times 5 = 30$ runs plus controls; roughly 3–5k A100-hours.

**Control arm.** Identical runs with paraphrases of a *held-out* benchmark of matched domain and difficulty injected instead — so the model is equally exposed to same-topic synthetic text but not to the scored items. This isolates contamination from domain-familiarity, the confound in §6.2.

**Detectors scored.** Min-K%++, neighbourhood attack, Oren exchangeability test, ConStat, and the accuracy gap $\Delta$ on functional variants. Each calibrated on the control arm to FPR $= 1\%$.

**Deciding number.** **TPR at FPR $1\%$ at L3, $k = 4$, 6.9B.** If the best detector exceeds $0.8$, paraphrase contamination is practically detectable and the field should standardize on that detector. If it is below $0.2$ while the benchmark accuracy gain from injection exceeds $5$ points, then contamination that moves leaderboards is undetectable by current means, and the correct response is held-out/rolling benchmarks, not better detectors. Report the accuracy gain alongside; a detector that only fires when contamination does not help is useless.

## 9. Key References

- **[Foundational]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Foundational]** Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, Eneko Agirre. *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark.* Findings of EMNLP, 2023. — arXiv:2310.18018
- **[SOTA]** Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023. — arXiv:2311.04850
- **[SOTA]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori B. Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR, 2024. — arXiv:2310.17623
- **[SOTA]** Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer. *Detecting Pretraining Data from Large Language Models.* ICLR, 2024. — arXiv:2310.16789
- **[SOTA]** Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Frank Yang, Hai Li. *Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models.* ICLR, 2025. — arXiv:2404.02936
- **[SOTA]** Jasper Dekoninck, Mark Niklas Müller, Martin Vechev. *ConStat: Performance-Based Contamination Detection in Large Language Models.* NeurIPS, 2024. — arXiv:2405.16281
- **[Negative result]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Empirical]** Hugh Zhang, Jeff Da, Dean Lee, et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS, 2024. — arXiv:2405.00332
- **[Method]** Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schölkopf, Mrinmaya Sachan, Taylor Berg-Kirkpatrick. *Membership Inference Attacks against Language Models via Neighbourhood Comparison.* Findings of ACL, 2023. — arXiv:2305.18462
- **[Mitigation]** Alon Jacovi, Avi Caciularu, Omer Goldman, Yoav Goldberg. *Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks.* EMNLP, 2023. — arXiv:2305.10160
- **[Survey]** Cheng Xu, Shuhao Guan, Derek Greene, M-Tahar Kechadi. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244

## 10. Worked Example

Take one GSM8K test item:

> *Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?* (answer 72)

L3 paraphrase, produced by a standard rewrite prompt:

> *In March, Priya distributed hair ties to 62 classmates. The following month she handed out only half that number. What is Priya's two-month total?* (answer 93)

Check the surface filters. Longest shared 13-gram: none. Longest shared character substring: `" half as many"` is gone; the longest common substring is `" in "` — 4 characters, far under the 50-character GPT-3 threshold. Token-level Jaccard over content words: shared set is $\{\text{month}\}$ against a union of ~24, so $J \approx 0.04$. Every deployed decontamination pipeline passes this item.

Now the detection side. Suppose a 6.9B model saw 500 such rewrites, one per GSM8K item, four times each — 2M tokens inside a 10B-token continue-pretrain, i.e. 0.02% of the stream.

- **Min-K% on the original item.** The original was never in training. Its low-probability tokens are `48`, `Natalia`, `72` — none of which appear in the paraphrase. The score moves by roughly the amount that generic grade-school-arithmetic exposure moves it, which the control arm also produces. Expected separation: small, and the control arm subtracts it.
- **Exchangeability test.** Requires that the model's log-likelihood depends on the canonical *order* of the test set. The rewriter emitted items independently; the injected stream was shuffled. $H_0$ holds by construction, so the test returns a uniform $p$-value. It is not a weak detector here — it is a correctly-calibrated detector of a different thing.
- **Accuracy gap $\Delta$.** The model learned the *solution schema* $n + n/2$, not the string. It now answers both the original and a fresh functional variant, so $\Delta \approx 0$ — and its GSM8K score is up several points.

That is the obstruction in one item. The signal the surface filters look for (string overlap) is zero; the signal the likelihood detectors look for (memorized tokens) is zero because the memorized object is a schema; and the signal the performance detectors look for (a gap between original and variant) is zero because schema learning generalizes. What remains — "the model is unusually good at this benchmark" — is indistinguishable from the model being good, which is the non-identifiability in §6.3 made concrete.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*