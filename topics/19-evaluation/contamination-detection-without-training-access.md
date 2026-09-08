---
id: 19-evaluation/contamination-detection-without-training-access
title: "Benchmark Contamination Detection Without Training Data Access"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Benchmark Contamination Detection Without Training Data Access

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/contamination-detection-without-training-access` · **Status:** open

## 1. Problem Statement

Given a released model $M$ whose training corpus $D$ is undisclosed, and a public benchmark $B = \{(x_i, y_i)\}_{i=1}^n$, decide whether $B$ (or a paraphrase/translation of it) influenced $M$'s training, and — harder — how much of $M$'s reported score on $B$ is attributable to that influence.

Three variants, with sharply different difficulty:

- **Measurement variant.** Define a contamination quantity that is identifiable from black-box access. Currently under-defined: "was this string in $D$" and "does $B$ overstate $M$'s capability" are different quantities and are routinely conflated.
- **Method variant.** Build a test with calibrated false-positive rate under access limited to logprobs, or to sampled text only. This is the empirically active line (Min-K%, exchangeability tests, performance-gap probes).
- **Theory variant.** Establish whether membership of a document in $D$ is even identifiable from $M$'s output distribution at frontier scale ($\ge 10^{13}$ tokens, $\le 2$ epochs, heavy dedup). Non-identifiability results would retire the method line, not just discourage it.

A solution is a procedure returning either a $p$-value against "$B \not\subset D$" with empirically validated Type-I error $\le \alpha$, or a corrected score estimate $\hat{s}_{\text{clean}}$ with coverage guarantees.

## 2. Formal Setting

Let $M$ define $p_\theta(x)$ over token sequences. Access levels: $\mathcal{A}_{\text{logits}}$ (full next-token distributions), $\mathcal{A}_{\text{logprob}}$ (scores of supplied sequences), $\mathcal{A}_{\text{text}}$ (samples only). Most frontier APIs since 2024 supply only $\mathcal{A}_{\text{text}}$ plus top-$k$ logprobs, so any method requiring per-token $\log p$ of an arbitrary string is already inapplicable to the models that matter most.

**Membership predicate.** $Z_i = \mathbb{1}[x_i \in D]$ under some match relation (exact, $n$-gram overlap, embedding-neighborhood). Measured as: a detector statistic $T(x_i)$ thresholded at $\tau$; reported as AUC or TPR at FPR $=0.01$ against a *member/non-member* split whose non-members are drawn post-cutoff.

**Loss-based statistics.** With $\ell(x) = -\log p_\theta(x)$ and $\bar{\ell}$ its per-token mean:
$$T_{\text{Min-K\%}}(x) = -\frac{1}{|S_K|}\sum_{t \in S_K} \log p_\theta(x_t \mid x_{<t}),\quad S_K = \text{$K\%$ lowest-logprob positions}.$$
Min-K%++ normalizes by the token-conditional mean and standard deviation of $\log p_\theta(\cdot \mid x_{<t})$ over the vocabulary, converting the statistic into a local-maximum test.

**Exchangeability statistic (Oren et al.).** If $B$ has a canonical published order and $M$ never saw it, the sequence log-likelihood is exchangeable over permutations $\pi$ of the $n$ examples:
$$\Pr\big[\textstyle\sum_i \log p_\theta(x_{\pi(i)}) > \sum_i \log p_\theta(x_i)\big] = 1/2 .$$
Measured as a permutation $p$-value over $m$ random shard permutations. This is the only test in the literature with a *valid* null derived from a stated assumption rather than a calibration set.

**Utility-side quantity.** $\Delta = s_B(M) - s_{B'}(M)$, the score gap between $B$ and a freshly constructed distribution-matched twin $B'$. Measured directly; requires paying for $B'$.

**Assumptions and their violations.**
- *Member/non-member exchangeability.* Violated: post-cutoff non-members differ in topic, style and vocabulary, so detectors learn the date, not the membership (Duan et al. 2024; Das et al. 2024).
- *Benchmark order is canonical and unshuffled in $D$.* Violated whenever the benchmark reached the corpus via a shuffled mirror, a leaderboard scrape, or a reformatted copy.
- *Contamination is verbatim.* Violated by rephrased, translated and synthetic-restatement contamination (Yang et al. 2023).
- *$B'$ is distribution-matched.* Never verifiable; item-writer effects confound $\Delta$.

## 3. State of the Art

**Established.**
- *Exchangeability/sharded likelihood test* (Oren, Meister, Chatterji, Ladhak, Hashimoto, ICLR 2024): provably calibrated under the stated null, and empirically detects an injected test set duplicated a small number of times in a 1.4B-parameter pretraining run. It is a test of *ordering* memorization, not of capability inflation.
- *Negative result on loss-based MIA at LLM scale* (Duan et al., COLM 2024): across Pythia and OLMo models to 12B on the Pile, loss, zlib, neighborhood and Min-K% attacks sit near chance, AUC $\approx 0.5$–$0.55$, once candidate sets are temporally and distributionally matched.
- *Blind-baseline critique* (Das, Zhang, Tramèr, 2024): on WikiMIA and similar benchmarks, a classifier with no model access at all — bag-of-words or date heuristics on the text — matches or beats published MIAs, which means the reported AUCs measure split construction, not membership.

**Claimed but unablated.**
- Min-K% (Shi et al., ICLR 2024) and Min-K%++ (Zhang et al., ICLR 2025) report AUC $\approx 0.70$–$0.85$ on WikiMIA. These are benchmark numbers on a split now known to be confounded; the methods have not been shown to hold up on a matched split.
- ReCaLL and other conditional-likelihood detectors report large gains on the same confounded suites.
- Prompt-based probes — "Time Travel in LLMs" (Golchin & Surdeanu, ICLR 2024), TS-Guessing (Deng et al., NAACL 2024) — report contamination on GPT-family models via completion of masked benchmark fields. Positive rates exist only as benchmark numbers with no calibrated null.

**Utility-side SOTA.** GSM1k (Zhang et al., NeurIPS 2024) and live/rolling benchmarks (LiveCodeBench, Jain et al., ICLR 2025; LiveBench, White et al., ICLR 2025) sidestep detection by rebuilding the test set. This is the only approach with a defensible measurement, and it costs a benchmark per model release.

## 4. What Is Known

- Verbatim memorization grows log-linearly in model size, duplication count and prompt-prefix length; Carlini et al. (ICLR 2023) measure this on GPT-Neo up to 6B, where a sequence duplicated $\sim 10^2$ times is extractable at rates orders of magnitude above singletons.
- Detection degrades with epochs: at $\le 1$–$2$ epochs and after aggressive dedup — the frontier regime — loss-based signal is at or near chance (Duan et al. 2024, up to 12B).
- Capability inflation is real and unevenly distributed: on GSM1k, held-out twin of GSM8K, gaps reach up to $\approx 13$ accuracy points for some Mistral and Phi checkpoints, while frontier models (GPT-4-class, Claude, Gemini) show near-zero or negative gaps (Zhang et al. 2024, $n=1250$ new items).
- Rephrased contamination defeats $n$-gram decontamination: 13-gram overlap filters pass paraphrased test items that still lift benchmark scores substantially (Yang et al. 2023, Llama-2 scale).
- Longitudinal evidence links pre-cutoff online popularity of a problem to model accuracy on it (Roberts et al., ICLR 2024, Codeforces/Project Euler), an association, not an identified effect.

## 5. What Is Not Known

- **Theoretically open.** Whether single-epoch membership is identifiable at all from $p_\theta$ at $10^{13}$-token scale. No lower bound exists showing the signal must vanish, and no upper bound showing it must persist. A stability/differential-privacy-style argument is plausible but unwritten.
- **Methodologically blocked.** The mapping from membership to score inflation. Nobody has defined a black-box-estimable quantity for "how many of $M$'s $B$-points come from having seen $B$." Every current statistic measures memorization traces; the number practitioners want is causal.
- **Empirically open.** Whether Min-K%-family detectors retain any signal on a properly matched split. The experiment is runnable today with open-weight models on known corpora; it has not been run at 70B+ with matched candidates.
- **Empirically open.** Whether contamination via RLHF/instruction data or synthetic distillation from a contaminated teacher leaves any black-box trace at all.

## 6. Why It Is Hard

Two obstructions, both structural.

1. **Non-identifiability under confounding.** The observable is $p_\theta(x)$. Low loss on $x$ has at least three sufficient causes: membership, near-duplicate membership, and genuine generalization from the same distribution. At single-epoch frontier scale the membership contribution is of the same order as the generalization contribution, so no threshold on $\ell(x)$ separates them. This is why matched-split evaluations collapse to AUC $\approx 0.5$.
2. **Absent ground truth for validation.** Validating a detector requires knowing $D$. For models where $D$ is known (Pythia, OLMo), the detector is unnecessary; for models where it is needed, no labels exist. The field substitutes proxy splits (WikiMIA), which encode a date shortcut — hence blind baselines winning. Every reported AUC on a proxy split is an evaluation that does not measure the thing it names.

## 7. Current Research (as of 2026)

- **Rolling and private benchmarks** as the pragmatic answer: LiveBench, LiveCodeBench, SWE-bench-style rebuilt splits, held-out twins. Well established.
- **Calibrated distribution-level tests** — dataset inference (Maini et al., NeurIPS 2024) aggregates weak per-example signals over a set, which is statistically the right move given per-example non-identifiability. Extending it to benchmark-sized $n \approx 10^3$ with only $\mathcal{A}_{\text{text}}$ is open. *(frontier — verify)*
- **Methodological hygiene** — SoK by Meeus, Shilov, Jain, Cretu, Cummings, de Montjoye (SaTML 2025) specifies randomized member/non-member splits from a single corpus as the minimum standard; adoption is partial. *(frontier — verify)*
- **Canary/watermark provenance**: benchmark authors embedding detectable canaries at publication (BIG-bench's canary string is the precedent). Prospective only — useless for existing benchmarks.

## 8. Concrete Next Experiment

**Question.** Does any black-box detector beat a blind baseline on membership when the split is randomized rather than temporal, at frontier-relevant scale?

**Setup.** Use OLMo 2 (7B and 13B) with its fully public Dolma-derived corpus. Sample $10{,}000$ documents; randomly assign half to a held-out shard *before* pretraining a matched pair of models — or, cheaper, use the existing released models and construct members/non-members by random split of a single homogeneous source (one Common Crawl dump, one arXiv month) with exact-dedup verification against $D$. Insert five benchmark-shaped corpora (1,000 items each) at duplication counts $\{1, 2, 4, 16, 64\}$ in a continued-pretraining run of $2\times10^{10}$ tokens.

**Arms.** (a) Min-K%++; (b) Oren exchangeability test; (c) dataset inference over the 1,000-item set; (d) **control arm: blind bag-of-words classifier with no model access**, trained on the same split.

**Deciding number.** TPR at FPR $=0.01$ for each arm at duplication count 1. If no model-access arm exceeds the blind control by $\ge 10$ percentage points at duplication 1 and 2, single-exposure contamination detection without training-data access is empirically dead, and the field should redirect entirely to rolling benchmarks and held-out twins. If the exchangeability test alone clears it, contamination detection is a test of *ordering* provenance, and benchmark publishers should be told to fix and publish a canonical order.

## 9. Key References

- **[Foundational]** N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramèr, C. Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA]** Y. Oren, N. Meister, N. Chatterji, F. Ladhak, T. B. Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR, 2024. — arXiv:2310.17623
- **[SOTA]** W. Shi, A. Ajith, M. Xia, Y. Huang, D. Liu, T. Blevins, D. Chen, L. Zettlemoyer. *Detecting Pretraining Data from Large Language Models.* ICLR, 2024. — arXiv:2310.16789
- **[SOTA]** J. Zhang, J. Sun, E. Yeats, Y. Ouyang, M. Kuo, J. Zhang, H. Yang, H. Li. *Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models.* ICLR, 2025. — arXiv:2404.02936
- **[Negative result]** M. Duan, A. Suri, N. Mireshghallah, S. Min, W. Shi, L. Zettlemoyer, Y. Tsvetkov, Y. Choi, D. Evans, H. Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Critique]** D. Das, J. Zhang, F. Tramèr. *Blind Baselines Beat Membership Inference Attacks for Foundation Models.* 2024. — arXiv:2406.16201
- **[Methodology]** M. Meeus, I. Shilov, S. Jain, M. Faysse, M. Rei, Y.-A. de Montjoye. *SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It).* IEEE SaTML, 2025.
- **[Utility-side]** H. Zhang, J. Da, D. Lee, V. Robinson, C. Wu, W. Song, T. Zhao, P. Raja, C. Zhuang, D. Slack, et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2405.00332
- **[Rephrased contamination]** S. Yang, W.-L. Chiang, L. Zheng, J. E. Gonzalez, I. Stoica. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023. — arXiv:2311.04850
- **[Aggregate test]** P. Maini, H. Jia, N. Papernot, A. Dziedzic. *LLM Dataset Inference: Did You Train on My Dataset?* NeurIPS, 2024. — arXiv:2406.06443
- **[Rolling benchmarks]** N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, I. Stoica. *LiveCodeBench: Holistic and Contamination-Free Evaluation of Large Language Models for Code.* ICLR, 2025. — arXiv:2403.07974
- **[Survey]** C. Xu, S. Guan, D. Greene, M.-T. Kechadi. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244

## 10. Worked Example

Take MMLU ($n \approx 14{,}000$ four-way items) and a released 70B model with logprob access.

**Loss-based route.** Score each item's question+answer string and compute Min-K% with $K=20$. Members are undefined — so build a proxy: MMLU items versus 14,000 items from a post-cutoff exam bank. Suppose Min-K% yields AUC $=0.78$. Now run the control: a logistic classifier on TF-IDF features of the *raw text only*, no model. On such splits this blind baseline routinely reaches AUC $\approx 0.75$–$0.9$ (Das et al. 2024). The 0.78 is consistent with a detector that has learned only "MMLU items are written in exam-bank style circa 2020." The obstruction is visible: the AUC is a property of the split, not of $\theta$.

**Exchangeability route.** MMLU has a canonical file order. Shard into 20 blocks of 700, compute the total logprob of the canonical order versus 1,000 random permutations. Suppose the canonical order lands at rank 340/1000, $p = 0.34$. This is a negative — but its power is unknown here. Oren et al.'s test detected an injected set at low duplication in a 1.4B run; nobody has characterized its power for a set that entered a $10^{13}$-token corpus once, through a shuffled HuggingFace mirror that destroyed the order the test relies on. A null result is uninformative because the assumption may simply be false.

**Utility route.** Commission 500 new MMLU-style items under the original writing guidelines. Observe $s_{\text{MMLU}} = 0.86$, $s_{\text{new}} = 0.81$. The 5-point gap is measurable, but item difficulty was never matched — two independent item-writing teams routinely differ by that much. To attribute the gap to contamination you need a second model, known-clean on both sets, to calibrate writer difficulty; that model does not exist for frontier-level content.

All three routes terminate in the same place: a number that is real, and an attribution that is not identified.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*