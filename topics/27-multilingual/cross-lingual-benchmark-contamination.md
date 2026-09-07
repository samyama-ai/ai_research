---
id: 27-multilingual/cross-lingual-benchmark-contamination
title: "Quantifying Cross-Lingual Contamination in Multilingual Corpora"
topic: 27-multilingual
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Quantifying Cross-Lingual Contamination in Multilingual Corpora

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/cross-lingual-benchmark-contamination` · **Status:** methodologically-blocked

## 1. Problem Statement

Nearly every multilingual benchmark is a **translation** of an English source: XNLI translates MultiNLI, XQuAD and MLQA translate SQuAD, MGSM translates GSM8K, XCOPA translates COPA, Global-MMLU translates MMLU. A model that memorised the English source has, in a semantic sense, seen the test set in all 100 target languages — but no substring of the target-language item appears in the pretraining corpus. String-matching decontamination reports zero overlap and is not wrong; it is answering a different question.

Three variants, of very different difficulty:

- **Measurement.** Given a pretraining corpus $D$, a benchmark $B_\ell$ in language $\ell$, decide for each item whether a *translation-equivalent* of it occurs in $D$ in any language. Output: a per-item binary label and a corpus-level rate.
- **Method.** Given only black-box or logit access to a trained model (corpus unavailable — true for GPT-4/5-class, Gemini, Claude), test the hypothesis "this benchmark's source items were in training" with calibrated false-positive control.
- **Theory.** Bound the inflation in reported accuracy attributable to cross-lingual contamination, separating it from legitimate cross-lingual transfer. This is the variant that is not merely unrun but ill-posed: transfer and contamination are the same computational phenomenon viewed from different sides.

Solving it means: a decontamination procedure that changes reported multilingual scores by a measurable, reproducible amount, and a stated identification argument for why the change is contamination rather than transfer.

## 2. Formal Setting

Let $D=\{d_1,\dots,d_N\}$ be a pretraining corpus, $\mathcal{L}$ the language set, and $B_\ell=\{(x_i^\ell,y_i)\}_{i=1}^{n}$ a benchmark whose items are translations of a common source $\{x_i^{\mathrm{src}}\}$ (usually $\mathrm{src}=\mathrm{en}$).

**Contamination indicator.** Fix a cross-lingual similarity $\mathrm{sim}:\mathcal{X}\times\mathcal{X}\to[0,1]$ and threshold $\theta$:

$$
c_i(\theta) \;=\; \mathbb{1}\!\left[\;\exists\, d\in D,\ \exists\, \ell'\in\mathcal{L}:\ \mathrm{sim}\!\left(d,\ x_i^{\ell'}\right)\ \ge\ \theta\right],
\qquad
\rho(\theta)=\frac{1}{n}\sum_{i=1}^{n} c_i(\theta).
$$

**As measured.** $\mathrm{sim}$ is not free. Three implementable choices, each with a distinct failure mode:
1. *String*: max $n$-gram Jaccard over $n=8{-}13$. Cross-lingually $\mathrm{sim}\approx 0$ by construction — this is the defect, not a property of the data.
2. *Translate-then-match*: apply an MT system $\tau_{\ell'\to\mathrm{en}}$ to every document, then $n$-gram match. Cost is $O(|D|)$ MT forward passes; for $|D|=10^{12}$ tokens this is comparable to pretraining itself.
3. *Embedding*: $\mathrm{sim}(d,x)=\cos(f(d),f(x))$ with a multilingual sentence encoder $f$ (LaBSE, SONAR). $\theta$ has no calibration that is stable across language pairs — LaBSE cosine for true bitext varies by 0.1–0.2 between high- and low-resource pairs, so a single $\theta$ trades false positives in one language against false negatives in another.

**Effect estimand.** The quantity of interest is not $\rho$ but the score inflation

$$
\Delta_\ell \;=\; \mathbb{E}\!\left[s(x_i^\ell)\mid c_i=1\right]-\mathbb{E}\!\left[s(x_i^\ell)\mid c_i=0\right],
$$

with $s$ the per-item score. $\Delta_\ell$ is a difference of conditional means, not a causal effect: items that appear on the web are systematically easier, shorter, and more templatic.

**Assumptions, and which fail.**
- *(A1) Benchmark items are i.i.d. draws from the task distribution.* Violated: translated benchmarks are the same $n$ source items permuted across $|\mathcal{L}|$ languages, so item effects are perfectly correlated across languages and per-language errors are not independent.
- *(A2) A corpus is monolingual if labelled so.* Violated: language-ID filtering leaks. Blevins & Zettlemoyer (EMNLP 2022) show "English" corpora carry non-trivial fractions of other languages.
- *(A3) $\mathrm{sim}$ is language-pair invariant.* Violated by encoder resource bias (above).
- *(A4) Contamination is monotone in $\theta$ and $\rho$ is the right summary.* Partially violated: paraphrase and translation contamination produce a heavy tail near threshold, so $\rho(\theta)$ is unstable in $\theta$ exactly where the interesting items are.

## 3. State of the Art

**Established (monolingual, English).**
- Oren et al., *Proving Test Set Contamination in Black Box Language Models* (ICLR 2024): an exchangeability test — if a model assigns higher log-likelihood to the benchmark's canonical item ordering than to random permutations, contamination is present. Gives a valid $p$-value under a stated null with no corpus access. This is the only method in the area with a proof.
- Carlini et al., *Quantifying Memorization Across Neural Language Models* (ICLR 2023): extractable memorisation grows log-linearly in model size, in example duplication count, and in prompt-context length.
- Min-K% Prob (Shi et al., ICLR 2024) for pretraining-data detection.

**Claimed but unablated / benchmark-number-only.**
- Golchin & Surdeanu's *Time Travel in LLMs* (ICLR 2024) guided-instruction completion, and Deng et al.'s TS-Guessing (NAACL 2024), report contamination rates on specific benchmarks. Neither has a false-positive rate measured against a model with a known-clean corpus; the numbers are point estimates without a null.
- Yang et al. (2023) show rephrased/translated test samples evade $n$-gram and embedding decontamination and still inflate scores — the closest existing result to this problem, but demonstrated by *fine-tuning* on rewritten test data, not observed in a natural corpus.
- Cross-lingual specifically: essentially nothing established. Multilingual evaluation papers (MEGA, EMNLP 2023; Global-MMLU, ACL 2025) note the translation-source risk in discussion sections and do not measure it.

**Systems SOTA for decontamination in practice** remains 13-gram or 50-character substring matching against the eval suite, monolingual, as used in the GPT-3, PaLM, Llama and OLMo reports. No released frontier model documents a cross-lingual decontamination pass.

## 4. What Is Known

- **Benchmark provenance is documented and narrow.** XNLI: 5,010 test + 2,490 dev items, translated from MultiNLI into 15 languages (Conneau et al., EMNLP 2018). XQuAD: 1,190 SQuAD-v1.1 dev QA pairs in 11 languages. MGSM: 250 GSM8K problems in 10 languages. XCOPA: 600 items in 11 languages. Belebele: 900 questions across 122 language variants. So a single English source set of order $10^3$ items determines the entire multilingual evaluation surface — the contaminating object is tiny and heavily duplicated on the web.
- **Language-ID leakage is measured.** Blevins & Zettlemoyer (EMNLP 2022) find non-English text in nominally English corpora at rates sufficient to explain a large share of "zero-shot" cross-lingual transfer in English-pretrained models; removing it degrades transfer.
- **Web-crawled multilingual corpora are dirty at audited scale.** Kreutzer et al. (TACL 2022) hand-audited 5 corpora across 200+ languages: for many low-resource languages under 50% of sentences were in the labelled language, and several CCAligned language pairs were 0% correct.
- **Duplication drives memorisation.** Carlini et al. (ICLR 2023): memorisation rises roughly log-linearly with duplicate count; items appearing $\ge 10^2$ times are extractable at materially higher rates than singletons.
- **Membership inference is weak at pretraining scale.** Duan et al. (COLM 2024) find MIA against LLMs performs near chance (AUC $\approx 0.5$–$0.55$) on non-duplicated pretraining data — so per-item contamination labels from MIA are not currently trustworthy.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition of "the same test item in another language". Translation equivalence is a graded, encoder-dependent judgement; every threshold choice is a free parameter that moves $\rho$ by tens of points. Until $\mathrm{sim}$ and $\theta$ are pinned by something external — human bitext judgements, or a synthetic corpus with ground-truth insertion — $\rho$ is not a measurement.
- **Methodologically blocked (secondary).** Contamination and cross-lingual transfer are not identified from observational data. Both predict "higher score on items whose English source is well represented in the corpus". No observational design separates them.
- **Empirically open.** The controlled-injection experiment (§8) is runnable today at 1–3B parameters for well under $10^5$ GPU-hours and has not been run. Nobody has measured how translated-duplicate count in a pretraining corpus maps to accuracy gain on the target-language benchmark.
- **Theoretically open.** Whether Oren et al.'s exchangeability test extends to translated benchmarks. The null there is invariance of log-likelihood to item permutation; under translation the model never saw the target-language strings, so the test statistic's power against a "source-language contamination, target-language evaluation" alternative is unproven — plausibly zero.

## 6. Why It Is Hard

**Non-identifiability, with a compute cost on top of it.**

The obstruction is that the treatment and the capability are the same object. "Model saw an English document semantically equivalent to test item $i$, and answers $i$ correctly in Swahili" is *both* the definition of cross-lingual contamination *and* the definition of successful cross-lingual transfer, which is the property multilingual benchmarks exist to measure. Any observational estimator of $\Delta_\ell$ is confounded by item difficulty, source-item web frequency, and target-language resource level — three variables that are themselves mutually correlated.

The secondary obstruction is cost: the only design that breaks the confound is intervention on the corpus, which requires pretraining from scratch. And exhaustive cross-lingual search over a $10^{12}$-token corpus needs either MT over the whole corpus or a dense index of $\sim10^{10}$ multilingual embeddings — both within a small constant factor of pretraining compute, which is why no lab has published one.

## 7. Current Research (as of 2026)

- **Dynamic and post-cutoff benchmarks.** LiveBench-style rolling evaluation, and multilingual variants built from post-training-cutoff native text rather than translation, sidestep rather than measure contamination. Cohere Labs' Global-MMLU line (culturally-sensitive vs culturally-agnostic item split) is the closest to a native-item multilingual suite. *(frontier — verify: several native-source multilingual suites announced in 2025–26; check item provenance before treating them as clean.)*
- **Corpus-transparency efforts** — AI2's OLMo/Dolma, HuggingFace FineWeb-2 (multilingual) — make the corpus-side search feasible in principle for open models. FineWeb-2 covers 1,000+ languages; nobody has published a cross-lingual eval-overlap audit of it. *(frontier — verify.)*
- **Contamination detection without corpus access:** extensions of exchangeability and Min-K% testing; survey coverage in Xu et al., *Benchmark Data Contamination of Large Language Models: A Survey* (2024).
- **Bitext mining infrastructure** (LASER/SONAR, NLLB) is the natural $\mathrm{sim}$ operator and is already deployed at web scale for mining — the tooling gap is smaller than the incentive gap.

## 8. Concrete Next Experiment

**Controlled cross-lingual injection.**

- **Scale.** Three 1.4B-parameter decoder models, identical architecture and seed, each trained on 300B tokens of a fixed multilingual corpus (e.g. FineWeb-2 subset, 10 languages, deduplicated against all eval sets by 13-gram in every language).
- **Arms.**
  - **Control:** clean corpus.
  - **Arm A (same-language):** inject the 250 MGSM problems *in Swahili and Thai*, at duplication counts $k\in\{1,10,100\}$ (stratified by item, so each item gets exactly one $k$).
  - **Arm B (cross-lingual):** inject the same 250 problems **in English only**, at the same $k$; the target-language strings never appear.
- **Evaluation.** 5-shot accuracy on MGSM in Swahili and Thai, per injected item, stratified by $k$.
- **The deciding number.** $\Delta_B(k{=}100)$ — the accuracy gain in Arm B over Control on Swahili/Thai items whose *English* source was duplicated 100×, with a 95% CI from item-level bootstrap over the 250-item set. If $\Delta_B(100) \ge 5$ accuracy points, cross-lingual contamination is real at 1.4B scale, string-based decontamination is insufficient, and every translated-benchmark number in the literature is suspect by an unmeasured amount. If $\Delta_B(100) \le 1$ point while $\Delta_A(100) \ge 10$, contamination is language-local at this scale and the field's current practice is defensible for models of this size.
- **Cost.** $3\times$ 300B-token runs at 1.4B $\approx$ 15–25k A100-hours total. This is affordable to any academic group with a mid-size cluster, and it has not been done.

## 9. Key References

- **[Foundational]** Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R. Bowman, Holger Schwenk, Veselin Stoyanov. *XNLI: Evaluating Cross-lingual Sentence Representations.* EMNLP 2018. — arXiv:1809.05053
- **[Foundational]** Mikel Artetxe, Sebastian Ruder, Dani Yogatama. *On the Cross-lingual Transferability of Monolingual Representations.* ACL 2020. — arXiv:1910.11856
- **[SOTA]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024. — arXiv:2310.17623
- **[SOTA]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR 2023. — arXiv:2202.07646
- **[SOTA]** Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer. *Detecting Pretraining Data from Large Language Models.* ICLR 2024. — arXiv:2310.16789
- **[Evidence]** Terra Blevins, Luke Zettlemoyer. *Language Contamination Helps Explain the Cross-lingual Capabilities of English Pretrained Models.* EMNLP 2022. — arXiv:2204.08110
- **[Evidence]** Julia Kreutzer et al. *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.* TACL 2022. — arXiv:2103.12028
- **[Evidence]** Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023. — arXiv:2311.04850
- **[Evidence]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM 2024. — arXiv:2402.07841
- **[Benchmark]** Freda Shi et al. *Language Models are Multilingual Chain-of-Thought Reasoners.* ICLR 2023. — arXiv:2210.03057
- **[Benchmark]** Lucas Bandarkar et al. *The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.* ACL 2024. — arXiv:2308.16884
- **[Survey]** Cheng Xu, Shuhao Guan, Derek Greene, M-Tahar Kechadi. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244
- **[Survey]** Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, Eneko Agirre. *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark.* Findings of EMNLP 2023. — arXiv:2310.18018
- **[Context]** Kabir Ahuja et al. *MEGA: Multilingual Evaluation of Generative AI.* EMNLP 2023. — arXiv:2303.12528

## 10. Worked Example

Take MGSM item `gsm8k-test-#12` in Swahili. Its English source is a GSM8K test problem that appears verbatim across GitHub mirrors, HuggingFace dataset viewers, blog posts and tutorial notebooks; conservatively $k \approx 10^2$–$10^3$ copies in a Common Crawl snapshot.

Run the three similarity operators against a 300B-token multilingual corpus:

| Operator | Swahili item vs corpus | Reported $\rho$ on MGSM-sw |
|---|---|---|
| 13-gram exact | 0 matches | 0.000 |
| LaBSE cosine, $\theta=0.80$ | 3 English docs, 1 Swahili news doc | 0.016 |
| LaBSE cosine, $\theta=0.65$ | 41 docs, mostly unrelated arithmetic text | 0.284 |

A $0.15$ move in a threshold with no principled setting swings the contamination rate from 1.6% to 28.4%. That is the block: $\rho$ is a function of a free parameter, so the "measurement" reports the analyst's prior.

Now the identification failure. Suppose the model scores 46% on MGSM-sw items whose English source is high-frequency on the web and 31% on low-frequency items — a 15-point gap. Two explanations fit exactly:

- **Contamination:** the model memorised the English solution and re-emits it through Swahili.
- **Transfer:** high-web-frequency problems are the templatic, easy ones (rate × time, simple percentage), and easy problems transfer better.

Regress score on $\log k$ controlling for problem length, operation count and Swahili token count and the gap shrinks — but there is no covariate set that removes it, because "the model saw a translatable statement of this problem" is the treatment *and* the mechanism of transfer. The only lever that separates them is randomising $k$ during pretraining, which is exactly §8. Until that run exists, every reported multilingual score carries an inflation term whose sign is known and whose magnitude is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*