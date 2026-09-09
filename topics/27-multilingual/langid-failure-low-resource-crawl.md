---
id: 27-multilingual/langid-failure-low-resource-crawl
title: "Language Identification Failure on Low-Resource Web Crawl"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Language Identification Failure on Low-Resource Web Crawl

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/langid-failure-low-resource-crawl` · **Status:** open

## 1. Problem Statement

**Input.** A web crawl snapshot $D = \{d_1,\dots,d_N\}$, $N \sim 10^{10}$ documents, plus a label set $\mathcal{L}$ of languages (typically ISO 639-3 code paired with ISO 15924 script).

**Output.** For each target language $\ell \in \mathcal{L}$, a subset $\hat{S}_\ell \subseteq D$ intended to contain the text actually written in $\ell$.

**Objective.** Maximise recovered volume subject to a precision floor: $\text{prec}(\hat S_\ell) \ge \tau$ (say $\tau = 0.9$) for every $\ell$, including those with a crawl prevalence below $10^{-7}$.

**Solved would mean:** a pipeline that, audited by native speakers on a random sample from $\hat S_\ell$, hits $\tau$ simultaneously for the ~1,500 languages with any web presence — not just the ~100 with enough data to train a classifier.

Three variants, routinely conflated:

- **Measurement.** We cannot cheaply tell how bad a given $\hat S_\ell$ is. Held-out accuracy on FLORES-200 or UDHR is not crawl precision; the only ground truth is native-speaker audit, and for many languages fewer than a dozen qualified annotators are reachable.
- **Method.** Build a classifier or filter chain that holds precision under extreme class imbalance and an unbounded, adversarial negative class (spam, boilerplate, MT output, closely-related languages).
- **Theory.** Under what assumptions is per-language precision on unlabeled crawl even *estimable* without labels? This reduces to mixture proportion estimation, which is not identifiable without an irreducibility assumption.

## 2. Formal Setting

Let $Y(d) \in \mathcal{L} \cup \{\bot\}$ be the true language of document $d$, with $\bot$ for non-linguistic or unlabelable text. The crawl prior is
$$\pi_\ell = \Pr_{d \sim D}[Y(d) = \ell],$$
**measured** by uniform sampling of $D$ followed by human labelling — the only unbiased estimator, and infeasible for $\pi_\ell < 10^{-6}$ (an expected 1 positive per $10^6$ annotations).

A classifier $f: d \mapsto \mathcal{L} \cup \{\bot\}$ with per-language recall $r_\ell = \Pr[f(d)=\ell \mid Y(d)=\ell]$ and false-positive rate $\varepsilon_\ell = \Pr[f(d)=\ell \mid Y(d)\neq\ell]$ yields
$$\text{prec}_\ell = \frac{\pi_\ell r_\ell}{\pi_\ell r_\ell + (1-\pi_\ell)\varepsilon_\ell} \approx \frac{\pi_\ell r_\ell}{\pi_\ell r_\ell + \varepsilon_\ell}.$$
The precision floor $\tau$ therefore demands
$$\varepsilon_\ell \le \frac{1-\tau}{\tau}\,\pi_\ell r_\ell,$$
a *specificity* requirement that scales with the prior, not with any quantity a benchmark reports.

**Measured quantities.**
- $\text{prec}_\ell$: fraction of a uniform sample of $n$ documents from $\hat S_\ell$ judged in-language by $\ge 2$ native speakers; report Wilson interval at $n=200$ ($\pm{\sim}7$ pp near 0.9).
- $\varepsilon_\ell$: estimated from the same audit as $\hat\varepsilon_\ell = |\hat S_\ell|(1-\widehat{\text{prec}}_\ell)/N$.
- Yield $|\hat S_\ell|$ in documents and in deduplicated bytes.

**Assumptions, and where they break.**
1. *Single language per document.* Violated: code-switching and multilingual boilerplate are the norm on low-resource pages.
2. *$\mathcal{L}$ is a partition.* Violated: macrolanguages (e.g. `ms`/`zsm`/`id`), dialect continua, and mutually intelligible pairs make $Y(d)$ genuinely set-valued.
3. *Test distribution matches crawl.* Violated by construction — FLORES-200 is professionally translated Wikipedia prose, single-domain and clean.
4. *Negative class is stationary.* Violated: crawl composition drifts, and MT-generated text now forms a growing share of low-resource web pages.
5. *Irreducibility* (needed for label-free precision estimation): the negative-class distribution contains no component identical to the positive class. Violated exactly where it matters — MT output in $\ell$ and translationese Bible text are near-copies of the positive class.

## 3. State of the Art

**Systems SOTA.**
- **GlotLID** (Kargaran, Imani, Yvon, Schütze, EMNLP Findings 2023) — FastText-based, ~1,665 languages at v1, ~2,100 in later releases; the widest open label set. *Established:* it labels more languages than any predecessor and is reproducible from released artifacts.
- **OpenLID** (Burchell, Birch, Bogoychev, Heafield, ACL 2023) — 201 languages, fully open training data with per-source provenance, deliberately built so errors are auditable. *Established:* high FLORES-200 macro-F1 (reported ≈0.98).
- **NLLB / No Language Left Behind** (NLLB Team, 2022) — 200-language LID plus cascaded filters; the mined bitext is the most-used artifact.
- **CCNet** (Wenzek et al., LREC 2020) and **fastText `lid.176`** (Joulin et al., 2016) remain the default in most pretraining pipelines despite covering ≤176 languages.
- **MADLAD-400** (Kudugunta et al., NeurIPS D&B 2023) — 419 languages from CommonCrawl, with a documented manual audit that led the authors to remove or flag a substantial set of languages.

**Claimed but unablated.** That wider label sets improve *crawl* precision. Every headline number for GlotLID/OpenLID/NLLB-LID is a benchmark number on FLORES-200, UDHR, or held-out splits of their own training sources. Almost no paper reports native-speaker-audited precision on a random sample of its own crawl output, per language. The one systematic exception is the audit literature (§4), which is negative.

**Theory SOTA.** Nothing language-specific. The relevant results are generic: open-set recognition (Scheirer et al., TPAMI 2013) and mixture proportion estimation (Blanchard, Lee & Scott, JMLR 2010; Scott, AISTATS 2015), which establish that the positive-class proportion is identifiable only under an irreducibility condition.

## 4. What Is Known

- **Kreutzer et al., TACL 2022** ("Quality at a Glance") audited 205 language-specific corpora across CCAligned, ParaCrawl, WikiMatrix, JW300 and OSCAR with speakers of the target languages. For a substantial minority of low-resource corpora, **under 50%** of sampled lines were in the labelled language; **at least 15 audited corpora contained essentially no correct text (≈0%)**. Scale: ~100 sentences sampled per corpus, ~50 annotators.
- **Caswell et al., COLING 2020** ("Language ID in the Wild") showed that for many of the ~1,600 languages they attempted, *unfiltered* CLD3-style LID output on web crawl had precision near zero, and that recovering usable corpora required cascades of wordlist, script, and cluster-consistency filters rather than a better classifier.
- **Precision collapses with document length.** Character-$n$-gram LID accuracy degrades sharply below ~50 characters; short-text degradation is reproduced across langid.py (Lui & Baldwin, ACL 2012 demo), CLD3, and fastText.
- **Domain skew is systematic.** Low-resource "corpora" are dominated by religious translations (Bible, JW300) and machine translation output; Kreutzer et al. document this directly.
- **Contamination is measurable downstream.** Blevins & Zettlemoyer (EMNLP 2022) showed nominally English pretraining corpora carry non-trivial non-English text — evidence the same filters leak in both directions.
- **Benchmark scores are high and crawl precision is low at the same time.** OpenLID reports ≈0.98 FLORES-200 macro-F1; audits of comparably-built pipelines report per-language crawl precision below 0.5 for dozens of languages. Both are correct measurements of different quantities.

## 5. What Is Not Known

- **Methodologically blocked (the dominant gap).** There is no accepted, cheap estimator of per-language crawl precision. Reported "LID accuracy" measures a clean, balanced, single-domain benchmark; the deployed quantity is precision under $\pi_\ell \sim 10^{-7}$ imbalance on adversarial text. No standard reporting format requires the second.
- **Theoretically open.** Whether $\text{prec}_\ell$ admits a nontrivial distribution-free lower bound from unlabeled crawl plus a small clean seed set. Irreducibility fails for MT output, so the classical MPE guarantees do not transfer; no impossibility theorem specific to this setting has been proved either.
- **Empirically open.** Whether a modern multilingual encoder or LLM-based verifier, applied as a second-stage filter, raises audited crawl precision for sub-$10^{-7}$ languages. Runnable today at ~$10^5$ GPU-hours plus annotation; nobody has published it with native-speaker audit as the outcome variable.
- **Open.** The right treatment of dialect continua and macrolanguages — whether the label should be a distribution over $\mathcal{L}$ rather than an element of it.

## 6. Why It Is Hard

**The specific obstruction is base-rate amplification combined with an evaluation that does not measure what it names.**

Precision degrades as $\varepsilon_\ell / \pi_\ell$. Every order of magnitude of rarity demands an order of magnitude better specificity. A classifier at 99.99% specificity — excellent by any benchmark — produces $10^6$ false positives on a $10^{10}$-document crawl, swamping a language with 1,000 true documents by 1000:1. No amount of accuracy improvement measured on a balanced test set constrains $\varepsilon_\ell$ at the $10^{-8}$ scale required, because balanced test sets have no resolution below $1/n_{\text{test}}$.

Compounding this: the negative class is not random. Its densest region is text that is *nearly* the target language — a related dialect, MT output, transliterated content — precisely the region where the classifier's margin is smallest and where irreducibility (§2, assumption 5) fails, so label-free correction is not identifiable. And the ground truth is genuinely scarce: for many languages, the pool of reachable annotators is single-digit.

## 7. Current Research (as of 2026)

- **Wider, auditable LID.** GlotLID (LMU Munich / CIS, Schütze group) and OpenLID (Edinburgh, Birch/Heafield) continue to expand coverage with released provenance. Growth is in label count; per-language crawl-precision reporting remains rare.
- **Cascade filtering over better classifiers.** Google's MADLAD-400 line and the CommonCrawl-derived corpora (HPLT, FineWeb-2 multilingual efforts) treat LID as one stage among script checks, perplexity filters, wordlists and dedup. *(frontier — verify: FineWeb-2's per-language audit protocol and its published precision figures.)*
- **LLM-as-verifier second stage.** Using a multilingual LLM to confirm a candidate document's language. Cheap per document only after aggressive first-stage recall; ablation against native-speaker audit is the missing piece. *(frontier — verify)*
- **MT-output detection** as an explicit negative class for low-resource crawl. *(frontier — verify)*
- **Community-in-the-loop auditing.** Masakhane, AmericasNLP, and the Aya/Cohere Labs multilingual data efforts supply the annotator pool that makes audits possible at all.

## 8. Concrete Next Experiment

**Question.** Does high benchmark LID accuracy predict crawl precision for rare languages?

**Scale.** One fixed CommonCrawl snapshot (~$3\times10^9$ documents after dedup). Choose 40 languages with estimated $\pi_\ell \in [10^{-8}, 10^{-6}]$ and a reachable annotator pool of $\ge 2$ native speakers each. Run three LID systems — fastText `lid.176`, OpenLID, GlotLID — at their default thresholds. From each $\hat S_\ell$, sample 200 documents uniformly; two native speakers label each as in-language / related-language / MT-output / not-language. Cost: 40 × 200 × 3 systems × 2 annotators = 96,000 judgements; deduplicate overlapping documents across systems to cut this by roughly half.

**Control arm.** The same 3 systems on 10 high-resource languages ($\pi_\ell > 10^{-2}$), audited identically, plus each system's FLORES-200 macro-F1 on exactly the 50 languages involved.

**Deciding number.** The **gap** $\Delta = \overline{F1}_{\text{FLORES}} - \overline{\text{prec}}_{\text{crawl}}$ on the 40 rare languages, minus the same gap on the 10 control languages. If $\Delta_{\text{rare}} - \Delta_{\text{control}} > 0.30$ with 200-sample Wilson intervals, benchmark accuracy is established as non-predictive of crawl precision in the rare regime, and every corpus paper reporting only FLORES-200 numbers is reporting the wrong quantity. If $\Delta_{\text{rare}} - \Delta_{\text{control}} < 0.10$, the base-rate argument overstates the practical problem and the field can keep its current reporting.

**Secondary readout.** Precision as a function of crawl subsample size $N' \in \{10^6, 10^7, 10^8, 10^9\}$. The base-rate model predicts $\text{prec}_\ell$ is flat in $N'$ while $\varepsilon_\ell$ is fixed — a *decline* would indicate the false-positive population is itself concentrated in the long tail of the crawl.

## 9. Key References

- **[Foundational]** Marco Lui, Timothy Baldwin. *langid.py: An Off-the-shelf Language Identification Tool.* ACL 2012 (System Demonstrations).
- **[Foundational]** Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov. *Bag of Tricks for Efficient Text Classification.* EACL 2017 — arXiv:1607.01759. (Basis of `lid.176`.)
- **[Foundational]** Isaac Caswell, Theresa Breiner, Daan van Esch, Ankur Bapna. *Language ID in the Wild: Unexpected Challenges on the Path to a Thousand-Language Web Text Corpus.* COLING 2020 — arXiv:2010.14571.
- **[Survey / audit]** Julia Kreutzer et al. *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.* TACL, 2022 — arXiv:2103.12028.
- **[SOTA]** Laurie Burchell, Alexandra Birch, Nikolay Bogoychev, Kenneth Heafield. *An Open Dataset and Model for Language Identification.* ACL 2023 (Short Papers).
- **[SOTA]** Amir Hossein Kargaran, Ayyoob Imani, François Yvon, Hinrich Schütze. *GlotLID: Language Identification for Low-Resource Languages.* Findings of EMNLP 2023 — arXiv:2310.16248.
- **[SOTA]** Sneha Kudugunta et al. *MADLAD-400: A Multilingual And Document-Level Large Audited Dataset.* NeurIPS 2023 Datasets & Benchmarks — arXiv:2309.04662.
- **[Systems]** Guillaume Wenzek et al. *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data.* LREC 2020 — arXiv:1911.00359.
- **[Systems]** NLLB Team. *No Language Left Behind: Scaling Human-Centered Machine Translation.* 2022 — arXiv:2207.04672.
- **[Theory]** Gilles Blanchard, Gyemin Lee, Clayton Scott. *Semi-Supervised Novelty Detection.* JMLR 11, 2010.
- **[Theory]** Walter J. Scheirer, Anderson de Rezende Rocha, Archana Sapkota, Terrance E. Boult. *Toward Open Set Recognition.* IEEE TPAMI 35(7), 2013.
- **[Related]** Terra Blevins, Luke Zettlemoyer. *Language Contamination Helps Explain the Cross-lingual Capabilities of English Pretrained Models.* EMNLP 2022 — arXiv:2204.08110.

## 10. Worked Example

Take a language with roughly 1M speakers and thin web presence — say Dyula (`dyu_Latn`). Assume a deduplicated crawl of $N = 3\times10^9$ documents and a true prevalence $\pi = 3\times10^{-7}$, so about **900 true documents exist**.

Give the classifier generous parameters: recall $r = 0.90$, false-positive rate $\varepsilon = 10^{-5}$ (i.e. 99.999% specificity — far better than any published system's measurable resolution).

$$\text{TP} = \pi N r = 3\times10^{-7}\cdot 3\times10^{9}\cdot 0.9 = 810$$
$$\text{FP} = (1-\pi)N\varepsilon \approx 3\times10^{9}\cdot 10^{-5} = 30{,}000$$
$$\text{prec} = \frac{810}{810+30{,}000} = 0.026$$

**2.6% precision.** A native-speaker audit of 200 sampled documents from $\hat S_{\text{dyu}}$ would find about 5 in-language. The rest would be Bambara, Jula-adjacent Manding varieties, French boilerplate, and MT output — because the false positives concentrate in the near-neighbour region, not uniformly.

To reach $\tau = 0.9$, invert the constraint:
$$\varepsilon \le \frac{1-\tau}{\tau}\pi r = \frac{0.1}{0.9}\cdot 3\times10^{-7}\cdot 0.9 = 3\times10^{-8}.$$

That is a **333× reduction** in false-positive rate, to a level of $3\times10^{-8}$. Verifying it empirically requires observing roughly $1/\varepsilon \approx 3\times10^{7}$ correctly-rejected negatives per expected false positive — so a test set of $10^{9}$ labelled negatives to estimate $\varepsilon$ to within a factor of two.

**This is the obstruction made visible.** FLORES-200 has 1,012 sentences per language. Its finest measurable false-positive rate is $\sim10^{-3}$. The quantity that determines whether the corpus is usable lives five orders of magnitude below the benchmark's resolution floor. A system can be perfect on FLORES-200 — zero errors, F1 = 1.00 — and still produce a 2.6%-precision Dyula corpus, with nothing in the reported numbers to warn you.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*