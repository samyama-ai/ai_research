---
id: 27-multilingual/mt-contaminated-web-collapse
title: "Model Collapse from Machine-Translated Web Text"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Model Collapse from Machine-Translated Web Text

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/mt-contaminated-web-collapse` · **Status:** open

## 1. Problem Statement

Web crawls for low-resource languages are dominated by machine-translated text. The text was produced by earlier MT systems, which were themselves trained on web crawls. Training a new multilingual model on that crawl is therefore a recursive self-consumption loop that runs *through the web*, not inside a lab.

Three distinct problems, routinely conflated:

- **Measurement.** Given a corpus $D_\ell$ in language $\ell$, estimate the fraction $\pi_\ell$ that is machine output, and estimate the *generational depth* $g$ — how many MT hops separate a document from human text. Solving means a detector with calibrated per-language error, not an English-trained classifier applied blind.
- **Method.** Given $\pi_\ell$ known or estimated, produce a filtering, reweighting, or mixing rule that yields a better model per unit compute than training on the raw crawl. Solving means a rule that beats the "keep everything" baseline on human-authored held-out text, at fixed token budget.
- **Theory.** Existing collapse theory assumes an *iterated closed loop*: generation $t+1$ trains only on generation $t$'s samples. The web is an *accumulating, mixed-depth, selection-biased* pool. Solving means a degradation bound as a function of $(\pi_\ell, g, \text{accumulation rate})$ that reduces to the known closed-loop results in the limit.

The decision predicate for the catalog: **does MT contamination at observed web rates cause measurable capability loss beyond ordinary low-quality-data effects, and does removing it help?** Neither direction is established.

## 2. Formal Setting

Let $p_\ell$ be the human-authored distribution over text in language $\ell$. Let $T_{s \to \ell}$ be an MT operator mapping a distribution over source language $s$ to $\ell$. The observed crawl distribution is a mixture over generational depth:

$$q_\ell = \sum_{g \ge 0} \pi_\ell(g)\, q_\ell^{(g)}, \qquad q_\ell^{(0)} = p_\ell, \qquad q_\ell^{(g)} = \mathbb{E}_{s}\big[T^{(g)}_{s \to \ell} \circ S_g \circ q_s^{(g-1)}\big]$$

where $S_g$ is a **selection operator** — the web does not translate a uniform sample of source text, it translates whatever is commercially worth translating.

Measured quantities:

- $\pi_\ell(g)$ — from a detector $h$ with per-language confusion matrix estimated on a human-audited seed set. In practice only $\hat{\pi}_\ell = \Pr[h(x) = \text{MT}]$ is available, and $\hat\pi$ is a biased estimate of $\pi$ unless the confusion matrix is applied as a correction.
- **Generational depth** $g$: no direct estimator exists. Proxies are multi-way parallelism (a sentence appearing in $k \ge 3$ aligned translations) and back-translation round-trip stability.
- **Diversity collapse**: tail mass loss. With unigram counts $c_v$ over a fixed vocabulary, measure $\mathrm{TTR}$ (type-token ratio) at fixed token count, and Zipf tail exponent $\alpha$ fitted on ranks $10^3$–$10^5$. Collapse predicts $\alpha$ increasing (tail thinning) generation over generation.
- **Capability loss**: $\Delta = \mathcal{L}_{p_\ell}(\theta_{\text{raw}}) - \mathcal{L}_{p_\ell}(\theta_{\text{filtered}})$, cross-entropy on a **human-authored, natively-written** held-out set — not a translated benchmark.

Assumptions, with the violated ones flagged:

1. *Held-out human text exists for $\ell$.* **Violated** for most of the ~200 languages in question; FLORES-200 and most evaluation sets are themselves translations from English.
2. *The detector is language-independent.* **Violated** — MT detection accuracy is measured almost entirely on high-resource pairs.
3. *One MT system generated the text.* **Violated** — crawls mix dozens of systems across a decade.
4. *Human text is the target.* Partly violated — human translation is also translationese, so $p_\ell$ itself contains translated material.

## 3. State of the Art

**Established (contamination side).** Thompson et al. (Findings of ACL 2024) audited 6.38B web sentences and found 57.1% sit in multi-way parallel sets spanning 3+ languages, with parallelism rising sharply as resource level falls. Kreutzer et al. (TACL 2022) audited 205 language corpora across five public multilingual datasets by hand; many low-resource subsets had under 50% correct-language content and 15 had essentially none. These are measurements, and they replicate.

**Established (collapse side, but in the closed loop).** Shumailov et al. (*Nature*, 2024) show perplexity divergence and tail loss under strict generational replacement. Dohmatob et al. (ICML 2024) prove that under synthetic contamination the power-law scaling $L(n) \sim n^{-c}$ acquires a finite floor — more data stops helping. Gerstgrasser et al. (COLM 2024) show that *accumulating* real plus synthetic data instead of replacing it bounds test error by a constant, i.e. collapse is an artifact of replacement.

**Claimed but unablated.** That web MT contamination causes model collapse in deployed LLMs. This is asserted in the framing of several contamination papers and widely repeated, but no paper has trained matched models on contaminated versus decontaminated crawls of the same language and reported the gap. The bridge from "the web is 57% parallel" to "models degrade" is currently an inference, not a result.

**Benchmark-number-only.** Reported multilingual gains from MT-filtering pipelines (MADLAD-400's audited filters, NLLB's data cleaning) come as end-task scores on translated benchmarks with many co-varying filter changes. They do not isolate MT provenance as the causal factor.

**Counter-evidence.** Artetxe et al. (EMNLP 2022) found that for low-resource languages, aggressive corpus quality filtering gave little or no downstream benefit relative to keeping the noisy data — data volume dominated cleanliness at the scales tested.

## 4. What Is Known

- 57.1% multi-way parallelism in a 6.38B-sentence multilingual crawl; translations into low-resource languages are systematically shorter and drawn from lower-quality source content (Thompson et al., ACL Findings 2024).
- Hand audit of 205 corpora: multiple low-resource subsets below 50% in-language; several at 0% (Kreutzer et al., TACL 2022, scale: manual annotation of ~100 sentences per corpus).
- Closed-loop collapse in language models: OPT-125M fine-tuned recursively on its own generations shows monotone perplexity increase and loss of low-probability events within ~5–10 generations (Shumailov et al., Nature 2024).
- Accumulation defeats replacement: with real data retained, error is bounded, verified for transformers up to ~125M params on TinyStories/Wikipedia-scale corpora (Gerstgrasser et al., COLM 2024).
- Tolerable synthetic fraction has a bound: Seddik et al. (COLM 2024) give conditions relating the number of synthetic samples to real samples under which collapse is avoided.
- MT output has lower lexical diversity than human translation of the same content — measured by TTR and MTLD on WMT-scale corpora (Vanmassenhove et al., EACL 2021).
- Cross-lingual capability of "English-only" models is partly explained by unintended non-English contamination of their pretraining data (Blevins & Zettlemoyer, EMNLP 2022).

## 5. What Is Not Known

- **Empirically open.** The decisive experiment — matched pretraining runs on contaminated versus MT-filtered crawls for the same low-resource language, evaluated on natively-authored held-out text — is runnable today at 1B parameters for under $50k of compute. Nobody has published it. This is the central gap.
- **Theoretically open.** No bound on degradation for the *open, accumulating, selection-biased* loop. All existing theory (Dohmatob, Seddik, Gerstgrasser) assumes a controlled generational structure. Whether web-scale mixing is closer to the benign accumulation regime or the malign replacement regime is unproven either way.
- **Methodologically blocked.** (a) Generational depth $g$ has no estimator. (b) Per-language MT detection has no calibrated error model outside high-resource pairs. (c) For most affected languages there is no natively-authored evaluation set, so the objective $\mathcal{L}_{p_\ell}$ cannot be computed at all.

## 6. Why It Is Hard

**The evaluation does not measure what it names.** For a language like Sinhala or Yoruba, the standard benchmarks (FLORES-200, XNLI, translated MMLU) are English texts rendered into $\ell$. A model trained on MT-heavy data is *better* at MT-style text. Filtering out MT data will therefore lower the benchmark score while possibly improving the model on native text. The measurement instrument is correlated with the contaminant. Every reported "filtering helps/hurts" number inherits this confound.

Secondary obstruction: **non-identifiability of provenance from surface statistics.** Machine translationese and human translationese share the same signature — shortened sentences, flattened lexis, source-language syntactic shadowing. A detector trained to separate them is separating fluency, not provenance, and its errors are correlated with exactly the tail-diversity statistic used to measure collapse.

## 7. Current Research (as of 2026)

- Data auditing at scale — AWS AI Labs (Thompson and colleagues) on multi-way parallelism; Google/MADLAD and the Cohere Labs Aya effort on documented, audited multilingual corpora.
- Collapse theory — Kempe and Dohmatob (NYU) on scaling-law modification; Gerstgrasser/Koyejo (Stanford) on accumulation regimes; Seddik et al. (MBZUAI) on sample-count bounds.
- Native-first evaluation — the shift away from translated benchmarks toward locally authored sets (Global-MMLU-style and community-authored African-language efforts). This is the prerequisite for the decisive experiment. *(frontier — verify current release status.)*
- Provenance watermarking of MT output, so that future crawls carry generational labels. Proposed; no deployment at web scale. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Language.** One with a real native evaluation set and high measured contamination — Sinhala or Nepali.

**Scale.** 1.4B-parameter decoder, 30B tokens, English-plus-$\ell$ mixture with $\ell$ held at 10% of tokens. Roughly $2 \times 10^{21}$ FLOPs per arm; four arms fit in ~4k H100-hours.

**Arms.**
1. Raw crawl for $\ell$.
2. MT-filtered: drop documents flagged by a multi-way-parallelism detector, then **backfill to the same token count** from the remaining raw pool.
3. **Control arm (essential):** drop a *random* equal-sized document subset and backfill identically. This separates "removing MT" from "removing data and re-sampling".
4. Perplexity-filtered: drop the same volume by a generic quality classifier, not an MT detector.

**Evaluation.** Cross-entropy on natively-authored held-out $\ell$ text (local news, forum posts, literature — never translated), plus Zipf tail exponent $\alpha$ of generated text at temperature 1.0.

**The deciding number.** $\Delta = \mathcal{L}_{\text{native}}(\text{arm 1}) - \mathcal{L}_{\text{native}}(\text{arm 2})$, with arm 3 as the null. If $\Delta > 0.05$ nats and exceeds the arm-1-vs-arm-3 difference by $3\times$, MT contamination is a distinct, actionable harm. If $\Delta < 0.02$ nats, it is not — and a decade of filtering effort is being spent on the wrong axis.

## 9. Key References

- **[Foundational]** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, Yarin Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024.
- **[Foundational]** Brian Thompson, Mehak Preet Dhaliwal, Peter Frisch, Tobias Domhan, Marcello Federico. *A Shocking Amount of the Web is Machine Translated: Insights from Multi-Way Parallelism.* Findings of ACL, 2024. — arXiv:2401.05749
- **[Foundational]** Julia Kreutzer et al. *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.* TACL 10, 2022.
- **[SOTA]** Elvis Dohmatob, Yunzhen Feng, Pu Yang, Francois Charton, Julia Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML, 2024. — arXiv:2402.07043
- **[SOTA]** Matthias Gerstgrasser et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM, 2024. — arXiv:2404.01413
- **[SOTA]** Mohamed El Amine Seddik, Suei-Wen Chen, Soufiane Hayou, Pierre Youssef, Merouane Debbah. *How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse.* COLM, 2024.
- **[Supporting]** Eva Vanmassenhove, Dimitar Shterionov, Matthew Gwilliam. *Machine Translationese: Effects of Algorithmic Bias on Linguistic Complexity in Machine Translation.* EACL, 2021.
- **[Supporting]** Mikel Artetxe, Vedanuj Goswami, Shruti Bhosale, Angela Fan, Luke Zettlemoyer. *Does Corpus Quality Really Matter for Low-Resource Languages?* EMNLP, 2022.
- **[Supporting]** Terra Blevins, Luke Zettlemoyer. *Language Contamination Helps Explains the Cross-lingual Capabilities of English Pretrained Models.* EMNLP, 2022.
- **[Supporting]** Sneha Kudugunta et al. *MADLAD-400: A Multilingual And Document-Level Large Audited Dataset.* NeurIPS Datasets & Benchmarks, 2023.

## 10. Worked Example

Take a hypothetical low-resource crawl of 2B tokens. Apply the multi-way-parallelism criterion: 57% of sentences are in 3+ way parallel sets, matching the ACL-2024 audit rate for this resource tier. So $\hat{\pi} = 0.57$, leaving 860M tokens after filtering.

Now the obstruction, made numeric.

Assume the detector has 90% recall on MT and 80% precision, plausible given that no calibration exists for this language and high-resource-trained detectors degrade. Of the 1.14B flagged tokens, 20% — 228M tokens — are human-authored text misflagged for being translationese-like. Human *translations* and formal register news prose are exactly what gets misflagged. So filtering removes about 21% of the language's genuine human text along with the machine text, in a setting where total human text is the binding constraint.

Two competing scaling effects:

- Contamination penalty. If the Dohmatob floor applies, the contaminated arm's loss saturates: doubling tokens past some $n^\ast$ buys nothing.
- Volume penalty. Chinchilla-style, dropping from 2B to 0.86B tokens for $\ell$ costs roughly $L \propto n^{-0.28}$, i.e. $(2/0.86)^{0.28} \approx 1.27$ — about a 27% increase in the language-specific loss term, before any quality gain is counted.

For filtering to pay, the contamination penalty must exceed a 27%-equivalent volume loss compounded by the 21% collateral removal of good text. Nobody has measured either side of that inequality for a single language. And the natural way to check — run both arms and compare on FLORES — is invalid, because FLORES *is* translated text, so it scores the contaminated arm higher by construction. The obstruction is not that the experiment is expensive. It is that the standard instrument reads the contaminant as signal.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*