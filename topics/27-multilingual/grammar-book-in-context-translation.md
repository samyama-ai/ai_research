---
id: 27-multilingual/grammar-book-in-context-translation
title: "Grammar-Book Prompting for Unseen Languages"
topic: 27-multilingual
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Grammar-Book Prompting for Unseen Languages

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/grammar-book-in-context-translation` · **Status:** partially-solved

## 1. Problem Statement

Give a language model, at inference time, the documentation a human field linguist would use for a language absent from pretraining — a reference grammar, a bilingual wordlist, a few hundred glossed example sentences — and ask it to translate. No gradient updates. The question is whether the model *uses the grammar* or merely does retrieval and analogy over the parallel material that happens to sit inside the same document.

Three variants, routinely conflated:

- **Measurement.** Define a score that separates "learned the language from its description" from "copied from nearby parallel text". Requires a language with near-zero web presence and a decomposable prompt corpus.
- **Method.** Maximise translation quality under a fixed documentation budget. Retrieval, chunking, and glossing pipelines all count; nothing forbids the model from ignoring the prose.
- **Theory.** Characterise what class of grammatical generalisation is reachable by in-context computation over a description versus only by weight updates. Essentially untouched.

Solving it means: an ablation-verified demonstration that declarative grammatical prose contributes measurable, non-substitutable quality on a language the model has never seen, replicated across several unrelated languages.

## 2. Formal Setting

Let $L$ be a target language with an *evidence bundle* $B = (G, W, P, T)$: grammar prose $G$, wordlist $W$ (entries $(w_L, w_{\text{en}})$), parallel sentences $P = \{(x_i, y_i)\}_{i=1}^{n}$, and monolingual text $T$. Held out is a test set $D = \{(x_j^\ast, y_j^\ast)\}_{j=1}^{m}$ disjoint from $P$.

A system is a map $f_\theta(\cdot \mid S)$ where $S \subseteq B$ is the retrieved context, $\theta$ frozen. Quality is chrF++ (character $n$-gram F-score, $\beta=2$), reported per direction:
$$Q(S) = \frac{1}{m}\sum_{j=1}^{m}\mathrm{chrF}\big(f_\theta(x_j^\ast \mid S),\, y_j^\ast\big).$$

The quantity of interest is the **grammar increment**, measured by removing prose while holding everything else fixed:
$$\Delta_G = Q(G \cup W \cup P) - Q(W \cup P).$$

Two confounds must be measured, not assumed away.

- **Leakage of parallel data into prose.** A grammar book contains thousands of glossed examples. Let $\rho$ be the fraction of test-set content $n$-grams recoverable from $G$'s example sentences by exact or fuzzy match. $\Delta_G$ is interpretable only alongside $\rho$; a large $\Delta_G$ at large $\rho$ is retrieval, not grammar.
- **Pretraining exposure.** Let $Q(\emptyset)$ be zero-context performance. If $Q(\emptyset) \gg 0$, $L$ is not unseen.

Budget must be stated: context tokens $C = |S|$, and quality reported as $Q$ at fixed $C$, since $Q$ rises with $C$ for trivial reasons.

**Assumptions known to be violated.**

1. *Test independence from $P$.* Violated: test sentences and prompt corpus usually come from one elicitation session, one speaker, one domain.
2. *Zero pretraining exposure.* Violated in degree for every language except a handful; even Kalamang has some web trace via the documentation project itself.
3. *chrF measures comprehension.* Violated: chrF rewards copied proper nouns and shared loanwords; for morphologically rich $L$ it is dominated by stem overlap.
4. *Single reference adequacy.* Violated: $m$ is typically 50–100 with one reference, so per-system 95% CIs span several chrF points.

## 3. State of the Art

**Benchmarks.** MTOB (*A Benchmark for Learning to Translate a New Language from One Grammar Book*, Tanzer, Suzgun, Visser, Jurafsky, Melas-Kyriazi, ICLR 2024) is the reference setting: Kalamang (kgv, ~200 speakers, West Papua), a 573-page grammar, a bilingual wordlist, and ~400 parallel sentences. ZhuangBench (Zhang, Wang, Zhao et al., *Teaching Large Language Models an Unseen Language on the Fly*, ACL Findings 2024) plays the same role for Zhuang with a dictionary-centric method, DiPMT++.

**Empirical SOTA.** Long-context frontier models with the whole bundle in the prompt. The Gemini 1.5 technical report (Google DeepMind, 2024) reports Kalamang chrF at or above the human-learner baseline in at least one direction using the full book. This is *a benchmark number only*: the report gives no ablation isolating prose from the book's embedded parallel examples.

**Established-by-ablation SOTA is weaker than the headline.** Aycock, Stap, Wu, Monz, Sima'an (*Can LLMs Really Learn to Translate a Low-Resource Language from One Grammar Book?*, ICLR 2025) decompose MTOB and find that the parallel sentences and wordlist carry nearly all of the gain; grammar prose contributes little once lexical evidence is present. Hus & Anastasopoulos (*Back to School: Translation Using Grammar Books*, EMNLP 2024) extend grammar-book prompting to 16 low-resource languages and report modest, inconsistent gains. LingoLLM / *Hire a Linguist!* (Zhang et al., ACL Findings 2024) shows that structured linguistic resources — dictionary, morphological analyser, grammar snippets — help, but the analyser and dictionary do the heavy lifting.

**Theory SOTA.** None specific to this problem. The nearest results are generic in-context-learning analyses (induction heads, implicit-gradient views), which say nothing about acquiring a grammar from its metalinguistic description.

## 4. What Is Known

- **Scale of the reference setting.** MTOB: one grammar (~573 pages, several hundred thousand tokens), a ~2k-entry wordlist, ~400 sentence pairs. Human-learner baselines sit in the low-to-mid 50s chrF; GPT-4-class models at MTOB publication sat in the mid-40s — below human, above no-context.
- **Zero-context performance is near floor.** Frontier models score in the single digits to low teens chrF on Kalamang with no bundle, confirming near-absence from pretraining. This is the strongest evidence in the area.
- **Lexicon dominance.** Across MTOB (kgv), ZhuangBench (za), and the 16-language EMNLP 2024 set, the wordlist plus parallel sentences recovers the large majority of the total gain. The grammar-prose increment $\Delta_G$ is typically within a few chrF of zero and not consistently positive across directions.
- **Dictionary-only prompting works surprisingly well.** Elsner & Needle (*Translating a low-resource language using GPT-3 and a human-readable dictionary*, SIGMORPHON 2023) get non-trivial output from dictionary lookup plus prompting alone.
- **Direction asymmetry.** Into-English is consistently easier than out-of-English at equal context; the English side benefits from the model's fluent decoder.
- **Long context is necessary but not sufficient.** Doubling retrieved chunks past the point where the relevant wordlist entries are present yields flat chrF.

## 5. What Is Not Known

- **Empirically open.** Whether $\Delta_G > 0$ robustly for *any* frontier model on a language where $\rho$ (test-content recoverable from prose examples) is measured and small. Existing ablations use one or two languages and do not report $\rho$. The experiment is runnable today; nobody has run it across enough languages to separate signal from the ±3 chrF noise floor of $m \approx 100$ test items.
- **Empirically open.** Whether grammar prose helps *specific phenomena* — clitic placement, evidentiality, alignment — even when corpus-level chrF is flat. No phenomenon-targeted test suite exists for any MTOB-class language.
- **Methodologically blocked.** Whether the model "learned the grammar". chrF on 50–100 sentences cannot distinguish rule application from analogical copying; there is no accepted measurement that does, and no ground truth on what the model internally used.
- **Theoretically open.** Whether any class of grammatical rule is in-context-learnable from description but not from equivalent-length examples. No separation result either way.

## 6. Why It Is Hard

**Confounded measurement, with a structural cause.** A reference grammar is not a prose description with examples appended — it is *mostly* interlinear glossed examples. Removing the prose without removing the examples is not a clean cut, and removing both changes the token budget. So the ablation that would establish $\Delta_G$ is not well defined on real grammars: the treatment and the control differ in two variables at once.

Compounding it: the pool of genuinely unseen, well-documented languages with a held-out test set is roughly a handful. Each one is a single sample, so per-language noise ($m \le 100$, single reference, ±3 chrF) is the same size as the effect being measured. You cannot buy statistical power with compute; you would have to commission new field documentation.

## 7. Current Research (as of 2026)

- **Ablation-first re-evaluation.** The Amsterdam ILLC line (Aycock, Monz, Sima'an) continues decomposing evidence bundles; the direction is towards measuring lexical-overlap confounds explicitly rather than reporting bundle-level chrF.
- **Grammar-book scaling to more languages.** Hus & Anastasopoulos (GMU) and follow-ups push beyond Kalamang to grammars scraped from descriptive-linguistics archives, trading documentation quality for language count.
- **Glossing as the intermediate representation.** SIGMORPHON/ComputEL work on automatic interlinear glossing (Ginn et al.) is being used as a pipeline stage: gloss first, translate second, so the grammar is consulted for morphology rather than for whole sentences. *(frontier — verify)*
- **New unseen-language benchmarks with contamination audits.** Several groups are reported to be building MTOB successors with pre-registered $\rho$ measurement and multi-reference test sets. *(frontier — verify)*
- **Long-context vs. RAG over the book.** Whether to stuff the grammar or retrieve from it is an open engineering question; retrieval currently matches stuffing at lower cost.

## 8. Concrete Next Experiment

**Question.** Is $\Delta_G$ positive after controlling for example leakage and token budget?

**Scale.** Four languages with near-zero web presence and full descriptive grammars (Kalamang plus three from archive holdings), $m = 300$ test sentences each, *two* independent references per sentence, both directions. Three frontier long-context models. Total ≈ 7.2k translations per arm — a few hundred GPU-hours of API inference, no training.

**Arms.**
1. $W \cup P$ — lexicon plus parallel sentences. **This is the control arm.**
2. $W \cup P \cup G_{\text{prose}}$ — add grammar prose with all interlinear examples *stripped*.
3. $W \cup P \cup G_{\text{full}}$ — unmodified grammar.
4. $W \cup P \cup \text{shuffled }G_{\text{prose}}$ — sentence-shuffled prose, matched token count. Placebo for "more tokens helps".

All arms padded to identical context length $C$ with neutral filler. Report $\rho$ per language.

**Deciding number.** $\Delta_G^{\text{clean}} = Q(\text{arm 2}) - \max\big(Q(\text{arm 1}), Q(\text{arm 4})\big)$, averaged over languages and directions. **If $\Delta_G^{\text{clean}} \ge +3$ chrF with a 95% bootstrap CI excluding zero, grammar prose is doing real work and the method problem is live. If the CI contains zero, the field should stop reporting book-level results and report lexicon-normalised ones.**

## 9. Key References

- **[Foundational]** Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, Luke Melas-Kyriazi. *A Benchmark for Learning to Translate a New Language from One Grammar Book.* ICLR 2024. — arXiv:2309.16575
- **[SOTA / ablation]** Seth Aycock, David Stap, Di Wu, Christof Monz, Khalil Sima'an. *Can LLMs Really Learn to Translate a Low-Resource Language from One Grammar Book?* ICLR 2025. — arXiv:2409.19151
- **[SOTA]** Chen Zhang, Xiao Liu, Jiuheng Lin, Yansong Feng. *Teaching Large Language Models an Unseen Language on the Fly.* Findings of ACL 2024. — arXiv:2402.19167
- **[Method]** Jonathan Hus, Antonios Anastasopoulos. *Back to School: Translation Using Grammar Books.* EMNLP 2024.
- **[Method]** Kexun Zhang, Yee Man Choi, Zhenqiao Song, Taiqi He, William Yang Wang, Lei Li. *Hire a Linguist!: Learning Endangered Languages with In-Context Linguistic Descriptions.* Findings of ACL 2024. — arXiv:2402.18025
- **[Foundational]** Micha Elsner, Jordan Needle. *Translating a low-resource language using GPT-3 and a human-readable dictionary.* SIGMORPHON 2023.
- **[Resource]** Eline Visser. *A Grammar of Kalamang.* Language Science Press, 2022.
- **[Systems]** Gemini Team, Google DeepMind. *Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context.* Technical report, 2024. — arXiv:2403.05530
- **[Survey]** NLLB Team et al. *No Language Left Behind: Scaling Human-Centered Machine Translation.* Nature, 2024.

## 10. Worked Example

Take one Kalamang→English test sentence containing the verb root *toni* ('say') with a subject-agreement prefix and the sequential suffix documented in Visser (2022, §on verbal morphology).

- **Arm 1 (wordlist + 400 parallel sentences).** The wordlist has *toni*; three of the 400 parallel sentences carry the same prefix on other roots. Model output: correct lexical content, correct tense by analogy with those three sentences. chrF ≈ 47.
- **Arm 3 (full grammar added).** Output identical in content, marginally different wording. chrF ≈ 48.

Naive reading: the grammar added +1. Now measure the confound. The grammar's morphology chapter contains **41 glossed examples** using that exact prefix, several sharing three or more content words with the test sentence. So $\rho$ for this item is high: the +1 is consistent with the model having retrieved a near-neighbour gloss, not with it having applied a rule.

Strip the examples (arm 2) and the prose alone says, in English, that the prefix marks a first-person subject. The model already inferred that from the three parallel sentences. The increment collapses to ≈ 0.

The obstruction is now visible and it is not about model capability: at $m = 100$ and one reference, the bootstrap CI on the +1 is roughly $\pm 3$ chrF. The measured effect is smaller than the measurement noise, and the only lever that shrinks the noise — more test sentences with more references — requires new field linguistics on a language with 200 speakers, not more compute.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*