---
id: 27-multilingual/morphological-generalization-agglutinative
title: "Morphological Generalization in Agglutinative Languages"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Morphological Generalization in Agglutinative Languages

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/morphological-generalization-agglutinative` · **Status:** open

## 1. Problem Statement

Agglutinative languages (Turkish, Finnish, Hungarian, Korean, Basque, Swahili, Quechua, Telugu, Georgian) build words by concatenating largely invariant morphemes. Paradigm size is effectively unbounded: a Turkish verb admits on the order of $10^4$–$10^6$ inflected forms; Finnish nouns ~2,000 including clitics. Any finite corpus therefore covers a vanishing fraction of the licit word forms, and the test distribution is dominated by forms never seen in training.

**The problem.** Given a model trained on text (or on lemma–tag–form triples) from an agglutinative language, produce and interpret morphologically *novel* words — combinations of a known stem with a tag bundle whose surface realization was never observed — at the accuracy a competent speaker reaches.

Three variants that are routinely conflated:

- **Measurement.** Define a test set whose items are genuinely novel *as morphology*, not merely as strings. Requires a split that controls stem overlap, tag-bundle overlap, and suffix-sequence overlap at once, plus a tokenizer-independent scoring rule.
- **Method.** Build a system whose accuracy on that set does not collapse relative to i.i.d. held-out text. Success predicate: accuracy gap $\Delta \le 5$ points between i.i.d. and morphologically-disjoint splits at matched data scale.
- **Theory.** Characterize which inductive biases make the compositional map from tag sequences to surface strings learnable from $n$ paradigm cells, under phonology (vowel harmony, consonant gradation, assimilation) that makes the map non-concatenative at the surface.

Solving it means the method variant holds across at least five typologically distinct agglutinative languages, including at least one with productive consonant gradation (Finnish) and one with non-suffixing morphology (Swahili prefixes, Georgian circumfixes).

## 2. Formal Setting

Let $\Sigma$ be the grapheme alphabet, $\mathcal{L} \subset \Sigma^*$ the lemma set, and $\mathcal{T}$ the set of morphosyntactic tag bundles (UniMorph schema). The inflection map is
$$f: \mathcal{L} \times \mathcal{T} \to \Sigma^*, \qquad f(\ell, t) = w.$$

A dataset is $D = \{(\ell_i, t_i, w_i)\}_{i=1}^n$. Write $L(D)$, $T(D)$ for the lemmas and tag bundles occurring in $D$, and $S(D)$ for the set of observed *affix sequences*, obtained by aligning $w$ to $\ell$ with a minimal edit script and reading off the residual suffix/prefix string decomposed by a gold segmenter.

**Generalization gap.** With train $D_{\text{tr}}$ and two test sets, define exact-match accuracy $A(D) = \frac{1}{|D|}\sum_i \mathbb{1}[\hat f(\ell_i,t_i) = w_i]$ and
$$\Delta = A(D_{\text{iid}}) - A(D_{\text{novel}}),$$
where $D_{\text{novel}}$ satisfies $L(D_{\text{novel}}) \cap L(D_{\text{tr}}) = \emptyset$ **or** $T(D_{\text{novel}}) \cap T(D_{\text{tr}}) = \emptyset$, and additionally $S(D_{\text{novel}}) \not\subseteq S(D_{\text{tr}})$ for the *unseen-combination* condition. $\Delta$ is the headline quantity; it is measured, not estimated.

**Paradigm coverage.** For a corpus $C$ of $N$ tokens,
$$\kappa(C) = \frac{|\{(\ell,t) : f(\ell,t) \text{ occurs in } C\}|}{\sum_{\ell \in \mathcal{L}} |\mathcal{T}_\ell|},$$
computed against a finite-state morphological analyzer that enumerates $\mathcal{T}_\ell$. For Turkish with a full verbal paradigm, $\kappa < 10^{-4}$ at $N = 10^9$.

**Tokenizer-induced confound.** For a subword tokenizer $\tau$, define morpheme-boundary recall
$$R_\tau = \frac{|B_\tau \cap B_{\text{gold}}|}{|B_{\text{gold}}|},$$
where $B$ are boundary positions. Any comparison of $\Delta$ across languages must report $R_\tau$, because $\Delta$ and $R_\tau$ are confounded.

**Assumptions, and which are violated.**
1. *Morphemes concatenate.* Violated: Finnish gradation (`katu` → `kadun`), Turkish vowel harmony, Korean irregular verb classes. The map is not a free monoid homomorphism.
2. *Gold tag bundles are language-independent.* Violated: UniMorph coverage and granularity differ per language; Turkish `-mIş` is annotated inconsistently as evidential vs. perfect across resources.
3. *Lemma-disjoint ⇒ morphologically novel.* Violated when lemmas share derivational stems (`göz` / `gözlük`), which leaks the phonological environment.
4. *Exact match is the right loss.* Violated where orthographic variants are both licit (Hungarian `-ban`/`-ba` in speech-influenced text).

## 3. State of the Art

**Systems/empirical SOTA.** Character-level transducers — the MED encoder–decoder line (Kann & Schütze, SIGMORPHON 2016) and the character transformer of Wu et al. (NAACL 2021) — remain the strongest inflection systems. On SIGMORPHON–UniMorph shared tasks with random splits, top systems exceed 90% exact match on Turkish, Finnish and Hungarian. **Established**: these numbers reproduce across teams and years. **Established but often misread**: the same architectures lose large amounts of accuracy under lemma-disjoint splits (Goldman, Guriel & Tsarfaty, ACL 2022).

**Shared-task SOTA under hard splits.** SIGMORPHON 2022 Shared Task 0 (Kodner, Khalifa et al.) explicitly partitioned test items into seen-lemma/seen-feature, unseen-lemma, unseen-feature, and both-unseen. The both-unseen cell is where every system degrades most; the 2023 iteration (Goldman et al.) reported the same ordering on a typologically broader set. These are **benchmark numbers only** — no controlled ablation isolates *why* the both-unseen cell is hard (data sparsity vs. inductive bias).

**LLM SOTA.** Large multilingual LLMs handle high-frequency Turkish/Finnish inflection well but are **claimed, not ablated**, to generalize morphologically: reported gains are not separated from memorization of the exact form in pretraining, and pretraining corpora are not searchable for most models. Treat all "GPT-class models do morphology" claims as unverified for nonce stems.

**Theory SOTA.** Weaker. Finite-state morphology (Koskenniemi two-level; Oflazer's Turkish analyzer, 1994) gives an exact, hand-built $f$ with provable coverage but no learning guarantee. There is no PAC-style sample-complexity result for learning $f$ over an unbounded paradigm under phonological alternation.

## 4. What Is Known

- **Lemma overlap inflates scores.** Goldman et al. (ACL 2022) showed that standard SIGMORPHON splits share lemmas between train and test; re-splitting by lemma drops average exact-match accuracy sharply — tens of points on languages with large paradigms, at the ~10k-example-per-language scale of the shared tasks.
- **Copying is not free.** Liu & Hulden (ACL 2021) showed transformers on inflection fail to copy long or unseen stem material unless copying bias is explicitly tuned (hallucinated data, or attention/regularization changes). Measured on SIGMORPHON-scale datasets.
- **Morphological complexity costs perplexity.** Cotterell et al. (NAACL 2018) and Mielke et al. (2019) measured, on a 21-language multi-text Bible corpus, that morphologically rich languages are harder to model at matched content; Park et al. (TACL 2021) replicated the direction across 92 languages.
- **Tokenizer quality drives much of the deficit.** Rust et al. (ACL 2021) showed monolingual tokenizers close a large share of the mBERT-vs-monolingual gap; Arnett & Bergen (COLING 2025) argued the morphological-complexity penalty largely disappears once dataset size in *bytes* is matched — i.e., part of the "morphology is hard" effect is a tokenization/data-budget artifact, not a modeling one.
- **Segmentation helps downstream in places.** Bostrom & Durrett (Findings of EMNLP 2020) found unigram-LM tokenization aligns better with morphology than BPE and improves downstream tasks; Hofmann et al. (ACL 2021) showed derivationally-aware segmentation improves BERT on complex words (English derivation, not agglutination).
- **Data augmentation is the reliable low-resource lever.** Anastasopoulos & Neubig (EMNLP 2019) reported large gains from hallucinated-stem augmentation plus multi-task training at 100–1,000 training examples per language.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted definition of a *morphologically novel* test item. Lemma-disjoint, tag-disjoint, and suffix-sequence-disjoint splits give different rankings; no benchmark controls all three at once with a gold segmenter. Until this is fixed, $\Delta$ is not comparable across papers or languages.
- **Methodologically blocked.** Pretraining-corpus contamination cannot be checked for closed LLMs, so no LLM morphological-generalization number is currently interpretable. Nonce-stem (wug) protocols partly route around this but have no standard scoring for agglutinative multi-suffix forms.
- **Empirically open.** Whether $\Delta \to 0$ with scale. Nobody has run a clean scaling sweep — say $10^7$ to $10^{10}$ Turkish tokens at 4–5 model sizes — measuring $\Delta$ on a fixed nonce-stem set with a fixed tokenizer. This is runnable today for well under 10k GPU-hours.
- **Empirically open.** Whether byte-level or morphologically-informed tokenization reduces $\Delta$ specifically, as opposed to reducing perplexity generally. The two are usually reported together and never separated.
- **Theoretically open.** No sample-complexity bound for learning a concatenative-plus-alternation morphology from $n$ observed paradigm cells; no proof that the transformer function class can represent unbounded suffix-chain composition with harmony at fixed depth (the finite-precision circuit results bear on this but do not settle it).

## 6. Why It Is Hard

The binding obstruction is **confounded measurement**, not compute.

Measured accuracy on an agglutinative test set mixes at least four causes that current benchmarks do not separate: (i) the string was memorized in pretraining; (ii) the tokenizer happened to place a boundary at a morpheme edge, so the task reduced to lookup; (iii) the model composed suffixes correctly; (iv) the gold annotation is right. Absent ground truth on (i) for LLMs and on (ii) except via a gold segmenter that exists for only a handful of languages, a reported number cannot be attributed. Two papers can report the same accuracy on the same language with opposite claims about compositionality and neither is checkable.

Secondary: exact match over a $10^5$-cell paradigm is a heavy-tailed target. Most cells never occur; frequency-weighted evaluation measures memorization, and uniform-over-cells evaluation measures a distribution no speaker produces. There is no principled weighting, so the evaluation does not measure the thing it names.

## 7. Current Research (as of 2026)

- **Harder splits.** The SIGMORPHON–UniMorph line (Kodner, Khalifa, Goldman, Vylomova, Cotterell and collaborators) has moved from random to feature/lemma-disjoint partitions; further partitioning by affix-sequence novelty is the natural next step. *(frontier — verify)*
- **Tokenizer-free modeling.** Byte- and character-level architectures (ByT5, CANINE, and successor byte-latent designs) are the main methodological bet for removing the tokenization confound. Whether they reduce $\Delta$ on agglutinative morphology specifically is unresolved. *(frontier — verify)*
- **Morphology-aware tokenizer evaluation.** Work scoring tokenizers against gold morpheme boundaries (MorphyNet, UniMorph-derived segmentations) is growing; the metric is not yet standard. *(frontier — verify)*
- **Cognitive-model framing.** Corkery, Matusevych & Goldwater (ACL 2019) and Kirov & Cotterell (TACL 2018) argue over whether seq2seq models match human wug behavior — the same measurement problem, with human data as the reference. Extension to agglutinative languages is thin.
- **Language-specific NLP groups** (Boğaziçi/METU for Turkish, Turku for Finnish, HUN-REN for Hungarian, SNU/KAIST for Korean) maintain the analyzers that make gold segmentation possible; these resources are the bottleneck asset.

## 8. Concrete Next Experiment

**Question.** Does morphological generalization gap $\Delta$ shrink with pretraining scale, or is it flat?

**Scale.** Turkish and Finnish. Pretrain decoder-only LMs at 4 sizes (70M, 350M, 1.4B, 6.7B) × 3 data budgets (1B, 10B, 100B tokens), byte-level tokenizer. ~24 runs; roughly 3–8k A100-hours total. Everything is public data (OSCAR/CulturaX/mC4 Turkish + Finnish).

**Task.** Nonce-stem inflection generated from Oflazer's Turkish FST and Omorfi for Finnish: 5,000 nonce stems that are phonotactically legal and absent from the corpus (verified by exact substring search over the training shards — this is the contamination control that closed LLMs cannot offer). Each stem is inflected into 20 cells spanning 1–5 suffixes, including harmony-triggering and gradation-triggering cells. Score exact match, few-shot with 8 in-context real-stem examples per cell type.

**Control arms.**
1. **Same models, real stems** matched for frequency band — isolates the novelty effect from task difficulty.
2. **BPE tokenizer** at every scale point — isolates tokenization from scale.
3. **Suffix-shuffled ablation**: the same cells with tag bundles randomly permuted (illegal orders). A model that composes should reject these; a lookup model is indifferent.

**Deciding number.** $\Delta_{\text{5-suffix}} = A(\text{real stem, 5 suffixes}) - A(\text{nonce stem, 5 suffixes})$, plotted against pretraining tokens. If $\Delta_{\text{5-suffix}}$ falls below **5 points** by 100B tokens at 6.7B parameters, scale solves it and the problem is empirically closed for these languages. If it stays above **20 points** and flat across the 100× data sweep, scale does not solve it, and the field should redirect to architectural or segmentation priors. Anything in between localizes the crossover.

## 9. Key References

- **[Foundational]** Jean Berko. *The Child's Learning of English Morphology.* Word, 1958. — the wug test; the protocol every novel-form evaluation descends from.
- **[Foundational]** Kemal Oflazer. *Two-level Description of Turkish Morphology.* Literary and Linguistic Computing, 1994.
- **[Foundational]** Ryan Cotterell, Sabrina J. Mielke, Jason Eisner, Brian Roark. *Are All Languages Equally Hard to Language-Model?* NAACL-HLT, 2018.
- **[SOTA]** Katharina Kann, Hinrich Schütze. *MED: The LMU System for the SIGMORPHON 2016 Shared Task on Morphological Reinflection.* SIGMORPHON @ ACL, 2016.
- **[SOTA]** Shijie Wu, Ryan Cotterell, Mans Hulden. *Applying the Transformer to Character-level Transduction.* EACL, 2021.
- **[SOTA]** Omer Goldman, David Guriel, Reut Tsarfaty. *(Un)solving Morphological Inflection: Lemma Overlap Artificially Inflates Models' Performance.* ACL, 2022.
- **[SOTA]** Ling Liu, Mans Hulden. *Can a Transformer Pass the Wug Test? Tuning Copying Bias in Word Inflection Tasks.* ACL (short), 2021.
- **[Benchmark]** Jordan Kodner, Salam Khalifa, Khuyagbaatar Batsuren et al. *SIGMORPHON–UniMorph 2022 Shared Task 0: Generalization and Typologically Diverse Morphological Inflection.* SIGMORPHON, 2022.
- **[Benchmark]** Khuyagbaatar Batsuren et al. *UniMorph 4.0: Universal Morphology.* LREC, 2022.
- **[Benchmark]** Khuyagbaatar Batsuren, Gábor Bella, Fausto Giunchiglia. *MorphyNet: a Large Multilingual Database of Derivational and Inflectional Morphology.* SIGMORPHON, 2021.
- **[Method]** Antonios Anastasopoulos, Graham Neubig. *Pushing the Limits of Low-Resource Morphological Inflection.* EMNLP, 2019.
- **[Method]** Kaj Bostrom, Greg Durrett. *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP, 2020.
- **[Method]** Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, Iryna Gurevych. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL, 2021.
- **[Analysis]** Catherine Arnett, Benjamin Bergen. *Why Do Language Models Perform Worse for Morphologically Complex Languages?* COLING, 2025.
- **[Analysis]** Hyunji Hayley Park, Katherine J. Zhang, Coleman Haley, Kenneth Steimel, Han Liu, Lane Schwartz. *Morphology Matters: A Multilingual Language Modeling Analysis.* TACL, 2021.
- **[Analysis]** Maria Corkery, Yevgen Matusevych, Sharon Goldwater. *Are We There Yet? Encoder-Decoder Neural Networks as Cognitive Models of English Past Tense Inflection.* ACL, 2019.
- **[Survey]** Sabrina J. Mielke, Zaid Alyafeai, Elizabeth Salesky et al. *Between Words and Characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP.* 2021.

## 10. Worked Example

Take the Turkish nonce verb stem **`sürel-`** (phonotactically legal, front-unrounded harmony class, not a Turkish word). Build the causative-passive-negative-past-3pl form:

$$\texttt{sürel} + \texttt{-t} + \texttt{-il} + \texttt{-me} + \texttt{-di} + \texttt{-ler} \;\to\; \textbf{süreltilmediler}$$

Every suffix choice is conditioned: `-t` (causative) rather than `-dIr` because the stem ends in a liquid after a polysyllable; `-il` rather than `-in` because the preceding segment is `t`; `-me` and `-di` take front unrounded vowels by harmony; `-di` stays voiced because `e` is a vowel. Six decisions, five of them phonologically conditioned. A competent speaker makes all six on a stem heard once.

Now the obstruction, in numbers. Tokenize the target with a standard 32k multilingual BPE vocabulary: the string splits into pieces like `sü` / `rel` / `til` / `med` / `iler` — 5 pieces, of which **0** align with the 6 gold morpheme boundaries at positions {5,6,8,10,12}. Boundary recall $R_\tau = 0/6 = 0$. The model is not being asked to compose morphemes; it is being asked to emit a specific 5-piece string it has never seen, where the pieces cut across every morpheme edge.

Suppose the model scores 0 on this item and 1 on the real stem `görüldüler`. Which of the following is the cause?

| Hypothesis | Predicts real=1, nonce=0 | Distinguishable with current benchmarks |
|---|---|---|
| Memorized `görüldüler` from pretraining | yes | no — corpus not searchable |
| BPE destroyed the boundaries | yes | only with a gold segmenter |
| No compositional suffix-ordering ability | yes | no — not separately probed |
| Harmony rule not learned | yes | only with a harmony-controlled minimal pair |

All four predict the same observation. That is the problem: the headline number — say "78% exact match on Turkish inflection" — is consistent with a model that composes morphology and with a model that does pure lookup. The experiment in §8 breaks the tie by making the corpus searchable (open pretraining data), fixing the tokenizer to bytes ($R_\tau = 1$ by construction), and reading $\Delta$ at suffix depth 5, where lookup and composition finally diverge.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*