---
id: 27-multilingual/code-switching-understanding
title: "Code-Switched Text Understanding"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Code-Switched Text Understanding

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/code-switching-understanding` · **Status:** open

## 1. Problem Statement

Code-switching (CS) is the alternation of two or more languages inside a single utterance, by fluent bilinguals, subject to grammatical constraints. The problem: build a model that understands such text as well as it understands each contributing language monolingually.

Three variants, usually conflated:

- **Measurement.** Given a model $f$ and a task $T$, quantify the *code-switching penalty* — the drop in $T$-performance attributable to switching rather than to domain, topic, dialect, transliteration, or corpus size. No standard estimator exists.
- **Method.** Given a pretraining corpus that is essentially monolingual-per-document, produce a model whose CS penalty is near zero without collecting large labelled CS corpora.
- **Theory.** Characterize when a model trained on languages $A$ and $B$ separately generalizes to the switched distribution $\mathcal{D}_{AB}$. Cross-lingual transfer theory says nothing about mixed-language inputs.

Solved would mean: for a task with a monolingual counterpart (NER, sentiment, QA, NLI), the CS penalty is within noise of zero on *naturalistic* CS data, and this holds for a language pair the model was not tuned on.

## 2. Formal Setting

Let $L=\{\ell_1,\ell_2\}$. A CS sentence is a token sequence $x=(x_1,\dots,x_n)$ with a per-token language tag $g=(g_1,\dots,g_n)$, $g_i\in\{\ell_1,\ell_2,\texttt{other},\texttt{ne},\texttt{ambig}\}$ — the tag set of the CALCS shared tasks (Molina et al., 2016). $g$ is **annotated**, not observed; inter-annotator agreement on `ambig` and `ne` is the weak link.

**Switch points.** $S(x)=\{i: g_i\neq g_{i+1},\; g_i,g_{i+1}\in\{\ell_1,\ell_2\}\}$, $s(x)=|S(x)|$.

**Code-Mixing Index** (Gambäck & Das, 2016), per utterance:
$$\mathrm{CMI}(x)=100\cdot\frac{n-\max_{\ell\in L} n_\ell + s(x)}{n},$$
with $n_\ell$ the token count of language $\ell$ and language-independent tokens excluded from $n$. Measured by counting tags; sensitive to whether punctuation, numerals and named entities are stripped — a choice that moves CMI by 5–15 points on the same corpus.

**M-index and I-index** (Guzmán et al., Interspeech 2017): $M=\frac{1-\sum_\ell p_\ell^2}{(k-1)\sum_\ell p_\ell^2}$ for $k$ languages with token shares $p_\ell$; $I = s(x)/(n-1)$, the switch rate. $M$ measures *balance*, $I$ measures *alternation*; corpora matched on one differ wildly on the other.

**CS penalty.** For task $T$ with metric $m$, model $f$:
$$\Delta_{\mathrm{CS}}(f,T)=\tfrac{1}{2}\big[m(f;\mathcal{D}_{\ell_1})+m(f;\mathcal{D}_{\ell_2})\big]-m(f;\mathcal{D}_{AB}).$$
Measuring this requires $\mathcal{D}_{\ell_1},\mathcal{D}_{\ell_2},\mathcal{D}_{AB}$ to differ *only* in switching. **Assumptions known to be violated in practice:**

1. *Matched domain/topic/authorship.* Violated: CS corpora are overwhelmingly social media (Hindi-English tweets, Spanish-English Twitter, SEAME conversational speech); monolingual references are news or Wikipedia.
2. *Consistent orthography.* Violated: Hindi-English CS is largely romanized with no spelling standard; the same word appears as *kya/kia/kyaa*. Script noise and switching are entangled.
3. *Switching is grammatical.* The Equivalence Constraint (Poplack, 1980) and Matrix-Language-Frame model (Myers-Scotton, 1993) predict switch points; real corpora contain performance errors and borrowings the constraints do not cover.
4. *Borrowing $\neq$ switching.* Violated by construction: single-word "switches" like *train*, *phone* in Hindi are established loans (Bali et al., CALCS 2014). CMI counts them as switches.

Non-identifiability: a measured $\Delta_{\mathrm{CS}}>0$ is a sum of switching effect, domain shift, script noise, and label noise. The decomposition is not identified by any existing dataset.

## 3. State of the Art

**Benchmarks (established as artifacts, not as measurements of switching).** LinCE (Aguilar, Kar, Solorio, LREC 2020) — LID, POS, NER, sentiment across Spanish-English, Nepali-English, Hindi-English, MSA-Egyptian. GLUECoS (Khanuja et al., ACL 2020) — Hi-En and Es-En, six tasks including NLI and QA. Both are leaderboards over held-out splits; neither ships a switching-matched monolingual control arm, so a leaderboard number is *not* an estimate of $\Delta_{\mathrm{CS}}$.

**Empirical SOTA.** Fine-tuned XLM-R / mBERT variants, sometimes with an intermediate stage on synthetic CS text. Khanuja et al. (2020) report that mBERT further pretrained on Equivalence-Constraint-generated synthetic CS data ("Modified mBERT") beats vanilla mBERT on most GLUECoS tasks — *established for Hi-En and Es-En, unablated for whether the gain comes from switching structure or simply from extra in-domain romanized text*.

**LLMs.** Zhang, Cahyawijaya, Cruz, Winata, Aji, *Multilingual Large Language Models Are Not (Yet) Code-Switchers* (EMNLP 2023): few-shot LLMs (including GPT-3.5-class) underperform fine-tuned smaller multilingual encoders on CS understanding tasks, while being reasonable at CS *generation* fluency judged by humans. Claimed but unablated: whether the deficit is CS-specific or the general few-shot-vs-fine-tuned gap on noisy social text.

**Generation.** Yong et al. (CALCS 2023) show LLMs prompted to produce code-mixed South-East Asian text generate output that native speakers reject as unnatural for most pairs except Singlish-adjacent ones. Established by human eval at small $n$; no automatic metric is validated against it.

**Theory SOTA.** Still the linguistic constraint models — Poplack's Equivalence Constraint and Sankoff & Poplack's (1981) formal grammar, Myers-Scotton's MLF. These are descriptive generalizations with known counterexamples, not learnability results. There is no PAC-style or transfer-theoretic statement about $\mathcal{D}_{AB}$ given $\mathcal{D}_{\ell_1},\mathcal{D}_{\ell_2}$.

## 4. What Is Known

- **CS is frequent in the wild.** Bali et al. (CALCS 2014) found roughly 17% of posts in a Hindi-English Facebook sample contained code-mixing, at the scale of a few thousand posts from Indian users.
- **Switch points are locally predictable.** Solorio & Liu (EMNLP 2008) predicted Spanish-English switch points above chance from lexical/POS features on ~40 minutes of transcribed bilingual conversation. Reproduced in spirit by every later LID shared task.
- **LID at the token level is largely solved for high-resource pairs.** CALCS 2016 shared task (Molina et al.) systems reached token-level accuracy in the mid-90s % on Spanish-English Twitter (~tens of thousands of tokens), while performance on `ambig`/`ne` classes stayed far lower.
- **Downstream CS tasks lag monolingual analogues.** LinCE NER for Spanish-English sits in the ~60s F1 for strong entries, against high-80s F1 for English CoNLL-03 NER — different data, so this is an *upper bound on the gap's plausibility*, not a measured $\Delta_{\mathrm{CS}}$.
- **Synthetic CS data helps language modelling.** Pratapa et al. (ACL 2018) showed Equivalence-Constraint-generated Es-En text lowers perplexity on real CS text relative to naive random mixing, on Bangor-Miami-scale corpora (~10^5 tokens of real CS).
- **Speech is harder.** SEAME (Lyu et al., 2010; ~192 hours Mandarin-English) mixed error rates for modern systems remain well above monolingual WER on either language.

## 5. What Is Not Known

- **Methodologically blocked.** Whether $\Delta_{\mathrm{CS}}$ is nonzero *at all* once domain, script, and annotation noise are controlled. No dataset exists in which the same content, same authors, same register appears in switched and unswitched form. Every published "code-switching is hard" number is confounded.
- **Methodologically blocked.** Whether CMI/M/I measure difficulty. They correlate with corpus identity, hence with domain; no study has shown that holding a model and domain fixed, performance decreases monotonically in CMI.
- **Empirically open.** Whether pretraining-scale exposure to naturalistic CS (rather than synthetic) closes the gap. Runnable: continue-pretrain a 7B model on $10^9$ tokens of filtered CS web text. Nobody has published this with a matched-compute monolingual-mixture control.
- **Empirically open.** Whether the LLM deficit reported by Zhang et al. (2023) persists for 2025–2026 frontier models under fine-tuning rather than prompting.
- **Theoretically open.** Any guarantee of the form: if $f$ has risk $\epsilon$ on $\mathcal{D}_{\ell_1}$ and $\mathcal{D}_{\ell_2}$ and the switching process satisfies constraint $C$, then risk on $\mathcal{D}_{AB}$ is bounded by $h(\epsilon, C)$. No such theorem exists, and no negative result either.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus absent ground truth for the confounder**.

1. CS text is produced in informal registers. Any CS-vs-monolingual comparison also compares Twitter to Wikipedia. The switching effect is not separable from the register effect with observational data.
2. The label $g_i$ that defines switching is itself contested. Borrowing versus switching has no operational test annotators agree on; Bali et al. framed the corpus paper around exactly this. So $\mathrm{CMI}$ — the independent variable in every difficulty claim — has annotation-dependent variance comparable to the effects being measured.
3. Romanization collapses the script signal. For Hi-En, En-Ur, Ar-En dialectal CS, subword tokenizers trained on canonical scripts fragment romanized tokens; measured "CS difficulty" partly reports tokenizer mismatch. Fertility (subwords per word) is rarely reported alongside CS results.
4. Constructing the missing control is expensive: it requires bilinguals producing paired switched/unswitched versions of the same content, which is elicitation, not scraping, and elicited CS is known to differ from spontaneous CS.

Compute is *not* the bottleneck. Data design is.

## 7. Current Research (as of 2026)

- **Benchmark consolidation and critique.** Winata, Aji, Yong, Solorio (Findings of ACL 2023) survey the field's trend line and argue most CS work is concentrated in a handful of pairs; Doğruöz, Sitaram, Bullock, Toribio (ACL 2021) argue NLP CS evaluation ignores sociolinguistic validity. Both point to the control-arm gap without closing it.
- **LLM code-mixing evaluation and prompting** — Aji/Winata-affiliated groups (SEACrowd/AISingapore orbit), Microsoft Research India (Sitaram, Choudhury lineage) on Indic CS. *(frontier — verify current 2026 outputs.)*
- **Synthetic CS generation beyond the Equivalence Constraint**, using LLMs as the generator with linguistic constraints as filters. Evaluation of naturalness remains human-only. *(frontier — verify.)*
- **CS speech and speech-LLMs**: SEAME and Indic CS ASR with foundation speech models; mixed error rate remains the reported metric. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Is there a code-switching penalty after domain and orthography are controlled?

**Scale.** Elicit a *paired* corpus: 3,000 sentences from ~60 Hindi-English bilinguals and ~60 Spanish-English bilinguals. Each participant writes each item twice — once naturally code-switched, once in a single language — same content, same author, same register, same session. Annotate NER and sentiment on both versions (double-annotated, report Cohen's $\kappa$). Budget: roughly 6,000 paired items, 2 annotators, on the order of a few hundred annotator-hours.

**Arms.**
- *Treatment:* CS version, evaluated with XLM-R-large fine-tuned on LinCE + a 2026 frontier LLM fine-tuned identically.
- *Control 1 (the key arm):* the monolingual version of the *same* items, same models, same annotation protocol.
- *Control 2:* a script control — CS items with the Hindi span in Devanagari versus romanized, to split orthography from switching.

**Deciding number.** The paired difference in NER F1, $\hat\Delta_{\mathrm{CS}} = F_1(\text{mono}) - F_1(\text{CS})$, with a 95% bootstrap CI over items.
- $\hat\Delta_{\mathrm{CS}} \le 2$ F1 with CI excluding 5 ⟹ the published CS gap is domain and script, not switching; the field should redirect to romanization and register robustness.
- $\hat\Delta_{\mathrm{CS}} \ge 8$ F1 with CI excluding 3 ⟹ switching itself is the difficulty, and $\Delta_{\mathrm{CS}}$ becomes a legitimate target.

Secondary: regress per-item $\Delta$ on CMI. A slope indistinguishable from zero falsifies CMI as a difficulty measure.

## 9. Key References

- **[Foundational]** Shana Poplack. *Sometimes I'll start a sentence in Spanish y termino en español: toward a typology of code-switching.* Linguistics, 1980.
- **[Foundational]** David Sankoff, Shana Poplack. *A Formal Grammar for Code-Switching.* Papers in Linguistics, 1981.
- **[Foundational]** Carol Myers-Scotton. *Duelling Languages: Grammatical Structure in Codeswitching.* Oxford University Press, 1993.
- **[Foundational]** Thamar Solorio, Yang Liu. *Learning to Predict Code-Switching Points.* EMNLP, 2008.
- **[Data]** Kalika Bali, Jatin Sharma, Monojit Choudhury, Yogarshi Vyas. *"I am borrowing ya mixing?" An Analysis of English-Hindi Code Mixing in Facebook.* CALCS workshop, EMNLP, 2014.
- **[Data]** Dau-Cheng Lyu, Tien-Ping Tan, Eng Siong Chng, Haizhou Li. *SEAME: a Mandarin-English code-switching speech corpus in South-East Asia.* Interspeech, 2010.
- **[Metrics]** Björn Gambäck, Amitava Das. *Comparing the Level of Code-Switching in Corpora.* LREC, 2016.
- **[Metrics]** Gualberto Guzmán, Joseph Ricard, Jacqueline Serigos, Barbara Bullock, Almeida Jacqueline Toribio. *Metrics for Modeling Code-Switching Across Corpora.* Interspeech, 2017.
- **[Benchmark]** Gustavo Aguilar, Sudipta Kar, Thamar Solorio. *LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation.* LREC, 2020.
- **[Benchmark]** Simran Khanuja, Sandipan Dandapat, Anirudh Srinivasan, Sunayana Sitaram, Monojit Choudhury. *GLUECoS: An Evaluation Benchmark for Code-Switched NLP.* ACL, 2020.
- **[Method]** Adithya Pratapa, Gayatri Bhat, Monojit Choudhury, Sunayana Sitaram, Sandipan Dandapat, Kalika Bali. *Language Modeling for Code-Mixing: The Role of Linguistic Theory based Synthetic Data.* ACL, 2018.
- **[SOTA]** Ruochen Zhang, Samuel Cahyawijaya, Jan Christian Blaise Cruz, Genta Indra Winata, Alham Fikri Aji. *Multilingual Large Language Models Are Not (Yet) Code-Switchers.* EMNLP, 2023.
- **[Survey]** A. Seza Doğruöz, Sunayana Sitaram, Barbara E. Bullock, Almeida Jacqueline Toribio. *A Survey of Code-switching: Linguistic and Social Perspectives for Language Technologies.* ACL, 2021.
- **[Survey]** Genta Indra Winata, Alham Fikri Aji, Zheng Xin Yong, Thamar Solorio. *The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges.* Findings of ACL, 2023.
- **[Shared task]** Giovanni Molina et al. *Overview for the Second Shared Task on Language Identification in Code-Switched Data.* CALCS workshop, EMNLP, 2016.

## 10. Worked Example

Take one Hi-En sentence from the social-media register:

> `mujhe kal ka meeting cancel karna hai but manager ne abhi tak reply nahi kiya`

Tokens $n=17$. Tag by language: Hindi = {mujhe, kal, ka, karna, hai, ne, abhi, tak, nahi, kiya} = 10; English = {meeting, cancel, but, manager, reply} = 5; language-independent/other = 2 (treat `to`-less; here assume 2 numerals/punctuation stripped).

With $n_{\text{eff}}=15$, $\max_\ell n_\ell = 10$, and counting maximal-span switches $s=4$:
$$\mathrm{CMI}=100\cdot\frac{15-10+4}{15}=60.0.$$

Now apply the borrowing test. *meeting*, *cancel*, *manager*, *reply* are all established loans in colloquial Hindi — they appear in monolingual Hindi speech with Hindi light-verb constructions (*cancel karna*). Reclassify them as Hindi borrowings. Then English = {but} = 1, $\max_\ell n_\ell=14$, $s=2$:
$$\mathrm{CMI}=100\cdot\frac{15-14+2}{15}=20.0.$$

**The same sentence scores 60 or 20 depending on one annotation decision with no agreed operational test.** A corpus-level difficulty claim of the form "F1 drops 9 points as CMI rises from 20 to 60" is therefore comparing two annotation conventions as much as two linguistic conditions.

Second half of the example: run an NER model on both surface forms. Romanized, XLM-R's tokenizer fragments *mujhe/karna/kiya* into 3–4 subwords each (fertility ≈ 2.8 subwords/word); rewrite the Hindi spans in Devanagari and fertility drops to ≈ 1.6. If model F1 improves on the Devanagari version, the "code-switching penalty" just measured was partly tokenizer fertility — with $s$ unchanged. That is the obstruction: the independent variable is unstable and the dependent variable moves for reasons that are not switching.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*