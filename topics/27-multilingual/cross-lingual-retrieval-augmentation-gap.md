---
id: 27-multilingual/cross-lingual-retrieval-augmentation-gap
title: "Retrieval Augmentation When the Corpus Language Differs"
topic: 27-multilingual
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Retrieval Augmentation When the Corpus Language Differs

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/cross-lingual-retrieval-augmentation-gap` · **Status:** partially-solved

## 1. Problem Statement

A user asks in language $\ell_q$. The evidence that answers the question exists only in language $\ell_c \neq \ell_q$ — typically English, because English Wikipedia has ~7M articles against ~130k for Swahili. The system must retrieve in $\ell_c$, condition a generator on $\ell_c$ passages, and answer in $\ell_q$. The question is how much utility is lost relative to the monolingual English pipeline, and where the loss sits.

Three variants, routinely conflated:

- **Measurement.** Define a cross-lingual RAG gap that is attributable — separating retrieval failure from cross-lingual reading failure from answer-language failure. Current benchmarks report one end-to-end number and cannot tell these apart.
- **Method.** Close the gap. Candidate levers: multilingual dense retrievers, query translation, document translation, cross-lingual reranking, generator prompting in the pivot language.
- **Theory.** Under what conditions does a single embedding space of dimension $d$ support both within-language semantic ranking and cross-language alignment without one degrading the other? No non-trivial result exists.

Solved would mean: for a held-out low-resource $\ell_q$ with no in-language corpus, end-to-end answer accuracy within 5 points of the English-query/English-corpus control, at equal retrieval depth $k$ and equal generator.

## 2. Formal Setting

Corpus $\mathcal{C} = \{p_1,\dots,p_N\}$ with a language label $\lambda(p_i)$. Query $q$ in language $\ell_q$; gold answer set $A^\star(q)$ expressed in language $\ell_a$. Retriever $R_k(q;\mathcal{C})$ returns $k$ passages; generator $G$ produces $\hat{a} = G(q, R_k(q))$.

**Measured quantities.**

- Retrieval quality: $\text{nDCG@10}$ against human relevance judgments, and recall $\text{R@}k = \Pr[\exists p \in R_k(q): p \text{ is judged relevant}]$. In XOR-style settings the judgment is *answer-bearing*: the passage contains a string that normalizes to a gold answer, which over-counts (string match ≠ support) and under-counts (paraphrase).
- Answer quality: token-F1 or exact match after language-specific normalization; for generative multilingual answers, $\text{F1}$ is computed in $\ell_a$ and is not comparable across scripts with different tokenization granularity.
- Answer-language compliance: $\text{LC} = \Pr[\text{lang-id}(\hat{a}) = \ell_q]$, measured with an off-the-shelf identifier whose own error rate on short strings is a nuisance term of several points.

**The gap.** With $U$ any of the above utilities,

$$\Delta_{\mathrm{CL}}(\ell_q) \;=\; U(\ell_q{=}\mathrm{en},\ \ell_c{=}\mathrm{en}) \;-\; U(\ell_q{=}\ell,\ \ell_c{=}\mathrm{en}).$$

**Decomposition.** Insert an oracle retriever $R^\star$ (gold passage always at rank 1):

$$\Delta_{\mathrm{CL}} \;=\; \underbrace{\big[U^{\mathrm{en}}_{R}-U^{\ell}_{R}\big]_{U \text{ held at } G^\star}}_{\Delta_{\text{ret}}} \;+\; \underbrace{\big[U^{\mathrm{en}}_{R^\star}-U^{\ell}_{R^\star}\big]}_{\Delta_{\text{gen}}} \;+\; \varepsilon_{\text{int}},$$

with $\varepsilon_{\text{int}}$ the non-additive interaction (a generator that reads $\ell_c$ poorly is also more sensitive to retrieval noise). $\varepsilon_{\text{int}}$ is almost never reported.

**Assumptions, and where they break.**

1. *Answer existence in $\ell_c$.* Assumed; violated for culturally local questions, where the English corpus simply lacks the fact. This makes $\Delta_{\mathrm{CL}}$ un-closeable in principle for that slice.
2. *Language-agnostic relevance.* Relevance is assumed independent of $\ell_q$ given meaning. Violated: for contested topics, the relevant answer differs by language community.
3. *Translation-invariant evaluation.* Violated — translated test sets carry translationese; queries translated from English are easier than native queries in the same language.
4. *Complete judgments.* Violated — MIRACL-style pools are shallow, and unjudged-but-relevant passages penalize systems unlike the pooled ones.

## 3. State of the Art

**Established (reproduced, ablated).**

- Hybrid sparse+dense retrieval beats either component in every multilingual benchmark it has been run on (Mr. TyDi, EMNLP MRL 2021; MIRACL, TACL 2023). This is robust across teams.
- Translate-query into English and retrieve monolingually is a strong baseline that many neural CLIR systems fail to beat once MT quality is decent (repeatedly observed in TREC NeuCLIR 2022–2023).
- Multilingual contrastive pretraining (mContriever, TMLR 2022; mE5, 2024; BGE-M3, ACL Findings 2024) improves zero-shot transfer to unseen languages over mDPR.

**Claimed but unablated.**

- That multilingual embedding models perform genuine *cross-lingual* matching rather than exploiting shared entity strings, numerals and code-switched fragments. The lexical-overlap control is rarely run.
- That English-pivot prompting is optimal for generation. Reported by Chirkova et al. (2024) on a limited language set with one retriever family; not replicated across generator scales.
- Instruction-following retrieval across languages (mFollowIR, ECIR 2025) shows sharp drops, but only for a handful of systems.

**Benchmark-number-only results.** MIRACL and NoMIRACL leaderboard entries, AfriQA baselines, and most multilingual-RAG numbers in model cards are single-configuration scores with no oracle-retrieval arm, so they do not license any claim about $\Delta_{\text{ret}}$ versus $\Delta_{\text{gen}}$.

## 4. What Is Known

- **MIRACL (TACL 2023), 18 languages, ~726k judgments.** BM25 averages nDCG@10 ≈ 0.39 over the 16 known-language splits; mDPR ≈ 0.41; the hybrid ≈ 0.53. Judged-relevance recall for BM25 at $k{=}100$ is well above 0.8 for high-resource languages and materially lower for Telugu, Bengali, Yoruba.
- **XOR-TyDi (NAACL 2021), 7 typologically diverse languages.** Cross-lingual retrieval against English Wikipedia recovers answers for a large fraction of questions unanswerable in-language — the original motivation — but end-to-end F1 sat far below English open-domain QA at the same scale.
- **CORA (NeurIPS 2021).** A single mDPR+mGEN model trained with cross-lingual data augmentation beat translate-test pipelines on XOR-TyDi, showing the gap is partly a *training-data* artifact, not an architectural limit.
- **Synthetic multilingual training data works.** SWIM-IR (NAACL 2024), ~28M LLM-generated query–passage pairs across 33 languages, improves dense retrieval on MIRACL without human annotation — evidence that annotation scarcity, not modeling, dominates the retrieval half.
- **Generators are not robust to irrelevant multilingual context.** NoMIRACL (2023) constructs a non-relevant subset; strong instruction-tuned LLMs answer confidently from unrelated passages at high rates rather than abstaining, and the rate varies by language.
- **Answer-language slippage is real.** Models conditioned on English passages drift into English answers unless instructed otherwise; an explicit output-language instruction recovers most of the compliance (Chirkova et al., 2024).

Scales: retrievers here are 100M–600M parameters; generators in the cited multilingual-RAG studies are 7B–70B plus GPT-4-class APIs.

## 5. What Is Not Known

- **Methodologically blocked.** The attribution $\Delta_{\mathrm{CL}} = \Delta_{\text{ret}} + \Delta_{\text{gen}} + \varepsilon_{\text{int}}$ is not measurable today: no benchmark ships parallel queries in $n$ languages against a *fixed* corpus with *shared* gold passages. Without that, retrieval and generation losses are confounded with query-set difficulty differences.
- **Methodologically blocked.** "Relevance" for questions where the answer is language-community-dependent has no agreed ground truth. Cross-lingual RAG on contested content (BordIRlines, 2024) shows the retrieved viewpoint tracks the query language, and there is no correct label.
- **Empirically open.** Does the cross-lingual gap shrink with generator scale at fixed retriever? Runnable today across a 1B→70B ladder; nobody has published the clean curve.
- **Empirically open.** How much of multilingual dense retrieval is lexical overlap? The control (entity-scrubbed, transliteration-normalized queries) is cheap and unrun at benchmark scale.
- **Theoretically open.** Whether a $d$-dimensional bi-encoder can be simultaneously optimal for within-language ranking over $L$ languages and cross-language alignment; no capacity lower bound analogous to the multilingual "curse of multilinguality" is proved for retrieval.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus absent ground truth on the same axis**. Every reported cross-lingual RAG number varies at least three things at once — query language, corpus composition, and question distribution — because multilingual QA sets are built per-language from native speakers, so the Swahili and English question sets are not the same questions. When they *are* the same questions (translated), translationese makes them easier and the gap shrinks for the wrong reason. Meanwhile the relevance judgments that would let you insert an oracle retriever exist only for pooled systems, so an oracle arm is unavailable exactly where the gap is largest (low-resource languages, shallow pools). Compute is not the binding constraint; annotation design is.

## 7. Current Research (as of 2026)

- **Synthetic cross-lingual training data at scale** — LLM-generated queries against non-English corpora, following SWIM-IR; now standard in open embedding releases (BGE, E5 lines).
- **Unified multi-functional retrievers** — BGE-M3-style models combining dense, sparse and multi-vector scoring in one checkpoint, motivated by the hybrid result in §4.
- **Abstention and grounding under multilingual noise** — NoMIRACL-derived evaluation of "knowing when you don't know" across languages *(frontier — verify)*.
- **Culturally-grounded and contested-evidence RAG** — measuring whether retrieved viewpoints track the query language (BordIRlines and follow-ons) *(frontier — verify)*.
- **Pivot-language interpretability** — whether generators internally translate to English before reasoning (Wendler et al., ACL 2024) and whether that explains the $\Delta_{\text{gen}}$ term *(frontier — verify)*.
- Groups: Waterloo (Lin, Thakur, Zhang — MIRACL/NoMIRACL/SWIM-IR), Naver Labs Europe (multilingual RAG), JHU HLTCOE (NeuCLIR, ColBERT-X), Masakhane (African-language QA).

## 8. Concrete Next Experiment

**Question:** is the cross-lingual RAG gap mostly retrieval or mostly generation?

**Build.** Take 1,000 MIRACL/XOR-TyDi questions whose gold supporting passage is in English Wikipedia. Have native speakers *author* — not translate — the same information need in 8 languages (en, fr, zh, ru, ar, bn, sw, te). Result: one fixed English corpus, one shared gold passage per question, 8 parallel query sets. Cost: ~8,000 annotations, a few weeks, well under one GPU-month of compute.

**Arms.** (a) mE5-large retriever + Llama-3-70B-Instruct; (b) same generator, **oracle retriever** (gold passage at rank 1) — this is the control that isolates $\Delta_{\text{gen}}$; (c) translate-query-to-English + English monolingual retriever; (d) lexical-overlap control: entity strings in queries replaced by language-neutral placeholders.

**Deciding number.** The ratio

$$\rho(\ell) \;=\; \frac{\Delta_{\text{gen}}(\ell)}{\Delta_{\mathrm{CL}}(\ell)} \;=\; \frac{F1^{\mathrm{en}}_{\text{oracle}} - F1^{\ell}_{\text{oracle}}}{F1^{\mathrm{en}}_{\text{real}} - F1^{\ell}_{\text{real}}}.$$

If $\rho < 0.3$ averaged over the four lower-resource languages, the gap is a retrieval problem and effort belongs in cross-lingual training data. If $\rho > 0.7$, retrievers are already adequate and the gap is the generator's cross-lingual reading and answer-language behaviour. Nobody has published $\rho$ for any language.

## 9. Key References

- **[Foundational]** Akari Asai, Jungo Kasai, Jonathan H. Clark, Kenton Lee, Eunsol Choi, Hannaneh Hajishirzi. *XOR QA: Cross-lingual Open-Retrieval Question Answering.* NAACL 2021. — arXiv:2010.11856
- **[Foundational]** Jonathan H. Clark et al. *TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages.* TACL 2020.
- **[Foundational]** Akari Asai, Xinyan Yu, Jungo Kasai, Hannaneh Hajishirzi. *One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval (CORA).* NeurIPS 2021. — arXiv:2107.11976
- **[SOTA]** Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, Jimmy Lin. *MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages.* TACL 2023. — arXiv:2210.09984
- **[SOTA]** Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings.* Findings of ACL 2024. — arXiv:2402.03216
- **[SOTA]** Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei. *Multilingual E5 Text Embeddings: A Technical Report.* 2024. — arXiv:2402.05672
- **[Method]** Nandan Thakur, Jianmo Ni, Gustavo Hernández Ábrego, John Wieting, Jimmy Lin, Daniel Cer. *Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval (SWIM-IR).* NAACL 2024. — arXiv:2311.05800
- **[Method]** Suraj Nair, Eugene Yang, Dawn Lawrie, Kevin Duh, Paul McNamee, Douglas W. Oard, James Mayfield. *Transfer Learning Approaches for Building Cross-Language Dense Retrieval Models (ColBERT-X).* ECIR 2022.
- **[Empirical]** Nadezhda Chirkova, David Rau, Hervé Déjean, Thibault Formal, Stéphane Clinchant, Vassilina Nikoulina. *Retrieval-Augmented Generation in Multilingual Settings.* KnowLLM Workshop, ACL 2024.
- **[Empirical]** Nandan Thakur, Luiz Bonifacio, Xinyu Zhang, Odunayo Ogundepo, Ehsan Kamalloo, et al. *NoMIRACL: Knowing When You Don't Know for Robust Multilingual Retrieval-Augmented Generation.* 2023.
- **[Empirical]** Odunayo Ogundepo et al. *AfriQA: Cross-lingual Open-Retrieval Question Answering for African Languages.* Findings of EMNLP 2023.
- **[Analysis]** Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL 2024. — arXiv:2402.10588
- **[Survey]** Yunfan Gao et al. *Retrieval-Augmented Generation for Large Language Models: A Survey.* 2023. — arXiv:2312.10997

## 10. Worked Example

Question, Telugu: *"అమెరికా అంతర్యుద్ధం ఎప్పుడు ముగిసింది?"* ("When did the American Civil War end?"). No Telugu Wikipedia article states the surrender date; English Wikipedia does, several times.

Pipeline: mE5-large over English Wikipedia, $k{=}5$, Llama-3-70B-Instruct.

- The gold passage ("Lee surrendered … April 9, 1865") is retrieved at rank 2. Retrieval succeeded.
- The generator answers **"April 9, 1865"** — in English, in Latin script. Language compliance fails.
- Scoring against the Telugu gold string "9 ఏప్రిల్ 1865" gives token-F1 = **0.33**: the numerals "9" and "1865" match after normalization, "ఏప్రిల్" vs "April" does not.
- Add the instruction "Answer in Telugu." The model emits "1865 ఏప్రిల్ 9". F1 = **1.0**.

The end-to-end number moved 0.33 → 1.00 with no change to retrieval, generator, or knowledge. Now note what the *unmodified* benchmark run reports: a 67-point gap versus the English control, which reads as a cross-lingual knowledge-transfer failure. It is a formatting-and-instruction failure. Without the oracle-retrieval arm and a compliance metric reported separately, the two are indistinguishable — and any method evaluated only on end-to-end F1 will be credited for fixing the wrong thing. That is the obstruction in §6 in one question.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*