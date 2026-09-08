---
id: 05-retrieval-and-agents/cross-lingual-retrieval-parity-low-resource
title: "Cross-Lingual Retrieval Parity for Low-Resource Languages"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Lingual Retrieval Parity for Low-Resource Languages

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/cross-lingual-retrieval-parity-low-resource` · **Status:** open

## 1. Problem Statement

Given a query in language $\ell$ and a corpus in language $\ell'$ (monolingual when $\ell=\ell'$, cross-lingual otherwise), a retriever must return documents ranked by relevance. **Parity** is the claim that retrieval quality does not depend on $\ell$ once query difficulty and corpus quality are held fixed. Today it does: on the same information need, systems score materially lower for Yoruba, Somali, Telugu or Swahili than for English or Chinese.

Three variants, routinely conflated:

- **Measurement variant.** Is the observed gap real, or an artifact of pooled judgments, tokenizer-dependent BM25 baselines, and translationese queries? Currently the binding constraint.
- **Method variant.** Build one embedding model that reaches within $\varepsilon$ of English nDCG@10 for $\ge 90\%$ of the languages in a fixed set, without per-language supervised retrieval data.
- **Theory variant.** Under a fixed parameter and vocabulary budget, is a parity-achieving multilingual representation possible at all, or does the *curse of multilinguality* impose a capacity/parity tradeoff with a nonzero floor?

Solving it means: a retriever, a test collection, and an ablation showing the residual per-language gap is within annotation noise.

## 2. Formal Setting

Languages $\mathcal{L}$, corpus $\mathcal{C}_\ell = \{d_1,\dots,d_{N_\ell}\}$, query set $\mathcal{Q}_\ell$. A dual encoder gives $s(q,d) = \langle f_\theta(q), g_\theta(d)\rangle$; retrieval returns the top-$k$ by $s$.

Per-language quality, measured exactly as the ranking is scored:

$$M_\ell = \frac{1}{|\mathcal{Q}_\ell|}\sum_{q\in\mathcal{Q}_\ell} \mathrm{nDCG}@10(\pi_q, r_q), \qquad \mathrm{nDCG}@10 = \frac{\sum_{i=1}^{10} \frac{2^{r_q(\pi_q(i))}-1}{\log_2(i+1)}}{\mathrm{IDCG}@10}$$

with $r_q(d)\in\{0,1\}$ from human judgments over a **pool** $P_q\subseteq\mathcal{C}_\ell$ of documents surfaced by a fixed set of contributing systems. Unjudged $\Rightarrow$ non-relevant, so $M_\ell$ is a *lower bound* whose slack grows with the fraction of unjudged top-10 documents (`judged@10`).

**Parity gap** against a reference language (English, $\ell_0$):

$$\Delta_\ell = M_{\ell_0} - M_\ell, \qquad \Delta_{\max} = \max_{\ell\in\mathcal{L}} \Delta_\ell, \qquad \bar\Delta = \frac{1}{|\mathcal{L}|}\sum_\ell \Delta_\ell .$$

Resource level is measured, not asserted: $R_\ell = \log_{10}(\text{pretraining tokens in }\ell)$ from the model's own corpus statistics, plus $T_\ell$, the count of supervised (query, positive) pairs in $\ell$. The empirical scaling claim under test is $\Delta_\ell \approx \alpha - \beta R_\ell$ for constants $\alpha,\beta>0$.

Assumptions, with the ones known to be violated marked:

1. Judgments are comparable across languages — **violated**: pool depth, assessor count and guidelines differ per language.
2. Queries express the same difficulty distribution across languages — **violated**: many low-resource query sets are translated from English, so they carry English topical structure and lack native-script code-switching.
3. Corpora are comparable — **violated**: Wikipedia in Yoruba has ~4 orders of magnitude fewer articles than English, so a relevant document may simply not exist.
4. $\mathrm{nDCG}@10$ with binary relevance is a valid utility proxy — untested for morphologically rich, low-resource languages where a single stem match may be decisive.

## 3. State of the Art

**Empirical SOTA.** Multilingual dense retrievers trained with large synthetic or mined supervision: multilingual-E5 (Wang et al., 2024), BGE-M3 (Chen et al., ACL Findings 2024), and mGTE-class models. BGE-M3 reports the best published MIRACL averages for an open model in its class, combining dense, sparse (lexical) and multi-vector scoring. Hybrid dense+BM25 fusion remains the strongest configuration on the low-resource tail — *established* across MIRACL, Mr. TyDi and CIRAL.

**Established.** (a) BM25 is not dominated on the tail: in the MIRACL paper (Zhang et al., TACL 2023) BM25 beats mDPR on several low-resource languages including Swahili, Telugu and Yoruba, while losing badly on English and Chinese. (b) Dense-only gains concentrate where pretraining data is abundant. (c) Translate-then-retrieve with NLLB-200 is a strong, frequently under-reported baseline for cross-lingual pairs.

**Claimed but unablated.** That synthetic-query generation (SWIM-IR, Thakur et al., NAACL Findings 2024) *closes* the low-resource gap: reported gains are aggregate benchmark numbers with no per-language ablation isolating synthetic-data volume from base-model pretraining share $R_\ell$. Similarly, MMTEB / MTEB multilingual leaderboard placements are benchmark numbers only — no controlled comparison of tokenizer fertility, pool depth, or corpus size across the compared systems.

**Theory SOTA.** No parity theorem. The nearest formal object is the capacity-vs-languages tradeoff described empirically as the *curse of multilinguality* (Conneau et al., ACL 2020), which is an observed regularity, not a bound.

## 4. What Is Known

- On MIRACL (18 languages, ~726k judgments, dev sets of hundreds of queries per language), the spread of nDCG@10 across languages for a single system exceeds 30 points; Yoruba and Telugu behave very differently from English under the same model. Scale: 100M–560M-parameter retrievers, corpora of $10^5$–$10^7$ passages.
- BM25 nDCG@10 varies by more than a factor of two across MIRACL languages purely with tokenization and morphology, independent of any neural component. Scale: same collection, sparse baseline.
- XOR-TyDi QA (Asai et al., NAACL 2021, 7 typologically diverse languages) established that cross-lingual answer retrieval is markedly harder than the monolingual case for the same questions — the gap is not only a language-modelling gap.
- Scaling the multilingual encoder past ~XLM-R-large size raises low-resource languages more than high-resource ones (Conneau et al., ACL 2020; measured at 550M parameters, 100 languages, 2.5TB CommonCrawl) — but the gap does not vanish.
- CIRAL (Ogundepo et al., FIRE 2023 evaluation track; Hausa, Somali, Swahili, Yoruba, English queries) shows cross-lingual English→African retrieval trails monolingual English retrieval by a wide margin under human judgments collected natively rather than by translation.

## 5. What Is Not Known

- **Methodologically blocked.** Whether $\Delta_\ell$ measures a system deficit or a collection deficit. `judged@10` is not reported per language on most leaderboards, and pools for low-resource languages were built from fewer and weaker contributing systems, so a genuinely better system is penalised for surfacing unjudged documents. Until pool bias is bounded per language, $\Delta_\ell$ is not identifiable.
- **Empirically open.** Whether per-language supervised data $T_\ell$ or pretraining share $R_\ell$ is the binding input. The 2-factor experiment (hold $R_\ell$ fixed, vary $T_\ell$ over 3 decades, per language) is runnable on <2k GPU-hours and has not been published.
- **Theoretically open.** Whether a fixed-capacity encoder admits a parity-achieving solution. No lower bound of the form "$\bar\Delta \ge c(|\mathcal{L}|, d, \theta)$" exists; equally, no construction proving $\bar\Delta\to 0$ is achievable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent ground truth**. Four causes of a low $M_\ell$ are not separable with current collections:

1. the retriever ranks badly;
2. the relevant document does not exist in $\mathcal{C}_\ell$ (corpus size varies by $10^4$);
3. the relevant document exists but was never pooled, so it scores 0;
4. the query is translationese and does not represent a real information need in $\ell$.

Causes 2–4 are properties of the collection. Any method paper that reports only $\Delta_\ell$ is attributing all four to cause 1. Fixing this needs native-speaker assessors per language — the cost scales with $|\mathcal{L}|$, not with compute, and is the reason the measurement has not been fixed by throwing GPUs at it.

## 7. Current Research (as of 2026)

- **Synthetic supervision at scale.** LLM-generated queries over native corpora (SWIM-IR lineage), now with per-language quality filtering. Google Research, Waterloo. Open question: do synthetic queries inherit English topical priors?
- **Sparse-dense unification.** BGE-M3-style multi-functional retrievers (BAAI) that keep an explicit lexical channel, motivated exactly by BM25's tail robustness.
- **Tokenizer fairness.** Work on per-language fertility (tokens per word) and its effect on effective context and embedding quality; vocabulary-expansion and byte-level retrievers *(frontier — verify)*.
- **Collection building.** MMTEB (Enevoldsen et al., 2025) broadened multilingual coverage; CIRAL continues African-language cross-lingual judgments. Both are the correct response to the measurement blockage.
- **Agentic multilingual RAG.** Retrieve-in-English-then-answer-in-$\ell$ pipelines with NLLB or LLM translation in the loop; strong in practice, rarely ablated against native retrieval *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** is the parity gap a system property or a pool artifact?

**Scale.** Six MIRACL/CIRAL languages spanning $R_\ell$: English, Chinese, Swahili, Telugu, Yoruba, Somali. 100 queries each (600 total). Corpus: existing MIRACL/CIRAL passage collections, unchanged.

**Procedure.** Run four retrievers (BM25, mDPR, multilingual-E5-large, BGE-M3) plus RRF fusion. Take the union of top-20 per system per query — roughly $600 \times 5 \times 20 \le 60{,}000$ documents, deduped to ~25k. Have **two native speakers per language** judge every unjudged document in that union under one shared guideline. Cost estimate: ~25k judgments at 30 s each ≈ 210 assessor-hours.

**Control arm.** Recompute the same runs against the *original* pools. The control is the identical system under the old judgments; the treatment is the identical system under the deepened pools.

**Deciding number.** The change in the mean parity gap,
$$\delta = \bar\Delta_{\text{old pools}} - \bar\Delta_{\text{deep pools}}.$$
If $\delta \ge 5$ nDCG@10 points, the published gap is substantially a measurement artifact and the field's method work is optimising against a biased target. If $\delta \le 2$ points, the gap is a genuine system deficit and effort belongs in modelling. Report `judged@10` per language in both arms; it is the diagnostic that makes $\delta$ interpretable.

## 9. Key References

- **[Foundational]** Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzmán, Grave, Ott, Zettlemoyer, Stoyanov. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. — arXiv:1911.02116
- **[Foundational]** Asai, Kasai, Clark, Lee, Choi, Hajishirzi. *XOR QA: Cross-lingual Open-Retrieval Question Answering.* NAACL 2021. — arXiv:2010.11856
- **[Benchmark]** Zhang, Thakur, Ogundepo, Kamalloo, Alfonso-Hermelo, Li, Liu, Rezagholizadeh, Lin. *MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages.* TACL, 2023. — arXiv:2210.09984
- **[Benchmark]** Zhang, Ma, Shi, Lin. *Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval.* MRL Workshop @ EMNLP 2021. — arXiv:2108.08787
- **[Benchmark]** Ogundepo, Zhang, Lin et al. *CIRAL: A Test Collection for Cross-Lingual Information Retrieval in African Languages.* FIRE evaluation track, 2023.
- **[SOTA]** Chen, Xiao, Zhang, Luo, Lian, Liu. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* ACL Findings 2024. — arXiv:2402.03216
- **[SOTA]** Wang, Yang, Huang, Yang, Majumder, Wei. *Multilingual E5 Text Embeddings: A Technical Report.* 2024. — arXiv:2402.05672
- **[Method]** Thakur, Ni, Hernández Ábrego, Wieting, Lin, Cer. *Leveraging LLMs for Synthetic Multilingual Retrieval Training Data (SWIM-IR).* NAACL Findings 2024. — arXiv:2311.05800
- **[Survey/Resource]** Enevoldsen et al. *MMTEB: Massive Multilingual Text Embedding Benchmark.* ICLR 2025. — arXiv:2502.13595
- **[Resource]** NLLB Team. *No Language Left Behind: Scaling Human-Centered Machine Translation.* 2022. — arXiv:2207.04672

## 10. Worked Example

Take one Yoruba query from a MIRACL-style set: *"Kí ni Ìjọba Ìbílẹ̀ Ìbàdàn Àríwá ń ṣe?"* (roughly: what does Ibadan North Local Government administer?).

- Yoruba Wikipedia holds on the order of $3\times10^4$ articles; English holds $\sim 7\times10^6$ — a ratio above 200:1. Cause 2 (document may not exist) is live before any model runs.
- BM25 over the Yoruba passages retrieves on diacritic-bearing surface forms. The default analyzer does not stem Yoruba, so *Ìjọba* and *ìjọba* may or may not match depending on Unicode normalisation. A single normalisation flag moves this query's reciprocal rank between 1.0 and 0.
- Suppose the run's top-10 contains 4 documents judged relevant, at ranks 1, 3, 7, 9, and 3 documents that are unjudged.
  $$\mathrm{DCG}@10 = \frac{1}{\log_2 2}+\frac{1}{\log_2 4}+\frac{1}{\log_2 8}+\frac{1}{\log_2 10} = 1 + 0.500 + 0.333 + 0.301 = 2.134$$
  With 4 known relevant documents, $\mathrm{IDCG}@10 = 1+0.631+0.500+0.431 = 2.562$, so $\mathrm{nDCG}@10 = 0.833$.
- Now suppose one of the three unjudged documents at rank 2 is in fact relevant. Then $\mathrm{DCG}@10$ gains $1/\log_2 3 = 0.631$ and $\mathrm{IDCG}@10$ rises to $3.062$: $\mathrm{nDCG}@10 = 2.765/3.062 = 0.903$.

One unjudged-but-relevant document moves this query by 7 nDCG points. `judged@10` here is $7/10$. On the English half of the same collection, deep pooling from many strong contributing systems typically leaves `judged@10` near 1.0, so the English score has no comparable slack. **The obstruction is visible:** the reported $\Delta_\ell$ between English and Yoruba includes a per-language measurement bias of the same order as the gaps method papers claim to close.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*