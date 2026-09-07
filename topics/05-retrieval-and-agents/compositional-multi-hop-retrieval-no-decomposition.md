---
id: 05-retrieval-and-agents/compositional-multi-hop-retrieval-no-decomposition
title: "Compositional Multi-Hop Retrieval without Query Decomposition"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compositional Multi-Hop Retrieval without Query Decomposition

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/compositional-multi-hop-retrieval-no-decomposition` · **Status:** open

## 1. Problem Statement

A multi-hop query names an entity only through a chain of relations: *"the director of the film that won Best Picture the year Kubrick died"*. The gold evidence set is two or more documents, and at least one of them (the bridge document) contains no lexical or semantic overlap with the query — it is reachable only *after* the first hop is resolved.

The standard fix is **decomposition**: an LLM rewrites the query into sub-queries, retrieves iteratively, and conditions each hop on the last. This works, and it costs one or more LLM calls plus one retrieval round trip per hop.

The open problem: **can a single retrieval pass — one encoding of the query, one index lookup, no LLM in the loop — return the full gold evidence set for compositional queries?**

Three variants, with different difficulty:

- **Measurement.** Is there a benchmark on which single-pass and decomposed retrieval can be compared with the decomposition advantage isolated from the LLM's parametric knowledge? Currently no: on HotpotQA an LLM often knows the bridge entity outright, so "decomposition helps retrieval" and "the LLM answered from memory" are not separated.
- **Method.** Build an index or encoder whose single-pass recall@$k$ on 2–4 hop queries matches iterative retrieval at equal or lower total FLOPs.
- **Theory.** For a fixed-dimension embedding, is compositional top-$k$ retrieval representable at all? Partially answered — negatively — for single dense vectors (§4).

Solved would mean: recall@20 within 2 points of a strong iterative baseline on MuSiQue-Ans and 2WikiMultiHopQA, at ≥5× lower end-to-end latency, without an LLM call at query time.

## 2. Formal Setting

Corpus $\mathcal{D} = \{d_1,\dots,d_N\}$, query $q$, gold evidence set $G(q) \subseteq \mathcal{D}$ with $|G(q)| = h$ (the hop count). A retriever is a scoring function $s: \mathcal{Q}\times\mathcal{D}\to\mathbb{R}$ and returns $R_k(q)$, the top-$k$ by $s$.

**Primary metric — evidence recall (measured, not proxied):**
$$\mathrm{ER}@k = \mathbb{E}_q\left[\mathbf{1}\{G(q)\subseteq R_k(q)\}\right]$$
Note this is *set* recall, not per-document recall. Per-document recall@$k$ overstates multi-hop performance: a system that always finds hop 1 and never hop 2 scores 0.5 on the per-document metric and 0.0 on $\mathrm{ER}@k$. Most reported "recall" numbers on HotpotQA are the per-document kind; check before comparing.

**Bridge margin.** For gold document $d^\star$ and query $q$, let $\rho(d^\star \mid q)$ be its rank under $s(q,\cdot)$, and $\rho(d^\star \mid q, d_1)$ its rank under a retriever conditioned on the first-hop document. The **conditioning gain** is
$$\Delta(q) = \log \rho(d^\star \mid q) - \log \rho(d^\star \mid q, d_1)$$
measured by running both retrievers over the same index. $\Delta(q) \approx 0$ means decomposition bought nothing on that query — these queries should be excluded when measuring compositional difficulty, and usually are not.

**Cost.** Total query-time FLOPs $C = C_{\text{enc}} + T\cdot(C_{\text{LLM}} + C_{\text{ANN}})$ with $T$ hops. Single-pass fixes $T=1$ and $C_{\text{LLM}}=0$. A method that "wins" on recall while burning $10^{12}$ extra FLOPs has not solved the stated problem; the comparison must be at matched $C$.

**Assumptions and which are violated:**
- *Gold evidence is complete and unique.* Violated. HotpotQA and MuSiQue contain queries answerable from documents outside $G(q)$; MuSiQue was constructed specifically to reduce this, and 2WikiMultiHopQA's templated questions reintroduce it.
- *Hops are independent of parametric knowledge.* Violated for any query about pre-cutoff entities in Wikipedia — the model may already know the bridge.
- *Relevance is binary and query-independent per document.* Violated: a bridge document is relevant only relative to the first hop.
- *Single-vector similarity is expressive enough.* Known false in general (§4).

## 3. State of the Art

**Iterative / decomposed (empirical SOTA).** MDR (Xiong et al., ICLR 2021) re-encodes the query with the hop-1 passage appended and beam-searches; Baleen (Khattab et al., NeurIPS 2021) adds a condenser over multi-hop context; IRCoT (Trivedi et al., ACL 2023) interleaves chain-of-thought with retrieval; Self-Ask (Press et al., EMNLP Findings 2022) and DecomP (Khot et al., ICLR 2023) decompose explicitly. These are *established* — ablations exist and have been rerun by later work. What is **claimed but under-ablated** is the attribution: papers report end-task EM/F1 gains, and rarely report $\Delta(q)$ or ER@$k$ separately from answer accuracy, so how much of the gain is retrieval versus the extra LLM reasoning pass is not isolated.

**Structure-augmented single-pass.** HippoRAG (Gutiérrez et al., NeurIPS 2024) builds an open-KG over the corpus and runs Personalized PageRank from query-linked nodes — one pass at query time, multi-hop reachability precomputed in the index. Reported single-step gains on MuSiQue and 2Wiki over iterative baselines at lower online cost. GraphRAG (Edge et al., 2024, Microsoft) does community summarization; its published evidence is on global sensemaking, not ER@$k$, so it is **a benchmark-adjacent claim, not a multi-hop retrieval result**.

**Late-interaction and multi-vector.** ColBERTv2 (Santhanam et al., NAACL 2022) scores token-level; this raises the representational ceiling but does not by itself resolve bridge entities absent from the query.

**Reasoning-intensive retrieval.** BRIGHT (Su et al., ICLR 2025) shows the gap directly: retrieval where relevance requires inference. BM25 lands around 14 nDCG@10; strong dense retrievers do not clearly beat it; adding LLM-generated reasoning to the query is what moves the number. That is decomposition by another name.

## 4. What Is Known

- **Dimension is a hard ceiling.** Weller et al., *On the Theoretical Limitations of Embedding-Based Retrieval* (2025), connect the set of top-$k$ document subsets a $d$-dimensional single-vector retriever can express to the sign rank of the query–document relevance matrix. Their LIMIT dataset instantiates this: a corpus of **46 documents** with simple 2-document gold sets on which SOTA embedders (including 4096-dim models) score far below saturation — recall@$k$ in the low tens of percent where BM25-style or multi-vector approaches do much better. Scale: tiny corpus, adversarially constructed, so it is an existence proof of the limitation, not an estimate of its frequency.
- **The compositionality gap is real and does not close with scale.** Press et al. (2022) measured, on Bamboogle and compositional Wikipedia 2-hop sets, that models answer both single hops correctly far more often than the composition; the *gap* stayed roughly constant from small to GPT-3-scale models even as single-hop accuracy rose. Scale: GPT-3 family, ~125 hand-built + templated compositional questions.
- **Iterative retrieval gives large, reproduced recall gains.** IRCoT reported retrieval recall improvements of up to ~21 points over one-step retrieval on HotpotQA / 2Wiki / MuSiQue / IIRC with GPT-3-class readers; the direction has been reproduced by HippoRAG and successors. Scale: Wikipedia-derived corpora, $N \sim 10^5$–$10^7$.
- **Benchmark difficulty is dataset-dependent by a wide margin.** MuSiQue (Trivedi et al., TACL 2022) was built by composing single-hop questions so that single-hop shortcuts fail; the human–model gap it reports is far larger than on HotpotQA, confirming that HotpotQA numbers overstate multi-hop capability.
- **Structure precomputation transfers some of the online cost offline.** HippoRAG's PPR retrieval is single-pass online and competitive with multi-step pipelines on MuSiQue/2Wiki — evidence that *some* of the decomposition work is index-side, not query-side.

## 5. What Is Not Known

- **Theoretically open.** The sign-rank result bounds *single-vector* retrievers. No analogous lower bound exists for late-interaction (multi-vector) scoring, or for graph-index retrieval with a learned diffusion. Whether a $d$-dimensional multi-vector retriever with $m$ vectors per document can represent all $h$-hop compositional top-$k$ sets over a corpus of size $N$ — and what $m$ must be as a function of $N, h$ — is unproven either way.
- **Empirically open.** No published study varies hop count $h \in \{2,3,4\}$ at fixed corpus and reports ER@$k$ for single-pass versus iterative *at matched FLOPs*. The experiment is runnable today on MuSiQue with commodity hardware. Nobody has run it as a controlled sweep.
- **Methodologically blocked.** Attribution of a multi-hop gain to retrieval versus parametric recall. Without a corpus of entities the reader model provably has not memorized, "the decomposition found the bridge" and "the LLM already knew the bridge" are not distinguishable from published numbers. This is the blocking issue for the measurement variant.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent per-hop ground truth**.

1. Benchmarks report end-task EM/F1. Retrieval quality, reader reasoning, and parametric memory all move that number, and they are not separable post hoc.
2. Gold evidence sets are annotated at the *document* level for the *answer*, not per hop. So $\Delta(q)$ — the quantity that says whether a query is genuinely compositional — cannot be computed from the released labels; it requires re-running a conditioned retriever and is therefore method-dependent.
3. Any dataset drawn from Wikipedia is inside every frontier model's pretraining. Building a held-out compositional corpus means synthesizing entities, which changes the retrieval distribution (synthetic names are lexically distinctive, so BM25 gets artificially strong).

Secondary: the theory cuts against the cheap solution. If the sign-rank bound is tight, no amount of training fixes single-vector single-pass retrieval; the fix must change the representation class or move work into the index, and both raise indexing cost superlinearly in $N$.

## 7. Current Research (as of 2026)

- **Index-side reasoning.** Graph and KG-augmented indices (HippoRAG line, Ohio State NLP; GraphRAG line, Microsoft Research) — precompute reachability so query time stays single-pass. Open question is index build cost at $N > 10^7$.
- **Multi-vector scaling.** ColBERT-descendant work on making late interaction cheap enough to be the default; motivated post-2025 by the dimension lower bound. *(frontier — verify current SOTA numbers.)*
- **Reasoning-augmented embedders.** Training encoders on reasoning-intensive relevance (BRIGHT-style, Stanford/Princeton groups). These blur the problem statement: if the encoder internally does the decomposition, "no decomposition" is a claim about interface, not computation. Whether that distinction matters is itself contested.
- **Agentic deep-research systems** amortize many hops with many LLM calls — they are the opposite corner of the cost/recall trade and serve as the upper-bound arm.

## 8. Concrete Next Experiment

**Question:** does the single-pass recall deficit grow with hop count, or is it a constant offset?

**Scale.** MuSiQue-Ans, filtered to 2-hop, 3-hop, and 4-hop subsets (roughly 1–2k questions each in the full set; use ≥500 per bucket), over the standard Wikipedia paragraph corpus ($N \approx 10^6$ passages, or the full English dump if compute allows). Fits on 4–8 A100-class GPUs in under a week including index builds.

**Arms (all at matched query-time FLOPs, reported):**
1. Single-pass dense (E5/GTE-class, $d = 1024$).
2. Single-pass late-interaction (ColBERTv2).
3. Single-pass graph index (HippoRAG-style PPR).
4. **Control:** iterative retrieval with oracle decomposition — use MuSiQue's released sub-question annotations as the sub-queries, so the LLM's reasoning quality is removed from the comparison. This is the arm that makes the result interpretable and is the one usually missing.

**Deciding number.** $\mathrm{ER}@20$ (full-set recall, not per-document) per hop bucket, plotted against $h$. The decisive quantity is the **slope of the gap** $g(h) = \mathrm{ER}@20_{\text{oracle-iterative}} - \mathrm{ER}@20_{\text{best single-pass}}$. If $g(4) - g(2) < 5$ points, the deficit is a constant offset and single-pass retrieval is a tractable engineering target. If $g(4) - g(2) > 15$ points, single-pass retrieval degrades compositionally and the sign-rank obstruction is binding in practice, not just adversarially.

## 9. Key References

- **[Foundational]** Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP, 2018. — arXiv:1809.09600
- **[Foundational]** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal. *MuSiQue: Multihop Questions via Single-hop Question Composition.* TACL, 2022. — arXiv:2108.00573
- **[Foundational]** Wenhan Xiong et al. *Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval.* ICLR, 2021. — arXiv:2009.12756
- **[Foundational]** Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, Mike Lewis. *Measuring and Narrowing the Compositionality Gap in Language Models.* Findings of EMNLP, 2023 (arXiv 2022). — arXiv:2210.03350
- **[SOTA]** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal. *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.* ACL, 2023. — arXiv:2212.10509
- **[SOTA]** Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su. *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.* NeurIPS, 2024. — arXiv:2405.14831
- **[SOTA]** Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia. *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* NAACL, 2022. — arXiv:2112.01488
- **[Theory]** Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee. *On the Theoretical Limitations of Embedding-Based Retrieval.* 2025. — arXiv:2508.21038
- **[Benchmark]** Hongjin Su et al. *BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.* ICLR, 2025. — arXiv:2407.12883
- **[Related]** Omar Khattab, Christopher Potts, Matei Zaharia. *Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval.* NeurIPS, 2021. — arXiv:2101.00436
- **[Survey]** Yunfan Gao et al. *Retrieval-Augmented Generation for Large Language Models: A Survey.* 2023. — arXiv:2312.10997

## 10. Worked Example

Query: *"What is the population of the city where the composer of the opera 'Nixon in China' was born?"*

Gold set $G(q)$: $d_1$ = "John Adams (composer)" (born Worcester, Massachusetts), $d_2$ = "Worcester, Massachusetts" (population 206,518, 2020 census).

**Single-pass dense, $d = 1024$.** The query embedding is dominated by *opera*, *Nixon in China*, *composer*, *population*, *born*. Cosine against the index:

| Doc | Rank | Why |
|---|---|---|
| Nixon in China | 1 | near-exact lexical + semantic match |
| John Adams (composer) — $d_1$ | 3 | strong |
| List of operas by John Adams | 6 | topical |
| Worcester, Massachusetts — $d_2$ | ~40,000 | shares only *population*; the query never says "Worcester" |

$\mathrm{ER}@20 = 0$. The failure is not a ranking error to be tuned away. $d_2$'s embedding has no reason to be near $q$'s — the only bridge is the string *Worcester*, which does not exist in $q$.

**Conditioned retriever.** Encode $[q; d_1]$. Now "Worcester, Massachusetts" is at rank 2. $\rho(d_2\mid q)\approx 4\times10^4$, $\rho(d_2\mid q,d_1)=2$, so $\Delta(q) = \ln(4\times10^4) - \ln 2 \approx 9.9$. A genuinely compositional query.

**Where the obstruction becomes visible.** Ask a frontier reader the question with *no retrieval at all* and it answers "about 206,000" from memory. So the end-task EM for the single-pass RAG pipeline is **1.0** while its $\mathrm{ER}@20$ is **0.0**. Any paper reporting EM on this query records a success for a retriever that retrieved nothing useful. That is the confound in §6, in one query: the headline metric and the quantity under study point in opposite directions, and only the ER/$\Delta$ pair separates them.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*