---
id: 05-retrieval-and-agents/optimal-chunking-granularity-dense-retrieval
title: "Optimal Chunking Granularity for Dense Retrieval"
topic: 05-retrieval-and-agents
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Chunking Granularity for Dense Retrieval

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/optimal-chunking-granularity-dense-retrieval` · **Status:** empirically-open

## 1. Problem Statement

A dense retrieval pipeline must cut a corpus into units before embedding it. The cut is irreversible at query time: whatever segmentation is chosen fixes both the vector index and the text handed to the reader. The question is what determines the best cut.

**Input.** A corpus $\mathcal{D}$ of documents, an embedding model $E$, a query distribution $Q$, a downstream reader $R$ with context budget $B$.
**Output.** A segmentation policy $\pi$ mapping each document to an ordered set of chunks.
**Objective.** Maximise end-task utility of the retrieved set, not chunk-level recall.

Three variants, of different difficulty:

- **Measurement.** Is there a metric under which "the right granularity" is even well-posed? Chunk-level relevance labels are defined relative to a chunking, so recall@k is not comparable across policies. This variant is the blocking one.
- **Method.** Given a corpus and model, find $\pi$ cheaply — ideally without re-embedding the corpus per candidate policy.
- **Theory.** Does an optimal granularity exist as a function of measurable corpus and model properties (topic-shift rate, embedding anisotropy, positional dilution of the encoder), or is it irreducibly task-specific?

Solving it means: a rule that predicts, from properties measurable before indexing, a chunking that is within a stated tolerance of the best policy in a candidate family — validated on held-out corpora the rule was not fitted on.

## 2. Formal Setting

Let a document be a token sequence $d = (t_1,\dots,t_n)$. A segmentation is a set of spans $\pi(d) = \{c_1,\dots,c_m\}$, $c_j = (a_j, b_j]$, with lengths $\ell_j = b_j - a_j$ measured in **encoder tokens**, not words or characters — the practical difference is a factor of $\approx 1.3$ for English under BPE tokenizers and much larger for code and non-Latin scripts. Overlap is $o = |c_j \cap c_{j+1}| / \ell_j$.

Each chunk has embedding $E(c) \in \mathbb{R}^h$, normalised. Retrieval scores $s(q,c) = \langle E(q), E(c)\rangle$ and returns the top $k$ chunks, subject to $\sum_{j \in \text{top-}k} \ell_j \le B$. **The budget constraint is the crux**: smaller chunks mean more of them fit, so granularity trades precision of each unit against the number of units.

Ground truth lives at the level of *answer spans*, not chunks. For query $q$ let $A(q) \subseteq \{1,\dots,n\}$ be the minimal supporting token set — annotated once, independent of $\pi$. Then define chunking-invariant coverage:

$$\mathrm{Cov}_k(q,\pi) = \frac{\big|A(q) \cap \bigcup_{j \in \mathrm{top}\text{-}k(q,\pi)} c_j\big|}{|A(q)|}$$

and dilution, the fraction of retrieved tokens that are not supporting:

$$\mathrm{Dil}_k(q,\pi) = 1 - \frac{\big|A(q) \cap \bigcup_j c_j\big|}{\sum_{j\in\mathrm{top}\text{-}k} \ell_j}.$$

End utility is $U(\pi) = \mathbb{E}_{q\sim Q}\big[u(R(q, \mathrm{top}\text{-}k(q,\pi)))\big]$ with $u$ the task score (EM, F1, faithfulness judgement). The problem is $\arg\max_\pi U(\pi)$ over a family $\Pi$ (fixed-length grids, sentence/paragraph boundaries, semantic-similarity splits, propositions, hierarchical summaries).

**Assumptions, and which are violated.**
1. *$E$ is length-invariant.* False. Embedding models are trained on a narrow length band; quality degrades outside it, and mean-pooled long inputs suffer positional dilution.
2. *$A(q)$ is contiguous and unique.* False for multi-hop and aggregation queries, where support is scattered across documents and $\mathrm{Cov}$ needs a set-cover formulation.
3. *Chunks are independent.* False — adjacent chunks are near-duplicates under overlap, inflating top-$k$ redundancy.
4. *$Q$ is known at index time.* False in production; granularity is chosen before the query distribution is observed.
5. *Reader utility is monotone in coverage.* False — "lost in the middle" positional effects mean added context can lower accuracy (Liu et al., TACL 2024).

## 3. State of the Art

**Established.** Passage-level retrieval beats whole-document retrieval for QA. DPR (Karpukhin et al., EMNLP 2020) fixed 100-word Wikipedia passages and this became the field default largely by inheritance, not by ablation. Classical IR reached the same conclusion earlier: Callan (SIGIR 1994) and Kaszkiel & Zobel (SIGIR 1997) showed fixed-length arbitrary passages match or beat structurally-motivated ones.

**Claimed but under-ablated.**
- *Propositions.* Dense X Retrieval (Chen et al., EMNLP 2024) decomposes Wikipedia into atomic propositions (FactoidWiki, ~2.5M pages) and reports large recall@$k$ gains for unsupervised retrievers at small $k$. The comparison is at fixed $k$, not fixed token budget, which favours short units mechanically.
- *Hierarchy.* RAPTOR (Sarthi et al., ICLR 2024) builds a recursive cluster-and-summarise tree and reports gains on QuALITY, NarrativeQA, QASPER. Cost of the LLM summarisation pass is not amortised into the comparison.
- *Late chunking* (Günther et al., 2024, arXiv:2409.04701) encodes the full document with a long-context model and pools per-chunk afterwards, so chunk vectors carry document context. Reported gains on BEIR subsets; ablation against simple context-prefixing is thin.
- *Contextual Retrieval* (Anthropic engineering report, 2024) prepends an LLM-generated situating sentence to each chunk and reports large reductions in failed-retrieval rate. Benchmark number only; no public reproduction across corpora at the time of writing.

**Negative result worth naming.** Qu, Tu & Bao, *Is Semantic Chunking Worth the Computational Cost?* (2024) find semantic (embedding-similarity-boundary) chunking gives no consistent advantage over fixed-size chunking across retrieval and QA benchmarks, at substantially higher cost.

No published method predicts granularity from corpus statistics without running the full index-and-evaluate loop.

## 4. What Is Known

- **Fixed-size beats "smart" splitting on average.** Reproduced across the 1994–2001 classical IR line and again in 2024 embedding-model settings. Effect sizes for semantic chunking are within noise on most BEIR datasets.
- **A broad optimum exists.** Across public reports the useful band is roughly 128–512 tokens with 10–20% overlap; within that band differences on BEIR-style nDCG@10 are typically 1–3 points, smaller than the gap between embedding models (often 5–15 points on the same datasets).
- **Long-input degradation is real.** Encoders trained at 512 tokens lose measurable quality when fed 2k+ tokens even when architecturally capable; the loss grows with the number of distinct topics in the input.
- **Reader utility is not monotone in retrieved tokens.** Liu et al. (TACL 2024) show accuracy drops when the supporting passage sits mid-context, at 20-document scale on NaturalQuestions-derived tasks.
- **Multi-vector sidesteps some of it.** ColBERT (Khattab & Zaharia, SIGIR 2020) scores at token granularity with late interaction, decoupling scoring granularity from storage granularity — at roughly an order of magnitude more index storage.
- **Corpus dependence is large.** Chunking choices that help on narrative corpora (NarrativeQA) differ from those on structured financial filings (Yepes et al., 2024), where layout-aware element boundaries beat fixed windows.

## 5. What Is Not Known

- **Methodologically blocked.** There is no standard chunking-invariant evaluation. Public benchmarks (BEIR, MTEB) ship pre-chunked corpora with passage-level qrels, so any policy that changes the segmentation destroys the label alignment. Token-level support annotations of the kind $\mathrm{Cov}_k$ requires exist only in fragments (SQuAD-style spans, QASPER evidence) and not at corpus scale.
- **Empirically open.** Nobody has run the full factorial — {policy} × {embedding model} × {corpus type} × {reader} at **fixed token budget** — at a scale that separates policy effects from model effects. The experiment is straightforward; it is just expensive.
- **Theoretically open.** No result relates optimal $\ell^\star$ to measurable quantities. A plausible conjecture — $\ell^\star$ scales inversely with topic-shift rate and with the encoder's effective positional capacity — has no proof and no clean empirical test.
- **Open.** Whether adaptive, per-document granularity beats a single global $\ell$ by more than the cost of computing it.

## 6. Why It Is Hard

**The measurement is confounded by its own labels.** Relevance judgements are collected against a fixed segmentation. Change the segmentation and you change the unit being judged, so recall@$k$ across policies compares different measurement instruments. Proposition-level retrieval "wins" partly because at fixed $k$ it retrieves shorter units and pays no penalty for retrieving fewer total tokens — the metric does not measure what it names.

**Second obstruction: cost coupling.** Evaluating one policy requires re-embedding the corpus. At 10M chunks and $10^{-2}$ USD per million tokens for a hosted embedding model, one pass over a 5B-token corpus is ~50 USD and hours of wall-clock; a 40-cell factorial with three seeds is 120 passes. That is why the factorial is unrun rather than unrunnable.

**Third: absent ground truth for support sets.** $A(q)$ must be annotated at token level, independent of chunking. That annotation does not exist for BEIR-scale corpora, and LLM-generated substitutes inherit the chunking bias of whatever pipeline produced them.

## 7. Current Research (as of 2026)

- **Late/contextual chunk encoding.** Jina AI (late chunking), Anthropic (contextual retrieval), and follow-on open reproductions. Direction: keep small retrieval units, restore document context in the vector.
- **Long-context embedding models** with 8k–32k windows, which reframe the question as "how much does granularity still matter once the encoder no longer degrades?" *(frontier — verify: whether measured degradation actually vanishes rather than moving.)*
- **Hierarchical and graph indices.** RAPTOR-style trees; graph-structured RAG variants that retrieve at entity granularity and expand. Groups: academic RAG labs plus Microsoft Research's graph-RAG line.
- **Learned/adaptive granularity.** Mix-of-Granularity-style routers that select a granularity per query. *(frontier — verify: reported gains are small and single-corpus.)*
- **Evaluation reform.** Interest in token-budget-normalised retrieval metrics is visible in workshop discussion but no adopted benchmark. This is the gap that would unblock the rest.

## 8. Concrete Next Experiment

**Question.** Does chunking policy matter once the comparison is normalised to a fixed retrieved-token budget?

**Scale.** One corpus with token-level evidence annotations at moderate size — QASPER (~1.5k papers, evidence spans annotated) plus a 500k-passage Wikipedia slice re-annotated for support spans on 2,000 queries by span-projection from NQ/SQuAD answers. Two embedding models of different context lengths (one 512-token, one 8k-token). One reader, fixed.

**Arms.** Policies: fixed-256, fixed-512, fixed-1024 (all at 15% overlap), sentence-boundary, semantic-similarity split, proposition decomposition, RAPTOR tree. Each evaluated at **$B \in \{2000, 8000\}$ retrieved tokens**, choosing $k$ per policy to exactly fill $B$.

**Control arm.** Fixed-512, 15% overlap, no contextualisation — the field default. Also run a *shuffled-boundary* control: fixed-512 chunks with boundaries offset by a random phase, which shares length statistics but destroys any boundary semantics. If semantic policies do not beat the phase-shifted control, boundary placement is not the mechanism.

**Deciding number.** $\Delta U$ = end-task F1 of the best non-default policy minus fixed-512 control, at $B = 2000$, with paired bootstrap CI over queries. **If the 95% CI excludes 0 and $\Delta U \ge 3$ F1 points**, granularity is a first-order design variable and the theory question is worth pursuing. If $|\Delta U| < 1$ point, granularity is a second-order knob and the field should spend its effort on encoders and contextualisation instead. Report $\mathrm{Cov}_k$ and $\mathrm{Dil}_k$ alongside, so the mechanism is visible.

Estimated cost: ~8 embedding passes over 500k passages plus reader inference on 2k × 14 conditions. Order 10³ USD, days not weeks.

## 9. Key References

- **[Foundational]** J. Callan. *Passage-Level Evidence in Document Retrieval.* SIGIR, 1994.
- **[Foundational]** M. Kaszkiel, J. Zobel. *Passage Retrieval Revisited.* SIGIR, 1997.
- **[Foundational]** V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, W. Yih. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP, 2020. — arXiv:2004.04906
- **[Foundational]** P. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS, 2020. — arXiv:2005.11401
- **[SOTA]** T. Chen, H. Wang, S. Chen, W. Yu, K. Ma, X. Zhao, H. Zhang, D. Yu. *Dense X Retrieval: What Retrieval Granularity Should We Use?* EMNLP, 2024. — arXiv:2312.06648
- **[SOTA]** P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, C. D. Manning. *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* ICLR, 2024. — arXiv:2401.18059
- **[SOTA]** M. Günther et al. *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models.* 2024. — arXiv:2409.04701
- **[SOTA]** O. Khattab, M. Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR, 2020. — arXiv:2004.12832
- **[Negative result]** R. Qu, R. Tu, F. S. Bao. *Is Semantic Chunking Worth the Computational Cost?* 2024. (arXiv preprint; identifier omitted as uncertain.)
- **[Evaluation]** N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2104.08663
- **[Evaluation]** N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, P. Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Domain]** A. J. Yepes, Y. You, J. Milczek, S. Laverde, R. Li. *Financial Report Chunking for Effective Retrieval Augmented Generation.* 2024. — arXiv:2402.05131
- **[Survey]** Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, H. Wang. *Retrieval-Augmented Generation for Large Language Models: A Survey.* 2023. — arXiv:2312.10997

## 10. Worked Example

A 2,400-token technical page. Query: *"what timeout does the retry policy use after the third failure?"* Support $A(q)$ is a 30-token sentence at position 1,180.

Two policies, both evaluated at budget $B = 2048$ retrieved tokens.

| Policy | $\ell$ | $k$ to fill $B$ | Chunk containing $A(q)$ | Rank | $\mathrm{Cov}$ | $\mathrm{Dil}$ |
|---|---|---|---|---|---|---|
| Fixed-512 | 512 | 4 | tokens 1024–1536 | 2 | 1.0 | $1 - 30/2048 = 0.985$ |
| Proposition | ~28 | 73 | the sentence itself | 41 | 1.0 | $1 - 30/2044 = 0.985$ |

Both recover the answer. **Dilution is identical** — because the budget, not the chunk size, sets how much irrelevant text reaches the reader. The apparent advantage of propositions evaporates under budget normalisation.

Now change the query to *"why was the backoff changed?"*, where the answer needs the sentence at 1,180 **and** a rationale paragraph at 300. Under fixed-512 both land in chunks ranked 2 and 3: $\mathrm{Cov}_4 = 1.0$. Under propositions, the rationale sentence scores 0.31 against the query in isolation — it says "this avoids thundering-herd behaviour" without naming backoff — and ranks 118th, outside $k=73$: $\mathrm{Cov} = 0.55$.

Now report the same two policies the way the literature does, at **fixed $k = 5$** instead of fixed budget. Propositions retrieve 140 tokens and hit the first query; fixed-512 retrieves 2,560. Propositions look 18× more "precise". The number changed by a factor of 18 with no change in either policy or corpus — only in the normalisation.

That is the obstruction in one table: the ordering of policies is set by the choice of metric normalisation, and the field has not agreed on one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*