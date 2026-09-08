---
id: 05-retrieval-and-agents/graph-versus-vector-retrieval-global-questions
title: "Knowledge Graph versus Vector Retrieval for Global Questions"
topic: 05-retrieval-and-agents
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Knowledge Graph versus Vector Retrieval for Global Questions

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/graph-versus-vector-retrieval-global-questions` · **Status:** empirically-open

## 1. Problem Statement

A **global question** is one whose answer depends on aggregate structure of a corpus rather than on any bounded set of passages: "What are the main themes across these 1,200 incident reports?", "Which two research groups' methods converged, and when?". Local questions ("what was the pressure reading on 3 March?") are answered by a handful of chunks; global questions are not.

The claim under test: **building an LLM-extracted knowledge graph plus hierarchical community summaries over a corpus produces better answers to global questions than top-$k$ dense vector retrieval over the same corpus, at a cost that is worth paying.**

Three variants, with different difficulty:

- **Measurement.** Define a scoring function for global answers that is not itself an LLM preference judgment correlated with output length and structure. Currently unsolved — this is the binding constraint.
- **Method.** Given any fixed scorer, find the retrieval structure maximizing quality per unit of indexing + query cost. Empirically open.
- **Theory.** Characterize the class of query distributions for which no chunk-level retriever with budget $k$ can match a graph-structured retriever. Theoretically open; no separation theorem exists.

Solving it means: a reproducible protocol on which graph retrieval either wins by a stated margin on a stated query class, or does not — with the indexing cost reported in the same table.

## 2. Formal Setting

Corpus $D = \{d_1,\dots,d_N\}$, chunked into $C = \{c_1,\dots,c_M\}$ with $|c_i| \approx 600$ tokens. A query $q \sim \mathcal{Q}$ is drawn from a query distribution. A retriever is a map $R: q \mapsto S \subseteq C \cup \mathcal{A}$, where $\mathcal{A}$ is a set of derived artifacts (entity nodes, relation edges, community summaries). A reader $g_\theta$ produces $a = g_\theta(q, S)$.

**Globality.** Measure it, do not assert it. Define the *minimal sufficient support* $k^*(q)$ as the smallest $|S|$ over subsets of raw chunks such that a fixed strong reader answers $q$ correctly (or within $\epsilon$ of the reference score):
$$k^*(q) = \min\{|S| : S \subseteq C,\ \mathrm{score}(g_\theta(q,S)) \ge \tau\}.$$
Estimated by greedy forward selection with an oracle scorer; the estimate upper-bounds the true minimum. A benchmark is *global* if the median $k^*$ over its queries exceeds the deployed budget $k$ — say $\tilde{k}^* > 20$. Most published "global" benchmarks have never had $k^*$ measured.

**Cost.** Two separate quantities, both in tokens, both measurable from provider logs:
$$T_{\text{index}} = \sum_{i=1}^{M}\big(\text{in}_i + \text{out}_i\big)_{\text{extract}} + \sum_{v \in \mathcal{C}}\big(\text{in}_v + \text{out}_v\big)_{\text{summarize}}, \qquad T_{\text{query}}(q) = \text{in}(q) + \text{out}(q).$$
$\mathcal{C}$ is the set of communities from a clustering (Leiden) of the extracted graph. Amortized cost per query over a workload of size $Q$ is $T_{\text{index}}/Q + \mathbb{E}[T_{\text{query}}]$. A method that wins on quality but has $T_{\text{index}}/Q$ larger than the token cost of simply feeding the whole corpus to a long-context reader has not won.

**Quality.** $\mathrm{score}(a)$ is either (i) exact/F1 match against a short reference, available only for local and multi-hop questions, or (ii) a pairwise LLM-judge win rate $w = \Pr[\text{judge prefers } a_{\text{graph}} \text{ over } a_{\text{vec}}]$ on axes like comprehensiveness and diversity.

**Assumptions, and which are violated.**
- *Judge validity*: $w$ tracks human preference. Known violated — LLM judges show position bias and length/verbosity bias, and graph pipelines emit longer, more list-structured answers than top-$k$ pipelines.
- *Extraction fidelity*: extracted triples are faithful to $D$. Violated — LLM entity/relation extraction has non-trivial hallucination and entity-resolution error, and the error is not measured in most GraphRAG evaluations.
- *Matched compute*: baselines get the same reader, same context budget, same prompt scaffolding. Frequently violated — the strongest published graph-vs-vector comparisons vary reader prompt and answer length simultaneously with retrieval structure.
- *Query distribution*: $\mathcal{Q}$ is fixed and representative. Violated when the evaluation queries are themselves LLM-generated *from the corpus*, as in the original GraphRAG protocol, which correlates query form with the index.

## 3. State of the Art

**Systems/empirical SOTA.**
- *GraphRAG* (Edge et al., Microsoft, 2024, arXiv:2404.16130): entity/relation extraction, Leiden community detection, hierarchical community summaries, map-reduce over community summaries at query time. Reported win rates of roughly 70–80% on comprehensiveness and diversity against naive top-$k$ RAG on two ~1M-token corpora (podcast transcripts, news). **Claimed but unablated**: the win is attributed to graph structure, but the comparison also changes the query-time aggregation (map-reduce over all communities vs. one-shot over $k$ chunks). A map-reduce-over-random-chunk-clusters arm is not reported.
- *RAPTOR* (Sarthi et al., ICLR 2024, arXiv:2401.18059): recursive clustering and summarization of chunks into a tree — hierarchy without a graph. +20% absolute on QuALITY accuracy with GPT-4 reported. This is the correct control for GraphRAG and is rarely run against it in the same table.
- *HippoRAG* (Gutiérrez et al., NeurIPS 2024, arXiv:2405.14831) and *HippoRAG 2* (2025, arXiv:2502.14802): personalized PageRank over an open KG; targets multi-hop, not global synthesis.
- *LightRAG* (Guo et al., 2024, arXiv:2410.05779): dual-level graph retrieval at lower index cost; evaluated with the same LLM-judge protocol as GraphRAG, so it inherits the same validity gap.

**Theory SOTA.** None specific. There is no separation result stating a query class where any budget-$k$ chunk retriever fails and a graph retriever succeeds. The nearest formal object is coverage/set-cover framing of query-focused summarization, which is descriptive, not comparative.

**Benchmark-number-only results.** Nearly all published graph-vs-vector "global" wins exist only as LLM-judge win rates on corpora chosen by the authors, with no human agreement study and no $k^*$ characterization of the queries.

## 4. What Is Known

- On **multi-hop** benchmarks with short references, graph/PPR retrieval gives real gains: HippoRAG reports up to ~20 points recall@5 improvement over ColBERTv2/Contriever on MuSiQue and 2WikiMultihopQA (dev sets, ~1k questions, corpora of ~10k–100k passages), and near-parity on HotpotQA. Reproduced by follow-on work.
- **Hierarchical summarization helps long single documents**: RAPTOR on QuALITY (~2k questions over ~5k-token stories) beats flat chunk retrieval; the gain does not require a graph.
- **Index cost is large and measured.** LLM entity/relation extraction over a ~1M-token corpus costs on the order of $10^6$–$10^7$ LLM tokens — one to two extraction passes over the corpus plus summarization of every community at every hierarchy level. Query-time cost moves the other way: root-level community answering uses tens of thousands of context tokens per query versus hundreds of thousands for map-reduce over source text.
- **Long-context readers are a strong, cheap control.** For corpora under a few hundred thousand tokens, stuffing the corpus is competitive with retrieval on many benchmarks; retrieval's advantage is cost, not accuracy, at that scale.
- **LLM judges prefer longer answers.** Verbosity and position bias in pairwise LLM judging is well documented (Zheng et al., MT-Bench/Chatbot Arena, NeurIPS 2023). Graph pipelines produce longer answers by construction.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no validated scorer for global answers. Without a scorer with measured human agreement, and with length controlled, the win rates in §3 are not evidence about retrieval structure. Every downstream comparison inherits this.
- **Empirically open.** Whether the graph *per se* contributes anything beyond hierarchical clustering + summarization. The ablation — replace the LLM-extracted graph with embedding $k$-means clusters, keep summarization and map-reduce identical — is cheap and largely unpublished.
- **Empirically open.** The crossover in $T_{\text{index}}/Q$: at what workload size $Q$ and corpus size $N$ does amortized graph indexing beat repeated long-context reading? No published curve.
- **Theoretically open.** Existence of a query class $\mathcal{Q}^*$ and a proof that $\mathbb{E}_{\mathcal{Q}^*}[\mathrm{score}]$ under any budget-$k$ chunk retriever is bounded away from that of a graph retriever. Also open: whether extraction error propagates sublinearly through community summarization.

## 6. Why It Is Hard

**Confounded measurement, plus an evaluation that does not measure what it names.** "Global question answering" is scored by an LLM judge on axes ("comprehensiveness", "diversity") that are monotone in answer length and list structure. The graph arm changes four things at once: what is retrieved, how much is retrieved, how it is aggregated (map-reduce vs single-shot), and how long the answer is. The reported win is a sum over these, and the standard protocol identifies none of them separately. Compounding this: the reference answer for a global question does not exist — there is no ground truth for "the main themes of 1,200 reports", so the scorer cannot be anchored. Secondary obstruction: index cost of $10^6$–$10^7$ tokens per corpus makes full factorial ablation across corpora expensive enough that most groups run one configuration.

## 7. Current Research (as of 2026)

- Cheaper graph construction: LightRAG-style incremental indexing, and extraction with small models — reduces $T_{\text{index}}$ by ~an order of magnitude, quality effect unablated *(frontier — verify)*.
- Purpose-built graph-RAG benchmarks with reference-anchored global questions rather than judge-only scoring; several appeared in 2025 *(frontier — verify)*.
- Memory-framed variants (HippoRAG 2 lineage, OSU) merging passage retrieval and graph traversal into one scoring pass.
- Long-context-vs-RAG cost curves, driven by falling per-token prices; the practical question is shifting from "which retriever" to "at what corpus size does retrieval still pay".
- Survey coverage: Peng et al., *Graph Retrieval-Augmented Generation: A Survey* (2024, arXiv:2408.08921).

## 8. Concrete Next Experiment

**Scale.** Three corpora at 0.3M, 1M, and 10M tokens (e.g. a news archive, a company's incident reports, a book set). 300 queries per corpus, half written by domain readers *without* seeing the index, half machine-generated, kept as separate strata. Measure $k^*(q)$ for each query by greedy oracle selection; report the distribution.

**Arms (identical reader, identical answer-length cap of 400 words, identical map-reduce scaffold):**
1. Top-$k$ dense retrieval, $k$ chosen to match the graph arm's query-time token budget.
2. RAPTOR-style hierarchical summary tree (clustering by embedding, no graph).
3. **Control arm:** *same hierarchy, shuffled membership* — clusters of the same sizes formed at random, summarized identically. Isolates whether the content of the grouping matters at all.
4. GraphRAG community summaries.
5. Long-context stuffing where the corpus fits.

**The deciding number.** $\Delta = w(\text{arm 4 vs arm 2})$, the length-controlled, position-balanced pairwise win rate of graph over non-graph hierarchy, restricted to human-written queries with $k^* > 20$, with the judge first validated against 200 human pairwise labels (report Cohen's $\kappa$; require $\kappa \ge 0.6$ or the result is void). **If $\Delta \le 0.55$ with a 95% CI excluding 0.60, the graph contributes nothing beyond hierarchy at this scale** and its $T_{\text{index}}$ is unjustified. Arm 3 calibrates the floor: if arm 3 already beats arm 1, the win was aggregation, not retrieval.

## 9. Key References

- **[SOTA]** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* 2024. — arXiv:2404.16130
- **[Foundational]** Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning. *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* ICLR 2024. — arXiv:2401.18059
- **[SOTA]** Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su. *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.* NeurIPS 2024. — arXiv:2405.14831
- **[SOTA]** Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su. *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models.* 2025. — arXiv:2502.14802
- **[SOTA]** Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang. *LightRAG: Simple and Fast Retrieval-Augmented Generation.* 2024. — arXiv:2410.05779
- **[Survey]** Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, Siliang Tang. *Graph Retrieval-Augmented Generation: A Survey.* 2024. — arXiv:2408.08921
- **[Foundational]** Lianmin Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 Datasets & Benchmarks. — arXiv:2306.05685
- **[Foundational]** V. A. Traag, L. Waltman, N. J. van Eck. *From Louvain to Leiden: guaranteeing well-connected communities.* Scientific Reports 9, 5233, 2019.
- **[Foundational]** Patrick Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. — arXiv:2005.11401

## 10. Worked Example

Corpus: 1,000 incident reports, 1,000 tokens each — $N = 10^6$ tokens, $M \approx 1{,}700$ chunks at 600 tokens. Query: *"What recurring root causes appear across the 2025 incidents?"*

**Cost accounting.** Extraction: 1,700 chunks × (600 in + ~400 out) ≈ $1.7\times10^6$ tokens. Leiden gives ~120 leaf communities; summarizing at three levels ≈ 160 summary calls × ~4,000 tokens ≈ $6\times10^5$. So $T_{\text{index}} \approx 2.3\times10^6$ tokens. Query time, root-level map-reduce over ~15 top-level communities: ~30k tokens. Vector arm at $k=40$: ~24k tokens, $T_{\text{index}} \approx 0$ beyond embedding.

Amortization: the graph arm pays off against long-context stuffing (1M tokens/query) once $2.3\times10^6/Q + 3\times10^4 < 10^6$, i.e. $Q > 2.4$ queries. Against the vector arm it is *never* cheaper — it is only ever justified by quality.

**Where the obstruction becomes visible.** Run both arms. The graph answer lists 9 root causes with community-level counts; the vector answer lists 4, drawn from the 40 chunks retrieved. GPT-4-class judge prefers the graph answer 78% of the time on "comprehensiveness". Now run the shuffled-cluster control (arm 3 of §8): random clusters of the same sizes, summarized and map-reduced identically. It lists 8 root causes and wins 71% against the vector arm. The graph's marginal win over the shuffled control is 78 → 71, i.e. $\Delta \approx 0.54$ — inside noise at 300 queries ($\pm 0.06$ at 95%).

Then check the answers. Of the 9 listed causes, 2 are extraction artifacts: entities merged across reports by a name collision ("Node-7" in two unrelated clusters). The judge cannot see this, because there is no reference list of true root causes. **The measured 78% win is mostly aggregation breadth, partly hallucinated breadth, and only marginally graph structure — and the protocol as normally run reports a single number that conflates all three.**

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*