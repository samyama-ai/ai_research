---
id: 05-retrieval-and-agents/generative-retrieval-billion-document-scaling
title: "Generative Retrieval Scaling to Billion-Document Corpora"
topic: 05-retrieval-and-agents
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Generative Retrieval Scaling to Billion-Document Corpora

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/generative-retrieval-billion-document-scaling` · **Status:** empirically-open

## 1. Problem Statement

Generative retrieval (GR) replaces an external index with a sequence model: given a query, the model *decodes* a document identifier (docid) directly, so retrieval quality, the index, and the parameters are one object. The open question is whether this survives corpus scale.

**Input.** A corpus $\mathcal{D} = \{d_1,\dots,d_N\}$ with $N \approx 10^9$, and a query $q$.
**Output.** A ranked list of $k$ docids produced by constrained decoding from a model $p_\theta(\cdot \mid q)$.
**Predicate.** Does there exist $\theta$, a docid scheme, and a decoding procedure such that GR matches a well-tuned dual-encoder or learned-sparse baseline on recall and nDCG *at equal or lower serving cost*, at $N = 10^9$?

Three variants, different difficulties:

- **Measurement.** Is there a corpus/query/qrel set at $N \gtrsim 10^8$ where "recall at 1B" is even estimable? Judgement pools at that scale are shallow and biased toward the systems that contributed to them.
- **Method.** Does any known docid scheme + training recipe hold quality as $N$ grows by three decades from the $8.8\times10^6$ where GR is actually studied?
- **Theory.** Is there a lower bound relating model parameters to $N$ for exact-identifier decoding — i.e. is GR *forced* to grow parameters linearly in corpus size?

## 2. Formal Setting

Docids are strings over a vocabulary $\mathcal{V}_{\text{id}}$ of length $L$: $\mathrm{id}(d) \in \mathcal{V}_{\text{id}}^L$, injective on $\mathcal{D}$, so $L \log_2|\mathcal{V}_{\text{id}}| \ge \log_2 N$. Three families: **atomic** ($L=1$, $|\mathcal{V}_{\text{id}}|=N$), **naive/semantic strings** (titles, n-grams, hierarchical cluster paths), **learned semantic IDs** (residual-quantized codes, $L\approx 4$–$32$, $|\mathcal{V}_{\text{id}}|\approx 256$–$4096$).

Retrieval score, as measured:

$$s_\theta(d \mid q) = \sum_{t=1}^{L} \log p_\theta\big(\mathrm{id}(d)_t \mid q, \mathrm{id}(d)_{<t}\big),$$

with the top-$k$ obtained by beam search of width $B$ constrained to a trie $T(\mathcal{D})$ with $N$ leaves.

**Quantities and how they are measured.**
- Quality: $R@k$ and $\mathrm{nDCG}@10$ against human qrels; report the *pool depth* and which systems contributed, since unjudged-is-irrelevant biases against a new system class.
- Serving cost: $C_{\text{infer}} = L \cdot B \cdot F_{\text{dec}} + F_{\text{enc}}$ FLOPs, plus trie memory $M_T$ (bytes actually resident), plus tail latency $p_{99}$ at fixed QPS on named hardware. Not "parameters".
- Index cost: total bytes to serve, $M_\theta + M_T$, versus a dual encoder's $N \cdot b$ bytes ($b=64$ for PQ-64, $1536$ for fp16-768d).
- Capacity: bits of corpus-specific information the weights must hold, $I \ge N\log_2 N$ bits for injective identification alone.
- Update cost: wall-clock and quality delta for inserting $\Delta N$ new documents without full retraining.

**Assumptions, and which break.**
1. *Static corpus.* Violated: web corpora churn on the order of percent per day; DSI-style models forget under sequential updates.
2. *Injective, stable docids.* Violated for learned semantic IDs — RQ codebooks collide, and collisions grow with $N$; tie-breaking suffixes are a patch, not a fix.
3. *Query distribution matches synthetic training queries.* Violated: doc2query-style generation is the dominant training signal and its distribution is model-defined, not user-defined.
4. *Quality is decomposable over docid tokens.* Assumed by beam search; the beam is not the argmax of $s_\theta$, and the gap grows with $L$.

## 3. State of the Art

**Empirical SOTA.**
- **DSI** (Tay et al., NeurIPS 2022) established the paradigm on NQ subsets of $10^4$–$3.2\times10^5$ documents.
- **NCI** (Wang et al., NeurIPS 2022) added a prefix-aware decoder and query generation; **SEAL** (Bevilacqua et al., NeurIPS 2022) used FM-index-constrained n-gram identifiers; **GenRet** (Sun et al., NeurIPS 2023) learned discrete docids by autoencoding.
- **Pradeep et al.** (EMNLP 2023 Findings), *How Does Generative Retrieval Scale to Millions of Passages?*, is the only careful scaling study: MS MARCO, full $8.8\times10^6$ passages, models to 11B parameters.
- **RIPOR** (Zeng et al., WWW 2024) is the strongest GR result at $8.8\times10^6$, using relevance-based docid initialization and prefix-oriented ranking optimization; it reports a large relative MRR@10 gain over prior GR at that scale.
- **TIGER** (Rajput et al., NeurIPS 2023) popularized RQ-VAE semantic IDs, but on *recommendation* item sets of $10^4$–$10^5$ — a scale three to five decades short.

**Baselines that GR must beat and largely does not:** ColBERTv2 (Santhanam et al., NAACL 2022), SPLADE++/v3 (Formal et al.), and simple BM25 + cross-encoder reranking, all of which serve $10^8$–$10^9$ documents in production today.

**Established vs. claimed.** *Established*: synthetic query generation is the single component whose removal collapses GR at millions of passages; naive scaling of model size does not recover the loss. *Claimed but unablated*: that learned semantic IDs are asymptotically better than atomic IDs — no paper varies $N$ across decades with the docid scheme as the only free variable. *Benchmark number only*: every headline GR result on MS MARCO dev is a single-corpus point estimate; there is no GR result at $N > 10^7$ from any group, so nothing above extrapolates by measurement.

## 4. What Is Known

- **Scale hurts, measured.** Pradeep et al. sweep $N \in \{10^5, 10^6, 8.8\times10^6\}$ on MS MARCO. Performance that looks competitive at $10^5$ is not at $8.8\times10^6$; their best configuration at full scale is an 11B-parameter model with synthetic queries and atomic IDs, reported at roughly $0.267$ MRR@10 on the dev set — below well-tuned dual encoders and far below ColBERTv2-class systems at that same corpus size.
- **Query generation dominates.** Ablations at $8.8\times10^6$ show removing synthetic queries is the largest single drop; architectural additions from DSI/NCI (prefix-aware decoding, doc-representation losses) contribute little at that scale.
- **GR is a dense retriever in disguise.** Wu et al. (SIGIR 2024) show generative retrieval with learned identifiers is equivalent in form to multi-vector dense retrieval, with docid-token embeddings playing the role of document vectors. This predicts that GR inherits, rather than escapes, the capacity requirements of a vector index.
- **Memorization has a price.** Allen-Zhu & Li (2024) measure ~2 bits of storable knowledge per parameter for transformer LMs across scales. Injective identification of $10^9$ documents needs $\ge N\log_2 N \approx 3\times10^{10}$ bits.
- **Updates are expensive or lossy.** DSI++ (Mehta et al., EMNLP 2023) documents catastrophic forgetting under document insertion; IncDSI (Kishore et al., ICML 2023) inserts documents in ~milliseconds by solving a constrained optimization over the output layer only, at the cost of freezing the encoder.

## 5. What Is Not Known

- **Empirically open (the core gap).** No GR system has been trained or evaluated at $N \ge 10^8$. MS MARCO v2 (138M passages) and ClueWeb22 (~10B pages) exist and are usable; the experiment is runnable with a few thousand accelerator-hours. Nobody has run it. The *shape* of the quality-vs-$\log N$ curve beyond $10^7$ is unmeasured.
- **Theoretically open.** No lower bound is known on $|\theta|$ as a function of $N$ for a decoder achieving $R@k \ge \rho$ under a fixed query distribution. The 2 bits/parameter figure is an empirical regularity, not a theorem about retrieval. Whether semantic (compressible) identifiers evade the $N\log_2 N$ counting argument is open.
- **Methodologically blocked.** "Recall at $10^9$" is not well defined with current judgements. TREC-style pools at $10^8$+ are built from contributed runs of existing systems; a GR system that surfaces a *different* relevant document is scored as wrong. Until pooled judgements include GR runs, or an LLM-judge protocol is calibrated against human agreement at that depth, the deciding metric is not measurable in an unbiased way.

## 6. Why It Is Hard

Two obstructions, both specific.

**Capacity is linear in $N$, and the decoder's output layer is worse than linear.** Atomic IDs require an output embedding matrix of $N \times d$: at $N=10^9$, $d=768$, fp16, that is $1.5$ TB of *parameters*, decoded through one softmax. Structured IDs shrink the softmax but push the cost into $L$ sequential decoder steps under a trie with $10^9$ leaves, where the trie itself is tens of GB and the beam must be wide enough that $B \ll k$ does not prune the answer at step 1. Neither branch has a demonstrated operating point at $10^9$.

**The evaluation does not measure what it names.** MRR@10 on MS MARCO dev, with one shallow judgement per query, rewards reproducing the annotator's single click. It cannot distinguish "GR failed to find the document" from "GR found a different, unjudged relevant document" — precisely the failure mode where a memorizing model and a matching model differ. Every scaling claim in the literature rests on this metric.

## 7. Current Research (as of 2026)

- **Learned semantic IDs at larger scale**, extending RQ-VAE codes from recommendation to web-scale text; Google (TIGER lineage) and academic IR groups (UMass CIIR, University of Amsterdam IRLab, Waterloo) are the visible actors. *(frontier — verify)*
- **Hybrid GR**: generate a coarse semantic-ID prefix to select a shard, then run dense/sparse retrieval inside it. This is the pragmatic path to $10^9$ and concedes the original claim that no index is needed. *(frontier — verify)*
- **GR inside agent loops**, where the model emits docids as tool calls and quality is judged by downstream task success rather than by qrels — attractive because it sidesteps the pooling problem, dangerous because it introduces a new confound (the agent's reasoning).
- **Theory of GR–dense equivalence**, following Wu et al. (SIGIR 2024) and Nguyen & Yates, aimed at converting the equivalence into a capacity bound.

## 8. Concrete Next Experiment

**Scale.** MS MARCO v2 passage, $1.38\times10^8$ passages, subsampled to a nested ladder $N \in \{10^6,\ 8.8\times10^6,\ 3.5\times10^7,\ 1.38\times10^8\}$ (each corpus a superset of the previous, all containing the judged documents for TREC DL 2021–2023 passage-v2 queries).

**Arms.** (a) GR with RQ-VAE semantic IDs, $L=8$, $|\mathcal{V}_{\text{id}}|=1024$, ~3B parameters, doc2query-T5 synthetic queries, held fixed across the ladder. (b) **Control:** a bi-encoder (or SPLADE++) trained on the *same* synthetic queries, same base checkpoint, matched training FLOPs and matched serving FLOPs+bytes. (c) BM25 floor. Judgement bias is addressed by re-judging the top-20 of every arm with a human-calibrated LLM judge on a 300-query sample, reporting Cohen's $\kappa$ against human labels.

**Deciding number.** Fit $R@100$ against $\log_{10} N$ over the four rungs for arms (a) and (b) and report the slope difference

$$\Delta = \frac{d\,R@100_{\text{GR}}}{d \log_{10} N} - \frac{d\,R@100_{\text{ctrl}}}{d\log_{10}N}.$$

If $\Delta \le -3$ recall points per decade, GR is disqualified at $10^9$ by extrapolation (a further $\ge 0.9$-decade gap to ClueWeb22 scale compounds it) and the paradigm's future is hybrid sharding, not end-to-end decoding. If $|\Delta| < 1$ point per decade, the $10^9$ run is justified. Cost estimate: ~2,000–5,000 A100-equivalent hours for the full ladder, both arms.

## 9. Key References

- **[Foundational]** Tay, Tran, Dehghani, Ni, Bahri, Mehta, Qin, Hui, Zhao, Gupta, Garg, Schuster, Dehghani, Metzler. *Transformer Memory as a Differentiable Search Index.* NeurIPS, 2022. — arXiv:2202.06991
- **[Foundational]** Metzler, Tay, Bahri, Najork. *Rethinking Search: Making Domain Experts out of Dilettantes.* SIGIR Forum, 2021. — arXiv:2105.02274
- **[SOTA / scaling]** Pradeep, Hui, Gupta, Lelkes, Zhuang, Lin, Metzler, Tran. *How Does Generative Retrieval Scale to Millions of Passages?* Findings of EMNLP, 2023. — arXiv:2305.11841
- **[SOTA]** Zeng, Luo, Zamani. *Scalable and Effective Generative Information Retrieval.* WWW, 2024. — arXiv:2311.09134
- **[SOTA]** Rajput, Mehta, Singh, Keshavan, Vu, Heldt, Hong, Tay, Tran, Samost, Kula, Chi, Sathiamoorthy. *Recommender Systems with Generative Retrieval.* NeurIPS, 2023. — arXiv:2305.05065
- **[Method]** Bevilacqua, Ottaviano, Lewis, Yih, Riedel, Petroni. *Autoregressive Search Engines: Generating Substrings as Document Identifiers.* NeurIPS, 2022. — arXiv:2204.10628
- **[Method]** Wang, Hou, Lu, Wu, Zhang, Wang, Zhu, Duan, Deng, Zhang et al. *A Neural Corpus Indexer for Document Retrieval.* NeurIPS, 2022. — arXiv:2206.02743
- **[Updates]** Mehta, Gupta, Tay, Dehghani, Tran, Rao, Najork, Strubell, Metzler. *DSI++: Updating Transformer Memory with New Documents.* EMNLP, 2023. — arXiv:2212.09744
- **[Updates]** Kishore, Wan, Lovelace, Artzi, Weinberger. *IncDSI: Incrementally Updatable Document Retrieval.* ICML, 2023.
- **[Theory]** Wu, Luo, Ren, de Rijke et al. *Generative Retrieval as Multi-Vector Dense Retrieval.* SIGIR, 2024. — arXiv:2404.00684
- **[Capacity]** Allen-Zhu, Li. *Physics of Language Models 3.3: Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[Baseline]** Santhanam, Khattab, Saad-Falcon, Potts, Zaharia. *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* NAACL, 2022. — arXiv:2112.01488

## 10. Worked Example

Take $N = 10^9$ and ask what the weights must hold before any semantics.

Injective identification alone needs $N \log_2 N = 10^9 \times 29.9 \approx 3.0\times10^{10}$ bits. At the measured ~2 bits per parameter (Allen-Zhu & Li, 2024), that is $1.5\times10^{10}$ parameters — a 15B-parameter model **entirely consumed by the docid table**, storing zero query–document relevance. Relevance is the expensive part: even 100 bits per document of retrievable content signature adds $10^{11}$ bits, i.e. another $5\times10^{10}$ parameters. Total: ~65B parameters, ~130 GB in fp16, decoded per query.

The control is unflattering. A dual encoder over the same $10^9$ documents with PQ-64 compression stores $10^9 \times 64\ \text{B} = 64$ GB — half the memory — and IVF-PQ touches perhaps $10^{-3}$ of it per query. GR touches *all* $130$ GB of weights on every decode step, $L=8$ times, times beam width $B=100$.

Now the throughput consequence. A 65B decoder at $B{=}100$, $L{=}8$ costs roughly $2 \times 65\times10^9 \times 100 \times 8 \approx 1.0\times10^{14}$ FLOPs per query. An A100 at ~200 TFLOP/s sustained gives ~2 queries/second/GPU at 100% utilization. The dual encoder is one 100M-parameter encoder pass plus an ANN probe: about $2\times10^{10}$ FLOPs, ~$10^4$ queries/second/GPU. **Four orders of magnitude**, before quality is discussed at all.

The obstruction is visible here and not in any published table: the entire GR literature operates at $N \le 8.8\times10^6$, where $N\log_2 N \approx 2\times10^8$ bits fits in ~0.1B parameters and is therefore *free*. Everything measured about GR has been measured in the regime where its dominant asymptotic cost is invisible.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*