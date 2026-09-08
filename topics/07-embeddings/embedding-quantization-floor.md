---
id: 07-embeddings/embedding-quantization-floor
title: "Quantization Floor for Embedding Retrieval Quality"
topic: 07-embeddings
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Quantization Floor for Embedding Retrieval Quality

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/embedding-quantization-floor` · **Status:** empirically-open

## 1. Problem Statement

Dense retrieval stores a vector per document. Storage and memory bandwidth scale linearly with bits per vector, so practitioners compress: `float32` → `float16` → `int8` → 1-bit binary → product-quantized codes. The question is where quality actually breaks.

**The problem.** Given a corpus $\mathcal{D}$, a query set $\mathcal{Q}$, an encoder $f$, and a bit budget $B$ per document vector, determine the *quantization floor*: the smallest $B$ at which a retrieval quality metric stays within tolerance $\varepsilon$ of the uncompressed baseline.

Three variants, routinely conflated:

- **Measurement variant.** Is "quality retention" even well defined? Retention measured against a benchmark with shallow, sparse relevance judgments (MS MARCO: ~1.1 positives per query) is not retention of ranking fidelity. A quantizer that shuffles ranks 3–100 can score 100% nDCG@10 retention. *This variant is methodologically blocked.*
- **Method variant.** Build the quantizer that attains the lowest $B$ at a fixed $\varepsilon$. Actively worked, with real progress (OPQ, JPQ, RepCONC, Matryoshka + binary).
- **Theory variant.** Prove a lower bound on $B$ as a function of corpus size $n$, intrinsic dimension, and required recall. Only loosely connected results exist (JL, Alon, Larsen–Nelson); none gives a tight floor for real embedding geometry.

Solving it means: a predictor $\hat{B}(\mathcal{D}, f, k, \varepsilon)$ that, without running the full retrieval evaluation, states the bit budget at which recall@$k$ degrades by more than $\varepsilon$ — and is validated out of sample.

## 2. Formal Setting

Encoder $f: \mathcal{X} \to \mathbb{R}^d$. Corpus embeddings $X = \{x_i\}_{i=1}^n$, queries $q \in \mathcal{Q}$. Score $s(q,x) = \langle q, x\rangle$ (unit-normalized, so inner product and cosine coincide).

A quantizer is a pair $(Q, \tilde{s})$ with codebook $Q: \mathbb{R}^d \to \{0,1\}^B$ and an approximate score $\tilde{s}(q, Q(x))$. **Measured as:** $B$ is the on-disk size of the index divided by $n$, in bits, *including* codebook and any residual/rescoring payload — not the nominal per-dimension width. This matters: "binary, 32× smaller" claims usually exclude the float vectors kept for rescoring.

Ground-truth top-$k$: $R_k(q) = \arg\text{top-}k_{i} \langle q, x_i \rangle$ under exact float32. Quantized: $\tilde{R}_k(q)$.

**Geometric fidelity** (judgment-free):
$$\text{Recall}_{k@k}(B) = \frac{1}{|\mathcal{Q}|}\sum_{q}\frac{|R_k(q) \cap \tilde{R}_k(q)|}{k}$$

**Task quality**, with relevance judgments $\mathrm{rel}(q,\cdot)$:
$$\text{nDCG@}k(B), \qquad \Delta(B) = \frac{\text{nDCG@}k(B)}{\text{nDCG@}k(\infty)}$$

**Floor:** $B^\star(\varepsilon) = \min\{B : \Delta(B) \ge 1 - \varepsilon\}$.

Rate–distortion framing: quantization noise on the score is $\delta(q,x) = \tilde{s}(q,Q(x)) - s(q,x)$. A retrieval error at rank $k$ requires the score gap
$$g_k(q) = s(q, x_{(k)}) - s(q, x_{(k+1)})$$
to be smaller than $|\delta(q,x_{(k)})| + |\delta(q,x_{(k+1)})|$. So the floor is governed by the **score-gap distribution near rank $k$**, not by mean squared reconstruction error $\mathbb{E}\|x - \hat{x}\|^2$ — which is what every off-the-shelf quantizer minimizes. Guo et al. (ICML 2020) made exactly this point and reweighted the loss anisotropically.

**Assumptions, and which fail.**
1. *Scores are the objective.* Fails when a reranker follows: the quantized stage only needs to preserve a candidate set, so $B^\star$ for recall@1000 ≪ $B^\star$ for nDCG@10-as-final.
2. *Judgments are complete.* Violated everywhere. MS MARCO dev has ~1.1 judged positives per query; nDCG@10 cannot see a swap between an unjudged relevant doc and a judged one.
3. *Query and document distributions match the quantizer's training data.* Violated under domain shift; BEIR is the standard stress test.
4. *Coordinates are roughly isotropic.* Violated — LLM-derived embeddings have heavy-tailed outlier dimensions, the same phenomenon that breaks naive `int8` weight quantization.
5. *$d$ is fixed.* Matryoshka training makes dimension truncation a competing axis, so the budget is $B = d' \cdot b$ and the trade between $d'$ and bits-per-dim $b$ is itself unresolved.

## 3. State of the Art

**Established (independently reproduced).**
- **Product quantization** (Jégou, Douze, Schmid, *TPAMI* 2011) and **OPQ** (Ge, He, Ke, Sun, *CVPR* 2013): the durable baseline. OPQ's learned rotation before PQ reliably beats plain PQ on SIFT/GIST-scale ANN benchmarks.
- **ScaNN / anisotropic vector quantization** (Guo et al., *ICML* 2020): weighting reconstruction error by its parallel component to the datapoint beats MSE-optimal quantization at equal bitrate on Glove-1M and related benchmarks. Reproduced in the ANN-Benchmarks ecosystem.
- **End-to-end learned quantization for retrieval**: JPQ (Zhan et al., *CIKM* 2021) and RepCONC (Zhan et al., *WSDM* 2022) train the encoder jointly with PQ centroids and recover most of the MS MARCO MRR@10 gap that post-hoc PQ loses at the same compression ratio.
- **Binary hashing with rescoring**: BPR (Yamada, Asai, Hajishirzi, *ACL* 2021) cuts a DPR index from ~65 GB to ~2 GB with a few-point drop in top-20 accuracy on Natural Questions, using a learned-to-hash layer plus reranking of a binary-retrieved candidate set.

**Claimed but unablated.**
- Vendor and library blog posts (Cohere int8/binary embeddings, 2024; Sentence-Transformers "Embedding Quantization", 2024) report ~90–96% nDCG@10 retention for binary embeddings and ~99%+ with float rescoring of a binary-retrieved shortlist. These are engineering reports on a handful of models and datasets, not controlled studies: the retained-float overhead is usually excluded from the bit count, the shortlist size is not swept, and no confidence intervals are given. Treat as existence proofs, not measurements of $B^\star$.
- "Matryoshka + binary composes multiplicatively" is widely repeated. No published ablation isolates the interaction term.

**Theory SOTA.** Johnson–Lindenstrauss gives $d' = O(\varepsilon^{-2}\log n)$ for pairwise distance preservation; Larsen and Nelson (*FOCS* 2017) proved this dimension is optimal for the worst case; Alon (2003) gave the near-matching lower bound for exact-ish embeddings. None of these bound *bits*, none is instance-adaptive, and all concern distance preservation rather than top-$k$ ordering.

## 4. What Is Known

- **`float32` → `int8` is close to free.** Scalar quantization with per-dimension calibration loses <1% nDCG@10 on MTEB-scale evaluations for common 768–1024-d models; 4× storage reduction. Scale: MTEB retrieval subset, ~15 datasets, models 100M–7B params.
- **Binary at 1 bit/dim costs single-digit percent, *if* rescoring is allowed.** BPR on NQ: index 65 GB → 2 GB (~32×), top-20 accuracy within a few points of DPR. Without the rescoring stage, the drop is substantially larger.
- **Learned beats post-hoc at high compression.** RepCONC/JPQ on MS MARCO passage (8.8M passages) hold MRR@10 far better than post-hoc OPQ at compression ratios in the 32–64× range; the advantage grows as bits shrink.
- **MSE is the wrong loss.** ScaNN's anisotropic result is the cleanest evidence that reconstruction error and retrieval error decouple.
- **Outlier dimensions dominate the error budget.** A small number of coordinates carry disproportionate score mass in transformer embeddings; uniform scalar quantizers spend their range on them.
- **The degradation curve is a cliff, not a slope.** Across reported studies, quality is near-flat from 32 to ~4 bits/dim and falls sharply below ~1–2 bits/dim. The *location* of the cliff varies by model and corpus and is not predicted by any published quantity.

## 5. What Is Not Known

- **Empirically open.** No study sweeps (bits/dim) × (dimension, via Matryoshka) × (corpus size $10^6$–$10^9$) × (rescoring depth) on one model family with a fixed evaluation, reporting *both* recall$_{k@k}$ and nDCG. The experiment is entirely runnable — it costs GPU-days, not GPU-years — and nobody has published it.
- **Empirically open.** Whether $B^\star$ grows with $n$. Score gaps near rank $k$ shrink as the corpus grows, so the floor should rise; the rate is unmeasured. All binary-embedding evidence comes from corpora of $10^6$–$10^7$.
- **Theoretically open.** A lower bound on bits per vector for top-$k$ *order* preservation with probability $1-\delta$ over a specified query distribution. JL bounds distances, not rankings, and is worst-case rather than instance-adaptive.
- **Theoretically open.** Whether an encoder can be trained so that its embedding geometry makes some target bitrate provably sufficient — i.e., quantization-aware representation learning with a guarantee, not just an empirical win.
- **Methodologically blocked.** "Quality retention" on sparse-judgment benchmarks. Until retention is defined against a judgment-free fidelity target (recall$_{k@k}$) *or* against deep judgments, reported retention numbers are not comparable across papers.

## 6. Why It Is Hard

**The primary obstruction is confounded measurement.** nDCG@10 on MS MARCO or BEIR conflates three effects: (a) genuine ranking damage from quantization, (b) judgment sparsity that hides damage below the judged set, and (c) the *lossy-compression-as-regularizer* effect, where quantization occasionally raises the metric by discarding spurious score precision. A single retention number cannot separate them, and papers report only that number.

**Secondary: the decisive quantity is a tail statistic.** The floor depends on $\Pr[g_k(q) < \delta]$ — the probability of a small score gap at the rank boundary. Tail estimates need many queries and exact top-$k$ ground truth over the full corpus, which is $O(nd)$ per query at $n = 10^9$.

**Third: non-identifiability across the budget.** A fixed byte budget can be spent on more dimensions at fewer bits or fewer dimensions at more bits, with or without a rescoring reserve. Published comparisons hold one axis fixed and vary another, so "binary is enough" and "binary is not enough" are both true at different points of an unswept surface.

## 7. Current Research (as of 2026)

- **Quantization-aware embedding training.** Extending the JPQ/RepCONC line to instruction-tuned LLM embedders; Matryoshka-style losses (Kusupati et al., NeurIPS 2022) combined with binarization objectives so a single checkpoint serves many budgets. *(frontier — verify current results.)*
- **Library-level defaults.** Faiss (Douze et al., 2024) and ScaNN continue to be the systems substrate; Sentence-Transformers and vector-DB vendors (Qdrant, Weaviate, Pinecone, Cohere) ship int8/binary paths with rescoring. Engineering-led, not measurement-led.
- **Outlier-aware scalar quantization** borrowed from weight-quantization work (LLM.int8, GPTQ lineage) applied to embeddings. *(frontier — verify.)*
- **Judgment-free evaluation.** Growing use of recall$_{k@k}$ against exact search as the primary quantizer metric, which is the right move for this problem.

## 8. Concrete Next Experiment

**Question.** Does the quantization floor rise with corpus size, and where is it?

**Scale.** One model family with Matryoshka dimensions (e.g. a 1024-d open embedder with 64/128/256/512/1024 heads). Three corpora at $10^6$, $10^7$, $10^8$ passages, nested (each a superset of the previous), drawn from a common source such as MS MARCO + a C4 slice. 10,000 held-out queries. Exact float32 top-1000 computed once per corpus with brute-force GPU search — roughly 10 A100-hours at $10^8$ × 1024-d, the dominant cost.

**Grid.** bits/dim $\in \{32, 8, 4, 2, 1\}$ × dimension $\in \{128, 256, 512, 1024\}$ × rescoring depth $\in \{0, 10k, 100k\}$ candidates. Report the true index bytes/vector for each cell.

**Control arm.** At each total-bytes budget, the control is the *dimension-truncated float32* index at the same bytes/vector (e.g. 128-d float32 = 512 B ≈ 1024-d 4-bit = 512 B). This isolates quantization from mere dimension reduction — the confound most published comparisons leave in.

**Deciding number.** $B^\star_{0.01}(n)$: the minimum bytes/vector at which $\text{Recall}_{100@100}$ against exact float32 stays $\ge 0.99$, plotted against $n$. If $B^\star_{0.01}(10^8) / B^\star_{0.01}(10^6) \le 1.2$, the floor is essentially corpus-size-independent and current binary practice extrapolates. If the ratio exceeds $2$, binary indexes validated at $10^6$ are unsafe at web scale and every vendor retention figure needs re-derivation.

## 9. Key References

- **[Foundational]** Hervé Jégou, Matthijs Douze, Cordelia Schmid. *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI, 2011.
- **[Foundational]** Tiezheng Ge, Kaiming He, Qifa Ke, Jian Sun. *Optimized Product Quantization.* CVPR, 2013.
- **[Foundational]** William B. Johnson, Joram Lindenstrauss. *Extensions of Lipschitz mappings into a Hilbert space.* Contemporary Mathematics, 1984.
- **[Theory]** Kasper Green Larsen, Jelani Nelson. *Optimality of the Johnson-Lindenstrauss Lemma.* FOCS, 2017.
- **[Theory]** Noga Alon. *Problems and results in extremal combinatorics I.* Discrete Mathematics, 2003.
- **[SOTA]** Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, Sanjiv Kumar. *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* ICML, 2020.
- **[SOTA]** Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, Shaoping Ma. *Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance.* CIKM, 2021.
- **[SOTA]** Jingtao Zhan et al. *Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval.* WSDM, 2022.
- **[SOTA]** Ikuya Yamada, Akari Asai, Hannaneh Hajishirzi. *Efficient Passage Retrieval with Hashing for Open-domain Question Answering.* ACL, 2021.
- **[SOTA]** Aditya Kusupati et al. *Matryoshka Representation Learning.* NeurIPS, 2022.
- **[Systems]** Matthijs Douze et al. *The Faiss library.* 2024. — arXiv:2401.08281
- **[Benchmark]** Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023.
- **[Benchmark]** Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks, 2021.
- **[Foundational]** Moses Charikar. *Similarity Estimation Techniques from Rounding Algorithms.* STOC, 2002.

## 10. Worked Example

Take a 1024-d unit-normalized embedder, corpus $n = 10^7$, $k = 10$.

**Budget arithmetic.** float32: $1024 \times 4 = 4096$ B/vector → 41 GB. Binary: 128 B/vector → 1.3 GB, nominally 32×. But the reported "99% retention with rescoring" configuration retains float32 (or int8) vectors for the top 1000 candidates *per query* — in practice the whole float index stays resident or on SSD. If it is resident, the true budget is 4224 B/vector, i.e. **1.03×, not 32×**. The 32× figure and the 99% figure describe different systems. That inconsistency is the obstruction in one line.

**Why the metric hides it.** Binarizing by sign is SimHash: $\Pr[\text{sign}(\langle q, x\rangle \text{ bits agree})]$ relates to angle by $1 - \theta/\pi$ per bit. With $d = 1024$ bits, the standard-error on the estimated angle is about $\sqrt{p(1-p)/1024} \approx 0.0156$ in agreement fraction, i.e. $\approx 0.049$ rad $\approx 2.8°$ of angular noise.

Now the score gap. For a typical strong retriever on MS MARCO-like data, cosine scores at ranks 10 and 11 differ by roughly $0.002$–$0.01$. At $\cos\theta \approx 0.75$ ($\theta \approx 0.72$ rad), $|d(\cos\theta)/d\theta| = \sin\theta \approx 0.66$, so $2.8°$ of angular noise is $0.049 \times 0.66 \approx 0.032$ of score noise — **3× to 16× larger than the gap it must resolve**. Ranks 10 and 11 are therefore swapped essentially at chance.

**And yet nDCG@10 barely moves.** With ~1.1 judged positives per query, a swap between rank 10 and rank 11 changes nDCG@10 only when exactly one of the two is the judged positive — a low-probability event. Meanwhile $\text{Recall}_{10@10}$ against exact float32 would fall to perhaps 0.6–0.7 under the same noise.

The same quantizer thus reads as "97% retention" on the benchmark and "30–40% of the true top-10 lost" on the judgment-free metric. Both numbers are correct. Neither alone is a quantization floor, and the literature reports the first.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*