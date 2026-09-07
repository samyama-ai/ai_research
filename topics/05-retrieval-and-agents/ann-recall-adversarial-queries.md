---
id: 05-retrieval-and-agents/ann-recall-adversarial-queries
title: "Provable Approximate Nearest Neighbor Recall Under Adversarial Queries"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Approximate Nearest Neighbor Recall Under Adversarial Queries

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/ann-recall-adversarial-queries` · **Status:** open

## 1. Problem Statement

Production retrieval stacks — RAG pipelines, agent memory, recommendation — use graph-based approximate nearest neighbor (ANN) indexes (HNSW, DiskANN, ScaNN) whose reported recall is an **average over a query distribution that matches the base distribution**. The open problem: give an index that has a *per-query* recall guarantee against a query chosen by an adversary who knows the index, or prove no practical index can.

Three variants, different difficulty:

- **Measurement.** Define adversarial recall so it is estimable. Recall@$k$ averaged over a benign query set is not it; the quantity that matters is $\min_q \mathrm{recall}(q)$ or a tail quantile, and the minimum is not estimable by sampling.
- **Method.** Build an index with high benign throughput whose worst-case recall over a specified query class is certifiable per query — e.g. it emits a certificate "no point within distance $r$ was missed" alongside each result.
- **Theory.** Prove or refute: for graph indexes with $O(n\log n)$ build time and $O(\text{polylog } n)$ query distance computations, there exist datasets and queries on which recall@10 is $O(1/k)$ — and characterize the data assumption (doubling dimension, aspect ratio) that rules this out.

Solving it means: a stated query class, a bound of the form $\Pr[\text{miss}] \le \delta$ that holds for every query in that class, and a benchmarked cost of the guarantee in QPS.

## 2. Formal Setting

Dataset $P = \{p_1,\dots,p_n\} \subset \mathbb{R}^d$, metric $\rho$ (usually $\ell_2$ or cosine). Index $\mathcal{I} = A(P; r)$ built by randomized algorithm $A$ with seed $r$. Query algorithm $Q(\mathcal{I}, q, B)$ returns $\hat{N}_k(q)$, $|\hat{N}_k| = k$, using at most $B$ distance evaluations.

**Recall, as measured.** Ground truth $N_k(q)$ from exhaustive scan (the only honest source; on $n = 10^9$, $d=128$ this is $\sim$ hours of GPU time per 10k queries).
$$\mathrm{rec}_k(q) = \frac{|\hat{N}_k(q) \cap N_k(q)|}{k}.$$
Benign recall is $\bar{R} = \mathbb{E}_{q\sim\mathcal{D}}[\mathrm{rec}_k(q)]$, estimated from a held-out query file. Adversarial recall over class $\mathcal{Q}$:
$$R^{\mathrm{adv}}_k(\mathcal{Q}) = \inf_{q \in \mathcal{Q}} \mathbb{E}_r[\mathrm{rec}_k(q)].$$
Useful query classes: $\mathcal{Q}_{\mathrm{free}} = \mathbb{R}^d$; $\mathcal{Q}_\epsilon(q_0) = \{q : \|q - q_0\| \le \epsilon\}$ (perturbation of a real query); $\mathcal{Q}_{\mathrm{adapt}}$ = queries produced by an adaptive attacker who has seen previous outputs of $\mathcal{I}$.

**Difficulty parameters.** Aspect ratio $\Delta = \max_{i\ne j}\rho(p_i,p_j)/\min_{i\ne j}\rho(p_i,p_j)$. Doubling dimension $\lambda$: smallest $\lambda$ with every ball coverable by $2^\lambda$ half-radius balls. Relative contrast $c(q) = \mathbb{E}[\rho(q,p)] / \rho(q, p_{(1)})$ — measured per query, and the empirical predictor of graph-search failure.

**Assumptions, and their violation.**
- *Queries are i.i.d. from the base distribution.* Violated by construction in RAG: queries are user text, the corpus is documents, and an attacker controls query text in prompt-injection settings.
- *The index is static and its seed is secret.* Violated: HNSW builds are deterministic given insert order and released model weights; embedding models are public, so the attacker can compute $\phi(\cdot)$ exactly.
- *Bounded doubling dimension.* Violated for text embeddings: measured intrinsic dimension of OpenAI/E5-class embeddings is in the 10–30 range locally but with heavy-tailed hubness, so $\Delta$ is large and the near-neighbor graph has hubs of very high in-degree.

## 3. State of the Art

**Theory SOTA (established).** LSH gives per-query guarantees against *any* query: Indyk–Motwani (STOC 1998) and Andoni–Indyk (FOCS 2006) solve $(c,r)$-ANN with $O(n^{1+\rho})$ space, $\rho = 1/c^2 + o(1)$ for $\ell_2$; the failure probability is over the hash seed and holds for adversarially chosen $q$ *provided the seed is unknown to the adversary*. This is the only widely-deployed family with a worst-case per-query bound. Data-dependent LSH (Andoni–Razenshteyn, STOC 2015) improves to $\rho = 1/(2c^2-1)$.

Lower bounds: Rubinstein (STOC 2018) shows under SETH that exact/near-exact bichromatic closest pair in $\ell_2$ admits no truly subquadratic algorithm, and rules out $(1+o(1))$-approximation — so worst-case-fast ANN requires genuine approximation slack.

**Systems SOTA (benchmark numbers only).** HNSW (Malkov & Yashunin, TPAMI 2020) and DiskANN (Jayaram Subramanya et al., NeurIPS 2019) reach recall@10 $\ge 0.95$ at $10^3$–$10^4$ QPS/core on SIFT1M–1B in ANN-Benchmarks (Aumüller et al., *Information Systems* 2020) and the NeurIPS'21 BigANN challenge (Simhadri et al.). **These are averages over benign query files.** No mainstream index ships a per-query certificate.

**Established but under-appreciated:** Indyk & Xu (NeurIPS 2023) give the first rigorous analysis of these implementations — a guarantee for "slow preprocessing" DiskANN in terms of doubling dimension and aspect ratio, and *counterexample constructions* on which HNSW and fast-preprocessing DiskANN fail. This is a proof, not a benchmark.

**Claimed but unablated:** that observed benign recall transfers to production query distributions; that quantization (PQ/OPQ, ScaNN's anisotropic loss, Guo et al. ICML 2020) degrades recall uniformly rather than concentrating loss on hard queries. Neither has a published tail-conditioned ablation.

## 4. What Is Known

- **Graph indexes have provable failure instances.** Indyk & Xu (NeurIPS 2023) construct datasets where HNSW's greedy search returns a point far from optimal; the failure is structural (graph connectivity), not probabilistic.
- **Benign-average recall hides a bad tail.** On SIFT1M with HNSW at $\bar{R}_{10} \approx 0.95$, the bottom 1% of queries by relative contrast routinely sit below recall 0.6 — reproduced across ANN-Benchmarks runs; the mean is carried by easy queries.
- **Corpus-side adversaries already work at scale.** Zhong et al. (EMNLP 2023, corpus poisoning) show that $\sim$50 adversarial passages inserted into a corpus of millions are retrieved for $>90\%$ of queries in some domains for dense retrievers. PoisonedRAG (Zou et al., USENIX Security 2025) reaches $\sim$90% attack success injecting 5 texts per target question into millions. These attack the *embedding geometry*, which is upstream of the index, and they demonstrate that "the query distribution is benign" is false in deployment.
- **Adaptivity provably breaks sketch-based data structures.** Ben-Eliezer, Jayaram, Woodruff, Yogev (PODS 2020) separate oblivious from adaptive streaming; the same mechanism (attacker learns the seed from outputs) applies to LSH: an adversary who observes $O(\text{poly})$ query answers can locate hash-boundary queries.
- **Intrinsic dimension predicts fragility.** Amsaleg et al. (WIFS 2017) show the perturbation needed to change a nearest neighbor shrinks as local intrinsic dimensionality grows, roughly as $\epsilon \sim 1/\mathrm{LID}$ — measured on standard image/text feature sets.

## 5. What Is Not Known

- **Theoretically open.** Whether any index with $\tilde{O}(n)$ space and polylog query cost admits a per-query recall bound against an adversary who knows the seed. No proof either way. Also open: a lower bound showing that graph-based search with degree $O(\log n)$ *must* have instances with recall $o(1)$ for a constant fraction of queries in $\mathcal{Q}_\epsilon$ of real embedding sets.
- **Empirically open.** The white-box attack on HNSW/DiskANN at $n=10^9$ has not been run. Nothing prevents it: build the index, gradient-descend a query to maximize $\rho(q, \hat{N}_1) - \rho(q, N_1)$ through the (differentiable) embedding model and the (non-differentiable, but simulatable) search. Nobody has published the resulting $R^{\mathrm{adv}}$ curve versus $\bar R$.
- **Methodologically blocked.** $\inf_q \mathrm{rec}_k(q)$ is not estimable from samples and current attacks give only an upper bound on it. There is no accepted *certificate* — no analogue of randomized smoothing for retrieval — so "robust recall" has no measurement whose value can be reported and compared.

## 6. Why It Is Hard

The obstruction is **absent ground truth at adversarial scale, compounded by a metric that does not measure what it names**. Two concrete parts:

1. *Ground-truth cost.* Verifying recall for one adversarial query at $n=10^9$, $d=128$ is a full 512 GB scan. An attack loop needs $10^4$–$10^6$ such evaluations. Benign benchmarks amortize this with a fixed 10k-query ground-truth file computed once; an adaptive attacker generates new queries, so the file cannot be precomputed. This is a $10^2$–$10^3$× cost multiplier over standard benchmarking, not a constant factor.
2. *Metric misnaming.* "Recall@10 = 0.95" is a mean over a distribution the deployment does not sample from. It is not an upper bound on failure rate for any particular query, and it degrades gracefully in reporting while degrading catastrophically per query. Reporting the mean makes an index look robust exactly when its tail is worst — a heavier tail moves the mean by little.

Non-identifiability adds a third: when retrieval fails in a RAG system, embedding error, index error and reranker error are confounded. Without a per-query certificate you cannot attribute failure to the index.

## 7. Current Research (as of 2026)

- **Provable analysis of deployed graph indexes** — Indyk's group (MIT) after the NeurIPS 2023 result; the open direction is tightening the doubling-dimension/aspect-ratio conditions to something measurable on real embeddings.
- **Robust/filtered ANN at billion scale** — the BigANN challenge series (Simhadri, Microsoft; Aumüller, ITU Copenhagen) added streaming and filtered tracks; adversarial tracks are proposed but not standard *(frontier — verify)*.
- **RAG-security** — corpus poisoning and prompt-injection lines (Cornell, Duke, Illinois). Nearly all work attacks the *embedding*; the index-level attack surface is comparatively unstudied.
- **Adversarially robust data structures** — Yogev, Stemmer, Woodruff; differential-privacy-based robustness transfers for sketches. Applying it to LSH to bound seed leakage is an open, tractable direction *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** how far below $\bar R$ does $R^{\mathrm{adv}}$ fall for a standard graph index under a white-box, seed-known attack?

- **Scale.** MS MARCO passages ($8.8\times10^6$ passages, $d=768$, E5-base embeddings) — large enough that exhaustive ground truth per query is $\sim$30 GPU-ms, so $10^5$ attack queries are affordable ($\sim$1 GPU-hour of verification). HNSW with $M=32$, $\mathrm{efC}=200$, $\mathrm{efS}=100$; verify $\bar R_{10} \ge 0.95$ on the standard dev queries first.
- **Attack arm.** Start from each of 1,000 dev queries $q_0$; run projected gradient ascent in embedding space inside $\|q-q_0\|_2 \le \epsilon$ ($\epsilon$ = 5% of mean nearest-neighbor distance) on the surrogate objective $\rho(q,\hat N_1(q)) - \rho(q, N_1(q))$, using a differentiable relaxation of greedy graph search (soft top-$m$ over each visited neighborhood), 200 steps.
- **Control arms.** (a) random perturbation of the same norm $\epsilon$; (b) the same attack against IVF-Flat with $n_{\mathrm{probe}}$ tuned to equal $\bar R_{10}$; (c) the same attack against an LSH index with a *secret* seed regenerated per query batch.
- **Deciding number.** $R^{\mathrm{adv}}_{10}$ on the 1,000 attacked queries. If it falls below **0.5** while the random-perturbation control stays above **0.93**, benign recall is established as non-predictive of per-query recall and the measurement problem is real. If the attack cannot push HNSW below 0.9 at $\epsilon$ = 5%, the practical concern is largely dismissed at this scale and the burden moves to the corpus-side attack.

## 9. Key References

- **[Foundational]** P. Indyk, R. Motwani. *Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality.* STOC, 1998.
- **[Foundational]** A. Andoni, P. Indyk. *Near-Optimal Hashing Algorithms for Approximate Nearest Neighbor in High Dimensions.* FOCS, 2006.
- **[Theory]** A. Andoni, I. Razenshteyn. *Optimal Data-Dependent Hashing for Approximate Near Neighbors.* STOC, 2015.
- **[Lower bound]** A. Rubinstein. *Hardness of Approximate Nearest Neighbor Search.* STOC, 2018.
- **[SOTA — theory of deployed indexes]** P. Indyk, H. Xu. *Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations.* NeurIPS, 2023.
- **[SOTA — systems]** Y. Malkov, D. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI, 2020.
- **[SOTA — systems]** S. Jayaram Subramanya, F. Devvrit, R. Kadekodi, R. Krishnaswamy, H. Simhadri. *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS, 2019.
- **[SOTA — quantization]** R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, S. Kumar. *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* ICML, 2020.
- **[Benchmark]** M. Aumüller, E. Bernhardsson, A. Faithfull. *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.* Information Systems, 2020.
- **[Benchmark]** H. Simhadri et al. *Results of the NeurIPS'21 Challenge on Billion-Scale Approximate Nearest Neighbor Search.* PMLR (NeurIPS Competition Track), 2022.
- **[Adversarial]** Z. Zhong, Z. Huang, A. Wettig, D. Chen. *Poisoning Retrieval Corpora by Injecting Adversarial Passages.* EMNLP, 2023.
- **[Adversarial]** W. Zou, R. Geng, B. Wang, J. Jia. *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models.* USENIX Security, 2025.
- **[Adaptivity]** O. Ben-Eliezer, R. Jayaram, D. Woodruff, E. Yogev. *A Framework for Adversarially Robust Streaming Algorithms.* PODS, 2020.
- **[Fragility]** L. Amsaleg, J. Bailey, D. Barbe, S. Erfani, M. Houle, V. Nguyen, M. Radovanović. *The Vulnerability of Learning to Adversarial Perturbation Increases with Intrinsic Dimensionality.* IEEE WIFS, 2017.
- **[Survey]** S. Har-Peled, P. Indyk, R. Motwani. *Approximate Nearest Neighbor: Towards Removing the Curse of Dimensionality.* Theory of Computing, 2012.

## 10. Worked Example

Take SIFT1M ($n = 10^6$, $d=128$), HNSW with $M=16$, $\mathrm{efSearch}=64$. Published ANN-Benchmarks operating point: $\bar R_{10} \approx 0.95$ at $\sim$8,000 QPS single-core.

Suppose — the empirically plausible shape — that the recall distribution is bimodal: 97% of queries at recall 0.98, 3% of queries at recall 0.05.
$$\bar R_{10} = 0.97(0.98) + 0.03(0.05) = 0.9520.$$
Now change the tail mass to 3% at recall 0.00 (total failure):
$$\bar R_{10} = 0.97(0.98) = 0.9506.$$
The mean moves by **0.0014** — smaller than run-to-run variance from build seed. A 3% total-failure population is invisible in the headline number.

Downstream, in an agent making 20 retrieval calls per task, the per-task probability that at least one call lands in that 3% is
$$1 - (1-0.03)^{20} = 0.456.$$
Nearly half of tasks touch a failed retrieval, from an index reported at 95% recall. And if an attacker can *choose* which queries land in the bad set — which is exactly what the $\epsilon$-ball attack in §8 tries to do — the rate is 100%, not 3%.

The obstruction is visible here: the reported number is insensitive to precisely the quantity that determines agent-level reliability, and no cheap estimator of the tail exists, because finding the tail requires either exhaustive ground truth over a very large query sample or a search over query space that is itself the unsolved attack problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*