---
id: 06-data-pipeline/optimal-deduplication-granularity
title: "Optimal Deduplication Granularity for Pretraining Corpora"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Deduplication Granularity for Pretraining Corpora

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/optimal-deduplication-granularity` · **Status:** empirically-open

## 1. Problem Statement

Deduplication of web-scale pretraining corpora is universal practice, but the *granularity* at which to do it is chosen by convention, not by evidence. Granularity has three independent axes:

- **Unit** — what object is compared: the document, the paragraph, a fixed-length token substring, or a semantic embedding.
- **Threshold** — how similar two units must be to count as duplicates.
- **Scope** — the partition of the corpus within which duplicates are collapsed: per-snapshot, per-source, or global.

**Input:** a raw corpus $\mathcal{D}$, a compute budget $C$, a model family, and a downstream evaluation suite.
**Output:** a granularity configuration $g = (\text{unit}, \tau, \text{scope})$ and the induced filtered corpus $\mathcal{D}_g$.
**Objective:** minimise final downstream loss under fixed $C$.

Three variants, of very different difficulty:

- **Measurement.** Given $g$, can we predict the quality of $\mathcal{D}_g$ without training on it? Currently no — every proxy (perplexity of a small model, corpus entropy, duplicate rate) has been shown to rank configurations differently from downstream loss.
- **Method.** Find $g^\star$ by search. Runnable, expensive, not run at frontier scale as a clean sweep.
- **Theory.** Characterise the optimal repetition profile of a training distribution as a function of model size and token budget. Open; the closest results are epoch-repetition scaling laws, which treat repetition as uniform over the corpus and so answer a strictly easier question.

Solving it means: a rule that maps $(\mathcal{D}, C, N)$ to $g^\star$ and is validated by held-out training runs, not a fixed recipe that happens to work on one crawl.

## 2. Formal Setting

Corpus $\mathcal{D} = \{d_1,\dots,d_M\}$ of documents. A granularity induces a segmentation into units $u \in U(\mathcal{D})$ and a similarity $\mathrm{sim}: U \times U \to [0,1]$.

**Similarity as measured.** For MinHash near-dup, $\mathrm{sim}$ is the Jaccard index over $n$-gram shingles,
$$J(u,v) = \frac{|S_n(u)\cap S_n(v)|}{|S_n(u)\cup S_n(v)|},$$
estimated from $h$ permutation hashes and thresholded by LSH banding into $b$ bands of $r$ rows ($h=br$). The *actually measured* quantity is not $J \ge \tau$ but the collision probability
$$P_{\text{dup}}(s) = 1-(1-s^{r})^{b},$$
a soft sigmoid in $s$, not a step at $\tau$. Reported thresholds ("0.75") name the inflection point of this curve, not a guarantee.

For exact-substring dedup the unit is any token span of length $\ge L$ (typically $L=50$), found via a suffix array; $\mathrm{sim} \in \{0,1\}$. For semantic dedup the unit is a document embedding $\phi(d)$ and $\mathrm{sim}$ is cosine within a $k$-means cluster.

**Multiplicity.** For unit $u$, let $m(u)$ be its count in $\mathcal{D}$. The corpus repetition profile is the distribution of $m$; it is heavy-tailed, roughly Zipfian. Dedup at granularity $g$ maps $m \mapsto \min(m, \kappa_g(u))$ for some cap implied by the scope — global scope gives $\kappa=1$, per-snapshot scope over $K$ snapshots gives $\kappa \le K$.

**Objective.** With $N$ parameters and $T$ tokens under $C \approx 6NT$,
$$g^\star = \arg\min_g \; \mathbb{E}_{x\sim \mathcal{P}_{\text{eval}}}\big[\ell(x;\theta_C(\mathcal{D}_g))\big],$$
where $\theta_C$ is the parameters after training to budget $C$. Note $|\mathcal{D}_g|$ varies with $g$, so at fixed $C$ aggressive dedup forces more epochs over less data — granularity and repetition are coupled, not separable.

**Assumptions, and which are violated.**
1. *Duplicate detection approximates semantic redundancy.* Violated: boilerplate licences and re-derived proofs are both "duplicates" but differ in value.
2. *Removal is value-neutral in aggregate.* Violated: FineWeb found global dedup removed high-quality repeated content and up-weighted the residual low-quality tail.
3. *Transitivity of near-duplication.* Violated: $J$-thresholded graphs have connected components far larger than any true dup cluster, so component-level removal deletes non-duplicates.
4. *Evaluation set is disjoint from training data.* Approximately violated at web scale; contamination interacts with dedup scope.

## 3. State of the Art

**Established (ablated, reproduced).**
- Removing exact and near duplicates does not hurt, and slightly helps, held-out perplexity while cutting memorised emission by roughly an order of magnitude (Lee et al., ACL 2022; Kandpal et al., ICML 2022). This is the one directional result that has replicated.
- Repetition of *specific* small subsets is disproportionately harmful: Hernandez et al. (2022) show a repeated 0.1% fraction can cost the equivalent of half the model's parameters.

**Claimed but unablated.**
- That aggressive semantic dedup is free: SemDeDup (Abbas et al., 2023) and D4 (Tirumala et al., NeurIPS 2023) report gains, but the ablations vary embedding model, cluster count, and removal rate together, so the granularity contribution is not isolated.
- That "global is better than local" — widely assumed in pipeline design, and contradicted by the FineWeb ablation.

**Benchmark-number-only.**
- Most production corpora (RefinedWeb, Dolma, RedPajama-v2, Nemotron-CC) publish one granularity configuration and its downstream scores. There is no sweep: a single point per corpus, with tokenizer, filters, and mixture all differing between them. These numbers cannot be used to rank granularities.

**Theory SOTA** is the data-constrained scaling law of Muennighoff et al. (NeurIPS 2023), which models return-to-repetition with an exponential decay in epoch count. It assumes uniform repetition and therefore says nothing about *which* units to deduplicate.

## 4. What Is Known

- **C4 duplication.** Lee et al. (2022) found a 61-word sequence appearing 61,036 times in C4 despite C4's original 3-sentence-span dedup. `NearDup` (MinHash) removed 3.04% of C4 documents; `ExactSubstr` ($L=50$) removed 7.18% of C4 tokens. Models: 1.5B parameters.
- **Memorisation is log-linear in duplicate count.** Carlini et al. (ICLR 2023), 125M–6B GPT-Neo: extraction rate grows roughly linearly in $\log m$. Kandpal et al. (2022): a sequence seen 10 times is emitted about $10^3$ times more often than one seen once.
- **Repetition budget.** Muennighoff et al. (2023), up to 9B parameters and 900B tokens: up to ~4 epochs of repeated data is nearly as valuable as fresh data; marginal value is near zero by ~16 epochs.
- **Scope matters and the sign is not the assumed one.** FineWeb (Penedo et al., NeurIPS D&B 2024): MinHash (5-grams, 112 hashes, 14 bands × 8 rows, ~0.75 threshold) applied *per CommonCrawl dump* beat the same method applied *globally across 96 dumps*, measured on 1.71B-parameter, ~28B-token ablation runs. Global dedup removed ~a further large fraction of tokens and lowered downstream accuracy.
- **Scale of removal by unit.** Document-level fuzzy dedup on CommonCrawl typically removes 30–60% of documents; substring-level removes single-digit to low-tens percent of *tokens* from an already document-deduped corpus. The two are not interchangeable.

## 5. What Is Not Known

- **Empirically open.** The clean sweep — unit × threshold × scope, ≥3 levels each, one trained model per cell, held-out eval, at ≥1B parameters — has not been published. Every ingredient exists. Nobody has paid for the grid.
- **Empirically open.** Whether $g^\star$ depends on $N$ and $T$. Plausible that larger models tolerate (or need) more repetition, but no cross-scale sweep exists, so extrapolating a 1B ablation to a 70B run is unjustified.
- **Methodologically blocked.** There is no accepted measurement separating "duplicate" from "legitimately repeated, high-value". Both fall under $J \ge \tau$. Without ground truth on which repeats are valuable, the search has no objective except full training runs.
- **Theoretically open.** No proof characterising the optimal multiplicity cap $\kappa^\star(u)$ as a function of a unit's information content and the model's capacity. The double-descent phenomenon from repeated subsets (Hernandez et al.) has no theory that predicts its onset from corpus statistics.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** Changing granularity changes corpus *size*, so at fixed compute it also changes epoch count and the effective mixture over sources. A quality delta between two granularities is a sum of at least three effects; no published ablation holds token count, epoch count, and source mixture fixed simultaneously.
2. **Absent ground truth.** "Redundant" has no label. The only ground truth is downstream loss, which costs one pretraining run per configuration — roughly $6NT$ FLOPs each, and the grid is multiplicative in three axes.
3. **Evaluation that does not measure what it names.** Dedup is justified by memorisation reduction and by benchmark accuracy. These are different objectives with different optima: memorisation is monotone-improving in aggressiveness, downstream accuracy is not. Papers that report only one of the two cannot identify $g^\star$.

## 7. Current Research (as of 2026)

- **Scope-aware pipelines.** Post-FineWeb, per-snapshot dedup with cross-snapshot filtering is becoming the default in open corpora (HuggingFace, AI2/Dolma, NVIDIA Nemotron-CC). The justification remains a single ablation at 1.7B parameters. *(frontier — verify)*
- **Fuzzy/semantic hybrids.** Embedding-cluster dedup layered on MinHash, following SemDeDup and D4 (Meta AI, Stanford). Open question is whether the semantic layer adds anything once thresholds are matched for removal rate.
- **Value-aware repetition.** Instead of removing duplicates, re-weighting multiplicity by a learned quality score — mixing dedup with quality classifiers (FineWeb-Edu, Nemotron-CC synthetic rephrasing). This reframes the problem as choosing $\kappa_g(u)$ per unit rather than a global threshold. *(frontier — verify)*
- **Contamination-driven scope.** Regulatory and benchmark-integrity work pushing toward global exact-substring dedup against eval sets specifically, decoupled from quality-driven dedup.

## 8. Concrete Next Experiment

**Scale.** 1.4B-parameter decoder, 30B training tokens per arm (~$2.5\times10^{20}$ FLOPs, ≈3–4 h on 64 H100s). Source pool: 12 CommonCrawl snapshots, ~600B raw tokens, identically filtered before the dedup stage.

**Grid (12 arms).** unit ∈ {document-MinHash, paragraph-MinHash, exact-substring $L=50$} × scope ∈ {per-snapshot, global} × threshold calibrated so that each (unit, scope) pair is run at two settings matched to **the same token removal rate** (25% and 50%). Removal-rate matching is the design element missing from prior work: it removes the size confound.

**Control arm.** No dedup beyond the pool's base filtering, trained on the same 30B tokens sampled from the full pool. Second control: random removal of 25%/50% of tokens, which isolates "dedup" from "less data".

**Fixed.** Tokenizer, architecture, LR schedule, token count, epoch count (all arms see 30B tokens, ≤1 epoch of their filtered pool).

**Deciding number.** Mean accuracy on a held-out suite (HellaSwag, ARC-C, MMLU, OpenBookQA, PIQA), reported as the delta against the random-removal control at the same removal rate. If no (unit, scope) cell beats random removal by more than $0.5$ points — larger than the seed-to-seed spread, which should be measured with 3 seeds on the control — then granularity choice is not the lever the field believes it is, and the result is publishable either way. Secondary number: extractable-memorisation rate at 50-token prefixes, to confirm the two objectives diverge.

## 9. Key References

- **[Foundational]** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. *Deduplicating Training Data Makes Language Models Better.* ACL 2022. — arXiv:2107.06499
- **[Foundational]** Andrei Z. Broder. *On the Resemblance and Containment of Documents.* SEQUENCES, 1997.
- **[SOTA]** Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, Thomas Wolf. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024 Datasets & Benchmarks. — arXiv:2406.17557
- **[SOTA]** Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, Julien Launay. *The RefinedWeb Dataset for Falcon LLM.* NeurIPS 2023 Datasets & Benchmarks. — arXiv:2306.01116
- Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, et al. *Scaling Laws and Interpretability of Learning from Repeated Data.* 2022. — arXiv:2205.10487
- Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR 2023. — arXiv:2202.07646
- Nikhil Kandpal, Eric Wallace, Colin Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML 2022. — arXiv:2202.06539
- Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos. *SemDeDup: Data-Efficient Learning at Web-Scale through Semantic Deduplication.* 2023. — arXiv:2303.09540
- Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari S. Morcos. *D4: Improving LLM Pretraining via Document De-Duplication and Diversification.* NeurIPS 2023 Datasets & Benchmarks. — arXiv:2308.12284
- Luca Soldaini, Rodney Kinney, Akshita Bhagia, et al. *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research.* ACL 2024. — arXiv:2402.00159
- Yanai Elazar, Akshita Bhagia, Ian Magnusson, et al. *What's In My Big Data?* ICLR 2024. — arXiv:2310.20707
- **[Survey]** Alon Albalak, Yanai Elazar, Sang Michael Xie, et al. *A Survey on Data Selection for Language Models.* TMLR, 2024. — arXiv:2402.16827

## 10. Worked Example

Take FineWeb's published MinHash setting: 5-gram shingles, $h=112$ hashes, $b=14$ bands, $r=8$ rows, nominal threshold $0.75$.

Actual detection probability from $P_{\text{dup}}(s)=1-(1-s^{8})^{14}$:

| true Jaccard $s$ | $s^8$ | $P_{\text{dup}}(s)$ |
|---|---|---|
| 0.50 | 0.0039 | 0.054 |
| 0.75 | 0.100 | 0.772 |
| 0.80 | 0.168 | 0.925 |
| 0.90 | 0.430 | 0.9996 |

So "threshold 0.75" means: 23% of true 0.75-duplicates survive, and 5% of pairs at $s=0.5$ — which are not duplicates in any editorial sense — are removed. The unit of the decision is a stochastic event, not a predicate.

Now push it through scope. Suppose a news wire story appears once in each of 12 snapshots at pairwise $s\approx0.93$ (boilerplate headers differ). Per-snapshot scope: multiplicity $m=12$ survives, near the "4 epochs is fine, 16 is not" boundary from Muennighoff et al. Global scope: $P_{\text{dup}}\approx 1.0$ for each pair, the component collapses to $m=1$. Twelve-fold reduction for that document.

Repeat this over the whole crawl and the global pass removes a large additional token fraction. The tokens it removes are exactly the ones that survived many crawls — the *persistently republished* material, which correlates with being edited, linked, and useful. What remains is the ephemeral tail. This is the mechanism FineWeb's ablation surfaced: the more principled-sounding scope produced the worse model at 1.71B parameters.

The obstruction is now visible. Nothing in the similarity computation distinguishes "republished because valuable" from "republished because spam". Both are $s\approx0.93$. To tell them apart you need the downstream number — and the 12-arm grid in §8 costs roughly $3\times10^{21}$ FLOPs, about 5,000 H100-hours, to produce a single row of the table the field has been assuming for four years.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*