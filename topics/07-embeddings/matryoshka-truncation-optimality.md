---
id: 07-embeddings/matryoshka-truncation-optimality
title: "Matryoshka Truncation Optimality"
topic: 07-embeddings
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Matryoshka Truncation Optimality

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/matryoshka-truncation-optimality` · **Status:** empirically-open

## 1. Problem Statement

Matryoshka Representation Learning (MRL) trains a $d$-dimensional encoder so that every prefix of the output vector is itself a usable embedding. Users then truncate: keep the first $m$ coordinates, drop the rest, re-normalize, index.

The question is whether the prefix is the *right* $m$ dimensions.

Three variants, different difficulty:

- **Measurement.** Given a fixed trained MRL encoder, is prefix truncation to $m$ dims as good as the best $m$-dimensional linear map of the same full embedding (PCA, learned projection, oracle subspace)? This is cheap to test and mostly untested.
- **Method.** Does an MRL-trained encoder truncated to $m$ match a model trained from scratch at width $m$ under an equal-compute, equal-data budget? Published comparisons exist but are not equal-compute.
- **Theory.** Under what conditions on the data distribution and loss is a nested (prefix-monotone) family simultaneously optimal at every $m \in \mathcal{M}$? No characterization exists. The obvious analogue — Eckart–Young–Mirsky — applies to reconstruction error under an orthogonal basis, not to a contrastive retrieval objective under an unconstrained basis.

Solving it means: a bound or a measured curve for the *truncation gap* $\Delta_m$ below, with the confounds in §6 controlled.

## 2. Formal Setting

Encoder $f_\theta:\mathcal{X}\to\mathbb{R}^d$, nesting set $\mathcal{M}=\{m_1<\dots<m_K=d\}$, typically $\{32,64,128,256,512,768\}$. Prefix operator $\pi_m(z)=z_{1:m}$, and $\hat\pi_m(z)=\pi_m(z)/\|\pi_m(z)\|_2$ since retrieval uses cosine.

MRL objective, as trained:
$$\mathcal{L}_{\mathrm{MRL}}(\theta)=\sum_{m\in\mathcal{M}} c_m\,\mathcal{L}_{\mathrm{InfoNCE}}\big(\hat\pi_m\circ f_\theta\big),\qquad c_m>0 .$$
In Kusupati et al. (2022) and in every open reproduction, $c_m=1$ for all $m$. That uniform choice is a convention, not a derived optimum.

**Quality**, measured: $Q_m = \mathrm{nDCG@10}$ on a held-out retrieval suite (BEIR subsets, or MTEB Retrieval), with the corpus re-encoded and re-indexed at width $m$ — exact search, no ANN, so index error does not enter.

**Truncation gap** (representational, within one model):
$$\Delta^{\mathrm{lin}}_m \;=\; \max_{W\in\mathbb{R}^{m\times d}} Q\big(\widehat{Wf_\theta}\big)\;-\;Q\big(\hat\pi_m\circ f_\theta\big).$$
Measured by fitting $W$ on held-in data (PCA, or a linear head trained with the same contrastive loss, encoder frozen) and scoring on held-out data.

**Bespoke gap** (across models, equal budget $C$ FLOPs and equal token count):
$$\Delta^{\mathrm{bes}}_m \;=\; Q\big(f^{(m)}_{\theta'}\big)\;-\;Q\big(\hat\pi_m\circ f_\theta\big),$$
where $f^{(m)}$ is the same architecture with output width $m$ trained standalone.

**Full-width tax:** $\tau = Q(f^{\mathrm{plain}}_d) - Q(f^{\theta}_d)$, the cost MRL pays at full width for being nestable.

Assumptions, and which fail:

1. *Cosine similarity is the deployed metric.* Holds for text retrieval; fails for reranking pipelines where truncation error is absorbed downstream.
2. *Prefix coordinates are ordered by importance.* Not enforced by anything — the loss is invariant to any rotation applied within a block only if block boundaries are respected, and $\mathcal{L}$ imposes no orthogonality between coordinate $i$ and $j$. **Known violated:** MRL embeddings are anisotropic and coordinates are correlated (Ethayarajh, EMNLP 2019, for the general phenomenon).
3. *Nesting is free at full width.* $\tau\approx 0$ is claimed, rarely reported with seeds.
4. *nDCG@10 differences of the size at stake ($\sim$1 point) exceed seed noise.* Not established for MTEB; run-to-run variance is usually unreported.

## 3. State of the Art

**Established.** Kusupati et al., *Matryoshka Representation Learning*, NeurIPS 2022 (arXiv:2205.13147): a single ResNet50/BERT trained with the nested loss gives, per the paper, up to $14\times$ smaller embeddings at equal ImageNet-1K top-1 versus fixed-feature baselines, and up to $2\%$ absolute gain on long-tail few-shot. The comparison arm is *post-hoc truncation of a non-MRL model* and *independently trained low-dim models*, not an optimal linear projection of the MRL model itself.

**Empirical/systems SOTA.** MRL is shipped: OpenAI `text-embedding-3` (Jan 2024, `dimensions` parameter), Nomic Embed (Nussbaum et al., 2024, arXiv:2402.01613), Jina Embeddings v3 (Sturua et al., 2024, arXiv:2409.10173), Gemini Embedding (Lee et al., 2025, arXiv:2503.07891). AdANNS (Kusupati et al., NeurIPS 2023, arXiv:2305.19435) uses low-dim prefixes for coarse ANN quantization and full width for reranking — the strongest evidence that prefixes are *useful*, not that they are *optimal*.

**Claimed but unablated.** (a) That truncation loses "little" quality — reported as one MTEB number per width, single seed, no PCA control. (b) That uniform $c_m$ is fine — no published sweep of $c_m$ against a measured Pareto front. (c) That MRL is free at full width — $\tau$ is reported as a single-seed delta in most model cards. (d) 2D/ESE variants nesting over both depth and width (Li et al., *ESE: Espresso Sentence Embeddings*, 2024, arXiv:2402.14776) report gains at small width but do not isolate the truncation question.

**Benchmark-number-only.** OpenAI's published table (`text-embedding-3-large`: MTEB 64.6 at 3072 dims, 62.0 at 256 dims; `ada-002`: 61.0 at 1536) is a blog table with no seeds, no ablation, and no PCA arm.

## 4. What Is Known

- Nested training beats post-hoc truncation of an ordinarily trained encoder by a large margin at small $m$ — ResNet50 / ImageNet-1K, 8–2048 dims, and BERT-base scale (Kusupati et al. 2022). This is the one robustly reproduced result.
- Quality degrades gracefully and monotonically in $m$ across every released MRL model; at production scale, `text-embedding-3-large` retains 62.0/64.6 MTEB ($-2.6$ points, $-4\%$ relative) at $256/3072$ dims, i.e. $12\times$ compression.
- Low-dim prefixes are good enough for *coarse* search: AdANNS reports matching or beating fixed-dim ANN baselines at equal memory on ImageNet and Natural Questions (NeurIPS 2023).
- Classical anchor: for squared-reconstruction error, the optimal rank-$m$ map is the top-$m$ PCA subspace (Eckart & Young, *Psychometrika* 1936; Mirsky 1960). A coordinate prefix attains this bound only if the learned basis coincides with the eigenbasis, which nothing in $\mathcal{L}_{\mathrm{MRL}}$ enforces.
- Random projection gives $\mathcal{O}(\varepsilon^{-2}\log n)$ dimensions for $(1\pm\varepsilon)$ pairwise-distance preservation (Johnson–Lindenstrauss, 1984) — a floor showing that useful $m$ can be far below $d$ for reasons unrelated to MRL.

## 5. What Is Not Known

- **Empirically open.** $\Delta^{\mathrm{lin}}_m$ — the gap between prefix truncation and the best linear $m$-dim map of the *same* MRL embedding. The experiment is a few GPU-days. Nobody has published it with seeds across widths. Likewise $\Delta^{\mathrm{bes}}_m$ under matched compute, and the $c_m$ sweep.
- **Theoretically open.** Whether a single parameterization can be simultaneously Bayes-optimal at all $m\in\mathcal{M}$ for a contrastive objective, or whether there is a strict frontier trade-off $\sum_m \alpha_m \Delta_m \ge B(\mathcal{M},\text{data})$. No lower bound is known even for Gaussian mixtures with $|\mathcal{M}|=2$.
- **Methodologically blocked.** "The first $m$ coordinates carry the most information" is not a well-posed claim under the loss's invariances: any invertible map that preserves each prefix subspace leaves $\mathcal{L}_{\mathrm{MRL}}$ unchanged, so coordinate importance is identifiable only up to within-block transformations. Attribution methods that score individual dimensions inherit this non-identifiability.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** Every published MRL-vs-baseline comparison varies training compute, data mixture, and loss weighting at the same time as width. A $-2.6$ MTEB point drop at $m=256$ cannot be attributed to *prefix truncation* rather than to *dimensionality* without the PCA arm, which is the one arm nobody runs.
2. **Non-identifiability of the basis.** The objective constrains nested *subspaces*, not coordinates. So "is the prefix optimal?" must be asked as "is the prefix subspace optimal?", and the two questions get conflated in reporting.
3. **Evaluation that does not measure the named thing.** MTEB averages 50+ heterogeneous tasks; a 1-point average move can come entirely from two classification tasks while retrieval nDCG is flat. Seed variance on individual BEIR sets is often $\pm0.3$–$0.7$ nDCG@10 and is not reported, so the effect size at stake sits near the noise floor.

Compute is *not* the obstruction here — that is what makes this empirically open rather than blocked.

## 7. Current Research (as of 2026)

- Adaptive-retrieval systems using prefixes as a cheap first stage and full width for rerank — the AdANNS line (Kusupati, Jain, Farhadi; UW/Google) and its successors in production vector databases.
- Post-hoc "matryoshka-fication" of frozen encoders via learned low-rank adapters, e.g. Google's Matryoshka-Adaptor line (2024) *(frontier — verify identifiers)*.
- Joint width×depth nesting (ESE / 2D-Matryoshka) and Matryoshka-style nesting in multimodal token budgets *(frontier — verify)*.
- Quantization-aware nesting: whether $m=256$ at int8 beats $m=1024$ at 2-bit at equal bytes. Actively discussed by vector-DB vendors; no controlled public study.

## 8. Concrete Next Experiment

**Scale.** One backbone, BERT-base or Qwen3-0.6B-Embedding class, $d=768$. Train on a fixed 20M-pair contrastive mixture, identical schedule, 5 seeds. Evaluate exact search on 6 BEIR sets (SciFact, NFCorpus, FiQA, TREC-COVID, ArguAna, SCIDOCS), nDCG@10.

**Arms** (all at $m\in\{32,64,128,256,768\}$):
- A: MRL prefix truncation (the treatment).
- B: **control** — same MRL checkpoint, frozen, top-$m$ PCA of full embeddings fit on 1M held-in corpus vectors.
- C: same checkpoint, frozen, linear $W\in\mathbb{R}^{m\times768}$ trained with InfoNCE on held-in data.
- D: bespoke width-$m$ encoder, equal FLOPs and equal tokens.

**Deciding number.** $\Delta^{\mathrm{lin}}_{64} = \max(Q_B,Q_C)-Q_A$ at $m=64$, averaged over 6 sets and 5 seeds, with a paired bootstrap 95% CI.

- $\Delta^{\mathrm{lin}}_{64} \le 0.5$ nDCG points with CI excluding 1.0 → prefix truncation is effectively optimal within the model; close the measurement variant, keep the theory variant open.
- $\Delta^{\mathrm{lin}}_{64} \ge 1.5$ points → truncation is leaving measurable quality on the table, and every deployed MRL API should ship a projection matrix instead of a slice. This is a shipping-relevant outcome from roughly 200 GPU-hours.

## 9. Key References

- **[Foundational]** Kusupati, Bhatt, Rege, Wallingford, Sinha, Ramanujan, Howard-Snyder, Chen, Kakade, Jain, Farhadi. *Matryoshka Representation Learning.* NeurIPS, 2022. — arXiv:2205.13147
- **[Foundational]** Eckart, Young. *The approximation of one matrix by another of lower rank.* Psychometrika, 1936.
- **[Foundational]** Johnson, Lindenstrauss. *Extensions of Lipschitz mappings into a Hilbert space.* Contemporary Mathematics, 1984.
- **[SOTA]** Kusupati, Wallingford, Ramanujan, Somani, Park, Pillutla, Jain, Kakade, Farhadi. *AdANNS: A Framework for Adaptive Semantic Search.* NeurIPS, 2023. — arXiv:2305.19435
- **[SOTA]** Nussbaum, Morris, Duderstadt, Mulyar. *Nomic Embed: Training a Reproducible Long Context Text Embedder.* 2024. — arXiv:2402.01613
- **[SOTA]** Sturua, Mohr, Akram, Günther, Wang, et al. *jina-embeddings-v3: Multilingual Embeddings With Task LoRA.* 2024. — arXiv:2409.10173
- **[SOTA]** Lee, Dai, Duddu, Ren, Zhang, et al. *Gemini Embedding: Generalizable Embeddings from Gemini.* 2025. — arXiv:2503.07891
- **[Related]** Li, Li. *ESE: Espresso Sentence Embeddings.* 2024. — arXiv:2402.14776
- **[Survey]** Muennighoff, Tazi, Magne, Reimers. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023. — arXiv:2210.07316
- **[Related]** Ethayarajh. *How Contextual are Contextualized Word Representations?* EMNLP, 2019. — arXiv:1909.00512

## 10. Worked Example

Take the one real production data point: `text-embedding-3-large`, MTEB 64.6 at 3072 dims, 62.0 at 256 dims (OpenAI, Jan 2024). Compare against `text-embedding-3-small`, 62.3 at 1536 dims.

Index 1M documents, fp32:

| Option | dims | bytes/vec | index size | MTEB |
|---|---|---|---|---|
| 3-large full | 3072 | 12,288 | 12.3 GB | 64.6 |
| 3-small full | 1536 | 6,144 | 6.1 GB | 62.3 |
| 3-large @ 256 | 256 | 1,024 | 1.0 GB | 62.0 |

Truncation buys $6\times$ less memory than the small model at $-0.3$ MTEB. That is the case for MRL, and it is real.

Now the obstruction. The drop from 64.6 to 62.0 is $2.6$ points. Decompose it:
$$2.6 \;=\; \underbrace{\big[Q_{3072}-Q^{\mathrm{PCA}}_{256}\big]}_{\text{cost of 256 dimensions}} \;+\; \underbrace{\Delta^{\mathrm{lin}}_{256}}_{\text{cost of using the \emph{prefix}}}.$$
Nobody has published $Q^{\mathrm{PCA}}_{256}$ for this model — the API returns slices, and the projection matrix is not exposed, so an external party cannot even fit PCA on the full-width outputs without paying to re-encode a corpus at 3072 dims. The two terms are unseparated at every scale where it matters.

Suppose $\Delta^{\mathrm{lin}}_{256}=1.0$ point of the 2.6 (a value fully consistent with published evidence, since no arm rules it out). Then shipping a $256\times3072$ projection matrix — 786K floats, 3 MB, computed once — would return `3-large@256` to 63.0 MTEB at identical serving cost and identical index size. That is a larger gain than most embedding model releases claim, obtainable by a matrix multiply.

Or $\Delta^{\mathrm{lin}}_{256}=0.0$ and the prefix is already optimal, in which case the entire family of "learned re-projection" proposals is dead.

The two worlds are distinguished by one number, and that number has never been measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*