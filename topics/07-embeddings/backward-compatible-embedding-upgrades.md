---
id: 07-embeddings/backward-compatible-embedding-upgrades
title: "Backward-Compatible Embedding Model Upgrades"
topic: 07-embeddings
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Backward-Compatible Embedding Model Upgrades

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/backward-compatible-embedding-upgrades` · **Status:** partially-solved

## 1. Problem Statement

A deployed retrieval or matching system holds an index of $N$ vectors produced by an old encoder $\phi_{\text{old}}$. A better encoder $\phi_{\text{new}}$ arrives. Re-encoding the whole index ("backfill") is often impossible: the source documents may be deleted, licence-restricted, or spread over systems with no batch path; the ANN structure must be rebuilt; and during any rolling migration the index is *mixed*.

**Input:** $\phi_{\text{old}}$, its index $\{\phi_{\text{old}}(x_i)\}_{i=1}^N$, a training set $D$, a compute budget.
**Output:** $\phi_{\text{new}}$ (optionally plus a light map $\psi$) such that queries embedded by $\phi_{\text{new}}$ retrieve correctly against old vectors.
**Decision predicate:** the *empirical compatibility criterion* of Shen et al. (CVPR 2020) — cross-space accuracy must beat old/old accuracy — with no loss of new/new accuracy relative to an unconstrained $\phi_{\text{new}}$.

Three variants, routinely conflated:

- **Measurement.** What does "compatible" mean beyond one cross-test number? Score calibration, threshold transfer, rank stability, and per-query regression are distinct and mostly unmeasured.
- **Method.** Train $\phi_{\text{new}}$ under a compatibility constraint, or learn $\psi$ post hoc from old to new space.
- **Theory.** Is there an inherent accuracy tax for compatibility? No non-trivial lower bound exists.

## 2. Formal Setting

Encoders $\phi:\mathcal{X}\to\mathbb{S}^{d-1}$ (L2-normalised). Retrieval uses inner product $s(q,x)=\langle\phi(q),\phi(x)\rangle$. Write $M(\phi_q,\phi_g)$ for a task metric evaluated with query encoder $\phi_q$ and gallery encoder $\phi_g$ — mAP, Recall@$k$, or TAR@FAR for verification. All four cells are measurable on a held-out set:

$$M_{\text{oo}}=M(\phi_{\text{old}},\phi_{\text{old}}),\quad M_{\text{nn}}=M(\phi_{\text{new}},\phi_{\text{new}}),\quad M_{\text{no}}=M(\phi_{\text{new}},\phi_{\text{old}}),\quad M_{\text{on}}=M(\phi_{\text{old}},\phi_{\text{new}}).$$

**Compatibility (as measured):** $M_{\text{no}} > M_{\text{oo}}$.
**Compatibility tax:** $\tau = M_{\text{nn}}^{\ast} - M_{\text{nn}}$, where $M_{\text{nn}}^{\ast}$ is the same architecture and data trained without the constraint. $\tau$ is the number practitioners actually trade against, and it is reported far less often than $M_{\text{no}}$.

**Regression rate** (Yan et al., CVPR 2021): fraction of queries correct under old and wrong under new,
$$\mathrm{NFR}=\frac{1}{|Q|}\sum_{q\in Q}\mathbf{1}[\text{correct}_{\text{old}}(q)\wedge\neg\,\text{correct}_{\text{new}}(q)].$$
Note $M_{\text{nn}}>M_{\text{oo}}$ with $\mathrm{NFR}>0$ is normal; aggregate gain hides per-query loss.

**Mixed-index objective.** With backfilled fraction $\rho\in[0,1]$ the served metric is not $\rho M_{\text{nn}}+(1-\rho)M_{\text{no}}$, because a single ranked list draws candidates from both spaces. Define
$$M_{\text{mix}}(\rho)=\mathbb{E}_{S\sim \text{Bern}(\rho)^N}\big[M(\phi_{\text{new}}, \text{index}_S)\big],$$
which requires score *comparability* across spaces, not just per-space ordering. This is the quantity deployments care about and the one almost nobody reports.

**Assumptions and their violations.**
1. *Fixed dimension* $d_{\text{old}}=d_{\text{new}}$ — violated by real upgrades (768→3072 in OpenAI's `text-embedding-3-large`); handled only by projection or Matryoshka-style nesting.
2. *Stationary data* — violated; new models are trained partly to fix distribution shift, so $M_{\text{oo}}$ on the new eval set is itself unstable.
3. *Old model available at training time* — violated for third-party API embeddings, where only vectors exist.
4. *Single old model* — violated after two upgrades; compatibility is not transitive and the constraint set grows.

## 3. State of the Art

**Established (ablated, reproduced).**
- **BCT** — Shen, Xiong, Xia, Soatto, *Towards Backward-Compatible Representation Learning*, CVPR 2020. Trains $\phi_{\text{new}}$ with an *influence loss*: the new features must remain classifiable by the **frozen old classifier head**. Reliably achieves $M_{\text{no}}>M_{\text{oo}}$ on face verification (IJB-C) and person re-ID. Its cost is a nonzero $\tau$, and BCT needs the old *classifier*, not just old vectors.
- **LCE** — Meng et al., *Learning Compatible Embeddings*, ICCV 2021. Replaces the frozen-head constraint with alignment plus boundary losses; matches or beats BCT compatibility at lower $\tau$.
- **FCT** — Ramanujan, Vasu, Farhadi, Tuzel, Pouransari, *Forward Compatible Training for Large-Scale Embedding Retrieval Systems*, CVPR 2022. Inverts the problem: the *old* model emits side-information, and a cheap learned transform maps old→new at upgrade time. Removes the constraint on $\phi_{\text{new}}$ entirely ($\tau=0$) at the cost of planning ahead and storing extra bytes per item.
- **PC-training** — Yan et al., *Positive-Congruent Training: Towards Regression-Free Model Updates*, CVPR 2021. Shows NFR is only weakly controlled by aggregate accuracy and can be reduced by distillation on the old model's correct predictions.

**Claimed but under-ablated.**
- **Hot-refresh / regression-free compatible training** (Zhang et al., ICLR 2022) targets $M_{\text{mix}}$ during rolling backfill. The rolling-window curves are reported at moderate index scale; behaviour at $10^9$ vectors with a real ANN structure is not ablated.
- **Post-hoc stitching** — learning $\psi:\mathbb{R}^{d_{\text{old}}}\to\mathbb{R}^{d_{\text{new}}}$ (linear or MLP) from paired embeddings. Widely used in industry, mostly reported as single benchmark numbers with no $\tau$ and no mixed-index evaluation.
- **Unpaired latent translation** — *Relative representations* (Moschella et al., ICLR 2023) and *vec2vec* (Jha, Zhang, Shmatikov, 2025) show latent spaces can be aligned without paired data. Retrieval-quality compatibility from these maps is a benchmark number, not an ablated deployment result.

**Adjacent, load-bearing.** *Matryoshka Representation Learning* (Kusupati et al., NeurIPS 2022) makes dimension changes cheap by nesting prefixes — it solves the $d$-mismatch axis, not the semantic-drift axis.

## 4. What Is Known

- BCT's frozen-head constraint is sufficient for the compatibility predicate on face recognition at IJB-C scale (~3.5M images gallery-side, millions of training identities): cross-test $M_{\text{no}}$ exceeds $M_{\text{oo}}$, and this replicates across follow-up papers (LCE, FCT, hot-refresh).
- $\tau>0$ for constraint-based methods in every published comparison; FCT's headline claim is that side-information-based forward compatibility removes it, measured on ImageNet-1k and large-scale retrieval sets (CVPR 2022).
- Real upgrades are large enough that people pay for backfill: OpenAI reported `text-embedding-3-large` at MTEB 64.6 and MIRACL 54.9 versus `text-embedding-ada-002` at 61.0 and 31.4 (January 2024). A 23-point MIRACL gain dwarfs any plausible compatibility tax — evidence that when the gap is big, compatibility is the wrong trade.
- MRL: prefix-truncated embeddings match full-size ImageNet-1k 1-NN accuracy at up to $14\times$ fewer dimensions, so dimension change alone need not force a backfill.
- NFR is not implied by accuracy: models with higher top-1 can flip a measurable fraction of previously-correct queries (CVPR 2021, ImageNet scale).

## 5. What Is Not Known

- **Theoretically open.** No lower bound on $\tau$. Nobody has proved a statement of the form: for a task family with Bayes-optimal margin $\gamma$, any $\phi_{\text{new}}$ satisfying $M_{\text{no}}\ge M_{\text{oo}}+\epsilon$ must lose at least $f(\epsilon,\gamma)$ accuracy. Also open: whether compatibility is transitive in any useful sense across a chain $\phi_1\to\phi_2\to\phi_3$.
- **Empirically open.** Whether compatible training holds at web scale for *text* retrieval — $10^8$–$10^9$ passages, BEIR/MIRACL-style heterogeneous evaluation, with an HNSW or IVF-PQ index rather than exhaustive search. Almost all compatibility evidence is vision (face, re-ID, ImageNet). Runnable today; unrun publicly.
- **Methodologically blocked.** $M_{\text{mix}}(\rho)$ has no standard definition or benchmark. Score comparability across spaces (does a 0.83 in the new space mean what 0.83 meant in the old?) is not part of any published compatibility protocol, yet thresholded systems — dedup, entity matching, safety filters — depend on exactly that.

## 6. Why It Is Hard

**The evaluation does not measure the thing it names.** The compatibility predicate $M_{\text{no}}>M_{\text{oo}}$ is a *ranking-only, single-space-gallery* statement. Deployment failures are calibration and mixing failures: a query scored against a half-backfilled index compares numbers drawn from two similarity distributions with different means and variances, and top-$k$ selection is not invariant to that. Passing the published test is compatible with a system that silently prefers whichever half of the index scores higher.

**Non-identifiability compounds it.** Compatibility constrains $\phi_{\text{new}}$ only up to the old model's decision structure; infinitely many $\phi_{\text{new}}$ satisfy it, and the criterion gives no way to pick the one that also preserves per-query behaviour. Hence $\mathrm{NFR}>0$ alongside aggregate gains.

**Ground truth for the counterfactual is absent.** $\tau$ requires training an unconstrained twin at the same scale and data — a second full pretraining run — which is why $\tau$ is the most-omitted number in the literature.

## 7. Current Research (as of 2026)

- Side-information / forward-compatible design (Apple ML and successors to FCT) — plan the next upgrade before shipping the current one.
- Compatibility for text retrieval and RAG stores, driven by vector-DB vendors; largely engineering blog posts, not ablated papers *(frontier — verify)*.
- Unpaired latent-space translation as a backfill-free path (Cornell/Shmatikov group's vec2vec; Italian groups on relative representations). Security work here doubles as a compatibility result: if embeddings can be translated without pairs, backfill can sometimes be replaced by a learned map *(frontier — verify)*.
- Matryoshka-style nesting is now standard in shipped embedding APIs, decoupling dimension change from model change.
- Regression control for LLM and reranker upgrades, extending PC-training's NFR framing to generative pipelines *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does compatible training survive a *mixed* text index at realistic scale, and at what tax?

**Scale.** MS MARCO passage (8.8M) plus BEIR's 13 public tasks; index built with HNSW ($M=32$, efC=200), not exhaustive search. Old model: a 110M-param bi-encoder, $d=768$. New model: 335M params, $d=1024$, trained on a superset of the data.

**Arms.**
1. **Control A — unconstrained new model** ($M_{\text{nn}}^{\ast}$), full backfill. Upper bound.
2. **Control B — no upgrade** ($M_{\text{oo}}$). Lower bound.
3. **BCT-style constrained** $\phi_{\text{new}}$.
4. **Post-hoc linear $\psi$** fitted on 1M paired embeddings.
5. **FCT-style** side-information transform.

**Protocol.** Sweep backfilled fraction $\rho\in\{0,0.25,0.5,0.75,1\}$ by randomly choosing which passages carry new vectors, serve queries with $\phi_{\text{new}}$, and measure nDCG@10 on the mixed index.

**Deciding number.** $\min_{\rho} \mathrm{nDCG@10}_{\text{mix}}(\rho)$ for each arm, reported against Control A's $\rho=1$ value. The claim "compatible training works for text retrieval" survives only if some arm keeps the worst-$\rho$ point within **2 nDCG@10 points** of Control A while $\tau\le 1$ point. If the worst point sits at $\rho\approx0.5$ and drops below Control B, the field's headline metric $M_{\text{no}}$ is confirmed to be measuring the wrong thing.

## 9. Key References

- **[Foundational]** Yantao Shen, Yuanjun Xiong, Wei Xia, Stefano Soatto. *Towards Backward-Compatible Representation Learning.* CVPR, 2020. — arXiv:2003.11942
- **[SOTA]** Vivek Ramanujan, Pavan Kumar Anasosalu Vasu, Ali Farhadi, Oncel Tuzel, Hadi Pouransari. *Forward Compatible Training for Large-Scale Embedding Retrieval Systems.* CVPR, 2022.
- **[SOTA]** Qiang Meng et al. *Learning Compatible Embeddings.* ICCV, 2021.
- **[SOTA]** Binjie Zhang et al. *Towards Universal Backward-Compatible Representation Learning* / *Hot-Refresh Model Upgrades with Regression-Free Compatible Training in Image Retrieval.* ICLR, 2022.
- **[Foundational]** Sijie Yan et al. *Positive-Congruent Training: Towards Regression-Free Model Updates.* CVPR, 2021.
- **[Adjacent]** Aditya Kusupati et al. *Matryoshka Representation Learning.* NeurIPS, 2022. — arXiv:2205.13147
- **[Adjacent]** Luca Moschella et al. *Relative Representations Enable Zero-Shot Latent Space Communication.* ICLR, 2023.
- **[Frontier]** Rishi Jha, Collin Zhang, Vitaly Shmatikov. *Harnessing the Universal Geometry of Embeddings.* 2025.
- **[Benchmark]** Nandan Thakur et al. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks, 2021.

## 10. Worked Example

A 500M-passage store, $d_{\text{old}}=768$ fp32 = 3.07 KB/vector → **1.54 TB**.

**Backfill is not compute-bound.** A 335M-param encoder at ~1,500 passages/s/GPU (256 tokens) does 500M passages in $3.3\times10^5$ GPU-seconds ≈ **93 GPU-hours** ≈ 12 hours on 8 GPUs. The blocker is not the GPU bill; it is that ~18% of the source passages were deleted under retention policy and cannot be re-encoded at all. So $\rho$ saturates at 0.82 — the index is *permanently* mixed.

**Now the obstruction.** Suppose (illustrative calculation, plausible magnitudes) new/new positive-pair cosine similarities have mean 0.71, sd 0.09, while cross-space new-query/old-doc scores under a BCT-style model have mean 0.63, sd 0.11 — the compatibility predicate still holds, $M_{\text{no}}=0.41$ nDCG@10 versus $M_{\text{oo}}=0.38$. But for a top-10 list drawn from a pool of both, a relevant old-space passage must beat a new-space distractor. With a mean gap of $0.71-0.63=0.08$ and pooled sd $\approx0.14$, an old-space relevant item loses to an equally-relevant new-space item with probability $\Phi(0.08/(0.14\sqrt2))\approx 0.66$. Across 100 candidates, the 18% never-backfilled tail is systematically pushed down the list.

Every published compatibility number in this scenario is green. The served system quietly stops returning the documents it can no longer re-encode — which is precisely the population the upgrade was supposed to protect. That gap between $M_{\text{no}}$ and $M_{\text{mix}}$ is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*