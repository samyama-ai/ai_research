---
id: 06-data-pipeline/multimodal-caption-noise-filtering
title: "Verifiable Filtering of Multimodal Caption Noise"
topic: 06-data-pipeline
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Verifiable Filtering of Multimodal Caption Noise

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/multimodal-caption-noise-filtering` · **Status:** open

## 1. Problem Statement

Web image–text pools are mostly misaligned: alt-text that names the seller, repeats the filename, or transcribes text rendered inside the image. Every large vision–language model trains on a filtered subset of such a pool. The problem is not "filter better" — it is **filter with a checkable guarantee**.

- **Input:** a candidate pool $\mathcal{P}$ of $N$ image–caption pairs, a training budget (samples seen, not pool size), an architecture, and a downstream evaluation suite.
- **Output:** a subset $S \subseteq \mathcal{P}$, plus a *certificate* — evidence that $S$ attains downstream utility within $\epsilon$ of the best subset the filter family can produce.
- **Decision predicate:** given two filters $\phi_1, \phi_2$, decide which yields higher downstream accuracy **without retraining on both**, with a stated error bar.

Three variants, routinely conflated:

- **Measurement:** define "caption noise" as a quantity that predicts downstream loss. Currently not well posed — see §6.
- **Method:** produce a filter that beats CLIP-score thresholding. Solved repeatedly (§3), each time by a heuristic validated only by full retraining.
- **Theory:** prove a bound relating a per-sample noise statistic to the excess risk of the model trained on the surviving subset. Open.

Solving it means: a filter whose accept/reject decisions come with an auditable per-sample justification, and a certificate that predicts the retrained model's evaluation score to within the run-to-run noise floor.

## 2. Formal Setting

Pool $\mathcal{P} = \{(x_i, c_i)\}_{i=1}^N$, $x_i$ an image, $c_i$ a caption. A latent alignment variable $a_i \in \{0,1\}$ marks whether $c_i$ describes $x_i$. A filter is $\phi: \mathcal{X}\times\mathcal{C} \to \{0,1\}$, giving $S_\phi = \{i : \phi(x_i,c_i)=1\}$, $n_\phi = |S_\phi|$.

**Training budget.** Fix total samples seen $T$ (e.g. $T=1.28\times10^8$ at DataComp *medium*). The model sees $T/n_\phi$ epochs of $S_\phi$. This is the operative constraint: filtering trades diversity for density at *fixed* compute, so $\phi$ is not evaluated on $S_\phi$'s purity but on the pair $(n_\phi, \text{purity})$.

**Utility, as measured.** With $\theta(S,\xi)$ the parameters from a contrastive run on $S$ under seed $\xi$,
$$U(\phi) = \mathbb{E}_\xi\big[\tfrac{1}{|\mathcal{E}|}\textstyle\sum_{e\in\mathcal{E}} \mathrm{acc}_e(\theta(S_\phi,\xi))\big],$$
$\mathcal{E}$ the 38 zero-shot/retrieval tasks of DataComp. Measured by one training run per filter; $\mathbb{E}_\xi$ is almost never estimated — single-seed numbers are reported as if noiseless.

**Noise score.** A filter is usually a threshold on a scalar $s(x,c)$: CLIP similarity $s_{\mathrm{clip}} = \langle f(x), g(c)\rangle / (\|f(x)\|\|g(c)\|)$ from a *reference* model, or a caption-model likelihood $s_{\mathrm{gen}} = \frac{1}{|c|}\log p_\psi(c \mid x)$.

**Certificate.** A procedure $\pi$ outputting $\hat U$ with
$$\Pr\big[|\hat U(\phi) - U(\phi)| \le \epsilon\big] \ge 1-\delta$$
at cost $\ll$ one training run. No published method achieves this for $\epsilon$ below the ~0.5 pp differences that decide filter rankings.

**Assumptions, and which fail.**

1. *$a_i$ is well defined and human-labelable.* Fails: OCR pairs (caption = text in image) are labelled aligned by humans yet are net-harmful to train on (T-MARS).
2. *Utility is monotone in purity at fixed $n$.* Fails: hard negatives and near-duplicates change sign depending on $T/n_\phi$.
3. *The reference scorer is independent of the evaluation.* Fails: OpenAI CLIP's scores were shaped by data overlapping DataComp's eval tasks — filtering is a covert distillation channel (Fang et al., 2024).
4. *Samples are exchangeable.* Fails: pools are dominated by domain clusters; removing one cluster shifts the whole subset's marginal.

## 3. State of the Art

**Established (ablated, reproduced by independent groups).**

- **CLIP-score thresholding** (Schuhmann et al., LAION-400M, 2021). Keep pairs above a similarity cut. At DataComp *medium* (128M pool, $T=128$M), the top-30% CLIP-L/14 filter gives **27.3%** ImageNet zero-shot vs **17.6%** for the unfiltered pool (Gadre et al., NeurIPS 2023). Reproduced widely.
- **Intersection filtering** — CLIP score $\cap$ image-embedding proximity to ImageNet-1k centroids — is the strongest hand-built DataComp baseline, **~32.8%** at medium; scaled to *xlarge* it yields DataComp-1B and **79.2%** ImageNet with ViT-L/14.
- **Data Filtering Networks** (Fang et al., ICLR 2024). Train a small model *specifically to filter*, on high-quality curated data. Establishes that filter quality is not predicted by the filter model's own zero-shot accuracy — an ablation others confirmed.
- **T-MARS** (Maini et al., NeurIPS 2023). Detect and mask text regions; re-score. Recovers pairs whose CLIP score came only from OCR matching, reported ~33% at medium — direct evidence that CLIP score measures the wrong thing.
- **MetaCLIP** (Xu et al., ICLR 2024). Balanced substring matching against a metadata query list, no learned scorer: 400M pairs → **70.8%** ViT-B/32 ImageNet vs OpenAI CLIP's 68.3%.

**Claimed but under-ablated.**

- **Recaptioning.** BLIP-2/LLaVA-generated captions (Nguyen et al., NeurIPS D&B 2023; CapsFusion, CVPR 2024; VeCLIP, ECCV 2024; Recap-DataComp-1B, 2024). Gains are real at small/medium; the *mixing ratio* of synthetic to raw captions is tuned per-scale and rarely ablated at more than two scales.
- **Sieve** (Mahmoud et al., CVPR 2024): caption-model likelihood as the filter signal. Benchmark numbers only; no analysis of which pairs it and CLIP score disagree on.
- **Datamodel-style selection** (DsDm, Engstrom et al., 2024) targets downstream loss directly. Demonstrated for LM pretraining; the multimodal instance is a benchmark number at best.

**No method in any of these families emits a certificate.** Every ranking is established by retraining.

## 4. What Is Known

- Filtering dominates architecture at fixed compute. DataComp *medium*: 17.6% → ~32.8% ImageNet from data selection alone, model and budget held fixed (128M samples seen, ViT-B/32).
- Optimal keep-rate is scale-dependent. ~30% is best at medium; aggressive filtering underperforms at *xlarge* where pool exhaustion binds. Measured across DataComp's 12.8M/128M/1.28B/12.8B ladder.
- Synthetic captions help less as scale grows. Nguyen et al. (2023) report clear gains at 12.8M/128M that compress at 1.28B — the raw-caption pool's diversity is worth more once there is enough of it.
- CLIP score systematically prefers OCR pairs. T-MARS: a large fraction of top-scoring DataComp images contain the caption as rendered text; masking them changes the ranking of filters.
- Filter strength ≠ filter-model strength. DFN: a ViT-B/32 filter trained on high-quality data outperforms filters from far stronger models.
- Deduplication is worth single-digit points and is orthogonal to alignment filtering (SemDeDup, Abbas et al., 2023).

## 5. What Is Not Known

- **Methodologically blocked.** "Caption noise" has no measurement that predicts utility. Human alignment labels and downstream contribution disagree in a known, non-random way (OCR pairs), so precision/recall against human labels is not a valid proxy. Until the target quantity is defined, filters cannot be compared except by retraining.
- **Theoretically open.** No bound of the form $U(\phi) \ge U^\star - g(\text{noise rate}, n_\phi, T)$ for contrastive objectives. Label-noise theory (Natarajan et al., NeurIPS 2013) covers classification with class-conditional flips; contrastive learning with a *shared* corrupted caption distribution and in-batch negatives is not covered — a corrupted caption is simultaneously a bad positive and a valid negative for other images.
- **Empirically open.** Whether a proxy-scale certificate (train at 12.8M, predict at 1.28B) can rank filters within 0.5 pp. Runnable today; nobody has published the full cross-scale rank-correlation matrix.
- **Empirically open.** Whether recaptioning replaces filtering or is complementary at $\ge 1$B scale, under matched samples-seen.
- **Unmeasured.** Seed variance of DataComp medium ImageNet accuracy. Reported filter gaps are often ~1 pp; the noise floor is not published.

## 6. Why It Is Hard

**The ground truth is absent and the available proxy is anti-correlated on an identifiable subpopulation.** The quantity we want is a sample's marginal contribution to downstream accuracy. The quantity we can label is human-judged alignment. These agree on random pairs and disagree on OCR pairs, product listings, and stock-photo boilerplate — exactly the high-mass regions of the pool. So filter evaluation is *confounded measurement*, not merely noisy.

Second obstruction: **non-identifiability at fixed compute.** $U$ is a function of the whole subset, not a sum over samples — in-batch negatives make every sample's value depend on which others survived. A per-sample score cannot be identified from subset-level outcomes without $O(N)$ counterfactual runs.

Third: **cost.** One DataComp *large* run is thousands of GPU-hours. Ranking ten filters with three seeds is a research group's annual budget, so almost every published comparison is single-seed at one scale.

## 7. Current Research (as of 2026)

- **Filter-model training as the primary lever** — DFN line at Apple; MetaCLIP curation line at Meta. Consensus is shifting from hand-built scores to trained selectors.
- **Recaption-then-filter pipelines** at scale (Recap-DataComp-1B; DataComp-style follow-ons). Open question is mixing ratio scaling law. *(frontier — verify)*
- **Datamodel / influence-based selection ported from LM to multimodal** (MIT Madry lab lineage). *(frontier — verify)*
- **Proxy-scale predictors of curation quality** — small-model rank transfer as an explicit object of study rather than an assumption. *(frontier — verify)*
- **Contamination auditing of filter models**, following the DFN observation that the reference scorer leaks eval information. Largely unaddressed in public work.

## 8. Concrete Next Experiment

**Question:** can filter rankings be certified at proxy scale?

- **Scale.** DataComp *medium* (128M pool, $T=128$M seen, ViT-B/32) as the reference, DataComp *small* (12.8M, $T=12.8$M) as the proxy. Eight filters: no-filter, CLIP-score at 15/30/45%, image-based $\cap$ CLIP-score, T-MARS, Sieve, DFN-public. **Three seeds each, both scales** — 48 runs, ~3–5k A100-hours total.
- **Control arm.** The same 8 filters ranked by their agreement with 5,000 human alignment labels drawn from the medium pool (precision at the filter's own keep-rate). This is the ranking a purity-based certificate would produce.
- **Deciding number.** Kendall's $\tau$ between the small-scale ranking and the medium-scale ranking on average accuracy over the 38 tasks, reported alongside the three-seed standard deviation at medium. **If $\tau \ge 0.85$ and the seed SD is $\le 0.4$ pp, proxy-scale certification is viable** and the field can stop retraining at target scale to compare filters. If $\tau < 0.6$, or if the human-label control ranks as well as the proxy runs, the current filtering literature's single-scale comparisons carry no transferable signal.

Nothing here needs a new method. It needs the seed replicates nobody pays for.

## 9. Key References

- **[Foundational]** Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. — arXiv:2103.00020
- **[Foundational]** Schuhmann et al. *LAION-5B: An open large-scale dataset for training next generation image-text models.* NeurIPS Datasets & Benchmarks 2022. — arXiv:2210.08402
- **[SOTA]** Gadre et al. *DataComp: In search of the next generation of multimodal datasets.* NeurIPS Datasets & Benchmarks 2023. — arXiv:2304.14108
- **[SOTA]** Fang et al. *Data Filtering Networks.* ICLR 2024. — arXiv:2309.17425
- **[SOTA]** Xu et al. *Demystifying CLIP Data.* ICLR 2024. — arXiv:2309.16671
- **[SOTA]** Maini et al. *T-MARS: Improving Visual Representations by Circumventing Text Feature Learning.* NeurIPS 2023. — arXiv:2307.03132
- Nguyen et al. *Improving Multimodal Datasets with Image Captioning.* NeurIPS Datasets & Benchmarks 2023. — arXiv:2307.10350
- Mahmoud et al. *Sieve: Multimodal Dataset Pruning Using Image Captioning Models.* CVPR 2024.
- Abbas et al. *SemDeDup: Data-efficient learning at web-scale through semantic deduplication.* 2023. — arXiv:2303.09540
- Engstrom, Feldmann, Madry. *DsDm: Model-Aware Dataset Selection with Datamodels.* ICML 2024. — arXiv:2401.12926
- **[Theory]** Natarajan, Dhillon, Ravikumar, Tewari. *Learning with Noisy Labels.* NeurIPS 2013.
- **[Survey]** Frénay, Verleysen. *Classification in the Presence of Label Noise: A Survey.* IEEE TNNLS, 2014.

## 10. Worked Example

Take the DataComp *medium* pool, 128M pairs, and the CLIP-L/14 top-30% filter: $n_\phi \approx 38$M, $T = 128$M, so ~3.4 epochs. Result: 27.3% ImageNet zero-shot vs 17.6% unfiltered.

Now audit the accepted set. T-MARS finds that a substantial share of the highest-scoring survivors are images whose caption is the text rendered in the image — a book cover captioned with its title, a meme captioned with its overlay. Score them three ways:

| Signal | Verdict on an OCR pair |
|---|---|
| CLIP score | very high (near-duplicate string match) — **accept** |
| Human alignment label $a_i$ | "the caption describes the image" — **accept** |
| Marginal downstream contribution | pushes the model toward reading text, not objects — **reject** |

Masking the text regions and re-scoring flips these pairs out and lifts medium-scale ImageNet to roughly 33% — about **+5.7 pp** from re-deciding a subpopulation that both the automatic score and the human label had accepted.

The obstruction is now visible as a number, not an argument. A certificate built on human alignment labels would have scored the CLIP-score filter's precision near its ceiling on exactly these pairs and predicted no headroom. The +5.7 pp it missed is roughly **six times** the ~1 pp differences that separate published filters. So the only currently trustworthy way to know that T-MARS beats CLIP-score filtering is to train both models — which is precisely what a verifiable filter is supposed to avoid.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*