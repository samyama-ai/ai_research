---
id: 06-data-pipeline/cross-lingual-data-allocation
title: "Cross-Lingual Data Allocation Under Transfer"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Lingual Data Allocation Under Transfer

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/cross-lingual-data-allocation` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed training-token budget and a pool of text in $L$ languages of wildly unequal size, decide how many tokens to draw from each language.

- **Input:** per-language pools $\{D_\ell\}_{\ell=1}^L$ with sizes $n_\ell$ (spanning ~6 orders of magnitude in web crawls), a token budget $B$, a model size $N$, and a utility weighting $w_\ell$ over languages.
- **Output:** an allocation $p \in \Delta^{L-1}$ (optionally a schedule $p(t)$), plus a repetition count where $p_\ell B > n_\ell$.
- **Objective:** minimize $\sum_\ell w_\ell \mathcal{L}_\ell(p, B, N)$, where $\mathcal{L}_\ell$ is held-out loss or a downstream metric in language $\ell$.

Three variants, different difficulty:

- **Measurement:** is per-language loss comparable across languages at all, given different tokenizers, corpus quality, and test-set provenance? Currently unresolved.
- **Method:** find a good $p$ cheaply — via proxy models, online reweighting, or a fitted mixing law. Several methods exist; none has been shown to transfer across the number of languages.
- **Theory:** when is the map $p \mapsto (\mathcal{L}_1,\dots,\mathcal{L}_L)$ predictable from few-language measurements, and is the resulting optimization convex? Open.

Solving it means: a procedure that, at cost $\ll B$, outputs $p$ whose realized loss vector is within measurement noise of the best allocation found by brute force at full scale.

## 2. Formal Setting

Let $V$ be the vocabulary, $x^{(\ell)}$ a document in language $\ell$, and $\theta(p, B, N)$ the parameters obtained by training a size-$N$ model on $B$ tokens sampled with mixture $p$.

**Per-language loss, as measured:**
$$\mathcal{L}_\ell(p,B,N) = \frac{1}{|H_\ell|}\sum_{x \in H_\ell} \frac{-\log P_\theta(x)}{\mathrm{bytes}(x)}$$
Bits-per-byte, not per-token: per-token loss is not comparable across languages because tokenizer fertility $f_\ell$ (tokens per byte) varies 1.5–4$\times$ between English and low-resource scripts. $H_\ell$ must be a held-out set deduplicated against $D_\ell$ at the document *and* near-duplicate level.

**Baseline family (temperature sampling):**
$$p_\ell(\alpha) = \frac{n_\ell^{\alpha}}{\sum_{k} n_k^{\alpha}}, \qquad \alpha \in [0,1]$$
$\alpha = 1$ is natural proportion, $\alpha = 0$ uniform. XLM-R uses $\alpha = 0.3$; massively multilingual MT commonly uses $T = 1/\alpha = 5$.

**Repetition.** Effective unique tokens $u_\ell = \min(p_\ell B,\, n_\ell)$ and epochs $e_\ell = p_\ell B / n_\ell$. Allocation is inseparable from repetition: for low-resource $\ell$, any nontrivial $p_\ell$ implies $e_\ell \gg 1$.

**Transfer.** A first-order model writes
$$\mathcal{L}_\ell(p) \approx A_\ell + \Big(\sum_{k} T_{\ell k}\, p_k \Big)^{-\beta_\ell} + \frac{C}{N^{\gamma}}$$
with $T_{\ell k} \ge 0$ the *effective-token exchange rate*: how many tokens of language $k$ count as one token of $\ell$. $T$ is not observed; it is fit.

**Assumptions, and which are violated:**

| Assumption | Status |
|---|---|
| Held-out sets are clean, comparable, in-language | **Violated.** Kreutzer et al. (TACL 2022) audited web corpora and found many with under 50% correct-language sentences. |
| Additivity/separability of transfer ($T$ constant in $p$) | **Violated or unverified.** Interactions are script- and family-dependent and change with $N$. |
| Repeated tokens are worth the same as fresh tokens | **Violated.** Value decays with epoch count. |
| Pool sizes $n_\ell$ are fixed and known | **Approximately violated.** Post-dedup, post-quality-filter sizes for low-resource languages fall by 50–90%. |
| Optimal $p$ is $N$-independent | **Contested** — this is the crux of proxy-model methods. |

## 3. State of the Art

**Established (ablated, reproduced):**

- Temperature sampling with $\alpha \approx 0.3$ beats $\alpha = 1$ and $\alpha = 0$ on multilingual encoders (Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale*, ACL 2020). The **curse of multilinguality** — fixed capacity, more languages, worse per-language performance, relieved by increasing $N$ — is reproduced across encoders and MT.
- The low-resource/high-resource trade-off in massively multilingual MT: raising temperature helps low-resource pairs and costs high-resource pairs (Arivazhagan et al., *Massively Multilingual NMT in the Wild*, 2019, arXiv:1907.05019).

**Claimed but not ablated across the multilingual axis:**

- **DoReMi** (Xie et al., NeurIPS 2023, arXiv:2305.10429) optimizes domain weights with a 280M proxy and transfers them to 8B; reported 2.6$\times$ training speedup on The Pile. Demonstrated over ~22 *domains*, not over 100+ languages with power-law-skewed pools.
- **Data mixing laws** (Ye et al., 2024, arXiv:2403.16952) and **RegMix** (Liu et al., 2024, arXiv:2407.01492) fit loss as a function of mixture from small runs. Validated on English domain mixtures; extrapolation to unseen language sets is untested.
- **Multilingual scaling laws** (Fernandes et al., *Scaling Laws for Multilingual Neural Machine Translation*, ICML 2023) fit per-language-pair loss as a function of mixture weight and model size, and report that weights and size act largely separably in their range. Measured on MT with a handful of language pairs.

**Benchmark-number-only:** NLLB-200 (2022) reports large chrF++ gains at 200 languages using a mined-and-curated mixture, but the allocation itself is not ablated against alternatives at fixed compute — the number does not isolate the allocation.

## 4. What Is Known

- $\alpha = 0.3$, 100 languages, 2.5 TB CommonCrawl: XLM-R Large (550M) reaches 80.9 average XNLI accuracy. At fixed capacity, expanding the language set degrades per-language quality; expanding $N$ from Base (270M) to Large restores it. (ACL 2020, scale: 270M–550M params.)
- MT at 103 languages / 25B sentence pairs: temperature raising gives multi-BLEU gains on low-resource pairs and measurable loss on high-resource pairs. (Scale: ~400M–6B param Transformers.)
- **Repetition value decay:** up to ~4 epochs, repeated tokens are worth close to fresh tokens; by ~16 epochs marginal value approaches zero (Muennighoff et al., *Scaling Data-Constrained Language Models*, NeurIPS 2023, arXiv:2305.16264; scale: up to 9B params, 900B tokens). This is the only well-measured input to the low-resource side of the allocation problem — and it was measured on English.
- Vocabulary allocation interacts with data allocation: language-clustered vocabularies improve multilingual models at fixed size (Chung et al., EMNLP 2020).
- Performance disparities across languages track data volume and are systematic, not incidental (Blasi et al., ACL 2022).

## 5. What Is Not Known

- **Empirically open (the main gap).** Whether mixture weights optimized at proxy scale ($\le$1B params) remain optimal at 10B–100B with $L > 50$ languages. The experiment is runnable; nobody has published a controlled multi-arm run at that $L$ and that scale.
- **Empirically open.** The exchange rate $T_{\ell k}$ between a *repeated* low-resource token and a *fresh* related-language token. No published measurement.
- **Methodologically blocked.** Whether $\sum_\ell w_\ell \mathcal{L}_\ell$ is the right objective at all: $w_\ell$ encodes a policy choice (speaker count? equal weight? downstream demand?) that no paper states as a decision variable.
- **Methodologically blocked.** Cross-language loss comparability. Bits-per-byte removes tokenizer effects but not corpus-difficulty effects; a lower bpb in Swahili than in English may reflect a more repetitive test set.
- **Theoretically open.** Conditions under which $p \mapsto \mathcal{L}(p)$ is convex, or under which few-language fits identify $T$. With $L$ languages, $T$ has $L^2$ entries and each full-scale evaluation gives $L$ numbers — identification requires $\ge L$ runs unless structure is assumed.

## 6. Why It Is Hard

The obstruction is **non-identifiability under a compute-constrained design**, compounded by confounded measurement.

- $L^2$ transfer parameters, $L$ observations per run. Assuming low-rank $T$ (by family or script) is what makes fitting tractable — and that assumption is exactly what is unverified.
- The decision has to be made once. A 100B-param pretraining run is not repeated for 8 mixture arms, so allocation is chosen from proxy runs whose validity is the open question. This is circular, not merely expensive.
- Allocation is confounded with quality filtering, deduplication, tokenizer fertility, and vocabulary allocation. Changing $p_\ell$ changes all four. No published run varies $p$ with the other three held fixed at $L>50$.
- The evaluation does not measure the thing it names: XNLI and FLORES are largely translated from English, so they reward English-anchored representations and understate the cost of starving a language of native text.

## 7. Current Research (as of 2026)

- **Learned/online mixtures.** DoReMi-style group-DRO reweighting and its multilingual descendants; online reweighting during pretraining. Google DeepMind, Stanford (Xie, Liang), Sea AI Lab (RegMix).
- **Predictive mixing laws** extended to languages rather than domains — fit on $\le$1B, extrapolate to $\ge$8B. *(frontier — verify)* Several 2025 papers claim per-language extrapolation error under 0.01 nats; independent replication is not yet visible.
- **Massively multilingual open corpora** — MADLAD-400 (Kudugunta et al., NeurIPS 2023 Datasets, arXiv:2309.04662), HPLT, FineWeb2 — which make the pool sizes $n_\ell$ auditable for the first time.
- **Curriculum over allocation:** anneal from uniform to proportional, or reserve low-resource data for a late high-quality phase. Widely used in industrial runs, rarely ablated in public.

## 8. Concrete Next Experiment

**Question:** do mixture weights fit at 300M transfer to 3B when $L = 24$?

- **Scale:** $L = 24$ languages from MADLAD-400 spanning $n_\ell$ from 500B down to 200M tokens, 6 families, 5 scripts. One shared 128k tokenizer, fixed for all arms. Train 300M-param models on 6B tokens (Chinchilla-ratio), 12 mixture arms: 4 temperature arms ($\alpha \in \{0,0.2,0.3,0.5\}$), 4 random Dirichlet arms, 4 arms proposed by a fitted mixing law. Then train **3 arms only** at 3B params / 60B tokens: the proxy-optimal, the $\alpha{=}0.3$ control, and the proxy-*worst* non-degenerate arm.
- **Control arm:** $\alpha = 0.3$ temperature sampling at both scales — the current default, so the comparison is against practice, not against a strawman.
- **Deciding number:** the **rank correlation between proxy-scale and target-scale per-language bits-per-byte across the 3 arms**, and specifically $\Delta = \mathcal{L}^{3B}_{\text{ctrl}} - \mathcal{L}^{3B}_{\text{proxy-opt}}$ averaged over the 12 lowest-resource languages, in bits-per-byte. Seed noise at 300M is ~0.005 bpb; require $\Delta > 0.02$ bpb with the ordering preserved for a positive result. If $\Delta \le 0$, proxy transfer fails and the field's default method is unsupported at $L=24$.
- **Cost:** ~12 $\times$ 6B + 3 $\times$ 60B token-runs $\approx$ 250B tokens of training, well under a single frontier run.

## 9. Key References

- **[Foundational]** Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzmán, Grave, Ott, Zettlemoyer, Stoyanov. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. — arXiv:1911.02116
- **[Foundational]** Arivazhagan, Bapna, Firat, Lepikhin, Johnson, Krikun, Chen, Cao, Foster, Cherry, Macherey, Chen, Wu. *Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges.* 2019. — arXiv:1907.05019
- **[SOTA]** Xie, Pham, Dong, Du, Liu, Lu, Liang, Le, Ma, Yu. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS 2023. — arXiv:2305.10429
- **[SOTA]** Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[SOTA]** Fernandes, Ghorbani, Garcia, Freitag, Firat. *Scaling Laws for Multilingual Neural Machine Translation.* ICML 2023.
- **[SOTA]** Liu, Zeng, He, Pang, Lin. *RegMix: Data Mixture as Regression for Language Model Pre-training.* 2024. — arXiv:2407.01492
- **[Data]** Kudugunta, Caswell, Zhang, Garcia, Xin, Kusupati, Stella, Bapna, Firat. *MADLAD-400: A Multilingual And Document-Level Large Audited Dataset.* NeurIPS 2023 Datasets & Benchmarks. — arXiv:2309.04662
- **[Survey/Audit]** Kreutzer, Caswell, Wang, Wahab, van Esch, et al. *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.* TACL 2022.
- **[Context]** Blasi, Anastasopoulos, Neubig. *Systematic Inequalities in Language Technology Performance across the World's Languages.* ACL 2022.

## 10. Worked Example

Two languages. English pool $n_{\text{en}} = 1{,}000$B tokens, Swahili pool $n_{\text{sw}} = 2$B. Budget $B = 100$B.

Temperature sampling at $\alpha = 0.3$:
$$n_{\text{en}}^{0.3} = e^{0.3 \ln(10^{12})} \approx 3.98\times10^{3}, \quad n_{\text{sw}}^{0.3} = e^{0.3\ln(2\times10^{9})} \approx 6.15\times10^{2}$$
$$p_{\text{sw}} = \frac{615}{3980+615} = 0.134$$

So the default heuristic allocates **13.4B Swahili tokens from a 2B pool — 6.7 epochs**, and 86.6B English tokens at 0.087 epochs.

Now price it. From the repetition results, the 5th–7th epoch of a token is worth materially less than the first, but the published decay curve was measured on English at 9B params with no other language present. Two plausible readings of the same evidence:

- **Reading A** (repetition decay dominates): the marginal 6.7th-epoch Swahili token is worth ~0.2 fresh tokens. Then $\alpha=0.3$ wastes ~8B token-slots; cut to $e_{\text{sw}} = 4$ ($p_{\text{sw}} = 0.08$) and spend the rest on English, which transfers to Swahili through shared structure.
- **Reading B** (transfer is weak across scripts/families and Swahili needs in-language signal): $T_{\text{sw},\text{en}} \approx 0.02$, so 8B extra English tokens buy the equivalent of 160M Swahili tokens — far less than 8B repeated Swahili tokens even at 0.2 efficiency. Keep $\alpha=0.3$ or raise it.

The two readings differ by the single unmeasured scalar $T_{\text{sw},\text{en}}$ against the epoch-decay factor. Both are consistent with every published number cited above. That is the obstruction: the default allocation heuristic sits precisely on the boundary between two opposite recommendations, and no experiment in the literature separates them.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*