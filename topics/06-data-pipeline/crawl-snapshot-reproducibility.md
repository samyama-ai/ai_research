---
id: 06-data-pipeline/crawl-snapshot-reproducibility
title: "Reproducibility of Web Crawl Snapshots Over Time"
topic: 06-data-pipeline
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Reproducibility of Web Crawl Snapshots Over Time

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/crawl-snapshot-reproducibility` · **Status:** methodologically-blocked

## 1. Problem Statement

A pretraining corpus is usually specified by a recipe: a crawl identifier (e.g. `CC-MAIN-2024-10`), a filter chain, a deduplication policy. The recipe is treated as if it named a fixed object. It does not. Re-running the same crawl specification at a later date, or from a different machine, returns different bytes: pages die, pages change, servers serve different content to different clients, and `robots.txt` policies tighten retroactively over the archive.

Three variants, of very different difficulty:

- **Measurement.** Given two crawls $\mathcal{S}_1, \mathcal{S}_2$ produced by the same nominal spec at times $t_1 < t_2$, define and compute a reproducibility score that separates (a) genuine change in the web from (b) change in the crawler's view of an unchanged web. *This is the blocked variant.*
- **Method.** Build a crawl/archive protocol whose output is bit-reproducible, or reproducible up to a stated tolerance, from a portable manifest — content hashes, fetch conditions, a fallback archive.
- **Theory.** Bound the downstream deviation: given a content-drift rate, how large can $|L(\theta_1) - L(\theta_2)|$ get for models trained on the two snapshots, relative to seed noise?

Solving it means: two labs, given the same manifest, obtain corpora whose difference is below a stated tolerance, and a model trained on either lands within seed variance on a fixed eval suite.

## 2. Formal Setting

Let $U$ be the URL space and $t$ wall-clock time. A fetch is not a function of $(u,t)$ alone but of the **crawler context** $c = (\text{user-agent}, \text{source ASN/geo}, \text{cookie/session state}, \text{JS rendering on/off}, \text{TLS/HTTP stack}, \text{rate})$:

$$f(u, t, c) \in \Sigma^* \cup \{\bot\},$$

with $\bot$ for non-retrieval (DNS failure, 4xx/5xx, robots exclusion). A **snapshot** is $\mathcal{S} = \{(u, f(u,t_u,c_u))\}_{u \in \hat U}$ where $\hat U \subset U$ is drawn by a frontier/scheduling policy $\pi$ — itself a function of prior crawls, so $\pi$ is path-dependent.

Measured quantities, on a fixed re-fetch list $\hat U$ of size $n$:

- **Availability decay** $A(\Delta) = \frac{1}{n}\sum_u \mathbf{1}[f(u,t+\Delta,c)\neq\bot]$, measured by HTTP status after redirect resolution.
- **Byte identity** $I = \frac{1}{n}\sum_u \mathbf{1}[\mathrm{SHA256}(f_1(u)) = \mathrm{SHA256}(f_2(u))]$ over pairs both $\neq\bot$.
- **Near-duplicate agreement** $J = \frac{1}{n}\sum_u \hat{\mathrm{Jac}}(\mathrm{MinHash}(x_1^u), \mathrm{MinHash}(x_2^u))$ on extracted text, 5-gram shingles, 128 permutations — the same estimator used in RefinedWeb/FineWeb dedup.
- **Corpus-level drift** $D = W_1$ or TV distance between token-level or domain-level distributions of the two post-filter corpora.
- **Downstream deviation** $\delta = |L_{\mathrm{eval}}(\theta_1) - L_{\mathrm{eval}}(\theta_2)|$, to be compared against $\sigma_{\text{seed}}$, the SD across seeds at fixed data.

Assumptions the standard framing rests on, and their status:

1. *A page has a content state at time $t$ independent of who asks.* **Violated** — geo/UA/JS/paywall/bot-detection variation is routine, and Carlini et al. (2024) turn it into an attack (split-view poisoning).
2. *Non-retrieval is missing-at-random.* **Violated** — decay correlates with domain age, language, and site size.
3. *`robots.txt` at crawl time governs the record permanently.* **Violated** — many pipelines re-apply current `robots.txt` retroactively, so an archived snapshot shrinks over time (Longpre et al., 2024).
4. *Extraction is deterministic given bytes.* **Approximately true but version-dependent** — trafilatura/resiliparse version bumps change text for the same WARC.

The obstruction is that $A$, $I$, $J$ jointly identify only the composite change; the factorization into web-change and crawler-context-change is not identified from a single re-crawl.

## 3. State of the Art

**Systems/empirical SOTA (established).**
- Common Crawl publishes monthly WARC dumps (~2–3B pages each) with immutable identifiers; the WARC file is reproducible *as an artifact*, and this is the only genuinely solved layer.
- Memento (Van de Sompel, Nelson, Sanderson; RFC 7089, 2013) gives datetime-negotiated access to archived representations — the closest thing to a temporal addressing standard.
- FineWeb (Penedo et al., NeurIPS 2024 D&B) is the strongest public demonstration that snapshots are *not* interchangeable: ablation models trained per-dump on 96 CommonCrawl dumps show non-monotone eval curves, and the anomalously strong 2021-49→2022-05 dumps were traced to benchmark-like content in the crawl, not to better data.
- Dolma (Soldaini et al., ACL 2024) and RefinedWeb (Penedo et al., NeurIPS 2023 D&B) publish full toolchains, which makes *pipeline* reruns reproducible given fixed WARCs — but not *crawl* reruns.

**Claimed but unablated.** That "recent dumps are better/worse" is widely asserted; the confound between dump recency, extraction-library version, and benchmark leakage has not been separated in any published factorial design. Dedup-threshold and extractor-version sensitivity are reported as single benchmark numbers in dataset cards, not as ablations with error bars.

**Theory SOTA.** Essentially none specific to crawls. The nearest formal results are distribution-shift generalization bounds (e.g. Ben-David et al., 2010, $\mathcal{H}\Delta\mathcal{H}$-divergence), which bound transfer under a divergence nobody estimates for two crawls.

## 4. What Is Known

- **Decay is large and roughly exponential.** Pew Research Center (2024) found 38% of webpages sampled that existed in 2013 were inaccessible in 2023; 8% of pages that existed in 2023 were gone within a year. Klein et al. (PLOS ONE, 2014) found roughly one in five scholarly articles suffers reference rot, over ~3.5M references.
- **Expired domains are purchasable and cheap.** Carlini et al. (IEEE S&P 2024) showed a fraction of URLs in LAION-400M, Conceptual Captions and similar sets point at domains available for registration; controlling ~0.01% of a dataset cost on the order of $60. This is direct evidence that *the same URL list yields adversary-chosen content later*.
- **Permission surfaces move.** Longpre et al. (2024) report a rapid tightening of `robots.txt` across the top C4/RefinedWeb/Dolma domains within a single year, with a large share of tokens from the most-used domains newly restricted — so retroactive-robots pipelines lose data with no code change.
- **Snapshot choice moves eval numbers.** FineWeb per-dump ablations (1.8B params, ~28B–350B tokens per arm) show dump-to-dump swings in aggregate benchmark score that exceed their seed noise band.
- **Machine-generated content is now a large share.** Thompson et al. (Findings of ACL 2024) find heavily multi-way-parallel, machine-translated content dominates web text in many lower-resource languages — a composition that changes fast between snapshots.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted estimator that decomposes observed snapshot difference into web-change, crawler-context-change, and pipeline-version-change. Every published "drift" number is the composite. Without the decomposition, "reproducible crawl" has no operational definition, and no tolerance can be stated.
- **Empirically open.** The factorial re-crawl — same URL list, crossed over user-agent × ASN/geo × rendering × elapsed time — has never been run at a scale large enough to train models on each cell. Cost is modest (tens of thousands of dollars); nobody has published it.
- **Empirically open.** Whether $\delta$ between two snapshots of the same nominal spec exceeds $2\sigma_{\text{seed}}$ at ≥1B params, once benchmark leakage is controlled.
- **Theoretically open.** No bound relating a measured drift rate $J$ or $D$ to worst-case $\delta$; and no identifiability result stating what set of re-crawls suffices to identify the context effect.

## 6. Why It Is Hard

**Non-identifiability plus absent ground truth.** There is no ground-truth "content of page $u$ at time $t$" against which a crawl can be scored: the server's response is a function of the requester. One re-crawl gives one composite difference and two unknowns (web change, context change), so the model is underdetermined by construction. Archives do not fix it — the Internet Archive's copy is itself one context-conditioned observation, and coverage of any given crawl's URL set is partial and non-random, so using it as reference imports its own selection bias.

Second obstruction: **evaluation that does not measure what it names.** The natural end-to-end test — train on two snapshots, compare benchmark scores — is contaminated, because the thing that changes between snapshots includes benchmark leakage (FineWeb's 2021-49 anomaly). A score difference is therefore not evidence about data quality drift.

Compute is *not* the binding constraint here; measurement definition is.

## 7. Current Research (as of 2026)

- **Dataset-card provenance.** HuggingFace/FineWeb, AI2/Dolma and the Data Provenance Initiative (Longpre et al.) continue to publish per-dump manifests and license/robots audits. Direction: manifests carrying per-record content hashes so a rerun is checkable *(frontier — verify)*.
- **Web-archive science.** ODU/LANL Memento lineage (Nelson, Van de Sompel) on temporal coherence of composite mementos — the closest formal work on "what time is this page".
- **Poisoning defenses.** Post-Carlini interest in integrity hashes distributed with URL lists; adopted patchily in image-text sets, rarely in text crawls.
- **Synthetic-contamination tracking.** Ongoing measurement of LLM-generated text share per dump *(frontier — verify)*; if the share grows monotonically, cross-snapshot comparison stops being a repeated measurement at all.

## 8. Concrete Next Experiment

**Design.** Take $n = 10^7$ URLs sampled stratified-by-domain-rank from a fixed dump (`CC-MAIN-2024-10`). Re-fetch them under a $2\times3\times2$ factorial: user-agent ∈ {CCBot, generic browser UA}; source ∈ {US cloud ASN, EU cloud ASN, residential-range proxy}; rendering ∈ {raw HTTP, headless-Chrome}. Repeat the full factorial at $\Delta = $ 0, 30, 180 days. Fix extraction to one pinned trafilatura version.

**Control arm.** The *same-context re-fetch at $\Delta \approx 0$* (hours apart, identical context). This measures the irreducible noise floor $J_0$ — CDNs, A/B tests, timestamps in markup — against which every other cell is scored.

**Second arm.** Train 1.4B-parameter models, ~30B tokens, matched token budget, on the $\Delta{=}0$ and $\Delta{=}180$ US/CCBot/raw cells, 3 seeds each, with a decontamination pass applied identically to both.

**Deciding number.** The variance decomposition of $1-J$: the fraction of text-level disagreement attributable to crawler context, $\rho = \sigma^2_{\text{context}} / (\sigma^2_{\text{context}} + \sigma^2_{\text{time}})$, at $\Delta = 180$ days. If $\rho < 0.1$, snapshot difference is genuinely temporal and a manifest-plus-archive protocol suffices — the problem becomes engineering. If $\rho > 0.3$, "the crawl at time $t$" is not a well-defined object and every published drift number is confounded. Report alongside $\delta$ versus $2\sigma_{\text{seed}}$.

## 9. Key References

- **[Foundational]** Van de Sompel, H., Nelson, M., Sanderson, R. *HTTP Framework for Time-Based Access to Resource States — Memento.* IETF RFC 7089, 2013.
- **[Foundational]** Raffel, C. et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (C4).* JMLR, 2020. — arXiv:1910.10683
- **[SOTA]** Penedo, G. et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.17557
- **[SOTA]** Penedo, G. et al. *The RefinedWeb Dataset for Falcon LLM.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.01116
- **[SOTA]** Soldaini, L. et al. *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research.* ACL, 2024. — arXiv:2402.00159
- **[Key result]** Carlini, N. et al. *Poisoning Web-Scale Training Datasets is Practical.* IEEE Symposium on Security and Privacy, 2024. — arXiv:2302.10149
- **[Key result]** Longpre, S. et al. *Consent in Crisis: The Rapid Decline of the AI Data Commons.* NeurIPS Datasets & Benchmarks, 2024.
- **[Key result]** Dodge, J. et al. *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus.* EMNLP, 2021. — arXiv:2104.08758
- **[Measurement]** Klein, M. et al. *Scholarly Context Not Found: One in Five Articles Suffers from Reference Rot.* PLOS ONE, 2014.
- **[Measurement]** Pew Research Center. *When Online Content Disappears.* 2024.
- **[Measurement]** Thompson, B., Dhaliwal, M., Frisch, P., Domhan, T., Federico, M. *A Shocking Amount of the Web is Machine Translated.* Findings of ACL, 2024.
- **[Theory]** Ben-David, S. et al. *A Theory of Learning from Different Domains.* Machine Learning, 2010.

## 10. Worked Example

Take 1,000 URLs from the `patents.google.com` and long-tail-blog strata of a 2024 dump and re-fetch once at $\Delta = 0$ (same context, 6 hours later) and once at $\Delta = 180$ days from a different ASN with a browser UA.

Illustrative arithmetic with the published decay rates: at $\Delta = 180$ days, expect $A \approx 0.96$ (Pew's ~8%/year decay, halved), so ~40 URLs return $\bot$. Of the 960 survivors, byte identity $I$ is typically far below 1 even at $\Delta = 0$ — a single injected CSRF token, ad slot or "last updated" string flips SHA256. Suppose the $\Delta{=}0$ control gives $I_0 = 0.31$, $J_0 = 0.991$. At $\Delta = 180$ with the context also changed, suppose $I = 0.18$, $J = 0.93$.

The tempting reading: "6% of text drifted in six months." That reading is unavailable. The 180-day cell changed *two* factors at once, and each is known to move text on its own — a browser UA past a bot wall returns full article bodies where CCBot got a stub; a different ASN returns a different consent interstitial or a geo-localized page. From $(J_0, J)$ alone, $\sigma^2_{\text{time}}$ and $\sigma^2_{\text{context}}$ enter only through their sum. Every allocation from (0.06, 0.00) to (0.00, 0.06) fits the data exactly.

Now push it downstream. Suppose 40 dead URLs concentrate in one stratum and their replacements — expired domains re-registered, per Carlini et al. — serve new content. At 0.4% of the list, that is above the 0.01% fraction shown sufficient for practical poisoning. So the same nominal spec, rerun, yields a corpus differing in a way that is (i) not measurable as web change versus crawler change, and (ii) potentially adversarially chosen. The obstruction is not that the difference is large. It is that no experiment on a single re-crawl can say what the difference *is*.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*