---
id: 06-data-pipeline/filtering-disparate-impact
title: "Filtering's Disparate Impact Across Dialects and Groups"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Filtering's Disparate Impact Across Dialects and Groups

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/filtering-disparate-impact` · **Status:** empirically-open

## 1. Problem Statement

Every large pretraining corpus is produced by a chain of filters: language ID, a "quality" classifier (usually fastText trained to separate curated reference text from raw crawl), heuristic rules (line length, punctuation ratio, stopword coverage), a blocklist of banned terms, a toxicity classifier, and near-deduplication. Each filter is a binary map on documents. The question is what these maps do to text produced by particular groups — speakers of African American English (AAE), Indian English, Nigerian English, low-resource languages, LGBTQ+ authors, disability communities — and whether the resulting corpus shift propagates into model behaviour.

Three variants, of very different difficulty:

- **Measurement.** Given a corpus $D$, a filter $f$, and a group label $g$, estimate the group-conditional retention rate $r_g = \Pr[f(x)=1 \mid g(x)=g]$ and the disparity $\Delta_{g,g'} = r_g - r_{g'}$. Hard because $g$ is never observed; it is imputed by a classifier with its own error profile.
- **Method.** Build a filter with comparable utility (measured as downstream loss or benchmark score at fixed token budget) and smaller disparity. Requires a definition of "comparable" and a chosen fairness criterion.
- **Theory.** Given a per-group retention rate, predict the change in model performance on that group's text. No such transfer function exists.

Solving the problem means: a reproducible per-group retention audit that survives classifier-error correction, plus a demonstrated causal link from retention gap to downstream capability gap, plus at least one filter that closes the second without paying the first.

## 2. Formal Setting

Let $\mathcal{X}$ be documents, $D = \{x_i\}_{i=1}^N$ the pre-filter crawl ($N \approx 10^{10}$ for a Common Crawl snapshot). A pipeline is a composition $f = f_K \circ \cdots \circ f_1$, $f_k: \mathcal{X} \to \{0,1\}$.

**Group variable.** $g(x) \in \mathcal{G}$ is latent. In practice it is replaced by $\hat{g}(x)$: for dialect, the mixed-membership posterior of Blodgett et al. (2016), giving $p_{\text{AAE}}(x) \in [0,1]$, thresholded (typically $>0.5$ or $>0.8$). Measured retention is therefore
$$\hat{r}_g = \frac{\sum_i \mathbb{1}[\hat g(x_i)=g]\,f(x_i)}{\sum_i \mathbb{1}[\hat g(x_i)=g]}.$$
With classifier confusion matrix $C$ ($C_{g'g} = \Pr[\hat g = g' \mid g = g]$), the true vector $r$ satisfies $\hat r \approx C^\top r$ up to base-rate weighting; recovering $r$ needs $C$ estimated on in-domain labelled data, which does not exist for web crawl. This is the identifiability crux.

**Disparity measures.** Retention ratio $\rho_{g,g'} = r_g / r_{g'}$ (the "80% rule" analogue), and representation shift
$$\delta_g = \frac{\pi_g^{\text{post}}}{\pi_g^{\text{pre}}} = \frac{r_g}{\sum_{g'} \pi_{g'}^{\text{pre}} r_{g'}},$$
where $\pi_g$ is the token share of group $g$. Note $\delta$ is what a model sees; $r$ is what an audit reports, and they differ whenever document lengths differ by group.

**Downstream quantity.** Train $\theta(f)$ on $f(D)$ under fixed token budget $B$ and compute budget $C$. The capability gap is
$$G_{g,g'}(f) = \mathcal{L}_g(\theta(f)) - \mathcal{L}_{g'}(\theta(f)),$$
with $\mathcal{L}_g$ measured as held-out bits-per-byte on a group-specific eval set, or task accuracy on dialect-perturbed benchmarks (VALUE / Multi-VALUE). The theory variant asks for $\partial G / \partial \log r_g$.

**Assumptions and which are violated.**
1. *Group membership is a document-level property.* Violated: code-switching, multi-author pages, quoted speech.
2. *The dialect classifier is unbiased with respect to the filter.* Violated by construction — both key on the same surface features (non-standard orthography, slang), so errors are correlated and $\hat\Delta$ is biased away from zero in an unknown direction.
3. *Filters are independent.* Violated: dedup removes what quality filters would have kept; order matters.
4. *Held-out group evals are drawn from the same distribution as the filtered-out text.* Violated — the eval sets are curated, the removed text is not.

## 3. State of the Art

**Established (audits, reproduced).** Dodge et al. (EMNLP 2021) audited C4 and showed the blocklist filter removes documents at markedly different rates by imputed dialect — AAE-classified documents removed several-fold more often than White-aligned English — and disproportionately removes non-offensive LGBTQ+ content. The audit code and the C4 artefacts are public and the finding has been re-derived on other snapshots. Gururangan et al. (EMNLP 2022) showed the GPT-3/Pile-style quality classifier assigns higher scores to writing from schools in wealthier, better-educated, urban areas, on a corpus of US high-school newspapers with linked school demographics — i.e. the "quality" signal encodes a language ideology, not an information-content measure.

**Established (downstream, but narrow).** Sap et al. (ACL 2019) showed toxicity classifiers flag AAE tweets as offensive at far higher rates, and that annotator dialect priming reduces it. Welbl et al. (Findings EMNLP 2021) and Xu et al. (NAACL 2021 workshop) showed detoxification raises LM perplexity on minority-dialect and marginalised-identity text more than on the general distribution — the clearest existing evidence that a filter/intervention gap propagates to model loss.

**Claimed but unablated.** Modern pipelines — FineWeb / FineWeb-Edu (Penedo et al., NeurIPS D&B 2024), DataComp-LM (Li et al., NeurIPS D&B 2024), Dolma (Soldaini et al., ACL 2024) — report benchmark gains from aggressive model-based quality filtering. None reports a per-group retention audit of its own classifier, and none ablates whether the gain would survive a disparity-corrected variant. The improvements are benchmark numbers (MMLU, HellaSwag, CORE) on English-centric evals that contain essentially no dialect variation, so they are silent on the question.

**Method SOTA.** There is no deployed filter designed for retention parity. The nearest things are dialect-aware data augmentation (Multi-VALUE, Ziems et al., ACL 2023) applied at finetuning time, and per-source rebalancing, neither of which touches the filter itself.

## 4. What Is Known

- C4 (Raffel et al., JMLR 2020) is ~156B tokens from one crawl snapshot; its blocklist alone drops on the order of a tenth of documents, and Dodge et al. measured that drop as strongly dialect-dependent, with AAE-imputed documents removed at roughly an order-of-magnitude higher rate than White-aligned English documents. Scale: full C4, 365M documents.
- Sap et al. (2019): on ~100k tweets, AAE-imputed tweets were labelled offensive by widely used classifiers at roughly twice the rate of other tweets; annotator dialect priming cut the gap materially.
- Kreutzer et al. (TACL 2022) hand-audited 205 language corpora from CCAligned/ParaCrawl/WikiMatrix/mC4 and found many low-resource languages with under 50% correct-language content — language ID failure is the dominant filter failure for non-English, and it is worst exactly where data is scarcest.
- Welbl et al. (2021): detoxification at 400M–2.7B scale increases LM loss on AAE-aligned and identity-mentioning text disproportionately; the effect grows with filter aggressiveness.
- Gururangan et al. (2022): quality-classifier score correlates positively with school-level wealth/education proxies across ~10k US high-school newspaper articles.
- Reproduced regularity: near-dedup removes boilerplate and improves loss (Lee et al., ACL 2022, at 1.5B scale) — the one filter with a clean, group-agnostic justification.

## 5. What Is Not Known

- **Empirically open.** Whether the measured retention gaps change downstream capability on dialect and group-specific tasks at frontier scale. Nobody has trained matched 1B–8B models on parity-corrected versus standard-filtered corpora and compared dialect-eval gaps. The experiment is runnable today for well under $10^5$ GPU-hours; it has not been run.
- **Empirically open.** The per-filter decomposition. Published audits look at blocklists and one quality classifier. The relative contribution of language ID, perplexity pruning, model-based quality scoring, and dedup to the total gap is unmeasured on any modern pipeline.
- **Methodologically blocked.** Correcting the audit for dialect-classifier error. Because the dialect classifier and the filter share surface features, $\hat\Delta$ is a biased estimate of $\Delta$ with no available in-domain confusion matrix. No published audit reports a corrected estimate or an error bar for this.
- **Theoretically open.** Any transfer function from retention rate to per-group loss. Scaling laws are fit on aggregate token counts; there is no proof or fit that says halving a subpopulation's tokens costs $\alpha$ nats on that subpopulation, and no result on whether cross-group transfer compensates.

## 6. Why It Is Hard

The binding obstruction is **non-identifiability from correlated measurement error**. The group label is imputed by a classifier keyed on non-standard orthography, slang, and morphosyntax — exactly the features the quality filter and blocklist penalise. So a document removed by the filter is also more likely to be *labelled* AAE, inflating the apparent gap; and a filtered-in AAE document written in standard orthography is likely labelled non-AAE, deflating the AAE base rate. The two errors push in opposite directions with unknown magnitudes, and there is no gold-labelled web-crawl sample to estimate the confusion matrix on. Every published number is therefore a point estimate with no defensible interval.

Second obstruction: **the evaluation does not measure what it names.** "Quality filtering improves the model" is established only on MMLU/HellaSwag-style evals written in edited Standard American English. Those evals cannot register a dialect capability loss, so the standard ablation is structurally incapable of detecting the harm. Third: **compute cost of the counterfactual** — the only clean answer requires training matched models, and matched-pair pretraining runs are rarely funded for fairness questions.

## 7. Current Research (as of 2026)

- Open-corpus groups (AI2 on Dolma/OLMo, HuggingFace on FineWeb, the DataComp-LM consortium) release filter code and intermediate artefacts, which makes retention audits possible without re-crawling. Audits of these specific pipelines are appearing but remain descriptive *(frontier — verify)*.
- Dialect robustness evaluation (Ziems, Yang and collaborators, Georgia Tech / Stanford) has moved from VALUE to multi-dialect perturbation suites covering Indian, Nigerian, Singaporean and Appalachian English — supplying the missing eval axis.
- Multilingual curation (CulturaX, HPLT, Common Corpus) is re-examining language ID thresholds after the Kreutzer audit; per-language retention curves are being published *(frontier — verify)*.
- Reweighting rather than filtering (DoReMi-style group-distributionally-robust domain weights, Xie et al., NeurIPS 2023) is the most plausible mitigation route: keep the document, change its sampling weight. Applied to dialect groups, this is untested.

## 8. Concrete Next Experiment

**Scale.** One Common Crawl snapshot, ~$3\times10^{11}$ raw tokens. Two 1.4B-parameter decoder models, Chinchilla-optimal $\approx 28$B tokens each, ~$4\times10^3$ A100-hours per run. Total under 10k GPU-hours including a 400M pilot.

**Arms.**
- *Control:* standard pipeline — fastText quality classifier (FineWeb-Edu style), C4 blocklist, MinHash dedup.
- *Treatment:* identical, except the quality classifier and blocklist are replaced by a **retention-parity-constrained** variant: adjust per-bucket thresholds so that $\rho_{g,\text{WAE}} \ge 0.8$ for AAE, Indian English and Nigerian English buckets ($\hat g$ from a multi-dialect classifier), keeping total retained tokens equal to the control to within 1%. Equal-token matching is the point — it removes "more data" as an explanation.

**Deciding number.** The dialect gap $G = \text{acc}_{\text{SAE}} - \text{acc}_{\text{dialect}}$, averaged over Multi-VALUE-perturbed GLUE/SQuAD, plus bits-per-byte on a held-out dialect corpus. Decision rule: if the treatment reduces $G$ by $\ge 3$ accuracy points while losing $\le 0.5$ points on aggregate MMLU/CORE, filtering disparity is a *causal* driver of dialect capability gaps and parity constraints are cheap. If $G$ moves by $<1$ point, the retention gap is not the mechanism, and attention should shift to eval-set and tokenizer effects. Report both with three seeds; the gap's seed variance at 1.4B is roughly 1 point, so the 3-point threshold is the smallest defensible one.

**Cheap prerequisite.** Hand-label 2,000 stratified crawl documents for dialect to estimate the classifier confusion matrix in-domain, and publish the corrected retention interval. This alone unblocks the measurement question at a cost of ~200 annotator-hours.

## 9. Key References

- **[Foundational]** Jesse Dodge, Maarten Sap, Ana Marasović, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, Matt Gardner. *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus.* EMNLP 2021. — arXiv:2104.08758
- **[Foundational]** Su Lin Blodgett, Lisa Green, Brendan O'Connor. *Demographic Dialectal Variation in Social Media: A Case Study of African-American English.* EMNLP 2016. — arXiv:1608.08868
- **[Foundational]** Maarten Sap, Dallas Card, Saadia Gabriel, Yejin Choi, Noah A. Smith. *The Risk of Racial Bias in Hate Speech Detection.* ACL 2019.
- **[SOTA]** Suchin Gururangan, Dallas Card, Sarah K. Dreier, Emily K. Gade, Leroy Z. Wang, Zeyu Wang, Luke Zettlemoyer, Noah A. Smith. *Whose Language Counts as High Quality? Measuring Language Ideologies in Text Data Selection.* EMNLP 2022. — arXiv:2201.10474
- **[SOTA]** Johannes Welbl, Amelia Glaese, Jonathan Uesato, Sumanth Dathathri, John Mellor, Lisa Anne Hendricks, Kirsty Anderson, Pushmeet Kohli, Ben Coppin, Po-Sen Huang. *Challenges in Detoxifying Language Models.* Findings of EMNLP 2021. — arXiv:2109.07445
- **[SOTA]** Albert Xu, Eshaan Pathak, Eric Wallace, Suchin Gururangan, Maarten Sap, Dan Klein. *Detoxifying Language Models Risks Marginalizing Minority Voices.* NAACL 2021. — arXiv:2104.06390
- **[SOTA]** Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, Thomas Wolf. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS Datasets & Benchmarks 2024. — arXiv:2406.17557
- **[SOTA]** Caleb Ziems, William Held, Jingfeng Yang, Jwala Dhamala, Rahul Gupta, Diyi Yang. *Multi-VALUE: A Framework for Cross-Dialectal English NLP.* ACL 2023. — arXiv:2212.08011
- **[Survey]** Julia Kreutzer et al. *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.* TACL 2022. — arXiv:2103.12028
- **[Survey]** Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell. *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?* FAccT 2021.
- **[Context]** Colin Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 21(140), 2020. — arXiv:1910.10683

## 10. Worked Example

Take 1,000,000 crawl documents. Suppose the true dialect split is 3% AAE, 97% other, and the true retention rates under a blocklist are $r_{\text{AAE}} = 0.60$, $r_{\text{other}} = 0.90$ — a real but moderate gap, $\rho = 0.67$.

Now impute dialect with a classifier that has recall 0.70 on AAE and false-positive rate 0.02 on other text, **and** whose errors correlate with the filter: assume among the 40% of AAE documents the blocklist removes, the classifier's recall is 0.85 (heavy non-standard orthography, easy to flag), while among retained AAE documents recall is 0.60.

Counts:
- True AAE = 30,000. Removed: 12,000, of which 0.85 → 10,200 labelled AAE. Retained: 18,000, of which 0.60 → 10,800 labelled AAE.
- True other = 970,000. Removed: 97,000, of which 2% → 1,940 labelled AAE. Retained: 873,000, of which 2% → 17,460 labelled AAE.

Measured: $\hat r_{\text{AAE}} = (10{,}800+17{,}460)/(10{,}200+10{,}800+1{,}940+17{,}460) = 28{,}260/40{,}400 = 0.70$. Measured other-retention $\approx (1{,}000{,}000\cdot0.891 - 28{,}260)/(1{,}000{,}000-40{,}400) = 0.898$. So $\hat\rho = 0.78$ against a true $\rho = 0.67$ — the audit *understates* the gap by 11 points and lands just above the 0.8 rule-of-thumb threshold that would have triggered concern.

Flip one assumption — make the classifier's recall on removed AAE 0.85 but let false positives concentrate in *removed* other-text (plausible, since both keyed on slang, say 4% removed vs 1.8% retained) — and the same true gap can be made to look worse than it is. Same ground truth, opposite conclusions, driven entirely by an unmeasured confusion matrix.

That is the obstruction in one calculation: without an in-domain, filter-stratified estimate of $C$, the published retention ratios are not just imprecise, their *direction of bias* is unknown. The 2,000-document hand-labelling prerequisite in §8 exists precisely to pin down the four numbers used above.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*