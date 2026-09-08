---
id: 06-data-pipeline/detecting-generated-text-in-crawls
title: "Detecting Machine-Generated Text in Web Crawls"
topic: 06-data-pipeline
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Detecting Machine-Generated Text in Web Crawls

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/detecting-generated-text-in-crawls` · **Status:** open

## 1. Problem Statement

**Input.** A web-scale crawl snapshot $C = \{d_1, \dots, d_N\}$, $N \approx 2$–$3 \times 10^9$ documents (Common Crawl), with no per-document provenance.

**Output.** Either (a) a per-document label $\hat{y}_i \in \{\text{human}, \text{machine}\}$, or (b) a scalar prevalence estimate $\hat{\pi}$ of machine-generated content, or (c) a retained subset $C' \subseteq C$ that maximizes downstream model quality.

Three variants that are routinely conflated:

- **Measurement.** Estimate $\pi$, and its drift across snapshots, to within a stated error bar. Population-level; does not require any single document to be labeled correctly.
- **Method.** Build a classifier $\hat{y}(d)$ that is useful at a crawl-scale false-positive budget — meaning FPR $\lesssim 10^{-3}$, because $0.1\%$ of $2.7\times10^9$ is 2.7 M wrongly discarded pages.
- **Theory.** Determine whether detection is possible *at all* as generator quality rises, i.e. as the total variation distance between machine and human text distributions goes to zero.

Solving it means: a curation rule with a *measured* effect on downstream loss, not a detector with a high in-domain AUROC.

## 2. Formal Setting

Let $\mathcal{H}$ be the distribution over human-authored documents and $\mathcal{M}$ over machine-generated ones. A crawl is a mixture

$$\mathcal{C}_\pi = (1-\pi)\,\mathcal{H} + \pi\,\mathcal{M}, \qquad \pi \in [0,1].$$

A detector is a score $s: \mathcal{X} \to \mathbb{R}$ with threshold $\tau$; $\hat{y}(d) = \mathbb{1}[s(d) > \tau]$. Measured quantities:

- **TPR/FPR**: $\mathrm{TPR}(\tau) = \Pr_{d\sim\mathcal{M}}[s(d)>\tau]$, $\mathrm{FPR}(\tau) = \Pr_{d\sim\mathcal{H}}[s(d)>\tau]$ — estimated on a labeled evaluation set, *never* on the crawl itself, because the crawl has no labels.
- **Precision under base rate**: $\mathrm{Prec} = \dfrac{\pi\,\mathrm{TPR}}{\pi\,\mathrm{TPR} + (1-\pi)\,\mathrm{FPR}}$. This is the number that matters and it is dominated by $(1-\pi)\mathrm{FPR}$ whenever $\pi$ is small.
- **Prevalence estimate** (no per-document labels needed): with the detector's operating point known, $\hat{\pi} = \dfrac{\hat{p} - \mathrm{FPR}}{\mathrm{TPR} - \mathrm{FPR}}$, where $\hat{p}$ is the observed positive rate. This is the Liang et al. (2024) distributional estimator in its simplest form.
- **Detectability ceiling**: any detector obeys $\mathrm{TPR} - \mathrm{FPR} \le \mathrm{TV}(\mathcal{M},\mathcal{H})$, hence $\mathrm{AUROC} \le \tfrac{1}{2} + \mathrm{TV} - \tfrac{\mathrm{TV}^2}{2}$ (Sadasivan et al., 2023).
- **Curation objective**: choose $C'$ minimizing $\mathbb{E}[\mathcal{L}_{\text{eval}}(\theta(C'))]$ under a fixed token budget $B$ — measured by actually pretraining on $C'$, not by detector accuracy.

Assumptions, with the ones known to be violated in practice marked:

1. Documents are i.i.d. draws. **Violated** — crawls contain near-duplicates, syndicated copies, and template farms; effective sample size is far below $N$.
2. Labels are binary. **Violated** — human-drafted/LLM-edited, LLM-drafted/human-edited, and machine-translated text occupy a continuum. Thompson et al. (2024) find machine translation is pervasive and predates ChatGPT.
3. The generator set is known and fixed. **Violated** — the crawl mixes hundreds of models, decoding settings, and system prompts, with composition drifting each snapshot.
4. Detector calibration transfers across domains. **Violated** — FPR measured on news transfers poorly to forums, code-adjacent text, or non-native English (Liang et al., 2023).
5. $\mathcal{H}$ is stationary. **Violated** — human writing style is itself shifting toward LLM-typical vocabulary, which contaminates the negative class.

## 3. State of the Art

**Theory SOTA (established).** Sadasivan et al. (*Can AI-Generated Text Be Reliably Detected?*, TMLR 2025; arXiv 2303.11156) give the TV bound above and a recursive-paraphrase attack realizing it. Chakraborty et al. (*On the Possibilities of AI-Generated Text Detection*, 2023) give the complementary positive result: for $\mathrm{TV} > 0$, detection to any target error is possible with $O(1/\mathrm{TV}^2)$ i.i.d. samples from the *same source* — a result about sources and corpora, not single documents. Zhang et al. (*Watermarks in the Sand*, ICLR 2024) prove strong watermarking is impossible given a quality oracle plus a random-walk perturbation oracle.

**Empirical SOTA (established).** Binoculars (Hans et al., ICML 2024) scores a document by the ratio of its cross-perplexity under an observer/performer LLM pair (Falcon-7B / Falcon-7B-Instruct); zero-shot, no training on any generator, reported >90% TPR on ChatGPT text at 0.01% FPR in-domain. Fast-DetectGPT (Bao et al., ICLR 2024) replaces DetectGPT's perturbation loop with a conditional-probability curvature statistic, ~340× cheaper. Ghostbuster (Verma et al., NAACL 2024) trains a classifier over features from weaker LMs.

**Claimed but unablated.** No published result shows that removing detector-flagged documents from a pretraining corpus improves a downstream model at any scale. Every headline detector number is a *benchmark number on a curated evaluation set* — usually balanced ($\pi = 0.5$), single-generator, single-domain — not a crawl measurement. Commercial detectors (GPTZero, Originality.ai, Pangram) report crawl-scale prevalence figures without a published false-positive audit.

## 4. What Is Known

- **Adversarial fragility, at scale.** RAID (Dugan et al., ACL 2024): 6 M+ generations, 11 models, 8 domains, 11 attacks. Detectors tuned to 5% FPR degrade sharply under paraphrase and, most severely, homoglyph substitution, where several open detectors fall to near-zero TPR.
- **Cross-domain collapse.** MAGE (Li et al., ACL 2024) shows out-of-distribution generator/domain pairs cost tens of AUROC points versus in-distribution.
- **Commercial-grade failure at deployment.** OpenAI's AI Text Classifier reported 26% TPR at 9% FPR and was withdrawn in July 2023 for low accuracy.
- **Demographic bias.** Liang et al. (*Patterns*, 2023): GPT detectors misclassified over half of TOEFL essays by non-native English writers as machine-generated, while near-correctly labeling US 8th-grade essays.
- **Prevalence is measurable at population level.** Liang et al. (ICML 2024) estimate ~10.6% of ICLR 2024 review sentences were substantially LLM-modified, against a <2% pre-ChatGPT baseline — a distributional estimate with no per-review claim.
- **Machine text predates LLMs.** Thompson et al. (2024) find that in a multi-way parallel web corpus, a majority of sentences appearing in 3+ languages are machine translations, concentrated in low-resource languages.
- **Recursive training degrades models.** Shumailov et al. (*Nature*, 2024) demonstrate model collapse when each generation fully replaces its training data; Gerstgrasser et al. (2024) show *accumulating* real plus synthetic data instead bounds the test error — the collapse result does not directly imply crawl filtering is necessary.

## 5. What Is Not Known

- **Empirically open.** Does filtering machine-generated text from a pretraining corpus improve downstream loss? Runnable today at 1B-parameter / 100B-token scale; nobody has published the ablation with a matched-token control. This is the central gap.
- **Empirically open.** What is $\pi$ for a 2025–2026 Common Crawl snapshot, with error bars? All circulating figures (10%, 30%, 50%) come from detectors whose crawl-domain FPR is unmeasured.
- **Methodologically blocked.** There is no ground truth for a crawl. Any labeled evaluation set is either synthetic (you generated the positives, so you know the generator — the easy case) or human-labeled (annotators are near chance on fluent LLM text). The quantity "fraction of the web that is machine-generated" has no operational definition for hybrid documents.
- **Theoretically open.** Whether *source-level* detection (Chakraborty-style, aggregating many documents from one domain) has a sample-complexity advantage that survives adversarial paraphrasing. The TV bound applies per-document; the multi-sample regime under attack is unresolved.

## 6. Why It Is Hard

The obstruction is **absent ground truth compounded by base-rate arithmetic**, not compute.

Compute is affordable: Binoculars over a full snapshot is $2 \times 2N_{\text{params}} \times T$ FLOPs $= 2 \times 2(7\times10^9)(2.7\times10^{12}) \approx 7.6\times10^{22}$ FLOPs, roughly $5\times10^4$ H100-hours, order $10^5$ USD — comparable to a mid-size training run's preprocessing. That is not the blocker.

The blocker: to convert an observed positive rate into a prevalence you need $\mathrm{FPR}$ *on crawl-distributed human text*, and you cannot measure it without labels you do not have. Pre-2022 snapshots give a partial anchor for the human class but not the current one, since human writing has since drifted. Meanwhile precision is set by $(1-\pi)\mathrm{FPR}$: at $\pi = 0.05$, an FPR error of 1 percentage point changes precision by tens of points. The evaluation also does not measure what it names — "detector AUROC on RAID" is not "loss reduction from filtering", and no published work bridges the two.

## 7. Current Research (as of 2026)

- **Zero-shot statistical detectors.** Maryland (Goldstein, Hans, Kirchenbauer), CUHK-Shenzhen (Bao) — cross-perplexity and curvature statistics; the live question is calibration transfer, not raw AUROC.
- **Watermarking at inference.** Google DeepMind's SynthID-Text (*Nature*, 2024) deploys tournament sampling in production; coverage of the open-weights ecosystem remains near zero, which caps crawl-level recall regardless of watermark strength. *(frontier — verify current deployment breadth.)*
- **Population-level quantification.** Stanford (Zou, Liang) — distributional estimators applied to peer review, job postings, consumer complaints; extension to full crawls is the natural next step. *(frontier — verify.)*
- **Curation-effect studies.** FineWeb/HuggingFace, DataComp-LM, and Allen AI (OLMo/Dolma) publish filter ablations; none isolates a machine-generated-text filter as an arm. *(frontier — verify.)*
- **Synthetic-data scaling laws.** Dohmatob et al. (*A Tale of Tails*, ICML 2024; *Strong Model Collapse*, 2024) — quantifying how a synthetic fraction distorts scaling exponents.

## 8. Concrete Next Experiment

**Question.** Does removing detector-flagged text from a pretraining corpus improve downstream loss, and what is the crawl's true $\pi$?

**Scale.** Two arms, each pretraining a 1.4B-parameter decoder on 100B tokens (~$3\times10^{21}$ FLOPs per run; ~4k H100-hours per arm).

- **Treatment arm.** Score a 2025 Common Crawl snapshot with Binoculars; drop the top-flagged documents until 10% of tokens are removed; backfill from unflagged documents to restore exactly 100B tokens.
- **Control arm (essential).** Drop 10% of tokens *at random* and backfill identically. This isolates the filter from the mere loss-of-data and duplication effects that a naive "filtered vs. unfiltered" comparison confounds.
- **Calibration sub-experiment.** Estimate FPR on human text by scoring 1M documents from CC-MAIN-2019-* (pre-LLM, so $\pi \approx 0$ except for machine translation, which is separately flagged by language-pair heuristics), stratified by domain and by inferred author language. Report FPR per stratum. Then compute $\hat{\pi} = (\hat{p} - \mathrm{FPR})/(\mathrm{TPR} - \mathrm{FPR})$ on the 2025 snapshot with stratum-weighted FPR.

**The deciding number.** The difference in mean downstream loss between the two arms on a held-out 2019-crawl validation set plus 8 standard benchmarks. Threshold: filtering is worth doing if it beats the random-drop control by $\ge 0.02$ nats of validation loss (roughly a 5–10% effective-compute gain at this scale). Below that, the filter is noise and the community should stop reporting detector AUROCs as if they were curation evidence.

**Secondary number.** The pre-LLM FPR spread across strata. If FPR varies by more than 3× between the lowest and highest strata, no single global threshold can be crawl-calibrated, and per-stratum thresholds become mandatory.

## 9. Key References

- **[Foundational]** Gehrmann, Strobelt, Rush. *GLTR: Statistical Detection and Visualization of Generated Text.* ACL (demo), 2019. — arXiv:1906.04043
- **[Foundational]** Mitchell, Lee, Khazatsky, Manning, Finn. *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature.* ICML, 2023. — arXiv:2301.11305
- **[Theory]** Sadasivan, Kumar, Balasubramanian, Wang, Feizi. *Can AI-Generated Text Be Reliably Detected?* TMLR, 2025 (first posted 2023). — arXiv:2303.11156
- **[Theory]** Chakraborty, Bedi, Zhu, An, Manocha, Huang. *On the Possibilities of AI-Generated Text Detection.* 2023. — arXiv:2304.04736
- **[Theory]** Zhang, Edelman, Francati, Venturi, Ateniese, Barak. *Watermarks in the Sand: Impossibility of Strong Watermarking for Language Models.* ICLR, 2024.
- **[SOTA]** Hans, Schwarzschild, Cherepanova, Kazemi, Saha, Goldblum, Geiping, Goldstein. *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text.* ICML, 2024. — arXiv:2401.12070
- **[SOTA]** Bao, Zhao, Teng, Yang, Zhang. *Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature.* ICLR, 2024.
- **[Benchmark]** Dugan, Hwang, Trhlík, Ludan, Zhu, Xu, Ippolito, Callison-Burch. *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors.* ACL, 2024.
- **[Benchmark]** Li, Li, Liu, Wang, Zhang, et al. *MAGE: Machine-Generated Text Detection in the Wild.* ACL, 2024.
- **[Watermarking]** Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein. *A Watermark for Large Language Models.* ICML, 2023.
- **[Watermarking]** Dathathri, See, Ghaisas, et al. *Scalable Watermarking for Identifying Large Language Model Outputs.* Nature, 2024.
- **[Prevalence]** Liang, Zhang, Cao, Wang, Ding, et al. *Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews.* ICML, 2024.
- **[Prevalence]** Thompson, Dhaliwal, Frisch, Domhan, Federico. *A Shocking Amount of the Web is Machine Translated: Insights from Multi-Way Parallelism.* Findings of ACL, 2024. — arXiv:2401.05749
- **[Bias]** Liang, Yuksekgonul, Mao, Wu, Zou. *GPT Detectors Are Biased Against Non-Native English Writers.* Patterns, 2023.
- **[Consequence]** Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. *AI Models Collapse When Trained on Recursively Generated Data.* Nature 631, 2024.
- **[Consequence]** Gerstgrasser, Schaeffer, Dey, et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM, 2024.
- **[Survey]** Wu, Yang, Zhan, Yuan, Cui, Wong. *A Survey on LLM-Generated Text Detection.* Computational Linguistics, 2025.

## 10. Worked Example

Take one snapshot: $N = 2.7\times10^9$ pages. Suppose the true machine fraction is $\pi = 0.05$ and you run Binoculars at its published in-domain operating point, TPR $= 0.90$.

**If the in-domain FPR of $10^{-4}$ held on the crawl:**

$$\mathrm{TP} = 0.05 \cdot 2.7\times10^9 \cdot 0.90 = 1.22\times10^8, \qquad \mathrm{FP} = 0.95 \cdot 2.7\times10^9 \cdot 10^{-4} = 2.6\times10^5$$

Precision $= 0.998$. Filtering is clean.

**Now use the number the crawl actually justifies.** Nothing in the literature measures Binoculars' FPR on forum posts, product listings, or non-native English pages at threshold. Suppose the realized crawl-average FPR is $0.02$ — well inside the range implied by the non-native-English miss rates in Liang et al. (2023) and the cross-domain drops in MAGE:

$$\mathrm{FP} = 0.95 \cdot 2.7\times10^9 \cdot 0.02 = 5.13\times10^7, \qquad \mathrm{Prec} = \frac{1.22\times10^8}{1.22\times10^8 + 5.13\times10^7} = 0.70$$

Thirty percent of what you discard is human text — 51 million pages — and by the bias result it is concentrated in non-native English and low-resource-language content, exactly the tail the corpus is thinnest in.

**The obstruction, made visible.** The two calculations differ by a factor of 200 in false positives, and *nothing in the crawl distinguishes them*, because there are no labels to estimate FPR against. Reversing the estimator makes it worse: an observed positive rate of $\hat{p} = 0.068$ yields $\hat{\pi} = (0.068 - 10^{-4})/(0.90 - 10^{-4}) = 0.075$ under the optimistic FPR and $\hat{\pi} = (0.068 - 0.02)/(0.88) = 0.055$ under the pessimistic one — and the same $\hat{p}$ with FPR $= 0.068$ gives $\hat{\pi} = 0$. The identical measurement supports "7.5% of the web is machine-generated" or "none of it is". That is why every circulating crawl prevalence figure should be read as a statement about the detector's uncalibrated threshold, not about the web.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*