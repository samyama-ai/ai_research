---
id: 30-synthetic-data/synthetic-contamination-detection-web
title: "Detecting Synthetic Contamination in Web-Scale Corpora"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Detecting Synthetic Contamination in Web-Scale Corpora

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/synthetic-contamination-detection-web` · **Status:** open

## 1. Problem Statement

**Input.** A web crawl $C$ of $N \sim 10^{10}$ documents (Common Crawl, FineWeb, RedPajama), with crawl timestamps and URLs, and no per-document provenance labels.

**Output, three variants of increasing difficulty:**

- **Measurement variant.** Estimate the scalar *contamination rate* $\alpha$ — the fraction of tokens in $C$ produced or substantially rewritten by a generative model — with a stated confidence interval. Solving this means an interval narrow enough to be actionable (say $\pm 1$ point absolute) that survives a held-out audit.
- **Method variant.** Produce a per-document score $s(x) \in [0,1]$ usable as a *filter*, such that reweighting $C$ by $s$ improves downstream pretraining loss relative to unfiltered $C$ at matched token budget. Solving this means a measured perplexity or benchmark gain, not a detector AUROC.
- **Theory variant.** Characterise when $\alpha$ is *identifiable* at all from an unlabelled corpus, given that generators are trained to imitate the human distribution and that human authors edit model output.

These come apart. A detector can have AUROC $0.95$ on a benchmark and still give a useless $\hat\alpha$ (Section 6). A perfect $\hat\alpha$ says nothing about which documents to drop. Conflating them is the usual reason the literature looks further along than it is.

## 2. Formal Setting

Let $\mathcal{X}$ be the space of documents. Human text is drawn from $H$, model text from $M_\theta$ for generator $\theta$ in an unknown, growing family $\Theta$ with unknown mixing weights $\pi$. The observed corpus is a mixture

$$C \sim (1-\alpha)\,H \;+\; \alpha \sum_{\theta \in \Theta} \pi_\theta\, M_\theta .$$

**Quantities as measured.**

- $\alpha$ is measured as a *token-weighted* fraction, $\alpha = \big(\sum_i \ell_i z_i\big)/\sum_i \ell_i$ with $\ell_i$ the token count and $z_i \in \{0,1\}$ the latent synthetic indicator. Document-weighted and token-weighted $\alpha$ differ by several points because model output is length-biased.
- A detector is a score $s: \mathcal{X}\to\mathbb{R}$ thresholded at $\tau$. Its operating point is $(\mathrm{TPR}(\tau), \mathrm{FPR}(\tau))$, estimated on a labelled probe set $P$.
- The **prevalence estimator** is the inverted positive rate. With $\hat q = \frac{1}{N}\sum_i \mathbb{1}[s(x_i) > \tau]$,
  $$\hat\alpha = \frac{\hat q - \mathrm{FPR}}{\mathrm{TPR} - \mathrm{FPR}}, \qquad \operatorname{Var}(\hat\alpha) \approx \frac{\hat q(1-\hat q)}{N(\mathrm{TPR}-\mathrm{FPR})^2}.$$
  Sampling noise vanishes at $N=10^{10}$; the error is entirely bias in $\mathrm{TPR}, \mathrm{FPR}$.
- The **distributional** alternative (Liang et al., ICML 2024) skips per-document decisions: fit $\alpha$ by maximum likelihood on corpus-level token or word-frequency statistics, $\hat\alpha = \arg\max_\alpha \sum_j \log\big[(1-\alpha)p_H(w_j) + \alpha\, p_M(w_j)\big]$, using reference $p_H, p_M$ estimated from certified-human and certified-model corpora.

**Assumptions, and their status.**

| Assumption | Status |
|---|---|
| $z_i$ is binary | **Violated.** Human-edited model output and model-edited human output are a continuum. |
| $\Theta$ is known and probeable | **Violated.** Closed API models change silently; open weights number in the thousands. |
| $P$ (probe set) has clean labels | **Violated.** Post-2022 "human" text is itself contaminated; pre-2022 text is distributionally shifted. |
| $\mathrm{FPR}$ measured on $P$ transfers to $C$ | **Violated.** FPR varies by domain, register, and author nativeness (Section 4). |
| $\mathrm{TV}(H, M_\theta) \gg 0$ | Weakening by construction — generators are trained to minimise it. |

## 3. State of the Art

**Theory.** Sadasivan et al. (*Can AI-Generated Text be Reliably Detected?*, TMLR 2025 / arXiv:2303.11156) give the operative bound: for any detector $D$,
$$\mathrm{AUROC}(D) \le \tfrac{1}{2} + \mathrm{TV}(M,H) - \tfrac{1}{2}\mathrm{TV}(M,H)^2,$$
so detection degrades to chance as generators approach the human distribution. Chakraborty et al. (*On the Possibilities of AI-Generated Text Detection*, ICML 2024) give the complement: detection stays possible with enough i.i.d. samples, sample complexity growing as $\mathrm{TV}$ shrinks. Both are **established**. Zhang et al. (*Watermarks in the Sand*, ICML 2024) prove strong watermarking is impossible against an attacker with a quality oracle and a perturbation oracle — relevant because watermarking is the only route that sidesteps the TV bound.

**Empirical.** Binoculars (Hans et al., ICML 2024) is the strongest training-free detector: a cross-perplexity ratio between two closely related LLMs, reported above $90\%$ TPR at $0.01\%$ FPR on ChatGPT-generated news and essays. **This is a benchmark number**, measured on curated in-domain pairs, not on crawl-distributed text. DetectGPT (Mitchell et al., ICML 2023) reaches AUROC $\approx 0.95$ on XSum with a white-box scorer. RAID (Dugan et al., ACL 2024) is the best adversarial benchmark; detectors that report $>0.9$ AUROC in-domain fall sharply under paraphrase and decoding-strategy shift.

**Claimed but unablated.** That detector scores are usable as *pretraining filters*. No published run shows a matched-token-budget pretraining comparison where a synthetic-text filter beats standard quality filtering. FineWeb (Penedo et al., NeurIPS D&B 2024) ablates dozens of filters; none is a synthetic-text detector.

## 4. What Is Known

- **Contamination is already large in specific slices.** Thompson et al. (ACL Findings 2024) found $57.1\%$ of sentences in the multi-way-parallel ($\ge 3$ languages) portion of Common Crawl are machine-translated, with quality degrading in low-resource languages. Measured on 6.38B sentences.
- **Machine text predates ChatGPT in canonical corpora.** Dodge et al. (EMNLP 2021) documented machine-translated patent text in C4 (156B tokens).
- **Population-level estimation works where per-document detection fails.** Liang et al. (ICML 2024) estimate $6.5\%$–$16.9\%$ of sentences in ICLR/NeurIPS/EMNLP 2024 peer reviews were substantially LLM-modified, against a near-zero pre-ChatGPT baseline; the same method puts LLM-modified sentences in CS arXiv abstracts near $17\%$ by late 2024.
- **Detectors have a large, structured FPR.** Liang et al. (*Patterns*, 2023): seven commercial detectors misclassified TOEFL essays by non-native English writers as AI-generated at an average $61.3\%$ rate, while near-$0\%$ on US 8th-grade essays. This is a bias term, not noise.
- **Paraphrase collapses detection.** Krishna et al. (NeurIPS 2023): DIPPER paraphrasing dropped DetectGPT detection from $70.3\%$ to $4.6\%$ at $1\%$ FPR.
- **Contamination matters for training.** Shumailov et al. (*Nature*, 2024) show tail loss and rising perplexity over 9 generations of recursive training (OPT-125M scale). Gerstgrasser et al. (COLM 2024) show that *accumulating* real plus synthetic data — the actual web regime — bounds the error rather than diverging. Dohmatob et al. (ICML 2024) show synthetic data changes the scaling-law exponent rather than merely shifting the constant.

## 5. What Is Not Known

- **Theoretically open.** Identifiability of $\alpha$ from an unlabelled mixture when $\Theta$ is unknown and $M_\theta$ is trained on $H$. The mixture is not identifiable in general — the classical two-component mixture identifiability results require $H$ and $M$ to be distinguishable families, which fine-tuning explicitly destroys. No proof either way for the realistic case with side information (timestamps, URL structure, cross-document duplication).
- **Empirically open.** The filtering experiment. Nobody has run a matched-token-budget pretraining ablation — one arm filtered by a synthetic-text detector, one not — at $\ge 10$B parameters. It is runnable today for roughly $10^{22}$ FLOPs.
- **Methodologically blocked.** $\alpha$ itself. There is no certified-human corpus after 2022 to calibrate FPR, and no certified-synthetic corpus representative of the web's generator mixture. Every reported web-scale contamination number inherits an uncalibrated FPR bias of unknown sign and magnitude.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth compounded into estimator bias**. From Section 2, $\hat\alpha$ divides by $(\mathrm{TPR}-\mathrm{FPR})$, so an FPR misestimate of $\delta$ propagates to $\delta/(\mathrm{TPR}-\mathrm{FPR})$ in $\hat\alpha$. With the *measured* domain-to-domain FPR spread of tens of points (Liang 2023), and $\alpha$ plausibly in the single digits, the bias exceeds the estimand. You cannot fix this by collecting more data: $N=10^{10}$ already drives sampling variance to zero.

The second obstruction is **non-identifiability by construction**. Every generator improvement shrinks $\mathrm{TV}(M,H)$, which the Sadasivan bound converts directly into a detection ceiling. Unlike most measurement problems, the thing being measured is actively optimised to be unmeasurable.

Watermarking (Kirchenbauer et al., ICML 2023; SynthID-Text, Dathathri et al., *Nature* 2024) escapes both — but only for participating generators, which is a shrinking and self-selected share of $\Theta$, and only until the text is paraphrased.

## 7. Current Research (as of 2026)

- **Population-level estimation over per-document classification.** The Stanford/Liang line (ICML 2024, COLM 2024) is the methodologically soundest direction: estimate $\alpha$ without ever labelling a document.
- **Temporal-baseline auditing.** Using pre-2022 crawl snapshots as a negative control to difference out detector bias. *(frontier — verify)*
- **Provenance infrastructure.** C2PA content credentials for images; the Data Provenance Initiative (Longpre et al., *Nature Machine Intelligence*, 2024) for dataset licensing and lineage. Both sidestep detection by carrying labels.
- **Model-collapse-aware curation.** Whether contamination needs detecting at all, given the accumulation result (Gerstgrasser et al.) — the open question is whether the accumulation bound holds when the synthetic share of *new* text keeps rising.
- **Detector robustness benchmarks.** RAID (UPenn/Dugan) and follow-ons, now covering adversarial decoding and multilingual settings. *(frontier — verify current leaderboard)*

## 8. Concrete Next Experiment

**Question.** Does filtering by a synthetic-text detector improve pretraining, and is $\hat\alpha$ stable enough to trust?

**Scale.** Two 1.4B-parameter decoder models, 100B tokens each, from a single FineWeb snapshot. Compute $\approx 1.2\times10^{21}$ FLOPs per arm — roughly 2k H100-hours total.

**Arms.**
1. **Treatment:** drop the top decile by Binoculars score.
2. **Control (essential):** drop a *random* decile of the same token count. This isolates the detector's signal from the effect of merely training on 10% less data plus whatever quality correlate the detector picks up.
3. **Bias probe:** run the same detector and threshold on a 2019 Common Crawl snapshot, where true $\alpha \approx 0$.

**The deciding number.** The 2019 positive rate $\hat q_{2019}$. If $\hat q_{2019} > 0.02$, every published web-scale contamination estimate using this detector class is dominated by FPR bias, and the measurement variant is confirmed methodologically blocked. If $\hat q_{2019} < 0.005$, the estimator is usable and $\hat\alpha$ on the modern snapshot becomes a real number. Secondary: treatment-minus-control validation perplexity, where a gain below $0.5\%$ means the detector is not a useful filter regardless of its AUROC.

## 9. Key References

- **[Foundational]** Sadasivan, Kumar, Balasubramanian, Wang, Feizi. *Can AI-Generated Text be Reliably Detected?* TMLR, 2025 (arXiv 2023). — arXiv:2303.11156
- **[Foundational]** Chakraborty, Bedi, Zhu, An, Manocha, Huang. *On the Possibilities of AI-Generated Text Detection.* ICML, 2024. — arXiv:2304.04736
- **[SOTA]** Hans, Schwarzschild, Cherepanova, Kazemi, Saha, Goldblum, Geiping, Goldstein. *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text.* ICML, 2024. — arXiv:2401.12070
- **[SOTA]** Liang, Zhang, Cao, Wang, Ding, Yang, Vodrahalli, He, Smith, Yin, McFarland, Zou. *Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews.* ICML, 2024. — arXiv:2403.07183
- **[Measurement]** Thompson, Dhaliwal, Frisch, Domhan, Federico. *A Shocking Amount of the Web is Machine Translated: Insights from Multi-Way Parallelism.* Findings of ACL, 2024. — arXiv:2401.05749
- **[Measurement]** Dodge, Sap, Marasović, Agnew, Ilharco, Groeneveld, Mitchell, Gardner. *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus.* EMNLP, 2021.
- **[Attack]** Krishna, Song, Karpinska, Wieting, Iyyer. *Paraphrasing Evades Detectors of AI-Generated Text: A Defense by Retrieval.* NeurIPS, 2023. — arXiv:2303.13408
- **[Bias]** Liang, Yuksekgonul, Mao, Wu, Zou. *GPT Detectors Are Biased Against Non-Native English Writers.* Patterns, 2023. — arXiv:2304.02819
- **[Consequence]** Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. *AI Models Collapse When Trained on Recursively Generated Data.* Nature 631, 2024.
- **[Consequence]** Gerstgrasser et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM, 2024. — arXiv:2404.01413
- **[Watermarking]** Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein. *A Watermark for Large Language Models.* ICML, 2023. — arXiv:2301.10226
- **[Watermarking]** Zhang, Edelman, Francati, Venturi, Ateniese, Barak. *Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models.* ICML, 2024. — arXiv:2311.04378
- **[Corpus]** Penedo, Kydlíček, Ben allal, Lozhkov, Mitchell, Raffel, Von Werra, Wolf. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.17557
- **[Survey]** Dugan, Hwang, Trhlík, Ludan, Zhu, Xu, Ippolito, Callison-Burch. *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors.* ACL, 2024. — arXiv:2405.07940

## 10. Worked Example

Take one Common Crawl snapshot: $N = 3.0\times 10^9$ documents. Run a detector with a *benchmark* operating point of $\mathrm{TPR}=0.94$, $\mathrm{FPR}=0.01$. Suppose the observed positive rate is $\hat q = 0.071$.

Point estimate:
$$\hat\alpha = \frac{0.071 - 0.01}{0.94 - 0.01} = 0.0656 .$$

Sampling standard error: $\sqrt{0.071 \cdot 0.929 / 3\times10^9}\,/\,0.93 = 5.0\times10^{-6}$. So the naive interval is $6.56\% \pm 0.001\%$ — apparently a precise measurement of web contamination.

Now the obstruction. The benchmark $\mathrm{FPR}=0.01$ was measured on curated essays. Run the same detector on a 2019 snapshot, where true $\alpha \approx 0$. Suppose it returns $\hat q_{2019} = 0.043$ — entirely false positives, and consistent in magnitude with the measured non-native-writer bias. Substituting the *in-situ* FPR:
$$\hat\alpha = \frac{0.071 - 0.043}{0.94 - 0.043} = 0.0312 .$$

The estimate halves. The sampling interval never overlapped the corrected value; it was $6{,}000\times$ narrower than the bias. And $\mathrm{FPR}=0.043$ is itself only a bound: web register drifted between 2019 and 2026 for reasons unrelated to LLMs (SEO templating, CMS boilerplate), so the negative control is imperfect in an unsigned direction.

This is the shape of the problem. The quantity is estimable to six decimal places of *precision* and one significant figure of *accuracy*, and closing that gap requires certified-human post-2022 text that does not exist.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*