---
id: 23-privacy-memorization/membership-inference-distribution-shift
title: "Membership Inference Under Distribution Shift"
topic: 23-privacy-memorization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Membership Inference Under Distribution Shift

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/membership-inference-distribution-shift` · **Status:** empirically-open

## 1. Problem Statement

A membership inference attack (MIA) takes a trained model $\theta$ and a candidate record $x$ and outputs a score for the decision "was $x$ in the training set $\mathcal{S}$?". The claim such an attack is meant to support is about *memorization*: that $\theta$ leaks something about $x$ beyond what the world already tells you.

Almost every reported MIA result on foundation models violates the premise. The members and non-members are not drawn from one distribution and split at random; they are separated by a proxy — documents before vs. after a cutoff date, books in one catalog vs. another, Wikipedia edits by year. So an attack can score well by detecting the proxy, not the model. **Blind baselines** — classifiers with no access to $\theta$ at all — reach or beat published MIA numbers on several standard benchmarks (Das, Zhang & Tramèr, 2024).

Three variants, with different difficulty:

- **Measurement.** Define an evaluation whose reported number is provably attributable to $\theta$ and not to $q_{\text{mem}} \neq q_{\text{non}}$. Currently the weakest link.
- **Method.** Build an attack that retains power when members and non-members are exchangeable (a randomized split) at pretraining scale. Empirically open: no attack clearly beats chance there.
- **Theory.** Characterize when the shift term and the memorization term in the Bayes-optimal test are separately identifiable from the observables an auditor actually has.

Solving it means: a protocol plus an attack such that, on a randomized member/non-member split of a real pretraining corpus, AUC is bounded away from $0.5$ by more than the blind-baseline ceiling, with a reported confidence interval.

## 2. Formal Setting

Let $\mathcal{X}$ be the record space, $\mathcal{S} \subset \mathcal{X}$ the training set of size $n$, $\theta = A(\mathcal{S})$ the model from training algorithm $A$. The auditor holds a candidate set $C = C_{\text{mem}} \cup C_{\text{non}}$ with gold labels, drawn as $C_{\text{mem}} \sim q_{\text{mem}}$, $C_{\text{non}} \sim q_{\text{non}}$.

**Attack.** A score $s: \mathcal{X} \times \Theta \to \mathbb{R}$. Measured performance is empirical ROC over $C$:
$$\widehat{\mathrm{AUC}}(s) = \frac{1}{|C_{\text{mem}}||C_{\text{non}}|}\sum_{x\in C_{\text{mem}}}\sum_{x'\in C_{\text{non}}} \mathbb{1}[s(x,\theta) > s(x',\theta)],$$
and, more usefully, $\mathrm{TPR}@\mathrm{FPR}=\alpha$ for $\alpha \in \{10^{-3}, 10^{-2}\}$ (Carlini et al., 2022). Both are computed on finite $C$; a $\mathrm{TPR}$ at $\alpha=10^{-3}$ needs $|C_{\text{non}}| \gg 10^3$ to be non-degenerate.

**The decomposition.** The Bayes-optimal membership log-odds for candidate $x$ splits:
$$\log\frac{\Pr[x \in \mathcal{S}\mid x,\theta]}{\Pr[x \notin \mathcal{S}\mid x,\theta]} \;=\; \underbrace{\log\frac{q_{\text{mem}}(x)}{q_{\text{non}}(x)}}_{\Delta_{\text{shift}}(x)\ \text{— no }\theta} \;+\; \underbrace{\log\frac{p(\theta\mid x\in\mathcal{S})}{p(\theta\mid x\notin\mathcal{S})}}_{\Delta_{\text{mem}}(x,\theta)}.$$
Only $\Delta_{\text{mem}}$ is a privacy leak. Under a **randomized split** ($q_{\text{mem}}=q_{\text{non}}$), $\Delta_{\text{shift}}\equiv 0$ and $\widehat{\mathrm{AUC}}$ is an unbiased estimate of memorization signal. Under a temporal or catalog split it is not.

**Blind ceiling.** Let $\mathcal{B}$ be the class of scores using $x$ only (bag-of-words, date regex, length, perplexity of an *independent* reference model). Define
$$\mathrm{Adv}_{\text{free}}(s) \;=\; \widehat{\mathrm{AUC}}(s) \;-\; \max_{b\in\mathcal{B}} \widehat{\mathrm{AUC}}(b).$$
This is the quantity a paper should report. It is a *lower bound* on true shift-free advantage only if $\mathcal{B}$ is rich enough — an assumption nobody verifies.

**Stratified alternative.** With observed nuisance covariate $T(x)$ (date bucket, domain, token count), report $\mathbb{E}_{T}[\widehat{\mathrm{AUC}}(s \mid T)]$. Valid only under **no unmeasured confounding**: $q_{\text{mem}}(x\mid T)=q_{\text{non}}(x\mid T)$.

**Assumptions known to be violated in practice.**
1. *Exchangeability of $C_{\text{mem}}, C_{\text{non}}$* — violated by construction in WikiMIA, BookMIA, and every cutoff-date benchmark.
2. *No unmeasured confounding given $T$* — violated: topic drift co-moves with date; nothing observed captures it.
3. *Disjointness of member and non-member text* — violated by near-duplicates; $n$-gram overlap between "non-members" and the corpus is often substantial.
4. *IID single-epoch training* — violated by data curricula, upsampling of high-quality sources, and deduplication.
5. *Known $n$ and one target model* — auditors of a released checkpoint have neither shadow models nor the corpus.

## 3. State of the Art

**Established (ablated, reproduced).**
- **LiRA** (Carlini, Chien, Nasr, Song, Terzis & Tramèr, S&P 2022): per-example Gaussian likelihood ratio over shadow models. On CIFAR-10 (WRN28-2, $n=25{,}000$), TPR $\approx 8.4\%$ at FPR $10^{-3}$, versus $\approx 0\%$ for loss-thresholding (Yeom et al., CSF 2018). This is a randomized-split setting — the number is a real memorization measurement.
- **RMIA** (Zarifzadeh, Liu & Shokri, ICML 2024) reaches comparable power with 1–2 shadow models rather than 64+. Difficulty calibration (Watson et al., ICLR 2022) and attack-$\mathrm{R}$ (Ye et al., CCS 2022) are the same idea from different directions: subtract a per-example baseline.
- **Failure at LLM pretraining scale** (Duan et al., COLM 2024): across Pythia 160M–12B on the Pile with a proper IID member/non-member split, MIA AUC sits near $0.5$ (roughly $0.51$–$0.55$) for loss, zlib, Min-K%, and reference-model attacks. Reproduced by several groups.
- **Blind baselines** (Das, Zhang & Tramèr, 2024): on WikiMIA, BookMIA and several LLM MIA benchmarks, a classifier with no model access matches or exceeds the published attack AUC. This is an ablation of the benchmark, and it holds.
- **SoK on randomized vs. non-randomized splits** (Meeus, Shilov, Jain, Faysse, Rei & de Montjoye, SaTML 2025): the same attacks that look strong on temporally split data collapse toward $0.5$ when the split is randomized over the same corpus.

**Claimed but unablated / benchmark-only.**
- **Min-K% Prob** (Shi et al., ICLR 2024) reports AUC $\approx 0.74$ on WikiMIA-128 with LLaMA-65B. That number is a benchmark artifact of a temporal split; the shift-free component is not isolated.
- **Min-K%++**, document-level aggregation (Meeus et al., USENIX Sec 2024) and various embedding-based detectors: all report gains on shift-contaminated benchmarks. Gains under randomized splits are largely unreported.
- **Scale-dependent success** (Puerto et al., NAACL 2025): MIA becomes measurable when the unit is a large collection and $n$-gram overlap is controlled. Directionally credible; the overlap control is not yet standard.

**Sidesteps that work.** Dataset inference (Maini, Jia, Papernot & Dziedzic, NeurIPS 2024) aggregates weak per-record signal over a *set* and calibrates with a held-out set from the same distribution — this makes the shift term cancel by construction, at the cost of no longer being a per-record claim. Canary-based auditing (Steinke, Nasr & Jagielski, NeurIPS 2023) removes the shift entirely by inserting known members.

## 4. What Is Known

- **Vision, randomized split, $n=25$k CIFAR-10:** LiRA TPR $8.4\%$ @ FPR $0.1\%$; loss-threshold $\approx 0\%$. Memorization signal is real and large in the low-FPR regime.
- **LLM pretraining, randomized split, Pythia 160M–12B, Pile:** AUC $\approx 0.5$ for all standard attacks; no monotone improvement with model size at fixed data (Duan et al., 2024). Scale: $\sim$300B tokens, one epoch.
- **LLM, temporal split (WikiMIA, 32–256 token chunks, $\sim$few thousand examples):** reported AUC $0.6$–$0.8$; blind baselines reach comparable values. The gap $\mathrm{Adv}_{\text{free}}$ is small and often within noise.
- **Duplication drives everything measurable.** Carlini et al. (ICLR 2023) show extraction rate grows log-linearly in model size, sequence duplication count, and prompt context length; a sequence seen once at 300B-token scale is near-unextractable. This bounds what per-record MIA can hope for.
- **Overfitting is sufficient but not necessary** (Yeom et al., CSF 2018): the generalization gap upper-bounds average-case advantage; single-epoch LLMs have near-zero train/test loss gap on held-in-distribution data, which is consistent with the observed $\mathrm{AUC}\approx0.5$.
- **A negative result on the legal use case:** Zhang, Das, Kamath & Tramèr (SaTML 2025) argue that MIA outputs cannot establish that a specific document was trained on, because the false-positive rate is not controllable without a validated non-member distribution — exactly what shift destroys.

## 5. What Is Not Known

- **Empirically open (primary).** Is there *any* attack with $\mathrm{Adv}_{\text{free}} > 0$, at statistical significance, on a randomized split of a $\ge 10^{11}$-token pretraining corpus for a $\ge 7$B model? The experiment is runnable — it needs a from-scratch pretraining run with a held-out shard drawn identically — and has not been run with a full modern attack suite and low-FPR reporting.
- **Empirically open.** Where is the per-record duplication threshold $k^\*$ at which membership becomes detectable at FPR $10^{-3}$, as a function of model size and token budget? Extraction curves exist; MIA curves at fixed low FPR do not.
- **Theoretically open.** Under what conditions are $\Delta_{\text{shift}}$ and $\Delta_{\text{mem}}$ identifiable from $(C, \theta)$ alone? Intuition says never without an anchor (a randomized subset, an inserted canary, or a second model trained without $x$), but there is no impossibility theorem stated at this generality.
- **Methodologically blocked.** There is no accepted definition of "sufficiently rich blind baseline class $\mathcal{B}$". $\mathrm{Adv}_{\text{free}}$ is only as trustworthy as $\mathcal{B}$, and every paper picks its own. Similarly, there is no standard for how much $n$-gram overlap between a "non-member" and the corpus is disqualifying.

## 6. Why It Is Hard

**The evaluation does not measure the thing it names — and the confound is non-identifiable from the auditor's data.** In the decomposition of §2, an auditor observing only $(x, \theta, \text{label})$ sees the sum $\Delta_{\text{shift}}+\Delta_{\text{mem}}$. Any procedure that separates them needs a sample from $q_{\text{mem}}$ with the membership label broken — that is, either a randomized split (requires controlling training, which excludes all released checkpoints) or a canary (requires inserting data before training). Neither is available to the third party who most wants the answer.

Two secondary obstructions compound it:

- **Compute.** The clean experiment is a pretraining run. At 7B/300B tokens that is $\sim$$10^{22}$ FLOPs plus a shadow arm; without a shadow arm you cannot run LiRA, which is the only attack with demonstrated low-FPR power.
- **Signal floor.** At one epoch over $10^{11}$ tokens, per-record influence on $\theta$ is minute. The effect being measured may genuinely be near zero, in which case the field is chasing benchmark artifacts. Distinguishing "no attack found it" from "there is nothing there" requires the negative result to be reported with power analysis, which is rare.

## 7. Current Research (as of 2026)

- **Benchmark repair.** The SaTML/S&P community (de Montjoye's group at Imperial; Tramèr's group at ETH Zürich) is pushing randomized-split benchmarks and mandatory blind baselines. Expect deprecation of WikiMIA-style temporal splits.
- **Set-level inference.** Dataset inference (CMU/Vector: Papernot, Dziedzic, Maini) and user inference (Kandpal, Pillutla, Oprea, Kairouz, Choquette-Choo & Xu, EMNLP 2024) trade per-record resolution for a valid null hypothesis. This is the direction with working statistics.
- **Auditing as the substitute.** One-run auditing (Steinke, Nasr, Jagielski) and tight DP auditing (Nasr et al., USENIX Sec 2023) give calibrated $\varepsilon$ lower bounds without needing a non-member distribution — for parties who control training.
- **Shift-aware calibration.** Quantile-regression attacks (Bertran et al., NeurIPS 2023) and importance-weighting the reference model to $q_{\text{non}}$ *(frontier — verify)*; the open question is whether reweighting can be validated without the very labels it is trying to earn.
- **Copyright litigation pressure** is generating claimed detectors faster than ablations *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does any attack have shift-free membership advantage at pretraining scale?

**Scale.** Pretrain two 1.4B-parameter decoder models on 100B tokens of a deduplicated corpus. Partition the corpus into shards; assign shards to model A and model B by a **random** draw, so every candidate document is a member for exactly one model and a non-member for the other, and $q_{\text{mem}}=q_{\text{non}}$ exactly. Candidate set: $|C|=200{,}000$ documents, stratified into duplication-count buckets $k \in \{1,2,4,8,16,64\}$ measured by exact-match on 50-gram shingles. Cost: $\sim$2 × $10^{21}$ FLOPs, order 4k A100-hours. Report at FPR $10^{-3}$, which $|C_{\text{non}}|=10^5$ can resolve.

**Attack arm.** Min-K%++, zlib ratio, reference-model ratio, and cross-model LiRA (each model is the other's single shadow — the symmetric design gives this for free).

**Control arm.** (a) Blind baselines: logistic regression on TF-IDF, document length, and source domain, trained on half of $C$, evaluated on the other half. (b) A **label-permutation null**: rerun the full pipeline with membership labels shuffled, to fix the finite-sample distribution of $\widehat{\mathrm{TPR}}@10^{-3}$.

**Deciding number.** $\mathrm{TPR}@\mathrm{FPR}=10^{-3}$ in the $k=1$ bucket, minus the blind-baseline TPR at the same FPR, with a 95% bootstrap CI over documents. If the lower bound exceeds $0.1\%$ (the chance rate), single-occurrence per-record membership is real at this scale. If the CI contains zero at $k=1$ but the lower bound clears $0.1\%$ at some $k^\*\le 64$, the field's honest claim becomes "membership inference is duplication detection above $k^\*$", and $k^\*$ is the number to publish.

## 9. Key References

- **[Foundational]** R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P, 2017. — arXiv:1610.05820
- **[Foundational]** S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha. *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE CSF, 2018. — arXiv:1709.01604
- **[SOTA]** N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022. — arXiv:2112.03570
- **[SOTA]** S. Zarifzadeh, P. Liu, R. Shokri. *Low-Cost High-Power Membership Inference Attacks.* ICML, 2024. — arXiv:2312.03262
- **[Key negative]** M. Duan, A. Suri, N. Mireshghallah, S. Min, W. Shi, L. Zettlemoyer, Y. Tsvetkov, Y. Choi, D. Evans, H. Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Key negative]** D. Das, J. Zhang, F. Tramèr. *Blind Baselines Beat Membership Inference Attacks for Foundation Models.* 2024. — arXiv:2406.16201
- **[Key negative]** J. Zhang, D. Das, G. Kamath, F. Tramèr. *Membership Inference Attacks Cannot Prove that a Model Was Trained On Your Data.* IEEE SaTML, 2025. — arXiv:2409.19798
- **[Survey / SoK]** M. Meeus, I. Shilov, S. Jain, M. Faysse, M. Rei, Y.-A. de Montjoye. *SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It).* IEEE SaTML, 2025. — arXiv:2406.17975
- **[Benchmark]** W. Shi, A. Ajith, M. Xia, Y. Huang, D. Liu, T. Blevins, D. Chen, L. Zettlemoyer. *Detecting Pretraining Data from Large Language Models.* ICLR, 2024. — arXiv:2310.16789
- **[Alternative framing]** P. Maini, H. Jia, N. Papernot, A. Dziedzic. *LLM Dataset Inference: Did you train on my dataset?* NeurIPS, 2024. — arXiv:2406.06443
- **[Auditing]** T. Steinke, M. Nasr, M. Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023. — arXiv:2305.08846
- **[Memorization scaling]** N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramèr, C. Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Calibration]** L. Watson, C. Guo, G. Cormode, A. Sablayrolles. *On the Importance of Difficulty Calibration in Membership Inference Attacks.* ICLR, 2022. — arXiv:2111.08440

## 10. Worked Example

Take WikiMIA-style construction concretely. Members: Wikipedia event pages created before 2017. Non-members: pages created after 2023. Suppose an attack reports $\mathrm{AUC}=0.74$.

Now compute the blind term. Post-2023 event pages contain tokens that are near-absent from pre-2017 text: "COVID-19", "ChatGPT", "SARS-CoV-2", "Wagner Group", and 2023–2024 date strings. Fit a logistic regression on TF-IDF unigrams over half the candidates, no model access. In this construction such a classifier separates the two halves at AUC in the high $0.8$s — Das, Zhang & Tramèr report blind baselines matching or exceeding the model-based attacks on this benchmark.

So:
$$\mathrm{Adv}_{\text{free}} = 0.74 - \max_{b\in\mathcal{B}} \widehat{\mathrm{AUC}}(b) < 0.$$

The attack is *worse* than not looking at the model. The reported $0.74$ is not evidence of memorization; it is a slightly lossy re-derivation of the publication date.

Try to rescue it by stratifying on the observed covariate $T$ = creation year. Within the 2016 bucket alone all candidates are members — the stratum is degenerate and contributes nothing to $\widehat{\mathrm{AUC}}(s\mid T)$. Every non-degenerate stratum must straddle the cutoff, and at the cutoff the topic distribution is still shifting. There is no cell in which $q_{\text{mem}}(x\mid T) = q_{\text{non}}(x\mid T)$ holds. The estimator has no support.

**That is the obstruction, visible in one benchmark.** The confound is not noise to be averaged out; it is perfectly collinear with the label the benchmark was built from. No amount of post-hoc adjustment recovers $\Delta_{\text{mem}}$, because the design never contained a comparison in which $\Delta_{\text{shift}}=0$. Only a randomized split or an inserted canary supplies one — which is why §8 spends 4k GPU-hours to build the split rather than trying to correct a broken one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*