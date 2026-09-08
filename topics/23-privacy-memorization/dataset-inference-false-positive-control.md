---
id: 23-privacy-memorization/dataset-inference-false-positive-control
title: "Dataset Inference for Ownership Without False Positives"
topic: 23-privacy-memorization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Dataset Inference for Ownership Without False Positives

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/dataset-inference-false-positive-control` · **Status:** partially-solved

## 1. Problem Statement

A data owner holds a corpus $\mathcal{S}$ (a book, a code repository, a scraped news archive). A model $\theta$ is deployed. The owner wants a procedure that outputs **"$\mathcal{S}$ was in the training set of $\theta$"** or **"insufficient evidence"**, with a calibrated bound on the probability of the first answer when $\mathcal{S}$ was *not* used.

The decision predicate is a hypothesis test, not a score. $H_0$: $\mathcal{S} \cap \mathcal{D}_{\text{train}}(\theta) = \emptyset$. The output must be a $p$-value valid under $H_0$, so that a legal or audit threshold $\alpha$ (say $0.01$) means what it says.

Three variants, of very different difficulty:

- **Measurement.** Does any statistic of $(\theta, \mathcal{S})$ separate members from non-members at the *set* level? Partly yes.
- **Method.** Can the null be constructed without a trusted non-member holdout that is distributionally identical to $\mathcal{S}$? Largely unsolved.
- **Theory.** Under what conditions on the training distribution and model class is set membership *identifiable* at all — as opposed to explained by the model having seen near-duplicates, later paraphrases, or the same underlying distribution? Open.

Solving it means: a test that, run on thousands of genuinely-unused suspect sets drawn adversarially, rejects at rate $\le \alpha$, and still rejects for genuinely-used sets of realistic size (a single 300-page book, not 10,000 documents).

## 2. Formal Setting

Let $\theta$ be trained on $\mathcal{D}_{\text{train}} \sim \mathcal{P}$. The claimant supplies $\mathcal{S} = \{x_1,\dots,x_n\}$ and a **validation set** $\mathcal{V} = \{x'_1,\dots,x'_m\}$ asserted to be exchangeable with $\mathcal{S}$ and disjoint from $\mathcal{D}_{\text{train}}$.

A per-example feature map $\phi: \mathcal{X} \to \mathbb{R}^k$ collects membership signals as actually computed:

- Log-loss: $\phi_1(x) = -\frac{1}{|x|}\sum_t \log p_\theta(x_t \mid x_{<t})$, measured by one forward pass at temperature 1.
- Min-$K$% ($K=20$): mean of the $K$% lowest token log-probabilities (Shi et al., ICLR 2024).
- zlib ratio: $\phi_3(x) = \phi_1(x)\cdot|x| / \text{zlib}(x)$, the compressed byte length, to normalize for intrinsic text complexity.
- Reference-model ratio: $\phi_1(x) - \phi_1^{\text{ref}}(x)$ with a second model $\theta_{\text{ref}}$ not trained on $\mathcal{S}$.
- Perturbation curvature (Neighbourhood/DetectGPT-style): $\phi_1(x) - \mathbb{E}_{\tilde x \sim q(\cdot|x)}\phi_1(\tilde x)$, estimated with $\sim 25$ mask-fill neighbours.

Dataset inference aggregates: fit $w \in \mathbb{R}^k$ on a *labelled* split (half of $\mathcal{S}$ against half of $\mathcal{V}$) by linear regression, then test on the held-out halves

$$T = \frac{\overline{w^\top\phi(\mathcal{S}_{\text{test}})} - \overline{w^\top\phi(\mathcal{V}_{\text{test}})}}{\sqrt{s_{\mathcal{S}}^2/n' + s_{\mathcal{V}}^2/m'}},\qquad p = \Pr[T_{\nu} > T].$$

Assumptions, and their status in practice:

1. **Exchangeability of $\mathcal{S}$ and $\mathcal{V}$ under $H_0$.** Required for the $t$-test to be valid. *Violated routinely*: benchmark holdouts are usually later in time than members (WikiMIA, arXiv-by-date splits), so topic and vocabulary shift alone drives $T$.
2. **$\mathcal{V} \cap \mathcal{D}_{\text{train}} = \emptyset$.** Unverifiable for a model whose training corpus is undisclosed; a "clean" holdout may itself be memorized, which *deflates* power rather than inflating error.
3. **i.i.d. examples within $\mathcal{S}$.** Violated: documents from one source share style, so $s_{\mathcal{S}}^2$ underestimates the variance of the set-level statistic and $\nu$ overstates degrees of freedom.
4. **Single-epoch, no deduplication effects.** Violated: near-duplicate removal and multi-epoch upsampling both change the member/non-member gap by more than most reported effect sizes.

## 3. State of the Art

**Established.**
- *Dataset Inference* (Maini, Yaghini, Papernot, ICLR 2021) introduced the set-level framing for image classifiers with a distance-to-decision-boundary feature, and showed detection with $\sim$50 private samples on CIFAR-10 while single-example MIA was near chance.
- *LLM Dataset Inference* (Maini, Jia, Papernot, Dziedzic, NeurIPS 2024) is the current reference method for LLMs: aggregate $\sim$50 MIA features by regression, then a $t$-test. On Pythia (410M–12B) with Pile train-vs-val splits it reports $p < 0.1$ using about 1,000 suspect examples, and — importantly — reports $p$ near uniform on *val-vs-val* pairs, i.e. no false positive on their negative control.
- *Data watermarks* (Wei, Wang, Jia, ACL Findings 2024) gives a genuinely valid null: inject random-hash or unicode watermarks into the owner's text before release, then test for elevated likelihood of the planted token sequence. The $p$-value is exact because the null is a known random construction, not an assumed exchangeable holdout.

**Claimed but unablated.**
- Every post-hoc MIA feature ported into set-level tests inherits an untested distributional assumption. Reported per-dataset $p$-values are benchmark numbers on Pile splits and near-duplicate-controlled versions of them; they are *not* evidence of FPR control against an adversarially chosen $\mathcal{V}$.
- Copyright traps for natural documents (Meeus et al., ICML 2024) show detectability of injected sequences repeated many times, but the required repetition count for short documents is a benchmark result on one 1.3B-parameter training run, not a general operating curve.

**Negative results that reset the SOTA.**
- *Blind baselines beat MIAs* (Das, Zhang, Tramèr, 2024): on WikiMIA and similar splits, a bag-of-words classifier with **no model access** reaches AUC comparable to or above published MIAs — the benchmarks measure temporal shift, not membership.
- *MIAs cannot prove a model was trained on your data* (Zhang, Das, Kamath, Tramèr, SaTML 2025): existing attacks flag data that is provably absent from training; without a valid null they cannot support an ownership claim.

## 4. What Is Known

- **Per-example MIA on pretrained LLMs is near chance.** Duan et al. (COLM 2024, MIMIR) measure AUC $\approx 0.5$–$0.6$ across Pythia 160M–12B on Pile domains, with the highest values driven by $n$-gram overlap between the "non-member" split and training data; after removing high-overlap examples, AUC falls back toward 0.5.
- **Set aggregation buys real power.** Averaging over $n$ examples shrinks the null standard error as $1/\sqrt{n}$, so an effect size of $d = 0.05$ per example — invisible at AUC 0.51 — becomes $p<0.01$ at $n\approx 2{,}000$ *if* the null is correct.
- **Duplication drives the signal.** Sequences duplicated $\ge 10$ times in the Pile are extractable/detectable at rates orders of magnitude above singletons (Carlini et al., ICLR 2023 quantification of memorization; Nasr et al. 2023 extraction). A single-copy 300-page book is the hard case, not the typical benchmark case.
- **Watermark tests are calibrated at realistic scale.** Wei et al. report detection with $p < 0.01$ for a random-sequence watermark repeated on the order of tens of times in a corpus of $\sim$10B tokens (BERT/LLM training runs), with an exact null.
- **The main published FPR control is against a friendly null.** In LLM Dataset Inference the negative control is a second draw from the *same* Pile validation pool — the most favourable possible exchangeability condition.

## 5. What Is Not Known

- **Methodologically blocked.** How to construct $\mathcal{V}$ for a real claim. A publisher suing over a 2019 novel has no distributionally identical, provably-unused sibling text. Without it, the $p$-value has no operational meaning. This is the blocking gap, not merely a hard one.
- **Theoretically open.** Whether set membership is identifiable when $\mathcal{S}$ is near-duplicated in the wild (fan quotes, review sites, translations). No separation theorem distinguishes "trained on $\mathcal{S}$" from "trained on 200 partial copies of $\mathcal{S}$", and both are legally distinct. No lower bound exists on the sample complexity of set-level DI as a function of duplication count $c$ and per-example effect size $d$.
- **Empirically open.** Nobody has run a large negative-control sweep: thousands of suspect sets that are *certainly* absent from a fully documented training run, scored with the published DI pipeline, to estimate the empirical FPR at $\alpha=0.01$. The training runs to do this exist (Pythia, OLMo, and their public corpora); the experiment has not been run at that scale.
- **Unknown for fine-tuning and RLHF.** All DI results assume a single pretraining pass; the effect of post-training on member/non-member gaps is unmeasured.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus an absent null**. Every feature in $\phi$ is a proxy for "$\theta$ assigns this text unusually high likelihood", and the dominant cause of unusual likelihood is not membership but distributional proximity: topic, register, publication date, and near-duplication. The test statistic $T$ is therefore a shift detector wearing a membership label — which is exactly what Das et al. showed by matching MIA AUC with a model-free classifier.

The secondary obstruction is **non-identifiability under duplication**. If $\mathcal{S}$ appears in paraphrase across the open web, then $H_0$ ("$\mathcal{S}$ was not used") and $H_1$ induce nearly the same likelihood profile; no amount of data at test time separates them, because the difference lives in the training corpus, which the auditor cannot see.

Compute is *not* the bottleneck: a DI run on a 7B model with 2,000 examples and 25 perturbation neighbours each is $\sim$50k forward passes, under an hour on one A100.

## 7. Current Research (as of 2026)

- **Proactive marking over post-hoc detection.** Watermarks and copyright traps (Wei/Jia at USC; de Montjoye's group at Imperial) trade coverage for an exact null. The open sub-problem is robustness: whether traps survive deduplication, canonicalization, and paraphrase-based data cleaning *(frontier — verify)*.
- **Benchmark reform.** The SaTML 2025 SoK (Meeus, Shilov, Jain, Faysse, Rei, de Montjoye) argues most LLM MIA evaluations are invalid by construction and proposes randomized-split designs where member/non-member assignment is made by coin flip *before* training. The CMU/Toronto/Vector line (Maini, Dziedzic, Papernot) continues on aggregation and calibration.
- **Test-side rigor.** Use of conformal $p$-values and e-values for anytime-valid auditing is being explored to replace the $t$-test's parametric null *(frontier — verify)*.
- **Randomized-controlled pretraining runs.** Small open training runs with pre-registered inclusion/exclusion of candidate corpora are the emerging gold standard; OLMo-class open-data models make this feasible.

## 8. Concrete Next Experiment

**The negative-control FPR audit.**

- **Scale.** Pythia-2.8B and OLMo-7B, both with fully public training corpora. Build $10{,}000$ suspect sets of $n=1{,}000$ documents each, drawn from sources verifiably *absent* from the corpus: post-cutoff news, post-cutoff arXiv, private/licensed text, and — critically — sources chosen to be *stylistically distant* from the owner's $\mathcal{V}$ (the adversarial arm, matching what a real claimant would supply).
- **Control arm.** For each suspect set, a matched $\mathcal{V}$ constructed the way a real claimant would: same publisher, adjacent time window, no guarantee of exchangeability. Second control: $\mathcal{V}$ drawn by random split of the *same* documents (the friendly null used in published work).
- **Procedure.** Run the LLM Dataset Inference pipeline unmodified; record the $p$-value for each set.
- **Deciding number.** The empirical rejection rate at $\alpha = 0.01$ under the realistic-$\mathcal{V}$ arm. If it is $\le 0.02$, DI is usable as evidence and the field should say so. If it exceeds $0.10$ — a 10x inflation — post-hoc DI is not an ownership test at any threshold, and only proactive marking remains. Report the same rate for the friendly null to quantify how much of the published FPR control is an artifact of the split.

Cost estimate: $10^7$ document scorings across two models, roughly 3–5 GPU-weeks on A100s. This is cheap relative to the claims resting on it.

## 9. Key References

- **[Foundational]** Pratyush Maini, Mohammad Yaghini, Nicolas Papernot. *Dataset Inference: Ownership Resolution in Machine Learning.* ICLR 2021. — arXiv:2104.10706
- **[SOTA]** Pratyush Maini, Hengrui Jia, Nicolas Papernot, Adam Dziedzic. *LLM Dataset Inference: Did you train on my dataset?* NeurIPS 2024. — arXiv:2406.06443
- **[Negative result]** Michael Zhang, Debeshee Das, Gautam Kamath, Florian Tramèr. *Membership Inference Attacks Cannot Prove that a Model Was Trained On Your Data.* IEEE SaTML 2025. — arXiv:2409.19798
- **[Negative result]** Debeshee Das, Jie Zhang, Florian Tramèr. *Blind Baselines Beat Membership Inference Attacks for Foundation Models.* 2024. — arXiv:2406.16201
- **[Empirical]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM 2024. — arXiv:2402.07841
- **[Method]** Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer. *Detecting Pretraining Data from Large Language Models.* ICLR 2024. — arXiv:2310.16789
- **[Method]** Johnny Tian-Zheng Wei, Ryan Yixiang Wang, Robin Jia. *Proving Membership in LLM Pretraining Data via Data Watermarks.* Findings of ACL 2024. — arXiv:2402.10892
- **[Method]** Matthieu Meeus, Igor Shilov, Manuel Faysse, Yves-Alexandre de Montjoye. *Copyright Traps for Large Language Models.* ICML 2024. — arXiv:2402.09363
- **[Foundational]** Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Hervé Jégou. *Radioactive Data: Tracing Through Training.* ICML 2020. — arXiv:2002.00937
- **[Foundational]** Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P 2022. — arXiv:2112.03570
- **[Survey]** Matthieu Meeus, Igor Shilov, Shubham Jain, Manuel Faysse, Marek Rei, Yves-Alexandre de Montjoye. *SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It).* IEEE SaTML 2025. — arXiv:2406.17975

## 10. Worked Example

A publisher claims a 2018 novel, 120k tokens $\approx$ 400 chunks of 300 tokens, was used to train a 7B model. They supply $\mathcal{V}$: 400 chunks from a *different* novel by the same author, published 2021 (after the model's data cutoff).

Suppose the true per-chunk membership effect is $d = 0.08$ standard deviations — plausible for single-copy text. With $n' = m' = 200$ after the fitting split:

$$\mathbb{E}[T] = d\sqrt{\frac{n'}{2}} = 0.08 \times 10 = 0.8 \quad\Rightarrow\quad p \approx 0.21.$$

Not significant. Now the confound. The 2021 novel is 3 years later, longer-sentenced, and shares less vocabulary with the pretraining corpus. Measure the shift directly: fit a bag-of-words logistic classifier to separate the two novels *without the model*. If it reaches AUC $0.78$ — routine for two books by the same author — that separation corresponds to a distributional gap of roughly $d_{\text{shift}} = \sqrt{2}\,\Phi^{-1}(0.78) \approx 1.09$ standard deviations in the discriminating direction.

The regression step in DI fits $w$ to maximize exactly this separation. Even if only $\sim$10% of $d_{\text{shift}}$ leaks into the likelihood features, the induced bias is $0.11$ — larger than the true effect $0.08$. Then

$$\mathbb{E}[T] = (0.08 + 0.11)\times 10 = 1.9 \quad\Rightarrow\quad p \approx 0.03.$$

The test now "rejects" at $\alpha=0.05$, and would reject with the same magnitude if the novel had never been trained on at all — because the bias term does not depend on membership.

**The obstruction, made visible:** the significant $p$-value is 58% attributable to the fact that the two books were written three years apart. No amount of additional suspect data fixes this; increasing $n$ scales bias and signal together, so $p \to 0$ under the null. The failure mode is inconsistency of the test, not low power.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*