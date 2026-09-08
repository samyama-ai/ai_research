---
id: 27-multilingual/cross-lingual-membership-inference
title: "Cross-Lingual Membership Inference for Language Attribution"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Lingual Membership Inference for Language Attribution

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/cross-lingual-membership-inference` · **Status:** open

## 1. Problem Statement

Given black-box or logit access to a multilingual model $\theta$ and a document that exists in several languages (an original plus translations), decide **which language edition was in the training corpus** — not merely whether the content was seen.

Three variants, of very different difficulty:

- **Measurement variant.** Define a membership score that is comparable across languages. A raw per-token loss is not: tokenizers segment languages at different rates, and base rates of loss differ by an order of magnitude between English and a low-resource language. Without a cross-lingually calibrated score there is no attribution decision to make.
- **Method variant.** Build an attacker that, given the family $\{x^{(\ell)}\}_{\ell \in L}$ of translation-equivalent documents, outputs the subset $S \subseteq L$ of editions present in $D_{\text{train}}$, at AUC materially above a blind baseline that uses no model access.
- **Theory variant.** Is language attribution *identifiable* at all? If a model trained only on English $x^{(\text{en})}$ generalizes cross-lingually well enough to lower loss on the Telugu $x^{(\text{te})}$ by as much as actual Telugu membership would, then no attack, however good, can separate the two hypotheses. This is a non-identifiability claim, and it is unproven in either direction.

Solving it means: a score $s$ and a decision rule with calibrated per-language FPR, validated against a model whose training corpus is *known* at the document-and-language level, beating a translation-free blind baseline by a stated margin.

## 2. Formal Setting

Let $D = \bigcup_{\ell \in L} D_\ell$ be a pretraining corpus partitioned by language. A **document family** is $X = \{x^{(\ell)}\}_{\ell \in L_X}$, all editions semantically equivalent (an original and its human translations). Ground truth is the membership vector $m \in \{0,1\}^{L_X}$, $m_\ell = \mathbb{1}[x^{(\ell)} \in D_\ell]$.

Per-token log-likelihood under $\theta$, with $x^{(\ell)}$ tokenized to $t^{(\ell)}_{1:n_\ell}$:

$$\mathcal{L}_\theta(x^{(\ell)}) = -\frac{1}{n_\ell}\sum_{i=1}^{n_\ell} \log p_\theta\!\left(t^{(\ell)}_i \mid t^{(\ell)}_{<i}\right).$$

**Fertility**, measured as $\phi_\ell = n_\ell / b_\ell$ with $b_\ell$ the UTF-8 byte length, is the quantity that breaks comparability: $\mathcal{L}$ is per token, $\phi_\ell$ varies by $3$–$5\times$ across $L$ for a single tokenizer, and the two are coupled. The byte-normalized score $\tilde{\mathcal{L}} = \phi_\ell \mathcal{L}_\theta(x^{(\ell)})$ (bits per byte) is the minimum fix; it is necessary and not sufficient.

Calibration against a reference model $\theta_{\text{ref}}$ (LiRA-style, Carlini et al. 2022) gives

$$s_{\text{ref}}(x^{(\ell)}) = \tilde{\mathcal{L}}_{\theta_{\text{ref}}}(x^{(\ell)}) - \tilde{\mathcal{L}}_{\theta}(x^{(\ell)}),$$

and the Min-K% family (Shi et al., ICLR 2024) replaces the mean with the mean over the $k\%$ lowest-probability tokens. The **attribution statistic** is the cross-edition contrast

$$\Delta_{\ell \to \ell'} = s(x^{(\ell)}) - s(x^{(\ell')}),$$

with the decision rule $\hat{m}_\ell = \mathbb{1}[\Delta_{\ell \to \ell'} > \tau_{\ell,\ell'}]$ and $\tau$ set on a per-language-pair null of known non-members.

Assumptions, and their status in practice:

1. *Held-out non-members are exchangeable with members.* **Violated.** Non-members are usually drawn from a later time window, so temporal shift, not membership, drives the score (Das, Zhang & Tramèr 2024; Duan et al. 2024).
2. *Translations are semantically equivalent and stylistically neutral.* **Violated.** Machine-translated probes carry translationese; human editions differ in length by $\pm 30\%$.
3. *One edition per family in $D$.* **Violated.** Web crawls contain both the original and machine translations of it; MADLAD-400 and CulturaX audits show large machine-translated fractions in mid- and low-resource languages.
4. *The reference model is membership-independent.* **Violated at scale** — any public reference model trained on Common Crawl has likely seen the same editions.
5. *Cross-lingual transfer does not lower loss on unseen editions.* **The crux, and false in general** — see §5.

## 3. State of the Art

**Established.** For pretraining-scale LLMs, monolingual (English) MIA is near-chance once the member/non-member split is controlled. Duan et al. (COLM 2024, MIMIR) report AUC $\approx 0.5$–$0.6$ across Pythia 160M–12B on Pile subsets with matched distributions; Das, Zhang & Tramèr (2024) show that blind baselines using date heuristics or bag-of-words classifiers — no model access at all — match or beat published MIA numbers on several benchmarks, which invalidates the benchmark rather than the attack. Dataset inference at the *collection* level (Maini et al., NeurIPS 2024) does work where per-document inference does not: aggregating hundreds of documents yields statistically significant $p$-values.

**Claimed but unablated.** Multilingual extensions of Min-K% and reference calibration have been reported on translated benchmark sets, but the published numbers are benchmark numbers only: the non-member arm is typically machine-translated or post-cutoff text, so the classifier can succeed by detecting translationese or recency. No multilingual MIA result to date has been ablated against a same-distribution, same-time-window non-member arm *(frontier — verify)*.

**Adjacent and solid.** Contamination detection with an exchangeability test (Oren et al., ICLR 2024) gives valid $p$-values for whether a *benchmark* was trained on, by testing sensitivity to example ordering; it detects sets, not documents, and has not been run per-language. Canary/trap insertion (Meeus et al., ICML 2024, "Copyright Traps") produces reliable detection but requires controlling the corpus before training.

## 4. What Is Known

- **Monolingual MIA is weak at pretraining scale.** AUC $0.5$–$0.6$, Pythia 160M–12B, Pile (Duan et al. 2024). Attack strength rises with duplication count and with document length, not with model size alone.
- **Blind baselines beat many published attacks.** Das et al. (2024) report blind methods exceeding reported MIA AUC on multiple LLM MIA benchmarks including WikiMIA-style splits.
- **Memorization is real but concentrated.** Carlini et al. (USENIX Security 2021) extracted verbatim training sequences from GPT-2; extraction rate scales with model size and duplication. Chang et al. (EMNLP 2023, name-cloze) find near-ceiling accuracy on widely circulated public-domain English books and near-chance on obscure or recent ones — memorization tracks web frequency, which in turn tracks language.
- **Cross-lingual transfer is strong enough to confound.** XLM-R (Conneau et al., ACL 2020) zero-shot transfers from English to 100 languages; Wendler et al. (ACL 2024) show Llama-2 pivots through an English-centric latent space on non-English input. So loss on the Telugu edition falls when the *English* edition is trained on.
- **Corpus quality asymmetry is documented.** Kreutzer et al. (TACL 2022) audited public multilingual crawls and found many low-resource corpora below 50% correct-language or usable content; this contaminates any "known member" label built from corpus metadata.

## 5. What Is Not Known

- **Theoretically open.** Whether $\Delta_{\ell \to \ell'}$ is identifiable. No bound exists relating the loss reduction from *direct* membership in $\ell$ to the reduction from cross-lingual transfer from $\ell'$. If the two are within the same order, attribution is information-theoretically impossible for a given query budget, and that is the result the field needs.
- **Empirically open.** Nobody has trained a model with a deliberately *disjoint* language-edition assignment — book A in English only, book B in Telugu only, book C in both — at pretraining scale ($\geq 1$B parameters, $\geq 100$B tokens) and measured attribution AUC. The experiment is runnable today for under a few thousand GPU-hours; it has not been run.
- **Methodologically blocked.** The score is not yet well defined cross-lingually. Bits-per-byte removes the first-order fertility effect but leaves language-specific entropy, script-level tokenizer artifacts, and per-language reference-model quality uncorrected. Until a null distribution per language pair is constructed from same-time, same-domain non-members, a reported cross-lingual AUC does not mean what its name says.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability compounded by a confounded null**.

Two distinct causes push $\tilde{\mathcal{L}}_\theta(x^{(\text{te})})$ down: the Telugu edition was trained on, or the English edition was and the model transfers. Both are membership of the *content*; only one is membership of the *edition*. There is no observable that separates them without an intervention on the corpus.

Second, the null arm is unavailable off the shelf. A "non-member" Telugu document that is post-cutoff, machine-translated, or drawn from a different domain differs from members in ways any classifier will pick up — which is exactly how blind baselines beat MIA in English. Low-resource languages have too few documents to build a matched null by filtering: for languages with under $10^5$ web documents, matching on date, domain, register, and translation provenance leaves an empty set.

## 7. Current Research (as of 2026)

- **Auditing-grade attacks with calibrated FPR** — the LiRA-descended line (CMU, Google DeepMind, ETH Zurich SPY Lab) has largely moved from per-example MIA to dataset inference and canary insertion, conceding that per-document inference at pretraining scale is not viable.
- **Multilingual contamination detection** for evaluation integrity: extending Oren et al.'s exchangeability test to translated benchmark copies (MMLU→translated variants) *(frontier — verify)*.
- **Copyright-motivated translation attribution** — whether a publisher can show a *translated* edition was ingested. Legally salient in the EU AI Act transparency regime; no peer-reviewed method with calibrated error rates yet *(frontier — verify)*.
- **Open pretraining corpora as testbeds** — OLMo/Dolma and Pythia/Pile provide document-level ground truth, but neither has enough non-English coverage to support per-language nulls.

## 8. Concrete Next Experiment

**Scale.** Train two 1.4B-parameter models on 100B tokens of a controlled multilingual mix (English, Spanish, Hindi, Telugu, Swahili). Insert 3,000 document families, each with human or high-quality translations in all five languages, under a randomized assignment: each family is included in exactly one language (600 families per language), and all other editions are held out. Duplication is fixed at 1, 4, and 16 copies (1,000 families each) to trace the memorization threshold.

**Control arm.** Model B, trained on the identical corpus with all 3,000 families removed. This gives a true non-member score for every edition of every family, from the same distribution, the same time window, and the same translator — removing every confounder that blind baselines exploit. Second control: a blind attacker with no model access, scoring editions by length, fertility, and $n$-gram overlap with the English edition.

**The deciding number.** Attribution AUC: over the $3000 \times 5$ (family, language) pairs, area under ROC for predicting $m_\ell = 1$ from $\Delta$ calibrated on Model B. **A cross-lingual attribution AUC $\geq 0.70$ at duplication 4, with the blind baseline at $\leq 0.55$, establishes that language attribution is feasible.** AUC $\leq 0.60$ at duplication 16 is strong evidence for the non-identifiability side, and would redirect the field to canaries and dataset-level inference.

## 9. Key References

- **[Foundational]** Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P, 2017. — arXiv:1610.05820
- **[Foundational]** Nicholas Carlini et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[SOTA]** Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022. — arXiv:2112.03570
- **[SOTA]** Weijia Shi et al. *Detecting Pretraining Data from Large Language Models.* ICLR, 2024. — arXiv:2310.16789
- **[SOTA]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[SOTA]** Pratyush Maini, Hengrui Jia, Nicolas Papernot, Adam Dziedzic. *LLM Dataset Inference: Did you train on my dataset?* NeurIPS, 2024. — arXiv:2406.06443
- **[Critique]** Debeshee Das, Jie Zhang, Florian Tramèr. *Blind Baselines Beat Membership Inference Attacks for Foundation Models.* 2024. — arXiv:2406.16201
- **[Method]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR, 2024. — arXiv:2310.17623
- **[Method]** Matthieu Meeus, Igor Shilov, Manuel Faysse, Yves-Alexandre de Montjoye. *Copyright Traps for Large Language Models.* ICML, 2024.
- **[Memorization]** Kent K. Chang, Mackenzie Cramer, Sandeep Soni, David Bamman. *Speak, Memory: An Archaeology of Books Known to ChatGPT/GPT-4.* EMNLP, 2023. — arXiv:2305.00118
- **[Multilingual]** Alexis Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL, 2020. — arXiv:1911.02116
- **[Multilingual]** Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL, 2024. — arXiv:2402.10588
- **[Survey/Audit]** Julia Kreutzer et al. *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.* TACL, 2022. — arXiv:2103.12028

## 10. Worked Example

Take one document family: a novel with an English original and a Telugu translation. Query a Llama-class model with a 32k-token tokenizer.

Fertility, from the same 1,000-sentence FLORES-200 devtest passage, bytes per token: English $\approx 3.9$, Telugu $\approx 1.1$ (Telugu is a non-Latin abugida largely absent from the merge table, so it falls back to near-byte-level segmentation). The Telugu edition of a 500 KB book is $\approx 455$k tokens; the English edition is $\approx 128$k.

Suppose per-token loss comes back at $\mathcal{L}(\text{en}) = 2.10$ and $\mathcal{L}(\text{te}) = 1.35$ nats. Taken naively, Telugu looks *more* memorized. Convert to bits per byte:

$$\tilde{\mathcal{L}}(\text{en}) = \frac{2.10}{\ln 2 \cdot 3.9} = 0.78, \qquad \tilde{\mathcal{L}}(\text{te}) = \frac{1.35}{\ln 2 \cdot 1.1} = 1.77.$$

The ranking inverts. The normalization is doing all the work, and it is the *only* step here with a defensible justification.

Now the obstruction. The English bits-per-byte for held-out contemporary English prose sits near $0.85$; for held-out Telugu prose near $1.85$. So the family scores $\Delta$ of $-0.07$ (English) and $-0.08$ (Telugu) against their own language nulls — indistinguishable, and both well inside the spread of the null, which for single documents has a standard deviation around $0.10$ bits per byte. A signal of $0.08$ against a noise of $0.10$ is not a detection.

Push the Telugu gap to a clean $-0.30$ and the problem is still not solved, because two hypotheses both predict it: the Telugu edition was ingested, or the English edition was ingested and the model transfers named entities, numbers, dates, and plot-specific vocabulary — all of which survive translation — into the Telugu likelihood. Only the corpus intervention in §8, which holds one edition out by construction, tells the two apart. That is the whole problem in one number.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*