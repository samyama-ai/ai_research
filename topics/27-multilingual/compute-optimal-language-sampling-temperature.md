---
id: 27-multilingual/compute-optimal-language-sampling-temperature
title: "Compute-Optimal Language Sampling Temperature"
topic: 27-multilingual
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Language Sampling Temperature

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/compute-optimal-language-sampling-temperature` · **Status:** empirically-open

## 1. Problem Statement

Multilingual pretraining corpora are power-law skewed: English supplies orders of magnitude more tokens than Swahili. Practitioners reweight by exponentiating the natural language proportions, $q_\ell \propto p_\ell^{\alpha}$ with $\alpha = 1/T$, and pick $\alpha$ by folklore ($0.3$, $0.5$, $0.7$). The problem: **is there a compute-optimal $\alpha^{*}$, and does it move with model size $N$ and token budget $D$?**

Three variants, different difficulty:

- **Measurement.** Given a corpus, a budget, and a target loss functional, estimate $\alpha^{*}$ from a small number of runs. Blocked less by compute than by the choice of functional — mean loss, min-max loss, and downstream-task-weighted loss have different optima.
- **Method.** Beat the best fixed $\alpha$ with a schedule or a per-language weight vector (DoReMi, MultiDDS, UniMax). Partially solved: UniMax beats temperature sampling, but the comparison is against a *fixed* $\alpha$, not against $\alpha$ tuned per scale.
- **Theory.** Predict $\alpha^{*}(N, D)$ from corpus statistics and a fitted scaling law, without running the sweep. Open.

A solution is a function $\hat{\alpha}(N, D, \{n_\ell\})$ whose predicted optimum lands within the confidence interval of a measured sweep at a scale held out from the fit.

## 2. Formal Setting

Languages $\ell = 1,\dots,L$. Unique token counts $n_\ell$ measured **after** deduplication and **after** tokenization by the actual tokenizer used — both matter, because the tokenizer's own vocabulary allocation depends on $\alpha$ (SentencePiece is fit on the sampled mixture, so $\alpha$ enters twice).

Natural proportions and sampling distribution:

$$p_\ell = \frac{n_\ell}{\sum_{k} n_k}, \qquad q_\ell(\alpha) = \frac{p_\ell^{\alpha}}{\sum_k p_k^{\alpha}}, \qquad \alpha = 1/T \in (0,1].$$

$\alpha = 1$ is natural sampling; $\alpha \to 0$ is uniform. Compute $C \approx 6ND$ FLOPs (Kaplan et al. 2020) with $N$ non-embedding parameters and $D$ total training tokens. Language $\ell$ receives $D_\ell = q_\ell D$ sampled tokens, i.e. $R_\ell = D_\ell / n_\ell$ epochs.

Repetition is not free. Using the data-constrained fit of Muennighoff et al. (2023), effective unique-token value is

$$D_\ell^{\text{eff}} = n_\ell\left[1 + R^{*}\left(1 - e^{-(R_\ell - 1)/R^{*}}\right)\right], \qquad R^{*} \approx 15.$$

Per-language loss, measured as held-out cross-entropy in nats per **byte** (not per token — per-token loss is not comparable across tokenizers whose fertility depends on $\alpha$):

$$L_\ell(N, D_\ell^{\text{eff}}) = E_\ell + \frac{A_\ell}{N^{a}} + \frac{B_\ell}{(D_\ell^{\text{eff}})^{b}} - T_\ell(q_{-\ell}),$$

where $T_\ell$ is a transfer term from related languages and is the part with no accepted parametric form. Objectives:

$$\alpha^{*}_{\text{mean}} = \arg\min_\alpha \sum_\ell w_\ell L_\ell, \qquad \alpha^{*}_{\text{minmax}} = \arg\min_\alpha \max_\ell \big(L_\ell - L_\ell^{\text{mono}}\big),$$

with $L_\ell^{\text{mono}}$ the loss of a monolingual model at the same $N$ and the same compute share.

**Assumptions known to be violated.** (i) Additivity — $T_\ell$ makes languages non-independent, so the mixture is not separable. (ii) Fixed tokenizer — vocabulary allocation co-varies with $\alpha$. (iii) $n_\ell$ is language-pure — web corpora leak English into every language bucket (Blevins & Zettlemoyer 2022), so $p_\ell$ is not identified. (iv) $R^{*}$ constant across languages — unmeasured for low-resource text, where duplication rates differ.

## 3. State of the Art

**Established.** Temperature sampling as a default is universal and its endpoints are documented: mBERT used exponential smoothing $\alpha = 0.7$; XLM (Lample & Conneau, NeurIPS 2019) used $\alpha = 0.5$; XLM-R (Conneau et al., ACL 2020) used $\alpha = 0.3$; mT5 (Xue et al., NAACL 2021) used $\alpha = 0.3$; massively multilingual NMT used $T = 5$ ($\alpha = 0.2$) (Arivazhagan et al. 2019). None of these papers report a sweep of $\alpha$ across model scales; the values are inherited.

**Empirical SOTA on the method variant.** UniMax (Chung et al., ICLR 2023) replaces $q_\ell \propto p_\ell^\alpha$ with an explicit epoch cap: distribute the budget as uniformly as possible subject to no language exceeding $N_{\text{epoch}}$ repeats. It reports gains over temperature sampling on XTREME-R at mT5 scales up to 13B. This is the strongest evidence that fixed $\alpha$ is the wrong *family*, not just the wrong value. **Claimed but unablated:** the baseline is $\alpha=0.3$ at every scale, so UniMax's advantage over a *per-scale-tuned* $\alpha$ is not established.

**Learned-weight SOTA.** DoReMi (Xie et al., NeurIPS 2023) fits domain weights with group DRO on a small proxy model and transfers them to an 8B model, reporting ~2.6× training speedup on The Pile — a domain-mixture result, applied to languages only by analogy. MultiDDS (Wang, Lipton & Tsvetkov, ACL 2020) optimizes language weights by gradient alignment in multilingual NMT. Data Mixing Laws (Ye et al. 2024) predicts loss as a function of mixture proportions and is the closest thing to a predictive theory, but is fit on English domains.

**Benchmark-number-only results.** All cross-lingual gains above are reported as XTREME/XTREME-R/XNLI scores at one or two scales. There is no published $\alpha \times N$ grid.

## 4. What Is Known

- **The curse of multilinguality is capacity-relative.** XLM-R (Conneau et al. 2020) shows XNLI accuracy rising then falling as languages are added at fixed capacity, and the fall being recovered by increasing $N$ — measured at Base (270M) and Large (550M) scales, 7 → 100 languages.
- **It is largely a data-budget and vocabulary effect, not an interference effect.** Chang et al. (EMNLP 2024), training over 250 languages at up to ~1B parameters, find that adding *related* language data helps low-resource languages and that much of the apparent curse disappears once per-language data and vocabulary allocation are controlled.
- **Repeats saturate.** Muennighoff et al. (NeurIPS 2023), at up to 9B parameters and 900B tokens, find ~4 epochs nearly as valuable as fresh data and returns approaching zero past ~16 epochs. High $\alpha^{-1}$ pushes low-resource languages to 40+ epochs.
- **Multilingual NMT loss follows a scaling law with mixture-dependent coefficients.** Fernandes et al. (ICML 2023) fit per-pair losses whose irreducible term and exponent shift with the language weighting.
- **Chinchilla's $D \propto N$ rule was fit on near-monolingual English data** (Hoffmann et al., NeurIPS 2022) and has never been re-fit under a skewed multilingual mixture.

## 5. What Is Not Known

- **Empirically open (dominant gap).** Whether $\alpha^{*}$ shifts with $N$ at fixed $D$, and with $D$ at fixed $N$. The experiment is a $10 \times 5$ grid at ≤1B parameters — entirely runnable on a few thousand GPU-hours, and not published.
- **Empirically open.** Whether $\alpha^{*}$ under mean loss and under min-max loss coincide, and by how much they diverge.
- **Theoretically open.** No proof that $\sum_\ell w_\ell L_\ell(\alpha)$ is unimodal in $\alpha$. If the transfer term $T_\ell$ is non-monotone, the sweep can have multiple local minima and grid search on a proxy model is unsound.
- **Methodologically blocked.** Cross-language loss comparison. Nats-per-token is tokenizer-dependent; nats-per-byte penalizes scripts with high byte-per-character cost (Devanagari, Amharic). No accepted normalizer exists, so $\max_\ell$ in the min-max objective is not well defined.
- **Methodologically blocked.** $p_\ell$ itself: language-ID error and cross-language contamination in web crawls mean the denominator of the temperature formula is estimated, not known.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability between three effects that $\alpha$ moves simultaneously**: (1) tokens allocated per language, (2) the number of repeats each language sees, and (3) the tokenizer's vocabulary share. A single final-loss curve over $\alpha$ cannot separate "the low-resource language needed more weight" from "the extra weight bought only re-reads" from "the tokenizer got better coverage". Fixing the tokenizer across $\alpha$ breaks (3) but makes the comparison unrepresentative of deployment, where the tokenizer is always co-fit. Compute is a secondary obstruction: a proper grid requires the sweep at *each* $N$, because the claim under test is that the optimum moves.

## 7. Current Research (as of 2026)

- **Mixture-law extrapolation applied to languages** — fitting loss-vs-proportion surfaces on small proxies and extrapolating to target scale, following Ye et al. (2024) and DoReMi. Groups: Shanghai AI Lab, Stanford, Google DeepMind. *(frontier — verify)*
- **Multilingual scaling laws with explicit language-count terms** — several 2024–2025 preprints fit $L(N, D, L)$ with a language-family similarity kernel. *(frontier — verify the specific parameterizations.)*
- **Epoch-capped and curriculum sampling** — UniMax variants with $\alpha$ annealed over training (high $\alpha$ early, low late). Reported informally in open-model tech reports; no controlled ablation known.
- **Byte- and token-free models** (MegaByte-style, BLT) sidestep the tokenizer confound and would make the loss comparison better-posed. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Scale.** Five decoder-only models, $N \in \{70\text{M}, 160\text{M}, 410\text{M}, 1.0\text{B}, 1.7\text{B}\}$, each trained at $D = 20N$ tokens (Chinchilla-matched), on a fixed 20-language subset of CC-100/mC4 spanning four families and a 200× resource range. Sweep $\alpha \in \{0.1, 0.2, \dots, 1.0\}$. Total: 50 runs, ~3,000 A100-hours.

**Control arms.** (a) A tokenizer frozen at $\alpha = 0.3$ for every cell, run alongside the co-fit tokenizer, to isolate the vocabulary channel. (b) A repeat-matched arm: for each $\alpha$, truncate every language at $R_\ell \le 4$ epochs and top up with the next-highest-resource language, isolating the repetition channel. (c) Per-language monolingual models at each $N$ for $L_\ell^{\text{mono}}$.

**The deciding number.** The slope $\partial \alpha^{*} / \partial \log_2 N$, where $\alpha^{*}$ minimizes uniformly-weighted held-out nats-per-byte. Across the 4.6 doublings from 70M to 1.7B: if $|\Delta \alpha^{*}| < 0.05$ (within the bootstrap CI of the sweep), $\alpha$ is scale-free and the folklore constant is vindicated. If $|\Delta \alpha^{*}| \ge 0.15$, every model trained with an inherited $\alpha$ is mis-weighted, and the sign of the slope tells the field which way. Report the same slope for arm (b): if it vanishes there, the scale dependence *is* the repetition law and needs no new theory.

## 9. Key References

- **[Foundational]** G. Lample, A. Conneau. *Cross-lingual Language Model Pretraining.* NeurIPS 2019. — arXiv:1901.07291
- **[Foundational]** A. Conneau, K. Khandelwal, N. Goyal, et al. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. — arXiv:1911.02116
- **[Foundational]** N. Arivazhagan, A. Bapna, O. Firat, et al. *Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges.* 2019. — arXiv:1907.05019
- **[SOTA]** H. W. Chung, N. Constant, X. Garcia, et al. *UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining.* ICLR 2023.
- **[SOTA]** S. M. Xie, H. Pham, X. Dong, et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS 2023. — arXiv:2305.10429
- **[SOTA]** N. Muennighoff, A. M. Rush, B. Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[SOTA]** J. Ye, P. Liu, Q. Sun, et al. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* 2024.
- **[Evidence]** T. Chang, C. Arnett, Z. Tu, B. Bergen. *When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages.* EMNLP 2024.
- **[Evidence]** P. Fernandes, B. Ghorbani, X. Garcia, et al. *Scaling Laws for Multilingual Neural Machine Translation.* ICML 2023.
- **[Evidence]** X. Wang, Z. C. Lipton, Y. Tsvetkov. *Balancing Training for Multilingual Neural Machine Translation.* ACL 2020.
- **[Evidence]** T. Blevins, L. Zettlemoyer. *Language Contamination Helps Explain the Cross-lingual Capabilities of English Pretrained Models.* EMNLP 2022.
- **[Foundational]** J. Hoffmann, S. Borgeaud, A. Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Survey]** NLLB Team. *No Language Left Behind: Scaling Human-Centered Machine Translation.* 2022. — arXiv:2207.04672

## 10. Worked Example

Three CC-100 languages, using the corpus sizes reported by Conneau et al. (2020): English $\approx 55{,}608$M tokens, Hindi $\approx 1{,}715$M, Swahili $\approx 275$M. Total $57{,}598$M, so $p = (0.9654,\ 0.0298,\ 0.00478)$.

At the field-default $\alpha = 0.3$: $p^{0.3} = (0.9894,\ 0.3485,\ 0.2013)$, normalizing to

$$q = (0.643,\ 0.226,\ 0.131).$$

Train at $D = 100$B tokens. Swahili receives $0.131 \times 100\text{B} = 13.1$B tokens from $0.275$B unique — **47.6 epochs**. Apply the repetition law with $R^{*} = 15$:

$$D_{\text{sw}}^{\text{eff}} = 0.275\left[1 + 15\left(1 - e^{-46.6/15}\right)\right] = 0.275 \times 15.32 = 4.21\text{B}.$$

So 13.1B tokens of compute (13% of the run) buy the learning value of 4.2B — **68% of the Swahili budget is dead weight**. Hindi at $q = 0.226$ gets 22.6B tokens over 1.715B unique = 13.2 epochs, $D^{\text{eff}} = 1.715 \times 13.6 = 23.3$B: essentially no waste.

Now the obstruction. Suppose a sweep finds $\alpha^{*} = 0.45$ at 1.7B parameters but $\alpha^{*} = 0.30$ at 160M. Three readings are consistent with that curve and the experiment above cannot distinguish them:

1. The larger model has more capacity per language, so it extracts more from *fresh* high-resource tokens — a genuine transfer effect.
2. The larger model, trained at $D = 20N$, sees more total tokens, pushing Swahili from 12 to 47 epochs; raising $\alpha$ merely trims wasted re-reads. The optimum moved because of $R^{*}$, not because of language transfer.
3. At $\alpha = 0.45$ the co-fit SentencePiece vocabulary shifts share toward Latin script, cutting English fertility and effectively increasing the English token budget again.

Only the frozen-tokenizer arm (b) and the repeat-capped arm (c) of §8 separate these. Without them, a measured $\Delta\alpha^{*}$ of $0.15$ is a number without a mechanism — which is why the sweep has never yielded a transferable rule.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*