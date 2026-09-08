---
id: 27-multilingual/curse-of-multilinguality-threshold
title: "The Curse of Multilinguality Threshold"
topic: 27-multilingual
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# The Curse of Multilinguality Threshold

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/curse-of-multilinguality-threshold` · **Status:** partially-solved

## 1. Problem Statement

A single model trained on many languages at fixed capacity eventually gets worse at each of them. Conneau et al. (2020) named this the *curse of multilinguality* and located a turn: adding languages helps low-resource languages up to a point, then hurts. The open problem is to characterize that point.

- **Measurement variant.** Given a training recipe, does a threshold $L^*$ exist — a language count beyond which per-language quality declines — and can it be estimated without confounding it with tokenizer change, data-quantity change, and benchmark coverage change? This variant is the one currently blocked.
- **Method variant.** Can architectural modularity (adapters, experts, per-language parameters) push $L^*$ past the number of languages anyone wants, making the threshold irrelevant? Partially solved: X-MOD (Pfeiffer et al., 2022) and X-ELM (Blevins et al., 2024) both report no degradation where dense baselines degrade.
- **Theory variant.** Is there a scaling law $L^*(N, D)$ — threshold as a function of parameters and tokens — with an exponent that can be predicted rather than fit after the fact? Open.

Solving it means: a predictive rule that, given $(N, D, V)$ and a language inventory, says how many languages fit before per-language loss on a fixed evaluation set starts rising, verified out-of-sample at a scale not used to fit it.

## 2. Formal Setting

Let $\mathcal{L} = \{1,\dots,L\}$ be the language inventory, $q_\ell$ the natural corpus share of language $\ell$, and the training mixture set by exponential smoothing:

$$p_\ell = \frac{q_\ell^{\alpha}}{\sum_{k} q_k^{\alpha}}, \qquad \alpha \in (0,1].$$

*Measured as:* $q_\ell$ is bytes of deduplicated text after language ID; $\alpha$ is a recipe constant ($\alpha = 0.3$ for XLM-R, $0.3$–$0.7$ elsewhere).

Let $N$ be non-embedding parameters, $V$ vocabulary size, $d$ hidden width, so total parameters are $N + Vd$ (tied embeddings). Let $D$ be training tokens. Define per-language quality in **bits per character**, not per token:

$$\mathrm{BPC}_\ell = \frac{\mathbb{E}_{x \sim \mathcal{D}_\ell}[-\log_2 P_\theta(x)]}{|x|_{\text{chars}}}.$$

*Measured as:* sum of token log-probabilities over a held-out corpus, divided by NFC-normalized Unicode character count. This is the only quantity comparable across models with different tokenizers; per-token loss is not.

Define the low-resource aggregate over a **fixed** evaluation core $\mathcal{C} \subseteq \mathcal{L}$ present in every arm:

$$Q(L) = -\frac{1}{|\mathcal{C}|}\sum_{\ell \in \mathcal{C}} \mathrm{BPC}_\ell(L), \qquad L^*(N,D,V) = \arg\max_L Q(L).$$

The curse is the claim $\partial Q/\partial L < 0$ for $L > L^*$ with $L^* < \infty$. Capacity per language is $\kappa = N/L$; the naive hypothesis is $L^* \propto N$, i.e. $\kappa^*$ constant.

**Assumptions, and which are violated.**
1. *Evaluation core held fixed as $L$ grows.* Usually violated — XNLI covers 15 languages, so arms with 7 vs 100 languages are scored on different intersections.
2. *Tokenizer held fixed.* Violated by construction: adding languages changes the learned vocabulary, changing characters-per-token and therefore loss-per-token.
3. *Per-language token count held fixed.* Violated when $D$ is held fixed and $\alpha$-sampling redistributes mass.
4. *Languages are exchangeable.* Violated: typological and script relatedness dominates transfer (Chang et al., 2023).
5. *Corpus quality constant across languages.* Violated — low-resource web crawls are markedly noisier.

## 3. State of the Art

**Established.**
- Conneau et al. (2020, ACL), XLM-R: at fixed capacity, XNLI accuracy on low-resource languages rises then falls as the language count goes $7 \to 30 \to 100$; raising capacity (Base $\to$ Large) recovers the loss. This is the original controlled demonstration and the source of the term.
- Pfeiffer et al. (2022, NAACL), X-MOD: language-specific modular components at 60 languages remove the degradation a dense baseline shows, and allow adding languages post hoc.
- Blevins et al. (2024, EMNLP), X-ELM: independently-trained cross-lingual expert LMs beat a dense multilingual model of matched total compute, with new languages addable without forgetting.
- Chang et al. (2023), 1,989 models over 252 languages: adding *related* languages helps a low-resource language; adding unrelated data mainly helps by adding tokens, and hurts high-resource languages.

**Claimed but unablated.**
- That $L^*$ scales linearly with $N$. This is folklore read off the Base/Large comparison; no paper fits the exponent.
- That instruction tuning inherits the same threshold. Shaham et al. (2024) show multilingual instruction following emerges from very few languages, which is evidence *against* a shared threshold, but for a different objective.

**Benchmark-number-only.** Most "no curse observed at $L$ languages" claims for massively multilingual models (Glot500-m at 511 languages, Imani et al. 2023; Aya-101 at 101, Üstün et al. 2024) rest on aggregate task scores against a *different* baseline model, not against a language-count sweep with matched compute and tokenizer. They do not measure $L^*$.

## 4. What Is Known

- **The turn is real at fixed capacity, at 270M–550M parameters.** XLM-R reports XNLI averages of roughly 76 (Base, 270M) and 81 (Large, 550M) at 100 languages; the low-resource degradation at 100 languages is erased by the capacity increase. Scale: encoder models, 2.5TB CommonCrawl, $V = 250{,}000$.
- **Embeddings dominate small multilingual models.** XLM-R Base: $V d = 250{,}000 \times 768 \approx 192$M embedding parameters against $\approx 85$M non-embedding. The "capacity" being diluted is mostly *not* where language modeling happens.
- **Relatedness beats count.** Chang et al. (2023), models up to ~45M parameters, 252 languages: the curse for low-resource languages largely disappears once the added data is typologically related; the residual harm falls on high-resource languages.
- **Modularity defers the turn.** X-MOD at 60 languages, X-ELM at 16: both report the dense baseline degrading where the modular model does not, at matched or lower inference FLOPs.
- **Tokenizer fertility varies up to ~15× across languages** for a shared vocabulary (Petrov et al., NeurIPS 2023), which is enough to reverse the sign of a per-token loss comparison.
- **Multilingual NMT admits joint scaling laws.** Fernandes et al. (ICML 2023) fit loss as a function of parameters and mixture weights for small language sets; the fits do not extend to hundreds of languages.

## 5. What Is Not Known

- **Theoretically open.** No result predicts $L^*$ from $(N, D, V)$ and a typological distance matrix. There is no proof that $L^*$ is finite for any fixed $N$, nor an interference bound analogous to a capacity theorem for multi-task models. The exponent $\beta$ in $L^* \propto N^{\beta}$ is unmeasured; $\beta = 1$ is assumed, never fit.
- **Empirically open.** Nobody has run a language-count sweep at frontier decoder scale ($\geq 7$B parameters, $\geq 1$T tokens) with tokenizer and per-language token counts controlled. The experiment is straightforward and expensive.
- **Methodologically blocked.** Whether the curse still exists once tokenizer change is removed. Every published sweep changes the vocabulary between arms, so "loss went up" and "the same text became more tokens" are not separated. Fixing this requires either a byte-level model or a frozen superset vocabulary — neither is standard.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by cost**, in three specific ways.

1. *Tokenizer non-comparability.* The independent variable (language count) mechanically changes the measuring instrument (the tokenizer). BPC fixes the units but not the modeling advantage of a better segmentation, so a BPC drop still conflates "learned the language" with "segments it more efficiently".
2. *Three variables move together.* Adding a language adds tokens, adds vocabulary, and subtracts effective per-language capacity. A single sweep cannot attribute the effect; disentangling needs a $3\times$ larger factorial design.
3. *Evaluation coverage shrinks with the question.* The benchmarks that support cross-lingual comparison (XNLI, 15 languages; FLORES-200) do not cover the tail languages whose treatment is the whole point, so the aggregate score at $L = 300$ is computed over a core that is not representative.

Cost then blocks the fix: a controlled 4-point sweep at 7B parameters over 1T tokens is roughly $4 \times 6ND \approx 1.7 \times 10^{23}$ FLOPs, several thousand H100-days.

## 7. Current Research (as of 2026)

- **Modular and expert routing** — Meta AI (X-MOD lineage), UW/AI2 (X-ELM), and MoE work where experts specialize by script. Direction: make $L^*$ unbounded rather than estimate it.
- **Vocabulary scaling** — treating $V$ as a first-class scaling variable rather than a constant; the open question is whether the optimal $V$ grows with $L$ or with $N$. *(frontier — verify)*
- **Byte- and character-level backbones** (Byte Latent Transformer-style, and CANINE/ByT5 lineage) as a way to delete the tokenizer confound outright. *(frontier — verify)*
- **Massively multilingual continued pretraining** — Glot500/MaLA (LMU Munich, CIS), Cohere Labs Aya — extending coverage to 500+ languages; these measure coverage benefit, not the threshold.
- **Data-centric accounts** — Chang, Arnett, Bergen and collaborators (UCSD) arguing the curse is mostly a data-quantity and relatedness artifact.

## 8. Concrete Next Experiment

**Question.** Does the curse survive removal of the tokenizer confound, and does $L^*$ grow with $N$?

**Scale.** Decoder-only LMs at $N \in \{150\text{M}, 400\text{M}, 1.2\text{B}\}$ non-embedding parameters, $D = 100$B tokens each, language counts $L \in \{10, 30, 100, 300\}$. Twelve runs plus controls; roughly $2\times10^{22}$ FLOPs total, ~1,500 H100-days.

**Controls.**
- *Frozen tokenizer:* one SentencePiece vocabulary ($V = 256$k) trained once on the full 300-language union and reused in every arm. Removes assumption 2.
- *Fixed per-language budget for the core:* an evaluation core $\mathcal{C}$ of 10 languages (2 high-, 4 mid-, 4 low-resource, 4 scripts) receives an identical token count in every arm; extra languages are added on top of $D$, not carved out of it. Removes assumption 3.
- *Control arm:* the $L = 10$ model, which sees exactly the core at the same per-language budget with the same vocabulary — the only difference from every other arm is the presence of other languages.

**Deciding number.** Fit $L^*(N) \propto N^{\beta}$ to the three per-$N$ maxima of $Q(L)$ measured in BPC on $\mathcal{C}$. Report $\hat\beta$ with a bootstrap 95% CI over the core languages.
- $\hat\beta \geq 0.75$: the curse is capacity dilution; scale dissolves it and modularity is an efficiency choice, not a necessity.
- $\hat\beta \approx 0$ (CI excludes 0.25): $L^*$ is capacity-independent, the curse is interference, and modular architectures are the only route past it.
- $Q(L)$ monotone increasing at every $N$ with the tokenizer frozen: the published curse was largely a tokenizer artifact — the most consequential outcome, and the one no existing experiment rules out.

## 9. Key References

- **[Foundational]** Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. — arXiv:1911.02116
- **[Foundational]** Naveen Arivazhagan et al. *Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges.* 2019. — arXiv:1907.05019
- **[SOTA]** Jonas Pfeiffer, Naman Goyal, Xi Victoria Lin, Xian Li, James Cross, Sebastian Riedel, Mikel Artetxe. *Lifting the Curse of Multilinguality by Pre-training Modular Transformers.* NAACL 2022. — arXiv:2205.06266
- **[SOTA]** Terra Blevins, Tomasz Limisiewicz, Suchin Gururangan, Margaret Li, Hila Gonen, Noah A. Smith, Luke Zettlemoyer. *Breaking the Curse of Multilinguality with Cross-lingual Expert Language Models.* EMNLP 2024. — arXiv:2401.10440
- **[Evidence]** Tyler A. Chang, Catherine Arnett, Zhuowen Tu, Benjamin K. Bergen. *When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages.* 2023. — arXiv:2311.09205
- **[Evidence]** Zirui Wang, Zachary C. Lipton, Yulia Tsvetkov. *On Negative Interference in Multilingual Models: Findings and a Meta-Learning Treatment.* EMNLP 2020.
- **[Evidence]** Patrick Fernandes, Behrooz Ghorbani, Xavier Garcia, Markus Freitag, Orhan Firat. *Scaling Laws for Multilingual Neural Machine Translation.* ICML 2023. — arXiv:2302.09650
- **[Measurement]** Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, Adel Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023.
- **[Coverage]** Ayyoob Imani et al. *Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages.* ACL 2023.
- **[Survey]** Ahmet Üstün et al. *Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model.* ACL 2024.
- **[Contrast]** Uri Shaham et al. *Multilingual Instruction Tuning With Just a Pinch of Multilinguality.* Findings of ACL 2024.

## 10. Worked Example

Take XLM-R Base as the instance: $d = 768$, 12 layers, $N \approx 85$M non-embedding, $V = 250{,}000$, so $Vd \approx 192$M embedding parameters — 69% of the model.

**Capacity accounting.** Non-embedding capacity per language:

$$\kappa(7) = \frac{85\text{M}}{7} = 12.1\text{M}, \qquad \kappa(100) = \frac{85\text{M}}{100} = 0.85\text{M}.$$

A 14× drop. If $L^*$ were set by a constant $\kappa^*$, this alone would explain the reported turn between 30 and 100 languages, and the Base→Large fix (which raises $N$ to ~300M non-embedding, $\kappa(100) = 3.0$M) follows.

**Where the account breaks.** Now do the same accounting on the tokenizer. Suppose Swahili is evaluated at 4.2 characters/token in the $L=100$ vocabulary with per-token cross-entropy 3.00 nats:

$$\mathrm{BPC} = \frac{3.00}{\ln 2 \times 4.2} = 1.03.$$

Retrain the vocabulary at $L = 300$; Swahili's share of the 250k slots shrinks and fertility rises to 3.4 characters/token. To hold BPC at 1.03 the model now needs per-token loss

$$3.4 \times 1.03 \times \ln 2 = 2.43 \ \text{nats},$$

a 19% *drop* in per-token loss for identical modeling quality. A sweep that reports per-token loss would record the $L = 300$ model as substantially better; a sweep reporting BPC records no change. The published curse literature mostly reports downstream accuracy, which is affected by fertility in the opposite direction (longer sequences, shorter effective context).

**The visible obstruction.** Two mechanisms — capacity dilution (14×, real, measurable) and fertility change (19% in the loss units, real, an artifact) — move together with $L$ and in opposite directions on the reported metric. No published run holds the second fixed. Until the frozen-vocabulary arm in §8 exists, $L^*$ is not identifiable from the data we have.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*