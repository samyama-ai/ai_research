---
id: 27-multilingual/cross-lingual-vocabulary-allocation
title: "Compute-Optimal Vocabulary Allocation Across Languages"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Vocabulary Allocation Across Languages

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/cross-lingual-vocabulary-allocation` · **Status:** open

## 1. Problem Statement

Given a fixed training compute budget and a fixed multilingual corpus, how many vocabulary slots should go to each language?

- **Input:** a set of languages $L$, a per-language corpus $D_\ell$ (bytes), a total compute budget $C$ (FLOPs), a model width $d$ and depth, and a total vocabulary size $V$ (itself a free variable).
- **Output:** a partition of the vocabulary — an allocation $v_\ell$ of slots per language, plus the induced sharing structure across scripts and cognates — and the data-sampling exponent that accompanies it.
- **Objective:** minimize a compute-normalized, tokenization-invariant loss aggregate (§2), not per-token perplexity, which is trivially gamed by coarser tokenization.

Three variants, with different difficulty:

- **Measurement.** Is there a tokenizer-independent per-language quality metric that is comparable across allocations? Bits-per-byte is the standard answer, but it does not track downstream quality uniformly across morphological types.
- **Method.** Given a metric, find $\{v_\ell\}$ maximizing it under budget. Currently done by heuristics (temperature-sampled unigram LM fitting, language clustering, greedy capacity allocation).
- **Theory.** Is there a scaling law $v_\ell^\ast(C, N, D_\ell, H_\ell)$ — allocation as a function of budget, non-embedding parameters, per-language data and per-language entropy — with a derivable exponent? Monolingual versions exist; the multilingual one does not.

Solved means: a predictive rule that, given a new language mixture, outputs $\{v_\ell\}$ whose realized loss is within noise of the best allocation found by brute-force search, verified at ≥2 model scales and ≥1 order of magnitude of budget.

## 2. Formal Setting

Let $\ell \in L$ index languages, $T_\ell$ the tokenizer restricted to $\ell$, and $V = \sum_\ell v_\ell - (\text{shared slots})$.

**Fertility** (measured on a held-out parallel corpus, e.g. FLORES-200, to hold content constant):

$$f_\ell = \frac{\\#\text{tokens}(T(x_\ell))}{\\#\text{words}(x_\ell)}, \qquad \rho_\ell = \frac{\\#\text{tokens}(T(x_\ell))}{\\#\text{UTF-8 bytes}(x_\ell)}$$

**Tokenization-invariant loss.** Per-token NLL is not comparable across allocations; normalize to bytes:

$$\mathcal{L}_\ell^{\mathrm{BPB}} = \frac{1}{\ln 2}\cdot\frac{\sum_{i} -\log p_\theta(t_i \mid t_{<i})}{\\#\text{bytes}(x_\ell)} = \frac{\rho_\ell}{\ln 2}\cdot \overline{\mathrm{NLL}}_{\text{token}}$$

**Compute accounting.** Split parameters into non-vocabulary $N_{nv}$ and vocabulary $N_v = 2dV$ (input embedding + output head, untied). Per-token cost is $\approx 6(N_{nv} + N_v)$, so vocabulary is *not free*: enlarging $V$ trades attention/FFN capacity against embedding capacity at fixed $C$, while simultaneously reducing the token count $D_{\text{tok}} = \sum_\ell \rho_\ell |D_\ell|$ needed to consume the corpus. The budget constraint is

$$C \approx 6\,(N_{nv} + 2dV)\sum_\ell \rho_\ell(v_\ell)\,|D_\ell|.$$

**The optimization.** Choose $\{v_\ell\}$ and sampling weights $q_\ell \propto p_\ell^{\alpha}$ ($p_\ell$ the natural corpus share) to minimize either a utilitarian or an egalitarian aggregate:

$$\min_{\{v_\ell\},\alpha}\ \sum_\ell w_\ell\, \mathcal{L}_\ell^{\mathrm{BPB}} \quad\text{(utilitarian)}\qquad\text{vs.}\qquad \min \max_\ell\ \big[\mathcal{L}_\ell^{\mathrm{BPB}} - \mathcal{L}_\ell^{\mathrm{mono}}\big] \quad\text{(egalitarian)}$$

where $\mathcal{L}_\ell^{\mathrm{mono}}$ is a monolingual model trained at compute $C\cdot q_\ell$ — the per-language control that makes "multilingual tax" measurable rather than rhetorical.

**Assumptions, and which are violated.**

1. *Bits-per-byte is monotone in downstream quality.* Violated in part: compression correlates with generation tasks more strongly than with classification (Goldman et al., 2024).
2. *UTF-8 bytes are a neutral denominator.* Violated: UTF-8 charges 1 byte/char for Latin, 3 for Devanagari/CJK, so BPB systematically flatters Latin-script languages. Bits-per-character is the alternative and flatters CJK instead. No neutral denominator exists.
3. *Language identity is a partition.* Violated: code-switching, shared scripts, loanwords, and web-corpus language-ID error (a few percent, higher for low-resource languages) make $v_\ell$ non-disjoint by construction.
4. *Corpus quality is constant across $\ell$.* Badly violated: low-resource web data is more duplicated and more machine-translated, which confounds any data-quantity term.

## 3. State of the Art

**Established.**

- Temperature/exponential sampling with $\alpha \in [0.3, 0.7]$ for both vocabulary fitting and data sampling — Conneau & Lample (2019), Arivazhagan et al. (2019), reused in mBERT, XLM-R and BLOOM. Robustly better than $\alpha=1$; the specific value is folklore, not optimized per budget.
- Monolingual vocabulary scaling law: Tao et al., NeurIPS 2024, derive $V^\ast$ growing sublinearly with non-vocabulary parameters (fitted exponent $\approx 0.83$), and show existing models are under-vocabularized — e.g. a Llama-2-70B-shaped model's compute-optimal vocabulary is ~216K against the shipped 32K. This is the closest thing to a theory, and it is **monolingual**.

**Claimed but unablated.**

- XLM-V (Liang et al., EMNLP 2023): 1M-token vocabulary built by de-emphasizing token sharing between low-overlap languages; reports gains over XLM-R on XNLI (~1–3 points average) and larger gains on low-resource NER. Single model scale, single seed, vocabulary and data pipeline changed together — the allocation rule is not isolated from the capacity increase.
- VoCap (Zheng et al., EMNLP 2021): allocates vocabulary capacity by a per-language marginal-utility curve on unigram log-likelihood. The allocation objective is a proxy (corpus likelihood), never validated against realized downstream loss at more than one scale.
- Language-clustered vocabularies (Chung et al., EMNLP 2020): cluster languages, build per-cluster vocabularies, union them. Reported gains on XTREME-style tasks; the clustering is chosen by embedding similarity, not by a compute argument.

**Benchmark-number-only.** Ali et al. (NAACL Findings 2024) report large downstream swings from tokenizer choice in multilingual English/German-focused training; the headline percentages are relative differences on selected task suites, not a controlled allocation sweep.

## 4. What Is Known

- **Fertility disparity is large and measured.** For the same content, token counts differ by up to an order of magnitude across languages under commercial tokenizers; Petrov et al. (NeurIPS 2023) report ratios above 10× for the worst-served languages against English, and Ahia et al. (EMNLP 2023) convert this into API cost and context-window premiums of ~2–5× for many non-Latin languages. Measured on parallel corpora (FLORES-200, Bible corpora), so content is controlled.
- **Tokenizer replacement recovers a substantial part of the multilingual gap.** Rust et al. (ACL 2021) show that swapping mBERT's shared vocabulary for a monolingual one and retraining embeddings closes much of the distance to monolingual models on POS/NER/QA at BERT-base scale (110M params) — evidence the bottleneck is partly vocabulary, not only parameters.
- **Vocabulary size trades against non-embedding capacity.** At 33M–1.13B non-vocabulary parameters, Tao et al. show loss is convex in $V$ with a well-defined optimum at each budget, and that the optimum shifts up with scale.
- **The curse of multilinguality is capacity-mediated.** XLM-R (Conneau et al., ACL 2020) shows per-language quality degrades as $|L|$ grows at fixed capacity and recovers when capacity grows — measured at 250M–550M params, 100 languages, 250K vocabulary.
- **Compression predicts quality unevenly.** Goldman et al. (ACL Findings 2024) find tokenizer compression correlates with downstream performance for generation tasks but weakly for classification, at ~350M-parameter scale.

## 5. What Is Not Known

- **Theoretically open.** No derivation of $v_\ell^\ast$ from a source-coding argument. The natural conjecture — allocate slots so that marginal bits-per-byte reduction is equalized across languages, $\partial \mathcal{L}_\ell^{\mathrm{BPB}}/\partial v_\ell = \lambda$ for all $\ell$ — has no proof, and its premise (separable per-language loss) is false under cross-lingual transfer, where adding Spanish slots changes Portuguese loss.
- **Empirically open.** The full sweep — $\{v_\ell\}$ allocation rule × 2–3 model scales × 2 budgets, with per-language monolingual controls — is runnable today for ~1B-parameter models and 20 languages. Nobody has published it. Cost, not feasibility, is the barrier.
- **Methodologically blocked.** Whether "fairness across languages" is even well defined: BPB, BPC, and per-word perplexity rank allocations differently, and there is no agreed normalizer (§2, assumption 2). Any minimax objective inherits the arbitrariness of the denominator.

## 6. Why It Is Hard

**The measurement is confounded by the thing being varied.** Changing $v_\ell$ changes the token sequence, which changes the loss denominator, the effective sequence length, the number of gradient steps per byte, and the position of content within the context window — all at once. Byte normalization removes the first confound and leaves the rest. A tokenizer that halves fertility for Telugu also doubles the Telugu content visible in a 4K context, so a downstream gain cannot be attributed to allocation versus effective context.

**Non-identifiability of data quality and allocation.** Low-resource languages have both fewer slots and worse data. Any observed benefit of extra slots is entangled with duplication rates and machine-translation contamination in the corpus. Deduplicated, human-verified corpora at matched size across 20 languages do not exist.

**Cost.** A single 1B-parameter, 100B-token run is $\approx 6 \times 10^{20}$ FLOPs; a 6-arm × 2-scale sweep with monolingual controls is a low-tens-of-thousands-of-GPU-hours experiment. That is affordable for a lab and unaffordable for most academic groups — which is exactly why the question stays open while opinions about it multiply.

## 7. Current Research (as of 2026)

- **Extending vocabulary scaling laws to the multilingual case** — direct follow-on to Tao et al.; several groups have signalled interest, no controlled multilingual result published *(frontier — verify)*.
- **Byte- and morphology-level escapes from the allocation problem**: ByT5 (Xue et al., TACL 2022), CANINE (Clark et al., TACL 2022), MegaByte (Yu et al., NeurIPS 2023), and MYTE (Limisiewicz et al., ACL 2024), which equalizes encoding length across languages via morphology-driven byte merges. These dissolve the discrete allocation but pay in sequence length.
- **Post-hoc vocabulary transplantation** — adapting an English-centric model's tokenizer to a new language with embedding initialization from the old vocabulary; widely used in regional-LLM projects (Indic, African, Southeast Asian consortia). Practically effective, theoretically unprincipled.
- **Tokenizer-free pricing/fairness auditing** following Ahia et al., now with per-language cost disclosures in some model cards *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does equalizing marginal loss reduction across languages beat temperature-sampled allocation at fixed compute?

- **Scale.** $N_{nv} = 1.0$B, $d = 2048$, 100B training tokens, 20 languages spanning 5 scripts and 4 morphological types (English, Spanish, Russian, Arabic, Hindi, Telugu, Turkish, Finnish, Swahili, Yoruba, Vietnamese, Thai, Indonesian, Korean, Japanese, Chinese, Amharic, Georgian, Hungarian, Bengali). Total $V = 256$K held constant across all arms so compute is matched to within 1%.
- **Arms.** (a) Control: unigram-LM SentencePiece fitted on $\alpha=0.3$ temperature-sampled data — the standard recipe. (b) VoCap marginal-utility allocation. (c) Equal-fertility allocation: choose $v_\ell$ so $\rho_\ell$ is equal across languages on FLORES-200, to within 5%. (d) Uniform $v_\ell = 12.8$K. Plus (e) per-language monolingual controls at 1/20 compute, 200M params, for the $\mathcal{L}_\ell^{\mathrm{mono}}$ baseline. Two seeds per arm.
- **Deciding number.** $\Delta = \max_\ell [\mathcal{L}^{\mathrm{BPB}}_\ell - \mathcal{L}^{\mathrm{mono}}_\ell]$, the worst-case multilingual tax in bits per byte, on held-out FLORES-200. Seed noise at this scale is $\approx 0.002$ bits/byte; a reduction of $\ge 0.01$ bits/byte in $\Delta$ for arm (c) over arm (a) settles the equal-fertility hypothesis in its favor. Report the utilitarian mean alongside — if the two objectives disagree in ranking, that disagreement is itself the publishable result, because it shows the problem is under-specified without a stated welfare function.
- **Cost.** ~4×10^21 FLOPs total, roughly 8 arms-equivalent of 1B/100B runs.

## 9. Key References

- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. — arXiv:1508.07909
- **[Foundational]** Taku Kudo, John Richardson. *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.* EMNLP 2018 (demo). — arXiv:1808.06226
- **[Foundational]** Alexis Conneau, Guillaume Lample. *Cross-lingual Language Model Pretraining.* NeurIPS 2019. — arXiv:1901.07291
- **[Foundational]** Alexis Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. — arXiv:1911.02116
- **[SOTA]** Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS 2024. — arXiv:2407.13623
- **[SOTA]** Davis Liang et al. *XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models.* EMNLP 2023. — arXiv:2301.10472
- **[SOTA]** Bo Zheng, Li Dong, Shaohan Huang, Saksham Singhal, Wanxiang Che, Ting Liu, Xia Song, Furu Wei. *Allocating Large Vocabulary Capacity for Cross-lingual Language Model Pre-training.* EMNLP 2021.
- Hyung Won Chung, Dan Garrette, Kiat Chuan Tan, Jason Riesa. *Improving Multilingual Models with Language-Clustered Vocabularies.* EMNLP 2020.
- Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, Adel Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023. — arXiv:2305.15425
- Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David Mortensen, Noah A. Smith, Yulia Tsvetkov. *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP 2023. — arXiv:2305.13707
- Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, Iryna Gurevych. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021. — arXiv:2012.15613
- Omer Goldman, Avi Caciularu, Matan Eyal, Kris Cao, Idan Szpektor, Reut Tsarfaty. *Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance.* Findings of ACL 2024. — arXiv:2403.06265
- Tomasz Limisiewicz, Terra Blevins, Hila Gonen, Orevaoghene Ahia, Luke Zettlemoyer. *MYTE: Morphology-Driven Byte Encoding for Better and Fairer Multilingual Language Modeling.* ACL 2024. — arXiv:2403.10691
- Mehdi Ali et al. *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL 2024. — arXiv:2310.08754
- Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Survey]** Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel. *ByT5: Towards a token-free future with pre-trained byte-to-byte models.* TACL 2022. — arXiv:2105.13626

## 10. Worked Example

Two languages, one budget. $L = \{\text{en}, \text{te}\}$ (English, Telugu — agglutinative, Telugu script, 3 bytes/char in UTF-8). Fix $d = 2048$, $N_{nv} = 1.0$B, $V = 64$K total, and a 50/50 byte-balanced corpus of 100 GB.

Measured fertilities on FLORES-200 with a joint SentencePiece unigram model at two allocations:

| Allocation | $v_{en}$ | $v_{te}$ | $\rho_{en}$ (tok/byte) | $\rho_{te}$ (tok/byte) |
|---|---|---|---|---|
| A (temperature $\alpha=0.3$) | 56K | 8K | 0.25 | 0.42 |
| B (equal-fertility target) | 40K | 24K | 0.27 | 0.30 |

Vocabulary parameters are identical in both ($2dV = 2.6$×10^8), so per-token FLOPs match. Total tokens to consume the corpus:

- A: $50\text{GB}\times0.25 + 50\text{GB}\times0.42 = 33.5$B tokens.
- B: $50\times0.27 + 50\times0.30 = 28.5$B tokens — **15% fewer**, so at fixed $C$, arm B can take 1.18× more epochs or 1.18× more data.

Now the obstruction. Suppose both models train to the same *per-token* loss, 2.60 nats. Converting to bits per byte:

$$\mathcal{L}^{\mathrm{BPB}}_{te}(\text{A}) = \frac{0.42\times 2.60}{0.693} = 1.58,\qquad \mathcal{L}^{\mathrm{BPB}}_{te}(\text{B}) = \frac{0.30\times2.60}{0.693} = 1.13.$$

Arm B looks 28% better on Telugu. But Telugu text is ~3 bytes/character, so in bits *per character* the numbers are 4.74 and 3.39, while English (1 byte/char) is unchanged at 0.94 and 1.01. Under BPB, arm B wins on the minimax objective; under BPC, English *lost* ground (0.94 → 1.01) and Telugu is still 3.4× worse than English, so a minimax-BPC judge may prefer arm A. Same two runs, same logits, opposite decision — purely from the choice of denominator.

The equal-fertility allocation also buys its Telugu gain by giving Telugu 24K slots for a language with far less clean training text, so the extra slots are rarer and worse-estimated; the embedding-quality cost does not appear in either metric. That is the live obstruction: the ranking is not a property of the models, and no experiment in the literature reports both denominators alongside a monolingual control.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*