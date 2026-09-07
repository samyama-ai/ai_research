---
id: 27-multilingual/compute-optimal-multilingual-vocabulary-allocation
title: "Compute-Optimal Vocabulary Allocation Across Unequal-Resource Languages"
topic: 27-multilingual
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Vocabulary Allocation Across Unequal-Resource Languages

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/compute-optimal-multilingual-vocabulary-allocation` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed training compute budget $C$ and a multilingual corpus covering languages $\ell = 1 \dots L$ with wildly unequal token counts (Common Crawl gives English $10^4$–$10^6\times$ the text of Yoruba or Quechua), choose a subword vocabulary of size $V$ **and** its allocation $\{V_\ell\}$ across languages so as to minimise a stated multilingual objective. The decision has two coupled parts: the global size $V$, which trades embedding/softmax parameters against sequence length, and the per-language share, which trades one language's fertility (tokens per word) against another's.

Three variants, of very different difficulty:

- **Measurement.** Is there a per-language quantity — fertility, compression rate, Rényi efficiency — that predicts downstream loss well enough to be optimised as a proxy? Currently contested.
- **Method.** Given a target allocation, produce a tokenizer that hits it. Largely solved (cluster-and-merge, capacity allocation, byte-level fallbacks).
- **Theory.** Does a compute-optimal $\{V, V_\ell\}$ exist as a function of $(C, L, \{D_\ell\})$, in the way Chinchilla gives $(N^\*, D^\*)$? No derivation exists that handles unequal $D_\ell$.

A solution is a rule $\{V_\ell^\*\} = f(C, \{D_\ell\}, \{\text{script}, \text{morphology}\})$ that beats proportional and uniform allocation, at matched FLOPs, on held-out per-language loss **normalised to a tokenization-invariant unit**.

## 2. Formal Setting

Let $N_{nv}$ be non-vocabulary parameters, $d$ the model width, $V$ vocabulary size. Vocabulary parameters are $N_v = 2Vd$ (tied input/output embeddings: $Vd$). Total $N = N_{nv} + N_v$.

**Fertility.** For language $\ell$ with tokenizer $T$, measured on a held-out corpus of $W_\ell$ whitespace/morpheme-delimited words yielding $t_\ell$ subwords:
$$\phi_\ell(T) = t_\ell / W_\ell .$$
Measured, not assumed; for scriptio continua (Chinese, Thai, Japanese) $W_\ell$ requires an external segmenter, and the choice of segmenter moves $\phi_\ell$ by tens of percent. This is the first place the measurement leaks.

**Compute.** With $D_\ell^{\text{word}}$ words of language $\ell$ in the corpus, training FLOPs are
$$C \approx 6\,(N_{nv} + 2Vd) \sum_\ell \phi_\ell D_\ell^{\text{word}} .$$
Raising $V$ lowers every $\phi_\ell$ (sublinearly, Heaps-law-like) but raises $N_v$ linearly. The optimum is interior.

**Objective.** Per-language cross-entropy in nats **per byte**, not per token, since per-token loss is not comparable across tokenizers:
$$\mathcal{L}_\ell^{\text{byte}} = \frac{1}{B_\ell}\sum_{i=1}^{t_\ell} -\log p_\theta(x_i \mid x_{<i}),$$
with $B_\ell$ the UTF-8 byte count of the same held-out text. Aggregate with a fairness weight $\alpha$:
$$\mathcal{J}_\alpha = \Big(\sum_\ell w_\ell (\mathcal{L}_\ell^{\text{byte}})^{\alpha}\Big)^{1/\alpha},$$
$\alpha = 1$ giving utilitarian mean, $\alpha \to \infty$ giving worst-language (Rawlsian) loss. **The choice of $\alpha$ is a policy decision, not an empirical one, and the literature almost never states it.**

**Allocation.** $V_\ell = |\{v : v \text{ assigned to } \ell\}|$ with $\sum_\ell V_\ell \geq V$ (inequality because scripts share tokens). Standard practice sets it implicitly by temperature sampling $q_\ell \propto p_\ell^{\alpha_s}$, $\alpha_s \approx 0.3$ (XLM-R), then training one BPE model on the resampled mixture.

**Assumptions known to be violated.** (i) Bytes-per-nat is script-neutral — false: UTF-8 charges Latin 1 byte and Devanagari 3, so $\mathcal{L}^{\text{byte}}$ silently rewards non-Latin scripts. (ii) Fertility is monotone in downstream quality — Goldman et al. (2024) show compression correlates with generation but weakly with classification. (iii) Languages are separable — shared subwords make $V_\ell$ non-identifiable at the boundary. (iv) Chinchilla-style $N \!\propto\! D$ holds per language — untested when $D_\ell$ spans six orders of magnitude.

## 3. State of the Art

**Theory SOTA.** Tao et al., *Scaling Laws with Vocabulary* (NeurIPS 2024), give the only compute-optimal vocabulary law: fit via IsoFLOP, derivative-of-loss, and a parametric fit, all three agreeing that optimal $V$ grows as a power of non-vocabulary parameters with exponent $\approx 0.83 < 1$ — vocabulary should grow slower than the rest of the model. **Monolingual English only.** No multilingual term.

**Empirical SOTA, allocation.** Zheng et al., VoCap (EMNLP 2021), allocate vocabulary capacity per language by a marginal-utility criterion rather than proportionally. Chung et al. (EMNLP 2020) cluster languages, build per-cluster vocabularies, and union them. Liang et al., XLM-V (EMNLP 2023), scale to a 1M-token vocabulary and report XNLI gains over XLM-R (250K), largest on low-resource languages. **Established as benchmark numbers; not ablated against a FLOP-matched larger-$N_{nv}$ control**, so how much of the gain is vocabulary allocation versus extra parameters is unresolved.

**Claimed but unablated.** That larger multilingual vocabularies "fix" low-resource languages. XLM-V's gains are reported at fixed architecture, not fixed compute. Over-tokenized Transformer work (Huang et al., 2025) claims decoupling input and output vocabulary sizes yields a log-linear input-vocabulary scaling benefit; the multilingual case is untested.

**Method SOTA.** MYTE (Limisiewicz et al., ACL 2024) uses morphology-driven byte encodings to equalise encoded length across 99 languages — the strongest existing attack on fertility disparity, evaluated at up to ~1B parameters.

## 4. What Is Known

- **Fertility disparity is large and measured.** Petrov et al. (NeurIPS 2023): across 17 languages and commercial tokenizers, tokenized length ratios reach ~15× between best- and worst-served languages. Ahia et al. (EMNLP 2023): up to ~5× API cost differences for semantically equivalent content, measured on GPT-3.5-class tokenizers.
- **Vocabulary is under-allocated in practice.** Tao et al. estimate Llama-2-70B's 32K vocabulary should be $\geq$ 216K at its compute; at 3B non-vocab parameters they show a predicted-optimal vocabulary (~35.8K vs 32K) improving ARC-Challenge accuracy at identical FLOPs. Scale: 33M–3B parameters, English.
- **Tokenizer choice materially changes downstream scores.** Ali et al. (NAACL 2024) train 2.6B-parameter multilingual models and find multilingual tokenizers reduce fertility and improve downstream tasks versus English-centric ones, at matched training setup.
- **Compression is not a sufficient proxy.** Schmidt et al. (EMNLP 2024) and Goldman et al. (ACL Findings 2024) both find that better compression does not reliably imply better downstream performance; the correlation is task-dependent.
- **Morphological complexity confound is partly explained away.** Arnett & Bergen (COLING 2025) argue the apparent penalty for morphologically complex languages is largely a dataset-size effect, not an intrinsic morphology effect.

## 5. What Is Not Known

- **Theoretically open.** No scaling law of the form $V_\ell^\* = f(C, D_\ell)$. Tao et al.'s exponent 0.83 is derived under a single language distribution; whether it holds per-language, or whether the correct object is a single $V$ with a per-language allocation obeying a different exponent, is unproven either way.
- **Empirically open.** The FLOP-matched sweep — hold $C$ fixed, vary $V \in \{32\text{K},\dots,512\text{K}\}$ and allocation temperature $\alpha_s \in \{0, 0.3, 0.7, 1\}$, measure per-language bits-per-byte — has never been run at $\geq$ 1B parameters across $\geq$ 30 languages. Runnable today for roughly $10^{22}$ FLOPs total.
- **Methodologically blocked.** The objective itself. There is no agreed tokenization-invariant, script-neutral per-language loss unit. Bits-per-byte is script-biased; bits-per-character is grapheme-cluster dependent; downstream benchmarks (XNLI, FLORES) are translationese and cover the wrong tail of languages.

## 6. Why It Is Hard

**Confounded measurement, compounded by non-identifiability.** Changing $V$ changes three things at once: parameter count, sequence length (hence tokens seen per FLOP), and the loss unit. A "win" for a larger vocabulary can be any of: more parameters, more effective epochs over the same bytes, or an easier prediction problem per token. Only a bits-per-byte metric at matched FLOPs disentangles them, and bits-per-byte is itself script-biased.

The allocation is separately non-identifiable: subwords are shared across languages that share scripts, so $V_\ell$ has no unique definition. Assigning a shared token to Spanish or Portuguese is a convention, and the two conventions give different "allocations" for the identical tokenizer. Any claim of the form "language $\ell$ received $V_\ell$ slots" is convention-dependent unless the paper states the attribution rule — most do not.

## 7. Current Research (as of 2026)

- Extending vocabulary scaling laws to multilingual mixtures — follow-on to Tao et al. (Sea AI Lab / NUS lineage) *(frontier — verify)*.
- Byte- and patch-level models that sidestep the allocation problem: MegaByte, MambaByte, and Byte Latent Transformer (Pagnoni et al., Meta, 2024), whose dynamic entropy-based patching makes effective vocabulary per language emergent rather than chosen. Whether this equalises across languages is untested at scale *(frontier — verify)*.
- Morphology-aligned encodings (MYTE line, Charles University / JHU).
- Tokenizer transplantation and vocabulary expansion for low-resource adaptation (continued pretraining with added tokens); large industrial adoption (128K–256K vocabularies in Llama-3, Gemma, Qwen) with no published FLOP-matched multilingual ablation.

## 8. Concrete Next Experiment

**Scale.** $N_{nv} = 1.2$B non-vocabulary parameters, $C = 1.2\times10^{22}$ FLOPs per run (roughly 25B–40B tokens depending on fertility), 40 languages spanning $10^{10}$ to $10^{6}$ bytes of available data, 4 scripts.

**Arms.** $V \in \{32\text{K}, 64\text{K}, 128\text{K}, 256\text{K}\}$ crossed with sampling temperature $\alpha_s \in \{0.3, 0.7\}$ — 8 runs. Crucially, **FLOPs are held constant by reducing training tokens as $V$ grows**, so larger vocabularies pay for themselves.

**Control arm.** $V = 32$K with the *saved* embedding parameters reinvested in depth/width, matched to total FLOPs. This is the arm that XLM-V never ran, and it is the one that decides whether vocabulary allocation, or merely parameter count, produces the low-resource gains.

**Deciding number.** The worst-language bits-per-byte, $\max_\ell \mathcal{L}_\ell^{\text{byte}}$, measured on held-out native (non-translated) text, reported alongside the mean. If the best $V$ under worst-language loss differs from the best $V$ under mean loss by more than one grid step, allocation is a genuine policy lever and the Chinchilla-style single-optimum framing is wrong. If they coincide, a single global $V$ suffices and per-language allocation is a distraction.

## 9. Key References

- **[Foundational]** Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzmán, Grave, Ott, Zettlemoyer, Stoyanov. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. — arXiv:1911.02116
- **[SOTA / theory]** Tao, Liu, Zhu, Zhang, Yang, Lin, Kan. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS 2024. — arXiv:2407.13623
- **[SOTA / allocation]** Zheng, Dong, Huang, Wang, Chi, Wei, Wang, Ma, Wei. *Allocating Large Vocabulary Capacity for Cross-lingual Language Model Pre-training.* EMNLP 2021. — arXiv:2109.07306
- **[SOTA / scale]** Liang, Bhosale, Artetxe, Li, Goyal, Mihaylov, Ott, Shleifer, Kalyan, Sridhar, Wang, Zettlemoyer. *XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models.* EMNLP 2023. — arXiv:2301.10472
- **[Measurement]** Petrov, La Malfa, Torr, Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023. — arXiv:2305.15425
- **[Measurement]** Ahia, Kumar, Gonen, Kasai, Mortensen, Smith, Tsvetkov. *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP 2023. — arXiv:2305.13707
- **[Method]** Limisiewicz, Blevins, Gonen, Ahia, Zettlemoyer. *MYTE: Morphology-Driven Byte Encoding for Better and Fairer Multilingual Language Modeling.* ACL 2024. — arXiv:2403.10691
- **[Ablation]** Ali, Fischer, Chelli, et al. *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL 2024. — arXiv:2310.08754
- **[Caution]** Goldman, Kaddour, Schmidt, et al. *Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance.* Findings of ACL 2024.
- **[Foundational]** Chung, Garrette, Tan, Riesa. *Improving Multilingual Models with Language-Clustered Vocabularies.* EMNLP 2020.
- **[Context]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556

## 10. Worked Example

Two languages, English ($\ell{=}1$) and Sinhala ($\ell{=}2$), $d = 2048$, $N_{nv} = 1.2\times10^9$, both evaluated on the same held-out content.

Take a 32K joint BPE vocabulary. Measured fertility: $\phi_1 \approx 1.3$ subwords/word, $\phi_2 \approx 6.5$ — a 5× gap, consistent with the ranges in Petrov et al. Growing $V$ to 128K and biasing merges toward Sinhala takes $\phi_2$ to about 3.0 while $\phi_1$ drifts to 1.25.

Cost of that change:
$$\Delta N_v = (128\text{K} - 32\text{K}) \times 2048 \times 2 \approx 3.9\times10^{8}.$$
Total parameters go from $\approx 1.33$B to $\approx 1.72$B, a 29% rise in per-token FLOPs. At fixed $C$, tokens seen drop by 22%.

Sinhala benefits: 54% fewer tokens for the same text, so at fixed $C$ it sees $0.78 / 0.46 \approx 1.7\times$ more Sinhala *bytes*. English pays: 22% fewer tokens for a 4% fertility gain — a net loss of roughly 19% of its byte throughput.

**Where the obstruction appears.** Report per-token loss and Sinhala looks worse: each token now carries more information, so $-\log p$ per token rises even if the model improved. Report bits-per-byte and Sinhala looks better than it is: Sinhala is 3 UTF-8 bytes per character against English's 1, so the same nats are divided by a 3× larger denominator. Two defensible metrics give opposite verdicts on the same pair of runs, and neither is wrong — the units differ. Until a script-neutral, tokenization-invariant loss unit is fixed and the fairness weight $\alpha$ is declared, the compute-optimal allocation is not a well-posed question, only a well-funded one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*