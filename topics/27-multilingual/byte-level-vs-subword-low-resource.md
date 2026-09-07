---
id: 27-multilingual/byte-level-vs-subword-low-resource
title: "Byte-Level Models Versus Subword Models for Low-Resource Scripts"
topic: 27-multilingual
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Byte-Level Models Versus Subword Models for Low-Resource Scripts

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/byte-level-vs-subword-low-resource` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed training-compute budget $C$ and a multilingual corpus dominated by high-resource languages, decide whether tokenizer-free (byte- or character-level) modeling beats subword modeling for languages whose scripts are poorly covered by the subword vocabulary — Ge'ez, Tibetan, N'Ko, Cherokee, Adlam, Khmer, Sinhala, and the long tail of Latin-script languages with heavy diacritics and rich morphology.

Three variants, routinely conflated:

- **Measurement.** Is there a comparison metric under which byte and subword models are commensurable at all? Per-token perplexity is not comparable across tokenizers; per-byte or per-character negative log-likelihood is, but does not translate linearly into downstream quality.
- **Method.** At matched *training FLOPs* and matched *inference latency*, does a byte-level (or dynamic-patch) architecture achieve lower per-character loss and higher downstream task score on low-resource scripts than a subword baseline whose vocabulary was allocated with the same information?
- **Theory.** Is the observed byte-model advantage on rare scripts an artifact of vocabulary allocation — recoverable by any subword model with a better-balanced vocabulary — or an irreducible consequence of the discrete vocabulary bottleneck?

Solved means: a scaling-law-backed statement of the form "for languages with corpus share below $\rho^\*$ and script coverage below $\kappa^\*$, byte-level modeling dominates subword modeling at compute budgets above $C^\*$", with the crossover constants estimated and the ablation isolating vocabulary allocation from architecture.

## 2. Formal Setting

Let $\mathcal{L}$ be a language set, $\ell \in \mathcal{L}$ with corpus share $\rho_\ell = N_\ell / \sum_{\ell'} N_{\ell'}$ measured in **UTF-8 bytes**, not documents or tokens (documents differ in length by script; token counts are tokenizer-dependent and therefore circular).

A tokenizer $T$ maps a byte string $b \in \{0,\dots,255\}^*$ to $T(b) \in V^*$. Define **fertility**

$$f_\ell(T) = \frac{\mathbb{E}_{b \sim D_\ell}[\,|T(b)|\,]}{\mathbb{E}_{b\sim D_\ell}[\,|b|_{\text{char}}\,]},$$

tokens per character, measured on a held-out in-language corpus (FLORES-200 devtest is the usual choice). Define **script coverage** $\kappa_\ell(T)$ as the fraction of in-language characters that appear in at least one multi-character vocabulary item; $\kappa_\ell \to 0$ is the byte-fallback regime, where the subword model degenerates to a byte model with a worse positional budget.

The only cross-tokenizer-comparable loss is **bits per character**:

$$\mathrm{BPC}_\ell(\theta) = \frac{1}{\log 2}\cdot\frac{\mathbb{E}_{b\sim D_\ell}\!\left[-\log p_\theta(T(b))\right]}{\mathbb{E}_{b\sim D_\ell}[\,|b|_{\text{char}}\,]},$$

where the numerator sums log-probabilities over *whatever* units the model emits. This requires the tokenizer to be lossless and deterministic — violated by tokenizers with unknown-token replacement or NFKC normalization that is not invertible.

Compute is matched in two distinct senses, which diverge for byte models:

$$C_{\text{train}} \approx 6\,P\,N_{\text{tok}}, \qquad C_{\text{inf}}(\ell) \approx 2P \cdot f_\ell(T)\cdot |b|_{\text{char}} .$$

A byte model has $f_\ell \approx 1$ token/char for Latin but $\approx 3$ for Ge'ez or Devanagari (UTF-8 uses 3 bytes for most non-Latin BMP characters), so matched training FLOPs and matched inference latency cannot both hold. **Assumptions known to be violated in practice:** (i) that $\rho_\ell$ is known — web-corpus language ID has error rates above 10% on low-resource languages; (ii) that held-out in-language test data is not machine-translated or LLM-generated — FLORES is human-translated, but most crawled low-resource evaluation data is not; (iii) that BPC comparisons hold under different training-data mixtures — they do not, and mixture is nearly always confounded with tokenizer in published comparisons.

## 3. State of the Art

**Established.** ByT5 (Xue et al., TACL 2022) is the reference controlled comparison: same architecture family and same pretraining corpus (mC4) as mT5, with byte inputs and parameters reallocated from the embedding table to the encoder. It is established that ByT5 matches or beats mT5 at comparable parameter counts on generative and noise-perturbed tasks, and is markedly worse in wall-clock efficiency on long non-Latin inputs. CANINE (Clark et al., TACL 2022) established that character-level input with downsampling beats mBERT on TyDi QA at fewer parameters. Charformer (Tay et al., ICLR 2022) established that learned subword segmentation (GBST) recovers most of the byte advantage at lower cost.

**Claimed but unablated.** Byte Latent Transformer (Pagnoni et al., 2024; ACL 2025) reports entropy-based dynamic patching matching Llama-3-class subword baselines at 8B parameters with large inference-FLOP savings and improved robustness on character-level and low-resource-translation tasks — but the low-resource claims are aggregate benchmark numbers, not per-language ablations with tokenizer allocation held fixed. MYTE (Limisiewicz et al., NAACL 2024) reports morphology-driven byte encoding that equalizes encoded-sequence length across languages and lowers perplexity disparity; the downstream-task transfer is reported on a small task set. MegaByte (Yu et al., NeurIPS 2023) and MambaByte (Wang et al., 2024) establish that byte modeling scales at all via patching / state-space recurrence, but neither ran a multilingual low-resource arm.

**Benchmark-number-only.** Nearly every claim of the form "byte models are better for low-resource languages" traces to a single aggregate score on FLORES-200, XTREME-UP, or Belebele, without a matched-compute control.

## 4. What Is Known

- **Fertility disparity is large and measured.** Petrov et al. (NeurIPS 2023) measured tokenized-length ratios of up to roughly $15\times$ between the best- and worst-served languages for widely used commercial tokenizers, on parallel text. Ahia et al. (EMNLP 2023) showed the same disparity converts directly into API cost and context-window disadvantage.
- **Tokenizer choice matters but is not dominant at scale.** Ali et al. (NAACL Findings 2024) trained 2.6B-parameter models with varied tokenizers and found downstream differences of a few points — non-negligible for multilingual settings, far from the order-of-magnitude framing.
- **Subword quality predicts downstream quality.** Rust et al. (ACL 2021) showed that a dedicated monolingual tokenizer recovers much of the gap between mBERT and monolingual BERT, at 110M-parameter scale — evidence that some "byte advantage" is really "vocabulary-allocation disadvantage".
- **Byte models are robust to noise.** ByT5 at Base/Large scale degrades far less than mT5 under character-level corruption — the most reproducible byte-vs-subword regularity.
- **Byte sequences are longer where it hurts most.** For Amharic (Ge'ez script), UTF-8 costs 3 bytes/character, so a byte model needs roughly $3\times$ the context length of a character model and often $6$–$10\times$ that of a well-fitted subword model.

## 5. What Is Not Known

- **Empirically open (primary).** No published experiment compares byte-level and subword models at matched training FLOPs *and* matched inference FLOPs *and* matched data mixture, with per-language reporting for $\rho_\ell < 10^{-4}$, at a scale above ~1B parameters. Runnable today; unrun.
- **Empirically open.** Whether the byte advantage is a low-compute phenomenon that closes as $C$ grows, or a persistent offset. No byte-vs-subword scaling law with a language-share covariate exists.
- **Theoretically open.** No result establishing whether an optimal subword vocabulary of size $|V|$ can match a byte model's per-character loss on a language of corpus share $\rho$; no lower bound on the loss cost of the discrete vocabulary bottleneck.
- **Methodologically blocked.** Cross-lingual downstream comparability. FLORES-200 chrF++ and Belebele accuracy are not calibrated to equal difficulty across languages, so a per-language delta of $+1.5$ chrF++ in Tigrinya and $+1.5$ in Sinhala are not the same evidence.

## 6. Why It Is Hard

The core obstruction is **confounded measurement compounded by a non-identifiable control arm**. Any byte-vs-subword comparison varies at least four things at once: vocabulary allocation, effective sequence length per document, tokens-per-parameter data ratio, and the fraction of compute in embeddings versus layers. The natural fix — hold the subword vocabulary fixed and swap architectures — is exactly what makes the comparison uninformative, because the subword arm's low-resource performance is a function of a vocabulary-fitting choice that has its own optimum.

Second obstruction: **cost asymmetry defeats matched-compute design**. Matching training FLOPs gives the byte model fewer effective documents on non-Latin scripts; matching document counts gives it more FLOPs. There is no neutral point, only a Pareto frontier, and almost no published work reports both axes.

Third: **absent ground truth for the tail**. For languages with a few million bytes of clean text, held-out sets are small enough that a chrF++ difference of 2 points is inside the resampling interval, and much crawled data is machine-translated, which inflates subword models trained on similar MT output.

## 7. Current Research (as of 2026)

- **Dynamic patching.** BLT-style entropy-driven patching (Meta AI) is the main line: keep byte inputs, restore subword-like compute cost. Multilingual extensions with per-language patch-rate analysis are in progress *(frontier — verify)*.
- **Equitable encodings.** MYTE and successors (Charles University / JHU line, Limisiewicz and collaborators) target length parity across languages as the primary objective rather than downstream score.
- **Tokenizer transplantation / vocabulary expansion.** Adapting a high-resource model's embedding matrix to a new script is currently the cheapest practical alternative and the strongest control arm the byte camp has to beat.
- **State-space and linear-attention byte models.** MambaByte-style recurrence removes the quadratic penalty of long byte sequences; whether it holds up on morphologically rich low-resource text at scale is untested *(frontier — verify)*.
- **Cost-fairness auditing.** Follow-ups to Ahia et al. on per-language inference cost as a deployment-equity metric.

## 8. Concrete Next Experiment

**Scale.** Train four 1.4B-parameter decoder models on an identical 300B-byte multilingual mixture (fixed by *bytes* per language, not documents), covering 40 languages including 12 with $\rho_\ell < 10^{-4}$ spanning five scripts (Ge'ez, Sinhala, Khmer, Tibetan, Adlam). Roughly $2.5\times10^{21}$ FLOPs per arm; ~4k H100-hours each.

**Arms.**
1. Subword, 256k SentencePiece vocabulary fit with the *default* high-resource-weighted sampling ($\alpha = 0.3$).
2. **Control arm:** subword, same 256k vocabulary size, fit with *byte-balanced* sampling so that fertility $f_\ell$ has coefficient of variation $< 0.15$ across the 40 languages. This is the arm that separates architecture from allocation, and it is the arm the literature omits.
3. Pure byte, matched training FLOPs (fewer bytes seen).
4. Dynamic-patch byte (BLT-style), matched training FLOPs.

**Deciding number.** For each low-resource language, report $\Delta\mathrm{BPC}_\ell = \mathrm{BPC}_\ell(\text{arm } k) - \mathrm{BPC}_\ell(\text{arm 2})$ on FLORES-200 devtest, at equal inference FLOPs per character. **The question is decided by the sign and magnitude of the mean $\Delta\mathrm{BPC}$ over the 12 low-resource languages for arms 3 and 4 against arm 2, with a bootstrap 95% interval.** If $\Delta\mathrm{BPC} \geq -0.02$ bits/char (byte models fail to beat a fairly-allocated subword model by more than 2 hundredths of a bit), the byte advantage is an allocation artifact and the practical recommendation is better vocabulary fitting. If $\Delta\mathrm{BPC} \leq -0.10$ bits/char, the vocabulary bottleneck is real and byte-level modeling is the right target for tail scripts.

## 9. Key References

- **[Foundational]** Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL, 2022. — arXiv:2105.13626
- **[Foundational]** Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting. *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL, 2022. — arXiv:2103.06874
- **[Foundational]** Yi Tay, Vinh Q. Tran, Sebastian Ruder, Jai Gupta, Hyung Won Chung, Dara Bahri, Zhen Qin, Simon Baumgartner, Cong Yu, Donald Metzler. *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* ICLR, 2022. — arXiv:2106.12672
- **[SOTA]** Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer. *Byte Latent Transformer: Patches Scale Better Than Tokens.* ACL, 2025. — arXiv:2412.09871
- **[SOTA]** Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS, 2023. — arXiv:2305.07185
- **[SOTA]** Tomasz Limisiewicz, Terra Blevins, Hila Gonen, Orevaoghene Ahia, Luke Zettlemoyer. *MYTE: Morphology-Driven Byte Encoding for Better and Fairer Multilingual Language Modeling.* NAACL, 2024. — arXiv:2403.10691
- **[Evidence]** Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, Adel Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS, 2023. — arXiv:2305.15425
- **[Evidence]** Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R. Mortensen, Noah A. Smith, Yulia Tsvetkov. *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP, 2023. — arXiv:2305.13707
- **[Evidence]** Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, Iryna Gurevych. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL, 2021. — arXiv:2012.15613
- **[Evidence]** Mehdi Ali et al. *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL, 2024. — arXiv:2310.08754
- **[Survey]** Sabrina J. Mielke, Zaid Alyafeai, Elizabeth Salesky, Colin Raffel, Manan Dey, Matthias Gallé, Arun Raja, Chenglei Si, Wilson Y. Lee, Benoît Sagot, Samson Tan. *Between Words and Characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP.* Preprint, 2021. — arXiv:2112.10508
- **[Benchmark]** NLLB Team et al. *No Language Left Behind: Scaling Human-Centered Machine Translation.* Nature, 2024 (FLORES-200). — arXiv:2207.04672

## 10. Worked Example

Take Amharic, one sentence of 100 characters, all Ge'ez (3 UTF-8 bytes each): 300 bytes.

| Arm | Units for the sentence | Fertility (units/char) |
|---|---|---|
| Byte model | 300 | 3.00 |
| Subword, high-resource-weighted 256k vocab, $\kappa_{am}\approx 0$ (byte fallback) | ~300 | ~3.00 |
| Subword, byte-balanced 256k vocab | ~30 | ~0.30 |

Suppose the byte arm reaches per-byte cross-entropy $0.62$ nats. Per character: $0.62 \times 3 / \ln 2 = 2.68$ bits/char. Suppose the balanced-vocabulary subword arm reaches $4.20$ nats/token at $0.30$ tokens/char: $4.20 \times 0.30 / \ln 2 = 1.82$ bits/char. On this hypothetical the subword arm wins by $0.86$ bits/char — but it also consumed $10\times$ fewer forward positions for the same text, so at matched *inference* FLOPs it could have been $10\times$ larger, and at matched *training* FLOPs it saw $10\times$ more Amharic text.

That is the obstruction in one table. The two arms differ in loss, in sequence length, in effective data seen, and in parameters-per-position — and no published low-resource comparison holds more than two of the four fixed. The default subword arm (row 2) is the one everyone actually benchmarks against, and it is degenerate: with $\kappa_{am} \approx 0$ it *is* a byte model, just one whose byte fallback wastes the embedding table. Beating it proves nothing about the vocabulary bottleneck.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*