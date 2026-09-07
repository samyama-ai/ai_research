---
id: 01-tokenization/byte-level-models-frontier-scale
title: "Tokenizer-Free Byte Models at Frontier Scale"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Tokenizer-Free Byte Models at Frontier Scale

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/byte-level-models-frontier-scale` · **Status:** empirically-open

## 1. Problem Statement

A tokenizer-free model consumes raw UTF-8 bytes and emits raw UTF-8 bytes. No BPE merge table, no vocabulary file, no fixed segmentation. The question is whether such a model can match or beat a subword-tokenized model of the same **training and inference compute budget** at frontier scale (roughly $\geq 10^{23}$ FLOPs, $\geq 10^{10}$ parameters, $\geq 10^{13}$ training bytes).

Three variants, of very different difficulty:

- **Measurement.** How do you compare a byte model and a token model at all? Cross-entropy per token is not comparable across vocabularies. The only shared unit is bits per byte (BPB) — but BPB does not capture inference cost, and downstream benchmarks confound tokenization with data mixture. *Partially blocked.*
- **Method.** Build a byte architecture whose compute-vs-quality curve does not fall behind BPE as scale grows. Requires dynamic compression of the byte stream, since a flat byte Transformer pays $\approx 4.4\times$ more positions per unit text in English and far more in code and non-Latin scripts. *Empirically open above ~8B parameters.*
- **Theory.** Is there a scaling-law statement — an exponent or an irreducible-loss term — that separates learned-segmentation models from fixed-vocabulary models? *Theoretically open; no separation result exists in either direction.*

**Solved** would mean: a byte model at $\geq 10^{23}$ training FLOPs that, under matched training FLOPs and matched inference FLOPs-per-byte-generated, is no worse than a strong BPE baseline on held-out BPB and on a fixed benchmark suite, with the multilingual/robustness gains that motivate byte models retained.

## 2. Formal Setting

Let $x \in \Sigma^{*}$ with $\Sigma = \{0,\dots,255\}$ be a UTF-8 byte string of length $L_b$. A tokenizer is a map $T: \Sigma^{*} \to V^{*}$ into vocabulary $V$, producing $L_t = |T(x)|$ tokens. Define the **fertility**

$$\rho = \frac{L_b}{L_t} \quad \text{(bytes per token, measured on the actual eval corpus, per language and per domain).}$$

Both model families define $p_\theta(x)$ over byte strings — the token model via $p_\theta(x) = \prod_i p_\theta(t_i \mid t_{<i})$ composed with a deterministic, injective decoder $V^* \to \Sigma^*$. The comparable quantity is

$$\mathrm{BPB}(\theta) = \frac{1}{L_b \ln 2}\sum_i -\ln p_\theta(t_i \mid t_{<i}) = \frac{\mathrm{NLL}_{\text{token}}}{\rho \ln 2}.$$

**Measurement caveat:** this identity holds only if $T$ is injective on the eval text and the model places no mass on strings outside $T(\Sigma^*)$. BPE is not surjective onto $\Sigma^*$ — the token model assigns zero probability to byte strings no tokenization reaches, so its "BPB" is a bound on a restricted support, not a density over $\Sigma^*$. This is a known, routinely ignored violation.

Compute is the second axis. With $N$ non-embedding parameters and $C \approx 6ND$ training FLOPs over $D$ positions (Kaplan et al. 2020; Hoffmann et al. 2022), a flat byte model at fixed text volume has $D_{\text{byte}} = \rho \, D_{\text{token}}$, so matched-quality byte training costs $\rho\times$ more unless the architecture compresses. Hierarchical byte models introduce a learned patcher $\pi_\phi: \Sigma^{*} \to \mathbb{N}^{*}$ giving patch lengths, with realized compression

$$\bar{\rho}_\pi = \mathbb{E}\big[L_b / |\pi_\phi(x)|\big],$$

and the honest cost metric is **FLOPs per byte**, split into local encoder/decoder cost and global backbone cost:

$$F_{\text{byte}} = \underbrace{f_{\text{loc}}}_{\text{per byte}} + \underbrace{f_{\text{glob}}/\bar{\rho}_\pi}_{\text{per patch, amortized}}, \qquad F_{\text{token}} = f_{\text{glob}}'/\rho .$$

Assumptions known to be violated: (i) $\rho$ is treated as a constant, but it varies by $2$–$4\times$ across languages (Petrov et al. 2023) and drops sharply on code, numerals, and rare Unicode; (ii) $C = 6ND$ ignores attention, which matters precisely when sequences are $\rho\times$ longer; (iii) inference cost is assumed proportional to FLOPs, but byte models generate more autoregressive steps, so wall-clock latency is memory-bandwidth-bound, not FLOP-bound.

## 3. State of the Art

**Empirical SOTA (established, flop-controlled).**
- **Byte Latent Transformer** (Pagnoni et al., Meta, 2024/2025): entropy-based dynamic patching, local encoder/decoder around a latent global Transformer. Trained to 8B parameters on 4T bytes; reported flop-matched parity with a Llama-3-tokenizer baseline, with better robustness on character-level and noised-input tasks. This is the largest published controlled comparison.
- **H-Net** (Hwang, Wang, Gu, 2025): end-to-end learned dynamic chunking with a routing module and no external patcher; at ~1.3B scale, one-stage matches a BPE Transformer and two-stage exceeds it on data-scaling curves, with larger gains on Chinese and code, where BPE fertility is worst.
- **EvaByte** (2025, HKU/Allen Institute collaboration): a ~6.5B flat byte model with multibyte prediction, trained on ~1.5T bytes; competitive with contemporaneous token models trained on far more data. Released as an open model.

**Earlier method line (established at small scale).** MegaByte (Yu et al., NeurIPS 2023) — fixed-size patches, ~1.5B; MambaByte (Wang et al., COLM 2024) — SSM backbone, ~350M–1.5B; SpaceByte (Slagle, NeurIPS 2024) — patch boundaries at space-like bytes; Dynamic Token Pooling (Nawrot et al., ACL 2023). ByT5 (Xue et al., TACL 2022), CANINE (Clark et al., TACL 2022), Charformer (Tay et al., ICLR 2022) established byte/character viability but at parameter-matched, not FLOP-matched, settings — ByT5 explicitly trades ~$1.2$–$10\times$ more compute for robustness.

**Claimed but unablated.** BLT's "better scaling trend" rests on a single family fit; the patcher's entropy model is itself a small BPE-free LM whose cost is often excluded from headline FLOP counts. Downstream benchmark wins for byte models are frequently reported without matching data mixture, and the $>7$B results exist as benchmark tables, not as independently reproduced ablations.

**Theory SOTA.** None specific. No scaling-law separation theorem between fixed-vocabulary and learned-segmentation models exists. The nearest formal results are tokenization-as-compression arguments and unigram/BPE optimality analyses, which do not bound end-model loss.

## 4. What Is Known

- **Fertility numbers.** Llama-3-class 128k-vocab BPE achieves $\rho \approx 3.9$–$4.4$ bytes/token on English web text, $\approx 2$–$3$ on code, and substantially lower on Devanagari, Burmese, Amharic. Petrov et al. (NeurIPS 2023) measured up to $15\times$ token-count disparity across languages for the same semantic content in some multilingual tokenizers — a direct inference-cost and context-length tax.
- **Byte models fix character-level failures.** BLT reports large gains on character manipulation and orthographic/noise-perturbation tasks versus a BPE Llama-3 baseline; ByT5 showed the same for noisy input at 300M–13B.
- **Hierarchy is required.** Flat byte Transformers lose decisively under matched FLOPs; every competitive result since 2023 uses patching, pooling, or an SSM backbone.
- **Parity is demonstrated to 8B / 4T bytes**, not beyond. Public frontier models (2024–2026) remain BPE-based.
- **Chinchilla-optimality is not transferable.** The $D \approx 20N$ rule was fit on token counts under a specific tokenizer; the equivalent byte-domain constant has not been re-fit at scale.

## 5. What Is Not Known

- **Empirically open.** Does parity hold at $10^{24}$–$10^{25}$ FLOPs? Whether the byte curve crosses above, below, or parallel to BPE beyond 8B is unmeasured because nobody has spent the compute on a matched pair. This is the central gap.
- **Empirically open.** Inference economics. Byte models make more sequential decode steps; whether multibyte prediction or patch-level decoding recovers tokens/second at serving batch sizes is untested at production scale.
- **Methodologically blocked.** Fair BPB comparison across support mismatch (§2) and fair FLOP accounting for the patcher. There is no agreed protocol; every paper uses its own.
- **Theoretically open.** No proof that learned segmentation can or cannot change the scaling exponent $\alpha$ in $L(C) = L_\infty + aC^{-\alpha}$, versus only the constant $a$. Both are consistent with all published data.

## 6. Why It Is Hard

The obstruction is **compute cost joined to confounded measurement**. A decisive experiment needs two runs at $\geq 10^{24}$ FLOPs — order $10^7$ USD — and only labs that already own a BPE frontier stack can afford it; for them the byte arm is a pure downside bet against a working pipeline. Worse, the cheap proxies mislead: at $<1$B parameters, sequence length is not the binding constraint and byte models look artificially good; benchmark suites are built from token-model-era tasks and under-weight exactly the character-level and low-resource cases where byte models win. So the measurement that is affordable does not answer the question, and the measurement that answers it is not affordable outside three or four organizations.

## 7. Current Research (as of 2026)

- **Meta FAIR** — BLT line; scaling the patcher and integrating with Llama-family training infrastructure.
- **CMU / Cartesia (Gu, Hwang, Wang)** — H-Net dynamic chunking, multi-stage hierarchies, SSM-Transformer hybrids. Most active on end-to-end learned boundaries.
- **EvaByte / open-model groups** — flat byte models with multibyte prediction and efficient attention kernels.
- **Multilingual equity work** — tokenizer fairness measurement feeding byte-model motivation (Petrov et al. and successors).
- *(frontier — verify)* Reports that closed frontier labs run internal byte-model ablations at $\geq 10$B are plausible but unpublished; treat any parity claim above 8B as unverified until a controlled pair appears.

## 8. Concrete Next Experiment

**Scale.** Two runs at $N = 8$B non-embedding parameters, $C \approx 3\times10^{23}$ FLOPs each, identical data: 6T bytes of a fixed, published mixture (web + code + 30% non-English), identical optimizer, identical context in **bytes** (not tokens) — say 32,768 bytes.

**Arms.**
1. *Control:* dense Transformer, 128k-vocab BPE trained on the same mixture.
2. *Treatment:* hierarchical byte model (BLT- or H-Net-style), with the patcher's FLOPs counted in the budget, and its parameters counted in $N$.

Add two $1.5$B calibration points per arm to fit $L(C)$.

**Deciding number.** Held-out **BPB on a byte-identical eval set**, reported as $\Delta = \mathrm{BPB}_{\text{byte}} - \mathrm{BPB}_{\text{BPE}}$ at matched $C$, together with the fitted exponent difference $\alpha_{\text{byte}} - \alpha_{\text{BPE}}$ from the three-point curves. Decision rule: byte models win the scaling argument iff $\alpha_{\text{byte}} - \alpha_{\text{BPE}} > 0$ with the 8B point satisfying $\Delta \leq 0$. If $\Delta \leq 0$ but $\alpha$ is equal within fit error, the honest conclusion is "constant-factor parity, no scaling advantage" — which changes the engineering case entirely. Secondary gate: decoded bytes/second/GPU at batch 64, which must be within $20\%$ of control for the result to matter in deployment.

## 9. Key References

- **[Foundational]** Yu, Simig, Flaherty, Aghajanyan, Zettlemoyer, Lewis. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS, 2023. — arXiv:2305.07185
- **[Foundational]** Xue, Barua, Constant, Al-Rfou, Narang, Kale, Roberts, Raffel. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL, 2022. — arXiv:2105.13626
- **[Foundational]** Clark, Garrette, Turc, Wieting. *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL, 2022. — arXiv:2103.06874
- **[SOTA]** Pagnoni, Pasunuru, Rodriguez, Nguyen, Muller, Li, Zhou, Yu, Weston, Zettlemoyer, Ghosh, Lewis, Holtzman, Iyer. *Byte Latent Transformer: Patches Scale Better Than Tokens.* Meta AI, 2024. — arXiv:2412.09871
- **[SOTA]** Hwang, Wang, Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.* 2025. — arXiv:2507.07955
- **[SOTA]** Wang, Gonzalez, Rush, Kuleshov et al. *MambaByte: Token-free Selective State Space Model.* COLM, 2024. — arXiv:2401.13660
- **[SOTA]** Slagle. *SpaceByte: Towards Deleting Tokenization from Large Language Modeling.* NeurIPS, 2024. — arXiv:2404.14408
- **[Method]** Nawrot, Chorowski, Łańcucki, Ponti. *Efficient Transformers with Dynamic Token Pooling.* ACL, 2023. — arXiv:2211.09761
- **[Method]** Tay, Tran, Ruder, Gupta, Chung, Bahri, Qin, Baumgartner, Yu, Metzler. *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* ICLR, 2022. — arXiv:2106.12672
- **[Measurement]** Petrov, La Malfa, Torr, Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS, 2023. — arXiv:2305.15425
- **[Scaling]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909

## 10. Worked Example

Take 1 GB of English web text. A 128k BPE tokenizer at $\rho = 4.4$ yields $2.27\times10^{8}$ tokens. A flat byte model sees $10^{9}$ positions. At $N=8$B, training FLOPs $\approx 6ND$:

- BPE: $6 \times 8\times10^9 \times 2.27\times10^8 \approx 1.09\times10^{19}$ FLOPs per GB.
- Flat byte: $4.8\times10^{19}$ FLOPs per GB — $4.4\times$ worse before attention.

Now the hierarchical arm, with a patcher achieving $\bar\rho_\pi = 4.5$, a local encoder/decoder of $0.4$B parameters running per byte, and a $7.6$B global backbone running per patch:

$$F_{\text{byte}} = 6(4\times10^{8}) + \frac{6(7.6\times10^{9})}{4.5} \approx 2.4\times10^{9} + 1.01\times10^{10} = 1.25\times10^{10} \text{ FLOPs/byte-of-text},$$
$$F_{\text{BPE}} = \frac{6(8\times10^{9})}{4.4} \approx 1.09\times10^{10}.$$

Byte cost is $15\%$ above BPE on English — recoverable, and this is roughly the regime where parity is reported at 8B.

Now switch the corpus to Hindi. BPE fertility falls to $\rho \approx 1.6$; the patcher, operating on bytes, holds near $\bar\rho_\pi \approx 4.2$. Then $F_{\text{BPE}} \approx 3.0\times10^{10}$ and $F_{\text{byte}} \approx 1.33\times10^{10}$ — the byte model is now $2.3\times$ **cheaper**.

The obstruction is visible here: the sign of the comparison flips with the eval corpus, and it flips using the same two models and the same arithmetic. A single BPB or benchmark number cannot decide the question, because $\rho$ — not the architecture — dominates the result, and $\rho$ is a property of the data mixture chosen by whoever runs the experiment. Any published parity claim that does not report per-language $\rho$ and $\bar\rho_\pi$ alongside BPB is unfalsifiable in the direction that matters.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*