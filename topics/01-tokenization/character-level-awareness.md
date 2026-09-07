---
id: 01-tokenization/character-level-awareness
title: "Character-Level Awareness in Subword Models"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Character-Level Awareness in Subword Models

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/character-level-awareness` · **Status:** open

## 1. Problem Statement

A subword language model never observes characters. Its input is a sequence of vocabulary IDs; the string "strawberry" may arrive as two or three opaque integers. Yet the model is routinely asked to spell, count letters, reverse strings, rhyme, do character-indexed edits, and reason about anagrams, and it does some of these well and others badly. The problem is to say precisely **what character-level information a subword model has, where it lives, and what limits its use**.

Three variants, of very different difficulty:

- **Measurement.** Define a metric of character awareness that separates *representational* access (the character identity is linearly decodable from the token embedding) from *functional* access (the model can condition a generation on it). Existing benchmarks conflate the two with instruction-following and arithmetic ability.
- **Method.** Given a fixed compute budget $C$, produce an architecture or training recipe whose character-level accuracy matches a byte-level model while keeping the sequence-length advantage of subwords. Solving it means: no loss on standard NLU/generation benchmarks, and character-task accuracy within noise of a byte baseline at equal FLOPs.
- **Theory.** Determine whether character composition is *learnable in the limit* from a token-only corpus — i.e. whether the map from token ID to character string is identified by co-occurrence statistics alone, or whether it requires the explicit character-spelling evidence (spelling games, hyphenation, typos, alternate segmentations) that happens to be in web text.

## 2. Formal Setting

Let $\Sigma$ be an alphabet (bytes or Unicode codepoints), $V \subset \Sigma^+$ a subword vocabulary of size $|V|$, and $\tau: \Sigma^* \to V^*$ the deterministic encoder. Write $\mathrm{str}(v) \in \Sigma^+$ for the surface form of $v \in V$, and $\ell(v) = |\mathrm{str}(v)|$.

**Fertility** — the measured cost of a tokenizer, computed on a held-out corpus $D$:
$$F(\tau, D) = \frac{\sum_{x \in D} |\tau(x)|}{\sum_{x \in D} |x|_{\text{words}}}.$$
Typical measured values: $F \approx 1.3$ for GPT-4o's tokenizer on English, $F \gtrsim 3$ on low-resource scripts; byte-level models have $F$ equal to bytes-per-word ($\approx 5.5$ English).

**Representational access.** Fix a layer $k$ and the model's hidden state $h_k(v) \in \mathbb{R}^d$ for token $v$ in a fixed neutral context. For character $c \in \Sigma$ define the containment probe
$$\hat{y}_c(v) = \sigma(w_c^\top h_k(v) + b_c), \qquad y_c(v) = \mathbb{1}[c \in \mathrm{str}(v)],$$
trained on a split of $V$ and scored by macro-$F_1$ on **held-out token types**. The held-out-type split is the measurement that matters: probes trained and tested on the same tokens measure memorisation of the probe, not of the model. A stronger variant is the positional probe $y_{c,i}(v) = \mathbb{1}[\mathrm{str}(v)_i = c]$, and the strongest is full spelling decode, scored by exact match.

**Functional access.** For a character task $T$ (spell, count, index, delete-$i$-th, reverse), accuracy
$$A_T(M) = \mathbb{E}_{v \sim P_V}\big[\mathbb{1}[M(\text{prompt}_T(v)) = \text{ans}_T(v)]\big],$$
with $P_V$ **stratified by token frequency and by $\ell(v)$** — unstratified averages are dominated by short frequent tokens and overstate awareness.

**The gap of interest** is $G_T = \text{probe-}F_1 - A_T$: information present but unused.

Assumptions, and which fail:
- *$\tau$ is deterministic.* Violated under BPE-dropout / subword regularisation (Kudo 2018; Provilkov et al. 2020), which is exactly the intervention that may inject character evidence.
- *Neutral context isolates the token.* Violated — contextual models' character info is context-dependent; a single "neutral" prompt is a choice, not a fact.
- *Character tasks are answerable at all.* Violated for tokens whose surface form is unreachable through the model's output vocabulary in the requested granularity (glitch tokens; Land & Bartolo 2024).
- *Equal-FLOPs comparison is fair across tokenizers.* Violated: fertility changes both FLOPs per string and the effective number of gradient updates per character.

## 3. State of the Art

**Established.**
- Character identity *is* recoverable from subword embeddings. Itzhak & Levy (NAACL 2022) and Kaushal & Mahowald (NAACL 2022) show linear probes on frozen embeddings (BERT, GPT-J, RoBERTa) reach roughly 85–90% on held-out-type character containment, well above frequency baselines, with accuracy rising with token frequency.
- Byte/character models remove the failure mode by construction. ByT5 (Xue et al., TACL 2022) beats mT5 on noisy and character-manipulation tasks at matched parameters, at a large inference cost (sequences $\sim4\times$ longer, reported up to $\sim$2× slower fine-tuning/inference at comparable size).
- Tokenization causes measurable downstream error in arithmetic: Singh & Strouse (2024) show digit-grouping choices shift multi-digit arithmetic accuracy by tens of points in frontier models — a direct demonstration that the segmentation, not the reasoning, is the binding constraint.

**Claimed but unablated.**
- That dynamic byte patching closes the gap "for free": Byte Latent Transformer (Pagnoni et al., 2024) reports matched scaling to Llama-3 up to 8B with better robustness to noise, but the character-awareness comparison is on a small task set and not stratified by token frequency. SpaceByte (Slagle, NeurIPS 2024) and MambaByte (Wang et al., COLM 2024) report competitive byte-level perplexity; neither ablates character tasks against a subword control at equal FLOPs.
- That "hierarchical/character-aware embeddings fix it": Charformer (Tay et al., ICLR 2022) and CANINE (Clark et al., TACL 2022) predate instruction-tuned character benchmarks; no published run measures $G_T$ for them.

**Benchmark-number-only.** CUTE (Edman, Schmid & Fraser, EMNLP 2024) is the cleanest existing instrument: models are near-ceiling on *containment* ("does 'strawberry' contain 'r'") but drop sharply on *manipulation* (insertion, deletion, substitution, swap) — the reported pattern is high-90s vs. frequently near or below 50% depending on model and subtask. It is a score, not a mechanism: no causal intervention, no probe-vs-behaviour decomposition.

## 4. What Is Known

- **Information is there.** ~85–90% held-out-type containment probe $F_1$ from static embeddings of models in the 100M–6B range (Itzhak & Levy 2022; Kaushal & Mahowald 2022). Scale of measurement: single-token English words, vocabularies of 30k–50k.
- **Use is not.** CUTE-style manipulation accuracy for frontier models (GPT-4-class, Llama-3-class, 2024) is far below containment accuracy on the same tokens — the gap $G_T$ is large and positive. This is the central empirical fact of the problem.
- **Frequency dependence.** Probe and behavioural accuracy both rise with token log-frequency; rare tokens and glitch tokens (Land & Bartolo, EMNLP 2024, "Fishing for Magikarp") can be unspellable and untypable by the model that owns them.
- **Compression is not the objective.** Schmidt et al. (EMNLP 2024) show tokenizer compression rate correlates only loosely with downstream quality — so "make fertility lower" is not a proxy for character competence.
- **BPE is not optimal even for LM pretraining.** Bostrom & Durrett (Findings of EMNLP 2020) find unigram-LM segmentation more morphologically aligned and slightly better downstream at BERT scale.

## 5. What Is Not Known

- **Theoretically open.** Whether the token→string map is identifiable from token co-occurrence alone. No proof that distributional statistics determine spelling, and no proof they cannot. A plausible negative result — two tokens with identical context distributions but different spellings are indistinguishable — has not been formalised or tested against corpus statistics.
- **Empirically open.** Whether a subword model trained at $\ge 7$B scale with an explicit character-supervision mixture (spelling, hyphenation, character-indexed edits at, say, 0.5–2% of tokens) closes $G_T$ without costing NLU. Runnable today; nobody has published a matched-FLOPs control.
- **Empirically open.** Whether byte-level and patch-based models (BLT, SpaceByte, MambaByte) actually beat subword models on stratified character tasks at equal FLOPs, or only on noise robustness.
- **Methodologically blocked.** There is no agreed measurement separating "the model lacks the information", "the model has it but cannot route it into a chain of thought", and "the model cannot emit the answer in the demanded output format". Until $G_T$ decomposes into these three, benchmark deltas are uninterpretable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability of the mechanism**. Every published character benchmark scores an end-to-end generation, which mixes at least four factors: (i) whether the character info is encoded, (ii) whether attention can address individual characters *inside* a token — it structurally cannot, since the token is one position — (iii) whether the model can serialise a per-character loop in its output, and (iv) instruction-following. A drop in accuracy is attributable to any of them. Probing fixes (i) alone and is itself contested: a strong probe can find information the model never uses, so probe $F_1$ upper-bounds availability without establishing use.

Second obstruction: **the control arm is expensive and unfair by default**. Comparing a byte model to a subword model requires a choice — equal parameters, equal FLOPs, equal tokens, or equal characters — and the four choices reverse the ranking. A clean answer needs a small scaling grid, not a single pair of runs, which puts the decisive experiment in the $10^{21}$–$10^{22}$ FLOP range rather than at toy scale.

## 7. Current Research (as of 2026)

- **Tokenizer-free / dynamic-patch models.** Meta FAIR's Byte Latent Transformer line (entropy-based patching), SpaceByte, MambaByte. Direction: keep byte-level input, recover subword-like compute cost via learned patch boundaries. *(frontier — verify current scale claims.)*
- **Diagnostic benchmarks and mechanism work.** CUTE (LMU Munich, Edman/Fraser) and follow-ups; glitch-token auditing (Cohere, Land & Bartolo). Interpretability groups probing where spelling lives across layers. *(frontier — verify.)*
- **Character-aware data mixtures.** Anecdotal reports that post-training on spelling and character-edit data lifts these tasks; no matched-control publication known. *(frontier — verify.)*
- **Tokenizer transplantation and vocabulary curricula.** Swapping or extending vocabularies post-hoc, mainly for multilingual fertility, with character competence as an untracked side effect.

## 8. Concrete Next Experiment

**Question.** Is $G_T$ a *representation* deficit or a *routing* deficit?

**Scale.** Four pretraining runs at 1.4B parameters, 30B tokens each (Chinchilla-ish, ~$1.2\times10^{21}$ FLOPs per run) — small enough to run on 64 A100-equivalents in days, large enough that character tasks are non-degenerate.

**Arms.**
1. **Control:** standard BPE, 32k vocab, unmodified data.
2. **+char-supervision:** identical, with 1% of tokens replaced by templated spelling/edit data ("banana → b a n a n a", "delete the 3rd letter of …").
3. **+BPE-dropout** ($p=0.1$), no extra data — tests whether variable segmentation alone injects character evidence.
4. **Byte baseline:** byte-level model at **equal FLOPs**, not equal parameters.

**Measurements.** For every arm: (a) held-out-type positional-spelling probe $F_1$ at each layer; (b) CUTE-style manipulation accuracy stratified into three token-frequency terciles and two length bins; (c) a control battery (HellaSwag, MMLU, perplexity) to catch regressions.

**The deciding number.** $\Delta G = G_T(\text{arm 1}) - G_T(\text{arm 2})$ on the middle frequency tercile, where $G_T = \text{probe-}F_1 - \text{manipulation accuracy}$.
- If $\Delta G \ge 20$ points with probe $F_1$ roughly unchanged between arms 1 and 2, the deficit is **routing**: the information was already there and supervision only taught the model to use it. Fix character awareness with post-training, not architecture.
- If $\Delta G < 5$ points and arm 2's gain comes with a matching probe-$F_1$ rise, the deficit is **representational**, and tokenizer-free architectures are the right lever.

## 9. Key References

- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Taku Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL, 2018. — arXiv:1804.10959
- **[Foundational]** Kaj Bostrom, Greg Durrett. *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP, 2020. — arXiv:2004.03720
- **[SOTA]** Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL, 2022. — arXiv:2105.13626
- **[SOTA]** Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting. *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL, 2022. — arXiv:2103.06874
- **[SOTA]** Yi Tay, Vinh Q. Tran, Sebastian Ruder, Jai Gupta, Hyung Won Chung, Dara Bahri, Zhen Qin, Simon Baumgartner, Cong Yu, Donald Metzler. *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* ICLR, 2022. — arXiv:2106.12672
- **[SOTA]** Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer. *Byte Latent Transformer: Patches Scale Better Than Tokens.* 2024. — arXiv:2412.09871
- **[SOTA]** Kevin Slagle. *SpaceByte: Towards Deleting Tokenization from Large Language Modeling.* NeurIPS, 2024. — arXiv:2404.14408
- **[Diagnostic]** Lukas Edman, Helmut Schmid, Alexander Fraser. *CUTE: Measuring LLMs' Understanding of Their Tokens.* EMNLP, 2024. — arXiv:2409.15452
- **[Diagnostic]** Itay Itzhak, Omer Levy. *Models in a Spelling Bee: Language Models Implicitly Learn the Character Composition of Tokens.* NAACL, 2022. — arXiv:2108.11193
- **[Diagnostic]** Ayush Kaushal, Kyle Mahowald. *What do tokens know about their characters and how do they know it?* NAACL, 2022. — arXiv:2206.02608
- **[Diagnostic]** Sander Land, Max Bartolo. *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.* EMNLP, 2024. — arXiv:2405.05417
- **[Empirical]** Aaditya K. Singh, DJ Strouse. *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs.* 2024. — arXiv:2402.14903
- **[Survey]** Craig W. Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omri Uzan, Yuval Pinter, Chris Tanner. *Tokenization Is More Than Compression.* EMNLP, 2024. — arXiv:2402.18376

## 10. Worked Example

Take the string `strawberry` under the `cl100k_base` (GPT-4-class) tokenizer. It encodes as three tokens: `st`, `raw`, `berry`. Ten characters, three positions.

Ask "how many `r`s?". The correct answer is 3, distributed as $0 + 1 + 2$ across the three tokens.

Now count what the model must do internally. There is no attention head that can attend to the second `r` of `berry`, because `berry` occupies one residual stream position. The per-character count must be stored *within* the embedding of `berry` as something a downstream MLP can add: the model needs a feature approximately equal to $\mathrm{count}_r(\mathrm{str}(v))$, then a three-term addition across positions. A containment probe only asks whether $\mathrm{count}_r > 0$ — a much easier binary feature, and exactly the one that scores in the high 90s on CUTE. The count feature is a different, higher-arity quantity, and nothing in the pretraining objective rewards it: predicting the next token after `berry` almost never requires knowing that it holds two `r`s.

Make the confound visible. Compare two prompts:

| Prompt | Segmentation | Typical outcome |
|---|---|---|
| "How many r's in strawberry?" | `st`/`raw`/`berry` (3 tokens) | Frequently wrong (2) |
| "How many r's in s t r a w b e r r y?" | ~10 tokens, one per letter | Reliably right (3) |

The spaced version is the same question with the same information content and the same model. Accuracy moves because the *addressing* changed: each character now has its own position, so induction-style heads can count them. This is the decomposition the field lacks a metric for. The first row's failure is usually reported as "the model doesn't know the spelling of strawberry" — but the probe says it does, and the second row says it can count. The deficit is that character-indexed information cannot be addressed while it is packed inside a token, and no current benchmark reports that separately from knowledge.

The obstruction, stated once: a benchmark number over the first row measures the sum of encoding, addressing, serialisation, and instruction-following, and every proposed fix — byte models, patching, character supervision — targets a different one of the four.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*