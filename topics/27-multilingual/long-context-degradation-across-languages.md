---
id: 27-multilingual/long-context-degradation-across-languages
title: "Long-Context Degradation Across Languages"
topic: 27-multilingual
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Context Degradation Across Languages

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/long-context-degradation-across-languages` · **Status:** empirically-open

## 1. Problem Statement

A model advertised at 128K or 1M tokens is evaluated, almost always, in English. The question is whether the *usable* context length is a property of the model or a property of the (model, language) pair — and if the latter, what causes the gap.

Three variants, with different difficulty:

- **Measurement.** Given a model $M$ and languages $\ell_1,\ell_2$, decide whether $M$ degrades faster in $\ell_2$ than in $\ell_1$ as input length grows. Solving this requires a length-matched, difficulty-matched, semantically-parallel evaluation. No such suite exists at $>32$K for more than a handful of languages. This variant is currently the bottleneck.
- **Method.** Given a fixed pretraining budget, close the gap: make $L^*_{\ell}$ (the length at which language $\ell$ retains 90% of its short-context accuracy) equal across a target language set. Solved iff the max-min ratio $\min_\ell L^*_\ell / \max_\ell L^*_\ell \geq 0.9$ on a held-out task family.
- **Theory.** Explain the gap from architecture. Does RoPE-based positional extrapolation interact with token fertility — the tokens-per-character ratio — in a way that predicts degradation rate? No proof either way exists.

The three are conflated in practice, which is why the problem resists progress: a paper reporting "Hindi degrades faster than English at 64K" has not separated positional failure from the fact that 64K Hindi tokens carry roughly half the text of 64K English tokens.

## 2. Formal Setting

Let $\mathcal{D}_\ell$ be a document distribution in language $\ell$, tokenizer $T$, and model $M$ with claimed context $L_{\max}$.

**Fertility.** For a text $x$ with $|x|_{\mathrm{chars}}$ characters (NFC-normalized, whitespace collapsed),
$$\phi_\ell \;=\; \mathbb{E}_{x\sim\mathcal{D}_\ell}\!\left[\frac{|T(x)|}{|x|_{\mathrm{chars}}}\right].$$
Measured on FLORES-200 devtest, which is parallel across 200 languages, so $\phi_\ell$ is comparable by construction. Typical values for a Llama-3-class 128K vocabulary: English $\approx 0.25$, Hindi $\approx 0.55$, Telugu $\approx 0.85$ tok/char. A *semantic* normalizer $\psi_\ell = \phi_\ell / \phi_{\mathrm{eng}}$ converts token budgets into English-equivalent content.

**Degradation curve.** For task family $\mathcal{T}$ (retrieval, aggregation, multi-hop, summarization), context length $n$ tokens, and needle depth $d\in[0,1]$ (fractional position of the target span),
$$A_\ell(n,d) \;=\; \Pr_{(c,q,y)\sim \mathcal{T}_\ell(n,d)}\big[M(c,q) = y\big].$$
Measured as exact match for retrieval, and as a calibrated judge score for generative tasks — which is itself language-dependent and a known confound (§6).

**Usable length.** With $A_\ell^0 = A_\ell(n_0,\cdot)$ at a short reference $n_0 = 2$K,
$$L^*_\ell(\tau) \;=\; \max\{\,n \le L_{\max} : \textstyle\min_d A_\ell(n,d) \ \ge\ \tau\, A_\ell^0 \,\},\qquad \tau = 0.9 .$$
The $\min_d$ matters: averaging over depth hides the lost-in-the-middle trough.

**The two comparisons.** Token-matched gap $G_{\mathrm{tok}} = L^*_{\mathrm{eng}} / L^*_\ell$, and content-matched gap $G_{\mathrm{sem}} = L^*_{\mathrm{eng}} / (\psi_\ell L^*_\ell)$. A result is only about *long-context ability* if $G_{\mathrm{sem}} \ne 1$. If $G_{\mathrm{tok}}\ne 1$ but $G_{\mathrm{sem}} = 1$, the finding is about tokenization, not attention.

**Assumptions, and which are violated.**
1. *Parallel difficulty:* translated haystacks pose equal task difficulty in every language. **Violated** — translationese is lower-perplexity and more literal, making retrieval easier; named-entity needles are often left in English, making them lexically salient in a non-Latin haystack.
2. *Language purity of the haystack:* **violated** — web-derived long documents in low-resource languages routinely contain English code-switching.
3. *Short-context baseline is a fair anchor:* **violated** where $A^0_\ell$ is already near chance; the ratio $\tau A^0_\ell$ then has no resolution.
4. *Depth is well-defined across languages:* holds by token index, but the same token depth is a different semantic depth when $\phi_\ell$ differs.

## 3. State of the Art

**Established (English, single-language).** *Lost in the Middle* (Liu et al., TACL 2024) showed a U-shaped accuracy-vs-position curve, reproduced widely. **RULER** (Hsieh et al., COLM 2024) showed that effective length is far below claimed length: of 10 models claiming $\ge$32K, most fell below the performance threshold well before their advertised limit; GPT-4 was the strongest and still degraded. RULER is English-only.

**Established (bilingual).** **LongBench** (Bai et al., ACL 2024) and **$\infty$Bench** (Zhang et al., ACL 2024) are English–Chinese. Both report per-language splits, but the splits are *not* parallel — different documents, different tasks — so cross-language differences are not attributable.

**Claimed but unablated.** **Multilingual needle-in-a-haystack** studies (Hengle et al., 2024, "Multilingual Needle in a Haystack") report that retrieval accuracy falls with linguistic distance from English and that the drop is largest at deep positions. The experiments do not control fertility; the reported gaps are $G_{\mathrm{tok}}$, not $G_{\mathrm{sem}}$. Vendor claims of "128K in 100+ languages" (Gemini, GPT-4-class, Qwen, Command-R) are **benchmark numbers with no public per-language depth curve** — they exist only as aggregate scores.

**Theory SOTA.** Position-interpolation analyses — Positional Interpolation (Chen et al., 2023), **YaRN** (Peng et al., ICLR 2024) — give frequency-domain accounts of RoPE extension. None models token statistics, so none predicts a language-dependent effect. There is no theorem relating fertility to extrapolation quality.

## 4. What Is Known

- **Fertility gaps are large and measured.** Ahia et al. (EMNLP 2023) and Petrov et al. (NeurIPS 2023) report up to $\sim$15$\times$ differences in tokens per equivalent text across languages for commercial tokenizers; even for well-covered languages, 2–4$\times$ over English is routine. Scale: FLORES-200 / parallel corpora, GPT-3.5–GPT-4-era vocabularies of 32K–100K.
- **Effective $\ll$ claimed, in English.** RULER, at 4K–128K on 10+ models of 7B–70B plus GPT-4: nearly all models lose substantial accuracy well before the claimed limit, with multi-hop and aggregation tasks collapsing earliest.
- **Length alone hurts reasoning.** Levy et al. (ACL 2024, FLenQA) held reasoning content constant and only padded: accuracy dropped sharply from 250 to 3000 tokens, in English. This isolates length from difficulty — the design that the multilingual literature has not yet copied.
- **Tokenizer quality predicts downstream quality.** Rust et al. (ACL 2021) showed that a language-specific tokenizer recovers much of the gap between mono- and multilingual encoders — established at BERT scale, on short contexts.
- **Bilingual long-context suites exist and show a gap.** LongBench reports lower Chinese than English scores for most open models, but with non-parallel data, so the gap is uninterpretable as a language effect.

## 5. What Is Not Known

- **Empirically open (the main gap).** Whether $G_{\mathrm{sem}} \ne 1$ — whether any language-dependent degradation survives fertility normalization. The experiment is entirely runnable today: it needs a parallel 128K-token haystack corpus in $\ge$20 typologically diverse languages and a few thousand GPU-hours of inference. Nobody has published it.
- **Empirically open.** Whether the depth of the lost-in-the-middle trough varies by script or by pretraining-token share. Requires per-language, per-depth curves; currently published only for English and Chinese.
- **Theoretically open.** Whether RoPE extension methods (PI, YaRN, NTK-scaling) are language-neutral. No result either way; the analyses are purely about frequency scaling.
- **Methodologically blocked.** Long-context *generation* quality — summarization, multi-document synthesis — across languages. LLM-judge scores are themselves less reliable in low-resource languages, and no calibrated cross-lingual judge exists. Until that is fixed, the measurement is not defined.

## 6. Why It Is Hard

**The measurement is confounded at the definition.** Fixing the token budget fixes neither the amount of information nor the number of sentences the model must span. Fixing the character budget changes the token count and hence the positional-encoding regime being tested. There is no normalization under which the arms are simultaneously matched on content, tokens, and difficulty — so every published cross-lingual long-context number is a mixture of at least two effects.

Compounding it: **absent ground truth at length**. There are no naturally occurring, quality-controlled 100K-token documents with annotated questions in Telugu, Amharic, or Yoruba. Translating English haystacks introduces translationese, which changes the task's difficulty in an unmeasured direction. Synthesizing them makes them unrepresentative of what the model saw in pretraining.

Third: **non-identifiability of cause**. A gap can arise from (a) fertility, (b) pretraining token share for $\ell$, (c) long-document scarcity for $\ell$ specifically — most long-context training data is English — or (d) positional-encoding interaction. Observational comparisons across existing models cannot separate these, because (a)–(c) covary almost perfectly.

## 7. Current Research (as of 2026)

- **Parallel long-context benchmarks.** Extensions of the multilingual needle work toward reasoning rather than retrieval — e.g. multilingual long-context reasoning suites released in 2025 by the Hengle/Chakraborty group *(frontier — verify)*.
- **Tokenizer-fair evaluation.** Growing use of byte- or character-normalized reporting; byte-level and dynamic-patching architectures (BLT-style) are the obvious control arm, since they remove $\phi_\ell$ from the comparison *(frontier — verify)*.
- **Long-context data curation for non-English.** Multilingual long-document mining for continued pretraining, at Cohere (Aya line), Alibaba (Qwen), and EleutherAI-adjacent groups *(frontier — verify)*.
- **Positional-encoding audits.** Per-language attention-entropy and RoPE-frequency probes; largely unpublished internal work *(frontier — verify)*.

## 8. Concrete Next Experiment

**"Fertility-controlled parallel haystack."**

- **Scale.** 12 languages spanning fertility $\phi \in [0.25, 0.9]$ (eng, deu, fra, rus, arb, hin, ben, tel, tha, zho, swh, amh). Two open models at 8B and 70B with claimed 128K (Llama-3.1 class) plus one API model. Lengths $n \in \{2\text{K}, 8\text{K}, 32\text{K}, 64\text{K}, 128\text{K}\}$ tokens, depths $d \in \{0.05,\dots,0.95\}$ (10 values), 200 items per cell. Total $\approx 12\times3\times5\times10\times200 = 360$K queries; dominated by the 128K cells, roughly 2–4K H100-hours.
- **Task.** Not vanilla NIAH. Use two-hop variable tracking (RULER's `vt`) built from a *natively multilingual* haystack — Wikipedia in $\ell$, not translated — with needles that are language-native entity strings, so no English lexical island exists.
- **Control arms.** (1) *Token-matched:* every arm gets $n$ tokens. (2) *Content-matched:* language $\ell$ gets $n/\psi_\ell$ tokens, equalizing characters. (3) *Fertility placebo:* English haystack re-tokenized with a deliberately inefficient tokenizer to raise $\phi_{\mathrm{eng}}$ to Telugu levels — isolating fertility from language identity. Arm 3 is the arm the literature is missing.
- **Deciding number.** $G_{\mathrm{sem}} = L^*_{\mathrm{eng}}/(\psi_\ell L^*_\ell)$ at $\tau=0.9$, reported per language with bootstrap CIs. **If $\max_\ell G_{\mathrm{sem}} < 1.25$ and the placebo arm reproduces the token-matched gap, the phenomenon is tokenization and the "long-context degradation across languages" framing is wrong.** If $G_{\mathrm{sem}} > 1.5$ for the low-resource half while the placebo arm shows $\approx 1$, there is a genuine language-dependent long-context deficit and the problem becomes a method problem.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[SOTA]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[SOTA]** Yushi Bai et al. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.* ACL, 2024. — arXiv:2308.14508
- **[SOTA]** Xinrong Zhang et al. *$\infty$Bench: Extending Long Context Evaluation Beyond 100K Tokens.* ACL, 2024.
- **[Foundational]** Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, Adel Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS, 2023. — arXiv:2305.15425
- **[Foundational]** Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R. Mortensen, Noah A. Smith, Yulia Tsvetkov. *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP, 2023. — arXiv:2305.13707
- **[Foundational]** Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, Iryna Gurevych. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL, 2021.
- **[SOTA]** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR, 2024. — arXiv:2309.00071
- **[SOTA]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024.
- **[Survey]** Amey Hengle, Prasoon Bajpai, Soham Dan, Tanmoy Chakraborty. *Multilingual Needle in a Haystack: Investigating Long-Context Behavior of Multilingual Large Language Models.* 2024.
- **[Foundational]** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 2024. — arXiv:2104.09864

## 10. Worked Example

Take one 128K-token English haystack and its Telugu translation. Measured fertilities on FLORES-200 devtest with a Llama-3 128K vocabulary are about $\phi_{\mathrm{eng}}=0.25$, $\phi_{\mathrm{tel}}=0.85$ tok/char, so $\psi_{\mathrm{tel}}\approx 3.4$.

- English arm: $128\text{K}$ tokens $\approx 512$K characters $\approx 85$K words $\approx$ a 340-page book.
- Telugu arm at the same 128K tokens: $\approx 150$K characters — roughly **one-third of a book**.

Suppose the measured curves give $L^*_{\mathrm{eng}}(0.9) = 64$K and $L^*_{\mathrm{tel}}(0.9) = 16$K tokens. The headline is a $4\times$ gap:
$$G_{\mathrm{tok}} = 64/16 = 4.0 .$$
Now normalize for content:
$$G_{\mathrm{sem}} = \frac{64}{3.4 \times 16} = \frac{64}{54.4} \approx 1.18 .$$

The $4\times$ becomes $1.18\times$. Read in characters, the model handles about 256K English characters and about 188K Telugu characters before dropping below the 90% threshold — a 27% shortfall, not a 4$\times$ collapse. Almost the whole reported gap is the tokenizer.

That is the obstruction, made concrete: the same experiment supports "Telugu long-context is 4$\times$ worse" and "Telugu long-context is 18% worse" depending on a normalization choice that no published multilingual long-context paper states explicitly. And even the 1.18 is not clean — the Telugu arm is translationese and its short-context anchor $A^0_{\mathrm{tel}}$ may be 0.82 rather than 0.97, so the $\tau A^0$ threshold sits at a different absolute accuracy in each arm. The fertility-placebo arm of §8 is the only thing that breaks the tie, and it has not been run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*