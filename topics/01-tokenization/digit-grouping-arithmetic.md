---
id: 01-tokenization/digit-grouping-arithmetic
title: "Digit Grouping Schemes for Arithmetic Accuracy"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Digit Grouping Schemes for Arithmetic Accuracy

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/digit-grouping-arithmetic` · **Status:** empirically-open

## 1. Problem Statement

A tokenizer must decide how to cut a digit string into tokens. Three families are in production use:

- **single-digit** (`1|2|3|4`) — T5, PaLM, Gemma, Qwen2.5, DeepSeek-V3;
- **left-to-right $k$-digit chunks** (`123|4` for $k=3$) — Llama 3, GPT-4o's `o200k` family;
- **right-to-left $k$-digit chunks** (`1|234`) — used by later Llama variants and available as a tokenizer option in several open stacks.

**Decision predicate.** Fix a training corpus, architecture, parameter count and compute budget. Does the choice of grouping scheme $g$ change end-of-training arithmetic accuracy by more than the seed-to-seed standard deviation, and if so, by how much, as a function of operand length?

Three variants, of different difficulty:

- **Measurement.** Build an accuracy-vs-operand-length curve that is attributable to $g$ alone. Blocked today by vocabulary-size and token-count confounds (§6).
- **Method.** Find the $g$ (or the encoding that replaces $g$ — index hints, Abacus embeddings, xVal) that maximizes arithmetic accuracy at fixed compute without degrading general language loss. Partially answered for toy models, unanswered at frontier scale.
- **Theory.** Prove that some grouping admits a constant-depth or $O(\log n)$-depth transformer circuit for $n$-digit addition, while another provably requires more. Open.

Solving it means: a rule of the form "use scheme $g^\*$ at vocabulary budget $B$", supported by a compute-matched ablation, not by cross-model correlation.

## 2. Formal Setting

Let $s = d_1 d_2 \cdots d_n \in \{0,\dots,9\}^n$ be a numeral, $d_1$ most significant. A grouping scheme is a map
$$g_{k,\alpha}(s) = (c_1, \dots, c_m), \quad c_i \in \bigcup_{j=1}^{k}\{0,\dots,9\}^j, \quad \mathrm{concat}(c_i)=s,$$
with $\alpha \in \{L,R\}$ the anchor. For $\alpha=L$ all chunks have length $k$ except the last; for $\alpha=R$ except the first. $m = \lceil n/k \rceil$. Single-digit is $k=1$.

**Vocabulary cost.** $|V_k| = \sum_{j=1}^{k} 10^j$: 10, 110, 1110 tokens for $k=1,2,3$. At embedding width $d$ this is $2d|V_k|$ parameters (input + output). At $d=4096$, $k{=}3$ costs $9.1{\times}10^6$ more parameters than $k{=}1$ — 0.11% of an 8B model. **Measured**, not estimated.

**Place-value index.** Chunk $c_i$ occupies decimal exponents $[\,e_i, e_i + |c_i| - 1\,]$. Under $\alpha=R$, $e_i = k(m-i)$: the exponent is a function of the chunk index alone. Under $\alpha=L$, $e_i = n - \sum_{j\le i}|c_j|$: it depends on $n \bmod k$.

**Alignment.** For a binary operation on operands of lengths $n, n'$, define
$$A_{k,\alpha}(n,n') = \frac{1}{\min(n,n')}\,\bigl|\{\,p : \text{digits at exponent } p \text{ fall at the same within-chunk offset in both operands}\,\}\bigr|.$$
$A_{k,R} \equiv 1$ for all $n,n'$. $A_{k,L} = 1$ iff $n \equiv n' \pmod k$, and is otherwise strictly less than 1.

**Objective, as measured.** Sample the length grid $\mathcal{G} = \{(n,n') : 1 \le n,n' \le N\}$ uniformly, one problem per cell, $R$ repeats. Report exact-match on the full answer string:
$$\mathrm{EM}(g) = \frac{1}{|\mathcal{G}|R}\sum_{(n,n')\in\mathcal{G}}\sum_{r=1}^{R} \mathbb{1}[\hat{y}=y],$$
and the length-generalization edge $L^\*(g) = \max\{n : \mathrm{EM}(g;n,n) \ge 0.9\}$.

**Assumptions, and which are violated.**
1. *Compute matching.* Schemes differ in tokens-per-number ($n$ vs $\lceil n/3\rceil$), so equal-token training is not equal-numeral training. **Violated by construction** — no scheme choice can satisfy both.
2. *Vocabulary neutrality.* Assumed the extra 1100 tokens do not shift BPE merges elsewhere. **Violated**: at fixed total vocabulary, digit tokens displace text merges.
3. *Corpus independence.* Web text contains years, IDs and prices whose frequency drives merges; naive BPE gives `2020` one token and `2021` possibly two. **Violated in every model that does not force digit splits.**
4. *Exact match is the right metric.* It conflates a single carry error with a total failure. **Known to be violated** as a proxy for "understands addition".

## 3. State of the Art

**Established (compute-matched ablations exist).**
- Reversing the *output* digit order — emitting least-significant first — takes NanoGPT-scale models (~10.6M params) from near-zero to near-100% on 3-digit addition with far fewer training examples (Lee et al., *Teaching Arithmetic to Small Transformers*, 2023). This is an ordering result, and it is the cleanest evidence that place-value locality, not capacity, is the binding constraint at small scale.
- Positional information dominates grouping. Abacus embeddings — a learned embedding of each digit's position *within its number* — let 16-layer decoder-only models trained on $\le$20-digit addition reach reported 99%+ on 100-digit addition, a $5\times$ length extrapolation (McLeish et al., NeurIPS 2024). Index hints achieve a similar effect (Zhou et al., 2023/2024).

**Claimed but unablated.**
- That right-to-left chunking beats left-to-right at frontier scale. Singh & Strouse (*Tokenization counts*, 2024) measure large accuracy gaps between L2R and R2L chunking on GPT-3.5/GPT-4-class models by manipulating input formatting (commas, spaces) — but the manipulation is at inference time on a fixed tokenizer, not a retrained control.
- That single-digit tokenization is why Qwen/DeepSeek score well on GSM8K-style arithmetic relative to Llama 3. This is **cross-model correlation only**: data mixture, RL post-training, and parameter count all differ.

**Benchmark numbers only.** Most public evidence — GSM8K, MATH, the *Number Cookbook* suite (Yang et al., 2024) — reports per-model scores under each model's native tokenizer. No entry in any of these tables isolates $g$.

## 4. What Is Known

- $|V_3| - |V_1| = 1100$ tokens; at $d{=}4096$ that is 9.1M parameters. Arithmetic, not experiment.
- Under L2R chunking with $k=3$, over the grid $1\le n,n' \le 20$, exactly $7^2+7^2+6^2 = 134$ of $400$ length pairs are place-value aligned. **66.5% of operand-length pairs are misaligned.** Under R2L, 0%.
- Pretrained LLMs implement addition partly via Fourier features over digit magnitude (Zhou et al., *Pre-trained large language models use Fourier features to compute addition*, NeurIPS 2024) — measured on GPT-2-scale and Llama-3-8B activations. The mechanism is magnitude-based, not chunk-symbol-based, which predicts grouping should matter *less* than positional encoding. Consistent with the Abacus result.
- Output-order reversal effect size: near-0% → near-100% on 3-digit addition at 10.6M params (Lee et al., 2023).
- Length-generalization edge with Abacus embeddings: train $\le 20$ digits, reported 99%+ at 100 digits, ~100M-param decoder-only models trained from scratch on arithmetic only.
- Continuous numeric encodings (xVal, Golkar et al., 2023) beat digit tokenization on regression-flavoured numeric tasks but do not produce exact multi-digit answers.

## 5. What Is Not Known

- **Empirically open.** The compute-matched pretraining ablation — same corpus, same parameter count, same tokens, three tokenizers — has not been run at $\ge$1B parameters on natural text. Every ingredient is available; nobody has published it. This is the core gap.
- **Empirically open.** Whether any grouping advantage survives chain-of-thought and RL post-training, which may launder tokenization damage into more steps rather than more errors.
- **Methodologically blocked.** How to compare schemes at equal compute when tokens-per-numeral differs by $3\times$. There is no agreed normalizer: equal tokens, equal numerals, or equal FLOPs give different rankings, and no published protocol commits to one.
- **Theoretically open.** No separation theorem. It is not proved that $n$-digit addition under L2R $k$-chunking requires asymptotically more depth than under R2L, despite the alignment argument suggesting it should. The RASP-L framework (Zhou et al.) makes the question askable but has not been used to prove a grouping separation.

## 6. Why It Is Hard

**The obstruction is confounded measurement, three-deep.**

1. *Vocabulary–merge coupling.* Changing $k$ changes 1100 vocabulary slots, which at fixed budget changes which text merges survive. General LM loss moves, so arithmetic deltas cannot be read as tokenizer-only effects.
2. *No compute normalizer.* $k{=}3$ sees each numeral in $\lceil n/3 \rceil$ tokens. Equal-token training gives $k{=}3$ roughly $3\times$ more numeral exposure; equal-numeral training gives it fewer gradient steps on digits. Both are defensible and they rank the schemes differently.
3. *Cross-model inference.* The available "evidence" is Llama-vs-Qwen comparisons where tokenizer is one of a dozen simultaneous differences. Non-identifiable.

Cost is real but secondary: three 1B-parameter runs at 30B tokens is roughly $3 \times 1.8\times10^{20}$ FLOPs, days on a modest node. The reason it is unrun is that no one owns the protocol, not that no one can afford it.

## 7. Current Research (as of 2026)

- **Positional-encoding substitutes.** Abacus-style within-number position embeddings and index hints (Maryland/UMD group around McLeish, Goldstein; Google/DeepMind on RASP-L length generalization). The live claim is that once position is explicit, grouping stops mattering — untested on natural-text pretraining. *(frontier — verify)*
- **Production tokenizer defaults.** Open frontier models have converged on single-digit or R2L 3-digit; L2R is now rare. The convergence is empirical folklore rather than a published ablation. *(frontier — verify)*
- **Numeric benchmarks that vary length systematically.** *Number Cookbook* (Yang et al., 2024) and successors report per-length curves rather than aggregate scores, which is the prerequisite for detecting $n \bmod k$ artefacts.
- **Hybrid numeric heads.** xVal-style continuous value channels alongside digit tokens, aimed at scientific-text models. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Three 1.4B-parameter decoder-only models, identical architecture, 30B tokens of the same corpus (e.g. a fixed FineWeb-Edu slice), 3 seeds each — 9 runs.

**Arms.** (a) single-digit; (b) L2R 3-digit; (c) R2L 3-digit. Hold total vocabulary at 32,768 by reserving 1,110 slots for digit tokens in *all three* arms — unused slots stay unused in arm (a). This kills confound 1: the BPE merge table over text is byte-identical across arms.

**Control arm.** Arm (c) plus Abacus-style within-number position embeddings, at $k=3$. If the tokenizer effect vanishes here, grouping is a positional-encoding problem, not a vocabulary problem.

**Compute normalizer.** Report both equal-token and equal-numeral-exposure curves. Pre-register equal-token as primary.

**Evaluation.** Zero-shot addition and multiplication over the grid $1 \le n,n' \le 20$, 200 samples per cell, exact match, no chain-of-thought.

**The deciding number.** $\Delta = \mathrm{EM}(\text{R2L}) - \mathrm{EM}(\text{L2R})$ restricted to the 266 misaligned cells ($n \not\equiv n' \bmod 3$), compared against the seed-to-seed standard deviation $\sigma$. If $\Delta > 3\sigma$ and $\Delta$ on the 134 aligned cells is within $1\sigma$, the alignment hypothesis is confirmed and grouping is causal. If $\Delta \le \sigma$ on both subsets, digit grouping does not matter at 1.4B and the field should stop citing it.

## 9. Key References

- **[Foundational]** Nayoung Lee, Kartik Sreenivasan, Jason D. Lee, Kangwook Lee, Dimitris Papailiopoulos. *Teaching Arithmetic to Small Transformers.* ICLR 2024. — arXiv:2307.03381
- **[Foundational]** Aaditya Singh, Daniel Strouse. *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs.* 2024. — arXiv:2402.14903
- **[SOTA]** Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, Tom Goldstein. *Transformers Can Do Arithmetic with the Right Embeddings.* NeurIPS 2024. — arXiv:2405.17399
- **[SOTA]** Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, Preetum Nakkiran. *What Algorithms can Transformers Learn? A Study in Length Generalization.* ICLR 2024. — arXiv:2310.16028
- **[Mechanism]** Tianyi Zhou, Deqing Fu, Vatsal Sharan, Robin Jia. *Pre-trained Large Language Models Use Fourier Features to Compute Addition.* NeurIPS 2024. — arXiv:2406.03445
- **[Encoding]** Siavash Golkar, Mariel Pettee, Michael Eickenberg, Alberto Bietti, et al. *xVal: A Continuous Number Encoding for Large Language Models.* NeurIPS 2023 AI4Science Workshop. — arXiv:2310.02989
- **[Survey/Benchmark]** Haotong Yang, Yi Hu, Shijia Kang, Zhouchen Lin, Muhan Zhang. *Number Cookbook: Number Understanding of Language Models and How to Improve It.* ICLR 2025. — arXiv:2411.03766
- **[Related]** Samy Jelassi, Stéphane d'Ascoli, Carles Domingo-Enrich, Yuhuai Wu, Yuanzhi Li, François Charton. *Length Generalization in Arithmetic Transformers.* 2023. — arXiv:2306.15400

## 10. Worked Example

Compute $456 + 7891 = 8347$ under $k=3$.

**R2L.** `456` → [`456`]. `7891` → [`7`, `891`]. Chunk exponents are $k(m-i)$: `456` and `891` both sit at exponents $[0,2]$. The model adds two aligned symbols, $456+891 = 1347$, propagates one carry into `7`, emits [`8`,`347`]. Two chunk-level operations, one carry. Circuit depth is $O(m)=2$.

**L2R.** `456` → [`456`]. `7891` → [`789`, `1`]. Now `456` covers exponents $[0,2]$ and `789` covers $[1,3]$. There is no digit position at which the two tokens' offsets agree; $A_{3,L}(3,4)=0$. To add, the model must decompose `789` into $\{7,8,9\}$, re-partition against $\{4,5,6\}$, and recompose — an operation on *sub-token* content that the embedding does not expose. Whatever the model learns must be memorized per unaligned pair, and there are $10^3 \times 10^3 = 10^6$ such pairs at $k=3$ versus $100$ digit pairs at $k=1$.

**Where the obstruction becomes visible.** The natural check is to compare a production L2R model (Llama-3-8B) with a production single-digit model (Qwen2.5-7B) on this grid. Suppose the L2R model scores lower on the 266 misaligned cells than the 134 aligned ones. That gap is *still* not attributable: the two models differ in 1B parameters, in pretraining mixture, in post-training, and in vocabulary size (128k vs 152k). The within-model aligned-vs-misaligned contrast is the only comparison that survives — and even it is confounded, because operand lengths with $n \equiv n' \pmod 3$ include the high-frequency cases $n=n'$, which are over-represented in web arithmetic. Removing that confound requires holding $n=n'$ fixed and varying only $n \bmod 3$, which leaves 20 usable cells out of 400. **The measurement runs out of statistical power before it runs out of compute.** That is why the ablation in §8, with the merge table held byte-identical, is the experiment that matters.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*