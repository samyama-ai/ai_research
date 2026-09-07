---
id: 01-tokenization/canonicality-bias
title: "Canonicality Bias in Subword Language Models"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Canonicality Bias in Subword Language Models

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/canonicality-bias` · **Status:** open

## 1. Problem Statement

A subword language model defines a distribution over *token sequences*, but users care about a distribution over *strings*. The map from strings to token sequences is one-to-many: `" strawberry"` can be segmented dozens of ways over a typical BPE vocabulary. Training uses exactly one of them — the **canonical** segmentation produced by the tokenizer's deterministic encoder (greedy merge order for BPE, Viterbi argmax for UnigramLM). Every other segmentation of the same string is off-distribution.

**Canonicality bias** is the resulting family of failures:

- **Measurement variant.** The model scores a string by its canonical tokenization, $p_\theta(\text{canon}(x))$, but the string's true probability is the sum over all tokenizations. Reported perplexity is therefore a *lower bound* on likelihood by an unknown amount, and the bound is not comparable across tokenizers.
- **Method variant.** At inference the model is fed prompts whose token prefix is not a canonical prefix of any completion — mid-word prompts, fill-in-the-middle, constrained decoding, tool-call splices. The model's conditional is undefined off the canonical manifold, and outputs degrade in ways that look like reasoning failures.
- **Theory variant.** Is the induced string distribution even well defined and consistent, and does marginalizing over tokenizations strictly improve it, or is the canonical restriction the right estimator under a finite-sample argument?

**Solved** would mean: (a) a tractable estimator of $p_\theta(x)$ over strings with certified error bars, (b) a decoding rule whose output distribution over strings is invariant to how the prompt was segmented, and (c) an account of when marginal and canonical training differ enough to change downstream behaviour.

## 2. Formal Setting

Let $\Sigma$ be a byte alphabet, $x \in \Sigma^*$ a string, $V \subset \Sigma^+$ a vocabulary. Define the **segmentation fiber**

$$\mathcal{T}(x) = \{ \mathbf{t} = (t_1,\dots,t_n) \in V^n : t_1 \cdot t_2 \cdots t_n = x \}.$$

The tokenizer is a function $\tau: \Sigma^* \to V^*$ with $\tau(x) \in \mathcal{T}(x)$; **canonical** means $\mathbf{t} = \tau(x)$. Measured as: run the shipped encoder (e.g. `tiktoken`, `sentencepiece`) — not a reimplementation, since merge-tie-breaking and pretokenizer regex differ.

The model gives $p_\theta$ over $V^*$. Two objectives:

$$\mathcal{L}_{\text{canon}}(\theta) = -\log p_\theta(\tau(x)), \qquad \mathcal{L}_{\text{marg}}(\theta) = -\log \sum_{\mathbf{t} \in \mathcal{T}(x)} p_\theta(\mathbf{t}).$$

By nonnegativity of the omitted terms, $\mathcal{L}_{\text{marg}} \le \mathcal{L}_{\text{canon}}$ always. Define the **canonicality gap** in bits per byte:

$$\Delta(x) = \frac{\mathcal{L}_{\text{canon}}(x) - \mathcal{L}_{\text{marg}}(x)}{|x| \ln 2}.$$

Measured by importance sampling: draw $\mathbf{t}^{(i)} \sim q(\cdot \mid x)$ from a proposal (BPE-dropout with rate $p$, or a Viterbi-sampling proposal), estimate $\hat{Z} = \frac{1}{N}\sum_i p_\theta(\mathbf{t}^{(i)})/q(\mathbf{t}^{(i)})$, and report $\hat{Z}$ with a bootstrap CI. $|\mathcal{T}(x)|$ grows exponentially in $|x|$, so exact summation is only feasible for $|x| \lesssim 15$ bytes or via a lattice DP that the transformer's unbounded context forbids.

Define **canonical mass** $\kappa(x) = p_\theta(\tau(x)) / \sum_{\mathbf{t}} p_\theta(\mathbf{t}) \in (0,1]$; $\Delta = 0 \iff \kappa = 1$.

**Prompt-boundary bias.** For a prompt $u$ and continuation $v$, the correct object is $p_\theta(v \mid u) = p_\theta(uv)/p_\theta(u)$ over strings. Feeding $\tau(u)$ instead conditions on the event "$u$ ends at a canonical token boundary", which is a strictly stronger event. Measured as: the total-variation distance between the next-byte distributions induced by $\tau(u)$ and by exact marginalization.

**Assumptions and their violations.**
- *Vocabulary is prefix-free or segmentation is unique* — false for all BPE and UnigramLM vocabularies.
- *Canonical mass $\kappa \approx 1$* — approximately true for English prose in-distribution, empirically false for rare words, code identifiers, non-Latin scripts, and numerals.
- *Proposal $q$ covers the support of $p_\theta$* — BPE-dropout proposals put zero mass on some fibers; the estimator is then biased low without indicating it.
- *The canonical encoder is deterministic across implementations* — false in practice; HuggingFace fast/slow tokenizers and byte-fallback settings disagree on some inputs.

## 3. State of the Art

**Established.**
- *Subword regularization* (Kudo, ACL 2018) and *BPE-dropout* (Provilkov et al., ACL 2020) train on sampled non-canonical segmentations. Both report consistent NMT gains — BPE-dropout up to $+2.3$ BLEU on low-resource pairs, ~$+0.9$ BLEU on high-resource — reproduced widely. This is the strongest evidence that canonical-only training leaves capability on the table.
- *Marginal-likelihood evaluation* (Cao & Rimell, EMNLP 2021) established that reported perplexities are one-sided bounds and gave the importance-sampling estimator now standard.
- *Exact next-byte correction* (Phan et al., NeurIPS 2024) gives an algorithm converting a token-level model into an unbiased next-byte predictor without retraining, via a lemma relating byte-level and token-level conditionals. This is a genuine method, not a heuristic.
- *Character-level conversion* (Vieira et al., 2024) formalizes the token-to-character LM conversion and its cost.

**Claimed but unablated.**
- "Token healing" (back up over the last token of the prompt, re-decode with a prefix constraint) is deployed in `guidance`, llama.cpp and several serving stacks. It is folklore-effective; there is no controlled study isolating its effect from the sampling changes it co-occurs with.
- Claims that marginalization "does not matter" rest on Chirkova et al. (EMNLP 2023), who measured the gap on English/Russian at GPT-2/Pythia scale. Their negative result has not been reproduced for code, math, or morphologically rich languages.

**Benchmark-number-only.** Reports that FIM (fill-in-the-middle) code models fail on mid-token cursors are near-universal in practitioner reports and appear as accuracy deltas in model cards, with no isolating ablation against a byte-exact baseline.

## 4. What Is Known

- **The gap is small for English prose.** Chirkova et al. (2023) report the marginal-vs-canonical log-likelihood gap under $0.5\%$ of perplexity for most settings, at model scales up to ~$7$B, using ~$10^2$ importance samples per sequence. Cao & Rimell (2021) report larger gains, up to roughly $1$–$2\%$ perplexity, on transformer LMs in the $10^8$-parameter range trained on English. The two are compatible: the gap shrinks as the model concentrates mass on canonical strings.
- **Canonical mass is high but not one.** For in-distribution English at GPT-2 scale, $\kappa$ estimates sit above $0.95$ for common words; the tail is where mass leaks.
- **Non-canonical prompts break models.** Bostrom & Durrett (EMNLP Findings 2020) showed the tokenizer choice itself moves downstream task accuracy by 1–2 points at BERT scale, with UnigramLM segmentations aligning better with morphology than BPE.
- **Segmentation choice is not intrinsically optimal.** Hofmann et al. (ACL 2022, FLOTA) improved classification accuracy simply by replacing the canonical segmentation with a morphology-respecting non-canonical one at inference — direct evidence that the canonical fiber element is not the best one.
- **Digit and glyph pathologies.** Number tokenization (right-to-left vs left-to-right grouping) changes arithmetic accuracy by tens of points at the multi-billion-parameter scale; character-counting failures ("how many r's in strawberry") are the popular face of the same effect.
- **Untrained tokens exist.** The `SolidGoldMagikarp` family (Rumbelow & Watkins, 2023) are vocabulary entries with near-zero training frequency and anomalous embeddings — the extreme case of a token that appears only off the canonical path.

## 5. What Is Not Known

- **Theoretically open.** No characterization of when $\mathcal{L}_{\text{marg}}$-trained models dominate $\mathcal{L}_{\text{canon}}$-trained ones. Canonical training is a *harder, lower-entropy* target and may act as a useful inductive bias; the bias–variance account is absent. Also open: whether the string distribution induced by a canonically trained autoregressive model is tight (non-leaking) at all.
- **Empirically open.** The gap $\Delta$ has never been measured at frontier scale ($\ge 10^{11}$ parameters) or on code and non-Latin scripts with a proposal known to cover the fiber. Runnable today; nobody has run it.
- **Methodologically blocked.** There is no agreed measurement of prompt-boundary bias for *instruction-tuned chat models*, because chat templates make the "string" the model conditions on ill-defined — special tokens have no byte realization, so $\mathcal{T}(x)$ is not even a well-formed object across the template boundary.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the estimator's failure mode**. The importance-sampling estimate $\hat{Z}$ is a lower bound in expectation over the log, and its variance is dominated by fiber elements the proposal rarely visits. A small measured $\Delta$ is therefore consistent with two opposite worlds: (i) the model genuinely concentrates on canonical segmentations, or (ii) the proposal fails to find the high-probability non-canonical segmentations. The two are indistinguishable from the estimate alone, because there is no tractable exact reference for $|x| > \sim 15$ bytes — the exact sum needs a lattice DP, and a transformer's conditional depends on the full token prefix, not on a bounded state, so the DP does not factorize.

A second obstruction: **the evaluation does not measure what it names.** Perplexity comparisons across tokenizers are routinely presented as model comparisons; they are $\mathcal{L}_{\text{canon}}$ under different segmentations, differing by an unmeasured $\Delta$ that depends on the tokenizer. Bits-per-byte partly normalizes the length term but not the fiber-mass term.

## 7. Current Research (as of 2026)

- **Byte-exact inference layers.** Follow-on work to Phan et al. (2024) on exact next-byte conditionals, and to Vieira et al. (2024) on token-to-character conversion, aimed at making constrained decoding and grammar-guided generation tokenizer-invariant. Active at ETH Zurich / Johns Hopkins (Cotterell, Eisner groups) and in the structured-generation tooling community. *(frontier — verify)*
- **Tokenizer-free and byte-level models.** MegaByte, MambaByte, and the Byte Latent Transformer line (Meta AI, 2024) sidestep the fiber entirely by removing $V$. The open question is whether they pay a compute premium large enough to make the canonicality question worth solving instead.
- **Tokenizer transplant and vocabulary swapping.** Growing practical interest in re-tokenizing a trained model, which makes cross-tokenizer likelihood comparability urgent. *(frontier — verify)*
- **Theory of tokenization.** Gastaldi et al. (2024) on when a tokenizer is consistent/injective as a map on distributions; this is the formal groundwork for the theory variant above.

## 8. Concrete Next Experiment

**Question:** is the small measured canonicality gap a property of models or an artifact of the proposal distribution?

**Scale.** Two model sizes, $1.4$B and $7$B, both open-weight with a public BPE tokenizer (Pythia-1.4B and an open $7$B). Evaluation set: $2{,}000$ strings of $\le 12$ bytes each — short enough that $|\mathcal{T}(x)|$ can be enumerated **exactly** by lattice expansion (typically $10^2$–$10^5$ fibers). Stratified: 500 common English words, 500 rare English words, 500 code identifiers (`snake_case`, `camelCase`), 500 numerals of 3–8 digits. Cost estimate: $\le 10^8$ forward token positions, roughly a few hundred GPU-hours on A100-class hardware.

**Control arm.** For each string, compute three quantities: (1) exact $\mathcal{L}_{\text{marg}}$ by full enumeration; (2) the importance-sampling estimate $\hat{\mathcal{L}}_{\text{marg}}$ with $N = 128$ BPE-dropout samples at $p = 0.1$, the standard protocol; (3) $\mathcal{L}_{\text{canon}}$.

**Deciding number.** The **estimator shortfall**

$$S = \frac{\hat{\mathcal{L}}_{\text{marg}} - \mathcal{L}_{\text{marg}}^{\text{exact}}}{\mathcal{L}_{\text{canon}} - \mathcal{L}_{\text{marg}}^{\text{exact}}} \in [0,1],$$

the fraction of the true gap the standard estimator fails to recover, reported per stratum.

- $S < 0.1$ on all strata: the existing negative results stand; the gap is real and small, and canonicality bias is an inference-time problem only.
- $S > 0.5$ on any stratum: every published gap measurement using dropout proposals is an underestimate by at least $2\times$, and the "marginalization doesn't matter" conclusion is withdrawn for that stratum.

## 9. Key References

- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Taku Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL, 2018. — arXiv:1804.10959
- **[Foundational]** Ivan Provilkov, Dmitrii Emelianenko, Elena Voita. *BPE-Dropout: Simple and Effective Subword Regularization.* ACL, 2020. — arXiv:1910.13267
- **[Foundational]** Kris Cao, Laura Rimell. *You should evaluate your language model on marginal likelihood over tokenisations.* EMNLP, 2021.
- **[SOTA]** Nadezhda Chirkova, Germán Kruszewski, Jos Rozen, Marc Dymetman. *Should you marginalize over possible tokenizations?* ACL (short), 2023.
- **[SOTA]** Buu Phan, Marton Havasi, Matthew Muckley, Karen Ullrich. *Understanding and Mitigating Tokenization Bias in Language Models.* NeurIPS, 2024.
- **[SOTA]** Tim Vieira, Ben LeBrun, Mario Giulianelli, Juan Luis Gastaldi, Brian DuSell, John Terilla, Timothy J. O'Donnell, Ryan Cotterell. *From Language Models over Tokens to Language Models over Characters.* 2024.
- **[Theory]** Juan Luis Gastaldi, John Terilla, Luca Malagutti, Brian DuSell, Tim Vieira, Ryan Cotterell. *The Foundations of Tokenization: Statistical and Computational Concerns.* 2024.
- **[Empirical]** Kaj Bostrom, Greg Durrett. *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP, 2020.
- **[Empirical]** Valentin Hofmann, Hinrich Schütze, Janet B. Pierrehumbert. *An Embarrassingly Simple Method to Mitigate Undesirable Properties of Pretrained Language Model Tokenizers.* ACL, 2022.

## 10. Worked Example

Take the GPT-2 BPE vocabulary and the string $x = $ `" strawberry"` (11 bytes, leading space).

Canonical: $\tau(x) = $ `[" straw", "berry"]`, 2 tokens. The fiber $\mathcal{T}(x)$ also contains `[" straw", "ber", "ry"]`, `[" str", "aw", "berry"]`, `[" s", "traw", "berry"]`, and — because GPT-2 has byte fallback — the all-single-byte segmentation, among a few thousand others.

Now the inference-time failure. Ask the model to complete the prompt `" strawber"`. Its canonical encoding is `[" straw", "ber"]`. The model has *never* seen `"ber"` follow `" straw"` in training, because whenever those bytes occurred, the encoder emitted `["  straw", "berry"]` instead. So the conditional $p_\theta(\cdot \mid [" straw", "ber"])$ is trained only on strings where `"ber"` is a *complete* word-fragment — the model's mass moves to continuations like `" Peninsula"` or `"g"` (as in `Strawberg`), and away from `"ry"`, which is the byte-correct answer with probability near 1.

Measure it. The byte-exact conditional is
$$p(\texttt{"r"} \mid \texttt{" strawber"}) = \frac{\sum_{\mathbf{t} \in \mathcal{T}(\texttt{" strawberr"})} p_\theta(\mathbf{t})}{\sum_{\mathbf{t} \in \mathcal{T}(\texttt{" strawber"})} p_\theta(\mathbf{t})} \approx 0.99,$$
because almost the only English continuation is `strawberry`. The naive token-level conditional $p_\theta(\text{next token starts with } \texttt{"r"} \mid [" straw", "ber"])$ is typically far lower — a total-variation gap of tens of percentage points on a single byte.

**Where the obstruction becomes visible.** To confirm the $0.99$, you must sum over the full fiber of an 11-byte string. With BPE-dropout at $p=0.1$ as the proposal, `[" s","t","r","a","w","b","e","r","r","y"]` is drawn with probability under $10^{-6}$ — so it never appears in $N=128$ samples. If that all-bytes fiber element happens to carry non-trivial mass under $p_\theta$ (plausible: byte tokens are heavily trained), the estimator misses it and reports a gap near zero. The measurement cannot tell you whether the model concentrated on the canonical path or whether you simply never looked at the paths where the mass went. That is exactly the non-identifiability the experiment in §8 is designed to break, by choosing strings short enough to enumerate exactly.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*