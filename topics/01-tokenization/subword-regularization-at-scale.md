---
id: 01-tokenization/subword-regularization-at-scale
title: "Subword Regularization Gains at Large Scale"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Subword Regularization Gains at Large Scale

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/subword-regularization-at-scale` · **Status:** empirically-open

## 1. Problem Statement

Subword regularization trains a model on *samples* from the set of valid segmentations of each string, rather than on the single canonical segmentation the tokenizer emits. Kudo (2018) and BPE-dropout (Provilkov et al., 2020) established gains of roughly 0.5–3 BLEU on neural machine translation at 10⁵–10⁷ sentence-pair scale. Essentially no production LLM uses it: Llama, Qwen, GPT-class and Gemma tokenizers are deterministic at train time.

The problem: **does subword regularization still help at 10⁹-parameter, 10¹¹-token pretraining scale, and if so, along which axis?**

Three variants, different difficulty:

- **Measurement.** Under a fixed FLOP budget, does stochastic segmentation lower held-out **bits-per-byte** relative to a deterministic control? Bits-per-byte, not per-token cross-entropy — the latter is not comparable across segmentation distributions.
- **Method.** If gains exist, are they a *data multiplier* (equivalent to more unique tokens, so they shrink as $D$ grows), a *capacity/robustness* effect (they persist), or a *repeated-data* effect (they appear only in multi-epoch regimes)?
- **Theory.** Is training on $\mathbb{E}_{q}[\text{loss}]$ a consistent estimator of anything a deterministic-tokenizer model is not, or is it strictly a regularizer whose benefit vanishes as $N,D \to \infty$?

Solving it means a published scaling-law fit that separates these three, not a single win/loss at one scale.

## 2. Formal Setting

Let $\Sigma$ be a byte alphabet, $V \subset \Sigma^+$ a vocabulary of size $|V|$, and $s \in \Sigma^*$ a string. Write $\mathcal{T}(s) = \{x \in V^* : \text{concat}(x) = s\}$ for the **segmentation lattice**. $|\mathcal{T}(s)|$ is exponential in $|s|$; it is computed only implicitly, by dynamic programming over the lattice in $O(|s| \cdot L_{\max})$ where $L_{\max}$ is the longest token.

**Sampling distribution.** For a unigram LM tokenizer with token log-probabilities $\log p_u(v)$, the training-time segmentation sampler is
$$q_\alpha(x \mid s) \;=\; \frac{p_u(x)^{\alpha}}{\sum_{x' \in \mathcal{T}(s)} p_u(x')^{\alpha}}, \qquad p_u(x) = \prod_i p_u(x_i),$$
with $\alpha \to \infty$ recovering the Viterbi (deterministic) segmentation and $\alpha \to 0$ giving uniform-over-lattice. In practice one samples from the $l$-best list ($l \approx 64$) rather than the full lattice. For BPE-dropout, $q_p$ is induced by dropping each merge with probability $p$ during the greedy merge loop; there is no closed form for $q_p$, which matters below.

**Training objective.**
$$\mathcal{L}_{\text{SR}}(\theta) = -\mathbb{E}_{s \sim \mathcal{D}} \; \mathbb{E}_{x \sim q(\cdot \mid s)} \big[ \log p_\theta(x) \big].$$
This is an upper bound on the negative *marginal* log-likelihood $-\log \sum_{x \in \mathcal{T}(s)} p_\theta(x)$ only up to the entropy term of $q$; the two objectives are not the same, and the gap is not controlled.

**Measurement.** All comparisons in bits per UTF-8 byte on a held-out corpus:
$$\text{bpb} = \frac{1}{\ln 2}\cdot\frac{-\log p_\theta(x^\star(s))}{|s|_{\text{bytes}}}, \qquad x^\star(s) = \arg\max_{x} p_u(x),$$
i.e. scoring the *deterministic* segmentation even for stochastically trained models, so that the two arms are compared on identical inputs. The marginal variant replaces the numerator with $-\log\sum_{x}p_\theta(x)$, estimated by importance sampling over $q$ (Cao & Rimell, EMNLP 2021).

**Scaling hypothesis to test.** Fit each arm to
$$L(N,D) = E + \frac{A}{N^{a}} + \frac{B}{D^{b}}$$
(Hoffmann et al., 2022, in bpb). A pure data multiplier means the SR arm satisfies $L_{\text{SR}}(N,D) \approx L_{\text{det}}(N, \rho D)$ for some $\rho > 1$ with $E, A, B, a, b$ shared — an effect that shrinks in absolute bpb as $D$ grows. A capacity effect means $E$ or $A$ differs.

**Assumptions, and which are violated.**
1. *$q$ is a fixed, string-independent noise process.* Violated: $q_\alpha$ depends on $p_u$, which is fitted to the same corpus, so the noise is correlated with the data.
2. *Train and test segmentation distributions match.* Deliberately violated — that is the point of the method, and it makes $\mathcal{L}_{\text{SR}}$ a biased estimator of test bpb.
3. *Segmentation is the only thing varying.* Violated in most published comparisons, which change tokenizer algorithm (BPE vs unigram) at the same time as adding stochasticity.
4. *Effective vocabulary is unchanged.* Violated: BPE-dropout raises token-frequency entropy and effectively upweights short tokens, so sequences lengthen by 5–20% at $p=0.1$, changing FLOPs per document.

## 3. State of the Art

**Established (ablated, reproduced).**
- Kudo (ACL 2018): unigram-LM subword regularization improves BLEU on WMT/IWSLT NMT, with gains concentrated in low-resource pairs; the $\alpha$ and $l$-best ablations are in the paper.
- Provilkov, Emelianenko & Voita (ACL 2020): BPE-dropout beats BPE on 8 language pairs; ablated over dropout rate $p$, corpus size, and vocabulary size, and shows gains *decay monotonically with corpus size* — the single most relevant finding for this problem.
- Wang, Ruder & Neubig (NAACL 2021): multi-view subword regularization improves cross-lingual transfer for XLM-R-scale encoders.

**Claimed but unablated at LLM scale.** The proposition "subword regularization would help pretraining LLMs" has no controlled public result. Absence of the technique from Llama/Qwen/Gemma tokenizers is a *choice*, not an ablation; no released technical report publishes a compute-matched SR-vs-deterministic pretraining comparison.

**Benchmark-number-only results.** Most tokenizer comparisons at scale (e.g. Dagan, Synnaeve & Rozière, ICML 2024, on tokenizers for code pretraining; Tao et al., NeurIPS 2024, on vocabulary-size scaling) report downstream benchmark deltas without a segmentation-stochasticity arm. Treat any inference about SR drawn from them as unsupported.

**Adjacent theory SOTA.** Cao & Rimell (EMNLP 2021) show one-best scoring understates a model's true likelihood and argue evaluation should marginalize. Chirkova et al. (ACL 2023) find that marginalizing at inference gives small and often negligible gains for models trained deterministically. Zouhar et al. (ACL 2023) give an information-theoretic ("noiseless channel") predictor of tokenizer quality — a *deterministic* criterion with no stochastic analogue.

## 4. What Is Known

- **Gains shrink with data.** BPE-dropout's own ablation (ACL 2020): on English↔German, improvements of ~2 BLEU at 10k sentence pairs fall to under ~0.5 BLEU at 4M+ pairs, and the paper reports the effect largely disappearing on the largest settings. Scale: 10⁴–10⁷ sentence pairs, ≤300M-parameter Transformers.
- **Optimal noise is small.** Best BPE-dropout rates cluster at $p \in [0.05, 0.1]$; unigram $\alpha \approx 0.2$–$0.5$ with $l \approx 64$ (Kudo 2018; Provilkov et al. 2020). Larger noise degrades.
- **Robustness gains are real and separate from clean accuracy.** SR-trained NMT models degrade less under misspellings and under source-side segmentation perturbation (Provilkov et al. 2020). Scale: IWSLT/WMT.
- **The one-best/marginal gap is nonzero.** Cao & Rimell (2021) measure it directly for unigram-LM-tokenized LMs; it is large enough to change model rankings in some settings. Scale: sub-1B models.
- **Tokenizer choice affects downstream loss at LLM scale.** Dagan et al. (ICML 2024), up to 7B parameters and hundreds of billions of tokens, show tokenizer swaps move code benchmarks by several points. This establishes that the *tokenizer* still matters at scale; it says nothing about *stochasticity*.

## 5. What Is Not Known

- **Empirically open (primary).** Whether a compute-matched SR arm beats a deterministic control at $N \ge 1$B, $D \ge 20$B tokens, in bpb. Nobody has published the run. It costs roughly a few thousand A100-hours per point — affordable, just not done in public.
- **Empirically open (secondary).** Whether SR gains reappear in the **repeated-data** regime (multi-epoch training on a fixed corpus, where extra segmentations are extra effective uniqueness). The BPE-dropout decay curve predicts gains vanish with unique data, but says nothing about epochs 2–8.
- **Theoretically open.** No proof that $\mathbb{E}_{q}$-training either does or does not asymptotically dominate deterministic training under a well-specified model. There is no known bound relating the excess risk of the SR objective to the marginal-likelihood objective it is often assumed to approximate.
- **Methodologically blocked.** $q_p$ for BPE-dropout has no tractable density, so importance-weighted marginal likelihood — the only tokenizer-independent likelihood — cannot be computed exactly for BPE-dropout models. This is why cross-arm comparisons default to one-best bpb, which is biased *against* the stochastic arm.

## 6. Why It Is Hard

Three specific obstructions.

1. **Confounded compute accounting.** BPE-dropout lengthens sequences by 5–20% at $p=0.1$. Matching token counts un-matches FLOPs and un-matches bytes seen; matching bytes un-matches steps. Every published NMT comparison matches epochs, which is none of these. At LLM scale, a 10% sequence-length difference is larger than the effect being measured.
2. **Evaluation that does not measure what it names.** Per-token perplexity is not comparable across segmentation distributions — a model can look better simply by producing longer sequences of easier tokens. Bits-per-byte fixes this but requires committing to one scoring segmentation, which reintroduces bias. The unbiased fix (marginal likelihood) is intractable for the most-used method (see §5).
3. **The effect is a small difference of large numbers, in a regime where the prior says it is zero.** The known corpus-size decay means the expected effect at 10¹¹ tokens is at or below seed noise ($\sim$0.002–0.005 bpb between pretraining seeds at 1B scale). Detecting it needs multiple seeds per arm, multiplying the cost by 3–5×.

## 7. Current Research (as of 2026)

- **Tokenizer-free and byte-level models** (ByT5, Xue et al., TACL 2022; CANINE, Clark et al., TACL 2022; and the byte-latent / dynamic-patching line at Meta AI, 2024–25) sidestep the question: if segmentation is learned or absent, subword regularization has nothing to regularize. *(frontier — verify current results.)*
- **Inference-time ensembling over segmentations** — the Chirkova et al. (2023) direction — remains the cheapest live probe, and is being revisited for robustness and for arithmetic/code tasks where canonical tokenization is known to be pathological. *(frontier — verify.)*
- **Vocabulary scaling laws** (Tao et al., NeurIPS 2024) fit $|V|$ jointly with $N$ and $D$; extending that fit with a stochasticity parameter $\alpha$ or $p$ is the natural next paper and, as far as public work goes, unwritten.
- **Tokenizer-quality intrinsics** (Zouhar et al., ACL 2023; Schmidt et al., EMNLP 2024, *Tokenization Is More Than Compression*) are converging on the finding that compression rate alone is a poor predictor of downstream loss — which weakens the main indirect argument that SR should not matter.

## 8. Concrete Next Experiment

**Scale.** 1.0B-parameter decoder-only Transformer, $|V| = 32{,}768$ unigram-LM (SentencePiece) tokenizer fitted once on the control corpus and frozen. Two data regimes: (a) **single-epoch**, 20B unique tokens; (b) **repeated**, 5B unique tokens × 4 epochs. Three seeds per arm. Total: 6 configurations × 3 seeds ≈ 18 runs, roughly 2×10²¹ FLOPs, order 5,000 A100-hours.

**Arms.**
- **Control:** deterministic Viterbi segmentation ($\alpha = \infty$).
- **Treatment A:** $q_\alpha$ sampling with $\alpha = 0.2$, $l = 64$, resampled every epoch.
- **Treatment B:** $\alpha = 0.2$ but with the *sequence-length* difference absorbed by truncating to a fixed byte budget per document, so all arms see identical bytes and identical FLOPs.

Same optimizer, schedule, and total FLOPs across arms; log bytes seen, tokens seen, and FLOPs separately for every arm.

**The deciding number.** Held-out **bits-per-byte on a common byte-aligned test set, scored with the deterministic segmentation for all arms**, at matched FLOPs. Decision rule: the treatment wins if $\Delta\text{bpb} = \text{bpb}_{\text{det}} - \text{bpb}_{\text{SR}} > 0.005$ with the 3-seed intervals disjoint. Below that, and given the corpus-size decay from Provilkov et al., the honest conclusion is *no effect at scale*.

**Secondary readout.** Fit $\rho$ in $L_{\text{SR}}(N,D) \approx L_{\text{det}}(N,\rho D)$ from the two data regimes. If $\rho > 1$ in the repeated regime and $\rho \approx 1$ in the single-epoch regime, subword regularization is a data-repetition remedy, not a regularizer — which is a publishable, actionable answer either way.

## 9. Key References

- **[Foundational]** Taku Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL 2018. — arXiv:1804.10959
- **[Foundational]** Taku Kudo, John Richardson. *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.* EMNLP 2018 (System Demonstrations). — arXiv:1808.06226
- **[SOTA]** Ivan Provilkov, Dmitrii Emelianenko, Elena Voita. *BPE-Dropout: Simple and Effective Subword Regularization.* ACL 2020. — arXiv:1910.13267
- **[SOTA]** Kris Cao, Laura Rimell. *You should evaluate your language model on marginal likelihood over tokenisations.* EMNLP 2021. — arXiv:2109.02550
- **[SOTA]** Nadezhda Chirkova, Germán Kruszewski, Jos Rozen, Marc Dymetman. *Should you marginalize over possible tokenizations?* ACL 2023 (short).
- Xinyi Wang, Sebastian Ruder, Graham Neubig. *Multi-view Subword Regularization.* NAACL 2021. — arXiv:2103.08490
- Tatsuya Hiraoka. *MaxMatch-Dropout: Subword Regularization for WordPiece.* COLING 2022.
- Vilém Zouhar, Clara Meister, Juan Luis Gastaldi, Li Du, Mrinmaya Sachan, Ryan Cotterell. *Tokenization and the Noiseless Channel.* ACL 2023.
- Craig W. Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omri Uzan, Yuval Pinter, Chris Tanner. *Tokenization Is More Than Compression.* EMNLP 2024.
- Gautier Dagan, Gabriel Synnaeve, Baptiste Rozière. *Getting the most out of your tokenizer for pre-training and domain adaptation.* ICML 2024. — arXiv:2402.01035
- Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS 2024. — arXiv:2407.13623
- **[Survey]** Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, Iryna Gurevych. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021.
- Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556

## 10. Worked Example

Take the string `unbelievably` with a 32k unigram vocabulary. Its lattice contains hundreds of segmentations; the top four by $p_u$ might be

| segmentation | $\log p_u$ | $q_{0.2}$ |
|---|---|---|
| `▁unbeliev` `ably` | −11.2 | 0.34 |
| `▁un` `believ` `ably` | −12.9 | 0.24 |
| `▁unbelievable` `y`\* | −14.1 | 0.19 |
| `▁un` `be` `liev` `ably` | −15.8 | 0.14 |

(\*illustrative; probabilities normalized over the 64-best list.) With $\alpha = 0.2$ the sampler is nearly flat over the top four — the intended effect. Mean length rises from 2.00 tokens to about 2.71, a **35% increase for this word**, and about 8% averaged over English text at $p_u$ from a 32k vocabulary.

Now carry it to the scaling question. Suppose a 1B-parameter control run reaches 0.720 bpb at 20B tokens. The BPE-dropout decay curve (Provilkov et al. 2020) puts the expected gain at this corpus size at or below the 4M-sentence-pair point where the effect measured under 0.5 BLEU — translated to bpb, plausibly $\Delta \approx 0.002$. Seed-to-seed standard deviation at this scale is also about 0.002 bpb. So a single-seed comparison has roughly the power of a coin flip.

Meanwhile, the 8% length increase means the SR arm, at equal *step* count, consumes 8% fewer bytes — and the $D^{-b}$ term with $b \approx 0.28$ predicts a bpb *penalty* of about $0.28 \times 0.08 \times (B/D^b) \approx 0.004$ bpb. **The accounting artefact is twice the size of the effect and points the other way.** That is the obstruction in one number: without byte-matched, FLOP-matched, multi-seed arms, the experiment will confidently report whichever sign its bookkeeping chose.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*