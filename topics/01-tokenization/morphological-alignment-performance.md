---
id: 01-tokenization/morphological-alignment-performance
title: "Morphological Alignment versus Downstream Performance"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Morphological Alignment versus Downstream Performance

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/morphological-alignment-performance` · **Status:** open

## 1. Problem Statement

A subword tokenizer splits `unhappiness` into either `un|happi|ness` (morpheme-aligned) or `unh|appiness` (not). The folk claim is that the first is better for a language model. The problem is to establish whether that is true, and under what conditions.

Three variants, of very different difficulty:

- **Measurement.** Define a scalar $A(T)$ — morphological alignment of tokenizer $T$ — that is comparable across tokenizers, languages, and vocabulary sizes, and is not a proxy for compression rate.
- **Method.** Build a tokenizer that raises $A$ without raising bytes-per-token cost, and show it improves downstream loss or task accuracy at fixed compute.
- **Theory.** Prove or refute: for a transformer trained on $n$ tokens with a fixed parameter budget, morpheme-aligned segmentation lowers achievable cross-entropy per byte, or improves sample efficiency on morphologically productive generalization.

Solving it means: a stated causal claim of the form "increasing $A$ by $\delta$ at fixed bytes-per-token changes downstream metric $M$ by $\Delta$", with the compression confound controlled, replicated at two model scales.

## 2. Formal Setting

Let $\Sigma$ be a byte alphabet, $V \subset \Sigma^+$ a vocabulary of size $|V|$, and $T: \Sigma^* \to V^*$ a deterministic segmenter. For a corpus $D$ of $B$ bytes, define:

**Compression (fertility).** Bytes per token
$$\phi(T,D) = \frac{B}{\sum_{d \in D} |T(d)|}.$$
Measured directly by running the tokenizer over a held-out corpus. Reported instead as *fertility* $1/\phi$ (tokens per word) in much of the literature; the two are not interchangeable across languages because word length varies.

**Alignment.** Let $G$ be a gold morpheme-segmentation lexicon: pairs $(w, s_w)$ with $s_w = (m_1,\dots,m_k)$ a boundary set $\partial s_w \subset \{1,\dots,|w|-1\}$. With $\partial T(w)$ the boundary positions induced by the tokenizer,
$$A(T; G) = \frac{1}{|G|}\sum_{(w,s_w)\in G} F_1\big(\partial T(w),\, \partial s_w\big).$$
Boundary-$F_1$, not exact-match, because exact-match saturates at 0 for any tokenizer that keeps frequent whole words intact. Measured over a fixed lexicon (MorphyNet, UniMorph-derived segmentations, SIGMORPHON 2022 data).

**Downstream metric.** Bits per byte on held-out text,
$$\mathrm{BPB} = \frac{1}{B\ln 2}\sum_{i} -\ln p_\theta(t_i \mid t_{<i}),$$
which is the only loss comparable across tokenizers — per-token perplexity is not, since it depends on $\phi$. Plus task accuracy on a fixed suite.

**The causal question.** Over a family of tokenizers, estimate
$$\beta = \frac{\partial\, \mathbb{E}[\mathrm{BPB}]}{\partial A}\bigg|_{\phi,\;|V|,\;C \text{ fixed}}$$
with $C$ the training FLOPs. The conditioning is the entire difficulty: $A$ and $\phi$ are negatively coupled in practice, since morpheme-faithful splits are longer.

**Assumptions, and which are violated.**
1. *A gold segmentation exists and is unique.* Violated: `-ing` in `building` (noun) versus `building` (verb); fusional morphs in Semitic templatic morphology have no contiguous boundary at all.
2. *Alignment is language-independent.* Violated: $A$ is computed against a lexicon whose coverage differs by two orders of magnitude between English and, say, Amharic.
3. *The tokenizer is deterministic and context-free.* Violated by subword regularization (Kudo, 2018) and by any inference-time sampling.
4. *Downstream tasks probe morphology.* Mostly false: MMLU-style benchmarks are dominated by knowledge, not composition.

## 3. State of the Art

**Established.**
- Bostrom & Durrett (Findings of EMNLP 2020) trained RoBERTa-scale (~125M) English models on identical data with BPE versus unigram-LM vocabularies of matched size. Unigram-LM recovers gold morphological boundaries far more often; downstream differences on GLUE/SQuAD are small and positive, typically under one point. This is the cleanest controlled arm in the literature, and it is single-scale, single-language.
- Rust et al. (ACL 2021) showed that replacing mBERT's shared tokenizer with a language-specific one recovers most of the monolingual-versus-multilingual gap. The mediator there is fertility, not alignment; the paper does not separate them.
- Schmidt et al., *Tokenization Is More Than Compression* (EMNLP 2024), trained 350M-parameter models over ~24 tokenizer configurations and found corpus token count does **not** reliably predict downstream accuracy. This is negative evidence against the strongest form of the compression-only hypothesis.

**Claimed but unablated.**
- Morphology-injected tokenizers (MorphPiece; morphologically-guided BPE variants) report downstream gains, but the comparison arms differ in $\phi$, in vocabulary construction corpus, or in both. No published morphology-aware tokenizer has been shown to beat a $\phi$-matched, $|V|$-matched baseline at fixed FLOPs, at more than one scale.
- Arnett & Bergen (COLING 2025) built *MorphScore* and reported that cross-language performance gaps track dataset size and fertility, not morphological alignment. Correlational, over pretrained checkpoints not trained for the comparison.

**Benchmark-number-only.** Most "our tokenizer is more morphological" results live entirely as a table of GLUE/XNLI numbers from a single seed. Seed variance at 100M–350M scale is typically ±0.5–1.0 accuracy points, which is the size of the reported effect.

## 4. What Is Known

- **Compression and downstream performance correlate, but weakly and task-dependently.** Goldman et al. (Findings of ACL 2024) found compression predicts generation-task performance better than classification performance, over models up to ~1B.
- **Tokenizer choice matters at scale for cost, clearly.** Ali et al. (NAACL Findings 2024) trained 2.6B and 6.7B models and reported that tokenizer choice changes training cost by tens of percent for multilingual setups, with monolingual tokenizers degrading multilingual downstream results.
- **Alignment and compression trade off.** At $|V| = 32$k English, morpheme-faithful segmentation increases token count on productive derivational vocabulary by roughly 20–40% relative to BPE, because BPE stores `-ization`-type strings whole.
- **Inference method matters as much as vocabulary.** Uzan et al. (ACL 2024) showed greedy longest-match inference over a fixed vocabulary changes both $A$ and $\phi$ substantially — so "the tokenizer" is a (vocabulary, inference rule) pair, and papers vary the pair while naming only one half.
- **Gold morphology helps when the task is morphology.** Hofmann et al. (ACL 2021) showed derivationally-correct splits improve complex-word classification at BERT-base scale by several points. This is the strongest positive result, and it is on a task selected to require composition.

## 5. What Is Not Known

- **Methodologically blocked:** whether $A$ measures anything tokenizer-intrinsic. Every published alignment score is a boundary-agreement statistic against a lexicon with unknown coverage bias, undefined for templatic and heavily fusional morphology, and unnormalized against the alignment a random segmenter of the same $\phi$ would achieve. Until $A$ is defined relative to a $\phi$-matched null, cross-tokenizer comparisons of $A$ are not interpretable.
- **Empirically open:** the sign and magnitude of $\beta$ at fixed $\phi$. The experiment is a small grid of pretraining runs at 1B parameters — expensive but entirely runnable. Nobody has published it with $\phi$ held fixed.
- **Theoretically open:** whether any statement of the form "morpheme-aligned segmentation strictly lowers achievable BPB" can hold. A transformer with enough capacity can compose subwords; there is no theorem that a misaligned segmentation costs anything beyond a sequence-length term.

## 6. Why It Is Hard

**Confounded measurement, with the confounder structurally coupled to the treatment.** You cannot raise $A$ without changing $\phi$, $|V|$, the token-frequency distribution, and the effective number of training tokens at once. A run with higher $A$ sees fewer bytes per step, so at fixed step count it trains on less text; at fixed bytes it uses more FLOPs. Both corrections are themselves confounds.

Second: **absent ground truth.** $G$ is a human artifact. Two morphologists disagree on `unhappiness` versus `un|happiness` (is `-ness` attached first?), and the disagreement rate is not reported in any alignment paper.

Third: **the evaluation does not measure what it names.** Downstream suites are chosen for coverage, not for morphological productivity. A tokenizer change that genuinely improves handling of unseen derived forms moves such suites by well under the seed noise floor.

## 7. Current Research (as of 2026)

- **$\phi$-controlled tokenizer ablation grids** at 1B scale, using tokenizers constructed to hit a target bytes-per-token — the direct attack on $\beta$ *(frontier — verify)*.
- **Tokenizer-free and byte-level models** (MambaByte, Byte Latent Transformer-style dynamic patching) sidestep the question by letting the model learn boundaries; the interesting measurement is whether learned patch boundaries land on morphemes.
- **Cross-lingual alignment metrics** with null-model normalization — Pinter's group (Ben-Gurion), Bergen's group (UCSD), and the SIGMORPHON community.
- **Tokenizer transfer** (Minixhofer et al., NeurIPS 2024) makes it cheap to swap vocabularies into a trained model, which offers a cheaper — if less clean — way to vary $A$ post hoc.

## 8. Concrete Next Experiment

**Scale.** Four 1.3B-parameter decoder models, identical architecture, identical 100B-byte English+Turkish corpus, identical optimizer and seed set (3 seeds each; 12 runs).

**Arms.** Construct four tokenizers at $|V| = 48$k, all tuned to $\phi = 4.0 \pm 0.05$ bytes/token on held-out text:
1. BPE (low $A$).
2. Unigram-LM (mid $A$).
3. Morphology-supervised vocabulary built from MorphyNet segmentations (high $A$).
4. **Control arm:** a *scrambled-morphology* tokenizer — take arm 3's vocabulary and permute merge decisions so $A$ falls to arm 1's level while $\phi$, $|V|$, and the token unigram entropy stay matched to arm 3.

Arm 4 is the point of the design: it isolates alignment from every distributional property that usually travels with it.

**Deciding number.** BPB on held-out Turkish Wikipedia, plus exact-match accuracy on a held-out set of 5,000 *novel* derived forms (attested, but zero-count in the pretraining corpus) in a cloze completion.

Decision rule: if arm 3 minus arm 4 differs by $\geq 0.010$ BPB, or $\geq 3$ points on novel-derived-form accuracy, with 95% CIs over seeds excluding zero, morphological alignment has an effect independent of compression. If $|{\Delta}| < 0.003$ BPB and $<1$ point, the alignment hypothesis is dead at this scale and the field should stop reporting $A$ as a tokenizer quality signal.

## 9. Key References

- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Taku Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL, 2018. — arXiv:1804.10959
- **[SOTA]** Kaj Bostrom, Greg Durrett. *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP, 2020. — arXiv:2004.03720
- **[SOTA]** Craig W. Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omar Uzan, Yuval Pinter, Chris Tanner. *Tokenization Is More Than Compression.* EMNLP, 2024.
- **[SOTA]** Omer Goldman, Avi Caciularu, Matan Eyal, Kris Cao, Idan Szpektor, Reut Tsarfaty. *Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance.* Findings of ACL, 2024.
- Mehdi Ali et al. *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL, 2024.
- Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, Iryna Gurevych. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL, 2021.
- Valentin Hofmann, Janet B. Pierrehumbert, Hinrich Schütze. *Superbizarre Is Not Superb: Derivational Morphology Improves BERT's Interpretation of Complex Words.* ACL, 2021.
- Omri Uzan, Craig W. Schmidt, Chris Tanner, Yuval Pinter. *Greed is All You Need: An Evaluation of Tokenizer Inference Methods.* ACL, 2024.
- Lisa Beinborn, Yuval Pinter. *Analyzing Cognitive Plausibility of Subword Tokenization.* EMNLP, 2023.
- Catherine Arnett, Benjamin K. Bergen. *Why do language models perform worse for morphologically complex languages?* COLING, 2025.
- **[Survey]** Khuyagbaatar Batsuren et al. *The SIGMORPHON 2022 Shared Task on Morpheme Segmentation.* SIGMORPHON Workshop, 2022.

## 10. Worked Example

Take the English word **`unfriendliness`** (14 bytes). Gold segmentation: `un|friend|li|ness`, boundaries $\partial s = \{2, 8, 10\}$.

GPT-2 BPE splits it `un|friend|liness` → $\partial T = \{2, 8\}$. Precision $2/2 = 1.0$, recall $2/3 = 0.667$, $F_1 = 0.80$, at 3 tokens.

A morphology-supervised tokenizer gives all three boundaries: $F_1 = 1.0$, at 4 tokens.

Now the obstruction. Over the 5,000-word derivational probe set, suppose alignment rises from $A = 0.62$ to $A = 0.89$ and the token count rises from 3.1 to 3.9 per word — a 26% increase, so $\phi$ falls from 4.4 to 3.5 bytes/token on this slice. At a fixed 100B-token budget, the aligned tokenizer sees 350 GB of text where the BPE tokenizer sees 440 GB. That is a 20% data reduction. Chinchilla-style scaling gives roughly $L \propto D^{-0.28}$ in the data-limited regime, so a 20% data cut costs about $1 - 0.8^{0.28} \approx 6\%$ of the loss-above-entropy gap — on the order of 0.02–0.04 BPB at 1B scale.

The measured alignment gain would have to buy back more than that before it shows any net benefit. Every published comparison that reports "morphological tokenizer wins by 0.4 accuracy points" has this term uncontrolled and unreported, with the same sign ambiguity: the aligned arm is simultaneously better-segmented and data-starved. The scrambled-morphology control in §8 exists precisely to cancel it — arm 4 pays the identical 20% data tax with none of the alignment.