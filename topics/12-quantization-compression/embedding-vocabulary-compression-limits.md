---
id: 12-quantization-compression/embedding-vocabulary-compression-limits
title: "Embedding and Vocabulary Compression Limits"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Embedding and Vocabulary Compression Limits

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/embedding-vocabulary-compression-limits` · **Status:** open

## 1. Problem Statement

The input embedding matrix and the output (unembedding/softmax) matrix of a language model hold $V \times d$ parameters. At $V = 128{,}000$ and $d = 4096$ this is $\approx 1.05\text{B}$ parameters per matrix — for a 7B model, roughly 15–30% of all weights, and a larger fraction of a small model. Unlike transformer blocks, these matrices are accessed sparsely (one row per token in, all rows out) and their weight distribution is dominated by token frequency, which is Zipfian.

**The question:** how far can $V \times d$ be compressed — by low-rank factorization, quantization, codebooks, tied weights, or vocabulary reduction — before task loss degrades beyond a stated tolerance, and what sets that floor?

Three variants, routinely conflated:

- **Measurement.** Given a compressed embedding/unembedding pair, what number reports the damage? Perplexity on a held-out corpus is dominated by high-frequency tokens; the loss from destroying rare-token geometry is nearly invisible in it.
- **Method.** Find the compression map achieving the best rate–distortion tradeoff at a given deployment budget.
- **Theory.** Prove a lower bound: for a model of quality $q$ over vocabulary $V$, at least $B(V, q)$ bits are needed in the embedding tables. No such bound exists that is both non-vacuous and predictive.

Solving it means: a rule that, given $(V, d, \text{model scale}, \text{tolerance }\epsilon)$, predicts the minimum embedding-table bits, and is verified to hold across at least two model families.

## 2. Formal Setting

Let $\mathcal{V}$ be a vocabulary, $|\mathcal{V}| = V$; $E \in \mathbb{R}^{V \times d}$ the input embedding; $U \in \mathbb{R}^{V \times d}$ the unembedding ($U = E$ when tied). A compression scheme is a pair of maps $(\mathcal{C}, \mathcal{D})$ with $\hat{E} = \mathcal{D}(\mathcal{C}(E))$, and a **rate**

$$R = \frac{|\mathcal{C}(E)| \text{ in bits}}{V d}\ \ \text{(bits per parameter)},$$

measured as the serialized size of *everything* needed at inference — codebooks, scales, indices, factor matrices — divided by $Vd$. Codebook and scale overhead is what most reported rates omit.

**Distortion** has three candidate definitions, and they disagree:

1. Weight-space: $\ \delta_W = \|E - \hat{E}\|_F / \|E\|_F$.
2. Corpus loss: $\ \delta_L = \mathbb{E}_{x \sim \mathcal{D}}[-\log p_{\hat{\theta}}(x)] - \mathbb{E}_{x \sim \mathcal{D}}[-\log p_{\theta}(x)]$, in nats/token.
3. Frequency-stratified loss: partition $\mathcal{V}$ into frequency deciles $\mathcal{V}_1 \dots \mathcal{V}_{10}$ by unigram count and report $\delta_L^{(k)} = \mathbb{E}[\Delta \text{NLL} \mid \text{target} \in \mathcal{V}_k]$.

Under a Zipfian unigram law $p(v) \propto \text{rank}(v)^{-\alpha}$ with $\alpha \approx 1$, the top 1% of tokens carry the large majority of token mass, so $\delta_L$ is a frequency-weighted average that essentially ignores $\mathcal{V}_{9}, \mathcal{V}_{10}$. Setting $\delta_L \le 0.01$ nats is compatible with $\delta_L^{(10)} > 0.5$ nats.

The **problem** is: characterize the frontier

$$R^\star(\epsilon) = \min \{ R : \delta_L(\hat{E}) \le \epsilon \}$$

and its stratified counterpart $R^\star_k(\epsilon)$, as a function of $V$, $d$, and non-embedding parameter count $N$.

**Assumptions, and which are violated:**

- *Embedding rows are approximately isotropic.* Violated — LM embedding spaces are anisotropic, with a dominant common direction and frequency-correlated norm (Gao et al., ICLR 2019; Ethayarajh, EMNLP 2019).
- *Distortion in $E$ maps monotonically to $\delta_L$.* Violated — a few outlier coordinates carry disproportionate loss (Dettmers et al., NeurIPS 2022, LLM.int8()).
- *$E$ and $U$ have the same compressibility.* Not established; $U$ feeds a softmax over all $V$ rows and is empirically more fragile.
- *Fixed $V$.* Violated by the design question: changing the tokenizer changes $V$, sequence length, and downstream loss jointly.

## 3. State of the Art

**Established (ablated, reproduced):**

- **Weight tying** ($U = E$): halves embedding parameters and *improves* perplexity at small scale (Press & Wolf, EACL 2017; Inan et al., ICLR 2017). Reproduced widely; still used in Gemma and most sub-3B models.
- **Factorized embedding** $E = A B$, $A \in \mathbb{R}^{V \times e}$, $B \in \mathbb{R}^{e \times d}$, $e \ll d$: ALBERT (Lan et al., ICLR 2020) uses $e=128$ with $V \approx 30{,}000$, $d$ up to 4096, with the embedding table shrinking from $Vd$ to $V e + e d$ — an ~$8\times$ reduction at $d=1024$ — at neutral-to-small GLUE cost when combined with cross-layer sharing.
- **Adaptive input/softmax** (Grave et al., ICML 2017; Baevski & Auli, ICLR 2019): assign frequency-dependent dimension $d_k$ per cluster. Established as a *strict improvement* in the perplexity-per-parameter tradeoff on WikiText-103 / Billion Word, not merely a speedup.
- **Compositional code learning** (Shu & Nakayama, ICLR 2018): $M$ codebooks of $K$ entries each, rate $M\log_2 K$ bits per token; reported 94–98% embedding compression with no loss on sentiment classification and machine translation (up to ~50M-parameter models).
- **Product quantization** for retrieval embeddings (Jégou et al., TPAMI 2011) and binary hashing (Yamada et al., ACL 2021, BPR: 32× index compression at ~2 point EM cost on Natural Questions).
- **Matryoshka Representation Learning** (Kusupati et al., NeurIPS 2022): nested prefixes let one $d=2048$ representation be truncated to $d=64$ with retrieval accuracy matched to independently trained low-$d$ models; reported up to $14\times$ faster large-scale retrieval at comparable accuracy.

**Claimed but unablated / benchmark-only:**

- Post-training quantization papers routinely *exclude* embeddings from the quantized set and report the model-wide compression ratio anyway. When embeddings are 20–30% of a small model, this inflates the headline.
- "4-bit embeddings are free" appears in llama.cpp-style deployment lore (Q4 token-embedding tensors) with perplexity as the only reported metric — a benchmark number, not an ablation, and precisely the metric blind to rare tokens.
- Claims that larger vocabularies are strictly better compute-efficient rest on fitted scaling laws, not on held-out extrapolation tests.

## 4. What Is Known

- **Optimal vocabulary grows sublinearly with model size.** Tao et al. (NeurIPS 2024, *Scaling Laws with Vocabulary*) fit $V_{\text{opt}} \propto N_{\text{nv}}^{\gamma}$ with $\gamma \approx 0.83$ over models from 33M to 3B non-vocabulary parameters, and predict a compute-optimal vocabulary of $\approx 216$K for a Llama-2-70B-scale budget against the 32K actually used; they verify empirically at 3B that 35K beats 32K under matched FLOPs. Scale: $\le 3$B trained, larger extrapolated.
- **Tokenizer choice moves downstream accuracy by single-digit points at fixed compute** (Ali et al., NeurIPS 2024, *Tokenizer Choice for LLM Training*), measured on 2.6B-parameter multilingual models — so vocabulary is not a free knob.
- **Outlier features dominate quantization damage** at $\ge$ 6.7B parameters (Dettmers et al., NeurIPS 2022); below that scale the phenomenon is largely absent, so small-scale embedding-quantization results do not transfer upward.
- **Bit-allocation regularity:** for weight-only PTQ of transformer blocks, 4-bit is near the accuracy-per-bit optimum across 19M–176B models (Dettmers & Zettlemoyer, ICML 2023). This was measured on *all* weights, not on embedding tables in isolation.
- **JL lower bound.** For $n$ points and distortion $\varepsilon$, dimension $\Omega(\varepsilon^{-2}\log n)$ is necessary (Larsen & Nelson, 2017) — with $n = V = 128$K and $\varepsilon = 0.1$, this gives $d \gtrsim 10^3$ *if* pairwise distances must be preserved. Which they need not be: the model only needs the logit ordering.

## 5. What Is Not Known

- **Theoretically open.** No non-vacuous lower bound on embedding-table bits as a function of $(V, d, \epsilon)$. The JL bound is the wrong bound — it preserves all pairwise distances, whereas the model needs only that the argmax and top-$k$ of $Ux$ be preserved for on-distribution $x$. No rate–distortion theory exists for "preserve the top-$k$ of a softmax over $V$ classes under a Zipfian query distribution."
- **Empirically open.** Nobody has published a matched sweep of embedding-only rate $R \in \{16, 8, 4, 3, 2, 1\}$ bits/param $\times$ model scale $\{1\text{B}, 7\text{B}, 70\text{B}\}$ with frequency-stratified loss reported. The compute is ~a few thousand GPU-hours of evaluation on existing checkpoints — cheap. It is unrun.
- **Empirically open.** Whether $E$ and $U$ have different rate floors, and by how much, at fixed model.
- **Methodologically blocked.** "How much did compression cost the rare tokens?" has no accepted metric. Perplexity is frequency-weighted; MMLU-style benchmarks use a handful of answer tokens; MTEB (Muennighoff et al., EACL 2023) measures retrieval quality but not tail-token fidelity. Until $\delta_L^{(k)}$ or an equivalent is standard, papers cannot be compared.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement under a Zipfian target distribution.** The metric everyone reports ($\delta_L$, i.e. perplexity) integrates against the same power law that makes the tail expensive to store. A scheme can delete the tail's geometry entirely and pay almost nothing in the reported number, so the field's optimization signal is systematically blind to the failure mode that matters — rare entity names, code identifiers, non-Latin scripts, numbers. This is not "hard because it's important"; it is a metric that does not measure what its name implies.

Secondary obstructions: (a) **non-identifiability** — $E$ and the first block, and $U$ and the last block, are related by an invertible reparameterization, so "embedding capacity" is not separable from block capacity without a fixed convention; (b) **scale-dependence** — outlier features appear only above ~6.7B, so a floor measured at 1B does not extrapolate; (c) **confounded design** — changing $V$ changes tokens-per-sequence, so vocabulary compression and compute budget cannot be varied independently without recalibration.

## 7. Current Research (as of 2026)

- **Vocabulary scaling laws.** Tao et al. and follow-ups fitting $V_{\text{opt}}(N)$; open question whether the fitted exponent holds above 10B *(frontier — verify)*.
- **Tokenizer-free / byte-level models.** MEGABYTE (Yu et al., NeurIPS 2023), Byte Latent Transformer (Pagnoni et al., Meta, 2024), CANINE (Clark et al., TACL 2022) — the limiting case $V = 256$, which dissolves the embedding table and moves the cost into a patching/local model.
- **Nested and adaptive-width representations.** Matryoshka-style training now standard in commercial embedding APIs (OpenAI `text-embedding-3`, Nomic, Jina) with dimension truncation exposed to users.
- **Quantization-aware embedding handling.** AQLM/QuIP#-lineage codebook methods being extended to embedding tables *(frontier — verify)*; most released kernels still leave `tok_embeddings` at 8-bit or higher.
- **Multilingual tail fidelity.** Groups working on low-resource languages report tokenizer-induced degradation as a first-order effect; this is the community most likely to produce the missing stratified metric.

## 8. Concrete Next Experiment

**Question:** does the embedding rate floor $R^\star$ depend on model scale, and does perplexity hide it?

**Scale.** Three open checkpoints of one family — Llama-3.1 8B, 70B, and a 1B (Llama-3.2). Fixed tokenizer, $V = 128{,}256$, $d \in \{2048, 4096, 8192\}$. No retraining: post-training compression plus a short calibration pass on 4M tokens.

**Arms.** Compress *only* $E$ and *only* $U$, separately, at $R \in \{16, 8, 4, 3, 2\}$ bits/parameter with group-wise round-to-nearest (group size 64, scales counted in $R$). Cross with rank-$e$ factorization $e \in \{d, d/4, d/16\}$.

**Control arm.** Identical calibration and evaluation with $E, U$ untouched at bf16 — this absorbs the calibration-pass effect, which is otherwise credited to compression.

**Measurement.** Report $\delta_L$ *and* $\delta_L^{(10)}$: mean $\Delta$NLL restricted to target tokens in the rarest unigram decile of a 200M-token held-out mixed corpus (English web, code, and three non-Latin-script languages), at least 500K scored positions per decile.

**The deciding number.** The ratio

$$\rho = \frac{\delta_L^{(10)}}{\delta_L} \ \text{at } R = 4 \text{ bits}.$$

If $\rho \le 2$ at all three scales, perplexity is an adequate proxy and the field can keep reporting it. If $\rho \ge 10$ at 70B while $\delta_L < 0.01$ nats — the outcome the anisotropy and outlier-feature literature predicts — then every existing "4-bit embeddings are free" claim is unsupported, and the stratified metric becomes mandatory. Cost: order 2,000 A100-hours, no training.

## 9. Key References

- **[Foundational]** Ofir Press, Lior Wolf. *Using the Output Embedding to Improve Language Models.* EACL, 2017. — arXiv:1608.05859
- **[Foundational]** Hakan Inan, Khashayar Khosravi, Richard Socher. *Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling.* ICLR, 2017. — arXiv:1611.01462
- **[Foundational]** Édouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou. *Efficient softmax approximation for GPUs.* ICML, 2017. — arXiv:1609.04309
- **[Foundational]** Alexei Baevski, Michael Auli. *Adaptive Input Representations for Neural Language Modeling.* ICLR, 2019. — arXiv:1809.10853
- **[Foundational]** Raphael Shu, Hideki Nakayama. *Compressing Word Embeddings via Deep Compositional Code Learning.* ICLR, 2018. — arXiv:1711.01068
- **[Foundational]** Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* ICLR, 2020. — arXiv:1909.11942
- **[SOTA]** Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024. — arXiv:2407.13623
- **[SOTA]** Aditya Kusupati, Gantavya Bhatt, Aniket Rege, et al. *Matryoshka Representation Learning.* NeurIPS, 2022. — arXiv:2205.13147
- **[SOTA]** Mehdi Ali, Michael Fromm, Klaudia Thellmann, et al. *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL / NeurIPS D&B, 2024. — arXiv:2310.08754
- **[SOTA]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[SOTA]** Tim Dettmers, Luke Zettlemoyer. *The case for 4-bit precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[SOTA]** Ikuya Yamada, Akari Asai, Hannaneh Hajishirzi. *Efficient Passage Retrieval with Hashing for Open-domain Question Answering.* ACL, 2021. — arXiv:2106.00882
- **[Theory]** Kasper Green Larsen, Jelani Nelson. *Optimality of the Johnson-Lindenstrauss Lemma.* FOCS, 2017. — arXiv:1609.02094
- **[Theory]** Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-Yan Liu. *Representation Degeneration Problem in Training Natural Language Generation Models.* ICLR, 2019. — arXiv:1907.12009
- **[Survey]** Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023. — arXiv:2210.07316
- **[Related]** Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS, 2023. — arXiv:2305.07185

## 10. Worked Example

Take Llama-3.1-8B: $V = 128{,}256$, $d = 4096$, untied $E$ and $U$. Each table is $128{,}256 \times 4096 = 525.3$M parameters; both together are $1.05$B of the $8.03$B total — **13.1%** of parameters, and at bf16, **2.10 GB** of the 16.06 GB checkpoint.

Quantize both to 4-bit with group size 64: per group of 64 weights, 64 codes $\times$ 4 bits = 256 bits, plus one fp16 scale and one fp16 zero-point = 32 bits. Effective rate

$$R = \frac{256 + 32}{64} = 4.5 \ \text{bits/param},$$

so the tables drop to $1.05\text{B} \times 4.5 / 8 = 0.59$ GB. Saving: 1.51 GB, or 9.4% of the checkpoint.

Now the obstruction. Under a Zipfian unigram law with $\alpha = 1$ over $V = 128$K, the normalizer is $H_V \approx \ln(128256) + 0.577 \approx 12.34$. The rarest decile — ranks 115,431 to 128,256 — carries mass

$$\sum_{r=115431}^{128256} \frac{1}{r \, H_V} \approx \frac{\ln(128256/115430)}{12.34} = \frac{0.1055}{12.34} \approx 0.0086.$$

Under 1%. So if 4-bit quantization left the top nine deciles untouched and inflated NLL on the rarest decile by a full **1.0 nat** — catastrophic, roughly "this token is now $e$ times less likely" — the corpus-level number moves by

$$\Delta \text{PPL-equivalent} = 0.0086 \times 1.0 = 0.0086 \ \text{nats/token},$$

which on a base loss of ~2.0 nats is a perplexity change from 7.389 to 7.453: **+0.9%**. That lands inside the run-to-run and calibration-set noise band typically reported for PTQ, and would be written up as "lossless."

The saving is real and worth having (1.51 GB). The claim that it is free is untested, and the standard metric cannot test it — a 1-nat degradation on 8.6% of the vocabulary's *type* mass hides under a 0.9% perplexity move. That gap between what is measured and what is claimed is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*