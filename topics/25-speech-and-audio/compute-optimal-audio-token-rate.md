---
id: 25-speech-and-audio/compute-optimal-audio-token-rate
title: "Compute-Optimal Audio Token Rate"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Audio Token Rate

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/compute-optimal-audio-token-rate` · **Status:** empirically-open

## 1. Problem Statement

An audio language model consumes discrete tokens produced by a tokenizer at some rate $R$ tokens per second of audio. Published systems span two orders of magnitude: 50 Hz HuBERT units (Lakhotia et al., TACL 2021), 75 Hz × 8 codebooks = 600 tokens/s for EnCodec (Défossez et al., TMLR 2023), 12.5 Hz × 8 = 100 tokens/s for Mimi (Moshi, 2024), 40 tokens/s for WavTokenizer (ICLR 2025), ~25 tokens/s for TAAE (Parker et al., ICLR 2025), ~5 Hz for SyllableLM (Baade et al., ICLR 2025). No published work fixes a training-compute budget and asks which $R$ minimizes downstream loss.

The problem: **given a fixed compute budget $C$ FLOPs and a fixed corpus of raw audio, what token rate $R$ minimizes end-task error, and how does the optimal $R^\ast$ scale with $C$?**

Three variants, of different difficulty:

- **Measurement.** Losses at different $R$ are not comparable — per-token cross-entropy on a 5 Hz vocabulary and a 600 tokens/s vocabulary measure different things. What is the compute-comparable quantity?
- **Method.** Build a tokenizer family whose only varying axis is $R$, holding reconstruction quality, codebook capacity and encoder compute fixed. No such family exists.
- **Theory.** Is there a rate–distortion argument predicting $R^\ast(C)$ from the entropy rate of speech, or is $R^\ast$ an artifact of architecture?

## 2. Formal Setting

Let a tokenizer $\mathcal{T}$ map waveform $x \in \mathbb{R}^{T f_s}$ ($f_s$ = sample rate, $T$ seconds) to tokens with **frame rate** $f$ Hz, **quantizer depth** $Q$ (RVQ codebooks per frame), **codebook size** $V$. Measured quantities:

$$R = f\,Q \ \ \text{tokens/s}, \qquad b = f\,Q\log_2 V \ \ \text{bits/s}, \qquad L = R\,T \ \ \text{tokens/utterance}.$$

Model has $N$ non-embedding parameters, trained on $D$ tokens. Training compute is measured, not estimated, but the standard proxy is $C \approx 6ND$; with attention over long audio contexts the quadratic term is not negligible, so measure $C$ as $6ND + 12 N_{\text{layer}} d_{\text{model}} L_{\text{ctx}} D$.

The corpus is $H$ hours of audio, so $D = 3600 H R$ and the epoch count is $E = D/(3600HR)$ — **$D$ and $R$ are not independent given a corpus.**

Objective. Let $\mathcal{E}$ be a downstream error measured *per second of audio*, not per token: e.g. sWUGGY/sBLIMP accuracy (Nguyen et al., ZeroSpeech 2021), spoken StoryCloze, or word error rate of a resynthesized continuation. Define

$$R^\ast(C) \;=\; \arg\min_{R}\ \min_{N,D \,:\, 6ND = C}\ \mathcal{E}\big(N, D, R\big).$$

For loss-based analysis the only rate-invariant likelihood is **bits per second of audio**:

$$\mathcal{L}_{\text{bps}} = R \cdot \mathbb{E}\big[-\log_2 p_\theta(t_i \mid t_{<i})\big],$$

which is comparable across $R$ only if the tokenizers are lossless w.r.t. the same target; they are not, so a distortion term must be carried alongside. Write the total as a rate–distortion pair $(\mathcal{L}_{\text{bps}}, \Delta)$ with $\Delta$ the tokenizer's reconstruction distortion (measured: ViSQOL, or WER of ASR on resynthesized audio).

Assumptions and their violation status:

| Assumption | Status |
|---|---|
| $C = 6ND$ | Violated at long context; audio at 50 Hz gives 10× text sequence lengths for the same seconds. |
| Loss comparable across tokenizers | **Violated by construction.** Different vocabularies, different support. |
| Tokenizer compute negligible | Violated for TAAE-class transformer codecs and for the $Q$-fold RVQ decoding stack. |
| Data unlimited (Chinchilla-style) | Violated: public speech is $\sim 10^5$–$10^6$ hours; low $R$ makes the run data-bound (§10). |
| Downstream metric is monotone in loss | Unestablished for speech; sBLIMP saturates while loss improves. |

## 3. State of the Art

**Established (ablated).**
- Cuervo & Marxer (EMNLP 2024) fit Kaplan-style power laws for speech LMs on HuBERT units and estimate speech LMs need roughly **3× more compute than text** to reach comparable linguistic performance, extrapolating that ~$10^5$–$10^6$ hours of speech is needed for text-level semantics. Single tokenizer, single rate — $R$ was not a variable.
- Hoffmann et al. (NeurIPS 2022) $N \propto C^{0.5}$, $D \propto C^{0.5}$ for text. Transfers to speech only by assumption.
- SpeechTokenizer (ICLR 2024) established, with ablation, that distilling HuBERT semantics into RVQ level 1 improves downstream LM quality at fixed bitrate. This is a *token-content* result, not a *token-rate* result.

**Claimed but unablated at LM scale.**
- Mimi's 12.5 Hz / 1.1 kbps design in Moshi is justified by real-time dialogue latency and by ablations on reconstruction, not by a compute-matched sweep of $f$ against downstream LM loss.
- WavTokenizer (ICLR 2025, 40–75 tokens/s single codebook) and TAAE (ICLR 2025, ~400 bps, ~25 tokens/s) report reconstruction and some generative-TTS numbers. **These are benchmark numbers**; neither paper holds LM training compute fixed while varying $R$.
- SyllableLM / Sylber (both ICLR 2025) report that ~5 Hz syllabic units give competitive semantic benchmarks with large training-compute reductions. Reported as speedup at matched *quality*, not as a point on a compute-optimal frontier.

**Theory SOTA.** None specific to audio token rate. The nearest formal anchor is the empirical estimate that speech transmits ~39 bits/s across languages (Coupé et al., *Science Advances*, 2019), which lower-bounds the useful $b$ for linguistic content but says nothing about paralinguistics or the optimal $R$ at which a transformer learns.

## 4. What Is Known

- 50 Hz HuBERT units, LM at ~150M params on ~6k h: sWUGGY ~0.72–0.80, sBLIMP ~0.55–0.57 (Lakhotia et al., TACL 2021). Baseline scale.
- Scaling exponents for HuBERT-unit speech LMs fit cleanly over ~$10^{18}$–$10^{20}$ FLOPs; the semantic gap to text is estimated at ~3× compute (Cuervo & Marxer, 2024). Measured at ≤1B params.
- Reconstruction quality is achievable at very low rate: TAAE reports high-quality 16 kHz speech at 400 bps (~25 tokens/s); WavTokenizer at 40 tokens/s single-codebook. So **low $R$ does not by itself destroy speech content** — at least for single-speaker clean speech.
- Mimi: 12.5 Hz, 8 codebooks, 1.1 kbps, streaming, used to train a 7B speech-text model in production (Moshi, 2024). Existence proof that 100 tokens/s suffices for full-duplex dialogue.
- Interleaved speech-text training changes the scaling picture: text-initialized speech LMs scale better in compute than speech-only ones (Maimon et al., 2025). This confounds any $R$ sweep that uses a text-pretrained init.
- Syllabic ~5 Hz units retain enough for lexical/semantic benchmarks (SyllableLM, ICLR 2025) but discard prosodic timing precision.

## 5. What Is Not Known

- **Empirically open (the main gap).** Nobody has run a compute-matched sweep over $R \in \{5, 12.5, 25, 50, 100, 300, 600\}$ tokens/s with a controlled tokenizer family, at $\ge 10^{20}$ FLOPs, reporting a downstream metric per second of audio. The experiment is runnable today for well under $10^{22}$ FLOPs. Whether $R^\ast$ grows, shrinks or is flat in $C$ is unmeasured.
- **Methodologically blocked.** There is no accepted rate-invariant loss. Comparing $\mathcal{L}_{\text{bps}}$ across tokenizers is contaminated by differing distortion $\Delta$; comparing downstream benchmarks is contaminated by benchmark saturation (sBLIMP near 0.55–0.60 for most systems). The measurement of "how much linguistic information a token carries" is not defined.
- **Theoretically open.** No theorem relates transformer sample complexity to the frame rate of a lossy discrete channel. It is not known whether the loss surface over $(N, D, R)$ is even unimodal in $R$, nor whether $R$ and $Q$ trade off along an equivalence class (is 12.5 Hz × 8 the same model as 50 Hz × 2?). Non-identifiability of $(f, Q, V)$ against $b$ is unresolved.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus a data–rate coupling**, not compute cost.

1. Changing $R$ changes the tokenizer, which changes distortion, vocabulary, and how much semantics sits in level 1 of the RVQ. Any observed difference in downstream error is attributable to at least four causes at once. No published tokenizer family varies $f$ alone.
2. Given a fixed corpus of $H$ hours, halving $R$ halves the tokens available. At Chinchilla-optimal $D$, low-$R$ arms become data-bound and must repeat epochs while high-$R$ arms do not (§10). The comparison then measures epoching, not rate.
3. The evaluation does not measure what it names. sBLIMP and sWUGGY are lexical/syntactic probes; they are blind to prosody, speaker, and non-speech audio — exactly the content that high $R$ buys. A sweep scored on sBLIMP will report that low $R$ wins, by construction.

## 7. Current Research (as of 2026)

- **Low-rate semantic tokenizers.** Berkeley/UT-Austin lines (Sylber, SyllableLM) push toward 4–5 Hz syllabic units. Open question they raise but do not answer: does 5 Hz survive scaling, or does it cap out?
- **Streaming split tokenizers.** Kyutai (Mimi/Moshi) and follow-ons optimize $f$ for latency; the LM-compute argument is secondary. *(frontier — verify whether any 2026 Kyutai release includes a rate ablation.)*
- **Scaling laws for speech and interleaved speech-text.** Cuervo & Marxer; Hebrew University (Maimon, Adi) on interleaved scaling. Adding $R$ as a third axis is the obvious next paper. *(frontier — verify)*
- **Codec benchmarking.** Codec-SUPERB and DASB (Discrete Audio and Speech Benchmark, Interspeech 2024) standardize downstream probes across tokenizers, which is the precondition for a rate sweep; neither yet fixes training compute.

## 8. Concrete Next Experiment

**Build one tokenizer family, vary only $f$.** Train five RVQ codecs on the same 60k h (Libri-Light), same encoder architecture and parameter count, same $Q=4$, same $V=2048$, differing only in decimation stride: $f \in \{6.25, 12.5, 25, 50, 100\}$ Hz, i.e. $R \in \{25, 50, 100, 200, 400\}$ tokens/s. Report $\Delta$ (ASR-WER on resynthesis) for each — arms whose $\Delta$ differs by more than 1 WER point are excluded rather than adjusted.

**Scale.** For each surviving $R$, train a Chinchilla-shaped ladder at $C \in \{3\times10^{19}, 3\times10^{20}, 3\times10^{21}\}$ FLOPs, $N \in [50\text{M}, 1.5\text{B}]$, $D$ set by $C=6ND$. ~20 runs; ≈$10^{22}$ FLOPs total, days on a 64-GPU node.

**Control arm.** $R = 50$ tokens/s HuBERT units at matched $C$ — the configuration Cuervo & Marxer fit, so the ladder's exponents can be checked against a published curve. Second control: identical arms with data capped to the *lowest*-rate arm's token count, isolating the epoching confound.

**Deciding number.** $R^\ast$ at each $C$, read from the compute-frontier, and specifically the sign and magnitude of $\gamma$ in $R^\ast \propto C^{\gamma}$. Scored on **spoken StoryCloze accuracy plus a prosody probe** (F0-contour continuation error), reported jointly. $|\gamma| < 0.05$ across two decades of compute means rate is compute-invariant and should be chosen purely for latency; $\gamma < -0.1$ means low-rate tokenizers become *more* favourable as models grow, and the field's 50 Hz default is wrong.

## 9. Key References

- **[Foundational]** Hoffmann, J. et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Kaplan, J. et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Lakhotia, K. et al. *On Generative Spoken Language Modeling from Raw Audio.* TACL 9, 2021.
- **[Foundational]** Hsu, W.-N. et al. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.* IEEE/ACM TASLP, 2021. — arXiv:2106.07447
- **[SOTA]** Cuervo, S. and Marxer, R. *Scaling Properties of Speech Language Models.* EMNLP, 2024. — arXiv:2404.00685
- **[SOTA]** Défossez, A. et al. *Moshi: a speech-text foundation model for real-time dialogue.* 2024. — arXiv:2410.00037
- **[SOTA]** Parker, J. D. et al. *Scaling Transformers for Low-Bitrate High-Quality Speech Coding.* ICLR, 2025.
- **[SOTA]** Ji, S. et al. *WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling.* ICLR, 2025.
- **[SOTA]** Baade, A. et al. *SyllableLM: Learning Coarse Semantic Units for Speech Language Models.* ICLR, 2025.
- **[SOTA]** Zhang, X. et al. *SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models.* ICLR, 2024. — arXiv:2308.16692
- **[SOTA]** Kumar, R. et al. *High-Fidelity Audio Compression with Improved RVQGAN.* NeurIPS, 2023. — arXiv:2306.06546
- **[SOTA]** Défossez, A. et al. *High Fidelity Neural Audio Compression.* TMLR, 2023. — arXiv:2210.13438
- **[SOTA]** Zeghidour, N. et al. *SoundStream: An End-to-End Neural Audio Codec.* IEEE/ACM TASLP, 2021. — arXiv:2107.03312
- **[SOTA]** Borsos, Z. et al. *AudioLM: a Language Modeling Approach to Audio Generation.* IEEE/ACM TASLP, 2023. — arXiv:2209.03143
- **[Context]** Coupé, C., Oh, Y., Dediu, D., Pellegrino, F. *Different languages, similar encoding efficiency: comparable information rates across the human communicative niche.* Science Advances 5(9), 2019.
- **[Survey]** Mousavi, P. et al. *DASB — Discrete Audio and Speech Benchmark.* Interspeech, 2024.
- **[Survey]** Cui, W. et al. *Recent Advances in Speech Language Models: A Survey.* 2024.

## 10. Worked Example

Budget $C = 10^{21}$ FLOPs. Assume Chinchilla shape $D = 20N$, so $C = 120N^2$:

$$N = \sqrt{10^{21}/120} = 2.9\times 10^{9}, \qquad D = 5.8\times10^{10}\ \text{tokens}.$$

Convert $D$ to hours of audio, $H = D/(3600R)$:

| $R$ (tok/s) | System | Hours needed | Epochs over Libri-Light (60k h) | Epochs over 1M h |
|---|---|---|---|---|
| 600 | EnCodec 75 Hz × 8 | 27k | 0.45 | 0.03 |
| 200 | HuBERT 50 Hz × 4 | 80k | 1.3 | 0.08 |
| 100 | Mimi 12.5 Hz × 8 | 160k | 2.7 | 0.16 |
| 25 | TAAE ~400 bps | 640k | 10.7 | 0.64 |
| 5 | SyllableLM syllables | 3.2M | 53 | 3.2 |

The obstruction is visible in the last two columns. At $10^{21}$ FLOPs and a 60k-hour corpus, the 5 Hz arm must see the data 53 times while the 600 tok/s arm sees it half once. If the 5 Hz arm loses, the paper cannot say whether the rate was wrong or the repetition was; if it wins, the result may be that repetition is cheap at low rate. Only a corpus above ~3M hours decouples the two at this budget — larger than any public speech corpus, and comparable to the largest private ones.

A second, subtler failure: the 5 Hz arm's tokens carry ≈ $5 \times 4 \times 11 = 220$ bits/s versus speech's ~39 bits/s of linguistic content (Coupé et al., 2019). Both the 5 Hz and 600 tok/s arms are far above the linguistic bound, so a lexical benchmark like sWUGGY cannot separate them — it is saturated with respect to the variable under test. The extra 580 bits/s at high rate buys speaker identity and prosody, which the benchmark does not score. Any sweep that reports only sWUGGY/sBLIMP will conclude "low rate is free," and that conclusion will be an artifact of the metric, not a measurement of $R^\ast$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*