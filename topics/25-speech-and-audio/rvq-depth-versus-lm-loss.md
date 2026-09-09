---
id: 25-speech-and-audio/rvq-depth-versus-lm-loss
title: "Residual Vector Quantizer Depth Versus Language Model Loss"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Residual Vector Quantizer Depth Versus Language Model Loss

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/rvq-depth-versus-lm-loss` · **Status:** empirically-open

## 1. Problem Statement

Neural audio codecs quantize a latent frame with a **residual vector quantizer** (RVQ): $K$ successive codebooks, each correcting the previous stage's error. Every downstream audio language model — VALL-E, AudioLM, MusicGen, Moshi — then models those $K$ token streams autoregressively.

$K$ sets a trade. Larger $K$ lowers reconstruction distortion, so the *ceiling* on output quality rises. Larger $K$ also multiplies tokens per second, and the added tokens are progressively closer to white noise, so the *achievable* fraction of that ceiling falls under a fixed compute budget.

The problem: **given a training-compute budget $C$ and a target generation quality, what is the optimal $K$ — and does an interior optimum exist at all, or is quality monotone in $K$ once modeling strategy is held fixed?**

Three variants, of very different difficulty:

- **Measurement.** Language-model cross-entropy is not comparable across $K$: the token alphabet, the sequence length, and the support of the distribution all change. There is currently no agreed scalar that makes "LM loss at $K=4$" and "LM loss at $K=16$" the same number. This variant is *methodologically blocked*.
- **Method.** Find a tokenizer/LM pair that decouples the two pressures — e.g. put all predictable structure in RVQ level 1 and treat levels $2..K$ as a cheap conditional decoder. Partially solved (SpeechTokenizer, Mimi).
- **Theory.** Prove that the per-level conditional entropy of an RVQ stack grows toward $\log_2 V$, and derive from that a compute-optimal $K^\star(C)$. Open.

## 2. Formal Setting

Audio $x \in \mathbb{R}^{T}$ at sample rate $f_s$. Encoder $E$ gives latents $z_{1:N} \in \mathbb{R}^{N \times d}$ at frame rate $f = N/(T/f_s)$ Hz. RVQ with $K$ codebooks $\mathcal{C}_1,\dots,\mathcal{C}_K \subset \mathbb{R}^{d}$, each of size $V$:

$$r_n^{(0)} = z_n, \qquad c_n^{(k)} = \arg\min_{c \in \mathcal{C}_k}\|r_n^{(k-1)} - c\|_2, \qquad r_n^{(k)} = r_n^{(k-1)} - c_n^{(k)}$$

Token $y_n^{(k)} \in \{1..V\}$ is the index of $c_n^{(k)}$. Reconstruction uses $\hat z_n = \sum_{k=1}^{K} c_n^{(k)}$.

**Measured quantities.**

- *Token rate*: $R(K) = K f$ tokens/s. *Bitrate*: $b(K) = K f \log_2 V$ bits/s (measured as file size of the packed indices, not as codebook entropy).
- *Distortion ceiling*: $D(K) = \mathbb{E}\|z - \hat z^{(K)}\|_2^2$; the perceptual proxy is ViSQOL or Mel-distance of $\mathrm{Dec}(\hat z^{(K)})$ against $x$ — a copy-synthesis measurement with no LM in the loop.
- *Per-level conditional entropy*, the quantity that actually governs LM difficulty:
$$h_k = H\!\left(y_n^{(k)} \,\middle|\, y_{<n}^{(1:K)},\, y_n^{(<k)}\right) \in [0, \log_2 V]$$
measured as the trained LM's mean NLL on held-out audio, restricted to positions of level $k$, in bits.
- *Total LM loss*: $\mathcal{L}(K) = \sum_{k=1}^{K} h_k$ bits/frame. **$\mathcal{L}$ is non-decreasing in $K$ by construction** — adding a codebook adds a term. Comparing raw $\mathcal{L}$ across $K$ is meaningless.
- *The comparable objective*: end-quality at matched compute,
$$\Phi(K; C) = \mathbb{E}\big[\, \mathrm{Qual}(\hat x) \,\big], \quad \hat x \sim \text{LM}_{K}\ \text{trained with FLOPs } C$$
with $\mathrm{Qual}$ an *end-to-end* metric on generated (not reconstructed) audio: WER of an ASR system on TTS output, speaker-similarity cosine, or FAD for music. The open question is the shape of $K \mapsto \Phi(K;C)$.

**Assumptions, and which are violated.**

1. *Codebooks are fully used*, so $\log_2 V$ is the true alphabet size. **Violated**: deep RVQ levels in EnCodec-style codecs show large dead-code fractions; DAC was designed specifically to fix this.
2. *Levels are conditionally independent given level 1*, the assumption behind NAR parallel decoding in VALL-E. **Violated**: parallel-pattern ablations in MusicGen degrade sharply relative to delay/flattening, which is direct evidence of residual cross-level dependence.
3. *Distortion ceiling transfers to generation quality.* **Violated in the observed direction**: codecs with better copy-synthesis do not reliably yield better LM samples.
4. *Compute per token is constant in $K$.* Holds for flattening; false for delay patterns and depth-transformer designs, which is precisely why $\Phi$ and not $\mathcal{L}$ must be the objective.

## 3. State of the Art

**Established.**

- RVQ with quantizer dropout for bitrate scalability — SoundStream (Zeghidour, Luebs, Omran, Skoglund, Tagliasacchi; *IEEE/ACM TASLP* 2022).
- High-fidelity RVQ codec at 24 kHz, 75 Hz frames, $V=1024$, $K \in \{2,4,8,16,32\}$ giving 1.5–24 kbps — EnCodec (Défossez, Copet, Synnaeve, Adi; *TMLR* 2023).
- Codebook under-utilization is a real failure mode and is fixed by low-dimensional factorized projection plus $\ell_2$-normalized lookup — DAC (Kumar, Seetharaman, Luebs, Kumar, Kumar; *NeurIPS* 2023), 9 codebooks at 44.1 kHz / ~8 kbps.
- Hierarchical split into semantic + coarse acoustic + fine acoustic levels is necessary for long-horizon coherence — AudioLM (Borsos et al.; *IEEE/ACM TASLP* 2023).
- Codebook-interleaving patterns matter and flattening is not required: the delay pattern reaches near-flattening quality at $\approx 1/K$ the decoding steps — MusicGen (Copet et al.; *NeurIPS* 2023), $K=4$ at 50 Hz.
- Distilling a self-supervised semantic teacher into RVQ level 1 improves downstream TTS — SpeechTokenizer (Zhang et al.; *ICLR* 2024); the same trick at $K=8$, $f=12.5$ Hz, ~1.1 kbps in Mimi/Moshi (Défossez et al., 2024).

**Claimed but unablated.** That a given production choice of $K$ (8 for VALL-E, 4 for MusicGen, 8 for Mimi) is *optimal*. These values were selected under fixed quality targets and engineering constraints; no published sweep isolates $K$ with tokenizer family, frame rate, LM size, and training FLOPs all held constant.

**Benchmark number only.** Single-codebook codecs — WavTokenizer (Ji et al.; *ICLR* 2025), and related BigCodec/Single-Codec lines — report competitive reconstruction and TTS scores at $K=1$. The comparison is against codecs of different frame rate and different decoder capacity, so it does not establish that $K=1$ dominates at matched compute.

## 4. What Is Known

- **Distortion falls with diminishing returns.** EnCodec at 24 kHz: going 1.5 → 3 → 6 → 12 → 24 kbps ($K = 2,4,8,16,32$) yields monotone but steadily shrinking MUSHRA gains; the top two doublings buy far less than the first.
- **Deep levels carry little structure.** In practice, RVQ level 1 dominates: after semantic distillation, level 1 alone supports intelligible content, and levels $2..K$ act as timbre/detail refinement. Measured at $K=8$, $f=12.5$ Hz in Mimi.
- **Sequence length is the binding cost.** Flattening $K=4$ at 50 Hz gives 200 tokens/s and quadratic attention cost; the delay pattern keeps 50 steps/s. MusicGen's ablation reports delay ≈ flatten in quality at a quarter the steps, and parallel (full conditional independence) clearly worse.
- **Speech LMs are data/compute-hungry relative to text.** Cuervo and Marxer, *Scaling Properties of Speech Language Models* (EMNLP 2024), estimate roughly three orders of magnitude more compute than text LMs for comparable linguistic ability — measured on semantic-unit LMs, not full RVQ stacks, which makes the acoustic-token overhead an *additional* unmeasured cost.
- **Text pretraining transfers.** TWIST (Hassid et al.; *NeurIPS* 2023) shows initializing a speech LM from a text LM improves loss and downstream metrics at up to 13B parameters.
- **Quantizer dropout has a cost.** DAC reports that SoundStream-style dropout across the whole batch degrades full-bandwidth reconstruction; applying it to a batch subset recovers it.

## 5. What Is Not Known

- **Methodologically blocked.** No accepted loss that is comparable across $K$. Candidates — bits/second, bits/second normalized by achievable distortion, rate–distortion–perception curves — have not been standardized, and none is used consistently across the codec-LM literature.
- **Empirically open.** The $K$-sweep at matched training FLOPs, matched tokenizer family, matched frame rate and matched end-metric has not been published at any scale above toy. It is runnable today for well under 10 GPU-months.
- **Theoretically open.** No proof of the rate at which $h_k \to \log_2 V$, hence no derivation of $K^\star(C)$. Also no theory relating RVQ depth to the *perception* term of rate–distortion–perception theory (Blau & Michaeli, ICML 2019), which is where "sounds real" lives.
- **Unknown whether the optimum is interior.** If $\Phi(K;C)$ is monotone increasing in $K$ once the interleaving pattern is chosen well, the whole problem dissolves into a sequence-length engineering question. Nobody has shown it either way.

## 6. Why It Is Hard

**Confounded measurement, plus a metric that does not measure what it names.**

The natural instrument — validation cross-entropy — is structurally incomparable across $K$: $\mathcal{L}(K)$ rises with $K$ by construction, and normalizing per token rewards the *most* noise-like tokenizers, since a level whose conditional entropy is exactly $\log_2 V$ contributes a constant that dilutes the mean. Both raw and normalized loss are therefore uninformative about $\Phi$.

Every alternative confounds $K$ with something else. Changing $K$ changes bitrate, sequence length, decoder difficulty, attention cost per second of audio, and the number of forward passes at inference — simultaneously. A clean sweep must hold FLOPs constant, which forces model size or data to move as $K$ moves, reintroducing scaling-law confounds. And the end-metrics themselves are weak: ASR-WER on TTS output saturates and is sensitive to the ASR system; FAD depends on the embedding network and is known to be unstable across implementations.

## 7. Current Research (as of 2026)

- **Low-frame-rate, semantic-first codecs.** Kyutai (Mimi, 12.5 Hz, $K=8$) and the SpeechTokenizer line push predictable structure into level 1 so that $K$ mostly buys fidelity, not modeling burden.
- **Single-codebook tokenizers.** WavTokenizer and successors argue $K=1$ with a large codebook and a strong decoder is sufficient — effectively the claim that $K^\star = 1$. *(frontier — verify)* whether these hold up under matched-compute LM training rather than reconstruction benchmarks.
- **Depth transformers.** Modeling the $K$ levels with a small secondary transformer per frame (the Moshi/RQ-Transformer family) decouples $K$ from the main model's sequence length; this changes the shape of $\Phi(K;C)$ and is the most likely route to an answer.
- **Interleaved speech–text scaling analyses.** Maimon and colleagues have extended speech-LM scaling work to interleaved text-speech training *(frontier — verify the exact scaling coefficients)*.
- **Vision analogue.** MAGVIT-v2 (Yu et al., *ICLR* 2024) shows tokenizer design, not model size, gated generative-LM quality in video — evidence the same question is tokenizer-bound in audio.

## 8. Concrete Next Experiment

**Scale.** One codec family (DAC architecture), one frame rate (50 Hz), one codebook size ($V = 1024$), retrained at $K \in \{1, 2, 4, 8, 16\}$ on 10k h of speech. For each $K$, train a decoder-only LM with the delay pattern at a **fixed 3×10²⁰ training FLOPs**, choosing model size per Chinchilla-style compute-optimal allocation (roughly 200M–400M params, ~10–20B tokens). Total ≈ 5 codecs + 5 LMs; under 10 A100-months.

**Control arm.** For each $K$, a **copy-synthesis oracle**: ground-truth tokens through the same decoder. This gives the ceiling $\Phi_{\max}(K)$ and separates "the codec cannot represent it" from "the LM cannot predict it."

**The deciding number.** The *realization ratio*

$$\rho(K) = \frac{\Phi(K; C)}{\Phi_{\max}(K)}$$

with $\Phi$ = 1 − ASR-WER on zero-shot TTS continuations (Whisper-large-v3, fixed). Report $\Phi(K;C)$ directly too.

- If $\Phi(K;C)$ peaks at some $K \in \{2,4,8\}$ and falls by more than 2 WER points at $K=16$, the interior optimum is real and $K^\star(C)$ is a legitimate scaling variable.
- If $\Phi$ is flat within noise (±0.5 WER) from $K=2$ to $K=16$ while $\rho$ falls monotonically, the LM is not the bottleneck and the field should optimize $K$ purely for inference cost.

Secondary readout: measure $h_k$ per level. If $h_k > 0.95\log_2 V$ for $k \ge 4$, the deep levels are provably unmodelable and belong in a non-autoregressive decoder, not in the LM.

## 9. Key References

- **[Foundational]** Zeghidour, Luebs, Omran, Skoglund, Tagliasacchi. *SoundStream: An End-to-End Neural Audio Codec.* IEEE/ACM TASLP, 2022. — arXiv:2107.03312
- **[Foundational]** Défossez, Copet, Synnaeve, Adi. *High Fidelity Neural Audio Compression.* TMLR, 2023. — arXiv:2210.13438
- **[Foundational]** Borsos, Marinier, Vincent, Kharitonov, Pietquin, Sharifi, Roblek, Teboul, Grangier, Tagliasacchi, Zeghidour. *AudioLM: A Language Modeling Approach to Audio Generation.* IEEE/ACM TASLP, 2023. — arXiv:2209.03143
- **[SOTA]** Kumar, Seetharaman, Luebs, Kumar, Kumar. *High-Fidelity Audio Compression with Improved RVQGAN.* NeurIPS, 2023. — arXiv:2306.06546
- **[SOTA]** Copet, Kreuk, Gat, Remez, Kant, Synnaeve, Adi, Défossez. *Simple and Controllable Music Generation.* NeurIPS, 2023. — arXiv:2306.05284
- **[SOTA]** Zhang, Zhou, Zhang, Yu, Zhou, Yu, Zhang, Qiu. *SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models.* ICLR, 2024. — arXiv:2308.16692
- **[SOTA]** Défossez, Mazaré, Orsini, Royer, Pérez, Jégou, Grave, Zeghidour. *Moshi: A Speech-Text Foundation Model for Real-Time Dialogue.* Technical report, Kyutai, 2024. — arXiv:2410.00037
- **[Related]** Wang, Chen, Zhou, Wu, Liu, Chen, Liu, Wang, Li, He, Zhao, Wei. *Neural Codec Language Models Are Zero-Shot Text to Speech Synthesizers (VALL-E).* 2023. — arXiv:2301.02111
- **[Related]** Cuervo, Marxer. *Scaling Properties of Speech Language Models.* EMNLP, 2024.
- **[Related]** Hassid, Remez, Nguyen, Gat, Conneau, Kreuk, Copet, Défossez, Synnaeve, Dupoux, Schwartz, Adi. *Textually Pretrained Speech Language Models (TWIST).* NeurIPS, 2023.
- **[Theory]** Blau, Michaeli. *Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff.* ICML, 2019. — arXiv:1901.07821
- **[Survey]** Ji, Jiang, Wang, Cheng, Yuan, et al. *WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling.* ICLR, 2025. — arXiv:2408.16532

## 10. Worked Example

Take an EnCodec-style stack: $f = 50$ Hz, $V = 1024$, so each level contributes $50 \times 10 = 500$ bits/s.

Suppose a trained LM reports per-level NLL in bits:

| level $k$ | 1 | 2 | 3 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| $h_k$ (bits) | 4.1 | 6.8 | 8.4 | 9.2 | 9.8 | 9.95 |
| $h_k/\log_2 V$ | 0.41 | 0.68 | 0.84 | 0.92 | 0.98 | 1.00 |

Total loss at $K=4$: $\mathcal{L}(4) = 4.1+6.8+8.4+9.2 = 28.5$ bits/frame = 1425 bits/s.
At $K=16$, adding levels 5–16 contributes roughly $12 \times 9.85 \approx 118$ bits/frame, giving $\mathcal{L}(16) \approx 146$ bits/frame = 7310 bits/s.

Now the trap. Per-token normalized loss:

$$\bar{\mathcal{L}}(4) = 28.5/4 = 7.13 \text{ bits/token}, \qquad \bar{\mathcal{L}}(16) = 146/16 = 9.14 \text{ bits/token}$$

$K=4$ looks better. Raw loss: $28.5 < 146$, $K=4$ looks better again. But both numbers are forced. Level 16 has $h_{16}/\log_2 V = 1.00$ — the LM has learned it is uniform noise and predicts uniform. That is *correct behaviour*, costs the model nothing to learn, and yet inflates every loss statistic. Meanwhile the $K=16$ codec's copy-synthesis ceiling is genuinely higher, so its samples may be better.

The obstruction is now visible: **the LM loss moves in the opposite direction to the thing we care about, for a reason that has nothing to do with model quality.** The only quantity that separates them is $\rho(K)$ in §8 — and computing $\rho$ requires the matched-FLOPs sweep that nobody has run. A cheap 3-line proxy check, computable today: if $h_k \ge 0.95\log_2 V$, level $k$ contributes no learnable structure and should be excised from the autoregressive loss entirely. On the table above that is every level from 8 up — half the tokens VALL-E-class models spend their sequence budget on.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*