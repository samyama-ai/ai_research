---
id: 25-speech-and-audio/audio-token-vocabulary-scaling-law
title: "Discrete Audio Token Vocabulary Scaling Law"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Discrete Audio Token Vocabulary Scaling Law

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/audio-token-vocabulary-scaling-law` · **Status:** empirically-open

## 1. Problem Statement

Speech and audio language models operate on discrete tokens produced by a neural codec or a clustered self-supervised representation. The tokenizer has at least three free design axes: codebook size $V$ (entries per quantizer), quantizer depth $Q$ (residual levels per frame), and frame rate $f$ (tokens per second per quantizer). Text LM scaling laws take the vocabulary as given; here it is a design variable that changes the sequence length, the entropy per token, and the acoustic information retained.

**The question.** Given a compute budget $C$ and a corpus of $H$ hours of audio, what $(V, Q, f)$ minimizes end-task loss, and how does the optimum move as $C$ grows?

Three variants, of different difficulty:

- **Measurement variant.** Define a loss that is comparable across tokenizers with different $V$, $Q$, $f$. Per-token cross-entropy is not: it falls trivially as $V$ shrinks or $f$ rises. This variant is currently the binding constraint.
- **Empirical variant.** Fit $V^\star(C)$, $Q^\star(C)$, $f^\star(C)$ from an IsoFLOP sweep, as Hoffmann et al. (2022) did for $(N, D)$ and Tao et al. (2024) did for text vocabulary.
- **Theory variant.** Derive the exponent in $V^\star \propto C^{\alpha}$ from a rate–distortion model of the tokenizer plus a capacity model of the LM. No such derivation exists for a multi-codebook residual quantizer.

**Solved** means: a published law, fit at $\geq 3$ orders of magnitude of compute, that predicts the held-out optimum $(V,Q,f)$ for an unseen budget within the noise of the sweep, and whose predicted optimum beats the standard configuration (e.g. 8×1024 @ 50 Hz) on a downstream metric that is not the training loss.

## 2. Formal Setting

**Tokenizer.** An encoder maps waveform $x$ at sample rate $s$ to frames at rate $f$ Hz. Each frame is quantized by $Q$ residual codebooks of size $V$, giving token tuple $z_t \in \{1..V\}^Q$. Bitrate, as measured:

$$R = Q \cdot f \cdot \log_2 V \quad \text{bits/s}.$$

EnCodec at $f=75$, $Q=8$, $V=1024$ gives $R = 6$ kbps. Mimi at $f=12.5$, $Q=8$, $V=2048$ gives $R=1.1$ kbps.

**Reconstruction distortion** $D_{\text{rec}}$: measured, not assumed — ViSQOL or PESQ on a held-out set, plus a phonetic probe (frame-level phone accuracy from a linear head on the token embeddings).

**Language model.** Non-embedding parameters $N_{\text{nv}}$; embedding + output head parameters $N_v = 2 \, d \, Q V$ for $d$-dimensional embeddings with per-codebook heads. Total $N = N_{\text{nv}} + N_v$. Compute, as measured: $C = 6 N \, T$ FLOPs where $T$ is the number of *token positions* consumed, $T = Q \, f \, H \cdot 3600$ for flat interleaving, or $f H \cdot 3600$ for a depth-transformer factorization where the $Q$ codebooks at a frame share a position.

**Normalized loss.** The only cross-tokenizer-comparable quantity is loss per second of audio:

$$\mathcal{L}_{\text{sec}} = \frac{1}{H\cdot 3600}\sum_{t} \sum_{q=1}^{Q} -\log_2 p_\theta(z_{t,q} \mid z_{<t}, z_{t,<q}) \quad \text{bits/s}.$$

Per-token cross-entropy $\mathcal{L}_{\text{tok}} = \mathcal{L}_{\text{sec}} / (Qf)$ is *not* comparable and must never be used to rank tokenizers.

**The objective** is a constrained minimization over the design space, with the tokenizer's own distortion as a floor:

$$\min_{V,Q,f,N,H} \; \mathcal{M}\big(\text{LM}(N,H;V,Q,f)\big) \quad \text{s.t.}\quad 6NT \leq C,$$

where $\mathcal{M}$ is a downstream metric — sWUGGY/sBLIMP lexical and syntactic accuracy (Nguyen et al., 2020), or WER of a resynthesized continuation.

**Assumptions, and which are violated.**
1. *$\mathcal{L}_{\text{sec}}$ tracks $\mathcal{M}$ monotonically.* **Violated.** A tokenizer that discards speaker identity lowers $\mathcal{L}_{\text{sec}}$ while destroying the information a voice-cloning task needs.
2. *Codebook entries are used uniformly.* **Violated.** RVQ codebooks suffer dead entries; effective vocabulary $V_{\text{eff}} = 2^{\mathbb{H}(z_q)}$ is often well below $V$, and deeper quantizers are closer to uniform than shallow ones.
3. *$C = 6NT$ holds.* Approximately violated at long context — attention is $O(T^2)$ and low-frame-rate tokenizers change the quadratic term disproportionately.
4. *The tokenizer is fixed while the LM scales.* Violated by design in any joint sweep; the codec must itself be retrained per $V$, and its own capacity is a confound.

## 3. State of the Art

**Established.**
- Chinchilla-style $(N, D)$ laws transfer to speech tokens with *different constants*: Cuervo & Marxer, "Scaling Properties of Speech Language Models" (EMNLP 2024), fit scaling laws on HuBERT units and estimate speech LMs need roughly three orders of magnitude more compute than text LMs to reach comparable linguistic competence — an extrapolation, flagged as such by the authors.
- Vocabulary is a scaling variable in text: Tao et al., "Scaling Laws with Vocabulary" (2024), fit $V^\star \propto N_{\text{nv}}^{\gamma}$ with $\gamma \approx 0.83$ across models from 33M to 3B, and show Llama-2-70B's 32K vocabulary is far below their predicted optimum (~216K). This is the closest analogue and it does **not** cover multi-codebook or acoustic tokens.
- Lower frame rate at fixed bitrate helps speech LMs: Mimi (Défossez et al., Moshi, 2024) runs at 12.5 Hz with $Q=8$, $V=2048$, versus EnCodec's 75 Hz — established as a systems result (real-time full-duplex dialogue), not as an ablated scaling claim.

**Claimed but unablated.**
- That single-codebook tokenizers with large $V$ (WavTokenizer, Ji et al., 2024: 4096 entries at 40–75 tokens/s) are strictly better for LM modeling. The reconstruction numbers are reported; the LM-side ablation at matched compute against an RVQ stack is not.
- That semantic distillation into the first quantizer (SpeechTokenizer, Zhang et al., ICLR 2024) improves downstream LM quality. Reported as benchmark deltas; the confound with codec capacity is not removed.

**Benchmark-number-only.** Most tokenizer comparisons — Codec-SUPERB (Wu et al., 2024) and DASB, the Discrete Audio and Speech Benchmark (Mousavi et al., 2024) — report a table at one model scale. They rank tokenizers; they do not give a scaling exponent, and DASB's own finding is that the ranking is task-dependent.

## 4. What Is Known

- **Unit count matters sublinearly at small scale.** GSLM (Lakhotia et al., TACL 2021) swept HuBERT k-means at $V \in \{50, 100, 200\}$; 100–200 units beat 50 on both ABX and resynthesis at a fixed ~150M-parameter LM trained on ~6K hours of LibriLight. The sweep stops at 200.
- **Text initialization dominates at 13B.** TWIST (Hassid et al., NeurIPS 2023) trained speech LMs up to 13B on ~150K hours; warm-starting from a text LM improved sBLIMP/sStoryCloze consistently. Vocabulary was held fixed at 500 HuBERT units — so the largest speech-LM scale run to date carries **no** vocabulary sweep.
- **Bitrate is not the sufficient statistic.** Mimi (1.1 kbps, 12.5 Hz) supports competitive LM behaviour where EnCodec at 6 kbps and 75 Hz does not, at comparable LM sizes. Same order of magnitude of bits/s, very different sequence lengths.
- **Codebook utilization is a real loss channel.** DAC (Kumar et al., NeurIPS 2023) shows factorized, L2-normalized codebook lookup raises utilization substantially over EnCodec's, at 9 codebooks × 1024 entries, 86 Hz, 44.1 kHz — i.e. nominal $V$ overstates $V_{\text{eff}}$ by a model-dependent factor.
- **Small-budget speech LMs are now cheap enough to sweep.** Slamming (Maimon et al., 2025) trains a competitive speech LM on a single A5000 in 24 hours, making a 20–40 point IsoFLOP grid affordable.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted loss that is comparable across $(V, Q, f)$ *and* validated to correlate with downstream metrics. $\mathcal{L}_{\text{sec}}$ is comparable but rewards information destruction; downstream metrics are comparable but noisy and task-dependent (DASB's central finding). Until this is fixed, any fitted exponent is a fit to an unvalidated target.
- **Empirically open.** No published IsoFLOP sweep varies $V$ over more than one octave for audio tokens at more than one model scale. The experiment is runnable today for under ~2,000 GPU-hours; nobody has published it.
- **Theoretically open.** No derivation of $V^\star(C)$ for residual quantization. The scalar-quantizer rate–distortion intuition does not extend to RVQ, where the marginal entropy of quantizer $q$ depends on the reconstruction error left by $q-1$, so the $Q$ codebooks are not exchangeable and $R = Qf\log_2 V$ overstates the true rate by the inter-codebook mutual information.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement compounded by non-identifiability.** Changing $V$ changes four things at once: (i) sequence length is unaffected but head parameters $N_v = 2dQV$ grow linearly, shifting the $N_{\text{nv}}/N$ split at fixed $C$; (ii) the codec must be retrained, so codec capacity and codebook utilization co-vary with $V$; (iii) the entropy per token changes, moving $\mathcal{L}_{\text{tok}}$ mechanically; (iv) the reconstruction floor $D_{\text{rec}}$ changes, so the LM is modeling a different signal. Three of the four move the loss in the same direction, so a single-arm sweep cannot attribute the effect. This is non-identifiability, not expense: adding compute does not separate the terms.

Second obstruction: **the evaluation does not measure what it names.** sWUGGY and sBLIMP were designed for lexical/syntactic probing of semantic units; they are near-blind to prosody and speaker identity, which is exactly the information large-$V$ acoustic codebooks buy. A sweep scored on sBLIMP will report that smaller vocabularies are better, and be right about the metric and wrong about the design question.

## 7. Current Research (as of 2026)

- **Low-frame-rate codecs for LMs.** Kyutai (Mimi/Moshi), and single-quantizer lines (WavTokenizer). The direction is trading $f$ down against $V$ up at roughly fixed bits/s. Whether this is a scaling law or a systems convenience is unresolved.
- **Speech-text scaling laws.** Maimon, Adi and colleagues (Hebrew University / SLAM lab) — "Scaling Analysis of Interleaved Speech-Text Language Models" (2025) — find interleaved training changes the compute exponent relative to speech-only. Vocabulary is held fixed.
- **Tokenizer benchmarking as a service.** DASB and Codec-SUPERB maintainers are extending to more downstream tasks; the scaling axis is not yet part of either. *(frontier — verify)*
- **Rate–distortion analysis of RVQ for generative modeling.** Sparse activity; no established result. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Fix the codec architecture (DAC-style, factorized lookup) and train 9 codecs on 1K hours of LibriLight: $V \in \{256, 1024, 4096\}$ crossed with $Q \in \{2, 4, 8\}$, all at $f = 25$ Hz. Report $V_{\text{eff}} = 2^{\mathbb{H}(z_q)}$ per codebook and ViSQOL for each. Then train speech LMs at three IsoFLOP budgets — $3\times10^{18}$, $3\times10^{19}$, $3\times10^{20}$ FLOPs — on each tokenizer, choosing $N$ Chinchilla-optimally within each budget. That is 27 LM runs; the largest is roughly a 1B model on 15K hours, about 1,500–2,500 A100-hours in total.

**Control arm.** A matched-bitrate, matched-$V_{\text{eff}}$ pair: $(V{=}1024, Q{=}8)$ versus $(V{=}4096, Q{=}5)$ at $f{=}25$ Hz — 2.0 kbps versus 1.5 kbps nominal, and re-tuned to equal measured $V_{\text{eff}} \cdot Q$. This arm isolates vocabulary shape from total bitrate; without it the sweep only re-discovers that more bits help.

**Deciding number.** Fit $\log V^\star = \alpha \log C + \beta$ to the argmin of downstream sWUGGY-in-context error at each budget. The single number is $\alpha$, with its bootstrap confidence interval. $\alpha$ indistinguishable from $0$ (CI containing zero across all three budgets) means vocabulary is *not* a scaling variable for audio and the field should fix $V$ and stop sweeping it. $\alpha > 0.3$ with a CI excluding zero means current 1024-entry codebooks are systematically undersized at frontier compute — the audio analogue of Tao et al.'s Llama-2 finding. Secondary readout: does the fitted $\alpha$ change sign when the target metric is swapped from sWUGGY to speaker-similarity of resynthesized continuations? A sign flip is direct evidence that the measurement, not the law, is the open problem.

## 9. Key References

- **[Foundational]** Lakhotia, Kharitonov, Hsu, Adi, Polyak, Bolte, Nguyen, Copet, Baevski, Mohamed, Dupoux. *On Generative Spoken Language Modeling from Raw Audio.* TACL, 2021.
- **[Foundational]** Zeghidour, Luebs, Omran, Skoglund, Tagliasacchi. *SoundStream: An End-to-End Neural Audio Codec.* IEEE/ACM TASLP, 2021.
- **[Foundational]** Borsos, Marinier, Vincent, Kharitonov, Pietquin, Sharifi, Roblek, Teboul, Grangier, Tagliasacchi, Zeghidour. *AudioLM: A Language Modeling Approach to Audio Generation.* IEEE/ACM TASLP, 2023.
- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022.
- **[SOTA]** Tao, Liu, Hou, Sun, King, Lyu. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* 2024. — closest existing law, text only.
- **[SOTA]** Cuervo, Marxer. *Scaling Properties of Speech Language Models.* EMNLP, 2024.
- **[SOTA]** Défossez, Mazaré, Orsini, Royer, Pérez, Jégou, Grave, Zeghidour. *Moshi: A Speech-Text Foundation Model for Real-Time Dialogue.* 2024. — the Mimi codec, 12.5 Hz.
- **[SOTA]** Kumar, Seetharaman, Luebs, Kumar, Kumar. *High-Fidelity Audio Compression with Improved RVQGAN.* NeurIPS, 2023. — codebook utilization.
- **[SOTA]** Hassid, Remez, Nguyen, Gat, Conneau, Kreuk, Copet, Défossez, Synnaeve, Dupoux, Schwartz, Adi. *Textually Pretrained Speech Language Models.* NeurIPS, 2023.
- **[SOTA]** Zhang, Zhou, Zhang, Wang, Wang, Qiu. *SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models.* ICLR, 2024.
- **[Survey]** Mousavi, Della Libera, Duret, Ploujnikov, Subakan, Ravanelli. *DASB — Discrete Audio and Speech Benchmark.* 2024.
- **[Survey]** Wu, Chung, Chen, et al. *Codec-SUPERB: An In-Depth Analysis of Sound Codec Models.* 2024.

## 10. Worked Example

Take a fixed budget $C = 3\times10^{19}$ FLOPs and 1,000 hours of speech at $f = 25$ Hz. Frame count is $25 \times 3.6\times10^6 = 9\times10^7$ frames.

**Arm A:** $Q=8$, $V=1024$. Positions (flat interleave) $T = 7.2\times10^8$. Nominal rate $R = 8 \times 25 \times 10 = 2{,}000$ bits/s.
**Arm B:** $Q=4$, $V=4096$. $T = 3.6\times10^8$. Nominal $R = 4\times25\times12 = 1{,}200$ bits/s.

From $C = 6NT$: Arm A affords $N = 3\times10^{19}/(6\times7.2\times10^8) \approx 6.9$M parameters; Arm B affords $\approx 13.9$M. Arm B gets twice the model for a nominally lower bitrate — the halved sequence length, not the vocabulary, dominates.

Now the embedding cost. At $d = 512$, $N_v = 2dQV$: Arm A is $2\times512\times8\times1024 \approx 8.4$M; Arm B is $2\times512\times4\times4096 \approx 16.8$M. **Both exceed their entire parameter budget.** At this compute scale the vocabulary tables alone eat the model, so $N_{\text{nv}} \to 0$ and the comparison is meaningless — unless you introduce weight tying or factorized heads, which is itself an untracked design choice that changes the answer.

Finally the utilization correction. Measure $\mathbb{H}(z_q)$ per codebook. A typical DAC-style stack at $V=1024$ gives near-full use on $q=1$ and progressively lower entropy on deep quantizers; suppose measured $V_{\text{eff}}$ across Arm A's 8 books averages $\sim 600$, and Arm B's 4 books average $\sim 3{,}000$. True rates become $8\times25\times\log_2 600 \approx 1{,}845$ and $4\times25\times\log_2 3000 \approx 1{,}155$ bits/s — the gap narrows but does not close, and the ordering of the arms on $\mathcal{L}_{\text{sec}}$ now depends on a quantity that was never held fixed.

**What this makes visible:** the head-parameter blow-up, the sequence-length effect, and the utilization gap all move together with $V$, and each one alone is large enough to flip the ranking. That is the obstruction — not compute.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*