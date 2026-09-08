---
id: 25-speech-and-audio/diffusion-versus-autoregressive-speech-synthesis
title: "Diffusion Versus Autoregressive Speech Synthesis Sample Efficiency"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Diffusion Versus Autoregressive Speech Synthesis Sample Efficiency

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/diffusion-versus-autoregressive-speech-synthesis` · **Status:** empirically-open

## 1. Problem Statement

Two families now dominate zero-shot text-to-speech: **autoregressive (AR) token models** that factorize discrete acoustic tokens left-to-right (VALL-E, TorToiSe, CosyVoice, Seed-TTS-AR), and **non-autoregressive (NAR) diffusion / flow-matching models** that denoise a continuous mel or latent sequence conditioned on text and a prompt (Grad-TTS, NaturalSpeech 2, Voicebox, E2 TTS, F5-TTS).

The question: **for a fixed data budget $D$ hours of paired speech and a fixed training-compute budget $C$ FLOPs, which family reaches a given speech-quality target first, and does the ordering flip as $D$ and $C$ grow?**

Three variants, of very different difficulty:

- **Measurement variant.** Is there a quality metric on which "sample efficiency" is even comparable across the two families? AR models are scored by token perplexity, diffusion models by an ELBO or a flow-matching regression loss; the two losses are not on a common scale, so the comparison must be made in *output* space (WER, speaker similarity, MOS), where the estimator has its own noise floor.
- **Method variant.** Given an engineering budget, build matched-compute AR and NAR systems that differ only in the generative factorization, and measure the data-efficiency curve of each.
- **Theory variant.** Prove a separation: exhibit a class of conditional speech distributions for which one factorization has strictly better sample complexity at fixed model capacity.

Solved means: a published curve of quality versus $D$ (and versus $C$) for both families, with matched tokenizers, matched data, matched inference compute, and a stated crossover point or a proof that none exists in the measured range.

## 2. Formal Setting

Let a training corpus be $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ with $x_i$ a text/phoneme sequence and $y_i \in \mathbb{R}^{T_i \times d}$ a waveform-derived feature sequence. **$D$ is measured as total audio duration in hours** after voice-activity trimming, not as utterance count; **$N$ is the utterance count**; both must be reported because $D/N$ (mean utterance length, typically 5–15 s in LibriHeavy-style corpora) shifts the effective context distribution.

**AR arm.** A codec quantizer $q: \mathbb{R}^{d} \to \{1,\dots,K\}^{Q}$ maps a frame to $Q$ codebook indices. The model maximizes
$$\mathcal{L}_{\mathrm{AR}}(\theta) = \frac{1}{\sum_i T_i}\sum_i \sum_{t=1}^{T_i} \log p_\theta\big(z_{i,t} \mid z_{i,<t}, x_i, \text{prompt}\big),$$
measured in nats per acoustic frame — **not** per token, since $Q$ is a free design choice and per-token numbers are incomparable across codecs.

**NAR arm.** With a probability path $p_t$ from noise $p_0 = \mathcal{N}(0,I)$ to data $p_1$, a conditional flow-matching model minimizes
$$\mathcal{L}_{\mathrm{FM}}(\phi) = \mathbb{E}_{t \sim U[0,1],\, y \sim p_1,\, y_t \sim p_t(\cdot|y)} \big\| v_\phi(y_t, t, x, \text{mask}) - u_t(y_t|y) \big\|_2^2,$$
a squared error in feature units. $\mathcal{L}_{\mathrm{AR}}$ and $\mathcal{L}_{\mathrm{FM}}$ **cannot be compared numerically**; only their induced sample distributions can.

**Compute accounting.** Training compute $C_{\mathrm{train}} \approx 6 N_{\mathrm{params}} \cdot N_{\mathrm{frames}}$ for both arms (Kaplan et al. 2020). Inference compute differs structurally: AR costs $T$ sequential forward passes; NAR costs $S$ solver steps over the full sequence, $S \in [8, 64]$ typically, plus classifier-free guidance doubling the per-step cost. **Report $C_{\mathrm{inf}}$ per second of audio generated**, since a fair "efficiency" claim must fix it.

**Quality.** Intelligibility $\mathrm{WER}$ from a fixed ASR model (state the checkpoint — Whisper-large-v3 and HuBERT-large-ASR give different absolute numbers on the same audio). Speaker similarity $\mathrm{SIM} = \cos(e(\hat y), e(y_{\mathrm{prompt}}))$ with $e$ a fixed WavLM-TDNN verification embedding. Naturalness by MOS with $n \ge 20$ raters per item and a reported 95% CI.

**Assumptions, and which are violated.**
1. *Both arms see identical data.* Violated in every published comparison: Voicebox used 60k h English, VALL-E 60k h LibriLight, F5-TTS ~100k h multilingual Emilia.
2. *The codec is neutral.* Violated — the residual VQ codec caps AR ceiling quality; NAR models on continuous latents have no such cap.
3. *MOS is a stable scale.* Violated — MOS is not comparable across listener panels or studies (Chiang et al., Interspeech 2023).
4. *WER measures naturalness.* Violated — a monotone, over-smoothed voice can score better WER than a natural one.

## 3. State of the Art

**Empirical SOTA (established).** Both families reach objective parity with ground truth on LibriSpeech-style zero-shot continuation. Reported benchmark numbers, *not* independently ablated against each other: VALL-E (Wang et al. 2023) ~5.9% WER / SIM ~0.58 on its own test split; Voicebox (Le et al., NeurIPS 2023) 1.9% WER / SIM 0.681 cross-sentence on filtered LibriSpeech test-clean; F5-TTS (Chen et al. 2024) ~2.4% WER / SIM ~0.66 on LibriSpeech-PC test-clean; ground-truth human audio sits near 2.2% WER under the same ASR. **These numbers come from different papers with different data, different test filtering, and different ASR scorers — the ranking they suggest is not evidence.**

**Claimed but unablated.** (a) "Diffusion/flow needs less data than AR because it does not have to model a discrete token ordering." No matched-data study supports this. (b) "AR generalizes better to long-form and expressive prosody because it models global sequential structure." Supported anecdotally by Seed-TTS and TorToiSe-lineage systems, not by a controlled curve. (c) "Non-autoregressive models are more robust (no repetition/skipping failure mode)." This is *structurally* true — the duration/alignment is externally imposed — and is the one asymmetry that is not just a benchmark artifact.

**Theory SOTA.** No separation theorem for speech. The nearest general results are diffusion sample-complexity bounds under score-estimation error (Chen et al., ICLR 2023 and successors) and autoregressive-model likelihood bounds; neither is instantiated for a speech-shaped conditional distribution, and neither gives a comparison.

## 4. What Is Known

- **Scale helps both.** TorToiSe (Betker 2023) established that AR TTS quality tracks parameters and data. Cuervo & Marxer (EMNLP 2024) fit power laws for speech *language* models and estimate that speech LMs need roughly **three orders of magnitude more data than text LMs** for comparable linguistic competence — measured on models up to ~800M params and up to ~100k hours.
- **Data thresholds are large.** Zero-shot voice cloning at the quality now considered baseline appears in the literature only above ~$10^4$ hours: NaturalSpeech 2 at 44k h, VALL-E and Voicebox at 60k h, F5-TTS at ~100k h. Below ~1k h both families degrade, but **no paper measures both degradation curves on the same corpus**.
- **Small-data regime.** Grad-TTS (ICML 2021) reached near-ground-truth MOS on single-speaker LJSpeech (~24 h). Single-speaker AR systems reached comparable MOS on the same corpus years earlier. At $D \approx 24$ h, single speaker, the two families are indistinguishable within MOS CIs.
- **Inference-compute asymmetry is real.** Flow-matching TTS produces audio in $S \approx 16$–32 network evaluations regardless of length; AR needs $O(T)$ steps (~75 frames/s of audio × $Q$ or 1 with a chunked decoder).
- **Hybrids beat neither cleanly.** CosyVoice and Seed-TTS-style stacks (AR semantic tokens → NAR acoustic generation) are competitive, which is itself evidence that the factorizations carry complementary inductive bias.

## 5. What Is Not Known

- **Empirically open (primary).** The matched-data, matched-compute curve. Every ingredient exists — public 100k-hour corpora (Emilia, Libri-Light/LibriHeavy), open AR and flow-matching implementations, fixed ASR and speaker scorers. Nobody has run both arms over $D \in \{10^2, 10^3, 10^4, 10^5\}$ hours with the same tokenizer front-end and published the two curves. Estimated cost: single-digit thousands of GPU-hours per arm at 300M params.
- **Methodologically blocked (secondary).** Naturalness. MOS is not comparable across panels, and automatic proxies (UTMOS, Saeki et al. 2022; Fréchet Audio Distance, Kilgour et al. 2019) are known to be gameable and to correlate weakly with human judgment out of their training domain. So "which family sounds better at 1k hours" has no trusted estimator, even though "which is more intelligible" does.
- **Theoretically open.** Whether a sample-complexity separation exists at all for conditional distributions with speech's structure (long-range prosodic dependence, one-to-many text→acoustics mapping, near-deterministic phonetic content).

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by a non-neutral tokenizer**. The AR arm must quantize; the NAR arm need not. Any observed gap therefore decomposes into (i) factorization effect, (ii) codec information loss, (iii) duration-model effect (NAR needs an external or learned alignment; AR gets it free), and (iv) inference-compute difference. Published comparisons vary all four at once. Isolating (i) requires either an AR model over continuous features (possible — MELLE, Meng et al., ACL 2025) or an NAR model over the same discrete codes, and both such controls break other parts of each recipe.

Second obstruction: **the deciding metric has no ground truth**. If the two families converge on WER and SIM but differ on prosody naturalness — the plausible outcome — the comparison rests on MOS, which is the quantity known not to transfer across studies.

## 7. Current Research (as of 2026)

- Continuous-valued AR TTS removing the codec confound (MELLE lineage, Microsoft) — the cleanest available control arm.
- Flow-matching TTS with fewer solver steps and distillation, narrowing the inference-compute asymmetry (F5-TTS, E2 TTS lines; SWivid and Microsoft groups).
- Hybrid AR-semantic + NAR-acoustic stacks as the production default (CosyVoice/Alibaba, Seed-TTS/ByteDance).
- Speech-LM scaling laws extended to *synthesis* quality rather than linguistic probing *(frontier — verify)*.
- Open large corpora (Emilia, ~100k h multilingual) making the matched-data experiment affordable for academic labs.

## 8. Concrete Next Experiment

**Scale.** Four data budgets $D \in \{100, 1\,000, 10\,000, 100\,000\}$ hours, sampled as nested subsets of one corpus (Emilia-EN or LibriHeavy) so smaller budgets are strict subsets. Two model sizes: 150M and 600M params. Train to matched $C_{\mathrm{train}}$ per cell (equal FLOPs, not equal epochs). Eight cells per arm, ~16 runs; ≈3–6k A100-hours total.

**Arms.**
- *AR:* decoder-only transformer over continuous mel frames with a Gaussian/flow output head (MELLE-style), so **no codec is in the loop**.
- *NAR:* conditional flow-matching over the identical mel features, same transformer backbone, same text front-end, same vocoder.
- *Control arm:* the same NAR backbone trained with an oracle duration/alignment taken from forced alignment, isolating the alignment advantage from the factorization advantage.

**Fixed:** vocoder, phonemizer, mel spec, speaker-prompt protocol (3 s prompt), ASR scorer (Whisper-large-v3), speaker encoder (WavLM-TDNN), and inference compute normalized to equal FLOPs per second of audio (tune NAR solver steps $S$ to match AR's cost).

**Deciding number.** $\Delta_D = \mathrm{SIM}_{\mathrm{NAR}}(D) - \mathrm{SIM}_{\mathrm{AR}}(D)$ at each $D$, with WER constrained to within 0.5 points absolute between arms. If $\Delta_D$ changes sign between $10^3$ and $10^5$ hours with non-overlapping bootstrap CIs (1000 resamples over test utterances), the crossover is real and the field's folk claim is confirmed. If $|\Delta_D| < 0.02$ at every $D$, the factorization is not the axis that matters and future effort should go to data and tokenization.

## 9. Key References

- **[Foundational]** Popov, Vovk, Gogoryan, Sadekova, Kudinov. *Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech.* ICML, 2021. — arXiv:2105.06337
- **[Foundational]** Kong, Ping, Huang, Zhao, Catanzaro. *DiffWave: A Versatile Diffusion Model for Audio Synthesis.* ICLR, 2021. — arXiv:2009.09761
- **[Foundational]** Wang, Chen, Zhou, Wang, Liu, Chen, Wu, Liu, Ko, Zhao, Huang, Li, Wei. *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E).* 2023. — arXiv:2301.02111
- **[SOTA]** Le, Vyas, Shi, Karrer, Sari, Moritz, Williamson, Manohar, Adi, Mahadeokar, Hsu. *Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale.* NeurIPS, 2023. — arXiv:2306.15687
- **[SOTA]** Shen, Ju, Tan, Liu, Leng, He, Qin, Zhao, Bian. *NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers.* ICLR, 2024. — arXiv:2304.09116
- **[SOTA]** Chen, Niu, Chen, Wang, Wang, Liu. *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.* 2024. — arXiv:2410.06885
- **[SOTA]** Anastassiou et al. *Seed-TTS: A Family of High-Quality Versatile Speech Generation Models.* 2024. — arXiv:2406.02430
- **[Control arm]** Meng, Chen, Wang, Wang, Liu, Qian, Wei. *Autoregressive Speech Synthesis without Vector Quantization (MELLE).* ACL, 2025.
- **[Method]** Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR, 2023. — arXiv:2210.02747
- **[Scaling]** Cuervo, Marxer. *Scaling Properties of Speech Language Models.* EMNLP, 2024. — arXiv:2404.00685
- **[Scaling]** Betker. *Better Speech Synthesis Through Scaling (TorToiSe).* 2023. — arXiv:2305.07243
- **[Evaluation]** Chiang, Huang, Lee. *Why We Should Report the Details in Subjective Evaluation of TTS More Rigorously.* Interspeech, 2023.
- **[Evaluation]** Saeki, Xin, Nakata, Koriyama, Takamichi, Saruwatari. *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022.* Interspeech, 2022. — arXiv:2204.02152
- **[Evaluation]** Kilgour, Zuluaga, Roblek, Sharifi. *Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms.* Interspeech, 2019.
- **[Survey]** Tan, Qin, Soong, Liu. *A Survey on Neural Speech Synthesis.* 2021. — arXiv:2106.15561

## 10. Worked Example

Take the two most-cited numbers as they are usually quoted side by side: Voicebox (NAR flow matching) at 1.9% WER / 0.681 SIM, VALL-E (AR codec LM) at ~5.9% WER / ~0.58 SIM. Read naively: NAR wins by 4 WER points, so NAR is the more sample-efficient family.

Now decompose the 4 points.

| Confound | Voicebox | VALL-E | Effect direction |
|---|---|---|---|
| Training data | 60k h, curated English | 60k h LibriLight (audiobook, noisier) | favors NAR |
| Output representation | continuous mel latent | 8-codebook EnCodec RVQ | favors NAR (codec ceiling) |
| Alignment | forced-aligned durations supplied | learned implicitly | favors NAR (no skip/repeat failures) |
| Test filtering | 4–10 s utterances, filtered test-clean | authors' own split | unknown |
| ASR scorer | HuBERT-L | HuBERT-L / Whisper varies by report | ±0.5–1.5 WER |

Three of five rows favor NAR *for reasons unrelated to the factorization*, and the fifth alone moves WER by up to 1.5 points. A conservative attribution: codec loss ≈1–2 points, alignment robustness ≈1–2 points (VALL-E's failures are heavy-tailed — a small fraction of repeated/dropped utterances dominates mean WER), scorer/split ≈1 point. That accounts for the whole gap before the generative factorization gets any credit at all.

The measurable consequence: **the reported 4-point gap has an error budget larger than itself.** Whatever "diffusion is more sample-efficient" means, this comparison cannot establish it, and no published comparison closes the table above. That is exactly the hole the Section 8 experiment fills — and it is why the status here is *empirically open* rather than *partially solved*.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*