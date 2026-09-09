---
id: 25-speech-and-audio/prosody-controllability-naturalness-tradeoff
title: "Prosody Controllability Versus Naturalness Trade-off"
topic: 25-speech-and-audio
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Prosody Controllability Versus Naturalness Trade-off

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/prosody-controllability-naturalness-tradeoff` · **Status:** open

## 1. Problem Statement

A text-to-speech system takes text $w$, a speaker identity $s$, and a **prosody control** $c$ — a pitch contour, a duration schedule, an emotion label, a natural-language instruction, or a reference utterance — and emits a waveform $\hat{x}$. Two things are wanted at once:

- **Controllability:** the realized prosody of $\hat{x}$ matches $c$.
- **Naturalness:** $\hat{x}$ is indistinguishable from a human recording of $w$ by $s$.

The folk claim is that these conflict: pushing the control harder degrades naturalness, and the most natural systems are the least steerable. The problem is to establish whether this conflict is real, and if so to characterize the frontier.

Three variants, with different difficulty:

- **Measurement variant.** Define a controllability axis and a naturalness axis that are simultaneously valid, so that a point on the plane means something. This is the binding constraint today.
- **Method variant.** Given fixed metrics, build a system that dominates the current frontier — same control error at higher naturalness, at fixed data and compute.
- **Theory variant.** Prove that a trade-off must exist for any generative model of a fixed family, or exhibit conditions under which it vanishes.

Solving it means: a reproducible frontier curve, with confidence intervals, over which system comparisons are decidable.

## 2. Formal Setting

Let $p^\*(x \mid w, s)$ be the distribution of human recordings. Let $\phi: \mathcal{X} \to \mathbb{R}^d$ be a **prosody read-out** — the function that extracts, from a waveform, the quantity the control names. Measured concretely:

- **Pitch:** $\phi_{f_0}(x) \in \mathbb{R}^T$, log-$F_0$ per 10 ms frame from a tracker (RAPT, pYIN, or CREPE), voiced frames only, $z$-normalized per speaker.
- **Duration:** $\phi_{\text{dur}}(x) \in \mathbb{R}^{|w|}$, per-phone durations from forced alignment (MFA), in frames.
- **Energy:** per-frame RMS in dB.
- **Categorical style:** $\phi_{\text{emo}}(x) \in \Delta^{K-1}$, the posterior of a *held-out* emotion classifier not used in training.

Control error for a target $c$ with read-out $\phi$:

$$\mathcal{E}(c) \;=\; \mathbb{E}_{\hat{x} \sim p_\theta(\cdot \mid w, s, c)} \big[\, \lVert \phi(\hat{x}) - c \rVert_2 \,\big] \quad\text{(RMSE in semitones, or ms, or } 1-\text{accuracy)}.$$

Naturalness is measured as CMOS against ground truth: listeners hear $(\hat{x}, x^\*)$ in random order and score on $\{-3,\dots,+3\}$. Write $\mathcal{N}(c) = \mathbb{E}[\text{CMOS}]$, so $\mathcal{N}=0$ means parity with human recordings. The object of interest is the **Pareto frontier**

$$\mathcal{F} = \{(\epsilon, \nu) : \nu = \max_\theta \mathcal{N}(\theta) \ \text{s.t.}\ \mathcal{E}(\theta) \le \epsilon \}.$$

A useful reframing is rate–distortion: a latent prosody code $z$ with capacity $I(z; x)$ nats, where the KL term in a VAE-TTS objective upper-bounds the rate. Battenberg et al. (ICASSP 2020) make this explicit — transfer fidelity rises with capacity, and so does leakage of speaker and channel information into $z$.

**Assumptions, and which are violated:**

1. *$c$ is realizable* — some human utterance of $w$ by $s$ has prosody $c$. **Violated by construction** in most control experiments: a $+6$ semitone shift on an angry utterance, or a reference from a different speaker, puts the target off the manifold. Then low $\mathcal{E}$ and high $\mathcal{N}$ are genuinely incompatible, and the "trade-off" is an artifact of the target set.
2. *$\phi$ is faithful* — the read-out measures the perceived attribute. Violated for emotion (classifier posteriors are not perception) and partially for $F_0$ (tracker octave errors on creaky and breathy voice).
3. *CMOS is stable across studies.* Violated: see §4.
4. *Prosody factorizes from timbre and content.* Violated — $F_0$ range co-varies with vocal effort, which changes spectral tilt.

## 3. State of the Art

**Empirical SOTA (naturalness, weak control).** StyleTTS 2 (Li et al., NeurIPS 2023) reports CMOS above ground truth on LJSpeech single-speaker; NaturalSpeech 3 (Ju et al., ICML 2024) reports human-parity CMOS on LibriSpeech test-clean using the FACodec factorization into content, prosody, timbre and acoustic detail. Voicebox (Le et al., NeurIPS 2023) reports 1.9% WER and 0.681 speaker similarity on filtered LibriSpeech zero-shot. These are benchmark numbers on the authors' own listening panels; cross-paper CMOS is not comparable (§4), so "human parity" is *claimed but not established* as a system-independent fact.

**Empirical SOTA (explicit control).** FastSpeech 2 (Ren et al., ICLR 2021) predicts pitch, energy and duration explicitly and allows direct override — the cleanest low-$\mathcal{E}$ arm, at a naturalness cost that was never measured *as a function of override magnitude*. Global Style Tokens (Wang et al., ICML 2018) and hierarchical VAE-TTS (Hsu et al., ICLR 2019) give reference-based control without a calibrated scale. PromptTTS (Guo et al., ICASSP 2023), InstructTTS (Yang et al., 2023) and Audiobox (Vyas et al., 2023) give natural-language control; their controllability numbers are attribute-classifier accuracies, which is assumption 2 above.

**Theory SOTA.** There is no theorem about this trade-off. The nearest formal results are rate–distortion–perception (Blau & Michaeli, ICML 2019): at fixed rate, forcing the output distribution to match the source (perfect perceptual quality) costs at most a factor of 2 in MSE distortion. This bounds a *related* trade-off — distortion versus realism — not controllability versus naturalness, and the mapping between the two is not established.

## 4. What Is Known

- **CMOS/MOS is not portable.** Le Maguer, King & Harte (Interspeech 2022) re-evaluated Blizzard Challenge 2013 systems alongside modern neural TTS; the same audio scored differently, and the relative ordering of older systems compressed. Wester, Valentini-Botinhao & Henter (Interspeech 2015) showed that with typical listener counts, MOS differences under about 0.3 are not reliably resolved. Chiang et al. (Interspeech 2023) showed cross-paper MOS comparison is invalid without shared anchors. Scale: dozens of systems, hundreds of listeners.
- **Capacity controls the trade-off.** Battenberg et al. (ICASSP 2020) showed that varying the VAE KL capacity in nats monotonically trades reference-transfer fidelity against speaker leakage and stability in Tacotron-class models on multi-speaker data.
- **Explicit predictors work at moderate magnitude.** FastSpeech 2 on LJSpeech (13 h, single speaker) reports MOS ≈ 3.83 against Tacotron 2 ≈ 3.70 and ground truth ≈ 4.30, with pitch/duration override qualitatively preserving quality at small deviations. No published curve of naturalness against override magnitude.
- **Automatic MOS predictors track humans in-domain only.** UTMOS (Saeki et al., Interspeech 2022) won VoiceMOS Challenge 2022 main track with high system-level correlation, but out-of-domain track performance dropped sharply — so the cheap axis is not a substitute for listeners under distribution shift, which is exactly what strong control induces.

## 5. What Is Not Known

- **Methodologically blocked (primary):** there is no agreed, realizability-controlled target distribution for $c$. Without it, $\mathcal{E}$ and $\mathcal{N}$ are measured against incompatible references and the frontier is undefined rather than merely unmeasured.
- **Empirically open:** nobody has published a naturalness-versus-control-error curve for a single system across a swept control magnitude with adequate listener power. The experiment is runnable today (§8).
- **Empirically open:** whether the frontier moves with scale. If it is a data/capacity artifact, a 100k-hour model should dominate a 1k-hour model at every $\epsilon$; if it is intrinsic, the curves should coincide near $\epsilon \to 0$.
- **Theoretically open:** no proof that any trade-off must exist for realizable targets. Plausibly $\mathcal{F}$ is flat on the realizable set and the observed conflict is entirely off-manifold extrapolation. Nobody has proved or refuted this.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by statistical cost**.

- Naturalness ratings are contaminated by the control target. A listener rating a $+6$-semitone utterance cannot separate "synthetic artifact" from "human would not say it that way". The CMOS reference $x^\*$ has the *original* prosody, so the rater is partly scoring prosody appropriateness, not signal quality. The evaluation does not measure what it names.
- Per-rating CMOS variance is large ($\sigma \approx 1$), so resolving frontier points 0.1 CMOS apart needs thousands of ratings per arm — and a frontier needs many arms (§10).
- **Non-identifiability:** at a fixed $c$, many prosodic realizations satisfy $\phi(\hat x) = c$ with very different naturalness. $\mathcal{E}$ does not pin down the output, so two systems at equal control error are not comparable.

## 7. Current Research (as of 2026)

- **Factorized codecs.** FACodec-style disentanglement (Microsoft Research Asia) and follow-ons treat prosody as a separate quantized stream, aiming to make control an edit on one factor rather than a perturbation of the whole. *(frontier — verify)* whether attribute-level editing beats direct override on a matched frontier.
- **Instruction-following TTS.** PromptTTS-lineage and Audiobox-lineage work (Microsoft, Meta) is moving from attribute labels to free-text descriptions, which makes $\phi$ harder to define — the controllability axis becomes an LLM-judge score with unknown calibration.
- **Flow-matching and diffusion backbones** (Voicebox, Matcha-TTS, E2/F5-style models) trade sampling steps for quality; the classifier-free guidance weight is itself an untracked controllability/naturalness knob. *(frontier — verify)*
- **Evaluation reform.** VoiceMOS Challenge organizers and the Blizzard/Interspeech evaluation community continue pushing shared anchors and reporting standards; adoption in TTS papers remains partial.

## 8. Concrete Next Experiment

**Sweep the control magnitude on one system and measure the curve.**

- **Scale.** One open TTS model with an explicit pitch predictor (FastSpeech 2 or StyleTTS 2), trained on LibriTTS-R (585 h). Test set: 200 utterances from 20 held-out speakers.
- **Design.** Two target families. (a) **Realizable:** $c$ taken from a *different real recording of the same sentence by the same speaker* (multi-take corpus or read-repeat elicitation). (b) **Synthetic:** $c$ = original contour scaled to $\pm\{0, 1, 2, 4, 6\}$ semitones of $F_0$ range. Five magnitude levels × 2 families = 10 arms.
- **Control arm.** Copy-synthesis of the *ground-truth* recording whose prosody is $c$ — a human utterance actually carrying the target. This upper-bounds naturalness at zero control error and separates "model degradation" from "target implausibility".
- **Ratings.** 1,600 CMOS ratings per arm (see §10), crowd panel, ground-truth anchors interleaved.
- **The deciding number.** The **CMOS slope on the realizable family**, $\partial \mathcal{N} / \partial \mathcal{E}$, in CMOS per semitone of residual $F_0$ RMSE. If $|{\partial \mathcal{N}}/{\partial \mathcal{E}}| < 0.05$ CMOS/semitone with a 95% CI excluding 0.15, the trade-off is an off-manifold artifact and the field should stop reporting it. If the slope is steep on realizable targets too, the trade-off is real and the frontier is the right object to optimize.

## 9. Key References

- **[Foundational]** Y. Wang, D. Stanton, Y. Zhang, et al. *Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis.* ICML, 2018. — arXiv:1803.09017
- **[Foundational]** R. Skerry-Ryan, E. Battenberg, Y. Xiao, et al. *Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron.* ICML, 2018. — arXiv:1803.09047
- **[Foundational]** W.-N. Hsu, Y. Zhang, R. Weiss, et al. *Hierarchical Generative Modeling for Controllable Speech Synthesis.* ICLR, 2019. — arXiv:1810.07217
- **[Foundational]** E. Battenberg, S. Mariooryad, D. Stanton, et al. *Effective Use of Variational Embedding Capacity in Expressive End-to-End Speech Synthesis.* ICASSP, 2020.
- **[Foundational]** Y. Blau, T. Michaeli. *Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff.* ICML, 2019.
- **[SOTA]** Y. Ren, C. Hu, X. Tan, et al. *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.* ICLR, 2021. — arXiv:2006.04558
- **[SOTA]** Y. A. Li, C. Han, V. S. Raghavan, G. Mischler, N. Mesgarani. *StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models.* NeurIPS, 2023.
- **[SOTA]** Z. Ju, Y. Wang, K. Shen, et al. *NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models.* ICML, 2024.
- **[SOTA]** M. Le, A. Vyas, B. Shi, et al. *Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale.* NeurIPS, 2023.
- **[Method]** Z. Guo, Y. Leng, Y. Wu, S. Zhao, X. Tan. *PromptTTS: Controllable Text-to-Speech with Text Descriptions.* ICASSP, 2023.
- **[Evaluation]** M. Wester, C. Valentini-Botinhao, G. E. Henter. *Are We Using Enough Listeners? No! An Empirically-Supported Critique of Interspeech 2014 TTS Evaluations.* Interspeech, 2015.
- **[Evaluation]** S. Le Maguer, S. King, N. Harte. *Back to the Future: Extending the Blizzard Challenge 2013.* Interspeech, 2022.
- **[Evaluation]** T. Saeki, D. Xin, W. Nakata, et al. *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022.* Interspeech, 2022.
- **[Survey]** X. Tan, T. Qin, F. Soong, T.-Y. Liu. *A Survey on Neural Speech Synthesis.* arXiv, 2021. — arXiv:2106.15561

## 10. Worked Example

Take FastSpeech 2 on LJSpeech. Override the pitch predictor to scale the $F_0$ contour range by $+4$ semitones. Measured outcome, typical of this setup:

- Realized $F_0$ RMSE against target: **1.2 semitones** (the model absorbs part of the override — control is not exact).
- CMOS against the original recording: suppose $-0.35$.

Now ask what that $-0.35$ means. The listener compares a $+4$-semitone utterance to a normally-spoken recording of the same sentence. Two causes are summed and not separable: vocoder/acoustic degradation from off-distribution conditioning, and the fact that a human would not say this sentence with that pitch range. Adding the copy-synthesis control arm — a real recording that *does* carry a $+4$-semitone range, resynthesized through the same vocoder — splits them. If that arm also scores $-0.30$, then $0.30$ of the $0.35$ is target implausibility, not model failure, and the model's own contribution is $-0.05$: no trade-off worth the name.

The statistical cost of resolving that split:

$$n = \left(\frac{1.96\,\sigma}{\text{half-width}}\right)^2 = \left(\frac{1.96 \times 1.0}{0.05}\right)^2 \approx 1{,}537 \ \text{ratings per arm}.$$

Ten arms plus a control arm each → about **17,000 CMOS ratings**. At roughly \$0.10 per rating and 30 ratings per listener-session, that is ~\$1,700 and ~570 sessions for *one system*. Comparing three systems triples it, and the anchors must be shared across all of them or the numbers do not combine.

That is the obstruction in one number: the measurement needed to decide whether the trade-off exists costs about $10^4$ human judgements per system, and every published "controllable TTS" paper spends about $10^3$ across *all* its conditions — enough to report a MOS table, not enough to resolve a frontier.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*