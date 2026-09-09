---
id: 25-speech-and-audio/long-form-audio-context-extension
title: "Long-Form Audio Context Extension Without Degradation"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Form Audio Context Extension Without Degradation

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/long-form-audio-context-extension` · **Status:** empirically-open

## 1. Problem Statement

An audio-language model is trained on inputs of at most $T_0$ seconds (typically 30 s for Whisper-style encoders, 30–300 s for current audio LLMs). Extend it to inputs of $T \gg T_0$ — an hour of meeting audio, a three-hour podcast, a full film soundtrack — such that **per-unit-time task quality does not degrade with $T$**.

- **Input:** waveform $x \in \mathbb{R}^{16000 \cdot T}$ plus a text instruction $q$.
- **Output:** text $y$ (transcript, answer, summary, or timestamped event list).
- **Decision predicate:** for a task metric $M$, is $M(T) \ge M(T_0) - \varepsilon$ for all $T$ up to the target, at inference cost sub-quadratic in $T$?

Three variants, routinely conflated:

- **Measurement.** Does a benchmark exist whose difficulty is *held fixed* while $T$ varies? Most long-audio benchmarks confound context length with task difficulty. This variant is methodologically blocked.
- **Method.** Given a fixed measurement, does any architecture (position interpolation, chunk-and-stitch, hierarchical pooling, SSM, retrieval) hold $M$ flat? Empirically open.
- **Theory.** Is there a lower bound on the state size a streaming model needs to answer arbitrary queries over $T$ seconds of audio? Partially known from streaming-complexity arguments; not instantiated for audio.

Chunk-and-stitch (Whisper's 30 s sliding window) *solves* transcription-shaped tasks and *cannot* solve tasks with cross-chunk dependency — speaker identity at minute 3 versus minute 58, "how many times did the alarm sound", global summarization. The problem is only interesting for the second class.

## 2. Formal Setting

**Tokenization.** An audio encoder $E$ maps the waveform to $N = \lceil r T \rceil$ tokens at frame rate $r$ Hz. Measured values: Whisper `large-v3` produces $r = 50$ Hz before pooling; Qwen2-Audio pools to $r = 25$ Hz; Audio Flamingo 2's audio representation runs near $r \approx 8$–$12$ Hz after windowed pooling; Moshi's Mimi codec runs at $r = 12.5$ Hz. So one hour costs $N \in [45{,}000,\ 180{,}000]$ tokens depending on $r$ — this is the single number that makes the problem hard, and it is a design choice, not a constant.

**Degradation curve.** Fix a task family $\mathcal{T}$ and a metric $M \in [0,1]$ (higher better). Define

$$D(T) \;=\; M(T_0) - \mathbb{E}_{(x,q,y)\sim \mathcal{D}_T}\big[M(f_\theta(E(x), q),\, y)\big].$$

The problem is solved for horizon $T^\star$ if $D(T) \le \varepsilon$ for all $T \le T^\star$, with $\varepsilon = 0.02$ absolute a reasonable convention.

**The confound to control.** $\mathcal{D}_T$ must be *difficulty-matched*: the query $q$ must depend on a span of the same intrinsic length regardless of $T$. Formally, let $S(q) \subseteq [0,T]$ be the minimal evidence support. A valid family holds $|S(q)|$ and the number of distractor spans fixed while varying $T$. Almost no published long-audio benchmark does this; they instead scale $T$ and let the questions get harder, which makes $D(T)$ uninterpretable.

**Cost.** Prefill FLOPs for dense attention scale as $\Theta(L d N^2)$ with $L$ layers, width $d$. KV cache bytes $= 2 L n_{kv} d_h N b$ for $b$ bytes/element. Both are measured, not estimated: report wall-clock prefill seconds and peak GPU memory at each $T$.

**Assumptions, and which are violated.**

1. *Frame rate is task-sufficient.* Violated: phoneme discrimination needs $r \gtrsim 25$ Hz; environmental-event counting survives $r \approx 5$ Hz. A single $r$ cannot be optimal for both.
2. *Position encoding extrapolates.* Violated. RoPE-based decoders degrade past the trained context without interpolation (Chen et al. 2023), and audio encoders trained on fixed 30 s windows have learned absolute positional embeddings that do not extend at all.
3. *Audio tokens and text tokens are exchangeable to the LLM.* Violated: audio token streams have far lower per-token entropy and much higher local redundancy, so text-derived context-extension recipes (YaRN, PI) are being transferred off-distribution.
4. *Metric additivity* — that hour-level WER is the duration-weighted mean of segment WERs. Violated whenever the model repeats or skips a segment, which is exactly the long-form failure mode.

## 3. State of the Art

**Established (ablated, reproduced):**

- **Chunked long-form ASR.** Whisper (Radford et al., ICML 2023) transcribes arbitrary durations by 30 s sliding windows with a text-conditioned buffer. WhisperX (Bain et al., Interspeech 2023) replaces this with VAD-based segmentation plus forced alignment and reports both large speedups and reduced timestamp error. Established: for transcription, chunking works and long-context modelling is not required.
- **Chunking's failure mode is real.** Whisper's window-boundary looping and hallucinated spans on silence are documented and reproduced (Koenecke et al., FAccT 2024) — hallucinated content in roughly 1% of transcribed segments in their sample, concentrated on non-speech and disfluent audio.
- **Positional interpolation transfers within text.** PI (Chen et al. 2023) and YaRN (Peng et al., ICLR 2024) extend LLaMA-class models 8–32× with small perplexity cost, with ablations.

**Claimed but not ablated for audio:**

- **Native long-audio audio-LLMs.** Audio Flamingo 2 (Ghosh et al., ICML 2025) trains to 5-minute inputs and introduces LongAudio / LongAudioBench for exactly this regime; Audio Flamingo 3 (NVIDIA, 2025) reports 10-minute support. These are benchmark numbers against other models, not degradation curves at fixed difficulty — no published $D(T)$ sweep isolates length from difficulty.
- **Million-token multimodal context.** Gemini 1.5 (Google DeepMind, 2024) reports near-perfect audio "needle" retrieval over ~9–11 hours of audio in a synthetic haystack. This is a single retrieval-shaped probe: it establishes that a needle *token* is reachable, not that reasoning over the haystack is intact. No open reproduction exists.
- **SSM/linear-attention audio backbones.** Mamba-class models (Gu & Dao, COLM 2024) give $O(N)$ inference and are being applied to audio, but no head-to-head at matched training FLOPs against dense attention on hour-scale audio reasoning has been published.

## 4. What Is Known

- **Token cost is dominated by frame rate.** At $r = 25$ Hz, 1 h $=$ 90,000 audio tokens. At Mimi's 12.5 Hz, 45,000. Downsampling audio 2× is worth more than any attention trick that yields a constant-factor speedup.
- **Encoder windows are hard limits, not soft ones.** Whisper's encoder takes exactly 3000 mel frames (30 s) with learned sinusoidal position embeddings; feeding longer input requires retraining or re-interpolating those embeddings. Measured at `large-v3` scale (1.55 B params).
- **Position bias exists in text and is expected in audio.** "Lost in the middle" (Liu et al., TACL 2024) shows accuracy on 20-document retrieval dropping by tens of points when the gold document sits mid-context, measured on GPT-3.5/Claude-class models. The audio analogue has not been measured with the same rigour.
- **Long-audio training data is scarce.** LongAudio (Audio Flamingo 2) is on the order of $10^5$ QA pairs over ~$10^3$ hours; by contrast ASR pretraining corpora run to $10^5$–$10^6$ hours (Whisper: 680 k h weak supervision; USM: ~12 M h unlabelled). The instruction data for hour-scale reasoning is 2–3 orders of magnitude thinner than the acoustic pretraining data.
- **Long-form ASR benchmarks exist and are non-saturated.** Earnings-21 (Del Rio et al., Interspeech 2021) provides 39 hours of full-length earnings calls with entity-dense reference transcripts; WERs there remain materially above LibriSpeech-clean for the same models.

## 5. What Is Not Known

- **Empirically open.** Does $D(T)$ stay flat to $T = 1$ h for any existing model under a difficulty-matched family? The experiment is runnable on 8×H100 today. Nobody has published the curve. Likewise: does frame-rate reduction ($25 \to 6$ Hz) trade off gracefully, and is the crossover task-dependent?
- **Empirically open.** Is audio context extension a *fine-tuning* problem or a *pretraining* problem? Whether $10^3$ h of long-audio instruction data suffices to unlock an LLM already competent at 128 k text tokens is untested at matched compute.
- **Methodologically blocked.** There is no accepted difficulty-matched long-audio benchmark. Without $|S(q)|$ held fixed, every reported "long-audio" score conflates length and difficulty, and a model can score well by ignoring context entirely (linguistic priors on podcast content are strong).
- **Theoretically open.** No lower bound on recurrent state size for audio question-answering over $T$ seconds. Streaming lower bounds exist for set-disjointness-style tasks and would transfer if a natural audio task were shown to embed one — the reduction has not been written.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent ground truth**.

- Length and difficulty are entangled by construction. Making a clip longer usually adds distractors, more speakers, and a harder question. A drop in $M$ therefore has at least three explanations and the published number distinguishes none.
- Hour-scale audio has no cheap ground truth. Reference transcripts for 39 h of earnings calls cost human-days; reference *summaries* or *event counts* over an hour have no single correct answer, so metrics fall back to LLM judges, which themselves have position bias over long contexts — the judge inherits the failure being measured.
- Synthetic needles dodge both problems and measure the wrong thing. Inserting a spoken sentence into a 9-hour haystack tests retrieval of an acoustically anomalous token. Real long-audio tasks require aggregation over evidence that is individually unremarkable — the two have never been shown to correlate.

Compute is a secondary obstruction: an $N = 90{,}000$-token prefill costs ~$400\times$ the attention FLOPs of a 4,500-token one, which bounds sweep size but does not block the experiment.

## 7. Current Research (as of 2026)

- **Long-audio instruction tuning.** NVIDIA's Audio Flamingo line (2024–2025) is the clearest public push at 5–10 minute inputs with a purpose-built benchmark.
- **Low-frame-rate audio tokenizers.** Mimi (Kymatique/Kyutai, Moshi, 2024) at 12.5 Hz and successor codecs at $\le$ 12.5 Hz make hour-scale sequences fit in text-scale context windows; the open question is what is lost. *(frontier — verify: several 2025–2026 codecs claim sub-10 Hz semantic rates.)*
- **Hierarchical / two-stage audio agents.** Segment, index, retrieve, then reason over retrieved spans. Strong engineering baseline; under-reported because it is not end-to-end.
- **Hybrid attention–SSM audio backbones.** Active in academia and at frontier labs. *(frontier — verify: no matched-FLOPs hour-scale audio comparison is public.)*
- **Native-audio frontier models.** Gemini, GPT-4o-class, and Qwen-Omni successors accept long audio; degradation curves are not published. *(frontier — verify.)*

## 8. Concrete Next Experiment

**"Fixed-Needle, Variable-Haystack" audio sweep.**

- **Scale.** 4 durations $T \in \{2, 10, 30, 60\}$ min; 400 items per duration; single open 7–8 B audio-LLM (e.g. Qwen2-Audio-class or Audio Flamingo 2) plus one frontier API model. Runs on 8×H100 in under 3 days including prefill.
- **Construction.** Take real meeting/podcast audio. Insert exactly one 15 s *evidence span* carrying a fact answerable by a 4-way multiple-choice question. Hold constant across all $T$: the evidence span itself, the question, the number of inserted distractor spans (3), and the answer-option set. Vary only surrounding real audio and needle position $p \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$. Difficulty is now length-invariant by construction — this is the part missing from existing benchmarks.
- **Control arms.** (a) $T = 2$ min, needle only — the ceiling. (b) Text-transcript-only arm: same items through an ASR pipeline plus a 128 k-context text LLM — isolates whether the deficit is audio-specific. (c) Chunk-and-vote arm: 30 s windows, per-window answer, majority vote — the trivial baseline any end-to-end model must beat.
- **Deciding number.** $\Delta = \text{Acc}(T{=}2\ \text{min}) - \text{Acc}(T{=}60\ \text{min})$, at fixed difficulty, averaged over $p$. If $\Delta \le 2$ points, end-to-end long-audio context is real and the field should move to aggregation tasks. If $\Delta \ge 15$ points while the transcript arm holds within 2 points, the bottleneck is the *audio* context path, not the language model — which redirects effort to encoders and tokenizers rather than to attention mechanisms.
- Secondary readout: the $p$-curve. A U-shape reproduces "lost in the middle" in the audio modality for the first time.

## 9. Key References

- **[Foundational]** Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever. *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML, 2023. — arXiv:2212.04356
- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[SOTA]** Sreyan Ghosh, Zhifeng Kong, Sonal Kumar, S. Sakshi, Jaehyeon Kim, Wei Ping, Rafael Valle, Dinesh Manocha, Bryan Catanzaro. *Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities.* ICML, 2025. — arXiv:2503.03983
- **[SOTA]** Gemini Team, Google. *Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context.* Technical report, 2024. — arXiv:2403.05530
- **[SOTA]** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR, 2024. — arXiv:2309.00071
- **[Method]** Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian. *Extending Context Window of Large Language Models via Positional Interpolation.* 2023. — arXiv:2306.15595
- **[Method]** Max Bain, Jaesung Huh, Tengda Han, Andrew Zisserman. *WhisperX: Time-Accurate Speech Transcription of Long-Form Audio.* Interspeech, 2023. — arXiv:2303.00747
- **[Method]** Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, Neil Zeghidour. *Moshi: A Speech-Text Foundation Model for Real-Time Dialogue.* 2024. — arXiv:2410.00037
- **[Method]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Benchmark]** Miguel Del Rio, Natalie Delworth, Ryan Westerman, Michelle Huang, Nishchal Bhandari, Joseph Palakapilly, Quinten McNamara, Joshua Dong, Piotr Żelasko, Miguel Jetté. *Earnings-21: A Practical Benchmark for ASR in the Wild.* Interspeech, 2021. — arXiv:2104.11348
- **[Benchmark]** S. Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, Dinesh Manocha. *MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark.* ICLR, 2025. — arXiv:2410.19168
- **[Related]** Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Chao Zhang. *SALMONN: Towards Generic Hearing Abilities for Large Language Models.* ICLR, 2024. — arXiv:2310.13289
- **[Related]** Allison Koenecke, Anna Seo Gyeong Choi, Katelyn X. Mei, Hilke Schellmann, Mona Sloane. *Careless Whisper: Speech-to-Text Hallucination Harms.* ACM FAccT, 2024.

## 10. Worked Example

**One 58-minute earnings call. One question: "What quarterly revenue figure did the CFO state?"** The figure is spoken once, at 41:20, in a 12-second span.

*Token budget.* At Qwen2-Audio's 25 Hz: $N = 58 \times 60 \times 25 = 87{,}000$ audio tokens. The evidence span is $12 \times 25 = 300$ tokens — **0.34%** of context.

*Dense-attention cost.* Attention FLOPs scale as $N^2$. Relative to a 30 s window ($N_0 = 750$): $(87{,}000/750)^2 = 1.35 \times 10^4$. One question costs 13,000× the attention work of one Whisper window.

*KV cache.* 32 layers, 8 KV heads, head dim 128, fp16: $2 \times 32 \times 8 \times 128 \times 87{,}000 \times 2\ \text{B} \approx 11.4$ GB — for the audio alone, before any text.

*Now the obstruction.* Suppose the model answers correctly 94% of the time at $T = 2$ min and 71% at $T = 58$ min. Four explanations are all consistent with that 23-point drop:

1. Positional dilution — attention mass over 87,000 tokens cannot concentrate on 300.
2. Position bias — 41:20 is 71% through, in the "middle" trough.
3. Acoustic distractors — the full call contains *nine other* dollar figures; the 2-minute clip contains one.
4. Encoder position embeddings extrapolated past their training range, corrupting the span's representation before the LLM sees it.

Existing long-audio benchmarks report the 23 points and stop. The experiment in §8 separates them: fixing the distractor count to 3 at every $T$ kills (3); sweeping $p$ isolates (2); the transcript-only control arm isolates (4) from (1), because ASR-then-text preserves the evidence but discards the audio encoder. Until that separation is run, "our model supports 1-hour audio" is a claim about the input shape, not about the model.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*