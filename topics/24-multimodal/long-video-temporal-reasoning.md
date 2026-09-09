---
id: 24-multimodal/long-video-temporal-reasoning
title: "Long-Video Temporal Reasoning Without Frame Subsampling"
topic: 24-multimodal
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Video Temporal Reasoning Without Frame Subsampling

> **Topic:** Multimodal Models · **ID:** `24-multimodal/long-video-temporal-reasoning` · **Status:** open

## 1. Problem Statement

Every deployed video-language model answers questions about an hour-long video by first throwing away 95–99.9% of its frames. A 1-hour video at 30 fps is 108,000 frames; typical inference uses 32 to 768. The question is whether the discarded frames carry information that the task needs, and whether any architecture can use them at a cost that scales sub-quadratically in video length.

Three variants, with different difficulty:

- **Measurement.** Build a benchmark on which a model's score is a monotone, non-saturating function of the sampling rate $r$ (frames per second) over the range $r \in [0.1, 30]$. No existing public benchmark does this: scores on most long-video suites plateau by $r \approx 1$ fps, which means they cannot detect whether dense-frame reasoning was achieved. This variant is **methodologically blocked**.
- **Method.** Given a video of $T$ seconds at native frame rate, produce answers to compositional temporal queries (order, duration, counting, causality, state change) using a computational budget growing as $O(T\log T)$ or better, matching the accuracy of a hypothetical dense-attention oracle over all frames.
- **Theory.** Characterise which temporal predicates are computable from a subsampled frame set, and prove separations: a class of queries answerable at rate $r$ but provably not at $r/2$ for any decoder.

Solved means: a model whose accuracy on a benchmark with certified sampling-rate sensitivity rises with $r$ up to native frame rate, at cost linear-ish in $T$, and where the gain survives an ablation against a same-compute subsampling baseline.

## 2. Formal Setting

A video is $V = (f_1,\dots,f_N)$, $N = \lceil T \cdot \mathrm{fps}\rceil$ frames. A **sampling policy** $\pi$ selects indices $S \subseteq [N]$, possibly query-dependent. An **encoder** $E$ maps each selected frame to $k$ tokens; total visual context is
$$ n = k\,|S| . $$
Measured, not assumed: $k$ is read off the model card. Qwen2.5-VL at $448\times448$ with $2\times2$ patch merging gives $k=256$; LLaVA-style pooling gives $k \in \{4, 64, 144\}$.

**Cost.** For a transformer with $L$ layers, $H_{kv}$ KV heads, head dim $d_h$, precision $b$ bytes:
$$ \mathrm{KV}(n) = 2\,L\,H_{kv}\,d_h\,b\cdot n \ \text{bytes}, \qquad \mathrm{FLOPs}_{\mathrm{attn}} \approx 4\,L\,H\,d_h\,n^2 . $$
Both are measured directly (peak device memory; profiler FLOP counts), not estimated from parameter count.

**Sampling-rate sensitivity.** For benchmark $B$ and model $M$, define
$$ \sigma_B(M) = \frac{\mathrm{Acc}_B(M, r=30) - \mathrm{Acc}_B(M, r=0.5)}{\mathrm{Acc}_B(\text{human}) - \mathrm{Acc}_B(\text{chance})} . $$
Measured by re-running the same checkpoint at both rates with everything else fixed. A benchmark with $\sigma_B \approx 0$ for all $M$ cannot be used to study this problem — that is the blocking condition in §5.

**Aliasing bound.** For an event of duration $\delta$ seconds and uniform sampling at rate $r$, the probability that at least one frame falls inside it is $\min(1, r\delta)$; the expected number of frames inside is $r\delta$, so **ordering** two sub-events requires $r\delta \geq 2$ in expectation. This is the Nyquist-style constraint the field mostly ignores.

Assumptions **known violated in practice**: (i) frames are i.i.d.-informative — false, adjacent frames are near-duplicates, which is exactly why subsampling works so well; (ii) benchmark questions require the full video — false, many are answerable from one frame or from the text alone (§4); (iii) uniform sampling is the right control — false, query-conditioned retrieval beats it, so uniform is a weak baseline that inflates apparent gains.

## 3. State of the Art

**Systems/empirical SOTA (established).** Long-context multimodal models exist and run. Ring Attention (Liu, Zaharia, Abbeel, ICLR 2024) and the Large World Model (Liu, Yan, Abbeel et al., 2024) demonstrate ~1M-token contexts covering roughly an hour of video at low frame rate. Gemini 1.5 Pro (Google, 2024) reports up to 10.5M tokens and near-perfect retrieval of a planted frame across hours of video. Qwen2.5-VL, LongVILA, Video-XL and LLaVA-Video push open-weight context to $10^5$–$10^6$ visual tokens with token merging, dynamic-fps encoding and time-aware position encodings.

**Claimed but unablated.** That these systems perform *temporal reasoning* over the long context, as opposed to retrieval plus a strong language prior. Needle-in-a-video-haystack recall is a retrieval measurement, not a reasoning one: the planted frame is visually anomalous and a single-frame detector suffices. Almost no published long-video system reports the sampling-rate sweep $\sigma_B$ that would separate the two.

**Benchmark-number-only results.** Video-MME (Fu et al., CVPR 2025) long-split scores in the high 60s for frontier models, MLVU and LongVideoBench leaderboards, and TemporalBench figures are all leaderboard entries without the frame-budget-matched control arm. Treat them as reported numbers, not as evidence about dense-frame reasoning.

**Theory SOTA.** Essentially absent. There is no published separation theorem for temporal predicates under subsampling. Sub-quadratic sequence models (Mamba, Gu & Dao 2023; linear attention) give the cost side but no accompanying expressivity result for video-specific temporal queries.

## 4. What Is Known

- **Single-frame bias is large and reproduced.** Buch et al. (CVPR 2022) showed an atemporal single-frame probe matches or beats video-language models on NExT-QA and MSRVTT-QA subsets. Lei, Berg & Bansal (ACL 2023) showed single-frame training matches or beats multi-frame training on several video-text retrieval and QA benchmarks. Two independent groups, different tasks.
- **Text-only and blind baselines are strong.** On EgoSchema (Mangalam et al., NeurIPS 2023 D&B; 3-minute clips, 5,000 questions), LLM-only baselines that never see pixels score well above chance (20%); human accuracy on the certified subset is about 76%.
- **Accuracy saturates in frame count.** Across LongVideoBench (Wu et al., NeurIPS 2024 D&B) and MLVU, open-weight models typically gain only a few points going from 8 to 64 frames and are flat or slightly worse beyond ~256 frames. Scale: 7B–34B models, videos of 3 minutes to 1 hour.
- **Fine-grained temporal understanding is far below human.** TemporalBench (Cai et al., 2024) reports a ~30-point gap between the best proprietary models and humans on multi-binary temporal-description accuracy. Vinoground (Zhang et al., CVPR 2025) reports frontier models near 35% group score on temporal counterfactual pairs against ~90% human, on videos averaging under 10 seconds — i.e. the failure is not about length.
- **Subtitles carry much of the signal.** Video-MME's with-subtitle condition improves frontier models by several points on the long split, indicating a large fraction of the answerable content is linguistic.
- **Cost is real.** KV-cache memory is linear in tokens with a large constant (§10), and full attention is quadratic; this is why every system subsamples.

## 5. What Is Not Known

- **Methodologically blocked (primary).** Whether any public long-video benchmark has $\sigma_B > 0.1$. Nobody has published the full sampling-rate sweep on a frontier model for Video-MME-long, MLVU, LVBench or LongVideoBench. Until that exists, "long-video reasoning" scores cannot be attributed to temporal processing.
- **Empirically open.** Whether dense sampling helps at all once compute is held fixed. The decisive comparison — $N$ tokens spent as 32 frames × 256 tokens versus 512 frames × 16 tokens — is runnable today on 8 H100s and has not been run as a clean sweep across token-per-frame budgets and video lengths.
- **Empirically open.** Whether query-conditioned retrieval over a dense index (retrieve-then-read) is strictly dominated by, equal to, or better than end-to-end long context at matched cost.
- **Theoretically open.** No proof that any natural temporal predicate class is inexpressible under rate-$r$ sampling but expressible at $2r$, for a decoder with unbounded capacity and a learned frame prior. The aliasing bound in §2 is an information argument about a *single* event, not a separation for a learned model that can exploit priors and context.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by absent ground truth**, not compute.

Benchmark questions are written by annotators who watched the video, then answer-checked by models. Questions that a strong language prior can guess survive; questions that genuinely need frame 41,203 are expensive to author and hard to verify. The result is a corpus whose difficulty lives mostly in language and single-frame recognition. A model can therefore improve its long-video score by getting better at reading subtitles and priors, and the benchmark cannot tell the difference. This is the classic "evaluation does not measure the thing it names."

Second obstruction: **non-identifiability of the gain**. Any dense-frame method changes three things at once — token budget, positional encoding range, and training distribution. Attributing an accuracy delta to "seeing more frames" requires holding the other two fixed, which almost no paper does.

Compute is a secondary but real barrier: native-rate attention over an hour of video is $\sim 10^{15}$ score entries per head per layer, so the oracle arm of the decisive experiment cannot be run directly and must be approximated (§8).

## 7. Current Research (as of 2026)

- **Sub-quadratic and hybrid backbones** for video: state-space and linear-attention video encoders, sliding-window plus global-token hybrids. Active at Meta, NVIDIA, Alibaba (Qwen-VL line), and academic groups (Berkeley, NUS/Show Lab).
- **Learned token compression**: query-conditioned pooling, memory banks (MA-LMM, CVPR 2024), streaming KV eviction for video. Empirically strong; theory absent.
- **Retrieval-augmented video QA**: index frames or clips, retrieve on the query, read a small window. Widely reported to match long-context at a fraction of cost *(frontier — verify: matched-compute comparisons are still rare)*.
- **Temporally certified benchmarks**: benchmarks built by construction so that the answer provably depends on sub-second structure (counterfactual pairs, synthetic event grammars). Vinoground and TemporalBench are the closest published instances.
- **Native-fps training** with time-aware position encodings (Qwen2.5-VL's absolute-time mRoPE variant) *(frontier — verify the ablation isolating the encoding from the extra frames)*.

## 8. Concrete Next Experiment

**The frame-budget isoquant sweep.**

- **Scale.** One 7B-class open video-LLM (e.g. Qwen2.5-VL-7B or LLaVA-Video-7B), 8×H100. 500 videos of 20–60 minutes drawn from LongVideoBench and LVBench, plus 500 *constructed* items: each is a 20–60 minute video with a 0.3–0.6 s two-sub-event insertion, and a question whose answer flips with the order of the two sub-events. Ground truth is known by construction, so no annotator prior can leak.
- **Arms.** Fix the visual token budget at $n \in \{16\text{k}, 64\text{k}, 256\text{k}\}$. Within each budget, sweep frames-per-token trade: $(|S|, k) \in \{(64,256), (256,64), (1024,16), (4096,4)\}$. **Control arm:** uniform subsampling at $|S|=32$, $k=256$ — the standard deployed configuration — plus a text-only (blind) arm and a subtitle-only arm to bound the language prior.
- **Deciding number.** $\sigma$ on the constructed split: accuracy at $(4096,4)$ minus accuracy at the $(32,256)$ control, normalised by (human − chance). **If $\sigma < 0.05$ at every token budget, dense frames buy nothing and the field's subsampling default is correct.** If $\sigma > 0.2$ at the largest budget, dense-frame architecture is worth the engineering, and the same sweep on the natural split tells you how much of existing benchmark headroom it explains.
- **Cost.** ~2,000 GPU-hours. No new pretraining; inference-only sweeps plus optional light finetuning at each $(|S|,k)$ to remove the train/test frame-count mismatch confound.

## 9. Key References

- **[Foundational]** Buch, Eyzaguirre, Gaidon, Wu, Fei-Fei, Niebles. *Revisiting the "Video" in Video-Language Understanding.* CVPR, 2022.
- **[Foundational]** Lei, Berg, Bansal. *Revealing Single Frame Bias for Video-and-Language Learning.* ACL, 2023. — arXiv:2206.03428
- **[Foundational]** Mangalam, Akshulakov, Malik. *EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2308.09126
- **[SOTA]** Liu, Zaharia, Abbeel. *Ring Attention with Blockwise Transformers for Near-Infinite Context.* ICLR, 2024. — arXiv:2310.01889
- **[SOTA]** Liu, Yan, Hashimoto, Abbeel et al. *World Model on Million-Length Video and Language with Blockwise RingAttention.* 2024. — arXiv:2402.08268
- **[SOTA]** Gemini Team, Google. *Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context.* Technical report, 2024. — arXiv:2403.05530
- **[Benchmark]** Fu et al. *Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis.* CVPR, 2025. — arXiv:2405.21075
- **[Benchmark]** Wu et al. *LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2407.15754
- **[Benchmark]** Cai et al. *TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models.* 2024.
- **[Benchmark]** Zhang et al. *Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos.* CVPR, 2025.
- **[Method]** Bolya, Fu, Dai, Zhang, Feichtenhofer, Hoffman. *Token Merging: Your ViT But Faster.* ICLR, 2023. — arXiv:2210.09461
- **[Method]** He, Fan, Ma, Zhu et al. *MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding.* CVPR, 2024.
- **[Method]** Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023. — arXiv:2312.00752

## 10. Worked Example

A 1-hour cooking video at 30 fps. At 0:41:07 the cook adds salt, then vinegar, 0.4 s apart. Question: "Which went in first?"

**Aliasing.** With uniform sampling at $r$ fps, expected frames landing inside the 0.4 s window is $r\delta$. Ordering needs two.

| $r$ (fps) | frames in window | ordering possible? |
|---|---|---|
| 0.5 | 0.2 | no |
| 1 | 0.4 | no |
| 5 | 2.0 | marginal |
| 30 | 12 | yes |

The deployed default (32 frames over 3,600 s, $r = 0.0089$ fps) gives $r\delta = 0.0036$ — the event is invisible 99.6% of the time. The model still answers, from the recipe prior. It is right more often than chance, and the benchmark scores it as temporal reasoning.

**Cost of not subsampling.** Qwen2.5-VL-7B: $L=28$, $H_{kv}=4$, $d_h=128$, bf16 ($b=2$). Per-token KV:
$$ 2\cdot 28\cdot 4\cdot 128\cdot 2 = 57{,}344\ \text{bytes} \approx 56\ \text{KiB}.$$

| policy | frames | tokens/frame | tokens $n$ | KV cache |
|---|---|---|---|---|
| deployed | 32 | 256 | 8,192 | 0.47 GB |
| 1 fps | 3,600 | 256 | 921,600 | 52.8 GB |
| 5 fps, pooled | 18,000 | 16 | 288,000 | 16.5 GB |
| native | 108,000 | 256 | 27.6M | **1.58 TB** |

Native-rate attention also costs $\sim n^2 = 7.6\times10^{14}$ score entries per head per layer.

**The obstruction made visible.** The 5-fps-pooled row fits on one 8×H100 node (16.5 GB KV) and clears the ordering threshold ($r\delta = 2$). So the compute barrier is *not* what blocks progress at this event scale. What blocks it is that if you run that configuration on Video-MME-long or MLVU, you will likely see a gain of one or two points inside noise — because those benchmarks contain almost no items whose answer flips on a 0.4 s ordering. You cannot tell whether the method failed or the ruler is blind. That is why §8 constructs its own items with known ground truth: the measurement has to be built before the method can be evaluated.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*