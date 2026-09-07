---
id: 11-inference-and-serving/attention-sink-necessity-streaming
title: "Attention Sink Necessity for Streaming Inference"
topic: 11-inference-and-serving
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Sink Necessity for Streaming Inference

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/attention-sink-necessity-streaming` · **Status:** partially-solved

## 1. Problem Statement

A streaming decoder serves an unbounded token sequence under a fixed KV-cache budget $B$. The naive policy — a sliding window that evicts the oldest key/value pairs — makes perplexity diverge the moment the *first few* tokens leave the cache. StreamingLLM's fix is to pin the first $k \approx 4$ tokens ("attention sinks") permanently and slide the rest. It works. The open problem is *why*, and whether the fix is necessary or incidental.

Three variants, routinely conflated:

- **Measurement.** Define "attention sink" so that a model can be scored as having one, independent of which tokens sit at position 0. Current definitions (mass on token 0, massive activation norm, sink logit magnitude) are correlated but not equivalent, and disagree on non-softmax attention.
- **Method.** Is there a streaming policy at budget $B$ with no pinned prefix that matches pinned-sink perplexity within $\epsilon$? Equivalently: is the sink a *load-bearing* computation or a *repairable artifact* of softmax normalization?
- **Theory.** Prove or refute: for any decoder trained with row-stochastic softmax attention and no null token, evicting the tokens carrying the dominant attention mass induces an $\Omega(1)$ shift in hidden-state norm that no positional-encoding or cache-compression scheme can correct without retraining.

Solving it means: a training-time or inference-time intervention such that streaming at budget $B$ is *exactly* the sliding-window policy, with no privileged tokens, and no perplexity penalty. Partially solved because two such interventions exist (learned sink logits; attention biases) but neither has been shown to dominate pinning at frontier scale, and the theory remains a set of mechanistic stories.

## 2. Formal Setting

Let the decoder have $L$ layers, $H$ heads per layer, head dim $d$. At step $t$, head $(\ell,h)$ computes query $q_t \in \mathbb{R}^d$ against cached keys $\{k_j\}_{j \in \mathcal{C}_t}$, where $\mathcal{C}_t \subseteq \{1,\dots,t\}$ is the retained index set, $|\mathcal{C}_t| \le B$:

$$a_{t,j} = \frac{\exp(q_t^\top k_j / \sqrt{d})}{\sum_{i \in \mathcal{C}_t} \exp(q_t^\top k_i / \sqrt{d})}, \qquad o_t = \sum_{j \in \mathcal{C}_t} a_{t,j} v_j .$$

**Sink mass** (measured): $\ \sigma^{(\ell,h)}_t = \sum_{j \le k} a_{t,j}$ with $k=4$, averaged over $t$ in a held-out stream; the model-level statistic is $\bar\sigma = \frac{1}{LH}\sum_{\ell,h}\sigma^{(\ell,h)}_t$. Measured by logging pre-eviction attention rows on a 100k-token corpus; requires eager attention, since fused FlashAttention kernels do not materialize $a$.

**Massive activation** (measured): $\ m_\ell = \max_{i,j} |x^{(\ell)}_{t,i}| \big/ \mathrm{median}_{i,j}|x^{(\ell)}_{t,j}|$ over residual-stream coordinates $i$ at layer $\ell$. Sun et al. report $m_\ell \sim 10^3$ concentrated in 1–4 fixed coordinates.

**Streaming loss gap** (the decision quantity): with $\mathcal{L}_B(\pi)$ the mean per-token NLL under eviction policy $\pi$ at budget $B$,

$$\Delta(B) = \mathcal{L}_B(\pi_{\text{window}}) - \mathcal{L}_B(\pi_{\text{sink}}), \qquad \pi_{\text{sink}}: \mathcal{C}_t = \{1..k\}\cup\{t-B+k+1..t\}.$$

$\Delta(B) > 0$ is the phenomenon. The **necessity predicate** is: $\exists\,\pi$ with no pinned prefix and $\mathcal{L}_B(\pi) \le \mathcal{L}_B(\pi_{\text{sink}}) + \epsilon$, $\epsilon = 0.02$ nats.

**Assumptions, and where they break.**
- *Softmax rows sum to 1.* Violated by design in gpt-oss-style learned sink logits and in "softmax+1"; the denominator gains a constant term, so $a$ is sub-stochastic and $\bar\sigma$ is undefined as written.
- *Position 0 is special.* Violated whenever a shared system prompt or BOS is prepended per request; the sink is then a *content* token, and its KV entries are shared across requests in production servers.
- *Perplexity tracks streaming quality.* Violated for retrieval-shaped tasks: a policy can hold perplexity flat while destroying long-range recall (needle-in-haystack), because the loss is dominated by locally predictable tokens.
- *Stationary stream.* Real serving traffic is bursty and multi-tenant; $\Delta(B)$ measured on PG19 does not transfer to chat traffic with turn boundaries.

## 3. State of the Art

**Systems/empirical SOTA — established.** StreamingLLM (Xiao et al., ICLR 2024) pins $k=4$ tokens plus a rolling window and holds perplexity stable over streams of ~4M tokens on Llama-2, MPT, Falcon, Pythia, with up to $22.2\times$ speedup over sliding-window-with-recomputation. LM-Infinite (Han et al., NAACL 2024) reaches the same stability with a $\Lambda$-shaped mask plus a distance ceiling on relative position, arrived at independently. Both are reproduced widely; both are *inference-time*, no retraining.

**Training-time SOTA — established but narrow.** Xiao et al. show a 160M model pretrained with one dedicated learnable sink token needs only that token pinned ($k=1$) instead of four. Bondarenko et al. (NeurIPS 2023) show clipped-softmax and gated attention remove activation outliers in BERT/OPT-scale models, improving post-training quantization. Darcet et al. (ICLR 2024) show register tokens remove sink-like artifacts in ViTs. gpt-oss (OpenAI, 2025) ships per-head learned sink logits in production — an existence proof at 20B/120B, but *not* an ablation: no public run of the same recipe without sink logits.

**Claimed but unablated.** That four is the right $k$ (the paper sweeps $k$ on a handful of models, not across data mixtures). That the sink is a "no-op"/over-mixing brake (Barbero et al., 2025) — the correlational evidence is strong, the causal test is not run. That removing sinks costs nothing on long-context tasks — most negative evidence is perplexity-only.

**Benchmark-number-only results.** KV-compression baselines that quietly retain the first tokens — H2O (Zhang et al., NeurIPS 2023), SnapKV, TOVA (Oren et al., EMNLP 2024) — report LongBench/needle scores where the sink contribution is not separated from the scoring heuristic.

## 4. What Is Known

- **Divergence is sharp, not gradual.** Llama-2-7B on a concatenated PG19 stream: pinned-sink perplexity stays near the dense value (~5–6 nats-equivalent) over millions of tokens; the pure sliding window explodes into the $10^3$ range within a few tokens of evicting position 0 (Xiao et al., ICLR 2024, 7B scale).
- **Mass concentration.** In Llama-2-7B, beyond layer 2 the first token receives a large majority of attention mass in most heads, largely independent of its semantic content (7B scale).
- **Massive activations.** LLaMA-2-7B/13B carry residual-stream coordinates ~$10^3\times$ the median at the first token and at weak delimiters; zeroing them collapses the model, fixing them to their mean does not (Sun et al., COLM 2024, 7B–70B).
- **Emergence is a training-dynamics fact.** Gu et al. (ICLR 2025) show the sink appears during pretraining after enough optimization steps and data, its strength varies with LR, batch size, and data distribution, and attention without softmax normalization (e.g. sigmoid attention, no denominator) can be trained to ~1B parameters *without* forming a sink.
- **Sink strength scales with pretraining context length.** Barbero et al. (2025) report stronger first-token sinks in models pretrained on longer contexts, across Gemma and LLaMA-3.1 8B/70B/405B.
- **Quantization coupling.** Outlier suppression at training time improves INT8 post-training quantization by several perplexity points at BERT/OPT scale (Bondarenko et al., 2023) — the sink is a quantization tax, not only a streaming aid.

## 5. What Is Not Known

- **Theoretically open.** No theorem states that softmax-normalized causal attention *must* allocate $\Omega(1)$ mass to a fixed token at convergence. The candidate mechanism — a no-op needed because rows are stochastic — has no proof, and no lower bound on $\Delta(B)$ as a function of $\bar\sigma$.
- **Empirically open.** No matched-compute, frontier-scale ($\ge 8$B, $\ge 1$T tokens) A/B of sink-free training (sigmoid attention, learned sink logit, or register tokens) against a standard softmax control, evaluated on *both* streaming perplexity and long-context retrieval. Gu et al.'s sink-free result stops at ~1B; gpt-oss has no control arm. This is the single largest gap and it is runnable — roughly $10^{22}$–$10^{23}$ FLOPs for a decisive pair.
- **Methodologically blocked.** "Does the model have a sink?" is not well defined once the softmax denominator gains a learned constant. Sink mass, massive-activation ratio, and sink-logit magnitude are three different scalars that agree on vanilla softmax models and diverge exactly on the architectures that would answer the question.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus non-identifiability**. Position 0 is simultaneously: the token with the smallest key-query distance under RoPE for every query; the only token visible to every causal row; and, in practice, BOS or a system prompt. Any experiment that evicts it changes three variables at once. Ablations that shuffle content at position 0 control for semantics but not for the other two.

The second obstruction is that **perplexity does not measure what streaming names**. $\Delta(B)$ is dominated by tokens whose prediction depends on the last few hundred positions, so a policy can look sink-free-and-fine at $\Delta(B) \approx 0$ while losing recall of facts $10^5$ tokens back. Separating "sink needed for numerical stability" from "sink needed for information routing" requires a retrieval metric that most sink papers do not report.

Third, the decisive experiment is a **pretraining** ablation, not an inference tweak: the sink is a property of the learned parameters. That puts the cost at frontier-run scale and outside the reach of the groups that publish most sink analyses.

## 7. Current Research (as of 2026)

- **Sink-free architectures.** Learned per-head sink logits shipped in gpt-oss; sigmoid/unnormalized attention studied by Gu et al. and follow-ons. Open question is whether they hold at 100B+ *(frontier — verify)*.
- **Mechanistic accounts.** Barbero et al. (over-mixing / rank-collapse brake, Google DeepMind + Oxford); Cancedda (ACL 2024) on spectral "dark signals" feeding sink formation; Sun et al. on massive activations as implicit bias terms.
- **Quantization-driven interest.** Sink and outlier suppression as a prerequisite for INT4/FP4 KV caches; several serving stacks now special-case sink KV entries in full precision *(frontier — verify)*.
- **Cache compression that assumes sinks.** SnapKV, PyramidKV, DuoAttention-style retrieval/streaming head splits — all inherit the pinned prefix as an unexamined constant.

## 8. Concrete Next Experiment

**Question.** Is the pinned prefix necessary, or is it a repairable softmax artifact?

**Scale.** Four 1.4B-parameter decoders, identical data (300B tokens, same order, same seed), identical optimizer, 8k context. Arms:
1. **Control:** standard softmax attention, RoPE, BOS.
2. Learned per-head sink logit $z_{\ell,h}$ appended to the softmax denominator: $a_{t,j} = e^{s_{t,j}}/(e^{z_{\ell,h}} + \sum_i e^{s_{t,i}})$.
3. Four prepended learnable register tokens, *not* pinned at eval.
4. Sigmoid attention, no row normalization.

Total ≈ $4 \times 2.5\times10^{21}$ FLOPs — one 8×H100 node-month per arm.

**Evaluation.** For each arm, run the pure sliding window $\pi_{\text{window}}$ at $B = 2048$ over a 2M-token held-out stream with **no pinned tokens at all** (arms 2–4 must not be given a free prefix), and in parallel a 32k-context needle-in-a-haystack retrieval suite.

**Deciding number.** $\Delta(2048)$ in nats/token for each arm, with the control's $\Delta(2048)$ as the reference.
- Any arm with $\Delta(2048) \le 0.02$ nats **and** needle accuracy within 2 points of its own dense-cache score ⇒ the sink is a repairable artifact; pinning is a workaround for an architecture choice.
- All arms $\Delta(2048) > 0.02$ ⇒ evidence the sink is load-bearing, and the theory variant becomes the live question.

Report $\bar\sigma$ and $m_\ell$ per arm as secondary evidence, but do not let them substitute for $\Delta$ — they are the quantities that stop being comparable across arms.

## 9. Key References

- **[Foundational]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[Foundational]** Chi Han, Qifan Wang, Hao Peng, Wenhan Xiong, Yu Chen, Heng Ji, Sinong Wang. *LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models.* NAACL 2024. — arXiv:2308.16137
- **[SOTA]** Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. *Massive Activations in Large Language Models.* COLM 2024. — arXiv:2402.17762
- **[SOTA]** Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, Min Lin. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR 2025. — arXiv:2410.10781
- **[SOTA]** Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu. *Why do LLMs attend to the first token?* 2025. — arXiv:2504.02732
- **[Related]** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS 2023. — arXiv:2306.12929
- **[Related]** Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. *Vision Transformers Need Registers.* ICLR 2024. — arXiv:2309.16588
- **[Related]** Zhenyu Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023. — arXiv:2306.14048
- **[Related]** Matanel Oren, Michael Hassid, Nir Yarden, Yossi Adi, Roy Schwartz. *Transformers are Multi-State RNNs.* EMNLP 2024.
- **[Related]** Nicola Cancedda. *Spectral Filters, Dark Signals, and Attention Sinks.* ACL 2024. (identifier omitted — unverified)

## 10. Worked Example

Take one head of Llama-2-7B at layer 10, decoding token $t = 5000$ with $B = 1024$. Log the pre-eviction row: suppose mass on the first four tokens is $\sigma = 0.62$, spread over the remaining 1020 positions is $0.38$.

Now evict positions 1–4. The unnormalized scores of the survivors are unchanged; only the denominator shrinks. Every surviving weight is multiplied by

$$\frac{1}{1-\sigma} = \frac{1}{0.38} = 2.63 .$$

The head output moves from $o_t = 0.62\,\bar v_{\text{sink}} + 0.38\,\bar v_{\text{ctx}}$ to $o_t' = \bar v_{\text{ctx}}$. The sink value vectors in these heads are small in norm — that is the point of a no-op — so $\|\bar v_{\text{sink}}\| \ll \|\bar v_{\text{ctx}}\|$ and

$$\frac{\|o_t'\|}{\|o_t\|} \approx \frac{1}{0.38} \approx 2.6 .$$

A $2.6\times$ norm inflation in one head, compounded across $L=32$ layers of residual writes the LayerNorm gains were never trained to see, is enough to push per-token NLL from ~5.4 to the $10^3$-perplexity regime reported for pure windowing — with **no information lost**: the evicted tokens carried almost no value mass. That is the obstruction in one line. The failure is a *normalization* failure, not an *information* failure, which is why pinning four near-zero-value tokens fixes it and why a learned denominator constant *should* fix it too.

But now try to verify that claim by measurement. In arm 2 above, the denominator contains $e^{z_{\ell,h}}$, so $\sigma$ is no longer computable — there are no sink tokens to sum over. In arm 4 there is no denominator at all, so the $1/(1-\sigma)$ inflation cannot arise and $\bar\sigma$ is identically undefined. The three architectures whose comparison would settle the question are precisely the three on which the standard sink metric does not exist. Only $\Delta(B)$, the end-to-end streaming loss gap, survives across all four arms — which is why Section 8 makes it the single deciding number.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*