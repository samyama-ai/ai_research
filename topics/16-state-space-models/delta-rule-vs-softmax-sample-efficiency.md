---
id: 16-state-space-models/delta-rule-vs-softmax-sample-efficiency
title: "Delta Rule Versus Softmax Attention Sample Efficiency"
topic: 16-state-space-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Delta Rule Versus Softmax Attention Sample Efficiency

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/delta-rule-vs-softmax-sample-efficiency` · **Status:** empirically-open

## 1. Problem Statement

Delta-rule linear recurrences (DeltaNet, Gated DeltaNet, RWKV-7) replace softmax attention's growing KV cache with a fixed-size matrix state updated by one step of online gradient descent per token. At matched parameter count they reach comparable perplexity on standard web-text pretraining. The open question is not final loss but **rate**: does a delta-rule model need more tokens to reach the same loss as a softmax Transformer, and if so, how does that penalty scale?

Three variants, of very different difficulty:

- **Measurement.** Define a sample-efficiency ratio that is not an artifact of the matching convention (parameters, FLOPs, wall-clock, or state bytes). Currently unsettled.
- **Method.** Find a delta-rule architecture whose ratio is $\le 1$ at $\ge 7$B parameters and $\ge 1$T tokens, with no attention layers. Unrun.
- **Theory.** Prove a separation (or its absence) in tokens-to-target-loss between a fixed-state online-regression memory and unbounded-cache attention, on a natural data distribution rather than a synthetic recall task. Open.

Solving it means: a reproducible ratio $\rho$, with confidence intervals over seeds, measured across at least a decade of token budget, plus an account of which data properties drive it.

## 2. Formal Setting

**The recurrence.** Per head, state $S_t \in \mathbb{R}^{d_v \times d_k}$, keys $k_t$ with $\|k_t\|_2 = 1$, values $v_t$, learning rate $\beta_t \in (0,1)$ produced by a sigmoid on a token-dependent projection:

$$S_t = S_{t-1}\left(I - \beta_t k_t k_t^\top\right) + \beta_t v_t k_t^\top, \qquad o_t = S_t q_t.$$

This is exactly one SGD step on $\ell_t(S) = \tfrac12\|S k_t - v_t\|_2^2$ with step $\beta_t$ (Schlag et al., 2021). Gated DeltaNet adds a scalar decay $\alpha_t$: $S_t = S_{t-1}(\alpha_t I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$.

**Memory budget, as measured.** Fixed state bytes $M_\Delta = b \cdot L \cdot H \cdot d_k d_v$ for $L$ layers, $H$ heads, precision $b$. Softmax cache at context $T$: $M_{\text{soft}}(T) = 2 b \cdot L \cdot H_{kv} d_h \cdot T$. Crossover $T^\star = d_k d_v H / (2 H_{kv} d_h)$ — for $H=16$, $d_k=d_v=128$, MHA, $T^\star \approx 1024$. Report $T^\star$ with every comparison; it is the only honest statement of what is being traded.

**Sample efficiency.** With loss $\mathcal{L}_A(N, D)$ for architecture $A$ at $N$ non-embedding parameters and $D$ tokens from a fixed corpus in a fixed order, define $D_A(\ell) = \min\{D : \mathcal{L}_A(N,D) \le \ell\}$ and

$$\rho(\ell, N) = \frac{D_\Delta(\ell)}{D_{\text{soft}}(\ell)}.$$

$\rho > 1$ means the delta rule is less sample-efficient. Measure $\ell$ on a held-out shard, in nats/token under the **same tokenizer**, on the smoothed training curve, not the last checkpoint.

**Assumptions, and which fail.**
1. *Matched $N$ implies matched compute.* False: at $T=4096$ a linear-attention layer is roughly $2\times$ cheaper in FLOPs, so a FLOP-matched comparison gives the delta rule more tokens for free. Report both $\rho$ curves.
2. *Loss curves are monotone and seed-stable.* Approximately true at $>10^{10}$ tokens; the published 1.3B comparisons are single-seed, and $\rho$ near 1 is inside seed noise.
3. *A single $\rho$ exists.* Violated by construction — $\rho$ depends on $\ell$, and the copying/recall literature predicts it grows as $\ell$ falls.
4. *Corpus is exchangeable.* Violated: recall-heavy data (code, long documents, multi-turn retrieval) is unevenly distributed across web-text shards.

## 3. State of the Art

**Empirical SOTA (established).** Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (NeurIPS 2024), gives a chunkwise WY-representation kernel making DeltaNet trainable at scale; models at 340M and 1.3B parameters, 100B tokens. Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025) adds decay and reports the strongest pure-linear results at 1.3B/100B on FineWeb-Edu. RWKV-7 "Goose" (Peng et al., 2025) uses a generalized (vector-valued $\beta$) delta rule at up to 2.9B parameters. All report per-token perplexity and zero-shot commonsense averages at a **single token budget** — none publishes a loss-versus-tokens curve against a matched Transformer, so none measures $\rho$.

**Claimed but unablated.** That gated delta-rule models "match or exceed Transformer++" at 1.3B. The reported gaps on WikiText perplexity are fractions of a point, from single seeds, at one budget; no seed variance is reported, and the Transformer baseline's hyperparameters are tuned less than the proposed model's. This is a benchmark number, not an ablation.

**Systems SOTA.** Hybrids dominate deployment. Waleffe et al. (NVIDIA, 2024) trained 8B-parameter Mamba-2, Transformer, and Mamba-2-Hybrid on 3.5T tokens; the hybrid beat the Transformer by 2.65 points averaged over 12 standard tasks, while pure Mamba-2 lagged on 5-shot MMLU and phonebook-style retrieval. Qwen3-Next and Kimi Linear (2025) ship Gated-DeltaNet-style layers in ~3:1 linear:attention stacks *(frontier — verify ratios and ablations)*.

**Theory SOTA.** Merrill, Petty & Sabharwal (ICML 2024) place fixed-state SSMs in uniform $\mathrm{TC}^0$: no $S_5$ word-problem solution at fixed depth. Grazzi et al. (ICLR 2025) show extending the eigenvalue range to $[-1,1]$ ($\beta_t \in (0,2)$) lets DeltaNet solve parity and modular arithmetic; DeltaProduct (Siems et al., 2025) composes $n_h$ Householder reflections per token to climb the state-tracking hierarchy. All are expressivity results. **None bounds sample complexity.**

## 4. What Is Known

- **Delta rule beats decay-only recurrence on recall.** DeltaNet solves MQAR (multi-query associative recall) at state sizes where GLA and Mamba fail, at 340M-scale ablations (Yang et al., 2024).
- **Recall explains most of the quality gap.** Arora et al., *Zoology* (ICLR 2024): associative-recall performance accounts for up to **82%** of the perplexity gap between attention and gated-convolution models at 70M–360M parameters on the Pile.
- **Recall is bounded by state, not parameters.** Arora et al., *Based* (ICML 2024): recall accuracy on real recall-intensive tasks is a smooth function of recurrent state size, at 355M–1.3B parameters — a Pareto frontier, not an architectural accident.
- **A hard separation exists on copying.** Jelassi et al., *Repeat After Me* (ICML 2024): Transformers learn string copying with far fewer training examples and generalize to lengths beyond training, while state-space models need state linear in string length; pretrained Pythia-410M beats Mamba-1.4B on copying and phone-book retrieval.
- **The gap closes at fixed budget on average web text.** At 1.3B/100B, Gated DeltaNet's zero-shot commonsense average is within ~1 point of a tuned Transformer++ — a regime where $\rho \approx 1$ at that particular $\ell$.

## 5. What Is Not Known

- **Empirically open.** The entire $\rho(\ell, N)$ surface. Nobody has published matched loss-versus-tokens curves for a delta-rule model and a softmax Transformer over a decade of $D$ at a fixed $N \ge 1$B, with $\ge 3$ seeds. The experiment is runnable today on ~$10^{4}$ GPU-hours. This is the central gap.
- **Empirically open.** Whether $\rho$ is stable or diverges as $\ell \to$ irreducible loss. The recall results predict divergence; the 1.3B results are consistent with stability. Both are compatible with all published data.
- **Theoretically open.** Any sample-complexity separation. There is no theorem of the form "learning distribution class $\mathcal{C}$ to error $\epsilon$ requires $\Omega(f(\epsilon))$ more samples with a rank-1-update fixed state than with attention." Expressivity results ($\mathrm{TC}^0$, $S_5$) say nothing about learning rate.
- **Methodologically blocked.** The matching convention. Parameter-matched, FLOP-matched, and state-byte-matched comparisons give different $\rho$ and can order the architectures differently. No community standard exists, so published claims are not comparable across papers.

## 6. Why It Is Hard

**Confounded measurement, compounded by cost.** The quantity of interest is a ratio of token budgets, but every published number is a single point on a curve whose position depends on three free choices — matching convention, corpus recall-density, and hyperparameter tuning effort — each capable of moving perplexity by more than the effect being measured. The delta rule's advantage is concentrated in the tail of the loss distribution (rare long-range recall events), which contributes a small fraction of average nats/token; average perplexity is therefore an evaluation that does not measure what it names. Resolving it requires curves, not points, and a 1.3B $\times$ 300B-token sweep with 3 seeds and 2 architectures is ~$10^{4}$–$10^{5}$ A100-hours — affordable for a lab, unaffordable for the ablation-per-hypothesis rate the question needs. The result is a literature of single-seed, single-budget benchmark tables.

## 7. Current Research (as of 2026)

- **Expressivity ladders.** DeltaProduct (Siems, Grazzi et al., ELLIS/Freiburg) trades $n_h$ Householder products per token for state-tracking power; the sample-efficiency cost of each rung is unmeasured.
- **Kernel efficiency.** `flash-linear-attention` (Songlin Yang, MIT) is the de facto reference implementation and makes the decisive experiment cheap enough to run.
- **Hybrid ratio search.** NVIDIA, Qwen, Moonshot (Kimi Linear), MiniMax — all converging on ~1 attention layer in 4 or fewer. The empirical fact that everyone hybridizes is the strongest indirect evidence that $\rho > 1$ somewhere *(frontier — verify: ratios are reported, ablations against pure-linear at matched tokens mostly are not)*.
- **Test-time-regression framing.** Treating the recurrence as online learning (delta rule = SGD; DeltaNet variants with momentum or Newton steps) invites importing online-learning regret bounds — the most plausible route to the theory variant.

## 8. Concrete Next Experiment

**Scale.** $N = 1.3$B non-embedding parameters, $D = 300$B tokens of FineWeb-Edu, context 4096, identical tokenizer, identical data order, 3 seeds per arm.

**Arms.** (a) Gated DeltaNet, pure linear. (b) **Control:** Transformer++ (RoPE, SwiGLU, GQA) with hyperparameters tuned to the *same* budget as the treatment arm — an equal-tuning protocol, logged. (c) Second control: Gated DeltaNet with 3:1 hybrid attention, to locate the hybrid on the same curve.

**Instrumentation.** Checkpoint every $2\times$ in tokens from 1B. Evaluate held-out loss on two shards: general web, and a recall-dense shard (long documents with $\ge 1$ verbatim repeated 32-gram). Report $M_\Delta$ and $T^\star$ for the configuration.

**The deciding number.** $\rho(\ell^\star) = D_\Delta(\ell^\star)/D_{\text{soft}}(\ell^\star)$ at $\ell^\star$ = the loss the Transformer reaches at 100B tokens, with a seed-derived 95% CI. **$\rho \le 1.1$ with the CI excluding 1.3** on both shards refutes a meaningful sample-efficiency penalty. **$\rho \ge 1.5$ on the recall-dense shard while $\rho \approx 1$ on general web** localizes the penalty to recall and explains the industry's hybrid consensus. A cost of roughly $2\times10^{4}$ H100-hours.

## 9. Key References

- **[Foundational]** Schlag, Irie & Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML, 2021. — arXiv:2102.11174
- **[SOTA]** Yang, Wang, Zhang, Shen, Kim & others. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS, 2024. — arXiv:2406.06484
- **[SOTA]** Yang, Kautz & Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR, 2025. — arXiv:2412.06464
- **[SOTA]** Peng et al. *RWKV-7 "Goose" with Expressive Dynamic State Evolution.* 2025. — arXiv:2503.14456
- **[Theory]** Merrill, Petty & Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[Theory]** Grazzi, Siems, Franke, Zela, Hutter & Pontil. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR, 2025. — arXiv:2411.12537
- **[Separation]** Jelassi, Brandfonbrener, Kakade & Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Empirical]** Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra & Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[Empirical]** Arora et al. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML, 2024. — arXiv:2402.18668
- **[Empirical]** Waleffe et al. *An Empirical Study of Mamba-based Language Models.* 2024. — arXiv:2406.07887
- **[Scaling]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556

## 10. Worked Example

Take a 1.3B Gated DeltaNet: $L = 24$, $H = 16$, $d_k = d_v = 128$. Fixed state:

$$M_\Delta = 2\ \text{bytes} \times 24 \times 16 \times 128 \times 128 = 12.6\ \text{MB}.$$

A GQA Transformer of the same size with $H_{kv}=4$, $d_h=128$ caches $2 \times 2 \times 24 \times 4 \times 128 \times T = 49{,}152\,T$ bytes. Crossover: $T^\star \approx 256$ tokens. Past 256 tokens of context the delta-rule model is strictly memory-bounded relative to attention; at $T = 128$K it holds $12.6$ MB against the Transformer's $6.3$ GB — a $500\times$ compression.

Now count what must fit. The delta rule stores associations as a rank-1 sum in $S \in \mathbb{R}^{128\times128}$ per head; with near-orthogonal keys, one head retrieves at most $\sim d_k = 128$ associations before interference. Sixteen heads, 24 layers gives a loose ceiling of $\sim$49K stored key-value pairs — but only if keys are orthogonal and $\beta_t$ is well calibrated, and the write is *lossy by design*: $S_{t-1}(I - \beta_t k_t k_t^\top)$ erases the component of every prior association along $k_t$.

**Where the obstruction becomes visible.** Suppose 0.1% of tokens in a 128K-token document require retrieving a fact written more than 100K tokens earlier, and the delta-rule model gets 60% of them right against attention's 95%. The loss penalty is $0.001 \times \left[\ln(0.95/0.60)\right] \approx 4.6 \times 10^{-4}$ nats/token. Typical held-out loss at this scale is ~2.0 nats/token, and seed-to-seed variation on a 100B-token run is on the order of $10^{-3}$ nats/token. **The effect is three to five times smaller than the noise floor of the metric everyone reports.** A 35-point absolute drop in long-range retrieval accuracy is invisible in average perplexity — which is precisely why the 1.3B/100B tables show parity, why practitioners nevertheless keep one attention layer in four, and why $\rho$ must be measured on a recall-stratified shard rather than on the corpus mean.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*