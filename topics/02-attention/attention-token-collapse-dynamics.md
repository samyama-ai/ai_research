---
id: 02-attention/attention-token-collapse-dynamics
title: "Attention Dynamics as Clustering and Token Collapse"
topic: 02-attention
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Dynamics as Clustering and Token Collapse

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-token-collapse-dynamics` · **Status:** partially-solved

## 1. Problem Statement

Stacked self-attention is an interacting-particle system: each token's update is a convex-combination pull toward other tokens. Iterating it contracts the token cloud. Depending on parameters the cloud goes to one point (single cluster / rank collapse), to a few points (metastable clustering), or stays spread. The problem has three variants that are routinely conflated.

- **Theory variant.** Given the idealized continuous-time attention flow, characterize the $t \to \infty$ limit as a function of $\beta$ (inverse temperature), $d$ (width), and the value/query/key matrices: how many clusters, on what timescale, and with what basins of attraction.
- **Measurement variant.** Given a *trained* transformer and real text, decide whether a given layer is in a collapsing regime, using a statistic that is not an artifact of layer norm, residual scale, or a handful of outlier tokens (attention sinks, massive activations).
- **Method variant.** Given that collapse is diagnosed, produce an architectural or optimizer intervention that removes it and *improves downstream loss at fixed compute* — not merely one that moves the diagnostic statistic.

Solving it means: (i) a proof of the cluster count and timescale for non-symmetric, non-identity $QKV$; (ii) a collapse statistic that predicts held-out loss degradation across scales; (iii) an intervention with a positive, ablated compute-matched delta.

## 2. Formal Setting

Tokens $x_1,\dots,x_n \in \mathbb{R}^d$, stacked as $X \in \mathbb{R}^{n\times d}$. One head with $Q,K,V \in \mathbb{R}^{d\times d}$:

$$A_{ij} = \frac{\exp(\beta \langle Qx_i, Kx_j\rangle)}{\sum_{k}\exp(\beta \langle Qx_i, Kx_k\rangle)}, \qquad \beta = 1/\sqrt{d_{qk}} .$$

**Continuous-time (residual-stream) limit.** Treating depth as time with layer step $\to 0$, and projecting onto the sphere $\mathbb{S}^{d-1}$ (the layer-norm surrogate):

$$\dot{x}_i(t) = \mathbf{P}_{x_i^{\perp}}\!\left(\sum_{j=1}^{n} A_{ij}(t)\, V x_j(t)\right), \qquad \mathbf{P}_{x^\perp} = I - xx^{\top}.$$

**Measured quantities.**

- *Residual to rank one* (Dong et al.): $\mathrm{res}(X) = \|X - \mathbf{1}x^{\top}\|$ with $x = \arg\min_x \|X-\mathbf{1}x^\top\|$, computed under the composite norm $\|\cdot\|_{1,\infty} = \sqrt{\|\cdot\|_1\|\cdot\|_\infty}$. Measured per layer on the post-attention activations, *before* residual add, on a fixed batch.
- *Token uniformity / cosine collapse*: $u(X) = \frac{2}{n(n-1)}\sum_{i<j} \cos(\tilde{x}_i,\tilde{x}_j)$ on mean-centred, norm-normalized tokens. $u \to 1$ is collapse.
- *Cluster count*: $\hat{k}(\varepsilon) = $ number of connected components of the $\varepsilon$-graph on $\{\tilde x_i\}$, reported as a curve over $\varepsilon$, not a single number.
- *Sink mass*: $s = \frac{1}{nH}\sum_{h,i} A^{(h)}_{i,1}$, the fraction of attention on the first (BOS) token.
- *Effective rank*: $\exp(H(\sigma/\|\sigma\|_1))$ on the singular values of $X$.

**Assumptions and their violations.** (a) $V=Q=K=I$ and $\beta$ shared — violated in every trained model; heads are low-rank and asymmetric. (b) Continuous depth — violated: real depth is 24–100 discrete steps, so asymptotic $t\to\infty$ statements may never be reached. (c) Sphere projection as layer norm — violated: RMSNorm has learned per-channel gains that break isotropy, and massive activations sit on 2–4 channels. (d) No MLP, no residual, no causal mask — all violated; the causal mask alone makes the dynamics non-exchangeable and gives token 1 a fixed point.

## 3. State of the Art

**Theory SOTA (established).** Geshkovski, Letrouit, Polyanskiy, Rigollet, *The emergence of clusters in self-attention dynamics* (NeurIPS 2023): for the flow above with $V=Q=K=I$ on $\mathbb{S}^{d-1}$, all tokens converge to a single cluster as $t\to\infty$ for almost every initialization; for $d \ge 2$ and generic data the convergence is exponential after a metastable phase in which $O(1)$ clusters persist for long times. Dong, Cordonnier, Loukas (ICML 2021): pure attention converges to rank one **doubly exponentially** in depth, $\mathrm{res}(\mathrm{SAN}^L(X)) \le \left(\frac{4\gamma h}{\sqrt{d_{qk}}}\right)^{(3^L-1)/2}\mathrm{res}(X)^{3^L}$; skip connections defeat this, MLPs slow it by a bounded factor.

**Theory SOTA (claimed, partially ablated).** Alcalde, Fantuzzi, Zuazua (2024) give exact cluster characterization for *hardmax* attention — clean, but hardmax is not the deployed nonlinearity. Castin, Ablin, Peyré, *How smooth is attention?* (ICML 2024) give Lipschitz constants for self-attention on compact domains; the constants are loose by orders of magnitude at realistic $n$.

**Empirical SOTA (established).** Attention sinks are the dominant empirical fact. Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (ICLR 2024). Sun et al., *Massive Activations in Large Language Models* (COLM 2024). Darcet et al., *Vision Transformers Need Registers* (ICLR 2024). Barbero et al., *Transformers need glasses!* (NeurIPS 2024) tie over-squashing to representational collapse of late-sequence tokens.

**Benchmark-number-only.** Anti-collapse ViT methods (DeepViT re-attention; Wang et al., *Anti-oversmoothing in deep vision transformers via the Fourier domain analysis*, ICLR 2022) report ImageNet top-1 gains at 24–32 layers, but not at compute-matched width-vs-depth controls, and not in the LLM regime at all. Treat as benchmark numbers, not mechanism.

## 4. What Is Known

- **Rank collapse is real without residuals, and absent with them.** Dong et al. (ICML 2021): removing skip connections from a 12-layer BERT-size model drives $\mathrm{res}(X)$ to $\sim 10^{-6}$ of its input value within ~6 layers; with skips it stays $O(1)$.
- **Rank collapse at init causes vanishing query/key gradients.** Noci et al. (NeurIPS 2022): in post-LN transformers the gradient norm w.r.t. $W_Q,W_K$ decays with depth; scaling the residual branch by $1/\sqrt{2L}$ removes it. Verified up to 32 layers.
- **Attention sinks are quantitatively extreme.** Sun et al. (COLM 2024): in LLaMA-2-7B, activations of magnitude $\approx 2\times10^3$ appear at 2 fixed channels on the BOS token against a median magnitude $\approx 0.1$ — four orders of magnitude. Xiao et al. (ICLR 2024): keeping 4 sink tokens plus a rolling KV cache keeps perplexity stable over 4M tokens where naive window eviction diverges.
- **Sinks are learned, not architectural.** Gu et al., *When Attention Sink Emerges in Language Models: An Empirical View* (ICLR 2025): sink formation depends on data distribution, optimization, and the softmax normalization; replacing softmax with non-normalized sigmoid attention removes the sink in models up to 1B params.
- **Vision transformers grow the same artifact and it costs accuracy.** Darcet et al. (ICLR 2024): DINOv2 ViT-L produces high-norm patch tokens ($\sim$ 10–100× median norm) at $\sim$2% of positions; adding 4 register tokens removes them and improves dense-prediction metrics.

## 5. What Is Not Known

- **Theoretically open.** Cluster count and timescale for generic non-symmetric $QKV$ with a causal mask. All sharp results assume $V=Q=K=I$ or hardmax and no mask. No theorem predicts the *number* of metastable clusters at finite depth $L\le 100$ for trained weights.
- **Theoretically open.** Whether the attention sink is the *mechanism that prevents* collapse (a "no-op" pressure valve) or a *symptom* of it. Barbero et al. (2025) argue the first-token sink acts as a bias that slows mixing; there is no proof.
- **Empirically open.** Whether any collapse statistic ($u$, effective rank, $\hat k$) measured at 1B params predicts loss degradation at 70B. Runnable; not run with matched data and tokenizer across scales.
- **Methodologically blocked.** "Collapse" has no agreed measurement. Cosine uniformity, effective rank and $\mathrm{res}(\cdot)$ disagree in sign on the same layer once 2 massive-activation channels are present, because those channels dominate the norm.

## 6. Why It Is Hard

**Confounded measurement, specifically.** Every scalar collapse statistic is computed on the residual stream, which is dominated by a handful of outlier channels and 1–4 outlier tokens. Removing the sink token changes $u(X)$ by more than a full doubling of depth does. So the statistic measures "how big is the sink" rather than "how mixed are the tokens", and the field's headline quantity does not measure what it names.

Second obstruction: **non-identifiability**. Layer norm makes the dynamics invariant to per-layer rescaling, so a layer that looks contracted can be an exact reparametrization of one that does not. Any diagnostic must be invariant to $X \mapsto cX$ and to per-channel gain absorption; none of the published ones are.

Third: **absent ground truth**. There is no independent label for "this layer collapsed too much". The only ground truth is downstream loss, and collapse interventions are entangled with regularization effects, so the compute-matched control is expensive.

## 7. Current Research (as of 2026)

- **Mean-field / PDE analysis of transformers** — Rigollet, Polyanskiy, Geshkovski, Letrouit (MIT/Sorbonne/Paris). Extending clustering theorems to time-varying and low-rank $QKV$, and to the measure-to-measure interpolation view. Live direction.
- **Sink mechanism** — Barbero, Veličković and collaborators (Google DeepMind / Oxford); Gu et al. (NUS/Sea AI Lab). Sink-as-bias, over-squashing, and softmax-free alternatives.
- **Register tokens and outlier-free architectures** — Meta FAIR (Darcet, Oquab, Bojanowski); outlier-free training via QK-norm and attention-bias terms is now standard in several open LLM recipes. *(frontier — verify which 2026 open models ship a learned no-op key.)*
- **Signal-propagation-driven depth scaling** — Noci et al. (ETH), residual scaling $1/\sqrt{L}$ and depth-$\mu$P style parametrizations.

## 8. Concrete Next Experiment

**Question.** Is sink-corrected token mixing, measured at small scale, predictive of loss at larger scale — or is the collapse statistic an artifact?

- **Scale.** Pretrain 4 decoder-only models on the same 100B-token corpus: 160M, 410M, 1.4B, 2.8B params, depths 12/24/24/32, fixed tokenizer and data order. Budget: roughly 3k–5k A100-hours total.
- **Instrumentation.** Every 1000 steps, on a held-out 256-sequence probe batch, log per layer: $u(X)$, effective rank, $\mathrm{res}(X)$, sink mass $s$ — each computed twice, once on all tokens/channels and once with the top-4 sink tokens and top-8 outlier channels excluded. Call the corrected uniformity $u^{\dagger}$.
- **Control arm.** Identical runs with a learned no-op key appended to every attention layer (softmax-off-by-one / attention bias), which is known to suppress the sink. Same seeds, same data, same step count.
- **Deciding number.** Spearman $\rho$ between mid-depth $u^{\dagger}$ at 20% of training and final held-out loss, computed *across* the 4 scales. $\rho \ge 0.8$ with the sign consistent in both arms means the corrected statistic is a real predictor and the collapse framing is usable. $|\rho| < 0.3$, or a sign flip between arms, means the diagnostic tracks the sink and not mixing — and the measurement variant is confirmed methodologically blocked. Secondary: final-loss gap between arms; a gap under 0.01 nats says sink suppression is diagnostic-only.

## 9. Key References

- **[Foundational]** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. *Attention Is All You Need.* NeurIPS, 2017. — arXiv:1706.03762
- **[Foundational]** Dong, Cordonnier, Loukas. *Attention is not all you need: pure attention loses rank doubly exponentially with depth.* ICML, 2021. — arXiv:2103.03404
- **[SOTA, theory]** Geshkovski, Letrouit, Polyanskiy, Rigollet. *The emergence of clusters in self-attention dynamics.* NeurIPS, 2023. — arXiv:2305.05465
- **[SOTA, theory]** Geshkovski, Letrouit, Polyanskiy, Rigollet. *A mathematical perspective on Transformers.* Bulletin of the AMS, 2025. — arXiv:2312.10794
- **[SOTA, empirical]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[SOTA, empirical]** Sun, Chen, Kolter, Liu. *Massive Activations in Large Language Models.* COLM, 2024. — arXiv:2402.17762
- **[SOTA, empirical]** Darcet, Oquab, Mairal, Bojanowski. *Vision Transformers Need Registers.* ICLR, 2024. — arXiv:2309.16588
- **[SOTA, empirical]** Gu, Xiang, Hua, Wang, Chen, Wang, You, Liu, Lin. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR, 2025. — arXiv:2410.10781
- Noci, Anagnostidis, Biggio, Orvieto, Singh, Lucchi. *Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse.* NeurIPS, 2022. — arXiv:2206.03126
- Barbero, Banino, Kapturowski, Kumaran, Araújo, Vitvitskyi, Pascanu, Veličković. *Transformers need glasses! Information over-squashing in language tasks.* NeurIPS, 2024. — arXiv:2406.04267
- Castin, Ablin, Peyré. *How Smooth Is Attention?* ICML, 2024. — arXiv:2312.14820
- Wang, Zheng, Chen, Wang. *Anti-Oversmoothing in Deep Vision Transformers via the Fourier Domain Analysis.* ICLR, 2022. — arXiv:2203.05962
- Alcalde, Fantuzzi, Zuazua. *Clustering in pure-attention hardmax transformers and its role in sentiment analysis.* Preprint, 2024. — arXiv:2407.01602

## 10. Worked Example

Take a 24-layer, $d=1024$ decoder, $n=512$ tokens of held-out text, layer 12.

**Raw statistic.** Mean pairwise cosine on normalized, mean-centred tokens: $u = 0.61$. Read naively: 61% of a fully collapsed cloud. Effective rank $= 41$ out of 512 possible directions.

**Now remove the BOS token only** — one token out of 512, 0.2% of the batch. On models of this class the BOS carries $\sim 2\times10^3$-magnitude activations on 2 channels while the median channel magnitude is $\sim 10^{-1}$. Its contribution to the batch mean is therefore $\frac{1}{512}\cdot 2\times10^3 \approx 3.9$ on those channels, against a median-token contribution of $\sim 10^{-1}$: the centring vector is set by one token. Dropping it and recomputing gives $u^{\dagger} \approx 0.2$–$0.3$ and effective rank roughly doubling.

**The obstruction, made visible.** The two numbers, $u=0.61$ and $u^{\dagger}\approx 0.25$, come from the same forward pass and differ by more than the effect of doubling depth from 12 to 24 layers ($\Delta u \approx 0.05$–$0.1$ in published depth sweeps). So the reported quantity is a function of the sink's magnitude, not of how mixed the 511 ordinary tokens are. A paper claiming "layer 12 is over-smoothed, $u=0.61$" and one claiming "layer 12 is healthy, $u^{\dagger}=0.25$" are both correct about their statistic and are not comparable.

The theory side does not rescue this: the clustering theorems assume $V=Q=K=I$ with no mask, a setting in which no sink exists, so they say nothing about which of the two numbers the asymptotic single-cluster result should be compared against. That gap — a sharp theorem for a system without the dominant empirical phenomenon, and a statistic dominated by that phenomenon — is why the problem is *partially solved* rather than open or closed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*