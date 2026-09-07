---
id: 02-attention/attention-entropy-collapse-predictor
title: "Attention Entropy as a Predictor of Training Collapse"
topic: 02-attention
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Entropy as a Predictor of Training Collapse

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-entropy-collapse-predictor` · **Status:** empirically-open

## 1. Problem Statement

Large transformer pretraining runs fail in characteristic ways: loss spikes that do not recover, divergence to NaN, and slow "silent" degradation where loss plateaus above the scaling-law prediction. Practitioners observe that attention entropy — the Shannon entropy of each head's softmax distribution over keys — falls sharply before or during many of these events. The question is whether that observation supports a **predictor**.

Three variants, of very different difficulty:

- **Measurement variant.** Define a scalar statistic $E_t$ from attention entropies at step $t$ that is cheap to compute during training, comparable across layers, heads, sequence lengths, and model widths, and stable under batch resampling. Not yet settled.
- **Method variant (the decision predicate).** Given a run's history $E_{1:t}$, emit at step $t$ a binary alarm $a_t \in \{0,1\}$ predicting that a collapse event occurs within a horizon $H$ steps. Solving it means: at a fixed false-alarm rate (say one per $10^4$ steps), detection recall exceeds that of the strongest cheap baseline — gradient-norm spikes, max attention logit, update-to-parameter RMS ratio — with lead time $\ge H$ large enough to act on.
- **Theory variant.** Prove that low attention entropy is *causal* for divergence under a specified optimizer and parameterization, rather than a co-symptom of a shared upstream driver (growing $\|W_Q W_K^\top\|_2$, Adam second-moment underflow). Currently only one direction is proved: bounds on entropy in terms of query–key spectral norm.

The catalog status is **empirically-open**: the run is affordable at 1B–8B scale and nobody has published the prospective, pre-registered version.

## 2. Formal Setting

Let a model have $L$ layers, $H_{\text{heads}}$ heads, head dimension $d_h$, sequence length $n$. For layer $\ell$, head $h$, batch element $b$, query position $i$:

$$A^{(\ell,h,b)}_{ij} = \mathrm{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt{d_h}} + m_{ij}\right), \qquad m_{ij} = -\infty \text{ for } j>i .$$

Per-query entropy, measured in nats:

$$\mathcal{H}^{(\ell,h,b)}_i = -\sum_{j\le i} A_{ij}\log A_{ij} \in [0, \log(i)] .$$

**Normalization matters and is not standardized.** Causal masking makes the maximum entropy position-dependent, so raw means are confounded with sequence length. Two candidate normalizations:

$$\tilde{\mathcal{H}}_i = \frac{\mathcal{H}_i}{\log i} \quad\text{(fraction of maximum)}, \qquad \mathcal{N}_i = \exp(\mathcal{H}_i) \quad\text{(perplexity / effective number of attended keys)}.$$

$\mathcal{N}_i$ is the more interpretable: a head with $\mathcal{N}=1.0$ is one-hot, $\mathcal{N}=i$ is uniform. Aggregate to a per-head scalar $E^{(\ell,h)}_t = \mathbb{E}_{b,i}[\tilde{\mathcal{H}}_i]$ and to a run-level statistic by a *low* quantile, not the mean — collapse is typically confined to a few heads:

$$E_t = Q_{0.05}\big(\{E^{(\ell,h)}_t\}_{\ell,h}\big).$$

The companion statistic is the max attention logit $M_t = \max_{\ell,h,b,i,j} |q_i^\top k_j / \sqrt{d_h}|$, which is what actually appears in the known bound. Zhai et al. (ICML 2023) prove, for inputs with bounded row norm $\|X\|_{2,\infty}\le \gamma$, an entropy lower bound of the form

$$\mathcal{H}_i \;\ge\; \log n - c\,\gamma^2\,\sigma\big(W_Q W_K^\top\big)\,\exp\!\big(c'\gamma^2 \sigma(W_Q W_K^\top)\big),$$

i.e. entropy is guaranteed high only while the query–key spectral norm stays small, and the guarantee degrades doubly fast. There is no matching upper bound, so low entropy is not *implied* to be dangerous.

**Assumptions known to be violated in practice.**
- *Bounded input norms.* Residual-stream norms grow roughly linearly with depth in real LMs; $\gamma$ is not constant.
- *Entropy is a per-head property.* Attention sinks (Xiao et al., ICLR 2024) put large mass on position 0 or on delimiter tokens; such heads have genuinely low entropy at convergence and are functional, not pathological. Any predictor must separate "sink-low" from "collapse-low".
- *Stationary data.* Curriculum changes, domain-mixture shifts, and long-context extension phases move $E_t$ for reasons unrelated to stability.
- *Collapse is a well-defined event.* It is not. Loss spikes that self-recover, spikes that do not, and NaNs are different events with possibly different mechanisms.

## 3. State of the Art

**Established (ablated, multiple tasks).**
- **σReparam** (Zhai et al., ICML 2023): spectral reparameterization $\hat W = (\gamma/\sigma(W))W$ on attention projections. Prevents entropy collapse and enables ViT training without learning-rate warmup and without LayerNorm; validated across ViT/ImageNet, machine translation, and speech recognition. This establishes that *controlling the quantity that bounds entropy* stabilizes training. It does not establish that entropy is the causal channel, since σReparam also changes the effective learning rate on those weights.
- **QK-LayerNorm** (Dehghani et al., ViT-22B, ICML 2023): LayerNorm on queries and keys before the dot product. Introduced explicitly to fix divergence caused by attention logits growing to $\sim 10^4$ and attention becoming one-hot. Reproduced independently in many open LM stacks (used in OLMo-2, Gemma-2 variants).
- **Small-scale proxies** (Wortsman et al., ICLR 2024): shows the attention-logit-growth instability, first seen at multi-billion scale, reproduces at tens of millions of parameters by raising the learning rate; qk-layernorm widens the stable learning-rate range by roughly three orders of magnitude. Also shows the instability is predictable from small-scale LR sweeps.

**Claimed but unablated.**
- That attention entropy provides *lead time* over gradient norm. Widely repeated in engineering write-ups and training logs; no published ROC curve, no matched false-alarm-rate comparison.
- That a monitoring threshold on $E_t$ can trigger an intervention (LR backoff, batch skip) that saves a run. Intervention-on-spike is standard practice (PaLM skipped ~200–500 batches around spikes), but the trigger used is loss/grad-norm, not entropy.

**Benchmark-number-only results.** Reported entropy curves in model cards and stability papers are single-run traces on single models. They are illustrations, not estimates of predictive performance.

## 4. What Is Known

- **Rank collapse.** Pure self-attention without skip connections or MLPs converges to rank-1 doubly exponentially in depth (Dong, Cordonnier, Loukas, ICML 2021). Skip connections and MLPs counteract it. Rank collapse and entropy collapse are related but distinct: entropy collapse is per-head over keys; rank collapse is over the token dimension of the output.
- **Signal propagation.** Noci et al. (NeurIPS 2022) show rank collapse causes vanishing gradients of the query/key parameters, and that scaling residual branches by $1/\sqrt{L}$ mitigates it — a depth-dependent, not entropy-dependent, fix.
- **Logit magnitudes at scale.** ViT-22B: attention logits reaching order $10^4$ with near-one-hot attention immediately preceding divergence, at 22B parameters. Wortsman et al. reproduce the same signature at ~40M–4.8B parameters under elevated learning rates.
- **Optimizer-side mechanism.** Molybog et al. (2023) attribute large-scale Adam instability to a mismatch between the second-moment estimate and the gradient's time scale after long stationary periods — a mechanism with no attention-entropy term at all, at OPT-175B scale. This is the leading confounder.
- **Sinks are normal.** Trained LMs reliably allocate large attention mass to the first token; entropy of such heads is low throughout training and does not indicate pathology (Xiao et al., ICLR 2024).

## 5. What Is Not Known

- **Empirically open.** Whether $E_t$ (or $M_t$) yields higher recall than gradient norm at matched false-alarm rate, with a usable lead time. The experiment needs on the order of 50–100 runs that actually diverge; each is affordable at 300M–1B scale. Nobody has published it.
- **Empirically open.** Whether the lead time, if any, survives to 70B+ scale, or whether entropy collapse becomes simultaneous with the loss spike.
- **Methodologically blocked.** What counts as a "collapse event". Without a labelling rule that separates recoverable spikes from terminal divergence, recall and precision are undefined. Also blocked: normalization across heads with different sink structure.
- **Theoretically open.** Whether low entropy is causal. There is no theorem showing that a low-entropy attention pattern, holding logit magnitude fixed, increases divergence probability. Non-identifiability is the core issue: $E_t \downarrow$ and $M_t \uparrow$ are two views of one underlying quantity in the proven bound.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus absent ground truth**, not compute.

1. *Confounded measurement.* $E_t$ and $M_t$ are analytically coupled through the softmax; the established theory bounds one by the other. Any observed predictive power of entropy is therefore not attributable to entropy specifically unless an intervention moves entropy while holding the logit spectrum fixed. Entropy regularization on the attention distribution is such an intervention, and it is rarely run.
2. *Absent ground truth.* Divergence is rare per run and dependent on seed, data order, and hardware nondeterminism. Estimating a ROC curve needs many positive events, which means deliberately destabilized runs — and destabilized runs may fail through a *different* mechanism than production 70B runs do.
3. *Evaluation that does not measure what it names.* Post-hoc traces show entropy dropping before a spike in the run that spiked. That is conditioning on the outcome. The quantity that matters — $P(\text{collapse} \mid E_t \text{ low})$ — requires the many runs where entropy dropped and nothing happened, which are not logged or published.

## 7. Current Research (as of 2026)

- **Architectural prophylaxis over prediction.** QK-norm, σReparam, tanh logit soft-capping (Gemma-2), and $1/\sqrt{L}$ residual scaling are now defaults in open training stacks. The field's revealed preference is to remove the failure mode rather than forecast it. *(frontier — verify: whether entropy monitoring persists in any frontier lab's production stability dashboard.)*
- **Attention-sink mechanism work.** Ongoing analysis of why sinks form and how they interact with quantization and long-context extension; relevant because sinks are the main false-positive source for any entropy predictor.
- **Small-proxy stability prediction.** Extending the Wortsman et al. programme: predicting large-scale instability from cheap small-scale LR sweeps, which competes directly with in-run monitoring as a solution to the same operational problem.
- **Optimizer-side diagnostics.** Update/parameter RMS ratios and second-moment staleness as alternative early-warning signals *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** 60 pretraining runs, 350M-parameter decoder-only LM (24 layers, $d=1024$, 16 heads), 30B tokens each, sequence length 2048. Roughly $6\times10^{20}$ FLOPs total — order 2,000 H100-hours. Induce a spread of outcomes by sampling learning rate log-uniformly over a range straddling the stability edge (found by a 6-run pilot), plus varied seeds and data orders. Log every 10 steps: $E_t$ (5th-percentile normalized head entropy), $M_t$, global grad norm, update/param RMS, loss.

**Labelling rule (fixed before running).** A run is a positive at step $t^\*$ if the 100-step moving-average loss rises by $\ge 0.15$ nats above its running minimum and fails to return within 2,000 steps, or produces NaN.

**Control arm.** Same alarm framework driven by global gradient norm alone (z-score over a 1,000-step window). Second control: max attention logit $M_t$. Third, causal arm: 12 runs at the same LR settings with an entropy floor imposed by an auxiliary loss $\lambda\max(0, \tau - \bar{\mathcal{H}})$ that raises entropy *without* spectral reparameterization.

**Deciding number.** Recall at a fixed false-alarm budget of one alarm per 10,000 steps, with lead time $H = 500$ steps. Entropy is a useful predictor only if

$$\text{Recall}_{E} - \max\big(\text{Recall}_{\text{gradnorm}}, \text{Recall}_{M}\big) \ge 0.20$$

with a bootstrap 95% CI excluding zero. If the gap is under 0.05, the entropy story is a redescription of logit growth and should be retired as a separate diagnostic. The causal arm decides the second question: if the entropy floor cuts the divergence rate by half at matched learning rate, entropy is on the causal path; if not, it is a symptom.

## 9. Key References

- **[Foundational]** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. *Attention Is All You Need.* NeurIPS, 2017. — arXiv:1706.03762
- **[SOTA]** Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, Josh Susskind. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML, 2023. — arXiv:2303.06296
- **[SOTA]** Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, Simon Kornblith. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- Mostafa Dehghani et al. *Scaling Vision Transformers to 22 Billion Parameters.* ICML, 2023. — arXiv:2302.05442
- Yihe Dong, Jean-Baptiste Cordonnier, Andreas Loukas. *Attention is not all you need: pure attention loses rank doubly exponentially with depth.* ICML, 2021. — arXiv:2103.03404
- Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, Aurelien Lucchi. *Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse.* NeurIPS, 2022. — arXiv:2206.03126
- Igor Molybog, Peter Albert, Moya Chen, Zachary DeVito, David Esiobu, Naman Goyal, Punit Singh Koura, Sharan Narang, Andrew Poulton, Ruan Silva, Binh Tang, Puxin Xu, Yuchen Zhang, Melanie Kambadur, Stephen Roller, Susan Zhang. *A Theory on Adam Instability in Large-Scale Machine Learning.* 2023. — arXiv:2304.09871
- Aakanksha Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR, 2023. — arXiv:2204.02311
- Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Survey]** Sho Takase, Shun Kiyono, Sosuke Kobayashi, Jun Suzuki. *Spike No More: Stabilizing the Pre-training of Large Language Models.* 2023. (arXiv preprint; identifier omitted.)

## 10. Worked Example

Take one head, $n=2048$, causal mask, query at position $i=1024$. Maximum entropy is $\log 1024 = 6.93$ nats, so $\mathcal{N}_{\max} = 1024$ effective keys.

Suppose the head has a sink on token 0 with mass $p_0$ and spreads the rest uniformly over the remaining 1023 keys. Then

$$\mathcal{H} = -p_0\log p_0 - (1-p_0)\log\frac{1-p_0}{1023}.$$

| $p_0$ | $\mathcal{H}$ (nats) | $\tilde{\mathcal{H}}$ | $\mathcal{N}=e^{\mathcal H}$ |
|---|---|---|---|
| 0.00 | 6.93 | 1.00 | 1024 |
| 0.50 | 3.81 | 0.55 | 45 |
| 0.90 | 1.03 | 0.15 | 2.8 |
| 0.99 | 0.13 | 0.019 | 1.14 |

Now the pathological case at the same entropy: attention concentrated on 2.8 effective *content* keys with no sink, e.g. $p=0.9$ on one mid-sequence token. $\tilde{\mathcal{H}}=0.15$ in both cases. **The statistic cannot tell them apart.** The first is a healthy trained sink head; the second is the one-hot pattern reported before ViT-22B divergence.

Now the logit side. To get $p_0 = 0.99$ with 1023 competitors at equal logits requires a logit gap of $\log(0.99 \cdot 1023 / 0.01) \approx 11.5$. To reach $\tilde{\mathcal{H}} = 10^{-3}$ needs a gap near 18. But observed pre-divergence logits are $\sim 10^4$ — four orders of magnitude past the point where entropy has already saturated at its floor.

That is the obstruction, quantitatively: **entropy is a saturating function of the logit spread.** Once $M_t \gtrsim 20$, $E_t$ is pinned near zero and carries no further information, while $M_t$ keeps rising by a factor of 500 before the run dies. Any monitor built on entropy alone loses resolution exactly in the regime where the divergence is developing — which is the strongest prior reason to expect the deciding number in §8 to come out near zero, and the reason the experiment is worth running rather than assuming.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*