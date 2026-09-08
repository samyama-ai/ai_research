---
id: 02-attention/attention-temperature-long-context
title: "Attention Temperature and Logit Scaling for Long Context"
topic: 02-attention
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Temperature and Logit Scaling for Long Context

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-temperature-long-context` · **Status:** empirically-open

## 1. Problem Statement

Softmax attention normalizes over all $n$ keys. As $n$ grows, a fixed budget of logit magnitude is spread over more competitors, so the attention distribution provably flattens. The standard fix is to rescale the logits — equivalently, lower the softmax temperature — by some function of context length. The problem: **what is the correct temperature schedule $\tau(n)$, and is any length-only schedule sufficient?**

Three variants, with different difficulty:

- **Theory.** Given a transformer trained at length $n_0$, does there exist a length-dependent logit scale $c(n)$ such that the layer's function on length-$n$ inputs extends the length-$n_0$ function in a specified sense (e.g. preserves argmax retrieval, or bounds per-head KL drift)? Chiang & Cholak (ACL 2022) prove $c(n)=\Theta(\log n)$ suffices for a *constructed* hard-attention-emulating network. Whether it suffices for a *trained* one is open.
- **Method.** Find $\tau(\cdot)$ — possibly per-head, per-layer, or input-adaptive — that maximizes long-context task accuracy at fixed training compute, ideally without retraining.
- **Measurement.** Define a quantity that separates "the softmax dispersed" from "the position encoding went out of distribution" from "the model never learned the retrieval circuit." No current benchmark does this cleanly.

Solving it means: a schedule with a derivation, an ablation isolating it from RoPE rescaling, and a gain that survives at $\ge$ 7B parameters and $\ge$ 128k tokens.

## 2. Formal Setting

For head $h$ at layer $\ell$, query index $i$, keys $j \le i$:

$$a_{ij} = \frac{\langle q_i, k_j\rangle}{\sqrt{d_h}} + b_{ij}, \qquad p_{ij} = \frac{\exp(a_{ij}/\tau)}{\sum_{j'\le i}\exp(a_{ij'}/\tau)}$$

with $d_h$ the head dimension and $b_{ij}$ any positional bias (ALiBi slope, zero for RoPE, which acts inside $\langle q_i,k_j\rangle$). Baseline is $\tau=1$; the "temperature" intervention sets $\tau = \tau(n)$ or multiplies logits by $c(n)=1/\tau(n)$.

Measured quantities, as they would actually be logged:

- **Attention entropy** $H_i^{(\ell,h)} = -\sum_j p_{ij}\log p_{ij}$, in nats, averaged over the last 256 query positions of a held-out document, per head per layer. Report also $H/\log i$ (normalized to $[0,1]$) so lengths are comparable.
- **Peak mass** $m_i = \max_j p_{ij}$, and **effective support** $k_i = \exp(H_i)$ ("how many keys the head is really reading").
- **Logit gap** $\Delta_i = a_{i,j^\star} - \max_{j\ne j^\star} a_{ij}$ where $j^\star$ is the ground-truth evidence token in a synthetic retrieval probe. This is the only quantity that separates *dispersion* from *wrong ranking*, and it needs a task with known $j^\star$.
- **Length generalization gap** $\mathcal{G}(n) = \mathcal{L}(n) - \mathcal{L}(n_0)$, per-token NLL on documents of true length $n$ versus training length $n_0$, matched for domain.

Standing assumptions and their status:

1. *Logits are bounded independent of $n$* — approximately true when QK-norm or weight decay is used, **violated** in unnormalized models where $\|q\|$ grows with training and attention sinks carry very large logits.
2. *Entropy loss is the binding constraint* — **violated in part**: RoPE out-of-distribution rotations degrade the ranking of $\langle q,k\rangle$ itself, which no temperature can repair.
3. *A single scalar $\tau$ per model* — **violated**: heads differ by orders of magnitude in entropy; "retrieval heads" are a small minority.
4. *Softmax denominator sums over informative keys only* — **violated** by sinks (Xiao et al., 2024), which absorb 30–90% of mass in some heads and act as an implicit learned temperature.

The bounding fact: if all logits lie in $[-B,B]$, then $m_i \le e^{2B}/(n-1+e^{2B}) \to 0$. Sharpness at length $n$ *requires* logit range growing like $\log n$.

## 3. State of the Art

**Theory SOTA (established).** Hahn (TACL 2020) shows soft attention cannot represent certain hard-attention functions with bounded logits. Chiang & Cholak (ACL 2022) close the gap by scaling logits with $\log n$, giving a transformer that recognizes PARITY at arbitrary length — the cleanest existence proof that the temperature must be length-dependent. Veličković et al. (2024, arXiv:2410.01104) restate this as "softmax cannot stay sharp out of distribution" and derive an adaptive-temperature head.

**Empirical SOTA (established by ablation).** YaRN (Peng et al., ICLR 2024) combines NTK-by-parts RoPE interpolation with an attention temperature $\sqrt{1/t} = 0.1\ln s + 1$ for extension factor $s$, applied by rescaling $q$ and $k$ so it is free at inference. Their ablation shows the temperature term alone improves long-context perplexity over the same interpolation without it — the strongest direct evidence the intervention is real. LongRoPE (Ding et al., ICML 2024) reuses a similar scaling in its 2048k extension.

**Claimed but unablated.** Scalable-Softmax (Nakanishi, 2025, arXiv:2501.19399) replaces $\exp(a)$ with $n^{sa}$, i.e. a logit scale of exactly $s\log n$, and reports better long-context retrieval; the reported scale is small (sub-1B) and the comparison is not matched for position-encoding treatment. Gemma-2's logit soft-capping and ViT-22B/Chameleon-style QK-norm are *deployed* temperature-adjacent interventions with no published isolating ablation of their long-context effect.

**Benchmark-number-only.** Most "needle in a haystack" pass rates attributed to temperature come from bundled recipes (interpolation + continued pretraining + temperature). They are not attributable to any single term.

## 4. What Is Known

- The $\log n$ requirement is a theorem for the bounded-logit construction (Chiang & Cholak, ACL 2022), not a heuristic.
- Entropy collapse in the *other* direction — logits too large, training instability — is real and fixable by $\sigma$Reparam / QK-norm (Zhai et al., ICML 2023, ViT scale up to 1B; Henry et al., Findings of EMNLP 2020).
- Attention sinks: the first few tokens absorb large mass; keeping 4 sink tokens plus a rolling window lets a 7B Llama-2 stream to 4M tokens with stable perplexity (Xiao et al., ICLR 2024). This is a denominator effect, i.e. an implicit temperature.
- YaRN's temperature is fit, not derived: $0.1\ln s + 1$ was chosen by sweep on LLaMA 7B/13B at $s\in\{8,16,32\}$; at $s=32$ it corresponds to a logit multiplier of $1/t \approx 1.82$.
- Naive extension without any correction degrades sharply: LLaMA-family perplexity blows up within a few thousand tokens past $n_0$ (Chen et al., 2023, arXiv:2306.15595, 7B/13B).
- Retrieval capability is concentrated: a small set of heads ($\sim$3–6% in 6B–34B models) accounts for most in-context retrieval, so a global $\tau$ mostly perturbs heads that do not matter.

## 5. What Is Not Known

- **Theoretically open.** Whether $c(n)=\Theta(\log n)$ is *necessary and sufficient* for trained (not constructed) transformers, under realistic assumptions on the empirical logit distribution. No lower bound matches the construction.
- **Theoretically open.** Whether a length-only schedule can exist at all when the informative-key count grows sublinearly in $n$ — the right variable may be effective support, not $n$.
- **Empirically open.** The isolating ablation at frontier scale: same checkpoint, same RoPE treatment, only $\tau$ varied, 7B+ and 128k+, measured on retrieval with known $j^\star$. Runnable today; not published.
- **Empirically open.** Per-head/per-layer $\tau$ versus one global scalar. Cheap to fit, never reported at scale.
- **Methodologically blocked.** Separating "softmax dispersed" from "RoPE ranking degraded." Perplexity and needle accuracy conflate them. The logit-gap probe $\Delta_i$ is the obvious instrument and is not standard.

## 6. Why It Is Hard

**Confounded measurement.** Every deployed long-context recipe changes the position encoding, the temperature, and the data mixture together. Temperature enters through the same product $\langle q,k\rangle$ that RoPE rotates, so a change in $\tau$ and a change in RoPE base $\theta$ produce nearly identical perplexity curves while doing different things to ranking. Perplexity is dominated by local tokens — a 128k model can lose all long-range retrieval and move NLL by under 0.02 nats/token, so the headline metric is nearly blind to the effect under study.

Secondary: **non-identifiability.** $\tau$ is absorbable into $\|q\|\|k\|$, so any global scale can be re-learned during continued pretraining; the intervention is only meaningful for a *fixed* checkpoint or under a fixed norm constraint (QK-norm makes it identifiable). Without stating which, "we tuned attention temperature" is underdetermined.

## 7. Current Research (as of 2026)

- Adaptive, input-dependent temperature: heads that set $\tau$ from their own entropy (Veličković et al., DeepMind). Reported at small scale only.
- $\log n$-in-the-architecture: Scalable-Softmax and variants that bake $s\log n$ into the normalizer so no post-hoc extension factor is needed *(frontier — verify at $\ge$7B)*.
- Attention-sink theory: sinks as learned no-op denominators, and whether removing them (via sigmoid or off-by-one softmax) removes the need for a temperature schedule *(frontier — verify)*.
- QK-norm as standard practice in open frontier models, which changes the problem: with normalized $q,k$, logits are bounded by construction and the $\log n$ term becomes explicit and tunable.
- Hybrid attention/SSM stacks, where only a few full-attention layers exist and per-layer $\tau$ is cheap to fit *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** One 7B RoPE model with QK-norm (so logit scale is identifiable), continued-pretrained to 128k once. Frozen thereafter. Inference-only sweep.

**Arms.**
- **Control:** the checkpoint as trained, $c=1$.
- **A:** global $c(n) = \alpha\log n$, $\alpha \in \{0, 0.1, 0.25, 0.5, 1.0\}$.
- **B:** YaRN's $c = (0.1\ln s + 1)^2$.
- **C:** per-head $c_h$ fit by minimizing retrieval loss on 512 held-out synthetic items, 1 scalar per head.
- **RoPE treatment held identical in every arm.** No retraining in any arm.

**Task.** Synthetic multi-key retrieval with known evidence position $j^\star$, 4 distractors of matched surface form, evaluated at $n \in \{4\text{k}, 16\text{k}, 64\text{k}, 128\text{k}\}$, $j^\star$ uniform over depth.

**The deciding number.** Retrieval accuracy at $n=128$k, arm C minus control, with 95% CI over 2000 items. **$\ge$ 10 points is a real effect; $\le$ 3 points means length-only temperature is a second-order knob and the failure is in the position encoding.** Secondary read-out: mean logit gap $\Delta$ at 128k — if $\Delta<0$ in the control (wrong key ranked first), no temperature can help and the diagnosis is RoPE, not softmax.

## 9. Key References

- **[Foundational]** Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. — arXiv:1706.03762
- **[Foundational]** Hahn. *Theoretical Limitations of Self-Attention in Neural Sequence Models.* TACL 2020. — arXiv:1906.06755
- **[Foundational]** Chiang, Cholak. *Overcoming a Theoretical Limitation of Self-Attention.* ACL 2022. — arXiv:2202.12172
- **[SOTA]** Peng, Quesnelle, Fan, Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR 2024. — arXiv:2309.00071
- **[SOTA]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[SOTA]** Veličković, Perivolaropoulos, Barbero, Pascanu. *softmax is not enough (for sharp out-of-distribution).* 2024. — arXiv:2410.01104
- **[Related]** Chen, Wong, Chen, Tian. *Extending Context Window of Large Language Models via Positional Interpolation.* 2023. — arXiv:2306.15595
- **[Related]** Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 2024. — arXiv:2104.09864
- **[Related]** Press, Smith, Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022. — arXiv:2108.12409
- **[Related]** Zhai et al. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML 2023. — arXiv:2303.06296
- **[Related]** Ding et al. *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.* ICML 2024. — arXiv:2402.13753
- **[Related]** Henry, Dachapally, Pawar, Chen. *Query-Key Normalization for Transformers.* Findings of EMNLP 2020.
- **[Frontier]** Nakanishi. *Scalable-Softmax Is Superior for Attention.* 2025. — arXiv:2501.19399

## 10. Worked Example

One head, one query. The target key beats every distractor by $\Delta = 4$ nats — a large, healthy margin at training length.

At $n = 4096$ keys:
$$m = \frac{e^{4}}{e^{4} + 4095} = \frac{54.6}{4149.6} = 0.0132$$

At $n = 131072$:
$$m = \frac{54.6}{54.6 + 131071} = 4.16\times10^{-4}$$

The ranking is still perfect — the right key wins every pairwise comparison — but its mass fell 32×, and the value vector it contributes is swamped by the sum of 131k near-uniform distractors. To hold $m = 0.5$ at 128k you need $\Delta' = \ln(n-1) = 11.78$ nats, i.e. a logit multiplier of $c = 11.78/4 = 2.95$.

Now the obstruction. YaRN at $s=32$ prescribes $1/t = (0.1\ln 32 + 1)^2 = 1.347^2 = 1.815$. That is **0.61× the value the dispersion argument demands**, and it was fit by perplexity sweep, not derived. Apply $c=1.815$: $\Delta' = 7.26$, $m = e^{7.26}/(e^{7.26}+131071) = 1424/132495 = 0.0107$ — still under 2% mass on the correct key.

So either (a) the dispersion argument is the wrong model of what breaks, (b) real logit gaps are far larger than 4 nats and only a mild correction is needed, or (c) YaRN's temperature is doing something other than restoring sharpness and the perplexity gain is incidental. The three are distinguishable only by measuring $\Delta$ on a task with known $j^\star$ — which is exactly the measurement nobody publishes. That is why the problem is empirically open rather than settled.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*