---
id: 15-mixture-of-experts/expert-choice-causal-leakage
title: "Causal Leakage in Expert-Choice Routing"
topic: 15-mixture-of-experts
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Causal Leakage in Expert-Choice Routing

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/expert-choice-causal-leakage` · **Status:** partially-solved

## 1. Problem Statement

Expert-choice (EC) routing inverts the usual assignment: instead of each token picking its top-$k$ experts, each expert picks its top-$k$ tokens from a pool. The pool is a batch of sequences, so the routing decision for token $t$ is a function of tokens at positions $>t$ and of unrelated sequences in the same batch. In a decoder-only language model trained with a causal mask, this is an information channel that bypasses the mask.

Three variants, with different difficulty:

- **Measurement.** Given a trained EC model, quantify how much of its training loss advantage comes from future-token information rather than from better load balance. Runnable today; the estimator is the contested part.
- **Method.** Train with EC's balancing benefits and decode autoregressively without an accuracy cliff. Partially solved — auxiliary causal predictors (Mixture-of-Depths) work at moderate scale, but the train/inference routing distributions are not matched by construction.
- **Theory.** Bound the leakage channel. Nothing rules out an EC model that is *behaviorally* causal despite a non-causal routing function; nothing proves one exists either.

A solution to the measurement variant is an estimator $\hat\Lambda$ in nats/token, with a control arm that isolates leakage from load balance, that ranks EC configurations consistently across two independent runs.

## 2. Formal Setting

Sequence $x_{1:T}$, layer $\ell$, hidden states $h_t \in \mathbb{R}^d$, $E$ experts, router $W_r \in \mathbb{R}^{d\times E}$. Affinity:

$$s_{t,e} = \left[\mathrm{softmax}(W_r^\top h_t)\right]_e .$$

**Token choice:** $A_t = \operatorname{top-}k_e(s_{t,\cdot})$ — a function of $h_t$ alone, hence of $x_{1:t}$ only.

**Expert choice:** with capacity $c$ per expert over a pool $\mathcal{P}$ of $N = |\mathcal{P}|$ token positions,

$$\mathcal{S}_e = \operatorname{top-}c_{\,t\in\mathcal{P}} \; s_{t,e}, \qquad a_{t,e} = \mathbb{1}[t \in \mathcal{S}_e], \qquad c = \left\lfloor \tfrac{N k}{E} \right\rfloor .$$

$a_{t,e}$ depends on every $s_{t',e}$, $t' \in \mathcal{P}$. If $\mathcal{P}$ spans the whole sequence, $a_{t,e}$ depends on $x_{t+1:T}$.

**Measured quantities.**

- *Flip rate.* $\delta = \frac{1}{NE}\sum_{t,e}\mathbb{1}[a_{t,e} \neq \tilde a_{t,e}]$, where $\tilde a$ is the assignment recomputed from the prefix only (prefix top-$c$, or the auxiliary predictor used at inference). Measured by running the router twice per batch; costs one extra router forward, negligible.
- *Leakage gap.* $\Lambda = \mathcal{L}_{\text{causal}} - \mathcal{L}_{\text{leaky}}$ in nats/token, where $\mathcal{L}_{\text{leaky}}$ evaluates the model with full-sequence EC routing and $\mathcal{L}_{\text{causal}}$ with prefix-only routing, same weights, same data.
- *Channel bound.* Routing per token per layer carries at most $E$ bits (the indicator vector $a_{t,\cdot}$), so over $L$ layers

$$I\big(a_{1:T,\cdot};\, x_{t+1:T} \mid x_{1:t}\big) \;\le\; L\,E\,\log 2 \quad \text{nats/token}.$$

- *Batch-composition sensitivity.* $\beta = \mathbb{E}\|p(\cdot\mid x_{1:t}, B) - p(\cdot\mid x_{1:t}, B')\|_1$ over two batches $B,B'$ containing the same sequence. Measured by re-running one sequence in different batch company.

**Assumptions, and which fail.** (i) *The pool is the training batch* — holds in training, fails at inference, where batch composition is a serving artifact. (ii) *$\Lambda$ isolates leakage* — violated: swapping the routing rule at eval also moves the model off its training distribution, so $\Lambda$ mixes leakage with distribution shift. (iii) *Softmax affinities are stable across the flip boundary* — violated near ties, where $s_{t,e}$ differences are $O(10^{-4})$ and the flip is numerically arbitrary. (iv) *Leakage is monotone in $\delta$* — no evidence; a flip on a low-information token costs nothing.

## 3. State of the Art

**Established.**

- EC routing itself: Zhou et al., *Mixture-of-Experts with Expert Choice Routing*, NeurIPS 2022. Perfect load balance by construction, no auxiliary balancing loss, variable experts per token. The headline results were measured on **encoder-decoder T5/GLaM-style models with a masked/span-corruption objective**, where bidirectional context is legitimate — so those numbers say nothing about the causal case. The paper states EC is not directly applicable to autoregressive decoding.
- The batch-level analogue in vision: Riquelme et al., *Scaling Vision with Sparse Mixture of Experts*, NeurIPS 2021 — Batch Prioritized Routing. Non-causal domain, so no leakage question arises.
- Assignment-style global routing: Lewis et al., *BASE Layers*, ICML 2021 — balanced assignment via auction; the same batch-global dependence, and the paper uses a token-choice fallback at inference.

**Partial method-level fix, claimed with limited ablation.**

- Raposo et al., *Mixture-of-Depths*, 2024 (arXiv:2404.02258). Uses per-sequence top-$k$ (an EC-style rule) and names the causality problem explicitly. Two remedies: a router auxiliary binary-cross-entropy loss predicting top-$k$ membership, and a small auxiliary MLP predictor consuming only the prefix. They report near-baseline performance with the predictor. What is *not* ablated: the flip rate $\delta$ as a function of depth and scale, and whether the residual gap grows with sequence length.
- Zhong et al., *Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training*, 2024 (arXiv:2405.03133). Introduces **causal segment routing**: expert weights for segment $i$ are computed from segment $i-1$, making the router causal at segment granularity. Established as a construction; the residual within-segment leakage is by design and not quantified as a channel.

**Benchmark-number-only.** Most reported EC-vs-token-choice comparisons in autoregressive settings appear as a single validation-perplexity delta with no causal-eval arm. Treat them as uninterpretable for this problem.

**Not a fix, often confused with one.** Wang et al., *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*, 2024 (arXiv:2408.15664), used in DeepSeek-V3: per-expert bias updated from *previous-step* statistics. Token choice at every step, so causal — it solves balance, not leakage.

## 4. What Is Known

- EC gives exact load balance at capacity $c$ with zero dropped tokens inside the pool, by construction (Zhou et al. 2022).
- On the MLM objective, EC reached a fixed training perplexity roughly $2\times$ faster in step count than top-1/top-2 gating in an 8B-parameter, 64-expert model trained on ~100B tokens (Zhou et al. 2022). Scale: 8B/64E. This number is from a non-causal objective.
- The routing channel is large: for $L=32$, $E=64$, the bound is $32\times64\times\ln 2 \approx 1420$ nats/token, against an English text entropy of roughly $0.7$–$1.0$ nats/token. The bound is therefore vacuous — it permits total leakage.
- The MoD line establishes that a prefix-only predictor can approximate a sequence-global top-$k$ decision well enough to preserve downstream quality at ~1B scale (Raposo et al. 2024). It does not establish $\delta$ or its scaling.
- Segment-level causal routing trains stably and matches token-choice MoE quality at ~0.3–1.5B parameters (Lory, 2024).

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed estimator of $\Lambda$ that separates leakage from the distribution shift induced by swapping the routing rule at evaluation time. Every published comparison changes two things at once. Until a matched-routing control exists, "how much does EC cheat?" has no defined answer.
- **Empirically open.** The scaling of $\delta$ and $\Lambda$ with $T$, $E$, $L$, and parameter count. Runnable at 1B/100B-token scale for well under 10k accelerator-hours; nobody has published the sweep with a causal-eval arm.
- **Theoretically open.** Whether there exists a batch-global balanced assignment rule that is *provably* prefix-measurable and retains EC's zero-drop guarantee. Perfect balance requires knowing the pool; a causal rule cannot. No impossibility theorem exists, and no construction achieves both.

## 6. Why It Is Hard

**Confounded measurement, plus non-identifiability.** Two mechanisms produce the same observable — a lower training loss under EC:

1. leakage (routing encodes the future),
2. better optimization (no token dropping, no auxiliary-loss gradient noise).

Both are removed by the same intervention (making routing causal), so the natural ablation cannot separate them. There is no held-out condition in which leakage is present and balance is not. The information-theoretic bound does not help: at $\approx 1420$ nats/token it is three orders of magnitude above the quantity of interest, so it is consistent with any measurement. And there is no ground truth — you cannot label which tokens' predictions were "helped by the future", because the router's influence is diffuse across the whole forward pass.

## 7. Current Research (as of 2026)

- **Causal surrogates for global assignment.** Auxiliary predictors (Google DeepMind, MoD line) and segment-causal routing (Princeton NLP / Meta AI, Lory). Both trade exactness for prefix-measurability.
- **Aux-loss-free balancing under token choice** (DeepSeek). Sidesteps the problem entirely and is now the dominant production choice; this is the main reason EC leakage remains under-measured — the field routed around it. *(frontier — verify: whether any frontier decoder-only model ships EC routing at all.)*
- **Batch-invariant inference.** Independent of MoE, the determinism work on batch-size-dependent kernels supplies exactly the $\beta$ metric this problem needs. *(frontier — verify.)*
- **EC in non-causal towers.** Encoders, vision, and retrieval towers still use EC/BPR freely, correctly.

## 8. Concrete Next Experiment

**Scale.** Decoder-only MoE, 1.3B active parameters, 64 experts, 24 layers, $T = 2048$, 100B tokens of a public corpus. About 4k H100-hours per arm; four arms.

**Arms.**
- **A (leaky):** EC routing, pool = full sequence, $k=2$, both training and evaluation.
- **B (causal-eval of A):** arm A's weights, evaluated with prefix top-$c$ routing.
- **C (control — balance without leakage):** token choice, top-2, with the aux-loss-free bias balancer. Matches A's load balance, has no future access.
- **D (predictor):** EC training with an MoD-style prefix-only auxiliary predictor, used at both train-time-eval and inference.

**The deciding number.** $\Lambda = \mathcal{L}_{B} - \mathcal{L}_{A}$, validation NLL in nats/token, reported with a 95% bootstrap CI over documents.

- $\Lambda < 0.01$ nats/token → leakage is negligible; EC's advantage is balance, and arm A vs C settles it directly.
- $\Lambda > 0.05$ nats/token → EC training loss is not comparable to token-choice training loss, and every published EC-vs-TC autoregressive perplexity table is invalid.

**Secondary readouts.** $\delta$ per layer (expect it to rise in later layers, where affinities sharpen); $\Lambda$ at $T \in \{512, 2048, 8192\}$ — if $\Lambda$ grows with $T$, leakage scales with context and the problem worsens on long-context models.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational / SOTA for EC]** Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[SOTA — causal surrogate]** Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro. *Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models.* 2024. — arXiv:2404.02258
- **[SOTA — causal segment routing]** Zhong, Xia, Chen, Lewis. *Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training.* 2024. — arXiv:2405.03133
- **[SOTA — balance without leakage]** Wang, Chen, Xie, Zhao, Dai, et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* 2024. — arXiv:2408.15664
- **[Related]** Lewis, Bhosale, Dettmers, Goyal, Zettlemoyer. *BASE Layers: Simplifying Training of Large, Sparse Models.* ICML, 2021. — arXiv:2103.16716
- **[Related]** Riquelme, Puigcerver, Mustafa, Neumann, Jenatton, Susano Pinto, Keysers, Houlsby. *Scaling Vision with Sparse Mixture of Experts.* NeurIPS, 2021. — arXiv:2106.05974
- **[Survey]** Fedus, Dean, Zoph. *A Review of Sparse Expert Models in Deep Learning.* 2022. — arXiv:2209.01667
- **[Scaling]** Clark, de las Casas, Guy, Mensch, et al. *Unified Scaling Laws for Routed Language Models.* ICML, 2022. — arXiv:2202.01169

## 10. Worked Example

One expert $e$, one layer, $T=8$, capacity $c=4$. Affinities:

| $t$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $s_{t,e}$ | 0.81 | 0.22 | 0.55 | 0.19 | 0.60 | 0.91 | 0.14 | 0.58 |

Full-sequence top-4: $\{6, 1, 5, 8\}$ (0.91, 0.81, 0.60, 0.58). Token 3 ($s=0.55$) is **excluded** — because token 8, which comes *after* it, has 0.58.

Prefix top-4 at $t=5$ (candidates $1..5$, take up to 4): $\{1, 5, 3, 2\}$. Token 3 is **included**.

Flip: $a_{3,e} = 0$ under training routing, $\tilde a_{3,e} = 1$ under causal routing. Here $\delta = 3/8 = 0.375$ for this expert. Token 3's residual stream differs between the two regimes by the expert's output, so its prediction of token 4 differs — and the difference is caused by $s_{8,e}$, a quantity that depends on $x_8$.

Now make the obstruction visible. Suppose the measured gap on a real 1.3B run is $\Lambda = 0.004$ nats/token with $\delta = 0.31$.

- Channel bound with $L=24$, $E=64$: $24 \times 64 \times \ln 2 = 1065$ nats/token. Ratio measured/bound $= 3.8 \times 10^{-6}$. The theory says nothing.
- Is $0.004$ leakage or distribution shift? Run the same swap on arm **C** (token choice), where routing is already prefix-measurable and the swap is a no-op: gap is exactly $0$. So arm C gives no shift baseline at all — the control does not exist, because there is no way to perturb a causal router by "the same amount" without also removing causality.
- The margin at the flip boundary is $0.58 - 0.55 = 0.03$ here; in a trained 64-expert router, median margins at rank $c$ are typically $O(10^{-3})$, so a bf16 rounding change flips assignments. $\delta$ is therefore partly numerical noise, and $\delta$ cannot be used as a proxy for $\Lambda$.

That is the whole difficulty in one table: a huge permissive bound, a tiny observed number, and no control arm that moves one mechanism without moving the other.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*