---
id: 08-loss-and-heads/multi-token-prediction-head-design
title: "Multi-Token Prediction Head Design and Depth"
topic: 08-loss-and-heads
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multi-Token Prediction Head Design and Depth

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/multi-token-prediction-head-design` · **Status:** empirically-open

## 1. Problem Statement

A multi-token prediction (MTP) model attaches $n$ output heads to a shared trunk and trains head $i$ to predict token $t+i$ from the trunk state at position $t$. Three knobs define the design: the **horizon** $n$, the **depth** $d$ (parameters and layers per head), and the **coupling** (heads independent and parallel, or chained so head $i$ conditions on head $i-1$'s output).

The problem: given a fixed training FLOP budget and a fixed inference latency target, choose $(n, d, \text{coupling})$. Solving it means a rule that predicts, before training, which configuration maximises either (a) the base model's own quality after MTP pretraining, or (b) the draft acceptance rate under self-speculative decoding.

Three variants, different difficulty:

- **Measurement.** Does the MTP auxiliary loss improve the *next-token* model, or only supply a cheap draft head? These are separate goods measured by the same training run and routinely conflated.
- **Method.** Find $(n, d)$ empirically. Runnable today; blocked by cost, since the effect reverses with model scale.
- **Theory.** Explain *why* horizon-$n$ supervision changes the trunk representation at all, given that teacher-forced next-token training is already a consistent estimator of the full sequence distribution.

## 2. Formal Setting

Sequence $x_{1:T}$ over vocabulary $\mathcal{V}$. Trunk $f_\theta$ maps a prefix to a latent $z_t = f_\theta(x_{1:t}) \in \mathbb{R}^{h}$. Head $i \in \{1,\dots,n\}$ is a map $g_{\phi_i}: \mathbb{R}^h \to \Delta(\mathcal{V})$. The loss is

$$\mathcal{L}(\theta,\phi) = -\sum_{t=1}^{T}\sum_{i=1}^{n} w_i \log P_{\phi_i}\big(x_{t+i} \mid z_t\big),$$

with weights $w_i$ (uniform in Gloeckle et al.; DeepSeek-V3 scales the $i>1$ terms by a constant $\lambda$).

**Measured quantities.**

- **Depth** $d$ = transformer (or residual MLP) blocks inside $g_{\phi_i}$ before the unembedding. Report also head parameter count excluding the tied unembedding matrix $W_U \in \mathbb{R}^{|\mathcal{V}|\times h}$, which dominates: at $|\mathcal{V}|=128\text{k}, h=4096$, $W_U$ is 524M parameters, so *tying* $W_U$ across heads is what makes $n=4$ affordable at all.
- **Factorization gap.** Parallel heads model $\prod_i P_{\phi_i}(x_{t+i}\mid z_t)$, a mean-field approximation of the true joint $P(x_{t+1:t+n}\mid x_{1:t})$. The error is the total correlation
  $$\mathrm{TC} = \mathbb{E}\Big[\mathrm{KL}\big(P(x_{t+1:t+n}\mid x_{1:t}) \,\big\|\, \textstyle\prod_i P(x_{t+i}\mid x_{1:t})\big)\Big],$$
  measurable as (sum of per-head cross-entropies) minus (chain-rule cross-entropy of the same model on the same span), in nats/token.
- **Acceptance rate** $\alpha_i$ = fraction of head-$i$ proposals accepted by exact speculative verification (Leviathan et al. 2023). Expected accepted length $\mathbb{E}[\ell] = 1+\sum_{i=1}^{n}\prod_{j\le i}\alpha_j$ under the chained scheme; wall-clock speedup is $\mathbb{E}[\ell]$ divided by the per-step cost ratio, *not* $\mathbb{E}[\ell]$ itself.
- **Compute matching.** Compare at equal training FLOPs *and* equal serving KV-cache footprint, not equal steps.

**Assumptions known to be violated.** (i) Head independence — adjacent tokens in text are strongly dependent; $\mathrm{TC}>0$ always. (ii) That $z_t$ carries enough information for token $t+n$ — under teacher forcing $z_t$ is trained on a single-step objective and provably need not (Bachmann & Nagarajan 2024). (iii) Stationarity in $i$ — head difficulty grows sharply with $i$, so uniform $w_i$ is not a neutral choice. (iv) Scale invariance — the sign of the MTP effect on base-model quality flips with parameter count.

## 3. State of the Art

**Established (ablated, multiple seeds or scales).**

- **Gloeckle et al., ICML 2024** ("Better & Faster Large Language Models via Multi-token Prediction"): $n=4$ parallel depth-1 transformer heads on a shared trunk, with sequential forward/backward per head to cap activation memory at one head's worth. Ablates $n \in \{1,2,4,6,8\}$ and model size 0.3B–13B on code. Result: benefit is **scale-dependent** — MTP is neutral-to-harmful below ~1B and clearly positive at 6.7B/13B.
- **Speculative decoding correctness** (Leviathan et al. ICML 2023; Chen et al. 2023): the verification step is distribution-preserving regardless of draft head quality. So any head design is *safe*; only speed varies. This removes quality from the draft-head design problem entirely.

**Claimed but not cleanly ablated.**

- **DeepSeek-V3 (2024)** uses a single *sequential* MTP module ($n=1$ extra, $d=1$ transformer block that consumes the previous token's embedding plus the trunk state) and attributes both quality gains and a 1.8× decoding speedup to it. There is no matched-FLOP arm without MTP at 671B, so the quality claim rests on smaller ablation models.
- **Medusa (Cai et al. 2024)**: $d=1$ residual-MLP heads, tree attention over candidate continuations; 2.2–2.8× speedups reported. Head depth is not swept.
- **EAGLE / EAGLE-2 / EAGLE-3 (Li et al. 2024, 2025)**: drafting on the *feature* sequence with a one-layer autoregressive head; large speedups on Spec-Bench. This is benchmark-number SOTA for drafting, and the ablations are about feature-vs-token input, not about $d$ or $n$ as a design axis.

**Prior art.** Blockwise parallel decoding (Stern et al., NeurIPS 2018) is the original $k$-head design; ProphetNet (Qi et al., EMNLP Findings 2020) introduced $n$-stream self-attention for future-$n$-gram prediction.

## 4. What Is Known

- **Scale reversal, measured at 0.3B–13B on code (Gloeckle et al. 2024).** With $n=4$ and 13B parameters, MTP pretraining improves pass@1 by roughly **+12% relative on MBPP and +17% relative on HumanEval** over a matched next-token baseline. At sub-billion scale the same recipe is neutral or worse.
- **Optimal horizon is small and dataset-dependent.** $n=4$ was best for code at 7B-scale byte-pair tokenization in the same ablation; $n=8$ was best for byte-level models. Beyond the optimum, quality degrades — the trunk spends capacity on unpredictable far tokens.
- **Self-speculative decoding from the trained heads gives up to ~3× on code** in the same paper, with no separate draft model.
- **Acceptance decays fast with $i$.** DeepSeek-V3 reports second-token acceptance of about **85–90%**; the third and beyond are not trained. Medusa/EAGLE trees exist precisely because per-position acceptance falls off, so breadth substitutes for depth.
- **Teacher forcing has a proven failure mode.** Bachmann & Nagarajan (ICML 2024) exhibit a path-star graph family where next-token teacher-forced training fails to learn a task that is trivially learnable, and show that predicting further ahead fixes it. This is an existence proof that horizon matters, not a statement about natural language.
- **Register-token variants work without extra heads.** MuToR (Gerontopoulos et al. 2025) interleaves learnable register tokens that predict $t+i$, adding negligible parameters and staying compatible with fine-tuning of an existing next-token model.

## 5. What Is Not Known

- **Empirically open.** The $(n,d)$ grid has never been swept at matched FLOPs above ~13B on general (non-code) pretraining data. Nobody has published a clean $d\in\{1,2,4\}$ sweep at fixed $n$; $d=1$ is a convention inherited from Stern et al., not a measured optimum.
- **Empirically open.** Whether parallel-independent or chained-sequential coupling wins at fixed head FLOPs. DeepSeek chose chained, Meta chose parallel, and no paper runs both in one controlled setting.
- **Theoretically open.** No characterization of which data distributions make horizon-$n$ supervision improve the trunk. The total correlation $\mathrm{TC}$ is the obvious candidate order parameter and has not been related to the observed gain by any bound.
- **Theoretically open.** Why the effect reverses with scale. The "small models lack capacity for the auxiliary task" story is folklore with no model.
- **Methodologically blocked.** There is no agreed metric separating *trunk improvement* from *head capacity*. Deleting the extra heads at inference (the standard protocol) confounds the two: any gain could come from the auxiliary gradient reshaping $z_t$, or from the extra heads acting as a regularizer on $W_U$'s gradient.

## 6. Why It Is Hard

**The primary obstruction is a sign flip located between the affordable scale and the interesting scale.** The effect is negative or null at $\le$1B, where a full $(n,d)$ grid costs a few thousand GPU-hours, and positive at $\ge$7B, where a single arm costs $10^5$ GPU-hours. So the cheap experiment answers the wrong question, and no scaling law exists to extrapolate across the reversal — the quantity being extrapolated changes sign, which is exactly where power-law fits are useless.

**Secondary: confounded measurement.** The two goods — base-model quality and draft acceptance — trade off. A deeper head raises $\alpha_i$ (better draft) while absorbing gradient that would otherwise pressure the trunk (worse base model). A paper reporting "MTP helps" without stating which good it measured has reported nothing decidable.

## 7. Current Research (as of 2026)

- **Sequential-module MTP in frontier open-weight models.** DeepSeek's $d=1$ chained module is now the default in several open MoE releases; the horizon stays at 1–2 extra tokens. *(frontier — verify: whether any 2026 release has shipped $n>2$ in production.)*
- **Draft-head architecture search.** EAGLE-3 and successors move drafting to multi-layer feature fusion and train the draft on the target's own outputs. SGLang/vLLM ship these as first-class serving features.
- **Lightweight retrofits.** MuToR-style register tokens and other adapter-only MTP, aimed at adding the objective to an already-pretrained model without re-pretraining.
- **Diffusion / any-order language models** as the alternative answer to the same question: predict a set rather than a chain. Relevant because they make the factorization gap explicit rather than implicit.
- **Theory of teacher forcing** following Bachmann & Nagarajan — small but active, mostly on synthetic graph tasks.

## 8. Concrete Next Experiment

**Question:** at fixed head FLOPs, does head *depth* or head *count* buy more?

- **Scale.** 7B dense transformer, 300B tokens, $|\mathcal{V}|=128$k, tied unembedding. About four arms × ~2×10$^{22}$ FLOPs; feasible on 256 H100s in under two weeks.
- **Arms**, all matched to within 2% of total training FLOPs by trimming tokens:
  1. **Control:** $n=1$, standard next-token.
  2. $n=4$, $d=1$ parallel heads (the Gloeckle configuration).
  3. $n=2$, $d=2$ parallel heads — same head FLOPs as arm 2, depth traded for horizon.
  4. $n=4$, $d=1$ **chained** heads (DeepSeek coupling), same head FLOPs as arm 2.
- **Measurements.** Next-token validation loss with all extra heads deleted; pass@1 on HumanEval/MBPP; $\alpha_1,\dots,\alpha_4$ under exact speculative verification; and $\mathrm{TC}$ in nats/token on held-out text.
- **The deciding number:** **next-token validation loss with the auxiliary heads deleted, arm 3 minus arm 2, in nats/token.** A gap $>0.005$ nats (roughly 3× the seed-to-seed noise at this scale) in either direction settles depth-versus-horizon at 7B. If $|\Delta| < 0.005$ while $\alpha_2$ differs by more than 5 points across arms, the conclusion is that head design is a *decoding-speed* knob only and does not touch the trunk — which is itself the answer to the measurement variant in Section 1.

## 9. Key References

- **[Foundational]** Mitchell Stern, Noam Shazeer, Jakob Uszkoreit. *Blockwise Parallel Decoding for Deep Autoregressive Models.* NeurIPS 2018. — arXiv:1811.03115
- **[Foundational]** Weizhen Qi, Yu Yan, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang, Ming Zhou. *ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training.* Findings of EMNLP 2020. — arXiv:2001.04063
- **[SOTA]** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction.* ICML 2024. — arXiv:2404.19737
- **[SOTA]** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. — arXiv:2401.10774
- **[SOTA]** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML 2024. — arXiv:2401.15077
- **[Foundational]** Yaniv Leviathan, Matan Kalman, Yossi Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. — arXiv:2211.17192
- **[Theory]** Gregor Bachmann, Vaishnavh Nagarajan. *The Pitfalls of Next-Token Prediction.* ICML 2024. — arXiv:2403.06963
- **[Systems]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Recent]** Anastasios Gerontopoulos, Spyros Gidaris, Nikos Komodakis. *Multi-Token Prediction Needs Registers.* NeurIPS 2025. — arXiv:2505.10518

## 10. Worked Example

Take a 7B model, $h=4096$, $|\mathcal{V}|=128$k, $d=1$ transformer head ($\approx 4h^2 + 8h^2 = 201$M parameters per head, unembedding tied and excluded).

**Head cost.** $n=4$ adds $3\times201\text{M}=604$M trainable parameters, +8.6% over the trunk, but only during training — heads are dropped at inference for the base model. Training FLOP overhead is about **+9%**, so a matched-FLOP control must be given 9% more tokens.

**The obstruction, made numeric.** Suppose arm 2 ($n=4,d=1$) reaches next-token loss 1.7400 nats and the control reaches 1.7460 — a 0.006-nat win, consistent with the reported code gains. Chinchilla-style scaling says loss falls roughly as $C^{-0.05}$ in training compute near this regime, so 0.006 nats off a 1.746 baseline is worth about
$$\Delta \log C \approx \frac{0.006/1.746}{0.05} \approx 0.069 \Rightarrow \text{about } 7\%\ \text{more compute}.$$
The MTP arm spent **9%** more compute on its heads. The measured "win" is inside the cost of producing it. Whether MTP is a genuine objective improvement or an expensive way to buy 7% of a scaling curve therefore depends on the *third* decimal place of validation loss — and seed-to-seed variance at 7B/300B tokens is itself around 0.002–0.003 nats.

That is the whole difficulty in one calculation: the effect size and the measurement noise and the accounting overhead are all the same order of magnitude, so the question cannot be settled by a single run at any scale where it is affordable, and requires multi-seed matched-FLOP arms at $\ge$7B — which nobody has published.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*