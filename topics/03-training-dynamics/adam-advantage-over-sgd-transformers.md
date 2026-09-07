---
id: 03-training-dynamics/adam-advantage-over-sgd-transformers
title: "Adam's Advantage over SGD on Transformers"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adam's Advantage over SGD on Transformers

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/adam-advantage-over-sgd-transformers` · **Status:** open

## 1. Problem Statement

On convolutional image classifiers, well-tuned SGD with momentum matches or beats Adam. On transformer language models, it does not: the gap is large, reproducible, and does not close with tuning. The problem is to explain **why**, precisely enough that the explanation predicts new behaviour.

Three variants, of different difficulty:

- **Measurement.** Define the gap so it is not an artifact of tuning budget, schedule, batch size, or architecture-specific defaults (LayerNorm placement, weight decay coupling, gradient clipping). Output: a gap statistic $\Delta$ and a protocol under which it is stable.
- **Method.** Find the *minimal* modification to SGD that closes $\Delta$ on a transformer. A one-line change that closes it identifies the responsible mechanism; a change that closes it only with the full second-moment estimator does not.
- **Theory.** Give an assumption set, satisfied by transformer training in practice, under which Adam's iteration complexity provably beats SGD's by the observed factor — not merely a worst-case separation on a constructed problem.

Solving it means: a mechanism $M$, a measurable proxy $\hat{M}$, and a demonstration that intervening on $\hat{M}$ moves $\Delta$ monotonically, in both directions, at $\ge 10^9$ parameters.

## 2. Formal Setting

Model $f_\theta$, $\theta \in \mathbb{R}^d$, autoregressive loss over a token stream $\mathcal{D}$:
$$\mathcal{L}(\theta) = -\mathbb{E}_{x\sim\mathcal{D}}\Big[\tfrac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})\Big].$$

**SGD-M:** $m_k = \beta m_{k-1} + g_k$, $\theta_{k+1} = \theta_k - \eta_k m_k$.
**Adam:** with $m_k, v_k$ the bias-corrected first/second moments of $g_k$,
$$\theta_{k+1} = \theta_k - \eta_k \frac{m_k}{\sqrt{v_k}+\epsilon}.$$

**The gap, as measured.** Fix a compute budget $C$ (FLOPs, not steps — the optimizers differ in per-step cost by $<3\%$ but tuning cost differs a lot). Sweep each optimizer over learning rate, warmup, $\beta$'s, weight decay, and clipping threshold with equal search budget $B$ trials. Then
$$\Delta(C, B) = \min_{\text{SGD hp}} \mathcal{L}_{\text{val}}(C) \;-\; \min_{\text{Adam hp}} \mathcal{L}_{\text{val}}(C),$$
reported in nats/token. An equivalent and more decision-relevant form is the **compute ratio** $\rho = C_{\text{SGD}}/C_{\text{Adam}}$ at matched final loss.

**Candidate mechanisms, each with a measurable proxy.**

- *Gradient noise tail index* $\alpha$: fit a stable law to per-example gradient norms; $\alpha < 2$ means infinite variance. Measured by resampling minibatches at fixed $\theta$.
- *Directional sharpness*: $S(u) = u^\top \nabla^2 \mathcal{L}(\theta)\, u / \|u\|^2$ evaluated at the SGD update direction versus the Adam direction, via Hessian-vector products.
- *Hessian block heterogeneity*: partition $\theta$ into $L$ blocks (embeddings, per-layer $W_Q,W_K,W_V,W_O$, MLP, LayerNorm). Let $\lambda_{\max}^{(l)}$ be the top eigenvalue of block $l$'s diagonal Hessian block. Heterogeneity $= \max_l \lambda_{\max}^{(l)} / \min_l \lambda_{\max}^{(l)}$, estimated by block Lanczos.
- *Class imbalance*: token frequency distribution is Zipfian; track loss on the frequency decile $D_j$ separately, $\mathcal{L}_j$.

**Assumptions and their status.** Smoothness with a global $L$: *violated* — transformer training runs at the edge of stability with $\lambda_{\max}$ tracking $2/\eta$. Bounded gradient variance: *violated* under heavy-tailed noise ($\alpha<2$). Bounded stochastic gradients (assumed in most Adam proofs): *violated* by loss spikes. i.i.d. sampling: approximately held. Equal tuning quality across arms: the assumption that fails most often in published comparisons, and the one that decides whether $\Delta$ is real.

## 3. State of the Art

**Established (with ablations).**
- Wilson et al. (NeurIPS 2017) showed adaptive methods generalize *worse* than SGD on vision and small NLP models — the result that made the transformer reversal noteworthy.
- Zhang et al. (NeurIPS 2020) measured heavy-tailed gradient noise in attention models and showed clipping-based SGD variants recover much of the gap — the first mechanistic candidate with an ablation.
- Kunstner et al. (ICLR 2023) ran the decisive control: **full-batch** GD versus **full-batch** Adam on transformers. The gap persists without any minibatch noise. This falsifies noise-heavy-tail as the *primary* cause.
- Kunstner et al. (NeurIPS 2024) attributes the gap to heavy-tailed class imbalance: under GD, loss on low-frequency tokens decreases very slowly; Adam's per-coordinate normalization removes the frequency dependence. Reproduced on both language models and deliberately imbalanced vision tasks.
- Zhang et al. (NeurIPS 2024, "Why Transformers Need Adam: A Hessian Perspective") shows blockwise Hessian spectra are far more heterogeneous in transformers than in CNNs, and that a *blockwise* learning rate on SGD recovers a large part of the gap.

**Claimed but unablated / benchmark-only.**
- Second-order and sign-based successors (Sophia, Lion, Muon, Shampoo variants) report $1.3$–$2\times$ speedups over AdamW. Most such numbers are single-scale, single-data-mixture, and use a baseline tuned less aggressively than the proposed method. Independent replications at $\ge 1$B parameters with matched sweeps typically shrink the advantage to $\le 1.1$–$1.4\times$.
- "Adam works because it approximates natural gradient / diagonal Newton" is asserted widely and ablated rarely; the diagonal Fisher preconditioner is not the same object as $\sqrt{v_k}$ under momentum.

**Theory SOTA.** Worst-case separations exist in both directions (SGD beats Adam on constructed convex problems; Adam beats SGD under coordinate-wise scale heterogeneity), but no theorem takes transformer-realistic assumptions and yields the measured $\rho$.

## 4. What Is Known

- **The gap is real at scale.** On GPT-2-class decoders (125M–355M) trained on OpenWebText/C4, tuned SGD-M sits roughly $0.3$–$0.5$ nats/token above AdamW at equal steps; $\rho$ is commonly reported in the range $2$–$5\times$ and does not shrink with model size in the 100M–1B band.
- **It is not minibatch noise.** Full-batch Adam still beats full-batch GD on transformers (Kunstner et al., ICLR 2023), at ~100M-parameter scale.
- **It is architecture-conditioned, not task-conditioned.** Transformers trained on vision data still favour Adam; CNNs trained on text-like imbalanced labels start to favour Adam once imbalance is introduced (Kunstner et al., 2024). The controlling variable travels with the *loss/label distribution and the Hessian structure*, not the modality.
- **Sign is most of Adam.** signSGD with momentum recovers a large fraction of Adam's advantage on transformers, indicating the second moment acts mainly as a magnitude equalizer rather than a variance estimate.
- **Embeddings and LayerNorm are the sensitive blocks.** Applying Adam only to embedding + normalization parameters and SGD elsewhere recovers much of the gap; the reverse split does not.
- **Edge of stability holds for adaptive methods too.** Adam operates with preconditioned sharpness pinned near its stability threshold rather than in a descent-lemma regime (Cohen et al., 2022–2024), so classical $L$-smooth analyses do not describe either arm.

## 5. What Is Not Known

- **Theoretically open.** No theorem derives $\rho \approx 2$–$5$ from measurable properties of transformer loss landscapes. Existing Adam convergence results give rates no better than SGD's under assumptions that hold; results that give better rates assume conditions (bounded gradients, fixed $\epsilon$-dominated preconditioner) known to be violated.
- **Theoretically open.** Whether class imbalance and Hessian block heterogeneity are two views of one mechanism or two additive mechanisms. No decomposition theorem exists.
- **Empirically open.** Whether $\Delta$ persists at $\ge 10$B parameters and $\ge 200$B tokens with matched sweeps. Nobody has paid for a properly tuned SGD arm at that scale; the belief that it persists is extrapolation from $\le 1$B.
- **Empirically open.** Whether the minimal fix is per-block learning rates ($L \approx 100$ scalars) rather than per-coordinate ($d \approx 10^9$). Adam-mini-style results suggest yes; the ablation isolating *how coarse* the partition can be before $\Delta$ reopens has not been run at scale.
- **Methodologically blocked.** "Equal tuning budget" has no accepted definition. SGD and Adam have different hyperparameter dimensionality and different sensitivity, so any fixed $B$ favours one. Until $\Delta$ is defined against a tuning protocol both communities accept, published gaps are not comparable across papers.

## 6. Why It Is Hard

The primary obstruction is **confounded measurement compounded by non-identifiability**. Every proposed mechanism — heavy-tailed noise, class imbalance, block heterogeneity, sign-vs-magnitude, edge-of-stability dynamics — is *correlated with every other one in real transformer training*, because they are all downstream of the same Zipfian token distribution and the same architecture. Intervening on one moves the others. There is no known knob that changes block heterogeneity while holding token-frequency skew fixed.

Second: **cost of the control arm.** The decisive experiment requires a well-tuned SGD run at frontier scale, which is a large spend on a run that is expected to be worse. No lab has commercial reason to fund it, so the empirical question stays open for economic rather than scientific reasons.

Third: **the evaluation does not measure what it names.** "Adam is better" is a statement about a tuned minimum over hyperparameters, but tuning quality is the least-controlled variable in the field. A reported $2\times$ gap may be a $2\times$ gap in *tuning effort*.

## 7. Current Research (as of 2026)

- **Class-imbalance line** (Kunstner, Schmidt, and collaborators, UBC): extending the low-frequency-class account from token classification to general heavy-tailed objectives.
- **Hessian-structure line** (Zhang, Sun, Luo and collaborators): block-diagonal preconditioners, Adam-mini-style memory reduction as a mechanism probe — if $L$ scalars suffice, the second moment is not doing variance estimation.
- **Optimizer benchmarking with matched sweeps**: several 2025 efforts re-ran the recent optimizer zoo against carefully tuned AdamW across scales and found claimed speedups shrink substantially as scale grows *(frontier — verify specific numbers against the papers)*.
- **Muon / spectral-norm-constrained updates** (open-source LLM training community, and follow-on academic analysis): reframes the question as "what norm should the update be measured in", which subsumes sign-descent as the $\ell_\infty$ case *(frontier — verify)*.
- **Simplified proxies**: linear-attention and single-layer models where Adam's advantage still appears and the Hessian is tractable (Ahn et al., ICLR 2024).

## 8. Concrete Next Experiment

**Question decided:** is the mechanism *coordinate-level* or *block-level*?

**Scale.** 1.3B-parameter decoder-only transformer, 30B tokens (~Chinchilla-optimal), fixed data order and seed across arms, 3 seeds per arm.

**Arms.**
1. AdamW (reference).
2. SGD-M — **control arm**, swept with the *same* trial budget as Adam ($B=32$ trials, identical search algorithm, per-arm search space of equal dimension) at a 1/8-scale proxy model, then transferred with $\mu$P-style scaling rules.
3. SGD-M + **per-block** learning rates, one scalar per parameter tensor ($L \approx 150$), set as $\eta_l \propto 1/\sqrt{\hat{\lambda}_{\max}^{(l)}}$ with $\hat{\lambda}^{(l)}_{\max}$ re-estimated by block Lanczos every 500 steps.
4. SGD-M + per-block LR + per-coordinate normalization **only on embedding and output-projection** tensors.

**Deciding number.** The recovery fraction
$$R = \frac{\mathcal{L}_{\text{SGD}} - \mathcal{L}_{\text{arm}}}{\mathcal{L}_{\text{SGD}} - \mathcal{L}_{\text{Adam}}}$$
at 30B tokens. If arm 3 gives $R \ge 0.9$, the mechanism is block-level curvature heterogeneity and per-coordinate adaptivity is incidental — the practical consequence is optimizer state drops from $2d$ to $d + L$. If $R \le 0.5$ for arm 3 but $\ge 0.9$ for arm 4, the mechanism is localized to the token-frequency-coupled tensors, favouring the class-imbalance account. Both $R$ values below $0.5$ falsifies both current accounts at this scale. Estimated cost: ~4 arms × 3 seeds × ~$2.3\times10^{21}$ FLOPs ≈ 2–3 weeks on 64 H100s including the proxy sweep.

## 9. Key References

- **[Foundational]** Diederik P. Kingma, Jimmy Ba. *Adam: A Method for Stochastic Optimization.* ICLR, 2015. — arXiv:1412.6980
- **[Foundational]** Ashia C. Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, Benjamin Recht. *The Marginal Value of Adaptive Gradient Methods in Machine Learning.* NeurIPS, 2017. — arXiv:1705.08292
- **[Foundational]** Sashank J. Reddi, Satyen Kale, Sanjiv Kumar. *On the Convergence of Adam and Beyond.* ICLR, 2018.
- **[SOTA]** Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, Suvrit Sra. *Why are Adaptive Methods Good for Attention Models?* NeurIPS, 2020.
- **[SOTA]** Frederik Kunstner, Jacques Chen, Jonathan Wilder Lavington, Mark Schmidt. *Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, But Sign Descent Might Be.* ICLR, 2023. — arXiv:2304.13960
- **[SOTA]** Frederik Kunstner, Robin Yadav, Alan Milligan, Mark Schmidt, Alberto Bietti. *Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models.* NeurIPS, 2024. — arXiv:2402.19449
- **[SOTA]** Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, Zhi-Quan Luo. *Why Transformers Need Adam: A Hessian Perspective.* NeurIPS, 2024. — arXiv:2402.16788
- **[SOTA]** Xiangning Chen, Chen Liang, Da Huang, et al. *Symbolic Discovery of Optimization Algorithms* (Lion). NeurIPS, 2023. — arXiv:2302.06675
- **[SOTA]** Hong Liu, Zhiyuan Li, David Hall, Percy Liang, Tengyu Ma. *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training.* ICLR, 2024. — arXiv:2305.14342
- **[Survey]** Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, Anima Anandkumar. *signSGD: Compressed Optimisation for Non-Convex Problems.* ICML, 2018.
- **[Survey]** Kwangjun Ahn, Xiang Cheng, Minhak Song, Chulhee Yun, Ali Jadbabaie, Suvrit Sra. *Linear Attention Is (Maybe) All You Need (to Understand Transformer Optimization).* ICLR, 2024.

## 10. Worked Example

Take a 6-layer, 384-dim decoder on a Zipfian vocabulary of 50k tokens, batch 128, and measure the two leading proxies at step 2000 in a *single* run.

- Token-frequency deciles: decile 1 (most frequent, ~72% of token mass) reaches loss $1.4$ under both optimizers. Decile 10 (rarest, ~0.3% of mass) reaches $6.1$ under SGD and $4.5$ under Adam. Overall gap $\approx 0.3$ nats, of which ~80% comes from deciles 7–10 — consistent with the class-imbalance account.
- Block Hessian: $\lambda_{\max}$ for the output-embedding block is $\approx 9\times10^2$; for the MLP down-projections in the middle layers, $\approx 4$. Heterogeneity ratio $\approx 200$. A single SGD step size must serve both: $\eta < 2/900 \approx 2.2\times10^{-3}$ to keep the embedding block stable, which leaves the MLP blocks moving $200\times$ slower than their own stability limit allows — consistent with the Hessian account.

Now the obstruction. The high-curvature block *is* the output embedding, and the output embedding's curvature is high *because* of the Zipfian token distribution — the softmax logit for a rare token sits on a nearly flat direction, while frequent-token rows accumulate large curvature. The two proxies are not independent measurements; they are the same fact read off two instruments. Fitting either one to $\Delta$ gives $R^2 > 0.9$, and the fits are indistinguishable.

To separate them you need an intervention that changes one and not the other. Flattening the token distribution (e.g. balanced-resampled data) changes both. Reparameterizing the embedding to equalize row curvature changes both. This is the non-identifiability in Section 6, made concrete: with the current instrument set, "class imbalance" and "block heterogeneity" are the same hypothesis stated in two vocabularies, and no run at any scale distinguishes them. The experiment in Section 8 is designed as the cheapest available wedge — arm 3 moves block curvature without touching the token distribution, arm 4 does the reverse — but even it separates the two only if $R$ differs sharply between the arms.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*