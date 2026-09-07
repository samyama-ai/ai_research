---
id: 03-training-dynamics/adam-vs-sgd-transformers
title: "Why Adam Beats SGD on Transformers"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Why Adam Beats SGD on Transformers

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/adam-vs-sgd-transformers` · **Status:** open

## 1. Problem Statement

On convolutional image classifiers, well-tuned SGD with momentum matches or beats Adam. On transformer language models, it does not: Adam/AdamW reaches a target validation loss in far fewer steps, and tuned SGD often plateaus at a loss SGD cannot cross within any practical budget. The gap is large, reproducible, and unexplained.

Three variants, of very different difficulty:

- **Measurement.** Is the gap a *convergence-speed* gap (same loss, more steps) or a *reachability* gap (SGD never gets there)? Under what tuning protocol, at what batch size, and does it survive as $\to$ compute-optimal scale? Requires a protocol in which "tuned SGD" is not a straw man.
- **Method.** What is the minimal modification to SGD that closes the gap? Per-parameter-block learning rates? Sign of the gradient? A preconditioner on the embedding/LayerNorm blocks only? Success here bounds the explanation from above.
- **Theory.** Give a property $P$ of the transformer loss landscape such that (i) $P$ holds empirically for transformer training, (ii) $P$ fails for ResNet/ImageNet training, and (iii) under $P$ there is a provable separation between Adam's and SGD's iteration complexity. No such triple is currently established.

Solving it means: an identified mechanism that *predicts* an intervention nobody has tried, and the intervention works.

## 2. Formal Setting

Model $f_\theta$, $\theta\in\mathbb{R}^d$ partitioned into blocks $\theta=(\theta^{(1)},\dots,\theta^{(B)})$ (embeddings, per-layer $Q,K,V,O$, MLP, LayerNorm gains). Data $(x,y)\sim\mathcal{D}$, loss $\mathcal{L}(\theta)=\mathbb{E}[\ell(f_\theta(x),y)]$, next-token cross-entropy over vocabulary $V$.

Stochastic gradient on batch $B_t$: $g_t=\frac{1}{|B_t|}\sum_{i\in B_t}\nabla_\theta \ell_i$.

SGD-M: $m_t=\beta m_{t-1}+g_t$, $\theta_{t+1}=\theta_t-\eta_t m_t$.
Adam: $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$, $\theta_{t+1}=\theta_t-\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)$, elementwise. As $\beta_1,\beta_2\to0$ this is signSGD: $\theta_{t+1}=\theta_t-\eta_t\,\mathrm{sign}(g_t)$.

**Quantities, as measured.**

- *Gap.* $G(C)=\mathcal{L}^{\mathrm{SGD}}_\star(C)-\mathcal{L}^{\mathrm{Adam}}_\star(C)$, where $\mathcal{L}_\star(C)=\min_{\text{hparams}}$ final validation loss at compute $C$, each optimizer given an *equal-cost* tuning budget (same number of trials, same search space dimension). Report in nats/token.
- *Gradient noise.* Full-batch gradient $g_\star=\nabla\mathcal{L}$ estimated by accumulating over the entire training set; noise $\Sigma=\mathrm{Cov}(g)$; tail index of $\|g_i-g_\star\|$ by Hill estimator.
- *Block heterogeneity.* Per-block Hessian spectra $\lambda(\nabla^2_{\theta^{(b)}}\mathcal{L})$ via SLQ (stochastic Lanczos quadrature). Heterogeneity = spread of $\lambda_{\max}^{(b)}$ across $b$, typically $10^2$–$10^4\times$ on transformers, $\approx10\times$ on CNNs (Zhang et al., 2024).
- *Directional sharpness.* $s(u)=u^\top\nabla^2\mathcal{L}\,u/\|u\|^2$ along the actual update direction $u$, by finite differences (Pan & Li, 2023).
- *Class imbalance.* Token frequency $\pi_c$ over $V$; measure per-frequency-decile loss $\mathcal{L}_c$, not just aggregate loss.

**Assumptions, and which are violated.** Smoothness with a global $L$: *violated* — transformer sharpness is direction- and block-dependent and grows during training. Bounded-variance noise: *violated* — the gradient-noise distribution is heavy-tailed at small batch (Zhang et al., 2020), though see §4 for why this is not the cause. Stationary objective: *violated* — LR warmup, schedule, and data ordering are part of the recipe, so "the optimizer" is never measured in isolation. Equal tuning: routinely violated in published comparisons; Adam's defaults are the product of a decade of community search, SGD's are not.

## 3. State of the Art

**Established, with ablation.**

- *Noise is not the cause.* Kunstner, Chen, Lavington & Schmidt (ICLR 2023) run **full-batch** GD vs full-batch Adam on transformers: the gap persists with zero gradient noise. This falsifies the heavy-tail-noise explanation as the primary mechanism and is the single most load-bearing ablation in the area. Sign descent with momentum recovers most of Adam's advantage.
- *Class imbalance reproduces the gap.* Kunstner, Yadav, Milligan, Schmidt & Bietti (NeurIPS 2024) show heavy-tailed class imbalance alone — even in linear softmax regression and MLPs — produces the Adam/GD separation: GD drives down loss on frequent classes and stalls on rare ones.
- *Hessian block heterogeneity.* Zhang, Chen, Ding, Li, Sra & Luo (NeurIPS 2024) measure per-block Hessian spectra: transformers show block-wise eigenvalue ranges differing by orders of magnitude; CNNs do not. A single global learning rate is therefore simultaneously too large for one block and too small for another. They show a blockwise-LR SGD variant closes much of the gap.

**Claimed but unablated / benchmark-only.**

- *Adam $\approx$ $\ell_\infty$-geometry steepest descent.* Xie & Li (ICML 2024) characterize AdamW's implicit bias as $\ell_\infty$-norm-constrained optimization. Suggestive; not tied to a measured landscape property that separates transformers from CNNs.
- *Adam as FTRL.* Ahn, Zhang, Kim & Sra (ICML 2024) reinterpret Adam through online learning of update rules. Explanatory framing, no separation theorem for transformers.
- *Optimizer equivalence at scale.* Zhao, Morwani, Brandfonbrener, Vyas & Kakade (2024, "Deconstructing What Makes a Good Optimizer for Language Models") find Adam, Adafactor, Lion and Signum land within a narrow band of each other at 150M–1.2B — but SGD stays outside it. This is a benchmark number under one tuning protocol, not a proof of mechanism.
- *Muon / Shampoo / SOAP* beat AdamW on wall-clock at 100M–1B (Jordan et al., 2024 speedruns; Vyas et al., 2024). Benchmark results; the winning ingredient (orthogonalized updates vs curvature) is not isolated.

## 4. What Is Known

- **Magnitude.** On a 6-layer transformer on PTB/WikiText, full-batch Adam vs full-batch GD leaves a gap of roughly $1$ nat/token in training loss at matched steps (Kunstner et al., 2023) — a factor of $\approx e$ in perplexity, not a tuning artifact.
- **Architecture specificity.** The same protocol gives a small or zero gap for ResNet-50/ImageNet, where tuned SGD-M matches Adam to within noise. The gap is transformer/language-specific, not a general deep-learning fact.
- **Sign is most of it.** Signum (sign + momentum) recovers the bulk of Adam's advantage on transformers at 100M scale; the $\sqrt{v}$ second moment adds little beyond sign once momentum is present (Bernstein et al., ICML 2018; Balles & Hennig, ICML 2018; Kunstner et al., 2023).
- **Rare tokens.** Under Zipf-distributed labels, GD's per-class loss on the bottom frequency deciles decreases $\sim\pi_c$-slowly, while Adam's is near frequency-independent (Kunstner et al., 2024, at linear-model and 100M-transformer scale).
- **Heterogeneity gradient.** Block Hessian $\lambda_{\max}$ spread: $\sim10^3\times$ on GPT-2-scale transformers vs $\sim10^1\times$ on CNNs (Zhang et al., 2024).
- **Not warmup.** The gap survives with warmup, gradient clipping, and LR schedules tuned separately per optimizer.

## 5. What Is Not Known

- **Theoretically open.** No theorem of the form "under measured property $P$ of transformer losses, Adam's iteration complexity is $O(\cdot)$ and SGD's is $\Omega(\cdot)$ with a separating factor." Existing separations rely on constructed objectives, not properties verified in real training runs.
- **Theoretically open.** Whether class imbalance, block heterogeneity, and $\ell_\infty$-geometry are three descriptions of one mechanism or three distinct contributions. They are correlated in every real transformer, so they are **non-identifiable** from observational runs.
- **Empirically open.** Does the gap shrink, hold, or grow with scale? Nobody has run a matched-tuning-budget SGD vs Adam sweep across $\ge4$ model sizes spanning $10^8$–$10^{10}$ params at Chinchilla-optimal tokens. Runnable; ~$10^5$ GPU-hours; unrun.
- **Empirically open.** Does per-block-LR SGD, with blocks set from *measured* Hessian spectra, close the gap at 1B+? Shown at small scale only.
- **Methodologically blocked.** "Tuned SGD" has no accepted definition. Adam has 3–4 effective hyperparameters plus decoupled weight decay; SGD has 2. Equal-trial-count and equal-cost protocols give different answers, and no comparison is protocol-free. AlgoPerf (Dahl et al., 2023) is the closest to a standard and still does not fix this for the SGD arm.

## 6. Why It Is Hard

**Non-identifiability is the obstruction.** In a real transformer, heavy-tailed token frequency, block-wise Hessian heterogeneity, and heavy-tailed gradient noise all co-occur and all correlate with the embedding/output layers. Each of the three published mechanisms predicts the *same* observable — Adam wins, sign of the gradient recovers most of the win, the effect concentrates in embedding-adjacent blocks. Observational runs cannot distinguish them; only an intervention that moves one factor while holding the others fixed can, and the obvious interventions (flatten the token distribution, equalize block curvature) change the task or the architecture, which changes the thing being explained.

Secondary: the tuning confound (§5) means any reported gap is a lower bound on SGD's best case, and compute cost makes the scaling question a $10^5$-GPU-hour experiment that no group has an incentive to run when Adam already works.

## 7. Current Research (as of 2026)

- Schmidt's group (UBC) and Bietti (Flatiron): class-imbalance mechanism, extending from linear models to full transformers.
- Sra (MIT/TUM), Luo and Zhang (CUHK-Shenzhen): Hessian-structure explanations and blockwise preconditioning (Adam-mini, blockwise-LR SGD).
- Kakade/Brandfonbrener/Vyas (Harvard): controlled optimizer comparisons at 150M–1.3B, SOAP.
- Jordan, Bernstein and the speedrun community: Muon — orthogonalized momentum updates — now standard in open speedrun baselines and reported in large-scale training reports *(frontier — verify claims about frontier-model adoption)*.
- Online-learning reinterpretations (Ahn, Cutkosky) aiming at a separation theorem rather than a bound.
- *(frontier — verify)* Several 2025–2026 preprints argue "gradient heterogeneity" subsumes both class imbalance and block heterogeneity; the unification is not yet independently reproduced.

## 8. Concrete Next Experiment

**Question decided:** is the gap caused by heavy-tailed label frequency, or by block curvature heterogeneity that exists independently of it?

**Scale.** 350M-parameter decoder transformer, 7B tokens (Chinchilla-optimal), batch 0.5M tokens, ~4k A100-hours per arm. Four arms, each tuned with an identical 24-trial random search over $(\eta,\text{warmup},\text{clip},\text{wd})$.

**Arms.**
1. AdamW, natural Zipf token distribution — reference.
2. SGD-M, natural distribution — control arm (the known-losing baseline).
3. SGD-M, **frequency-flattened** objective: per-token loss reweighted by $w_c\propto \pi_c^{-1}$, normalized, so the effective class distribution is uniform. Same data, same architecture, same tokenizer.
4. AdamW, frequency-flattened — to confirm the reweighting has not broken the task.

Measure block Hessian $\lambda_{\max}^{(b)}$ by SLQ at 10%, 50%, 100% of training in all arms.

**Deciding number.** $\Delta=\big(\mathcal{L}_2-\mathcal{L}_1\big)-\big(\mathcal{L}_3-\mathcal{L}_4\big)$, the reduction in the Adam–SGD gap caused by flattening, in nats/token on a held-out set evaluated *under the natural distribution*. If $\Delta\ge0.5$ nat (i.e. flattening removes over half a ~0.9-nat gap), class imbalance is the dominant mechanism. If $\Delta\le0.1$ nat while block heterogeneity stays at $\ge10^2\times$, imbalance is incidental and curvature structure is the mechanism. Intermediate values with a measured drop in heterogeneity indicate the two are the same effect — itself a publishable resolution.

## 9. Key References

- **[Foundational]** Diederik P. Kingma, Jimmy Ba. *Adam: A Method for Stochastic Optimization.* ICLR, 2015. — arXiv:1412.6980
- **[Foundational]** Lukas Balles, Philipp Hennig. *Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients.* ICML, 2018.
- **[Foundational]** Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, Anima Anandkumar. *signSGD: Compressed Optimisation for Non-Convex Problems.* ICML, 2018.
- **[SOTA]** Frederik Kunstner, Jacques Chen, Jonathan Wilder Lavington, Mark Schmidt. *Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be.* ICLR, 2023.
- **[SOTA]** Frederik Kunstner, Robin Yadav, Alan Milligan, Mark Schmidt, Alberto Bietti. *Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models.* NeurIPS, 2024.
- **[SOTA]** Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, Zhi-Quan Luo. *Why Transformers Need Adam: A Hessian Perspective.* NeurIPS, 2024.
- **[Related]** Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, Suvrit Sra. *Why Are Adaptive Methods Good for Attention Models?* NeurIPS, 2020.
- **[Related]** Yan Pan, Yuanzhi Li. *Toward Understanding Why Adam Converges Faster Than SGD for Transformers.* OPT workshop, NeurIPS, 2022/2023.
- **[Related]** Shuo Xie, Zhiyuan Li. *Implicit Bias of AdamW: $\ell_\infty$-Norm Constrained Optimization.* ICML, 2024.
- **[Related]** Kwangjun Ahn, Zhiyuan Zhang, Yunbum Kook, Yan Dai. *Understanding Adam Optimizer via Online Learning of Updates: Adam is FTRL in Disguise.* ICML, 2024.
- **[Benchmark]** George E. Dahl et al. *Benchmarking Neural Network Training Algorithms.* 2023.
- **[Benchmark]** Rosie Zhao, Depen Morwani, David Brandfonbrener, Nikhil Vyas, Sham Kakade. *Deconstructing What Makes a Good Optimizer for Language Models.* 2024.

## 10. Worked Example

Take a linear softmax model on $V=50{,}000$ tokens with Zipf frequencies $\pi_c\propto 1/c$. Normalizer $H_{50000}=\sum_{c=1}^{50000}1/c\approx 10.9$. The most frequent token has $\pi_1\approx 1/10.9\approx 9.2\times10^{-2}$; the 50,000th has $\pi_{50000}\approx 1/(50000\cdot10.9)\approx1.8\times10^{-6}$.

The gradient of cross-entropy w.r.t. the output-embedding row for class $c$ has magnitude $\Theta(\pi_c)$ near initialization. With a single global learning rate $\eta$, GD's per-step progress on class $c$ scales as $\eta\pi_c$. Stability caps $\eta$ by the sharpest direction — set by $\pi_1$. So the steps needed to fit class $c$ scale as $\pi_1/\pi_c$:

$$\frac{\pi_1}{\pi_{50000}}\approx\frac{9.2\times10^{-2}}{1.8\times10^{-6}}\approx 5\times10^{4}.$$

Adam divides by $\sqrt{v}\propto|g_c|$, so its per-class step is $\Theta(\eta)$ regardless of $\pi_c$: the $5\times10^4$ factor disappears. That is the class-imbalance story, and it is quantitatively the right size — the Zipf tail carries $\sim40\%$ of the total loss mass, so a $5\times10^4$ slowdown on it is easily worth $\approx1$ nat/token.

**Now the obstruction.** Compute the Hessian of the same model. The block for class $c$ has $\lambda_{\max}\approx\pi_c(1-\pi_c)\|x\|^2$. The spread across blocks is

$$\frac{\lambda_{\max}^{(1)}}{\lambda_{\max}^{(50000)}}\approx\frac{\pi_1}{\pi_{50000}}\approx5\times10^{4},$$

the same number. The block-heterogeneity account and the class-imbalance account predict an identical separation with an identical constant, because on this model heterogeneity *is* imbalance. No measurement taken during a normal training run can tell them apart: they are the same statistic viewed twice. The experiment in §8 exists precisely to break that tie by moving $\pi_c$ while watching whether $\lambda_{\max}^{(b)}$ follows.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*