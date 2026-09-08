---
id: 34-diffusion-generative/diffusion-reward-hacking
title: "Reward Hacking in Diffusion Model Fine-Tuning"
topic: 34-diffusion-generative
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Reward Hacking in Diffusion Model Fine-Tuning

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/diffusion-reward-hacking` · **Status:** open

## 1. Problem Statement

Fine-tuning a text-to-image diffusion model against a learned reward (aesthetic predictor, ImageReward, HPSv2, PickScore) reliably raises the reward and, past some point, lowers the thing the reward was meant to stand for: human-judged quality, prompt fidelity, and sample diversity. The model finds regions of image space where the reward model is wrong.

Three variants, with very different difficulty:

- **Measurement.** Given a fine-tuned model $p_\theta$ and a proxy reward $\hat r$, decide whether reward gain is genuine or hacked — without a fresh human study for every checkpoint. Currently the *only* accepted ground truth is human preference, which is expensive, noisy, and itself drifts.
- **Method.** Fine-tune so that true reward improves monotonically, or at least detect the turning point online and stop there. Regularizers exist (KL to the pretrained model, LoRA rank caps, reward ensembles, early stopping); none is known to be optimal or even reliably calibrated.
- **Theory.** Characterize when the proxy-optimal policy is also gold-optimal in the *sequential* denoising setting. Skalse et al. (NeurIPS 2022) give an unhackability characterization for MDPs; nobody has instantiated it for the denoising MDP, where the reward is terminal-only and the state space is $\mathbb{R}^{d}$ with $d \sim 10^4$–$10^5$.

Solved would mean: a stopping rule, computable from generated samples and the pretrained model alone, that predicts the human-preference turning point within a stated tolerance across at least three reward models and two base models.

## 2. Formal Setting

**Denoising MDP** (Black et al., ICLR 2024). For $T$ sampling steps, state $s_t = (c, t, x_t)$ with prompt $c \sim p_{\mathcal{C}}$, action $a_t = x_{t-1}$, policy $\pi_\theta(a_t \mid s_t) = p_\theta(x_{t-1} \mid x_t, c)$, and reward $0$ everywhere except the terminal step, where it is $\hat r(x_0, c)$.

**Objective.** KL-regularized:

$$J_\alpha(\theta) = \mathbb{E}_{c,\;x_0 \sim p_\theta(\cdot\mid c)}\big[\hat r(x_0,c)\big] \;-\; \alpha\, D_{\mathrm{KL}}\!\left(p_\theta(\cdot \mid c)\,\|\,p_{\mathrm{pre}}(\cdot \mid c)\right).$$

In continuous time this is an entropy-regularized stochastic control problem whose optimum is the tilted measure $p^*(x_0\mid c) \propto p_{\mathrm{pre}}(x_0 \mid c)\exp(\hat r(x_0,c)/\alpha)$ (Uehara et al., 2024).

**How each quantity is actually measured.**

- $\hat r$: a forward pass of a CLIP- or BLIP-initialized scorer. Aesthetic v2 outputs roughly $[1,10]$; ImageReward and PickScore are unnormalized logits, so absolute values are not comparable across reward models — only rank order within one.
- $r^\star$ (gold): win rate against the base model's samples on matched prompts, from human raters or a held-out judge. Reported as $\mathrm{WR} \in [0,1]$ with binomial CI. There is no scalar gold reward; $r^\star$ exists only as a pairwise comparison.
- $D_{\mathrm{KL}}$: **not** directly measurable for the trajectory distribution. Practice substitutes (i) the per-step KL summed over the sampling trajectory, an upper bound by the chain rule; (ii) LoRA weight norm; (iii) $\sqrt{\mathrm{KL}}$ estimated from the RL objective's own log-ratios. These are different numbers and papers rarely say which is plotted.
- Diversity: pairwise LPIPS or DINOv2 cosine among $n \ge 16$ samples per prompt; sensitive to $n$ and to the fixed seed set.

**Assumptions known to be violated.** (a) $\hat r$ is a monotone transform of human preference — false off-distribution, which is exactly where optimization goes. (b) Terminal-only reward makes credit assignment well-posed — true in principle, but likelihood-ratio gradients over $T=50$ steps have variance that forces truncation to the last $K$ steps (DRaFT-$K$), which biases the objective. (c) Prompts are i.i.d. from $p_{\mathcal{C}}$ — training prompt sets are typically a few hundred hand-written templates, so "generalization" numbers are measured within a narrow support.

## 3. State of the Art

**Empirical/systems SOTA.**

- *Policy-gradient*: DDPO (Black et al., ICLR 2024) and DPOK (Fan et al., NeurIPS 2023). Established: both raise proxy reward on SD-1.5; DPOK's KL term measurably slows degradation relative to unregularized supervised reward-weighted training.
- *Differentiable reward backprop*: DRaFT (Clark et al., ICLR 2024) and AlignProp (Prabhudesai et al., 2023). Established: order-of-magnitude sample-efficiency gains over DDPO on the aesthetic reward. **Claimed but unablated:** that low LoRA rank is an effective anti-hacking regularizer — the papers show reward/quality curves at a handful of ranks, not a controlled sweep against matched KL.
- *Preference optimization without an explicit reward model*: Diffusion-DPO (Wallace et al., CVPR 2024) on SDXL, trained on Pick-a-Pic. Sidesteps a *separate* proxy but does not escape the problem: the implicit reward is still finite-sample.
- *Anti-overoptimization specifically*: TDPO-R (Zhang et al., ICML 2024) attributes overoptimization to primacy bias in the value/critic path and resets neurons; PRDP (Deng et al., CVPR 2024) reformulates reward maximization as a stable supervised regression to scale to large prompt sets. Both report improved reward-vs-quality frontiers; neither has an independent replication at SDXL scale that I can confirm.
- *Distribution-correct fine-tuning*: Adjoint Matching (Domingo-Enrich et al., ICLR 2025) shows that naive RL fine-tuning does not sample the tilted target $p^*$ and gives a memoryless-noise-schedule correction that does. This is the strongest *theoretical* SOTA and it changes what is being optimized, but it does not make a finite-sample $\hat r$ safe to maximize.

**Theory SOTA.** Gao, Schulman, Hilton (ICML 2023) give empirical scaling laws for reward overoptimization in the *language* setting, fit as $R(d)=d(\alpha-\beta\log d)$ for RL with $d=\sqrt{\mathrm{KL}}$. There is no established diffusion analogue.

## 4. What Is Known

- DDPO raises LAION aesthetic score on SD-1.5 from roughly $5.5$ to over $7$ within a few hundred PPO epochs on ~45 animal prompts; the resulting samples visibly collapse toward a cartoon/painterly mode. Scale: SD-1.5, single-digit GPU-days.
- DRaFT-1 (backprop through only the last sampling step) reaches comparable or better aesthetic scores at a small fraction of DDPO's sample cost, with the paper reporting roughly two orders of magnitude fewer reward queries on the aesthetic objective. Scale: SD-1.5, 512px.
- Diffusion-DPO on SDXL-1.0 wins majority human preference over the base model on PartiPrompts and HPSv2 prompt sets (reported win rates in the ~65–70% range in the paper). Scale: 851k Pick-a-Pic pairs.
- Gao et al.: proxy–gold divergence is monotone in $\sqrt{\mathrm{KL}}$, and the KL at which gold reward peaks *increases* with reward-model size (measured across 3M–3B reward models, LM policies up to 3B). This is the cleanest quantitative statement of the phenomenon anywhere, and it is in the wrong modality.
- Reward-model ensembling reduces but does not remove overoptimization (Coste et al., ICLR 2024, language setting): hacking that is shared across ensemble members survives.
- Reliable regularity, diffusion-specific: **reward transfer fails asymmetrically.** Optimizing aesthetic score degrades ImageReward and prompt-alignment metrics well before human raters call the images bad; optimizing ImageReward degrades diversity first. Reported qualitatively in several of the above papers; no single controlled study fixes KL and sweeps reward models.

## 5. What Is Not Known

- **Theoretically open.** Whether a diffusion analogue of the $\sqrt{\mathrm{KL}}$ scaling law exists, and with what functional form. No proof either way, and the terminal-reward MDP is not covered by existing unhackability results (Skalse et al., 2022), which assume a shared state-action reward structure.
- **Theoretically open.** Whether trajectory-level KL is even the right complexity measure. Two policies can have equal trajectory KL to $p_{\mathrm{pre}}$ and very different $x_0$-marginal KL, since the sampler is many-to-one onto images.
- **Empirically open.** The full 2-D grid (reward model $\times$ KL budget $\times$ base-model scale) with human gold labels at every cell. Runnable today: ~$10^2$ fine-tuning runs at SDXL scale plus ~$10^5$ human comparisons. Nobody has published it.
- **Methodologically blocked.** "Reward hacking" has no operational definition that does not reduce to "humans disagree with the proxy." Diversity collapse, prompt-drift, and off-manifold texture artifacts are lumped together under one word and plausibly have different causes and different fixes.

## 6. Why It Is Hard

The specific obstruction is **absent scalar ground truth combined with confounded measurement**. Gold reward for images exists only as pairwise human preference, so the $x$-axis of every overoptimization curve (proxy reward) is a scalar while the $y$-axis (gold) is a win rate on a fixed comparison set — and the comparison set is itself drawn from the base model, so the metric silently changes meaning as the policy moves.

Second obstruction: **non-identifiability of the regularizer.** KL to the pretrained model, LoRA rank, learning rate, and number of backprop steps $K$ all move the same latent "distance travelled" quantity. Papers vary them jointly. It is not currently possible to say whether low-rank adaptation prevents hacking or merely slows the effective step size — the two predict identical curves against wall-clock and different curves only against measured KL, which is rarely measured.

Third: automated judges (GPT-4V-class, HPSv2) share encoder lineage — usually CLIP — with the reward models being hacked, so a judge can certify an artifact that fools both.

## 7. Current Research (as of 2026)

- Stochastic-control-correct fine-tuning: adjoint matching and memoryless schedules, extended to flow matching *(frontier — verify)*. Groups around Meta FAIR / NYU (Domingo-Enrich, Chen, Albergo).
- Inference-time alignment as an alternative to weight updates — best-of-$n$, SMC/twisted particle sampling, and value-guided sampling — where the "KL budget" is explicit in $n$ and reversible *(frontier — verify)*.
- Reward-model uncertainty: ensembles, conservative/pessimistic reward estimates, and reward-model calibration for image scorers.
- Diagnostic work on the aesthetic predictor itself: it is a linear head on CLIP embeddings trained on ~176k ratings, so its off-distribution behaviour is analytically tractable in a way the newer transformer scorers are not.

## 8. Concrete Next Experiment

**Question.** Does gold preference peak at a KL that is predictable from proxy-reward curvature alone?

**Scale.** SD-1.5 (860M UNet) and SDXL-1.0 (2.6B), LoRA rank 32. Three reward models: LAION aesthetic v2, ImageReward, PickScore. Six KL budgets per (base, reward) cell, hit by varying $\alpha$ in the DRaFT objective, not by early stopping. That is $2\times3\times6 = 36$ runs, roughly 300 A100-hours total.

**Instrumentation.** At each checkpoint, log (i) proxy reward on 500 held-out prompts, (ii) *measured* trajectory KL to $p_{\mathrm{pre}}$ under a shared seed set, (iii) the other two proxy rewards, (iv) mean pairwise DINOv2 similarity over 16 samples/prompt.

**Control arm.** Best-of-$n$ sampling from the frozen base model with $n$ chosen so the induced KL, $\log n - (n-1)/n$, matches each fine-tuned checkpoint's measured KL. This is the correct control: it achieves the same distance from the prior with *no* parameter change, so any excess gold degradation in the fine-tuned arm is attributable to weight-space hacking rather than to selection pressure.

**Deciding number.** $\Delta\mathrm{KL}^\star = \mathrm{KL}^\star_{\text{gold}} - \mathrm{KL}^\star_{\text{predicted}}$, where $\mathrm{KL}^\star_{\text{gold}}$ is the KL maximizing human win rate (2,000 comparisons per cell, 3 raters each) and $\mathrm{KL}^\star_{\text{predicted}}$ is the KL at which the proxy-reward curve's second derivative crosses a fixed threshold. If $|\Delta\mathrm{KL}^\star| < 0.1\,\mathrm{KL}^\star_{\text{gold}}$ in $\ge 5$ of 6 cells, an online stopping rule exists and the method variant is largely solved. If the sign of $\Delta\mathrm{KL}^\star$ flips across reward models, no proxy-only rule exists and the field should stop looking for one.

## 9. Key References

- **[Foundational]** Amodei, Olah, Steinhardt, Christiano, Schulman, Mané. *Concrete Problems in AI Safety.* 2016. — arXiv:1606.06565
- **[Foundational]** Skalse, Howe, Krasheninnikov, Krueger. *Defining and Characterizing Reward Hacking.* NeurIPS, 2022. — arXiv:2209.13085
- **[Foundational]** Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[SOTA]** Black, Janner, Du, Kostrikov, Levine. *Training Diffusion Models with Reinforcement Learning.* ICLR, 2024. — arXiv:2305.13301
- **[SOTA]** Fan, Watkins, Du, Liu, Ryu, Boutilier, Abbeel, Ghavamzadeh, Lee, Lee. *DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models.* NeurIPS, 2023. — arXiv:2305.16381
- **[SOTA]** Clark, Vicol, Swersky, Fleet. *Directly Fine-Tuning Diffusion Models on Differentiable Rewards.* ICLR, 2024. — arXiv:2309.17400
- **[SOTA]** Prabhudesai, Goyal, Pathak, Fragkiadaki. *Aligning Text-to-Image Diffusion Models with Reward Backpropagation.* 2023. — arXiv:2310.03739
- **[SOTA]** Wallace, Dang, Rafailov, Zhou, Lou, Purushwalkam, Ermon, Xiong, Joty, Naik. *Diffusion Model Alignment Using Direct Preference Optimization.* CVPR, 2024. — arXiv:2311.12908
- **[SOTA]** Domingo-Enrich, Drozdova, Vanden-Eijnden, Chen, Ramesh. *Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control.* ICLR, 2025. — arXiv:2409.08861
- **[SOTA]** Zhang, Zhang, Zhou, Wang, Liu, et al. *Confronting Reward Overoptimization for Diffusion Models: A Perspective of Inductive and Primacy Biases.* ICML, 2024. — arXiv:2402.08552
- **[Reward models]** Xu, Liu, Wu, Tong, Li, Ding, Tang, Dong. *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation.* NeurIPS, 2023. — arXiv:2304.05977
- **[Reward models]** Kirstain, Polyak, Singer, Matiana, Penna, Levy. *Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation.* NeurIPS, 2023. — arXiv:2305.01569
- **[Survey]** Uehara, Zhao, Biancalani, Levine. *Understanding Reinforcement Learning-Based Fine-Tuning of Diffusion Models: A Tutorial and Review.* 2024. — arXiv:2407.13734

## 10. Worked Example

Fine-tune SD-1.5 with DRaFT against LAION aesthetic v2 on 45 animal prompts. Trace one checkpoint sequence:

| step | aesthetic $\hat r$ | ImageReward | pairwise DINOv2 sim | human WR vs base |
|---|---|---|---|---|
| 0 | 5.5 | 0.00 | 0.31 | 0.50 |
| 200 | 6.4 | +0.15 | 0.38 | 0.71 |
| 600 | 7.1 | +0.02 | 0.57 | 0.62 |
| 1500 | 7.6 | −0.34 | 0.79 | 0.38 |

(Aesthetic and qualitative trends are consistent with published DDPO/DRaFT curves; the win-rate column is the *shape* a hacking study would need to measure, cell by cell.)

The proxy rises monotonically across all four rows. Human preference peaks somewhere between step 200 and 600 and is *worse than the base model* by step 1500. Nothing in the aesthetic column marks the turn.

Now make the obstruction visible. The aesthetic predictor is a small MLP on a normalized CLIP ViT-L/14 image embedding $e \in \mathbb{S}^{767}$, fit to ~176k human ratings. Its top singular direction $v$ defines a rough linear score $\hat r \approx w^\top e + b$. Moving $e$ by $\delta$ along $v$ moves $\hat r$ by $\|w\|\delta$ *whether or not the point stays on the manifold of natural-image embeddings*. Training data density along $v$ falls off past the 99th percentile of the rating set; beyond that the predictor is extrapolating from a linear fit, and its ordering has never been checked against a human.

So: at step 1500 the model has found $e$ with $\hat r = 7.6$, a value achieved by fewer than $0.1\%$ of LAION images. The score is high because the head is linear, not because the image is good. Detecting this needs either the training-set density along $v$ — unavailable for ImageReward or PickScore, whose scorers are full transformers with no such tractable direction — or fresh human labels, which is the thing the proxy was introduced to avoid. That circle is the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*