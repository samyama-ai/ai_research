---
id: 18-rl-for-llms/preference-noise-tolerance-bounds
title: "Preference Data Noise Tolerance Bounds"
topic: 18-rl-for-llms
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Preference Data Noise Tolerance Bounds

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/preference-noise-tolerance-bounds` · **Status:** partially-solved

## 1. Problem Statement

Preference datasets used for RLHF and DPO contain wrong labels. Human annotators agree with each other roughly three times in four; LLM-judge annotation adds its own systematic errors. The question: **how much label noise can a preference-optimization algorithm absorb before the aligned policy is worse than the reference policy, and does that threshold depend on the algorithm?**

Three variants, with different difficulty:

- **Theory.** Given a corruption model with rate $\epsilon$, bound the excess reward loss of the learned policy as a function of $\epsilon$, sample size $n$, and KL budget. Find the critical $\epsilon^\star$ above which no estimator recovers the reward ordering.
- **Method.** Build an estimator whose degradation under $\epsilon$ matches the information-theoretic rate without being told $\epsilon$.
- **Measurement.** Estimate $\epsilon$ on a real dataset. This is the blocked variant: on real data, $\epsilon$ is not identified from preference labels alone.

Solving it means: a procedure that takes a preference corpus, returns a calibrated $\hat\epsilon$ with an interval, and a guarantee of the form "at this $\hat\epsilon$ and this $n$, the post-training policy beats the reference under the *true* reward with probability $\geq 1-\delta$."

## 2. Formal Setting

Prompts $x \sim \rho$; response pair $(y_1,y_2) \sim \pi_{\text{ref}}(\cdot\mid x)^{\otimes 2}$. A latent reward $r^\star: \mathcal{X}\times\mathcal{Y}\to\mathbb{R}$ induces the Bradley–Terry (BT) probability

$$p^\star(y_1 \succ y_2 \mid x) = \sigma\!\big(\Delta(x,y_1,y_2)\big), \qquad \Delta = r^\star(x,y_1)-r^\star(x,y_2),\ \ \sigma(u)=\tfrac{1}{1+e^{-u}}.$$

The **clean** label is $z^\star \sim \text{Bern}(p^\star)$. The **observed** label is $z$, produced by a corruption channel. Two channels in use:

- Homogeneous flip: $\Pr[z \neq z^\star] = \epsilon$, independent of $(x,y_1,y_2)$.
- Instance-dependent / adversarial: an adversary flips an $\epsilon$-fraction chosen after seeing the data, $\frac{1}{n}\sum_i \mathbb{1}[z_i\neq z_i^\star] \le \epsilon$.

Observed preference probability under the homogeneous channel:

$$q(x,y_1,y_2) = \epsilon + (1-2\epsilon)\,\sigma(\Delta). \tag{1}$$

**How each quantity is actually measured.**

- $\epsilon$: estimated from repeat annotation. With two independent annotators on the same pair, the measured agreement rate is $A = \mathbb{E}[q^2+(1-q)^2]$. This is one scalar per pair and $q$ already mixes $\epsilon$ and $\Delta$ — see §6.
- $r^\star$: never observed. Proxied by a "gold" reward model trained on held-out clean-ish labels (Gao et al., 2023) or by a fixed LLM judge.
- Policy quality: win rate of $\pi_\theta$ against $\pi_{\text{ref}}$ under the gold proxy, reported at matched $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ in nats, since win rate without a KL axis is not comparable across methods.
- Reward range $B = \sup |r^\star|$: unmeasurable; entered as an assumption, and bounds degrade exponentially in it.

**DPO objective** with implicit reward $\hat r_\theta = \beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$:
$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\log\sigma(\hat r_\theta(y_w) - \hat r_\theta(y_l))$.

**Assumptions known to be violated.** (i) BT transitivity — human preferences are cyclic in aggregate; (ii) noise independent of the instance — real errors concentrate on close pairs and on long responses (length bias); (iii) a single annotator population — heterogeneous preferences look like noise but are structured; (iv) $y_1,y_2 \sim \pi_{\text{ref}}$ — most corpora are off-policy mixtures.

## 3. State of the Art

**Theory SOTA (established).**
- Zhu, Jordan & Jiao (ICML 2023) give finite-sample bounds for BT-MLE with linear reward: estimation error $\tilde O(\sqrt{d/n})$ scaled by $\gamma^{-1}$ where $\gamma \asymp e^{-B}$ — i.e. *intrinsic* BT stochasticity already costs exponentially in reward range, before any label flipping.
- Chowdhury, Kulkarni & Mahdavi (ICML 2024), rDPO: an unbiased debiased loss for known flip rate $\epsilon$,
$$\tilde{\mathcal{L}} = \frac{(1-\epsilon)\,\ell(y_w,y_l) - \epsilon\,\ell(y_l,y_w)}{1-2\epsilon},$$
with policy sub-optimality inflated by $(1-2\epsilon)^{-1}$. This is the preference-learning instance of Natarajan et al. (NeurIPS 2013) unbiased loss correction.
- Azar et al. (AISTATS 2024), IPO: DPO's optimum is unbounded when empirical preferences are deterministic; IPO's squared loss keeps the solution finite for any $\epsilon$, so IPO is noise-tolerant *by regularization* rather than by correction.

**Empirical SOTA (claimed, partially ablated).** rDPO and cDPO (label smoothing at rate $\epsilon$; Mitchell, 2023 note) beat DPO under *synthetically injected* flips at Pythia-scale (1.4B–2.8B) on IMDb-sentiment and Anthropic-HH. Robust-outlier methods (Bukharin et al., NeurIPS 2024, R³M — sparse outlier term on reward residuals) report gains under corrupted feedback. **Unablated:** none of these has been shown to help when $\epsilon$ is *native* to the corpus rather than injected, and all assume $\epsilon$ is supplied.

**Benchmark-number-only results.** RewardBench (Lambert et al., 2024) accuracies above 90% are a curated-set number; they do not translate into a noise-rate estimate for a training corpus, and the correlation between RewardBench score and downstream policy win rate is weak.

## 4. What Is Known

- **Human agreement floor.** InstructGPT (Ouyang et al., 2022): held-out labelers agree with training labelers $72.6 \pm 1.5\%$; training-labeler internal agreement $77.3\%$. Summarization (Stiennon et al., 2020): labeler–researcher agreement $\approx 77\%$; researcher–researcher $\approx 73\%$. Scale: tens of thousands of comparisons, 6B–175B policies.
- **Reward-model ceiling.** Anthropic HH reward models reach $\approx 67$–$70\%$ held-out pairwise accuracy at 52B (Bai et al., 2022) — consistent with the annotator floor, so the RM is near the noise limit, not underfit.
- **Overoptimization law.** Gao, Schulman & Hilton (ICML 2023): with $d=\sqrt{\mathrm{KL}}$, gold reward follows $d(\alpha-\beta d)$ for best-of-$n$ and $d(\alpha-\beta\log d)$ for RL, across proxy RMs 3M–3B trained on up to 100k comparisons. The peak-then-decline shape is the operational signature of noise tolerance being exceeded.
- **Correction works when $\epsilon$ is known and homogeneous.** Unbiasedness of the rDPO/Natarajan estimator is a theorem; variance inflation $(1-2\epsilon)^{-2}$ is tight for the flip channel.
- **Ensembles and regularization buy margin.** Reward-model ensembles (Coste et al., ICLR 2024) and WARM weight-averaging (Ramé et al., ICML 2024) delay the overoptimization turn at 7B scale, without changing the underlying $\epsilon$.

## 5. What Is Not Known

- **Methodologically blocked.** Estimating $\epsilon$ on real preference data. Equation (1) shows $\epsilon$ and the margin distribution of $\Delta$ are not separately identified from labels alone; repeat annotation measures $q$, not $(\epsilon,\Delta)$. Every empirical robustness claim therefore rests on injected noise.
- **Theoretically open.** A critical threshold $\epsilon^\star$ for *instance-dependent* corruption on the token-level (not bandit) MDP. Known bounds cover homogeneous flips or adversarial fractions with a coverage assumption; nothing is proven for noise correlated with $|\Delta|$, which is the empirically dominant case.
- **Theoretically open.** Whether any estimator without knowledge of $\epsilon$ can match the $(1-2\epsilon)^{-1}$ rate, or whether adaptivity to unknown $\epsilon$ is provably impossible under BT.
- **Empirically open.** The $\epsilon$–scale interaction. Whether larger policies tolerate more noise (better prior over $r^\star$) or less (higher capacity to fit flipped pairs) has not been run past ~7B with a controlled $\epsilon$ sweep and matched-KL evaluation.

## 6. Why It Is Hard

**Non-identifiability, exactly.** From (1), $(\epsilon, \Delta)$ and $(\epsilon', \Delta')$ give identical label distributions whenever $\epsilon+(1-2\epsilon)\sigma(\Delta) = \epsilon'+(1-2\epsilon')\sigma(\Delta')$. A one-parameter family of $(\epsilon,\Delta)$ pairs fits any dataset. No amount of repeat labeling from the *same* population breaks it: repeats estimate $q$, and $q$ is the confounded quantity. Breaking it needs an exogenous anchor — pairs with known ground-truth margin, or an annotator-covariate model with a testable exclusion restriction.

Secondary obstructions: absent ground truth for $r^\star$ (the gold RM shares the annotator population's errors, so the evaluation inherits the noise it is meant to measure); and the fact that the two policy-level failure modes — noise and KL over-optimization — produce the same downstream signature (win rate rising then falling), so an ablation that varies $\epsilon$ without pinning KL measures neither.

## 7. Current Research (as of 2026)

- Robust-loss variants beyond rDPO: distributionally-robust and $\ell_1$-outlier objectives for corrupted feedback (Georgia Tech / Bukharin line; Penn State / Mahdavi line).
- Uncertainty-aware and ensemble reward models as a noise buffer (DeepMind WARM line; UC Berkeley overoptimization line).
- Annotator-heterogeneity models that reinterpret "noise" as unmodelled preference diversity — mixture-of-BT and Nash-learning formulations (DeepMind, Munos et al.). *(frontier — verify)*
- LLM-judge miscalibration measurement: whether judge-generated preference corpora have a *lower* effective $\epsilon$ than human ones on verifiable subsets, and whether that transfers off-distribution. *(frontier — verify)*
- RLVR (verifiable rewards) as an end-run: on tasks with a checkable answer, $\epsilon$ is near zero, which is why the open problem now concentrates on non-verifiable domains.

## 8. Concrete Next Experiment

**Scale.** Llama-3.2-3B and Qwen2.5-7B base + SFT. 60k UltraFeedback-style prompt/pair records. Gold reward: an 8B RM trained on a disjoint 40k split, held fixed.

**Design.** Take the cleanest available split (triple-annotated, unanimous) as $\epsilon \approx 0$ baseline. Inject flips at $\epsilon \in \{0, 0.05, 0.10, 0.20, 0.30, 0.40\}$, two channels: uniform, and margin-targeted (flip only pairs with $|\hat\Delta| < \tau$, matching the real error profile).

**Arms.** DPO (control), cDPO with true $\epsilon$, rDPO with true $\epsilon$, rDPO with $\hat\epsilon$ estimated from a 2k triple-labeled subset, IPO. All at three $\beta$ values, evaluated on a KL-matched grid at $\mathrm{KL} \in \{5,10,20\}$ nats.

**Deciding number.** $\epsilon^\star$ = the largest injected flip rate at which an arm retains $\ge 95\%$ of its $\epsilon=0$ gold-reward gain at $\mathrm{KL}=10$ nats. One number per arm per channel. Theory predicts rDPO-with-known-$\epsilon$ has $\epsilon^\star$ near $0.45$ under the uniform channel; if the measured $\epsilon^\star$ is below $0.30$, the unbiased-correction story is variance-limited in practice, not bias-limited. The gap between rDPO-true-$\epsilon$ and rDPO-$\hat\epsilon$ prices the measurement problem directly. Cost: ~30 training runs, ~2k A100-hours.

## 9. Key References

- **[Foundational]** R. A. Bradley, M. E. Terry. *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika, 1952.
- **[Foundational]** N. Natarajan, I. Dhillon, P. Ravikumar, A. Tewari. *Learning with Noisy Labels.* NeurIPS, 2013.
- **[Foundational]** P. Christiano et al. *Deep Reinforcement Learning from Human Preferences.* NeurIPS, 2017. — arXiv:1706.03741
- **[Foundational]** N. Stiennon et al. *Learning to Summarize from Human Feedback.* NeurIPS, 2020. — arXiv:2009.01325
- **[Foundational]** L. Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS, 2022. — arXiv:2203.02155
- **[Foundational]** Y. Bai et al. *Training a Helpful and Harmless Assistant with RLHF.* 2022. — arXiv:2204.05862
- **[SOTA, theory]** B. Zhu, M. I. Jordan, J. Jiao. *Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons.* ICML, 2023. — arXiv:2301.11270
- **[SOTA, theory]** S. R. Chowdhury, A. Kulkarni, M. Mahdavi. *Provably Robust DPO: Aligning Language Models with Noisy Feedback.* ICML, 2024. — arXiv:2403.00409
- **[SOTA, theory]** M. G. Azar et al. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* AISTATS, 2024. — arXiv:2310.12036
- **[SOTA, empirical]** L. Gao, J. Schulman, J. Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[SOTA, empirical]** R. Rafailov et al. *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.* NeurIPS, 2023. — arXiv:2305.18290
- **[SOTA, empirical]** T. Coste, U. Anwar, R. Kirk, D. Krueger. *Reward Model Ensembles Help Mitigate Overoptimization.* ICLR, 2024.
- **[SOTA, empirical]** A. Ramé et al. *WARM: On the Benefits of Weight Averaged Reward Models.* ICML, 2024.
- **[Robustness]** A. Bukharin et al. *Robust Reinforcement Learning from Corrupted Human Feedback.* NeurIPS, 2024.
- **[Survey/benchmark]** N. Lambert et al. *RewardBench: Evaluating Reward Models for Language Modeling.* 2024.
- **[Note]** E. Mitchell. *A Note on DPO with Noisy Preferences and Relationship to IPO* (cDPO), 2023.

## 10. Worked Example

Take the InstructGPT number: two annotators agree on $72.6\%$ of pairs. Model each as an independent draw from the observed distribution $q$. Agreement is $q^2+(1-q)^2$; solving $q^2+(1-q)^2 = 0.726$ gives $q = 0.851$.

Now decompose $q = 0.851$ using (1):

| Hypothesis | $\epsilon$ | required $\sigma(\Delta)$ | implied $\Delta$ | DPO's optimal $\beta\log\frac{\pi}{\pi_{\text{ref}}}$ gap |
|---|---|---|---|---|
| A: all BT stochasticity | $0.00$ | $0.851$ | $1.74$ | $1.74$ |
| B: mild flipping | $0.10$ | $0.939$ | $2.73$ | $2.73$ |
| C: heavy flipping | $0.15$ | $1.000$ | $\infty$ | unbounded |

All three produce *identical* label counts and identical measured annotator agreement. The data cannot choose among them. But they prescribe different training: under A, plain DPO is correct and cDPO/rDPO under-fit; under C, plain DPO is driven to an unbounded log-ratio and only the KL penalty stops it, while rDPO with $\epsilon=0.15$ is the unbiased estimator.

The cost of guessing wrong is not symmetric. Apply the rDPO correction with $\epsilon=0.15$ when the truth is $\epsilon=0$: the estimator stays consistent for the *assumed* channel but its variance is inflated by $(1-2\epsilon)^{-2} = (0.7)^{-2} = 2.04\times$, so effective sample size drops from 60k to about 29k. Apply $\epsilon=0$ when the truth is $0.15$: the target margin is wrong by $2.73-1.74 = 0.99$ nats of implicit reward, and at $\beta=0.1$ that is a $9.9$-nat error in the policy log-ratio on exactly the pairs the model is most confident about.

That is the obstruction in one table: the choice that matters most — how much to correct — is the one quantity the preference data provably cannot reveal.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*