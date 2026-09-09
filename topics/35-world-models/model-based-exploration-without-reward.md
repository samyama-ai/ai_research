---
id: 35-world-models/model-based-exploration-without-reward
title: "Model-Based Exploration Without Reward Signals"
topic: 35-world-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Model-Based Exploration Without Reward Signals

> **Topic:** World Models & Planning · **ID:** `35-world-models/model-based-exploration-without-reward` · **Status:** partially-solved

## 1. Problem Statement

An agent is dropped into an environment with no reward function. It acts for a budget of $N$ steps, learns a world model, and is then handed a reward function it has never seen. How should it act during the $N$ steps so that the model supports near-optimal planning for *any* reward in a class $\mathcal{R}$?

Three variants, with different difficulty:

- **Theory variant (reward-free exploration, RFE).** Tabular or low-rank MDP, known $S$, $A$, $H$. Output an estimated model $\hat{P}$ such that for every $r \in \mathcal{R}$ the greedy policy on $\hat{P}$ is $\varepsilon$-optimal. Solved up to constants in the tabular case.
- **Method variant.** Deep world model over pixels or proprioception. Output a replay buffer plus a learned latent dynamics model. Success predicate: expert-normalized return after a small fine-tune budget, versus a task-aware agent given the same total steps.
- **Measurement variant.** What is a *good* reward-free dataset, independent of the downstream task suite? No accepted definition. This is where the field is actually stuck.

Solving it means: a single unsupervised interaction phase whose model beats task-specific training on a downstream suite the explorer was not tuned against, with the advantage surviving a matched-compute control.

## 2. Formal Setting

Controlled Markov process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, H, \mu)$ with no reward. A reward class $\mathcal{R} \subseteq \{r: \mathcal{S}\times\mathcal{A}\to[0,1]\}$ is revealed at test time. For $r \in \mathcal{R}$ let $V^\star_r$ be the optimal value under true $P$, and $\hat{\pi}_r$ the optimal policy under the learned model $\hat{P}$.

**Objective (measured as):** run the planner on $\hat{P}$, then execute $\hat{\pi}_r$ in the *true* environment for $K$ episodes and average the return. The reported quantity is the empirical suboptimality
$$\hat{\Delta} = \max_{r \in \mathcal{R}_{\text{test}}}\Big(V^\star_r - \tfrac{1}{K}\sum_{k=1}^{K} G_k(\hat{\pi}_r)\Big),$$
with $V^\star_r$ replaced in practice by a task-specific expert's score, giving *expert-normalized return* rather than true regret.

**Latent world model.** Encoder $q_\phi(z_t \mid z_{t-1}, a_{t-1}, o_t)$, dynamics $p_\theta(z_t \mid z_{t-1},a_{t-1})$, trained by an ELBO $\mathcal{L}(\theta,\phi)$. The measured proxies for "knowledge gain":

- **Ensemble disagreement:** $u_t = \frac{1}{M}\sum_{i=1}^M \lVert f_{\theta_i}(z_t,a_t) - \bar{f}(z_t,a_t)\rVert_2^2$ over $M$ one-step heads (typically $M{=}5$–$10$).
- **Information gain:** $I(\Theta; z_{t+1}\mid z_t,a_t)$, never computed exactly; approximated by disagreement or by a variational KL $\mathbb{E}[\mathrm{KL}(q(\Theta\mid \mathcal{D}_{t+1})\Vert q(\Theta\mid\mathcal{D}_t))]$.
- **State entropy:** $\hat{H}(z)$ from a $k$-NN estimator, $\hat{H} \propto \sum_i \log \lVert z_i - z_i^{(k)}\rVert$, $k \approx 12$ over a minibatch of 512 latents.

The intrinsic reward $r^{\text{int}}$ is one of these, and the agent trains an actor by imagined rollouts of horizon $15$ inside $p_\theta$.

**Assumptions, and which break.**
1. *Ergodicity / resettability.* Assumed; violated wherever exploration is irreversible (a broken robot, a consumed resource).
2. *Deterministic or low-noise transitions.* Assumed by prediction-error and disagreement bonuses; violated by aleatoric noise — the noisy-TV failure, where $u_t$ stays high forever.
3. *$\mathcal{R}$ is expressible in the learned latent.* Assumed; violated when the reward depends on a variable the ELBO discarded as low-pixel-variance (a small object, a wrist force).
4. *Bounded model class containing $P$.* Assumed by every RFE theorem; false for pixel dynamics.
5. *Test tasks independent of the explorer's design.* Routinely violated — benchmarks and bonuses co-evolved.

## 3. State of the Art

**Theory SOTA (established).** Jin, Krishnamurthy, Simchowitz, Yu (ICML 2020) formalized RFE and gave $\tilde{O}(S^2AH^5/\varepsilon^2)$ episodes. Ménard et al. (RF-Express, ICML 2021) and Zhang, Du, Ji (ICML 2021) reduced this to $\tilde{O}(H^2SA/\varepsilon^2)$ up to logs, matching the $\Omega(H^2SA/\varepsilon^2)$ lower bound. Kaufmann et al. (ALT 2021) gave an anytime variant. These are proved, not merely claimed — but they are tabular, and the exponent on $S$ makes them vacuous for pixels.

**Empirical SOTA.** Plan2Explore (Sekar et al., ICML 2020) is the reference model-based method: latent disagreement over a one-step ensemble, planned in imagination, zero-shot and few-shot on DMC. Model-Based Active Exploration (Shyam et al., ICML 2019) planned directly for Jensen–Shannon divergence across an ensemble. LEXA (Mendonca et al., NeurIPS 2021) added goal-conditioned achievement. On the Unsupervised RL Benchmark (URLB, Laskin et al., NeurIPS D&B 2021: 12 tasks, 3 domains, 2M reward-free steps then 100k fine-tune steps), Rajeswar et al. (ICML 2023) showed Dreamer-based pretraining from pixels reaches near-expert on most tasks, and that *what you collect* and *what you train the model on* can be decoupled.

**Claimed but unablated.** (a) That intrinsic bonuses in these systems measure epistemic rather than aleatoric uncertainty — disagreement is a proxy whose gap to $I(\Theta;z')$ is not measured. (b) That reward-free pretraining transfers *across* domains, not just across tasks within a domain — URLB fine-tunes in the same embodiment it pretrained in. (c) Go-Explore (Ecoffet et al., *Nature* 2021, >43,000 on Montezuma's Revenge) is often cited as exploration SOTA, but its cell representation is hand-specified or downsampled-image based and its return phase uses the extrinsic reward; it is not reward-free in the sense here.

## 4. What Is Known

- **Tabular RFE is closed to log factors.** $\tilde{O}(H^2SA/\varepsilon^2)$ upper, $\Omega(H^2SA/\varepsilon^2)$ lower (Ménard et al. 2021; Zhang et al. 2021; Jin et al. 2020).
- **Pure curiosity gets far in sparse-reward Atari.** Burda et al. (ICLR 2019) trained on 54 environments with *no* extrinsic reward; a purely curiosity-driven agent on Montezuma's Revenge visited roughly 20 of 24 first-level rooms. RND with extrinsic reward exceeded average human score on that game (human ≈ 4,753).
- **Disagreement beats prediction error under noise.** Pathak et al. (ICML 2019) showed ensemble disagreement is invariant to aleatoric noise in expectation, unlike $\lVert \hat{s}'-s'\rVert^2$; verified on noisy-image Atari and on a real robot at ~10k interactions.
- **Model-based exploration is sample-efficient at DMC scale.** Plan2Explore ran 1M–2M reward-free steps on DeepMind Control and matched or approached a task-aware Dreamer given equal steps on several tasks; the gap is largest on tasks needing states rarely visited under any intrinsic bonus.
- **Scaling the model helps downstream, not obviously exploration.** DreamerV3 (Hafner et al., *Nature* 2025) shows monotone improvement with model size on fixed-reward tasks; no comparable curve exists for reward-free coverage.
- **Noisy-TV is real, not hypothetical.** Prediction-error agents provably stall at a stochastic observation source; Jarrett et al. (ICML 2023) reformulated curiosity in hindsight to remove it, measured on noise-injected Atari.

## 5. What Is Not Known

- **Theoretically open.** No RFE sample-complexity bound for latent-variable / nonlinear world models under realizability failure. Bounds exist for linear MDPs and low Bellman rank; none for a learned representation whose error is itself driven by the exploration policy — the coupling is unresolved.
- **Theoretically open.** Whether any single scalar intrinsic reward can be $\mathcal{R}$-universal. Intuition says no (the optimal exploration policy depends on $\mathcal{R}$), but there is no impossibility theorem for a natural class.
- **Empirically open.** Does reward-free pretraining pay off at $10^8$–$10^9$ steps and $\geq$1B-parameter models, or does the advantage vanish once the model is large enough to learn from near-random data? Runnable; unrun at that scale.
- **Empirically open.** Cross-embodiment transfer: pretrain reward-free in domain $A$, fine-tune in domain $B$. URLB does not test it.
- **Methodologically blocked.** There is no task-independent measure of dataset quality for a reward-free buffer. Every reported number is expert-normalized against a fixed suite, so "better exploration" and "better match to the suite" are not separable.

## 6. Why It Is Hard

The core obstruction is **confounded measurement compounded by non-identifiability of the uncertainty being maximized**.

1. *The evaluation does not measure what it names.* URLB-style scores measure downstream fit to 12 chosen tasks. An explorer tuned on those tasks and one that genuinely covers the state space produce the same number. Without a task-free coverage metric, the field cannot tell them apart.
2. *Epistemic and aleatoric uncertainty are not separable from finite data.* Disagreement $u_t$ shrinks with more data under noise-free dynamics and does not under noise, but at any finite $n$ the two contributions are not identifiable from samples alone — you cannot tell "I have not seen this" from "this is random" without a model-class assumption that pixel dynamics violate.
3. *Representation and exploration are coupled.* The latent $z$ is trained on data the exploration policy chose; a variable never varied is never represented, and a variable never represented cannot be sought. This is a fixed point, not a bug, and no analysis handles it.
4. *Compute.* A clean answer to the scaling question needs a matched-compute grid at $10^8$ steps across several embodiments — on the order of $10^4$ GPU-hours.

## 7. Current Research (as of 2026)

- **Video and action-conditioned generative world models as explorers** — Genie-style latent-action models (Bruce et al., ICML 2024) used to propose exploratory goals rather than to compute bonuses. *(frontier — verify)*
- **LLM-proposed goals as the reward-free objective**: a language model enumerates candidate goals, the world model scores achievability. Active at DeepMind, Meta AI, and academic groups (Berkeley, CMU). *(frontier — verify)*
- **Hindsight and distributional fixes to curiosity** under stochasticity (Jarrett et al. line, DeepMind).
- **Reward-free theory beyond tabular**: low-rank and general function approximation, block MDPs (Princeton, MSR, Cornell).
- **Coverage metrics independent of downstream tasks** — occupancy entropy in a *frozen, externally trained* representation. Small literature; the obvious unlock for §5's blocked item.

## 8. Concrete Next Experiment

**Question:** is the benefit of model-based reward-free exploration a coverage effect or a benchmark-fit effect?

- **Scale.** DMC + MetaWorld, 4 embodiments. Reward-free phase: 2M environment steps, DreamerV3-class model (~50M params), 5 seeds. Fine-tune: 100k steps per downstream task, 20 tasks, **10 of which are held out and were never used to tune any hyperparameter of any arm.**
- **Arms.** (1) Plan2Explore latent disagreement. (2) $k$-NN state entropy (APT-style). (3) **Control arm: random-action data collection, same 2M steps, same model, same fine-tune budget.** (4) Second control: random actions with $4\times$ the data (8M steps) — isolating whether exploration buys anything a cheap step budget cannot.
- **Deciding number.** Mean expert-normalized return on the **10 held-out tasks**, arm 1 minus arm 3. If the gap on held-out tasks is under **5 points** while the gap on the 10 tuned tasks exceeds 15, the reported advantage is benchmark fit, not exploration. If arm 4 erases the gap, exploration is buying step-efficiency only, worth exactly its compute ratio.
- Secondary readout: occupancy entropy of each buffer in a *frozen* pretrained visual encoder, correlated against held-out return. A correlation below $r = 0.5$ would confirm §5's methodological block.

## 9. Key References

- **[Foundational]** J. Schmidhuber. *A Possibility for Implementing Curiosity and Boredom in Model-Building Neural Controllers.* SAB, 1991.
- **[Foundational]** R. Houthooft, X. Chen, Y. Duan, J. Schulman, F. De Turck, P. Abbeel. *VIME: Variational Information Maximizing Exploration.* NeurIPS, 2016. — arXiv:1605.09674
- **[Foundational]** D. Pathak, P. Agrawal, A. Efros, T. Darrell. *Curiosity-driven Exploration by Self-supervised Prediction.* ICML, 2017. — arXiv:1705.05363
- **[SOTA, method]** R. Sekar, O. Rybkin, K. Daniilidis, P. Abbeel, D. Hafner, D. Pathak. *Planning to Explore via Self-Supervised World Models.* ICML, 2020. — arXiv:2005.05960
- **[SOTA, method]** P. Shyam, W. Jaśkowski, F. Gomez. *Model-Based Active Exploration.* ICML, 2019. — arXiv:1810.12162
- **[SOTA, theory]** C. Jin, A. Krishnamurthy, M. Simchowitz, T. Yu. *Reward-Free Exploration for Reinforcement Learning.* ICML, 2020. — arXiv:2002.02794
- **[SOTA, theory]** P. Ménard, O. D. Domingues, A. Jonsson, E. Kaufmann, E. Leurent, M. Valko. *Fast Active Learning for Pure Exploration in Reinforcement Learning.* ICML, 2021.
- **[SOTA, theory]** Z. Zhang, S. S. Du, X. Ji. *Nearly Minimax Optimal Reward-free Reinforcement Learning.* ICML, 2021.
- **[Benchmark]** M. Laskin, D. Yarats, H. Liu, K. Lee, A. Zhan, K. Lu, C. Cang, L. Pinto, P. Abbeel. *URLB: Unsupervised Reinforcement Learning Benchmark.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2110.15191
- **[SOTA, empirical]** S. Rajeswar, P. Mazzaglia, T. Verbelen, A. Piché, B. Dhoedt, A. Courville, A. Lacoste. *Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels.* ICML, 2023.
- **[Established result]** Y. Burda, H. Edwards, D. Pathak, A. Storkey, T. Darrell, A. Efros. *Large-Scale Study of Curiosity-Driven Learning.* ICLR, 2019. — arXiv:1808.04355
- **[Established result]** D. Pathak, D. Gandhi, A. Gupta. *Self-Supervised Exploration via Disagreement.* ICML, 2019. — arXiv:1906.04161
- **[Noise fix]** D. Jarrett, C. Tallec, F. Altché, T. Mesnard, R. Munos, M. Valko. *Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments.* ICML, 2023.
- **[Context]** D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. *Mastering Diverse Control Tasks through World Models.* Nature, 2025.
- **[Context]** A. Ecoffet, J. Huizinga, J. Lehman, K. O. Stanley, J. Clune. *First Return, Then Explore.* Nature, 2021.

## 10. Worked Example

**Setup.** A $10\times10$ gridworld, $H=20$, deterministic movement, plus one cell containing a "TV": entering it emits an observation drawn uniformly from $2^{16}$ images. Total $S=100$, $A=4$. Reward class $\mathcal{R}$: reach any single designated cell.

**Tabular theory says.** With $\tilde{O}(H^2SA/\varepsilon^2)$ episodes, $\varepsilon=0.1$, $H=20$, $S=100$, $A=4$: $20^2 \cdot 400 / 0.01 = 1.6\times10^7$ episodes before logs. Guaranteed $\varepsilon$-optimal for *every* $r\in\mathcal{R}$. The TV is harmless — a tabular count-based method visits it, counts saturate, the bonus $\propto 1/\sqrt{n}$ decays.

**Deep model does.** Replace the tabular model with a latent world model and a prediction-error bonus $r^{\text{int}}_t = \lVert f_\theta(z_t,a_t)-z_{t+1}\rVert^2$. On the TV cell, $z_{t+1}$ is drawn fresh each visit, so the residual converges to the observation variance $\sigma^2 > 0$ rather than to $0$. With planning horizon 15 and discount $\gamma = 0.99$, the imagined value of parking on the TV is
$$V^{\text{int}} \approx \sigma^2\frac{1-\gamma^{15}}{1-\gamma} \approx 13.9\,\sigma^2,$$
while any genuinely novel cell offers a bonus that decays to zero within ~50 visits. For $\sigma^2$ comparable to early-training residuals elsewhere, the argmax policy is *stay at the TV forever*. Coverage stalls at whatever fraction of the grid was reached before the TV was found — empirically the first 15–30 cells.

**Swap the bonus for ensemble disagreement** ($M=5$). Now each head fits the same conditional mean, so $u_t \to 0$ at the TV as $n$ grows, and coverage completes. This is the Pathak et al. 2019 result reproduced in miniature.

**The obstruction made visible.** Both agents are then scored on $\mathcal{R}_{\text{test}}$ = "reach one of 12 designated cells." If those 12 cells happen to sit in the region reachable before the TV, the prediction-error agent scores identically to the disagreement agent — expert-normalized return near 1.0 for both — while its state coverage differs by a factor of 5. The benchmark number does not see the failure. That is the measurement problem in §6, at a scale where the ground truth is countable: 100 states, and no reported metric in the standard evaluation protocol counts them.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*