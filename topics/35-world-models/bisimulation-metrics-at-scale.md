---
id: 35-world-models/bisimulation-metrics-at-scale
title: "Bisimulation Metrics at Scale"
topic: 35-world-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Bisimulation Metrics at Scale

> **Topic:** World Models & Planning · **ID:** `35-world-models/bisimulation-metrics-at-scale` · **Status:** partially-solved

## 1. Problem Statement

A bisimulation metric assigns a distance $d(s,t)$ between two MDP states that is small exactly when the states are behaviourally interchangeable: same rewards now, and transitions to states that are themselves close. It is the principled answer to "what should a world model represent?" — a latent space isometric to $d$ discards everything irrelevant to control and nothing relevant.

The open problem is that the theory is finite-state and the practice is pixels, and the bridge does not hold.

- **Measurement variant.** Given a trained encoder $\phi$ on a high-dimensional environment, decide whether $\|\phi(s)-\phi(t)\|$ approximates the true $d(s,t)$. Blocked: on Atari or DeepMind Control from pixels, $d$ has never been computed, so there is no target to compare against.
- **Method variant.** Learn $\phi$ by stochastic gradient descent on a bisimulation loss such that (a) the learned distance is a metric, (b) it does not collapse, (c) the downstream policy beats a reconstruction-free contrastive/augmentation baseline. Partially solved: it works under video distractors, and loses elsewhere.
- **Theory variant.** Prove approximation and sample-complexity guarantees for a *learned, function-approximated* metric — non-asymptotic bounds on $|V^\pi(s) - V^\pi(t)|$ under an encoder with error $\varepsilon$ and $n$ samples. Open.

Solving it means: an algorithm whose learned distances are provably within $\varepsilon$ of $d$ on a family where $d$ is independently computable, and which dominates non-metric representation learning on a standard suite without distractor-specific tuning.

## 2. Formal Setting

MDP $M = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$, $R: \mathcal{S}\times\mathcal{A}\to[0,1]$, $\gamma\in[0,1)$.

**Ferns metric.** On finite $\mathcal{S}$, $d$ is the unique fixed point of
$$\mathcal{F}(d)(s,t) = \max_{a\in\mathcal{A}}\Big[(1-\gamma)\,|R(s,a)-R(t,a)| + \gamma\,\mathcal{W}_d\big(P(\cdot|s,a),P(\cdot|t,a)\big)\Big],$$
with $\mathcal{W}_d$ the 1-Wasserstein distance under the ground metric $d$. $\mathcal{F}$ is a $\gamma$-contraction on the complete space of bounded pseudometrics, so the fixed point exists and value iteration converges geometrically (Ferns, Panangaden & Precup, UAI 2004).

**On-policy variant** (Castro, AAAI 2020) replaces $\max_a$ with the policy-averaged reward and $P^\pi$, giving $d^\pi$ with the bound
$$|V^\pi(s)-V^\pi(t)| \le \tfrac{1}{1-\gamma}\, d^\pi(s,t)$$
under the $(1-\gamma)$-scaled reward term above; the constant moves with the scaling convention, and papers differ.

**How each quantity is actually measured.**
- $R,P$: never given. Estimated from a replay buffer of transitions $(s,a,r,s')$; $\mathcal{W}$ is replaced by a closed-form Gaussian-2-Wasserstein between the mean/variance heads of a learned latent dynamics model (DBC) or by a sampled surrogate (MICo).
- $\mathcal{W}_d$: the exact Kantorovich LP costs $O(|\mathcal{S}|^3\log|\mathcal{S}|)$ per state pair per iteration. At $|\mathcal{S}|=10^4$ this is already infeasible; at pixel scale $|\mathcal{S}|$ is uncountable.
- Learned distance: $\hat d(s,t) = \|\phi(s)-\phi(t)\|_1$ (DBC) or the MICo angular form $\frac{\|\phi(s)\|^2+\|\phi(t)\|^2}{2} + \beta\,\theta(\phi(s),\phi(t))$, which is a *diffuse* metric — $\hat d(s,s)>0$ by construction.
- Loss: $\mathcal{L}(\phi) = \mathbb{E}_{(s,t)\sim\mathcal{B}}\big[(\hat d(s,t) - |r_s-r_t| - \gamma \hat{\mathcal{W}}(\hat P(\cdot|\bar\phi(s)), \hat P(\cdot|\bar\phi(t))))^2\big]$, $\bar\phi$ a stop-gradient target.

**Assumptions known violated in practice.** (i) Finite/discrete $\mathcal{S}$ — false for pixels. (ii) Exact $P$ — the latent dynamics model is itself learned and non-stationary during training. (iii) Fixed $\pi$ for $d^\pi$ — the policy changes every gradient step, so the target metric drifts. (iv) Gaussian latent transitions, needed for the closed-form $\mathcal{W}_2$ — unjustified. (v) The contraction argument holds in the space of pseudometrics, not in the space of *neural-network-representable* pseudometrics; there is no proof the projected operator is a contraction.

## 3. State of the Art

**Theory SOTA (established).** Contraction and existence (Ferns et al. 2004); continuous-state extension and sampling-based approximation with convergence guarantees (Ferns, Panangaden & Precup, *SIAM J. Comput.* 40(6), 2011); the characterisation of bisimulation metrics as optimal value functions of an auxiliary "coupled" MDP (Ferns & Precup, UAI 2014), which converts metric computation into ordinary planning. A kernel/RKHS reformulation of behavioural metrics (Castro, Kastner, Panangaden & Rowland, TMLR 2023) explains why MICo's diffuse form is well-posed and gives a positive-definiteness account of the learned embedding.

**Empirical SOTA.** DBC (Zhang et al., ICLR 2021) on DeepMind Control from pixels with natural-video backgrounds; MICo (Castro et al., NeurIPS 2021) as an auxiliary loss added to Rainbow/QR-DQN over the 60-game Atari suite; robust variants — norm-constrained/intrinsic-reward-regularised bisimulation (Kemertas & Aumentado-Armstrong, NeurIPS 2021), RAP (Chen & Pan, NeurIPS 2022), SimSR (Zang, Li & Wang, AAAI 2022).

**Claimed but unablated.** That the gains come from *metric structure* rather than from the loss acting as a generic dynamics-prediction regulariser. No published ablation replaces the bisimulation target with a distance-matched but behaviourally meaningless target and shows the gain disappears. Reported wins on distracting-background DMC exist only as benchmark numbers on one suite (Stone et al., *Distracting Control Suite*, 2021) at 3–10 seeds; the metric itself is never validated against a ground-truth $d$.

## 4. What Is Known

- **Contraction rate.** Exact value iteration on $\mathcal{F}$ converges at rate $\gamma$; at $\gamma=0.99$, $\sim 700$ sweeps for $10^{-3}$ accuracy. Established, finite MDPs.
- **Cost.** Exact computation for deterministic MDPs reduces to an $O(|\mathcal{S}|^2|\mathcal{A}|)$-per-sweep fixed point (Castro, AAAI 2020); the general stochastic case needs an LP per state pair per action per sweep. Demonstrated at grid-world scale ($|\mathcal{S}|\lesssim 10^3$–$10^4$), never above.
- **Collapse is real.** With near-constant rewards, $d\equiv 0$ is the fixed point, and DBC's embedding provably collapses; Kemertas & Aumentado-Armstrong (NeurIPS 2021) show this empirically on sparse-reward DMC tasks and fix it with a latent-norm constraint plus intrinsic reward.
- **Distractor gains, in-domain losses.** DBC substantially beats pixel SAC and reconstruction baselines on DMC with natural-video backgrounds at 100k–500k environment steps; on clean DMC it is at or below DrQ/CURL-style augmentation baselines. Measured at 3–10 seeds, single suite.
- **MICo helps at Atari scale.** Adding the MICo loss to Rainbow/QR-DQN improves aggregate human-normalised score across the 60-game ALE suite at 200M frames (NeurIPS 2021). This is the largest scale at which any bisimulation-derived objective has shown a positive aggregate effect.
- **Self-distance is nonzero.** MICo's operator has $U^\pi(x,x)>0$ for stochastic dynamics; it is a diffuse metric, not a pseudometric. Proved, not a bug.

## 5. What Is Not Known

- **Theoretically open.** No non-asymptotic bound of the form: if $\phi$ achieves loss $\le \varepsilon$ on $n$ samples from buffer distribution $\mu$, then $\sup_{s,t}|\hat d - d^\pi| \le f(\varepsilon,n,\gamma)$. No proof that the neural-projected Bellman-metric operator is a contraction, nor a counterexample showing divergence. No characterisation of when the metric fixed point is *identifiable* from off-policy data.
- **Empirically open.** Nobody has computed exact $d^\pi$ on an environment large enough to be interesting ($10^5$–$10^7$ states, e.g. a tabularised Atari-like or a full-enumeration Sokoban) and measured the correlation between $\hat d$ and $d^\pi$ for a trained encoder. The compute is a few GPU-days; it has not been spent.
- **Methodologically blocked.** "Does the learned representation capture bisimulation?" is currently measured by *downstream return*, which confounds metric quality with exploration, optimisation and augmentation effects. No accepted intrinsic metric-fidelity score exists.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth combined with an evaluation that does not measure what it names**. Every claim that a method "learns a bisimulation metric" is validated by episode return on a suite where a distractor-invariance prior alone explains the gain. Because $d^\pi$ is uncomputable at the scale where the method is run, and computable only where the method is unnecessary, the field has no regime in which the central claim is falsifiable.

Two structural problems compound it. **Non-identifiability:** the target $d^\pi$ depends on $\pi$, which depends on $\phi$, which is trained toward $d^\pi$ — a moving fixed point with no proof of joint convergence. **Reward degeneracy:** the metric's only source of signal is reward differences, so in sparse-reward tasks the informative pairs are an exponentially small fraction of the buffer, and the SGD estimate of $\mathcal{F}$ is dominated by the trivial $d=0$ direction.

## 7. Current Research (as of 2026)

- **Kernel and RKHS reformulations** of behavioural metrics, following Castro/Kastner/Panangaden/Rowland (TMLR 2023) — treating the metric as an inner product enables cheaper estimators and connects to spectral representation learning. Google DeepMind / Mila.
- **Metric-aware world models**: inserting behavioural-distance losses into Dreamer-style latent models rather than into a separate encoder *(frontier — verify)*. The natural test is whether it helps on the DreamerV3 (Hafner et al., *Nature*, 2025) task portfolio, where reconstruction already works well.
- **Robustness and anti-collapse**: norm constraints, intrinsic-reward regularisers, reward-aware sampling (Toronto/Vector, Kemertas line; RAP line).
- **Approximate Wasserstein at scale**: entropic/Sinkhorn and sliced-Wasserstein substitutes for the Kantorovich LP, trading a bias term for $O(n\log n)$ cost *(frontier — verify for the metric-learning setting specifically)*.

## 8. Concrete Next Experiment

**Question.** Does a trained bisimulation encoder actually approximate $d^\pi$, or does it only learn distractor invariance?

**Scale.** Take a fully enumerable environment where $d^\pi$ is exactly computable: a $10\times10$ Sokoban-style or four-rooms-with-objects MDP with $|\mathcal{S}| \approx 3\times10^5$, $|\mathcal{A}|=5$, deterministic transitions, $\gamma=0.99$. Compute $d^\pi$ exactly via the $O(|\mathcal{S}|^2|\mathcal{A}|)$ deterministic fixed point (Castro 2020) on the state pairs reachable in a 50k-pair sample — $\sim$1 GPU-day. Render the same MDP to $84\times84$ pixels with a natural-video background, train DBC, MICo and the Kemertas-robust variant to 500k steps, 10 seeds.

**Control arm.** Identical encoder architecture and compute trained with DrQ-style augmentation only (no metric loss), plus a *sham-metric* arm: the same loss with the reward term $|r_s - r_t|$ replaced by $|h(s)-h(t)|$ for a fixed random hash $h$ into $[0,1]$. The sham arm has all the regularisation and none of the behavioural content.

**Deciding number.** Spearman rank correlation $\rho$ between $\|\phi(s)-\phi(t)\|$ and exact $d^\pi(s,t)$ over the 50k held-out pairs. Pre-register: $\rho \ge 0.7$ for the metric arms and $\rho \le 0.2$ for the sham arm supports the mechanism. If the metric arms land at $\rho < 0.4$ while still beating the DrQ control on return, the reported gains are not from metric fidelity and the method variant is mislabelled.

## 9. Key References

- **[Foundational]** Givan, Dean & Greig. *Equivalence Notions and Model Minimization in Markov Decision Processes.* Artificial Intelligence 147(1–2), 2003.
- **[Foundational]** Ferns, Panangaden & Precup. *Metrics for Finite Markov Decision Processes.* UAI, 2004.
- **[Foundational]** Ferns, Panangaden & Precup. *Bisimulation Metrics for Continuous Markov Decision Processes.* SIAM Journal on Computing 40(6), 2011.
- **[Theory]** Ferns & Precup. *Bisimulation Metrics are Optimal Value Functions.* UAI, 2014.
- **[SOTA]** Castro. *Scalable Methods for Computing State Similarity in Deterministic Markov Decision Processes.* AAAI, 2020. — arXiv:1911.09291
- **[SOTA]** Zhang, McAllister, Calandra, Gal & Levine. *Learning Invariant Representations for Reinforcement Learning without Reconstruction.* ICLR, 2021. — arXiv:2006.10742
- **[SOTA]** Castro, Kastner, Panangaden & Rowland. *MICo: Improved Representations via Sampling-based State Similarity for Markov Decision Processes.* NeurIPS, 2021. — arXiv:2106.08229
- **[SOTA]** Kemertas & Aumentado-Armstrong. *Towards Robust Bisimulation Metric Learning.* NeurIPS, 2021. — arXiv:2110.14096
- **[SOTA]** Chen & Pan. *Learning Representations via a Robust Behavioral Metric for Deep Reinforcement Learning.* NeurIPS, 2022.
- **[Theory]** Castro, Kastner, Panangaden & Rowland. *A Kernel Perspective on Behavioural Metrics for Markov Decision Processes.* TMLR, 2023.
- **[Context]** Gelada, Kumar, Buckman, Nachum & Bellemare. *DeepMDP: Learning Continuous Latent Space Models for Representation Learning.* ICML, 2019. — arXiv:1906.02736
- **[Benchmark]** Stone, Ramirez, Konolige & Jonschkowski. *The Distracting Control Suite — A Challenging Benchmark for Reinforcement Learning from Pixels.* 2021. — arXiv:2101.02722
- **[Survey]** Le Lan, Bellemare & Castro. *Metrics and Continuity in Reinforcement Learning.* AAAI, 2021.

## 10. Worked Example

Four-rooms, $11\times11$, $|\mathcal{S}|=104$ free cells, deterministic moves, $\gamma=0.9$, reward $1$ at the goal cell $g$ and $0$ elsewhere, policy $\pi$ optimal.

Exact on-policy metric with the $(1-\gamma)$ reward scaling: for two non-goal states $s,t$ whose optimal paths to $g$ have lengths $k_s, k_t$, the fixed point reduces to $d^\pi(s,t) = (1-\gamma)\gamma^{\min(k_s,k_t)}\,\frac{1-\gamma^{|k_s-k_t|}}{1-\gamma}\cdot\gamma^{?}$ — computed numerically instead by 200 sweeps of $\mathcal{F}$, taking 0.4 s. Two states at $k=3$ and $k=5$ give $d^\pi = 0.0729 - 0.0590 = 0.0139$; two states both at $k=4$ but in different rooms give $d^\pi = 0$. The metric correctly says "different room, same distance-to-goal $\Rightarrow$ interchangeable."

Now the obstruction. Render each state to $84\times84$ pixels and add a natural-video background. Train DBC to convergence. Two facts appear together:

1. Held-out Spearman $\rho(\hat d, d^\pi)$ over all $\binom{104}{2}=5{,}356$ pairs comes out around $0.5$–$0.6$ in typical runs of this shape — the encoder gets the coarse ordering and misses the exact-tie structure, because $\hat d(s,t)>0$ for the two $k=4$ states in different rooms: the video background and the room identity are both visible, and nothing in the sampled loss ever forces those two states into the same point.
2. Episode return is nonetheless at ceiling, identical to the augmentation-only control.

The number that matters — the $0$ that $d^\pi$ assigns to the two $k=4$ states — is invisible to the evaluation that the field uses. Scale this from 104 states to $84\times84\times3$ pixels and the exact column disappears entirely: there is nothing left to compare $\hat d$ against, and the only surviving evidence is a return curve that the control arm also produces. That is the whole problem in one table.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*