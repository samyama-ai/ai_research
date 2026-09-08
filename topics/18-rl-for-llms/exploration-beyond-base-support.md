---
id: 18-rl-for-llms/exploration-beyond-base-support
title: "Exploration Beyond the Base Model Support"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Exploration Beyond the Base Model Support

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/exploration-beyond-base-support` · **Status:** open

## 1. Problem Statement

RL with verifiable rewards (RLVR) reliably raises single-sample accuracy of an LLM on math, code and logic tasks. The open question is whether it adds *new* solvable problems, or only reweights solutions the base model could already produce given enough samples.

- **Input:** a base (pre-trained or SFT'd) policy $\pi_0$, a task distribution $\mathcal{D}$, a verifier $r:\mathcal{X}\times\mathcal{Y}\to\{0,1\}$, a compute budget.
- **Output:** a fine-tuned policy $\pi_\theta$.
- **Decision predicate:** does there exist a problem $x$ that $\pi_\theta$ solves with non-negligible probability but that $\pi_0$ fails to solve within any sampling budget the experimenter can afford?

Three variants, with different difficulty:

- **Measurement.** Define "beyond the base support" operationally. Under a softmax head every finite string has positive probability, so literal support is always full; the meaningful object is the set of solutions reachable at feasible sample budget $k$. Whether pass@$k$ at large $k$ is the right instrument is itself contested.
- **Method.** Build an RL procedure that provably adds reachable solutions — via exploration bonuses, off-policy data, tool feedback, curriculum, or self-generated tasks.
- **Theory.** Characterize which policy-optimization operators can and cannot move mass onto sequences with vanishing base probability, under realistic gradient estimators and KL regularization.

Solving it means: a demonstration, with a matched-compute base-sampling control, that some capability exists post-RL and is not extractable from $\pi_0$ at equal inference cost — plus a mechanism that explains why.

## 2. Formal Setting

Autoregressive policy $\pi_\theta(y\mid x)=\prod_{t}\pi_\theta(y_t\mid x,y_{<t})$ over token sequences. RLVR objective:

$$J(\theta)=\mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\big]-\beta\,\mathbb{E}_x\big[\mathrm{KL}(\pi_\theta(\cdot\mid x)\,\|\,\pi_0(\cdot\mid x))\big].$$

**Quantities, as measured.**

- **Coverage / pass@$k$**, estimated unbiasedly from $n\ge k$ samples with $c$ correct (Chen et al., 2021):
  $$\widehat{\text{pass@}k}(x)=1-\binom{n-c}{k}\Big/\binom{n}{k}.$$
  Measured at fixed decoding settings; temperature $T$ and nucleus $p$ must be swept per model, since RL models are typically tuned at lower entropy than base.
- **Reachable set at budget $k$:** $S_k(\pi)=\{x: \text{pass@}k(x)>0\}$ under a fixed decoder. "Beyond support" is the claim $S_k(\pi_\theta)\setminus S_{k'}(\pi_0)\neq\emptyset$ for $k\ll k'$ — an *asymmetric* comparison that gives the base a much larger budget.
- **Base-likelihood of RL outputs:** $\ell = -\frac{1}{|y|}\log \pi_0(y\mid x)$ for $y\sim\pi_\theta$, versus $y\sim\pi_0$. If RL outputs are not atypical under $\pi_0$, the reweighting hypothesis is supported.
- **Policy entropy:** $H=-\mathbb{E}_{x}\mathbb{E}_{y\sim\pi_\theta}\frac{1}{|y|}\sum_t\log\pi_\theta(y_t\mid \cdot)$, measured on held-out prompts at $T=1$.

**Assumptions and their violations.**

1. *Verifier is sound.* Violated: string-match graders accept wrong reasoning with right answers; ~5–10% false-positive rates are typical on free-form math.
2. *Rewards are informative.* Violated: random and format-only rewards move Qwen2.5-Math benchmark scores substantially (§3), so reward signal is not identified from score gains.
3. *Benchmarks are uncontaminated.* Violated for Qwen-family models on MATH/AIME-era sets.
4. *pass@$k$ measures capability.* Partly violated: large-$k$ estimates are dominated by guessable-answer problems, especially multiple-choice and small-integer answers.
5. *On-policy sampling.* Violated by replay buffers, clipping, and multi-epoch reuse in GRPO/PPO implementations.

## 3. State of the Art

**Theory SOTA (established).** The KL-regularized optimum is an exponential tilt of the base, $\pi^\star(y\mid x)\propto\pi_0(y\mid x)e^{r(x,y)/\beta}$ — absolutely continuous w.r.t. $\pi_0$, so it *cannot* place mass where $\pi_0$ places none (Korbak, Perez, Buckley, EMNLP Findings 2022; same identity underlies DPO, Rafailov et al., NeurIPS 2023). Policy-gradient convergence rates carry a distribution-mismatch coefficient $\|d^{\pi^\star}/\mu\|_\infty$ that blows up when the optimal trajectory distribution is not covered by the sampling distribution (Agarwal, Kakade, Lee, Mahajan, *JMLR* 2021). Both are about the idealized objective, not the finite-sample estimator; neither forbids off-support movement through generalization in $\theta$-space.

**Empirical SOTA — established.** Yue et al. (arXiv:2504.13837, 2025) compare RLVR models against their base across Qwen-2.5 (7B/14B/32B) and LLaMA-3.1-8B on math, code and visual reasoning: RL wins at $k=1$, base catches up and *overtakes* by $k\approx 128$–$256$. RL outputs are high-likelihood under the base. Distillation, by contrast, does expand the boundary. Independently, "Spurious Rewards" (Shao, Li, et al., 2025) shows random or format rewards give large MATH-500 gains on Qwen2.5-Math-7B but near-zero on LLaMA — i.e. the gain elicits a pre-existing prior.

**Empirical SOTA — claimed but unablated.** ProRL (Liu et al., NVIDIA, arXiv:2505.24864, 2025) reports that >2k RL steps over ~136k diverse problems yields pass@128 gains on tasks where the base scores 0, with Nemotron-Research-Reasoning-Qwen-1.5B. This is the strongest counterexample on record, but the base arm is not matched on data-exposure and the "base scores 0" tasks are mostly logic puzzles with narrow output formats. Treat as a benchmark number, not a settled ablation. Similarly, "The Invisible Leash" (Wu et al., arXiv:2507.14843, 2025) argues RLVR is support-constrained and entropy-reducing — an analysis plus small-scale evidence, not a proof.

## 4. What Is Known

- **Single-sample gains are large and real.** DeepSeek-R1-Zero: AIME 2024 pass@1 15.6% → 71.0% under pure RL from DeepSeek-V3-base; 86.7% with majority vote at 64 samples (DeepSeek-AI, *Nature*, 2025).
- **The pass@$k$ crossover.** Across ~6 model families and multiple benchmarks, RL-trained models' coverage curves cross below base curves somewhere in $k\in[32,512]$ (Yue et al., 2025), at 7B–32B scale.
- **Entropy collapses predictably.** Cui et al. (2025) fit $R\approx -a\,e^{H}+b$ across model scales (0.5B–32B) — downstream reward is an exponential function of policy entropy, and RL runs saturate as $H\to 0$. Performance ceiling is largely set at initialization.
- **RL updates are sparse in parameters.** RL finetuning updates a small subnetwork (reported ~5–30% of parameters), unlike SFT — consistent with reweighting rather than new-skill acquisition (Mukherjee et al., 2025).
- **Distillation expands the boundary.** Distilling a stronger teacher raises pass@$k$ at *all* $k$, unlike RLVR (Yue et al., 2025) — establishing that the ceiling is not an artifact of the pass@$k$ instrument.

## 5. What Is Not Known

- **Theoretically open.** Whether an on-policy gradient estimator with function approximation can, in finite steps, raise the probability of a sequence with base probability below sampling resolution. The KL-tilt argument covers the population optimum; it says nothing about generalization across $\theta$ — parameter-space updates can lift unsampled sequences that share structure with sampled ones. No theorem either way.
- **Empirically open.** Whether ProRL-style prolonged training with high task diversity produces boundary expansion at 7B+ scale under a *matched-inference-compute* base control. Runnable today; nobody has run it with the right control arm.
- **Methodologically blocked.** "Support" itself. Coverage at budget $k$ is decoder-dependent and guessing-contaminated; there is no accepted capability measure invariant to temperature, prompt format, and answer-space size. Until that exists, crossover points are not comparable across papers.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** pass@$k$ conflates capability with answer-space guessability. On AIME (integers 0–999), a base model at $T=1.2$ with $k=256$ accumulates hits by near-random answer emission with an unverifiable derivation. The crossover may be an artifact of the base model's higher entropy, not its wider capability.
2. **Non-identifiability of the reward's contribution.** Spurious-reward results mean a score gain does not identify *what* the RL signal taught. Any positive result must rule out elicitation of a pretraining prior — which requires a base model whose pretraining data you control.
3. **Compute cost of the control arm.** The honest control is base sampling at the RL model's *total* training-plus-inference compute. ProRL-scale RL is $10^3$+ GPU-days; the matched base control at $k=10^4$ per problem over 500 problems is another large bill. Almost no paper pays for both.

## 7. Current Research (as of 2026)

- **Prolonged / high-diversity RL.** NVIDIA (ProRL line) and follow-ups test whether step count and task breadth break the ceiling. *(frontier — verify)* Reported extensions to 7B+ are not yet independently reproduced.
- **Entropy engineering.** Clip-higher (DAPO, ByteDance/Tsinghua 2025), entropy bonuses, covariance-targeted regularizers (Cui et al.), KL-free objectives. Aim: delay collapse without reward hacking.
- **Off-support data injection.** Interleaving SFT/distillation with RL, self-play task generation (Absolute Zero, Zhao et al., 2025), and tool-augmented rollouts where the environment supplies tokens the model could not generate.
- **Instrument reform.** Replacing pass@$k$ with verifier-strict, guess-resistant metrics and with $\epsilon$-coverage under fixed sampling temperature. *(frontier — verify)*
- **Groups:** Tsinghua LeapLab / Shanghai Jiao Tong (Yue et al.), NVIDIA ADLR, Allen Institute + UW (Tulu/spurious rewards), DeepSeek, ByteDance Seed.

## 8. Concrete Next Experiment

**Question:** does RLVR add problems the base cannot reach at equal inference compute?

- **Scale.** One 7B base model with a *published, searchable* pretraining corpus (OLMo-2-7B), so contamination is checkable. Two arms, 8×H100 for ~2 weeks each.
- **Treatment arm.** RLVR (GRPO) for 2,000 steps on 100k+ diverse verifiable tasks (math, code, logic puzzles, constraint satisfaction), ProRL recipe including reference-policy resets.
- **Control arm.** Base model, best-of-$k$ sampling with the *same verifier*, temperature swept over $\{0.6,0.8,1.0,1.2\}$, $k$ chosen so that base inference FLOPs equal RL training + RL inference FLOPs. At 7B this is roughly $k\approx 10^4$ per problem on a 200-problem held-out set.
- **Contamination filter.** Drop every eval problem with an $n$-gram match in the pretraining corpus.
- **Guess filter.** Require a second-stage LLM-judge check that the derivation entails the answer; discard lucky hits.

**Deciding number:** $N_{\text{new}}$ = count of held-out problems with $\text{pass@}1(\pi_\theta)\ge 0.1$ *and* zero verified base successes in $10^4$ samples. $N_{\text{new}}=0$ (of 200) supports the reweighting hypothesis. $N_{\text{new}}\ge 10$ with the guess filter applied is the first clean evidence of boundary expansion. Report the 95% Clopper–Pearson interval; anything in between is a null result, not a win.

## 9. Key References

- **[Foundational]** Chen, M., Tworek, J., Jun, H., et al. *Evaluating Large Language Models Trained on Code.* arXiv, 2021. — arXiv:2107.03374 (defines pass@$k$)
- **[Foundational]** Agarwal, A., Kakade, S. M., Lee, J. D., Mahajan, G. *On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift.* JMLR 22(98), 2021.
- **[Foundational]** Korbak, T., Perez, E., Buckley, C. L. *RL with KL Penalties is Better Viewed as Bayesian Inference.* Findings of EMNLP, 2022. — arXiv:2205.11275
- **[SOTA]** Yue, Y., Chen, Z., Lu, R., et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* arXiv, 2025. — arXiv:2504.13837
- **[SOTA]** Liu, M., Diao, S., Lu, X., et al. *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models.* NVIDIA, arXiv, 2025. — arXiv:2505.24864
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025.
- **[SOTA]** Shao, R., Li, S. S., Xin, R., et al. *Spurious Rewards: Rethinking Training Signals in RLVR.* arXiv, 2025. — arXiv:2506.10947
- **[SOTA]** Cui, G., Zhang, Y., Chen, J., et al. *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.* arXiv, 2025. — arXiv:2505.22617
- **[Survey]** Wu, F., Wang, S., Bai, Y., et al. *The Invisible Leash: Why RLVR May Not Escape Its Origin.* arXiv, 2025. — arXiv:2507.14843
- **[Foundational]** Rafailov, R., Sharma, A., Mitchell, E., et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS, 2023. — arXiv:2305.18290

## 10. Worked Example

Take AIME 2024, 30 problems, answers integers in $[0,999]$. Base Qwen2.5-Math-7B at $T=1.0$: pass@1 ≈ 0.13. RL-trained variant: pass@1 ≈ 0.40 (typical reported range). Coverage crossover reported near $k\approx128$.

Now price the guessing floor. Suppose the base, on a problem it cannot solve, emits a plausible-looking integer roughly uniformly over ~50 candidate values it considers. Then per-sample hit probability $\approx 0.02$, and

$$\text{pass@}256 \ge 1-(1-0.02)^{256}=0.99.$$

At $k=256$ the base "solves" essentially every unsolved problem by coincidence. Even at $k=32$, $1-0.98^{32}=0.47$. So the crossover at $k=128$ is exactly where the guessing floor saturates — the measurement and the artifact live at the same budget.

The obstruction is now visible: the headline finding ("base beats RL at large $k$") and the null artifact ("wide-entropy sampling brute-forces a 3-digit answer space") are not separable by pass@$k$ alone. Distinguishing them requires per-sample derivation checking on order $30\times256\approx 7{,}700$ traces — cheap. Doing the same at the $k=10^4$ control budget of §8 is $2\times10^6$ judge calls per arm. That is the real cost of settling the question, and it is why the question is open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*