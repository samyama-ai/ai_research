---
id: 18-rl-for-llms/critic-free-baseline-variance-llm-rl
title: "Baseline Variance Reduction Without a Learned Critic"
topic: 18-rl-for-llms
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Baseline Variance Reduction Without a Learned Critic

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/critic-free-baseline-variance-llm-rl` · **Status:** partially-solved

## 1. Problem Statement

Post-training an LLM with policy-gradient RL requires an advantage estimate. PPO-style pipelines learn a value network of roughly the policy's size; this doubles memory and adds a second optimization problem that is itself unstable at LLM scale. Critic-free methods (GRPO, RLOO, ReMax) replace the learned value function with a **statistic of $G$ sampled completions to the same prompt**. The question is how much variance that trade actually costs, and whether a better critic-free baseline exists.

Three variants, which are routinely conflated:

- **Measurement.** Given a fixed policy $\pi_\theta$, prompt distribution, and sampling budget, *measure* the variance of the gradient estimator under each baseline. Nobody publishes this; papers publish downstream accuracy instead.
- **Method.** Find the baseline $b$, computable from $G$ i.i.d. rollouts and no extra parameters, that minimizes estimator variance per unit compute. "Per unit compute" is essential: a $G$-sample group baseline costs $G$ generations, and generation dominates the step.
- **Theory.** Characterize the variance gap between the best critic-free baseline and the optimal state-dependent baseline $b^\star(s)$, as a function of $G$, reward sparsity, and the credit-assignment granularity (sequence-level vs. token-level).

Solved would mean: a baseline rule with a proved variance bound relative to $b^\star$, plus a measured gradient-SNR curve at $\geq 7$B showing the predicted ordering and matching the downstream accuracy ordering.

## 2. Formal Setting

A prompt $x \sim \mathcal{D}$, a completion $y = (y_1,\dots,y_T)$ sampled from $\pi_\theta(\cdot\mid x)$, and a terminal (verifier) reward $R(x,y) \in \{0,1\}$ for math/code, or a scalar reward-model score. The objective is $J(\theta) = \mathbb{E}_{x,y}[R(x,y)]$, with

$$\nabla_\theta J = \mathbb{E}\Big[\big(R(x,y) - b(x)\big)\sum_{t=1}^{T}\nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\Big].$$

Any $b$ that does not depend on $y$ leaves the estimator unbiased.

**Baselines as measured in code.** With $G$ completions $y^{(1..G)}$ per prompt and $R_i = R(x,y^{(i)})$:

$$b_{\text{GRPO}} = \frac{1}{G}\sum_{j} R_j, \qquad A_i^{\text{GRPO}} = \frac{R_i - b_{\text{GRPO}}}{\mathrm{std}(R_{1..G}) + \epsilon}, \qquad b^{(i)}_{\text{RLOO}} = \frac{1}{G-1}\sum_{j\neq i} R_j .$$

RLOO's per-sample baseline is independent of $y^{(i)}$, so $A^{\text{RLOO}}$ is unbiased. GRPO's is not: including $R_i$ in the mean shrinks each advantage by exactly $\frac{G-1}{G}$ in expectation (a scale effect, absorbable into the learning rate) and the $1/\mathrm{std}$ term introduces a genuine, non-absorbable prompt-level reweighting.

**Optimal scalar baseline** (Weaver & Tao, UAI 2001; Greensmith et al., JMLR 2004): with $g = \sum_t \nabla_\theta\log\pi_\theta(y_t\mid\cdot)$,

$$b^\star(x) = \frac{\mathbb{E}[\|g\|^2 R]}{\mathbb{E}[\|g\|^2]} \neq \mathbb{E}[R] \text{ in general}.$$

The mean-reward baseline is optimal only when $\|g\|^2 \perp R$ — false for LLMs, where correct completions are systematically shorter and $\|g\|^2$ grows with $T$.

**The quantity to measure.** For parameter block $k$ and $N$ prompts, the per-step signal-to-noise ratio

$$\mathrm{SNR}_k = \frac{\|\mathbb{E}[\hat g_k]\|^2}{\mathrm{tr}\,\mathrm{Cov}(\hat g_k)},$$

estimated by splitting a large batch into $M$ disjoint microbatches and comparing between- and within-microbatch variance. This is a two-minibatch measurement, not a theoretical object.

**Assumptions known to be violated.** (i) Rewards i.i.d. within a group — violated once a shared prefix or shared seed is used. (ii) $b$ independent of the sample — violated by GRPO's mean and its std normalization. (iii) Stationary $\pi_\theta$ within a step — violated by multi-epoch inner loops with clipping. (iv) Reward bounded and noiseless — violated by reward models, and by verifiers with false negatives on formatting.

## 3. State of the Art

**Established.**
- *RLOO* (Ahmadian et al., ACL 2024) shows on 6.9B Pythia and 7B LLaMA that leave-one-out REINFORCE with $k=2$–$4$ matches or beats PPO on RLHF win-rate at lower memory, and that PPO's value network and GAE are largely unnecessary in the single-terminal-reward regime.
- *GRPO* (Shao et al., DeepSeekMath, 2024) is the systems SOTA by adoption: group-mean baseline, no critic, used in DeepSeek-R1 and most open reasoning-RL stacks.
- *Dr. GRPO* (Liu et al., 2025) identifies two biases in GRPO — division by group std, and length normalization — and shows removing them fixes an inflation of response length on wrong answers.
- *The Mirage of Action-Dependent Baselines* (Tucker et al., ICML 2018) is the cautionary precedent: several published "variance-reducing" baselines gave their gains from implementation artifacts, not variance; measured variance was unchanged.

**Claimed but unablated.** That GRPO's std normalization improves optimization (it is a per-prompt learning-rate rescaling, difficulty-weighted, and no paper isolates it with a variance measurement). That larger $G$ buys accuracy monotonically. That token-level vs. sequence-level advantage assignment matters for variance rather than for length bias.

**Benchmark-number-only results.** DAPO (Yu et al., 2025) reports 50 on AIME 2024 with Qwen2.5-32B at about half the steps of a DeepSeek-R1-Zero-style baseline, bundling four changes (clip-higher, dynamic sampling, token-level loss, overlong shaping); the attribution across them is by ablation on one benchmark, not by variance measurement. GSPO (Zheng et al., Qwen, 2025) reports stability gains from sequence-level importance ratios, again as downstream curves.

## 4. What Is Known

- **Unbiasedness.** RLOO's estimator is exactly unbiased; GRPO's unnormalized version is unbiased up to the constant $\frac{G-1}{G}$; the std-normalized version is biased in a way that does not vanish as $G\to\infty$ (Liu et al., 2025).
- **Variance floor.** For a Bernoulli reward with prompt success rate $p$, the group-mean baseline gives per-sample advantage variance $p(1-p)\cdot\frac{G-1}{G}$; RLOO gives $p(1-p)\cdot\frac{G}{G-1}$. Both collapse to zero as $p\to 0$ or $p\to 1$ — the *dominant* practical failure is degenerate groups, not baseline choice. DAPO's dynamic sampling exists precisely to discard all-correct/all-wrong groups.
- **Scale.** RLOO's advantage over PPO was measured at 6.9B (Pythia) and 7B (LLaMA) on TL;DR-summarization and HH-style preference data. GRPO's headline results are at 7B (DeepSeekMath) and larger for R1.
- **Critic quality.** VinePPO (Kazemnejad et al., 2024) shows PPO's learned value network is a poor value estimator on math reasoning — it identifies the correct higher-value branch only marginally above chance in their measurement — which is why replacing it with Monte-Carlo group statistics loses little.
- **Classical theory.** Greensmith, Bartlett & Baxter (JMLR 2004) give explicit variance bounds for baseline and actor-critic estimators in general MDPs; these bounds are stated in terms of mixing time and are vacuous at LLM sequence lengths.

## 5. What Is Not Known

- **Theoretically open.** The variance gap between the best $G$-sample critic-free baseline and $b^\star(x)$, as a function of $G$ and the reward distribution. No LLM-regime bound exists. Also open: whether *any* unbiased, parameter-free baseline can match an oracle per-token critic on sparse terminal reward, or whether a $\Omega(T)$ variance penalty is unavoidable.
- **Empirically open.** The head-to-head SNR measurement. Nobody has published $\mathrm{SNR}$ vs. $G$ for GRPO / Dr. GRPO / RLOO / PPO-critic at matched *generation* budget on the same 7B model. The experiment is fully runnable — a few thousand GPU-hours — and simply unrun.
- **Methodologically blocked.** "Variance reduction" as currently used is not well defined for LLM RL. Estimators differ in bias, in effective per-prompt learning rate, and in length weighting, so a lower measured $\mathrm{tr\,Cov}$ does not imply faster optimization. Until the community fixes a normalization under which the estimators are comparable, "lower variance" and "better" are separate claims neither of which implies the other.

## 6. Why It Is Hard

**Confounded measurement, plus non-identifiability of the mechanism.** Every proposed baseline change simultaneously alters (a) estimator variance, (b) the implicit per-prompt learning rate (std normalization is a difficulty reweighting), and (c) the length weighting of the token-level loss. Downstream accuracy — the only number reported — is a function of all three. This is the exact structure Tucker et al. (2018) found in the action-dependent-baseline literature, where the claimed variance reduction was absent and the gains came from a value-function fit. The obstruction is not compute: it is that the published metric does not measure the thing it names, so ten years of "baseline" results are not decidable against each other.

Secondary: the optimal baseline $b^\star$ requires $\mathbb{E}[\|g\|^2 R]$, needing per-sample gradient norms across billions of parameters, which vanilla training loops do not expose.

## 7. Current Research (as of 2026)

- **Bias-correction line.** Dr. GRPO (Sea AI Lab / NUS) removing std and length normalization; DAPO (ByteDance Seed / Tsinghua) with token-level loss and dynamic sampling; GSPO (Qwen team) moving importance ratios to sequence level. All argue about *what to normalize by*, none report variance.
- **Rollout-allocation line.** Adaptive $G$ per prompt: spend generations where $\hat p(1-\hat p)$ is largest. Appears in several open stacks (verl, OpenRLHF) as difficulty filtering. *(frontier — verify)* whether any published work gives an allocation rule with an optimality argument rather than a heuristic threshold.
- **Cheap-critic revival.** Value heads on frozen trunks, or single-scalar-per-token critics distilled from Monte-Carlo returns, as a middle path between GRPO and PPO. *(frontier — verify)*
- **Process-reward baselines.** Using step-level verifiers to build a per-prefix baseline without a parametric critic, in the VinePPO tradition (Mila).

## 8. Concrete Next Experiment

**Scale.** Qwen2.5-7B base (or Llama-3.1-8B), math RL on ~10k verifiable problems, 300 optimizer steps, matched *generation* budget of 4096 completions per step across all arms.

**Arms.** (1) GRPO, $G=8$, std-normalized. (2) GRPO without std (Dr. GRPO). (3) RLOO, $G=8$. (4) GRPO $G=32$ with 4× fewer prompts. (5) **Control arm:** PPO with a learned critic, $G=1$, same 4096 generations — the arm the critic-free literature claims to match.

**Measurement.** At steps 0, 50, 150, 300, freeze the policy and estimate $\mathrm{SNR}_k$ by splitting the 4096 completions into $M=16$ microbatches of 256 and computing between- vs. within-microbatch gradient covariance on the final two transformer blocks. Log alongside pass@1 on held-out AIME/MATH.

**The deciding number.** The Spearman rank correlation $\rho$ between arm ordering by $\mathrm{SNR}$ (averaged over the four checkpoints) and arm ordering by final pass@1. If $\rho \geq 0.8$, gradient variance is the operative mechanism and the field can optimize it directly. If $|\rho| < 0.4$ — the Tucker outcome — then "variance reduction" is the wrong name for what these methods do, and the ordering is driven by the implicit reweighting, which is then the thing to study.

## 9. Key References

- **[Foundational]** Ronald J. Williams. *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning, 1992.
- **[Foundational]** Lex Weaver, Nigel Tao. *The optimal reward baseline for gradient-based reinforcement learning.* UAI, 2001.
- **[Foundational]** Evan Greensmith, Peter L. Bartlett, Jonathan Baxter. *Variance reduction techniques for gradient estimates in reinforcement learning.* JMLR 5:1471–1530, 2004.
- **[Foundational]** Wouter Kool, Herke van Hoof, Max Welling. *Buy 4 REINFORCE samples, get a baseline for free!* Deep RL Meets Structured Prediction workshop, ICLR 2019.
- **[Cautionary]** George Tucker, Surya Bhupatiraju, Shixiang Gu, Richard E. Turner, Zoubin Ghahramani, Sergey Levine. *The Mirage of Action-Dependent Baselines in Reinforcement Learning.* ICML, 2018. — arXiv:1802.10031
- **[SOTA]** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300
- **[SOTA]** Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, Sara Hooker. *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.* ACL, 2024. — arXiv:2402.14740
- **[SOTA]** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin. *Understanding R1-Zero-Like Training: A Critical Perspective.* 2025. — arXiv:2503.20783
- **[SOTA]** Qiying Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476
- **[Analysis]** Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, Nicolas Le Roux. *VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment.* 2024. — arXiv:2410.01679
- **[Survey]** John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR, 2016. — arXiv:1506.02438

## 10. Worked Example

One prompt, verifier reward, $G=8$. Suppose the true success rate is $p=0.25$, so a typical group is 2 correct, 6 wrong.

- **RLOO.** A correct sample: $b = 1/7 \approx 0.143$, $A = 0.857$. A wrong sample: $b = 2/7 \approx 0.286$, $A = -0.286$. Mean advantage over the group: $\frac{2(0.857) + 6(-0.286)}{8} = 0.0$. Unbiased, correct sign, magnitudes proportional to surprise.
- **GRPO.** $b = 0.25$, $\mathrm{std} = \sqrt{0.25\cdot 0.75}\approx 0.433$. Correct: $A = 0.75/0.433 = 1.73$. Wrong: $A = -0.25/0.433 = -0.58$.

Now the obstruction. Take a second prompt with $p = 0.5$ (4 correct, 4 wrong): $\mathrm{std} = 0.5$, correct $A = 1.0$, wrong $A = -1.0$. Compare the *hard* prompt ($p=0.25$) to the *medium* one ($p=0.5$) under each rule:

| | RLOO, correct | RLOO, wrong | GRPO, correct | GRPO, wrong |
|---|---|---|---|---|
| $p=0.25$ | +0.857 | −0.286 | +1.73 | −0.58 |
| $p=0.50$ | +0.571 | −0.571 | +1.00 | −1.00 |

RLOO upweights the rare correct answer on the hard prompt by $0.857/0.571 = 1.50\times$ relative to the medium prompt. GRPO upweights it by $1.73/1.00 = 1.73\times$. The two estimators therefore point in *different directions* in parameter space once prompts of mixed difficulty are batched — not because one has lower variance, but because the $1/\mathrm{std}$ term is a difficulty-dependent learning rate.

Estimated per-sample advantage variance for the hard prompt: RLOO $\approx p(1-p)\frac{G}{G-1} = 0.214$; GRPO $\approx p(1-p)\frac{G-1}{G}/(p(1-p)) = 0.875$ after normalization — GRPO's is *larger* in raw units, yet GRPO is the method that trained R1. A measured variance comparison here would rank RLOO first and predict the wrong winner. That is the whole problem: the number the field would naturally measure does not, on this instance, order the methods the way accuracy does, and no one has run the experiment at scale to find out whether that inversion holds or is an artifact of the toy Bernoulli.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*