---
id: 18-rl-for-llms/off-policy-correction-stale-rollouts
title: "Off-Policy Correction for Stale Rollouts in Async RL"
topic: 18-rl-for-llms
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Off-Policy Correction for Stale Rollouts in Async RL

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/off-policy-correction-stale-rollouts` · **Status:** partially-solved

## 1. Problem Statement

Asynchronous RL post-training decouples generation from optimization: a pool of inference workers samples trajectories from a snapshot $\mu = \pi_{\theta_{t-\eta}}$ while the learner has already advanced to $\pi_{\theta_t}$. The rollouts are **stale** by $\eta$ gradient steps. Staleness buys throughput — no pipeline bubble waiting for the slowest 32k-token generation — and costs correctness: the gradient estimator is off-policy.

- **Input:** trajectories $\{(x, y^{(i)}, r^{(i)}, \log \mu(y^{(i)}\mid x))\}$ generated under $\mu$, plus current parameters $\theta_t$.
- **Output:** an update $\Delta\theta$ approximating an on-policy update for $\pi_{\theta_t}$.
- **Decision predicate:** does there exist a correction that holds final task accuracy within noise of the synchronous control while permitting $\eta \gg 1$?

Three variants, different difficulty:

- **Measurement.** Quantify how far off-policy a given batch is, in a way that predicts degradation. Not solved: KL, ESS and clip-fraction are all used and none is validated as a predictor.
- **Method.** Engineering a correction (truncated IS, decoupled clipping, sequence-level ratios) that empirically preserves accuracy at $\eta \le 4$. Largely solved for small $\eta$.
- **Theory.** Bound the bias/variance of the estimator as a function of $\eta$, sequence length $T$, and clip threshold. Open. Existing off-policy guarantees (Retrace, V-trace) assume a fixed behavior policy and bounded per-step corrections; neither holds here.

## 2. Formal Setting

A prompt $x \sim \mathcal{D}$, a response $y = (y_1,\dots,y_T)$, terminal reward $r(x,y) \in [0,1]$ (verifier output, measured as a binary pass/fail from a unit-test or math-checker). Policies are token-level: $\pi_\theta(y\mid x) = \prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t})$.

**Staleness.** $\eta$ = number of learner optimizer steps between the snapshot that produced $y$ and the snapshot being updated. Measured by tagging each rollout with its generating version id; report the *distribution* of $\eta$ over a batch, not the max, because heavy tails dominate.

**Sequence importance ratio.**
$$w(y) = \frac{\pi_\theta(y\mid x)}{\mu(y\mid x)} = \exp\!\Big(\sum_{t=1}^{T}\big[\log\pi_\theta(y_t\mid\cdot) - \log\mu(y_t\mid\cdot)\big]\Big)$$
Measured by re-scoring $y$ under the trainer's forward pass and comparing to logprobs *returned by the inference engine*. These differ even at $\eta=0$ because of kernel, batching and precision differences.

**Effective sample size**, the standard diagnostic:
$$\mathrm{ESS} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2}, \qquad \frac{\mathrm{ESS}}{N}\approx \exp(-\mathrm{Var}[\log w]) \text{ for log-normal } w.$$

**Token-level PPO/GRPO surrogate**, with group-normalized advantage $\hat A$:
$$\mathcal{L} = -\mathbb{E}\Big[\tfrac{1}{T}\sum_t \min\big(\rho_t \hat A,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A\big)\Big], \quad \rho_t = \frac{\pi_\theta(y_t\mid\cdot)}{\mu(y_t\mid\cdot)}.$$

**Decoupled objective** (Hilton et al., 2022; adopted by AReaL) introduces a proximal anchor $\pi_{\mathrm{prox}}$ distinct from the behavior policy:
$$\mathcal{L}_{\mathrm{dec}} = -\mathbb{E}_{\mu}\Big[\tfrac{\pi_{\mathrm{prox}}}{\mu}\min\big(\tfrac{\pi_\theta}{\pi_{\mathrm{prox}}}\hat A,\ \mathrm{clip}(\cdot)\hat A\big)\Big].$$

**Assumptions, and which break.**
1. *$\mu$ is known exactly.* Violated — inference-engine logprobs disagree with trainer logprobs by $10^{-3}$–$10^{-2}$ nats/token in practice.
2. *Support: $\pi_\theta \ll \mu$.* Nominally true (softmax has full support) but numerically false after low-probability tokens are pruned by top-$p$ sampling.
3. *Single behavior policy.* Violated — a batch mixes rollouts from many snapshots, so $\mu$ is a mixture whose weights are unknown unless logged.
4. *Bounded per-step ratio.* Violated at $T = 10^4$: even tiny per-token drift compounds multiplicatively.

## 3. State of the Art

**Systems/empirical SOTA (established, ablated).**
- **AReaL** (Fu et al., 2025): staleness-bounded rollout queue plus the decoupled PPO objective. Reports up to $2.77\times$ end-to-end speedup at matched or better AIME24 accuracy for 1.5B/7B reasoning models. The ablation that matters is present: with staleness uncapped and no decoupled objective, accuracy collapses; with $\eta \le 4$ and the decoupled loss, it recovers.
- **Asynchronous RLHF** (Noukhovitch et al., ICLR 2025): one-step-off-policy ($\eta=1$) matches synchronous PPO on TL;DR summarization and general instruction-following up to 8B, with online DPO more robust to staleness than PPO. Establishes that $\eta=1$ is essentially free.
- **GSPO** (Zheng et al., Qwen, 2025): sequence-level, length-normalized importance ratio $ (\pi_\theta/\mu)^{1/T}$ with sequence-level clipping. Claimed to stabilize MoE training where token-level GRPO diverges. Established as a training-stability result on Qwen3; the isolated contribution to *staleness* tolerance is not ablated separately from its MoE routing-drift benefit.
- **INTELLECT-2** (Prime Intellect, 2025): decentralized async at 32B with two-step delayed policy and aggressive clipping. Benchmark numbers only; no controlled staleness sweep.

**Theory SOTA.** Retrace$(\lambda)$ (Munos et al., NeurIPS 2016) and V-trace (Espeholt et al., ICML 2018) give convergence for clipped per-step corrections under a *fixed* behavior policy and bounded horizon. Truncated IS has classical bias/variance results (Ionides, JCGS 2008). Neither line covers the LLM regime: $T\sim 10^3$–$10^4$, terminal-only reward, mixture behavior policy, and a bandit-like (not TD) objective. **No bound exists relating $\eta$ to the bias of the clipped GRPO gradient.**

**Claimed but unablated.** Truncated importance sampling to repair the vLLM↔trainer logprob mismatch is now standard in open frameworks; the claim that it fixes late-training collapse circulated first through practitioner reports and framework code rather than a controlled paper, and the sequence-length dependence of the fix is unmeasured.

## 4. What Is Known

- $\eta = 1$ is safe. Matched performance vs. synchronous PPO at 2.8B and 8B on TL;DR/HH (Noukhovitch et al., ICLR 2025).
- $\eta \le 4$ with decoupled PPO is safe for 1.5B–7B reasoning models on AIME24/AMC; throughput gain up to $2.77\times$ (AReaL, 2025).
- Uncapped staleness degrades. AReaL's own ablation shows accuracy loss when the rollout queue is unbounded, at 1.5B scale.
- The behavior policy is not the policy you think. Trainer-vs-inference logprob divergence is nonzero at $\eta=0$; ignoring it makes even "on-policy" GRPO silently off-policy.
- Sequence-level IS is variance-catastrophic at long $T$. Direct consequence of $\mathrm{Var}[\log w]$ growing linearly in $T$; this is why GSPO length-normalizes rather than sums.
- Partial rollouts (Kimi k1.5, 2025) create staleness *within* a single trajectory — prefix generated by an older policy than the suffix — and were still trainable at frontier scale.

## 5. What Is Not Known

- **Theoretically open.** No bias bound for clipped token-level PPO/GRPO as a function of $(\eta, T, \epsilon)$. No proof that the decoupled objective is consistent when $\mu$ is an unknown mixture. Whether any unbiased-and-finite-variance estimator exists at $T=10^4$ with terminal reward is open; the log-normal variance argument suggests not.
- **Empirically open.** The staleness scaling law. Nobody has published a controlled sweep of $\eta \in \{1,2,4,8,16,32\}$ crossed with model scale $\{1.5\text{B}, 7\text{B}, 32\text{B}\}$ at fixed total tokens. The prevailing belief that "larger models tolerate more staleness" is untested. Runnable today; costs a few hundred thousand GPU-hours.
- **Methodologically blocked.** *How to measure off-policyness.* ESS, mean $|\log w|$, clip fraction and reverse KL are all reported, none validated against downstream degradation, and they disagree — a batch can have ESS/$N = 0.9$ token-wise and $0.05$ sequence-wise. Until one diagnostic is shown to predict final accuracy loss, staleness budgets are tuned by trial.

## 6. Why It Is Hard

**The specific obstruction is non-identifiability of the correction's contribution, compounded by multiplicative variance in $T$.**

1. *Variance is structural, not fixable by a better estimator.* $\log w$ is a sum of $T$ terms. Any correction that is unbiased must weight by $w$, whose variance grows exponentially in $T$. Every deployed method therefore clips, which trades an unbounded variance for an unquantified bias. There is no known way to measure that bias — it requires the on-policy gradient you cannot afford to compute.
2. *Confounded measurement.* In every published async system, staleness co-varies with batch composition, generation-length distribution (long rollouts are systematically staler, so staleness correlates with difficulty), and the numeric mismatch of the inference engine. An accuracy drop at $\eta=8$ cannot be attributed to off-policyness rather than to the fact that the stalest 5% of samples are also the hardest prompts.
3. *Absent ground truth.* The reference "on-policy" run is itself off-policy by the engine mismatch, so there is no zero point on the axis.

## 7. Current Research (as of 2026)

- **Framework teams** (AReaL/Ant-Tsinghua, slime/THUDM, veRL/ByteDance, ROLL/Alibaba) are converging on: bounded staleness queue + decoupled or truncated-IS objective + logged behavior logprobs. This is now the default stack.
- **Sequence-level vs. token-level ratios** — GSPO (Qwen) vs. token-level clipping. Active disagreement; the deciding ablation at matched staleness has not been published.
- **Fixing the engine mismatch at the source** — bitwise-consistent or logprob-consistent inference kernels between vLLM/SGLang and FSDP/Megatron. If solved, it removes confound (3). *(frontier — verify)*
- **Fully asynchronous / decentralized regimes** — Prime Intellect, and trajectory-balance objectives (TBA, 2025) that are off-policy by construction rather than by correction. *(frontier — verify)*
- **Theory** — extending V-trace-style contraction results to terminal-reward, mixture-behavior settings. No published result yet.

## 8. Concrete Next Experiment

**The staleness–scale sweep with a clean control.**

- **Scale.** Qwen3-8B base, GRPO on a fixed math+code verifier set, 4k prompts/epoch, 8 samples/prompt, 16k max response length, fixed 400 optimizer steps and identical data order across arms. About 6k H100-hours total.
- **Arms.** $\eta \in \{0, 1, 4, 16\}$ crossed with correction $\in$ {none, token-clip GRPO, truncated-IS $\bar\rho=2$, decoupled PPO}. **Critically: staleness is injected artificially** — run fully synchronous, buffer old snapshots, and score each rollout under a deliberately delayed $\theta_{t-\eta}$. This removes the length/difficulty confound, because $\eta$ is then independent of prompt.
- **Control arm.** $\eta=0$ with trainer-computed logprobs used as $\mu$ (no engine mismatch). This is the true zero point.
- **Deciding number.** $\Delta$AIME25 pass@1 (32-sample average) at step 400, relative to control. If the best correction holds $|\Delta| < 1.5$ points at $\eta=16$, staleness is a solved engineering knob and async can be run wide open. If $\Delta < -5$ points at $\eta=16$ under every correction, the variance obstruction is real and the field should stop tuning clip thresholds and start bounding $\eta$.
- **Secondary output.** Regress $\Delta$ on each diagnostic (sequence ESS, token ESS, clip fraction, reverse KL). The one with highest $R^2$ becomes the standard measure — that alone unblocks Section 5's methodological gap.

## 9. Key References

- **[Foundational]** Precup, Sutton, Singh. *Eligibility Traces for Off-Policy Policy Evaluation.* ICML, 2000.
- **[Foundational]** Munos, Stepleton, Harutyunyan, Bellemare. *Safe and Efficient Off-Policy Reinforcement Learning.* NeurIPS, 2016. — arXiv:1606.02647
- **[Foundational]** Espeholt et al. *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.* ICML, 2018. — arXiv:1802.01561
- **[Foundational]** Ionides. *Truncated Importance Sampling.* Journal of Computational and Graphical Statistics, 17(2), 2008.
- **[Foundational]** Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization Algorithms.* 2017. — arXiv:1707.06347
- **[Method]** Hilton, Cobbe, Schulman. *Batch Size Invariance for Policy Optimization.* NeurIPS, 2022. — arXiv:2110.00641 (source of the decoupled PPO objective)
- **[Method]** Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300 (GRPO)
- **[SOTA]** Noukhovitch, Huang, Xhonneux, Hosseini, Agarwal, Courville. *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models.* ICLR, 2025. — arXiv:2410.18252
- **[SOTA]** Fu et al. *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning.* 2025. — arXiv:2505.24298
- **[SOTA]** Zheng et al. *Group Sequence Policy Optimization.* Qwen Team, 2025. — arXiv:2507.18071
- **[Systems]** Sheng et al. *HybridFlow: A Flexible and Efficient RLHF Framework.* EuroSys, 2025. — arXiv:2409.19256
- **[Systems]** Kimi Team. *Kimi k1.5: Scaling Reinforcement Learning with LLMs.* 2025. — arXiv:2501.12599 (partial rollouts)
- **[Systems]** Prime Intellect Team. *INTELLECT-2: A Reasoning Model Trained Through Globally Decentralized Reinforcement Learning.* 2025. — arXiv:2505.07291

## 10. Worked Example

Take a 7B model, response length $T = 4096$, staleness $\eta = 4$ at learning rate $1\times10^{-6}$.

Measure the per-token log-ratio $\delta_t = \log\pi_\theta(y_t) - \log\mu(y_t)$. Empirically, four AdamW steps at this LR move a 7B policy by roughly $\mathrm{std}[\delta_t] \approx 0.02$ nats with near-zero mean on its own samples. Treat $\delta_t$ as weakly correlated. Then

$$\mathrm{Var}[\log w] \approx T\,\sigma^2 = 4096 \times (0.02)^2 = 1.64, \qquad \frac{\mathrm{ESS}}{N} \approx e^{-1.64} = 0.19.$$

So the *sequence-level* estimator throws away 81% of a batch of 512 — effectively 99 usable samples — before any reward signal is considered. Push to $T = 16384$: $\mathrm{Var} = 6.55$, ESS/$N = 0.0014$, under one usable sequence per 512. Sequence IS is dead at reasoning lengths.

Now the token-level view of the *same batch*: $\rho_t = e^{\delta_t}$ has $\mathrm{Var}[\log\rho_t] = 4\times10^{-4}$, token ESS/$N = 0.9996$, and with $\epsilon = 0.2$ the clip fraction is $\approx 0$. Token-level diagnostics say the batch is perfectly on-policy. Sequence-level diagnostics say it is unusable.

**That is the obstruction, in one batch.** The two standard measurements of "how off-policy am I" differ by a factor of 140 in ESS on identical data, and neither is wrong — they measure different estimators. Token clipping does not report the sequence-level distribution shift it is silently ignoring; that shift is exactly the quantity the terminal reward is attached to. Until Section 8's regression identifies which diagnostic predicts $\Delta$ accuracy, every published staleness budget is a hyperparameter found by running the job and seeing whether it broke.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*