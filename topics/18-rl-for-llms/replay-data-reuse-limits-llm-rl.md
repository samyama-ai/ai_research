---
id: 18-rl-for-llms/replay-data-reuse-limits-llm-rl
title: "Replay and Data Reuse Limits in LLM RL"
topic: 18-rl-for-llms
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Replay and Data Reuse Limits in LLM RL

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/replay-data-reuse-limits-llm-rl` · **Status:** empirically-open

## 1. Problem Statement

RL post-training of an LLM spends most of its wall-clock in generation, not in gradient steps. On a typical math-reasoning setup, rollout is 60–80% of step time. The obvious lever is to reuse each generated rollout for more than one gradient update — raise the *replay ratio*. The question is how far that can be pushed before the run degrades, and what "degrades" even means.

Three variants, with different difficulty:

- **Measurement.** Given a fixed generation budget $G$ (total sampled tokens) and a fixed prompt set, what is the empirical relationship between reuse factor $\mu$ (gradient passes per rollout batch) and final task reward? Is there a $\mu^\star>1$ that dominates $\mu=1$ at equal $G$, and does $\mu^\star$ move with model scale?
- **Method.** Design an off-policy correction — importance-sampling clipping, trust region, staleness bound, prioritization, or periodic reset — that extends the usable range of $\mu$ and of rollout *staleness* $s$ (how many optimizer steps old the behaviour policy is) without collapsing entropy or degrading pass@$k$ for large $k$.
- **Theory.** Bound the excess regret or the bias of the policy-gradient estimator as a function of $(\mu, s)$ for the LLM setting specifically: sparse terminal binary reward, sequence-level action spaces of size $|\mathcal{V}|^T$ with $T\sim10^3$–$10^4$, and a KL anchor to a reference model.

Solving it means: a predictive rule for $\mu^\star(N, G, s)$ that holds across at least two model scales and two task families, plus a mechanistic account of what fails past it.

## 2. Formal Setting

A policy $\pi_\theta$ maps prompt $x\sim\mathcal{D}$ to a response $y=(y_1,\dots,y_T)$. Terminal reward $r(x,y)\in\{0,1\}$ from a verifier (RLVR) or $r\in\mathbb{R}$ from a reward model. Objective:

$$J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\big]-\beta\,\mathbb{E}_x\big[\mathrm{KL}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right)\big].$$

**Quantities, as measured.**

- **Reuse factor** $\mu$ = optimizer steps taken on one rollout batch. Measured by counting `optimizer.step()` calls between two calls to the generation engine, not by epochs over a nominal dataset. With minibatching, $\mu = B_{\text{rollout}}/B_{\text{train}}$ when each sample is seen once per pass, times the number of passes.
- **Staleness** $s(y)$ = number of optimizer steps applied to $\theta$ between the checkpoint $\theta_{\text{old}}$ that generated $y$ and the checkpoint $\theta$ consuming it. In synchronous PPO-style loops $s\in\{0,\dots,\mu-1\}$; in asynchronous systems $s$ is a distribution with a tail, and must be logged per sample, not assumed.
- **Generation budget** $G$ = total tokens sampled, the actual FLOP-dominant resource. Compare arms at equal $G$, not equal optimizer steps — this is the control that most papers omit.
- **Sequence importance ratio** $\rho(x,y)=\prod_{t=1}^{T}\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t\mid x,y_{<t})}$, computed in log-space. Token ratios $\rho_t$ are what PPO/GRPO actually clip; $\log\rho$ has variance growing roughly linearly in $T$, so for $T=8{,}000$ the sequence ratio is numerically degenerate.
- **Effective sample size** $\mathrm{ESS}=\big(\sum_i w_i\big)^2/\sum_i w_i^2$ over the batch's normalized ratios $w_i$. This is the cheapest live diagnostic of how off-policy the batch has become; almost nobody reports it.
- **Diversity** measured as policy entropy $H_t$ per token and as pass@$k$ at $k\in\{1,16,256\}$ with temperature fixed and stated.

**Assumptions, and which break.**

1. *The behaviour policy is known.* Violated in practice: the inference engine (vLLM/SGLang) and the training engine compute different logprobs for the same tokens — kernel, precision, and batching differences — so the recorded $\pi_{\theta_{\text{old}}}$ is not the sampler. Every run is silently off-policy at $\mu=1$.
2. *Clipping bounds the bias.* PPO's ratio clip zeroes gradients outside the trust region but gives no bound on the bias of the surviving estimator; the objective is not a lower bound on $J$ once $s>0$.
3. *Rewards are stationary.* False when a learned reward model is being exploited — reuse amplifies whatever the reward model gets wrong.
4. *Prompts are i.i.d. and reward-informative.* Under GRPO's group-relative advantage, prompts where all $n$ samples agree contribute exactly zero gradient; the fraction of such prompts rises during training, so the effective batch shrinks in a way $\mu$ does not capture.

## 3. State of the Art

**Systems/empirical SOTA.** Asynchronous, partly off-policy pipelines are now standard. AReaL (Fu et al., 2025) trains with a bounded staleness parameter plus a decoupled PPO objective and reports ~2.5$\times$ throughput at matched final accuracy on math benchmarks. Noukhovitch et al. (ICLR 2025) show asynchronous RLHF with one-step-stale rollouts matches synchronous PPO on TL;DR summarization and instruction-following at lower wall-clock. DAPO (Yu et al., 2025) uses dynamic sampling — discard prompts with all-correct or all-wrong groups and resample — which is a data-*selection* intervention that raises the value of each generated token; it reports AIME'24 avg@32 of 50 on Qwen2.5-32B. GSPO (Zheng et al., Qwen, 2025) replaces token-level ratios with a length-normalized sequence-level ratio, motivated exactly by the variance blowup in §2, and reports more stable long-run training, especially for MoE.

**Established vs. claimed.** Established: asynchrony with small bounded staleness is roughly free at the scales tested. Claimed but unablated: that the specific correction (decoupled objective, sequence ratio, TIS) is what buys the stability, rather than the accompanying hyperparameter changes — these papers ship several changes at once and few report the equal-generation-budget control arm. Numbers like DAPO's AIME score are benchmark points on one model family, not a $\mu$-sweep.

**Theory SOTA.** Off-policy correction theory is inherited from deep RL and not adapted to sequence-level LLM structure: Retrace$(\lambda)$ (Munos et al., NeurIPS 2016) gives contraction for arbitrary behaviour policies with truncated traces; V-trace (Espeholt et al., ICML 2018) gives a fixed point that is a *biased* policy, biased toward the behaviour policy as truncation tightens. Neither is used in production LLM RL. Tapered off-policy REINFORCE (Le Roux et al., 2025) is one of the few LLM-specific analyses of the on/off-policy tradeoff.

## 4. What Is Known

- **Small $\mu$ helps, large $\mu$ hurts.** PPO's original ablation (Schulman et al., 2017) used $\mu\approx$ 3–10 epochs per rollout batch on control tasks. LLM practice has quietly converged to $\mu\in\{1,2,4\}$; DeepSeek-R1-style GRPO recipes and Kimi k1.5 (2025) both sit at the low end. No public sweep at 30B+ scale reports $\mu>8$ working.
- **Repeated data has a ceiling in the supervised analogue.** Muennighoff et al. (NeurIPS 2023) found up to ~4 epochs of repeated pretraining data are nearly as good as fresh data, with returns decaying to near-zero by ~16 epochs, measured at 8.7B params / up to 178B tokens. This is the closest quantitative anchor and it is for cross-entropy, not policy gradients.
- **Replay ratio barriers are breakable with resets, in small RL.** D'Oro et al. (ICLR 2023) reached replay ratio 16 on Atari-100k by periodically resetting network parameters; Nikishin et al. (ICML 2022) identify primacy bias as the mechanism. Fedus et al. (ICML 2020) show replay *capacity* and *oldest-policy age* are separately causal, at DQN scale. None of this has been tested on a >1B-parameter LLM.
- **Off-policy alignment underperforms online at matched data.** Tang et al. (2024) find a persistent gap between offline (DPO-style) and online RLHF across several tasks and budgets, with the gap not closed by better offline losses or more offline data.
- **Long RL runs collapse diversity.** Yue et al. (2025) report RLVR raising pass@1 while *lowering* pass@256 relative to the base model on math and code — the sampling boundary narrows. Reuse plausibly accelerates this, but that specific ablation is not in the paper.

## 5. What Is Not Known

- **Empirically open.** The $\mu$-sweep at fixed generation budget $G$ across two model scales. Runnable today on 8B and 32B models for ~$10^4$ GPU-hours; nobody has published it. Likewise the interaction $\mu\times s$: is one stale-by-8 pass worth more or less than eight fresh-but-reused passes?
- **Empirically open.** Whether parameter resets or optimizer-state resets transfer from Atari to LLM RL. Cheap to test on 1B–8B; untested publicly.
- **Theoretically open.** No bias bound for clipped surrogate objectives at staleness $s>0$ in the sequence setting. No characterization of when GRPO's group-relative advantage remains a valid advantage estimate under reuse — the group baseline is computed under $\pi_{\theta_{\text{old}}}$ and goes stale with the samples.
- **Methodologically blocked.** "Degradation" has no agreed metric. Reward goes up while pass@256 goes down; entropy collapse is diagnostic but not itself bad; benchmark scores saturate. Without a settled diversity-preserving objective, a $\mu$-sweep can be scored to produce whichever answer the author prefers.
- **Methodologically blocked.** The inference/training logprob mismatch (§2, assumption 1) means the measured $\rho$ is not the true ratio, so ESS and clip-fraction diagnostics are biased by an unquantified amount that varies by serving stack.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability of the failure mode**. Raising $\mu$ changes three things at once: the estimator becomes off-policy (bias), the optimizer takes more steps on the same data (overfitting to a batch), and the entropy schedule changes (exploration). All three produce the same surface signature — reward plateaus, then pass@$k$ for large $k$ falls. Nothing in the standard logging separates them.

The second obstruction is cost asymmetry. The interesting regime is *long* RL — the runs where reuse would pay off are 10k+ steps (ProRL, Hilton-style scaling), so a clean sweep of five $\mu$ values at two scales is a multi-hundred-thousand-GPU-hour experiment, and short proxy runs are known to mis-rank recipes: ScaleRL (Khatri et al., 2025) fits sigmoidal compute–performance curves precisely because early-run ranking does not predict the asymptote.

## 7. Current Research (as of 2026)

- **Asynchronous RL infrastructure** — AReaL (Ant/Tsinghua), slime, verl, and the OpenRLHF line — treating staleness as a first-class tunable with bounded queues rather than strict on-policy sync.
- **Sequence-level off-policy corrections** — GSPO (Qwen), truncated importance sampling to repair the inference/training mismatch, and tapered REINFORCE variants. *(frontier — verify: which of these is load-bearing is not yet separately ablated.)*
- **Data selection as an alternative to reuse** — DAPO-style dynamic sampling, difficulty-matched curricula, and prompt-level prioritization (the LLM analogue of PER, Schaul et al., ICLR 2016). *(frontier — verify.)*
- **Scaling-law framing** — ScaleRL (Meta, 2025) fits $\text{pass rate}=A-\frac{B}{1+(C/\text{compute})^{\alpha}}$ and argues recipes should be compared by fitted asymptote $A$, which is the right frame for a $\mu$-sweep and has not yet been applied to one.
- **Diversity-preserving objectives** — entropy floors, clip-higher, and pass@$k$-aware rewards, aimed at the metric gap in §5.

## 8. Concrete Next Experiment

**Scale.** Qwen3-8B-Base, RLVR on a 40k-problem verified math set, group size $n=16$, max response 8k tokens. Fixed generation budget $G=6\times10^{11}$ sampled tokens per arm (~2–3k GPU-days total for all arms on H100s; feasible for one lab).

**Arms.** $\mu\in\{1,2,4,8,16\}$, all at equal $G$ — so the $\mu=16$ arm takes 16$\times$ the optimizer steps of $\mu=1$ on the same generated tokens. Learning rate held fixed; a second $\mu=16$ arm with lr scaled by $1/\sqrt{\mu}$ guards against the trivial confound. Add a $\mu=16$ + periodic-reset arm (reset the value head and optimizer state every 500 steps, D'Oro-style).

**Control arm.** $\mu=1$, matched $G$, matched wall-clock-agnostic. This is the arm the literature usually omits.

**Logging.** Per-step ESS, clip fraction, token entropy, and the inference-vs-training logprob KL (to quantify assumption 1).

**Deciding number.** Fit the ScaleRL sigmoid to each arm's pass@1-vs-$G$ curve and report the fitted asymptote $A$. The single decisive quantity is $\Delta A = A(\mu{=}4)-A(\mu{=}1)$, with pass@256 at final checkpoint as a mandatory co-primary. Decision rule: if $\Delta A > 0.02$ absolute AND pass@256 drops by less than 1 point, reuse at $\mu=4$ is established as a free 4$\times$ on generation cost. If $\Delta A \le 0$, the field's default $\mu=1$ is correct and the remaining question is entirely about asynchrony, not reuse.

## 9. Key References

- **[Foundational]** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. *Proximal Policy Optimization Algorithms.* arXiv preprint, 2017. — arXiv:1707.06347
- **[Foundational]** Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare. *Safe and Efficient Off-Policy Reinforcement Learning.* NeurIPS, 2016. — arXiv:1606.02647
- **[Foundational]** Lasse Espeholt et al. *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.* ICML, 2018. — arXiv:1802.01561
- **[Foundational]** Tom Schaul, John Quan, Ioannis Antonoglou, David Silver. *Prioritized Experience Replay.* ICLR, 2016. — arXiv:1511.05952
- **[Foundational]** William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, Will Dabney. *Revisiting Fundamentals of Experience Replay.* ICML, 2020. — arXiv:2007.06700
- **[Foundational]** Pierluca D'Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc G. Bellemare, Aaron Courville. *Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier.* ICLR, 2023.
- **[Foundational]** Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, Aaron Courville. *The Primacy Bias in Deep Reinforcement Learning.* ICML, 2022. — arXiv:2205.07802
- **[SOTA]** Zhihong Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv preprint, 2024 (GRPO). — arXiv:2402.03300
- **[SOTA]** Qiying Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv preprint, 2025. — arXiv:2503.14476
- **[SOTA]** Chujie Zheng et al. *Group Sequence Policy Optimization.* arXiv preprint, 2025. — arXiv:2507.18071
- **[SOTA]** Michael Noukhovitch, Shengyi Huang, Sophie Xhonneux, Arian Hosseini, Rishabh Agarwal, Aaron Courville. *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models.* ICLR, 2025. — arXiv:2410.18252
- **[SOTA]** Wei Fu et al. *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning.* arXiv preprint, 2025. — arXiv:2505.24298
- **[SOTA]** Devvrit Khatri et al. *The Art of Scaling Reinforcement Learning Compute for LLMs.* arXiv preprint, 2025. — arXiv:2510.13786
- **[Evidence]** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Evidence]** Yuxiao Tang et al. *Understanding the Performance Gap Between Online and Offline Alignment Algorithms.* arXiv preprint, 2024. — arXiv:2405.08448
- **[Evidence]** Yang Yue et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* arXiv preprint, 2025. — arXiv:2504.13837
- **[Evidence]** Mingjie Liu et al. *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models.* arXiv preprint, 2025. — arXiv:2505.24864
- **[Survey]** Nicolas Le Roux et al. *Tapered Off-Policy REINFORCE: Stable and Efficient Reinforcement Learning for LLMs.* arXiv preprint, 2025. — arXiv:2503.14286

## 10. Worked Example

Take one GRPO step: 512 prompts, $n=16$ samples each, mean response length 4,000 tokens. Rollout batch = $512\times16\times4{,}000 = 3.28\times10^7$ generated tokens. Training minibatch = 128 prompt-groups, so $\mu=4$ passes.

**The accounting that motivates reuse.** Generation on an 8B model at ~4k tok/s/GPU across 64 GPUs takes ~128 s. Four gradient passes on $3.28\times10^7$ tokens cost roughly $6ND \approx 6\cdot 8\times10^9\cdot 3.28\times10^7 \approx 1.6\times10^{18}$ FLOPs, about 25 s on the same hardware at 40% MFU. So $\mu{=}4$ raises step compute by ~20% and cuts generation calls by 4$\times$: a nominal 3$\times$ wall-clock win.

**Where it goes wrong.** At pass 4, $\theta$ is three optimizer steps from $\theta_{\text{old}}$. Suppose per-token log-ratio drift is a modest $\sigma=0.004$ nats. Over $T=4{,}000$ tokens the sequence log-ratio has standard deviation $\sigma\sqrt{T}=0.004\cdot63\approx0.25$, so $\rho$ ranges over roughly $[0.6, 1.6]$ across the batch — already enough that ESS falls from 2,048 to a few hundred. Push to $T=16{,}000$ (long-CoT) and the same per-token drift gives $\sigma\sqrt{T}=0.5$, $\rho\in[0.4,2.7]$: most sequences sit outside a $1\pm0.2$ trust region and get clipped to zero gradient. The batch you paid $3.28\times10^7$ tokens for contributes almost nothing on pass 4.

**Why the measurement is confounded.** Now add GRPO's group baseline. If 30% of prompts have all-16 correct and 12% all-16 wrong, 42% of groups have zero advantage before reuse even starts — the true effective batch is 297 prompts, not 512. The observed "reuse stopped helping at $\mu=4$" is therefore consistent with three different causes: importance-weight collapse (fixable with a sequence-level ratio), zero-advantage groups (fixable with dynamic sampling), or genuine overfitting to the batch (fixable only with resets or fresh data). Reward curves alone cannot tell them apart. Logging ESS and the zero-advantage fraction alongside the $\mu$-sweep is what turns this from a folklore heuristic into a measurement — and that is precisely what §8 asks for.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*