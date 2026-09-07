---
id: 35-world-models/amortized-versus-search-planning-tradeoff
title: "Amortized versus Search-Based Planning Tradeoff"
topic: 35-world-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Amortized versus Search-Based Planning Tradeoff

> **Topic:** World Models & Planning · **ID:** `35-world-models/amortized-versus-search-planning-tradeoff` · **Status:** open

## 1. Problem Statement

An agent with a learned world model can spend compute in two places: **train time**, distilling behaviour into a feed-forward policy (amortized planning), or **decision time**, unrolling the model under a search operator such as MCTS, MPC, or beam search (search-based planning). The two are partly interchangeable — AlphaZero distils search results back into the policy, and the policy in turn guides search.

The problem: **characterize the exchange rate, and say when it breaks.**

- **Measurement variant.** Given a task family, a model class, and a total lifetime compute budget $C$, measure the Pareto frontier over $(C_{\text{train}}, C_{\text{act}})$ at fixed return. Report the local exchange rate $\partial \log C_{\text{train}} / \partial \log C_{\text{act}}$ along an iso-performance curve.
- **Method variant.** Build an agent that allocates search depth per state rather than uniformly, and beat a compute-matched fixed-budget agent.
- **Theory variant.** Prove (or refute) that for some natural task family the amortization gap is bounded below by a function that grows with problem size — i.e. no polynomial-size feed-forward policy matches a search agent with $n$ model calls.

Solving it means: given a task and a deployment budget, predicting the optimal split *before* training, within a stated error bar.

## 2. Formal Setting

MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)$; learned model $\hat{P}_\phi, \hat{r}_\phi$. Amortized policy $\pi_\theta$. A search operator $\mathrm{S}[\pi_\theta, \hat{P}_\phi, n]$ maps a state to an improved action distribution using $n$ model calls.

**Return.** $J(\pi) = \mathbb{E}\big[\sum_t \gamma^t r_t\big]$, measured as the mean over $\ge 100$ evaluation episodes with seed-stratified sampling; report the standard error, since Atari/Hex-scale variance routinely exceeds the effect being claimed.

**Compute, measured not assumed.**
$$C_{\text{train}} = F_{\text{train}}\ \text{(FLOPs)},\qquad C_{\text{act}} = n \cdot f_{\hat{P}} + f_{\pi} \ \ \text{FLOPs per decision},$$
with $f_{\hat{P}}$ the FLOPs of one model expansion (dynamics + value head) and $f_\pi$ one policy forward pass. Both are counted from profiled operator FLOPs, not parameter counts, because MCTS is latency-bound and sequential: wall-clock and FLOPs diverge by more than an order of magnitude on accelerators.

**Amortization gap.** For budget $n$,
$$\Delta(n) = J\big(\mathrm{S}[\pi_\theta,\hat P_\phi,n]\big) - J(\pi_\theta).$$

**Iso-performance frontier.** $\mathcal{F}(J_0) = \{(C_{\text{train}}, C_{\text{act}}) : J = J_0\}$. The **exchange rate** is the local slope $\eta = -\,\partial \log C_{\text{train}} / \partial \log C_{\text{act}}$ on $\mathcal{F}$. **Crossover budget** $n^\*(C)$ is the $n$ maximizing $J$ at fixed lifetime $C = C_{\text{train}} + N_{\text{episodes}} T C_{\text{act}}$.

**Assumptions, and which are violated.**
1. *Model accuracy independent of search depth* — violated: compounding error makes $\hat P_\phi$ rollouts diverge, and MuZero-style models are value-equivalent, not state-predictive, so depth beyond training horizon is unreliable.
2. *Search is policy improvement* — holds only approximately; MCTS with a UCT-style prior is exactly regularized policy optimization in the limit (Grill et al., ICML 2020), but finite-$n$ trees are biased.
3. *Train and act compute are fungible* — violated in deployment: acting compute recurs per decision and is latency-constrained; training compute is paid once.
4. *Stationary task distribution* — violated whenever the deployment distribution differs from training, which is exactly the regime where search should pay.

## 3. State of the Art

**Established (ablated, reproduced).**
- AlphaGo Zero (Silver et al., *Nature* 2017): the raw policy network reaches ~3055 Elo; the same network with MCTS reaches ~5185 Elo. A ~2000-Elo amortization gap at fixed weights, from 1600 simulations per move.
- Hamrick et al. (ICLR 2021), "On the role of planning in model-based deep RL": ablating MuZero across Atari/9×9 Go/control finds search's dominant contribution is to *training* (better targets, better data), while test-time search adds relatively little once the policy is trained — and generalization gains from search appear mainly out of distribution.
- Grill et al. (ICML 2020): MCTS visit distributions are the solution to a regularized policy-optimization problem; replacing search with the closed-form solution matches MuZero at small simulation counts. This is a *theoretical* reduction with matching empirics.

**Claimed but incompletely ablated.**
- Jones (2021), "Scaling scaling laws with board games": in Hex, train-time and test-time compute trade off along a straight line in log-log space at fixed Elo. The reported exchange rate (roughly an order of magnitude of test-time compute for a comparable multiple of train-time compute) is measured on one game family at small scale; it has not been reproduced on a second domain.
- Snell et al. (2024, arXiv:2408.03314): optimal test-time compute allocation on MATH lets a small model beat a ~14× larger model on easy/medium problems, but the advantage inverts on hard problems. Single model family, single benchmark.

**Benchmark-number-only.** Most LLM "planning" results (Tree-of-Thoughts, best-of-$n$, self-consistency) report accuracy at unreported or non-FLOP-matched decode budgets. PlanBench (Valmeekam et al., NeurIPS 2023) shows the underlying amortized planning ability is weak, which makes the search-vs-amortize comparison there largely uninterpretable.

## 4. What Is Known

- **Gap magnitude, board games.** AlphaGo Zero: 1600 sims ⇒ ~2100 Elo over raw policy (19×19 Go, 40-block net). AlphaZero played at ~80k sims/s (chess) versus Stockfish's ~70M positions/s and still won — search *quality* is not search *count*.
- **Gap shrinks with training.** MuZero Atari (Schrittwieser et al., *Nature* 2020) uses 50 simulations per move; reducing test-time sims degrades performance far less than reducing training-time sims (Hamrick 2021), at 200M-frame scale on 57 games.
- **Search enables large action spaces only with sampling.** Sampled MuZero (Hubert et al., ICML 2021) matches full search using ~20 sampled actions in action spaces of size $10^3$–$10^9$ (DM Control, real-time strategy-scale).
- **Short-horizon MPC is data-efficient.** PETS (Chua et al., NeurIPS 2018) reaches asymptotic model-free performance on continuous control in ~$10^2$–$10^3$ trials, with horizon 25–30 CEM planning. TD-MPC2 (Hansen et al., ICLR 2024) at 317M parameters across 104 tasks shows short-horizon (3–5 step) planning over a learned latent model beats the amortized actor it contains.
- **Model-free nets can implicitly plan.** Guez et al. (ICML 2019), "An investigation of model-free planning": recurrent networks with no explicit model solve Sokoban-style tasks at rates comparable to search agents, and improve with extra "thinking" steps at test time. Amortization can absorb search.
- **Search-trace distillation works.** Searchformer (Lehnert et al., COLM 2024): a transformer trained on A\* execution traces solves Sokoban with ~26.8% fewer search steps than A\* itself after bootstrapping, at 175M parameters.

## 5. What Is Not Known

- **Theoretically open.** Whether there is a natural MDP family with a *provable* amortization gap — a lower bound showing any polynomial-size feed-forward policy is $\varepsilon$-suboptimal while $n$-call search is near-optimal. Circuit-depth arguments for sequential problems exist in the next-token setting (Bachmann & Nagarajan, ICML 2024, on the pitfalls of teacher-forcing for planning-like tasks) but do not yield an MDP-level separation with learned models.
- **Empirically open.** The exchange rate $\eta$ has never been measured on more than one domain family with matched FLOP accounting. Runnable today; nobody has run the 2-D sweep at a scale where both axes span three decades.
- **Empirically open.** Whether $n^\*$ should be state-dependent, and how much is left on the table by uniform budgets. Adaptive-compute results exist for LLM decoding; none for learned-model MCTS with FLOP-matched controls.
- **Methodologically blocked.** "Planning" in LLM agents is not operationally distinguished from retrieval of memorized plans; without a definition, the amortized/search comparison has no denominator (Momennejad et al., NeurIPS 2023).

## 6. Why It Is Hard

**The measurement is confounded three ways at once.**
1. *Search appears on both axes.* Search generates the training targets that make the amortized policy good. Removing test-time search from an agent trained with search does not measure amortization — it measures a distribution shift. A clean comparison needs two separately trained agents, doubling cost.
2. *FLOPs are the wrong currency for the deployment constraint.* MCTS is a sequential dependency chain: 800 simulations cannot be batched within one decision. An agent that is FLOP-cheaper can be wall-clock 50× slower. Papers that FLOP-match are not latency-matched, and vice versa; almost none report both.
3. *Model error is depth-coupled.* The amortization gap $\Delta(n)$ is a property of $(\pi_\theta, \hat P_\phi)$ jointly. Improving $\hat P_\phi$ raises the ceiling for search; improving $\pi_\theta$ lowers the floor. The two are non-identifiable from a single agent's ablation curve.

Add the base cost: a three-decade 2-D sweep at Atari scale is $\mathcal{O}(10^2)$ full training runs.

## 7. Current Research (as of 2026)

- **Test-time compute scaling laws for reasoning models.** OpenAI o-series and DeepSeek-R1-style RL-on-reasoning results push amortization of search into the weights; the open question is whether distilled chains recover the search agent's out-of-distribution robustness. *(frontier — verify)*
- **Latent world-model control.** DreamerV3 (Hafner et al., *Nature* 2025) is amortized-with-imagination; TD-MPC2 is search-in-latent-space. Direct compute-matched comparison between the two families is still absent.
- **Search-trace distillation.** Meta FAIR (Searchformer line) and follow-ups on distilling A\*/MCTS traces into sequence models.
- **Adaptive compute allocation** — process-reward-model-guided budget selection per problem; DeepMind and academic groups. Mostly LLM-side. *(frontier — verify)*

## 8. Concrete Next Experiment

**Two-domain, FLOP-and-latency-matched exchange-rate sweep.**

- **Scale.** Two domains: 9×9 Go (self-play, discrete) and DM Control humanoid (continuous, learned latent model). MuZero-style agent, 5 model sizes spanning $10^6$–$10^8$ parameters × 5 training-simulation counts $\{1, 4, 16, 64, 256\}$ = 25 runs per domain. Evaluate every checkpoint at test-time budgets $n \in \{1,2,4,\dots,1024\}$ — evaluation is cheap, so the sweep costs 50 training runs, ~$10^{21}$ FLOPs total.
- **Control arm.** For each cell, a *distillation twin*: an agent of identical architecture trained purely on the search agent's visit distributions, then evaluated at $n=1$. This separates "search as target generator" from "search as decision procedure" — the confound in §6.1.
- **Deciding number.** The fitted exchange rate $\eta = -\partial \log C_{\text{train}} / \partial \log C_{\text{act}}$ on the iso-performance frontier, with a bootstrap CI, in each domain. If the two domains' $\eta$ agree within a factor of 2, the tradeoff is a transferable law and $n^\*$ can be predicted before training. If they differ by more than 5×, no domain-general exchange rate exists and the field should stop quoting one.
- Secondary: $\Delta(n)$ for the distillation twin. If it is $\ge 80\%$ closed at $n=1$, test-time search is a training artifact, not a capability.

## 9. Key References

- **[Foundational]** R. S. Sutton. *Integrated architectures for learning, planning, and reacting based on approximating dynamic programming.* ICML, 1990.
- **[Foundational]** T. Anthony, Z. Tian, D. Barber. *Thinking Fast and Slow with Deep Learning and Tree Search.* NeurIPS, 2017. — arXiv:1705.08439
- **[Foundational]** D. Silver et al. *Mastering the game of Go without human knowledge.* Nature 550, 2017.
- **[SOTA]** J. Schrittwieser et al. *Mastering Atari, Go, chess and shogi by planning with a learned model.* Nature 588, 2020. — arXiv:1911.08265
- **[SOTA]** J. B. Hamrick et al. *On the role of planning in model-based deep reinforcement learning.* ICLR, 2021. — arXiv:2011.04021
- **[SOTA]** A. L. Jones. *Scaling Scaling Laws with Board Games.* 2021. — arXiv:2104.03113
- **[SOTA]** C. Snell, J. Lee, K. Xu, A. Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** N. Hansen, H. Su, X. Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR, 2024.
- **[SOTA]** L. Lehnert et al. *Beyond A\*: Better Planning with Transformers via Search Dynamics Bootstrapping.* COLM, 2024.
- **[Theory]** J.-B. Grill et al. *Monte-Carlo Tree Search as Regularized Policy Optimization.* ICML, 2020.
- **[Theory]** G. Bachmann, V. Nagarajan. *The Pitfalls of Next-Token Prediction.* ICML, 2024.
- **[Empirical]** A. Guez et al. *An Investigation of Model-Free Planning.* ICML, 2019.
- **[Empirical]** T. Hubert et al. *Learning and Planning in Complex Action Spaces.* ICML, 2021.
- **[Empirical]** K. Chua, R. Calandra, R. McAllister, S. Levine. *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models.* NeurIPS, 2018.
- **[Survey/Critique]** K. Valmeekam et al. *PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.* NeurIPS Datasets & Benchmarks, 2023.
- **[Survey/Critique]** I. Momennejad et al. *Evaluating Cognitive Maps and Planning in Large Language Models with CogEval.* NeurIPS, 2023.
- **[SOTA]** D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. *Mastering diverse control tasks through world models.* Nature, 2025.

## 10. Worked Example

Take AlphaGo Zero's published numbers and try to price the tradeoff.

- Raw policy net: 3055 Elo, one forward pass, $f_\pi \approx 2\times10^{9}$ FLOPs (40-block residual net, 19×19 input).
- Same weights + 1600 sims: 5185 Elo. Acting cost $C_{\text{act}} \approx 1600 \times 2\times10^{9} = 3.2\times10^{12}$ FLOPs per move — an 800× increase for +2130 Elo.

Now ask the question the catalog cares about: **how much extra training compute buys 2130 Elo without search?** Using Jones's Hex-fitted log-linear tradeoff and treating 800× test compute as exchangeable at roughly parity, the naive answer is "about 3 orders of magnitude more training compute." AlphaGo Zero's training was ~$10^{23}$ FLOPs; $10^{26}$ FLOPs is beyond any Go run ever done. So the extrapolation is untestable by construction.

**Where the obstruction becomes visible.** The 3055-Elo policy was itself trained on 1600-sim MCTS targets. Its 3055 Elo already contains the search. Setting $n=1$ at evaluation does not produce "the amortized agent" — it produces a search agent with its search amputated, evaluated off the state distribution it was trained on. The measured $\Delta(1600) = 2130$ Elo is therefore an upper bound on the true amortization gap of an unknown, possibly large size: no published run trains a Go agent of the same size *without* search targets to serve as the control.

Second, the currency is wrong. 1600 sequential sims at ~1 ms each is ~1.6 s per move; a policy-only move is ~2 ms. FLOP-matching says 800×; latency-matching says 800× too, but only because MCTS cannot batch — on hardware where the policy net batches 512-wide, the *throughput* ratio is closer to $4\times10^{5}$. Which number you quote changes the recommended $n^\*$ by nearly three orders of magnitude.

The experiment in §8 exists to supply the missing control arm and to report both currencies. Until it is run, every "planning beats scaling" claim is a single-domain number with an uncontrolled baseline.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*