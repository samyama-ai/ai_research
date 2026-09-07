---
id: 35-world-models/amortized-versus-explicit-search
title: "Amortized Planning Versus Explicit Search at Scale"
topic: 35-world-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Amortized Planning Versus Explicit Search at Scale

> **Topic:** World Models & Planning · **ID:** `35-world-models/amortized-versus-explicit-search` · **Status:** empirically-open

## 1. Problem Statement

A planner can spend compute in two places. **Amortized**: pay at training time, distill the answers into weights, emit an action in one forward pass. **Explicit**: pay at decision time, unroll a model, expand a tree, sample and re-rank.

The question is not which is better in the abstract — it is the **exchange rate**, and whether that exchange rate survives scale.

- **Measurement variant.** For a task family and a skill target $s$, trace the iso-skill frontier in the plane (train FLOPs, per-decision test FLOPs). What is its local slope, and how does the slope vary with the *depth* of the required lookahead?
- **Method variant.** Build a policy that, given a fixed total FLOP budget, allocates between the two automatically and beats either pure arm.
- **Theory variant.** Characterize the problem classes for which a bounded-depth amortized policy of size $\mathrm{poly}(n)$ can match a depth-$d$ search, and those for which it provably cannot.

Solving it means: given a task and a budget, predicting the optimal split *before* running the experiment.

## 2. Formal Setting

MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)$ with a learned model $\hat{P}_\theta$. Two decision rules:

$$\pi^{\text{am}}_\theta(a\mid s) = f_\theta(s), \qquad \pi^{\text{ex}}_{\theta,B}(a \mid s) = \mathrm{Search}(s, \hat{P}_\theta, f_\theta; B)$$

where $B$ is a test-time budget (MCTS simulations, beam width, samples $k$, CoT tokens).

**Quantities as measured.**

- $C_{\text{train}}$: total training FLOPs, counted as $6ND$ ($N$ non-embedding params, $D$ tokens/transitions) *plus* the cost of generating supervision. For distillation-from-search pipelines the teacher's search cost belongs here, and is routinely omitted.
- $C_{\text{test}}$: FLOPs per *decision*, not per episode. $C_{\text{test}} \approx 2N \cdot B \cdot L$ for $B$ rollouts of length $L$.
- $s$: skill. Elo from head-to-head play in games; pass@1 under a verifier-free protocol elsewhere. pass@$k$ with an oracle checker is a *coverage* measure, not a skill measure, and must not be used on the amortized arm's side of the comparison.
- $d(x)$: instance planning depth — the minimum number of sequentially dependent decisions on any optimal solution path. Measurable exactly in Sokoban/Hex endgames/mate-in-$k$; only estimable in open-ended text tasks.

**Exchange rate.** On the iso-skill set $\{(C_{\text{train}}, C_{\text{test}}) : s = s_0\}$, define

$$\alpha(s_0, d) \;=\; -\,\frac{\partial \log C_{\text{train}}}{\partial \log C_{\text{test}}}\Bigg|_{s = s_0,\, d}.$$

$\alpha \approx 1$ means one order of magnitude of thinking buys one order of magnitude of training. $\alpha \to 0$ means search is worthless. The empirical claim under test is that $\alpha$ decays in $d$.

**Assumptions, and which fail.**

1. *The learned model is accurate enough for search to be a contraction.* Violated: compounding error in $\hat P_\theta$ makes deep search actively harmful in pixel-based domains.
2. *Skill is scalar and transitive.* Violated in games (Elo non-transitivity) and in reasoning benchmarks with heterogeneous $d$.
3. *Search and network are separable.* Violated: MuZero-style training uses search targets, so $\theta$ is not the same object across the two arms.
4. *FLOPs are the cost.* Violated at deployment, where latency and memory bandwidth, not FLOPs, bind.

## 3. State of the Art

**Established (with ablation).**

- Jones, *Scaling Scaling Laws with Board Games* (2021, arXiv:2104.03113), on 3×3–9×9 Hex with AlphaZero: iso-Elo curves in (train, test) compute are close to straight lines, giving a measured exchange rate near one order of magnitude of test-time compute per order of magnitude of train-time compute. This is the single cleanest measurement of $\alpha$ in existence — and it is at a scale of hours on one GPU.
- Silver et al., *Mastering the game of Go without human knowledge* (Nature, 2017): raw policy network ≈ 3055 Elo; full AlphaGo Zero with search ≈ 5185 Elo. Same weights, ~2100 Elo from search alone.
- Danihelka et al., *Policy improvement by planning with Gumbel* (ICLR 2022): policy improvement guaranteed with as few as 2 simulations; Gumbel MuZero matches MuZero on Go at drastically reduced $B$. Search's value is not monotone in $B$ in the way folklore assumes.

**Claimed, partially ablated.**

- Snell et al., *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters* (2024, arXiv:2408.03314): compute-optimal test-time scaling beats a ~14× larger model on easy/medium MATH questions, and *loses* on the hardest bin. The depth-dependence of $\alpha$ shows up here directly, but on one benchmark, one model family.
- Ruoss et al., *Amortized Planning with Large-Scale Transformers: A Case Study on Chess* (NeurIPS 2024 D&B): a 270M-parameter transformer distilled from Stockfish 16 action-values over ~15B annotated positions reaches Lichess blitz Elo ≈ 2895 against humans with **no search**. Grandmaster strength, amortized. It still loses to the teacher, and fails on long forced lines.

**Benchmark number only, no ablation.** Reported test-time-scaling curves for reasoning models (o-series, R1-style) are near-universally plotted against token count, not FLOPs, without a FLOP-matched retraining control arm. They establish that more tokens help; they say nothing about $\alpha$.

## 4. What Is Known

- Search substitutes for parameters, at a measured rate, in small board games (Hex, ≤9×9, Jones 2021).
- Distillation of search into weights works far further than expected: 2895 Elo blitz chess in one forward pass, 270M params, 15.3B action-value targets (Ruoss et al. 2024).
- The substitution is bounded: raw-network AlphaGo Zero is ~2100 Elo below its searched self (Nature 2017); Stockfish still beats the 270M distillate.
- Sampling coverage scales as a near-log-linear function of samples: SWE-bench Lite coverage 15.9% → 56% from 1 → 250 samples with DeepSeek-Coder-V2-Instruct (Brown et al., *Large Language Monkeys*, 2024). Coverage, not solve rate — selection is the bottleneck.
- Search traces are learnable: Lehnert et al., *Beyond A\*: Better Planning with Transformers via Search Dynamics Bootstrapping* (2024), Searchformer solves Sokoban tasks with ~26.8% fewer search steps than A\* after bootstrapping; Gandhi et al., *Stream of Search* (2024), show a model trained on search traces exceeds one trained only on optimal solutions.
- **Theory.** Constant-depth log-precision transformers lie in uniform $\mathsf{TC}^0$ (Merrill & Sabharwal, TACL 2023); with $T$ chain-of-thought steps they simulate $O(T)$ serial steps (Li, Liu, Zhou, Ma, ICLR 2024; Merrill & Sabharwal, ICLR 2024). So a fixed-depth amortized policy cannot represent an inherently serial depth-$d$ computation for $d$ beyond its depth, unless the circuit-class hierarchy collapses. Serial test-time compute is not a convenience; for some $d$ it is the only option.

## 5. What Is Not Known

- **Empirically open (the core gap).** Whether $\alpha$ measured on 9×9 Hex transfers to $10^{23}$-FLOP language models. Nobody has produced a FLOP-matched iso-skill frontier at frontier scale. The experiment is runnable — it costs a model-family sweep, not a new idea.
- **Empirically open.** Whether $\alpha(d)$ decays smoothly or has a cliff at the depth where the amortized policy's effective serial budget is exhausted.
- **Theoretically open.** Whether there is a natural planning-problem family with a $\mathrm{poly}(n)$-size *amortized* policy achieving $\epsilon$-optimality while every $\mathrm{poly}$-time search needs depth $\omega(1)$ — i.e. a genuine separation rather than the trivial $\mathsf{PSPACE} \not\subseteq \mathsf{P/poly}$ observation.
- **Methodologically blocked.** $d(x)$ for natural-language tasks. Without a depth metric that is not itself defined by "the model got it wrong", the depth-dependence claim is untestable outside synthetic domains.
- **Methodologically blocked.** Fair accounting of teacher search cost in distillation. The 270M chess model's $C_{\text{train}}$ excludes Stockfish's ~15B position evaluations; included, the amortized arm may be the more expensive one.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by an unaffordable control arm**.

To measure $\alpha$ you need iso-skill points at two very different $C_{\text{test}}$ values with everything else held fixed. But in every strong system, search participates in training (MuZero targets, ExIt, R1-style RL on sampled traces), so the "no-search" arm has different weights, not the same weights with search switched off. Removing that confound requires retraining a family of models per test-budget setting — an $O(k)$ multiplier on pretraining cost. That is why the only clean $\alpha$ measurement is on a board game small enough to sweep exhaustively.

Second, the benchmarks used at scale do not measure what they name. pass@$k$ with an oracle verifier measures coverage of a sampling distribution; it flatters the explicit-search arm by giving it a selector the amortized arm does not have.

## 7. Current Research (as of 2026)

- **Test-time-compute scaling laws for LLMs** — DeepMind and Berkeley lines following Snell et al.; the open item is FLOP-matched retraining controls rather than token-count curves.
- **Search-trace distillation** — Searchformer/Stream-of-Search successors; whether iterated distillation of one's own search closes the depth gap or saturates *(frontier — verify)*.
- **Learned-model fidelity as the binding constraint** — MuZero/Dreamer descendants; evidence that beyond a horizon, deeper rollouts in a learned model reduce return.
- **Adaptive compute allocation** — routing per-instance between one pass and deep search, conditioned on a difficulty estimate. Most published gains here are on benchmarks with leaked difficulty labels *(frontier — verify)*.
- **Circuit-complexity accounts of CoT depth** — Merrill, Sabharwal, and others; extending the $\mathsf{TC}^0$/serial-steps results to approximate and noisy computation.

## 8. Concrete Next Experiment

**Domain.** Sokoban, depth-stratified: instances bucketed by exact optimal solution length $d \in \{5, 10, 20, 40\}$, generated so that all buckets share state-space size and branching factor.

**Scale.** Decoder-only transformers at $N \in \{30\mathrm{M}, 100\mathrm{M}, 300\mathrm{M}, 1\mathrm{B}\}$, each trained at 3 data scales ($10^8$–$10^{10}$ transitions) — 12 checkpoints, roughly $10^{20}$–$10^{21}$ total FLOPs. Affordable on ~64 GPUs for a week.

**Arms.** For every checkpoint, evaluate at $B \in \{1, 4, 16, 64, 256, 1024\}$ beam/MCTS budget. **Control arm:** the same checkpoints at $B=1$, with $C_{\text{test}}$ raised only by longer CoT — this isolates serial thinking from external tree search. **Second control:** teacher search FLOPs charged to $C_{\text{train}}$ for the distilled arm.

**Decision number.** Fit $\alpha(d)$ per depth bucket by regressing $\log C_{\text{train}}$ on $\log C_{\text{test}}$ along the 80%-solve iso-skill contour. The question is settled by

$$\rho = \frac{\alpha(d{=}40)}{\alpha(d{=}5)}.$$

$\rho > 0.8$: the exchange rate is depth-invariant, and Jones' Hex slope is a general law — amortization scales. $\rho < 0.4$: search is irreplaceable exactly where planning is hard, and no amount of pretraining substitutes for it. Report $\rho$ with bootstrap CIs over instances; the fit needs ≥5 iso-skill points per bucket.

## 9. Key References

- **[Foundational]** Silver, Schrittwieser, Simonyan, et al. *Mastering the game of Go without human knowledge.* Nature 550, 2017.
- **[Foundational]** Anthony, Tian, Barber. *Thinking Fast and Slow with Deep Learning and Tree Search.* NeurIPS, 2017. — arXiv:1705.08439
- **[Foundational]** Schrittwieser, Antonoglou, Hubert, et al. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.* Nature 588, 2020. — arXiv:1911.08265
- **[SOTA — exchange rate]** Jones. *Scaling Scaling Laws with Board Games.* 2021. — arXiv:2104.03113
- **[SOTA — amortization]** Ruoss, Delétang, Medapati, et al. *Amortized Planning with Large-Scale Transformers: A Case Study on Chess.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2402.04494
- **[SOTA — test-time scaling]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA — sampling]** Brown, Juravsky, Ehrlich, et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA — search distillation]** Lehnert, Sukhbaatar, McVay, Rabbat, Tian. *Beyond A\*: Better Planning with Transformers via Search Dynamics Bootstrapping.* COLM, 2024. — arXiv:2402.14083
- **[SOTA — search distillation]** Gandhi, Lee, Grand, et al. *Stream of Search (SoS): Learning to Search in Language.* COLM, 2024. — arXiv:2404.03683
- **[Theory]** Merrill, Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL 11, 2023.
- **[Theory]** Li, Liu, Zhou, Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR, 2024.
- **[Method]** Danihelka, Guez, Schrittwieser, Silver. *Policy improvement by planning with Gumbel.* ICLR, 2022.

## 10. Worked Example

Take the two chess data points and try to compute an exchange rate.

- **Amortized arm** (Ruoss et al. 2024): $N = 2.7\times10^8$, $D \approx 1.5\times10^{10}$ action-value targets. Naive $C_{\text{train}} \approx 6ND \approx 2.4\times10^{19}$ FLOPs. Per-move cost $C_{\text{test}} \approx 2N \approx 5\times10^{8}$ FLOPs. Result: Lichess blitz Elo ≈ 2895.
- **Explicit arm** (AlphaZero-lineage, 800 simulations/move): per-move cost $\approx 800 \times 2N$, i.e. ~$10^{3}$× the amortized arm's test cost. Comparable or stronger play at far lower $C_{\text{train}}$.

Slope estimate: 3 orders of magnitude of test-time compute against roughly 1–2 orders of train-time compute, so $\alpha \sim 0.3$–$0.6$ — apparently much worse than Jones' Hex value near 1.

Now the obstruction. That number is not an exchange rate, because:

1. **Teacher cost is missing.** Stockfish 16 was run at 50 ms/move over $1.5\times10^{10}$ positions. At ~$10^{9}$ FLOP/s of effective search work, that is order $10^{18}$ FLOPs of *search* charged to nobody. Fold it into $C_{\text{train}}$ and the amortized arm's cost moves by a factor that is itself uncertain to ±1 order of magnitude, because Stockfish FLOPs are not measured, they are estimated from wall-clock.
2. **Skill is measured on different pools.** 2895 is Elo against *humans* on Lichess; AlphaZero Elo is against engines. The two scales are not affine-comparable — the distillate's known weakness on long forced lines is exactly the failure mode humans rarely punish and engines always do.
3. **The weights differ.** The 270M model was never trained with search in the loop; AlphaZero's were. There is no "same weights, search off" point anywhere in this comparison.

So the cleanest pair of numbers in the field yields an $\alpha$ whose error bars span the entire interesting range, and whose sign of disagreement with the Hex measurement cannot be attributed to depth, accounting, or opponent pool. That is why §8 insists on one domain, one weight family, exact $d$, and teacher FLOPs charged in.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*