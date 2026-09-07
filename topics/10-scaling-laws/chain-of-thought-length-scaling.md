---
id: 10-scaling-laws/chain-of-thought-length-scaling
title: "Scaling Behavior of Chain-of-Thought Length"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Behavior of Chain-of-Thought Length

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/chain-of-thought-length-scaling` · **Status:** empirically-open

## 1. Problem Statement

Chain-of-thought (CoT) turns serial token generation into computation: a model emits $T$ intermediate tokens before its answer, so inference FLOPs scale roughly as $2NT$ for an $N$-parameter dense model. The question is what the return curve on $T$ looks like, and whether it is a *law* in the sense that Kaplan/Chinchilla loss curves are laws.

Three variants, routinely conflated:

- **Measurement.** Given a fixed model and task distribution, what is the functional form of accuracy versus generated reasoning length $T$? Power law, exponential-saturating, or non-monotone with a peak?
- **Method.** Given a joint budget $C = C_{\text{train}} + C_{\text{inference}}$, how should compute be split between parameters $N$, training tokens $D$, and per-query serial length $T$? Is there a Chinchilla-style optimality condition for $T$?
- **Theory.** Does the *required* CoT length for a problem family scale with a known complexity parameter (circuit depth, serial time), and does a trained model's emitted length track that requirement?

A solution to the measurement variant is a fitted form $\mathrm{Acc}(T)$ with held-out predictive validity across at least a decade of $T$ and a decade of $N$. A solution to the method variant is an allocation rule $T^\star(N, C, \text{task})$ that beats fixed-budget baselines at a scale not used for fitting. The theory variant is separately open and mostly disjoint from both.

## 2. Formal Setting

Let $p_\theta$ be an autoregressive model with $N$ non-embedding parameters. On query $x$, it samples a reasoning trace $z \sim p_\theta(\cdot \mid x)$ and answer $y$. Define:

- **Emitted length** $T(x) = |z|$, measured in generated tokens up to the answer delimiter — not words, not "steps". Tokenizer-dependent; comparisons across tokenizers require normalization by bytes per token.
- **Serial budget** $B$: a cap enforced by truncation, by "budget forcing" (appending a stop or a `Wait` token), or by prompt instruction. These are not interchangeable — instruction-set budgets are not obeyed.
- **Inference cost** $C_{\text{inf}} \approx 2N(T + |x|) + \text{attention terms} \; \Theta(T^2 d)$. The quadratic term is nonneglible past $T \sim 10^4$ and is where the KV cache dominates wall-clock.
- **Task success** $\mathrm{Acc}(B) = \mathbb{E}_{x}\big[\mathbb{1}\{\hat y(x; B) = y^\star(x)\}\big]$, with the verifier fixed and stated (exact match, unit tests, or a graded proof checker).

Two competing parametric families are in play:

$$\mathrm{Acc}(B) = a - b\,B^{-\alpha} \qquad \text{versus} \qquad \mathrm{Acc}(B) = a\big(1 - e^{-\lambda B}\big) - c\,B$$

The second admits a maximum at finite $B$ — the "overthinking" regime. Distinguishing them requires data past the peak, which most published sweeps do not have.

For parallel scaling the natural quantity is **coverage**, $\mathrm{cov}(k) = \mathbb{E}_x[\mathbb{1}\{\exists i \le k: \hat y_i = y^\star\}]$, i.e. pass@$k$. Serial and parallel scaling are different axes and must be plotted against a common $C_{\text{inf}}$, not against $k$ and $T$ separately.

**Assumptions known to be violated in practice.** (i) That $T$ is an exogenous control variable — it is not; RL-trained reasoning models choose $T$ as a function of perceived difficulty, so $\mathrm{Acc}(T)$ measured by conditioning on emitted length is confounded by difficulty selection. (ii) That the trace is causally load-bearing — traces can be post-hoc, and correct answers occur with invalid traces. (iii) That the verifier is exact — LLM judges and answer-extraction regexes contribute several points of noise at the margin where these curves are decided. (iv) i.i.d. task difficulty within a benchmark: AIME-style sets have 15–30 items, so single-run accuracy has a standard error near 10 points.

## 3. State of the Art

**Established (ablated, reproduced).**
- CoT prompting raises accuracy on multi-step arithmetic and symbolic tasks, with the gain emerging only above roughly $10^{10}$ parameters (Wei et al., NeurIPS 2022). Reproduced widely.
- Self-consistency (parallel sampling + majority vote) improves GSM8K over greedy CoT by ~10–18 points at PaLM-540B scale (Wang et al., ICLR 2023).
- Coverage grows log-linearly in the number of independent samples across four orders of magnitude of $k$ (Brown et al., 2024, arXiv:2407.21787). This is the best-characterized inference-scaling regularity that exists.
- Test-time compute allocated by a process reward model plus adaptive revision beats best-of-$N$ at matched FLOPs, up to ~$4\times$ efficiency on MATH (Snell et al., 2024, arXiv:2408.03314).

**Claimed but not cleanly ablated.**
- That RL-induced length growth *causes* the accuracy gain. DeepSeek-R1-Zero shows average response length rising from hundreds to $\sim10^4$ tokens during RL while AIME 2024 pass@1 goes 15.6% → 71.0% (DeepSeek-AI, *Nature*, 2025; arXiv:2501.12948). Length and capability move together; no arm holds length fixed while training.
- That "budget forcing" extrapolates. s1 reports AIME24 gains of a few points from appended `Wait` tokens (Muennighoff et al., 2025, arXiv:2501.19393), with the authors themselves noting saturation after a small number of extensions.
- **Benchmark-number-only:** o1/o3-class results are reported as accuracy-versus-test-time-compute plots with unlabeled axes (Jaech et al., 2024, arXiv:2412.16720). No exponent, no reproducible protocol. Treat as existence claims, not measurements.

**Theory SOTA** is separate and stronger in kind. Constant-depth transformers with $T$ CoT steps simulate size-$T$ boolean circuits, so polynomial CoT strictly exceeds $\mathsf{TC}^0$ under standard assumptions (Li, Liu, Zhou, Ma, ICLR 2024, arXiv:2402.12875). Log-precision transformers with $\Theta(\log n)$ CoT steps characterize $\mathsf{L}$, and with polynomial steps characterize $\mathsf{P}$ (Merrill & Sabharwal, ICLR 2024).

## 4. What Is Known

- **Serial length buys expressivity, provably.** Feng et al. (NeurIPS 2023) show a constant-size autoregressive transformer with CoT solves arithmetic and linear-equation tasks that a fixed-depth direct-answer transformer cannot without width polynomial in input length.
- **Coverage exponent.** On SWE-bench Lite, resolve rate rises from 15.9% (one attempt) to 56% with 250 samples from the same model (Brown et al., 2024) — measured at DeepSeek-Coder-V2-Instruct scale. Log-linear over the full sweep.
- **Small-model-plus-compute can beat big-model-direct.** Snell et al. (2024) report a small PaLM-2-class model with optimized test-time compute matching or exceeding a ~14× larger model on easy/medium MATH questions — but *not* on the hardest bucket, where pretraining scale still wins.
- **Overthinking is real and measurable.** On simple arithmetic ("2+3=?"), o1-like models emit up to ~$1.9\times$ more tokens than needed with no accuracy gain (Chen et al., 2024, arXiv:2412.21187).
- **Length collapse at the difficulty cliff.** Shojaee et al. (2025, Apple) report that on controllable puzzle families, reasoning-token count *decreases* as problems pass a complexity threshold, even with budget remaining — measured on Claude 3.7 Sonnet Thinking and DeepSeek-R1 class models.

## 5. What Is Not Known

- **Empirically open.** The functional form of $\mathrm{Acc}(B)$ under *exogenous* budget control, swept over $B \in [10^2, 10^5]$ tokens and $N \in [1\mathrm{B}, 100\mathrm{B}]$, with difficulty stratified. Runnable today on open weights; nobody has published the full grid. The joint train/inference allocation frontier $T^\star(N, C)$ is likewise runnable and unrun.
- **Theoretically open.** Whether there is any lower bound tying required $T$ for a *learned* model to a complexity measure of the task. Existing results are constructive upper bounds on expressivity; they say nothing about what gradient descent produces.
- **Methodologically blocked.** Whether a trace's length is causally load-bearing. There is no accepted measure separating "computation carried in the trace" from "computation carried in the forward pass, with the trace as scaffolding". Until that measure exists, $\mathrm{Acc}(T)$ curves are correlational.

## 6. Why It Is Hard

The central obstruction is **confounded measurement via difficulty selection**. Reasoning models allocate length endogenously: hard items get long traces, and hard items are also the ones they fail. Any scatter of accuracy against emitted $T$ therefore slopes *downward*, the opposite of the causal effect. Forcing $T$ removes the confound but introduces a different one — a forced budget is off-policy relative to the RL-trained length distribution, so the model is being evaluated outside its training support.

Second obstruction: **cost asymmetry**. A single point on the grid at $N = 32\mathrm{B}$, $B = 64\mathrm{k}$, 30 items, 64 seeds is $\sim10^{18}$ FLOPs of generation and hours of serial decode that cannot be batched away, because the KV cache grows as $\Theta(T)$ per sequence. Filling a $5 \times 5$ $(N, B)$ grid with enough seeds to resolve a 3-point difference on a 30-item benchmark is a several-hundred-GPU-day job — small for pretraining, large for an evaluation nobody is paid to run.

Third: **benchmark granularity**. AIME 2024 has 30 problems. A one-item change is 3.3 points. Most published length-scaling deltas are within one or two items.

## 7. Current Research (as of 2026)

- **Length-penalized RL.** Adding a token cost to the RL objective to cut overthinking while holding accuracy — pursued at DeepSeek, Qwen, and in the open-weight reproduction community. Reported to cut mean length substantially at flat accuracy on math sets *(frontier — verify)*.
- **Adaptive routing.** Predicting per-query $T$ from a difficulty estimator and short-circuiting easy items; the natural test of whether endogenous allocation is any good.
- **Latent / continuous CoT.** Recurrent-depth and looped-transformer variants that add serial compute without emitting tokens, which would decouple "reasoning" from "length" and make the measurement problem tractable *(frontier — verify)*.
- **Faithfulness measurement.** Anthropic-style work on whether traces mention the cues that actually drive the answer; directly relevant to the methodological block in §5.

## 8. Concrete Next Experiment

**Question:** under exogenous budget control, is $\mathrm{Acc}(B)$ monotone-saturating or peaked?

- **Scale.** Three open reasoning models spanning a decade of parameters: ~1.5B, ~7B, ~32B (e.g. the R1-distill family, whose training recipe is public). Budgets $B \in \{256, 1\mathrm{k}, 4\mathrm{k}, 16\mathrm{k}, 64\mathrm{k}\}$ tokens, enforced by hard truncation plus a forced answer prefix so every arm produces a parseable answer. Tasks stratified into three a-priori difficulty bins of 200 items each — enough that 1 item is 0.5 points — drawn from a program-generated family (multi-digit multiplication, Tower-of-Hanoi at varied $n$) where required serial steps are known analytically. 32 seeds per cell. Total: $3 \times 5 \times 600 \times 32 \approx 2.9\times10^5$ generations.
- **Control arm.** Matched-FLOPs parallel sampling: at each $C_{\text{inf}}$, spend it on $k = C_{\text{inf}} / (2N \bar T_{\text{free}})$ independent short traces with majority vote, instead of one long trace. This isolates serial from parallel compute.
- **Deciding number.** The fitted curvature sign of $\mathrm{Acc}$ against $\log B$ in the hardest difficulty bin, at 32B. If the fitted peak $B^\star$ is finite with the upper confidence bound below $64\mathrm{k}$ in $\ge 2$ of 3 model scales, the saturating power law is falsified and CoT length has an optimum — which makes $T$ a hyperparameter to tune, not a resource to buy. If $\mathrm{Acc}$ is monotone with slope $> 0$ at $B = 64\mathrm{k}$ and the serial arm dominates the matched-FLOPs parallel arm, serial compute is a genuine scaling axis and the allocation question in §1 becomes the priority.

## 9. Key References

- **[Foundational]** Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le, Zhou. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS, 2022. — arXiv:2201.11903
- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Theory]** Li, Liu, Zhou, Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR, 2024. — arXiv:2402.12875
- **[Theory]** Merrill, Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024. — arXiv:2310.07923
- **[Theory]** Feng, Zhang, Gu, Ye, He, Wang. *Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective.* NeurIPS, 2023. — arXiv:2305.15408
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025. — arXiv:2501.12948
- **[SOTA]** Muennighoff, Yang, Shi, Li, Fei-Fei, Hajishirzi, Zettlemoyer, Liang, Candès, Hashimoto. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- **[Counterpoint]** Chen et al. *Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs.* 2024. — arXiv:2412.21187
- **[Counterpoint]** Shojaee, Mirzadeh, Alizadeh, Horton, Bengio, Farajtabar. *The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity.* Apple, 2025.
- **[Related]** Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR, 2023. — arXiv:2203.11171
- **[Related]** Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050

## 10. Worked Example

Take a 32B reasoning model, $2N \approx 6.4\times10^{10}$ FLOPs per generated token. Fix a per-query budget of $C_{\text{inf}} = 1.0\times10^{15}$ FLOPs — about $1.6\times10^{4}$ generated tokens.

Two ways to spend it on one AIME problem:

| Arm | Allocation | Tokens each | Samples |
|---|---|---|---|
| Serial | one trace, budget-forced | 16,000 | 1 |
| Parallel | short traces + majority vote | 2,000 | 8 |

Now the obstruction. Run both on AIME 2024 (30 items) and suppose serial scores 21/30 = 70.0% and parallel scores 19/30 = 63.3%. The 6.7-point gap is **two problems**. The binomial standard error at $p=0.67$, $n=30$ is $\sqrt{0.67 \cdot 0.33/30} = 8.6$ points; a paired McNemar test on 2 discordant pairs has $p \approx 0.5$. The measurement cannot distinguish the arms.

Push to significance and the cost appears. To resolve a 5-point difference at 80% power you need roughly $n \approx 600$ paired items, or 30 items × 32 seeds with per-seed variance accounted. That is $600 \times 2 \times 1.0\times10^{15} = 1.2\times10^{18}$ FLOPs for *one cell* of the grid in §8 — before the $\Theta(T^2)$ attention term, which at $T = 1.6\times10^4$ and $d = 5120$ adds ~$10^{12}$ FLOPs per sequence and, more importantly, a KV cache of order 10 GB per long sequence that caps batch size and makes the serial arm wall-clock-bound rather than FLOP-bound.

Separately, condition on emitted length in the *unforced* arm. Traces above the median length will score worse than traces below it — perhaps 45% versus 80% — because the model writes more when it is stuck. Read naively, that says CoT length hurts. It says nothing of the kind: it is difficulty selection, and it is why the forced-budget design is mandatory and why the honest status of this problem is empirically open rather than settled by the plots already in circulation.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*