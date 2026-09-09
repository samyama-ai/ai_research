---
id: 17-reasoning/minimal-sufficient-reasoning-trace-length
title: "Minimal Sufficient Reasoning Trace Length"
topic: 17-reasoning
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Minimal Sufficient Reasoning Trace Length

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/minimal-sufficient-reasoning-trace-length` · **Status:** open

## 1. Problem Statement

Given a model and a task instance, how many intermediate tokens does the model *need* to emit before its answer is correct — and is that number knowable?

- **Input:** a fixed autoregressive model $M$ with parameters $\theta$, a task instance $x$, a decoding procedure, and a correctness predicate $c(\hat y, y^\star)\in\{0,1\}$.
- **Output:** an integer $L^\star(M,x)$ — the smallest number of intermediate ("reasoning") tokens such that $M$ answers $x$ correctly with probability at least $\tau$.
- **Solving it** means either (a) computing or tightly bounding $L^\star$ *before* generation, or (b) proving that no such per-instance quantity is well defined for the class of models in use.

Three variants, of very different difficulty:

- **Measurement.** Define $L^\star$ so that two labs measuring it on the same model and instance get the same number. Currently they do not — the answer depends on whether the short trace is *forced* or *sampled*, and on what counts as a token of reasoning.
- **Method.** Train or prompt a model to emit a trace of length near $L^\star$ — spend little on easy instances, much on hard ones. This is the adaptive-compute problem, and it is partially addressed.
- **Theory.** Lower-bound the number of serial steps required for a fixed-depth transformer to decide a language, as a function of instance complexity. Circuit-complexity results give asymptotic answers for families, not for instances.

## 2. Formal Setting

Let $M_\theta$ be a decoder-only transformer of depth $d$, width $m$, and vocabulary $V$. On input $x$ it emits a trace $z = (z_1,\dots,z_L) \in V^L$ followed by an answer $\hat y$ delimited by a fixed answer marker. Write $p_\theta(z,\hat y \mid x)$.

**Success probability at budget $L$.** Measured, not assumed: sample $n$ traces under the deployment decoder (temperature $T$, top-$p$), truncate generation at $L$ trace tokens, force the answer marker, and count.

$$s_M(x, L) \;=\; \mathbb{E}_{(z,\hat y)\sim p_\theta(\cdot\mid x),\,|z|\le L}\big[\,c(\hat y, y^\star)\,\big], \qquad \hat s = \tfrac1n\sum_{i=1}^n c(\hat y_i, y^\star).$$

**Minimal sufficient trace length.**

$$L^\star_\tau(M,x) \;=\; \min\{\, L \in \mathbb{N} \;:\; s_M(x,L') \ge \tau \ \ \forall L' \ge L \,\}.$$

The $\forall L' \ge L$ clause matters: $s_M(x,\cdot)$ is **not** monotone. Overthinking makes it fall back below $\tau$ at large $L$, so the naive "first crossing" definition is unstable.

**Aggregate budget curve.** For a dataset $D$, $\bar s(L) = |D|^{-1}\sum_{x\in D} s_M(x,L)$; the reported quantity in most papers is $\bar s$, not $L^\star$ per instance.

**Excess length (overthinking).** $\Delta(M,x) = \mathbb{E}[|z|] - L^\star_\tau(M,x)$, measured in decoded tokens under the same tokenizer.

**Two non-equivalent measurement protocols**, which the literature conflates:

- *Truncation:* let $M$ generate freely, cut at $L$, force an answer. Measures whether the answer is already determined by the prefix.
- *Reconstruction:* find a short trace $z'$ (by deletion, compression, or a second model), place it in a fresh context, and measure $s$. Measures whether a short trace *suffices as evidence*.

These give different numbers on the same instance (see §10).

**Assumptions, and which are violated:**

- *The trace is the computation.* Violated. Pfau, Merrill & Bowman (COLM 2024) show filler tokens (`...`) recover accuracy on some tasks with no semantic content, so token count and computation are not interchangeable.
- *The trace is causally responsible for the answer.* Violated. Turpin et al. (NeurIPS 2023) show stated reasoning can be post-hoc; Lanham et al. (2023) show answers often survive heavy truncation.
- *Correctness is binary and well-specified.* Holds for AIME/GSM8K-style tasks; fails for open-ended reasoning, which is why nearly all measurement is on math and symbolic sets.
- *$L^\star$ is a property of $(M,x)$.* Weakly violated: it depends on the prompt, the tokenizer, and the decoder temperature, none of which are fixed across papers.

## 3. State of the Art

**Theory SOTA (established).**

- Merrill & Sabharwal (ICLR 2024): log-precision transformers with $O(\log n)$ CoT steps recognize exactly the languages in $\mathsf{L}$; with $\mathrm{poly}(n)$ steps, exactly $\mathsf{P}$. Without CoT they are confined to uniform $\mathsf{TC}^0$.
- Li, Liu, Zhou & Ma (ICLR 2024): a constant-depth transformer with $T$ CoT steps simulates any Boolean circuit of size $T$; hence CoT length buys serial depth that the architecture lacks.
- Feng et al. (NeurIPS 2023): constant-depth, log-precision transformers cannot compute arithmetic-expression evaluation directly unless $\mathsf{TC}^0=\mathsf{NC}^1$; with CoT, $O(n)$ steps suffice.

These are *family-level* results. None yields a per-instance $L^\star$.

**Empirical SOTA (established).**

- Budget forcing (Muennighoff et al., *s1: Simple test-time scaling*, 2025): appending "Wait" to extend, or the answer delimiter to truncate, moves AIME/MATH accuracy monotonically over a bounded range on Qwen2.5-32B. The control knob works; the per-instance optimum is not extracted.
- Length-controlled RL: Aggarwal & Welleck (*L1*, 2025) train a policy conditioned on a requested token budget, hitting targets closely and beating budget-forced s1 at matched length.

**Claimed but unablated.**

- "Longer reasoning is better." Widely asserted from o1/R1 curves. Ballon et al. (2025) report the opposite within a fixed model: on AIME, *longer* o3-mini traces correlate with *lower* accuracy, because length tracks instance difficulty and floundering. Correlational; no intervention arm.
- Trace-compression methods (Concise Thoughts, token-budget prompting) report token reductions with "minimal accuracy loss" on GSM8K/MATH — benchmark numbers only, usually without a matched-compute control and without checking the non-monotonicity of $s_M(x,\cdot)$.

## 4. What Is Known

- **CoT's benefit is concentrated, not general.** Sprague et al. (ICLR 2025), meta-analysis of 100+ papers plus 14 models: CoT gives large gains on math and symbolic reasoning (double-digit accuracy points) and near-zero average gain elsewhere — so $L^\star \approx 0$ for most non-symbolic tasks. Measured at 8B–175B scale.
- **Answers are often decided early.** Lanham et al. (2023, 810M–175B Claude-family models): truncating CoT frequently leaves the final answer unchanged; faithfulness of CoT to the answer is not monotone in model size and is lowest for the largest models tested.
- **Filler tokens can substitute for content.** Pfau et al. (COLM 2024): on $3\mathrm{SUM}$-style synthetic tasks, repeated `.` tokens recover much of the CoT gain, but only with dense supervision — contentless length helps in a narrow regime.
- **Overthinking is real and large.** Chen et al. (2024/2025) document o1-style models emitting hundreds of tokens on trivial arithmetic; their efficiency training cuts output tokens on MATH500 by ~49% for QwQ-32B-Preview with accuracy preserved. Measured at 32B.
- **Sequential and parallel compute trade off.** Snell et al. (2024): compute-optimal allocation across revision depth and sampling width beats a fixed policy by up to ~4× fewer FLOPs at matched accuracy on MATH, at PaLM-2-scale.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no agreed definition of $L^\star$. Truncation and reconstruction protocols disagree; "reasoning tokens" are not separable from formatting and restatement; and $s_M(x,\cdot)$ is non-monotone, so the minimum is protocol-dependent rather than intrinsic. Until this is fixed, cross-paper numbers are not comparable.
- **Theoretically open.** No instance-level lower bound. Circuit results bound worst-case length over an input family; nothing bounds the steps a *given* $\theta$ needs on a *given* $x$. Whether an instance-level bound is even computable (it looks Kolmogorov-like, hence uncomputable in general) is unsettled.
- **Empirically open.** Nobody has published a per-instance $L^\star$ distribution — the histogram of minimal sufficient lengths over a full benchmark for one frontier model, with the truncation/reconstruction gap quantified. The experiment is runnable today; it is a sampling cost, not a research obstacle.
- **Open:** whether a model's own internal states predict $L^\star$ before decoding starts. Confidence probes exist; none has been evaluated against a rigorously measured $L^\star$.

## 6. Why It Is Hard

- **Non-identifiability.** The trace is both computation and evidence. If a short trace fails, you cannot tell whether the computation needed more steps or the model merely needed more tokens to *condition* itself. These have different fixes and no experiment separates them without intervening on activations.
- **Confounded measurement.** Length correlates with difficulty, with failure mode (looping), and with RL training artifacts simultaneously. The Ballon et al. negative correlation is fully explained by difficulty confounding unless length is randomized — and randomizing length changes the distribution the model was trained on.
- **Cost.** A credible $L^\star$ estimate needs a binary search over $L$ with $n\ge 32$ samples per $(x,L)$ point. At ~10 budget points and 500 instances that is $1.6\times10^5$ long generations per model, before ablations.
- **The evaluation does not measure what it names.** Aggregate accuracy-vs-budget curves are named "test-time scaling" but are averages over a heterogeneous mixture of instances; the mean of $L^\star$ tells you nothing about its tail, which is what a deployment budget must cover.

## 7. Current Research (as of 2026)

- **Length-controlled RL** — length as a conditioning variable or a reward penalty (CMU/Welleck's L1 line; concurrent work from Qwen and DeepSeek teams on length-penalized GRPO). Established as a control mechanism, not as an $L^\star$ estimator.
- **Latent / implicit reasoning** — Coconut (Hao et al., 2024), CoT internalization (Deng et al., 2024), pause tokens (Goyal et al., ICLR 2024). If reasoning moves into continuous state, token-counted $L^\star$ stops being the right variable at all. *(frontier — verify)*
- **Difficulty-conditioned routing** — predicting a budget from the prompt and dispatching to a short or long policy; deployed in commercial "thinking effort" controls. Published ablations are thin. *(frontier — verify)*
- **Faithfulness auditing** as a measurement primitive — Anthropic's line from Lanham/Turpin, now applied to reasoning models; the natural home for a standardized truncation protocol.

## 8. Concrete Next Experiment

**The $L^\star$ histogram, with both protocols.**

- **Scale.** One open reasoning model at 32B (e.g. QwQ-32B or a DeepSeek-R1 distill, so weights are fixed and reproducible). $|D| = 500$ instances spanning MATH500 (easy tail) and AIME 2024–2025 (hard tail). Budgets $L \in \{0, 32, 64, 128, 256, 512, 1024, 2048, 4096\}$ trace tokens. $n = 32$ samples per $(x,L)$, temperature 0.6, $\tau = 0.9$. Cost: ~$1.4\times10^5$ generations, order 10 GPU-days on 8×H100.
- **Arm A (truncation).** Free generation, hard cut at $L$, forced answer marker.
- **Arm B (reconstruction).** Greedy sentence-level deletion from a correct full trace to find the shortest surviving subsequence; re-evaluate that trace in a fresh context.
- **Control arm.** Filler tokens: replace the trace with $L$ copies of a neutral token, same budgets, same $n$. This isolates the length effect from the content effect, following Pfau et al.
- **The deciding number.** The rank correlation $\rho$ between $L^\star_{0.9}$ measured under Arm A and under Arm B, over the 500 instances. If $\rho \ge 0.8$, $L^\star$ is a stable per-instance property and the field can standardize on either protocol and start predicting it. If $\rho \le 0.4$, minimal sufficient length is a protocol artifact, and every published "token reduction with minimal accuracy loss" claim needs restating with its protocol attached. Secondary number: the fraction of instances where the filler control matches Arm A at the same $L$ — the share of apparent reasoning that is length, not content.

## 9. Key References

- **[Foundational]** Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., Zhou, D. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS, 2022. — arXiv:2201.11903
- **[Foundational]** Nye, M., et al. *Show Your Work: Scratchpads for Intermediate Computation with Language Models.* 2021. — arXiv:2112.00114
- **[Theory]** Merrill, W., Sabharwal, A. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024. — arXiv:2310.07923
- **[Theory]** Li, Z., Liu, H., Zhou, D., Ma, T. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR, 2024. — arXiv:2402.12875
- **[Theory]** Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., Wang, L. *Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective.* NeurIPS, 2023. — arXiv:2305.15408
- **[Measurement]** Lanham, T., et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic technical report, 2023. — arXiv:2307.13702
- **[Measurement]** Turpin, M., Michael, J., Perez, E., Bowman, S. R. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS, 2023. — arXiv:2305.04388
- **[Measurement]** Pfau, J., Merrill, W., Bowman, S. R. *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models.* COLM, 2024. — arXiv:2404.15758
- **[SOTA]** Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Candès, E., Hashimoto, T. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- **[SOTA]** Aggarwal, P., Welleck, S. *L1: Controlling How Long a Reasoning Model Thinks with Reinforcement Learning.* 2025. — arXiv:2503.04697
- **[SOTA]** Snell, C., Lee, J., Xu, K., Kumar, A. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025. — arXiv:2501.12948
- **[Survey]** Sprague, Z., Yin, F., Rodriguez, J. D., Jiang, D., Wadhwa, M., Singhal, P., Zhao, X., Ye, X., Mahowald, K., Durrett, G. *To CoT or Not to CoT? Chain-of-Thought Helps Mainly on Math and Symbolic Reasoning.* ICLR, 2025. — arXiv:2409.12183
- **[Survey]** Sui, Y., et al. *Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models.* 2025. — arXiv:2503.16419
- **[Related]** Chen, X., et al. *Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs.* 2025. — arXiv:2412.21187
- **[Related]** Goyal, S., Ji, Z., Rawat, A. S., Menon, A. K., Kumar, S., Nagarajan, V. *Think Before You Speak: Training Language Models with Pause Tokens.* ICLR, 2024. — arXiv:2310.02226
- **[Related]** Ballon, M., Algaba, A., Ginis, V. *The Relationship Between Reasoning and Performance in Large Language Models — o3-mini Solves It Right.* 2025.

## 10. Worked Example

One GSM8K-style instance, run under both protocols on a 32B reasoning model. Illustrative numbers of the kind this measurement produces.

> *A shop sells pens at \$3 and notebooks at \$7. Maya buys 4 pens and some notebooks, spending \$54 in total. How many notebooks?*

Free generation emits a 213-token trace: restates the problem (38 tokens), computes $4\times3=12$ (21), sets $54-12=42$ (24), divides $42/7=6$ (19), then verifies twice and restates (111).

**Arm A (truncation), $\hat s$ over $n=32$:**

| $L$ | 0 | 32 | 64 | 128 | 213 (full) |
|---|---|---|---|---|---|
| $\hat s$ | 0.44 | 0.53 | 0.91 | 1.00 | 1.00 |

First crossing of $\tau=0.9$ is at $L=64$. So $L^\star_A = 64$.

**Arm B (reconstruction).** Greedy sentence deletion yields a 41-token trace: `4*3=12. 54-12=42. 42/7=6.` Re-scored in a fresh context, $\hat s_B = 0.72$ — *below* $\tau$. Restoring the problem restatement (79 tokens total) gives $\hat s_B = 0.94$. So $L^\star_B = 79$.

**Control (filler).** 64 copies of `.` in place of the trace: $\hat s = 0.47$, statistically indistinguishable from $L=0$. Length alone buys nothing here; content does.

**The obstruction, made visible.** $L^\star_A = 64 < L^\star_B = 79$, and the 41-token trace — which contains every arithmetic step needed — is *insufficient* as evidence while being *sufficient* as computation. The gap is not noise: the deleted 38 tokens were pure restatement, carrying no derivation. Whatever they do, they are not "reasoning steps," yet removing them costs 22 accuracy points. Any definition of $L^\star$ that counts tokens will therefore either include them (and overstate the reasoning) or exclude them (and understate the budget). That is why §5 classes this as methodologically blocked before it is empirically open: the histogram in §8 is cheap to produce, but it is not interpretable until the two protocols are shown to agree.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*