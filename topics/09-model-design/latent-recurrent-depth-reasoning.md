---
id: 09-model-design/latent-recurrent-depth-reasoning
title: "Latent Recurrent Depth as a Substitute for Chain of Thought"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Latent Recurrent Depth as a Substitute for Chain of Thought

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/latent-recurrent-depth-reasoning` · **Status:** empirically-open

## 1. Problem Statement

Chain of thought (CoT) buys extra serial computation by writing intermediate tokens into the context and re-reading them. Latent recurrent depth buys it by iterating a weight-tied block on the hidden state, emitting nothing. The question is whether the second can replace the first.

Three variants, often conflated:

- **Measurement.** Given a fixed inference FLOP budget and a fixed task, does a recurrent-depth model at recurrence count $r$ beat a same-budget CoT model? "Same budget" is the whole difficulty: latent depth pays per *context token*, CoT pays per *generated token*, and these scale differently with prompt length.
- **Method.** Is there a training recipe — curriculum over $r$, truncated backprop, latent-space RL — that makes latent iterations carry task-relevant serial state rather than converge to a fixed point after 4–8 steps?
- **Theory.** Does $r$ recurrences of a $k$-layer block have the same expressive power as $r$ CoT steps of the same block, at matched precision and matched per-step width?

Solved would mean: a recurrent-depth model that, at iso-FLOP and iso-training-tokens, matches or beats a CoT baseline on a serial-reasoning benchmark suite, with an ablation showing the gain comes from the recurrence and not from the extra parameters or data.

## 2. Formal Setting

A recurrent-depth LM factors as prelude $P_\theta$, core $R_\theta$, coda $C_\theta$, with layer counts $\ell_P, \ell_R, \ell_C$ and width $d$. For input embeddings $e \in \mathbb{R}^{n \times d}$:

$$s_0 \sim \mathcal{N}(0, \sigma^2 I), \qquad s_i = R_\theta(s_{i-1}, e_P), \; i = 1..r, \qquad e_P = P_\theta(e), \qquad y = C_\theta(s_r)$$

**Measured quantities.**

- *Serial depth.* $D_{\text{lat}} = \ell_P + r\,\ell_R + \ell_C$ layer applications per token. For CoT with $m$ thought tokens, $D_{\text{cot}} = m \cdot L$ where $L$ is total depth — but only along the causal chain; parallel-decodable prefixes do not count.
- *Inference FLOPs.* With $N_P, N_R, N_C$ non-embedding parameters per block, and $T$ total tokens processed (prompt + answer),
$$F_{\text{lat}} \approx 2T\,(N_P + rN_R + N_C), \qquad F_{\text{cot}} \approx 2(T + m)\,N.$$
Attention terms $O(n^2 d)$ are added separately; for the recurrent model they are incurred $r$ times.
- *Convergence.* $\delta_i = \lVert s_i - s_{i-1}\rVert_2 / \lVert s_i \rVert_2$, measured per token position. The recurrence is *doing work* only where $\delta_i$ stays above the noise floor.
- *Depth–accuracy curve.* $A(r)$ on a held-out task, with $r^\star = \min\{r : A(r) \ge 0.99\max_{r'} A(r')\}$ the saturation depth.

**Assumptions, and which fail.** (i) *Weight tying is not a capacity bottleneck* — violated: Saunshi et al. (ICLR 2025) find looped models lag full-depth models on memorization-heavy tasks even as they match on reasoning. (ii) *$A(r)$ is monotone in $r$* — violated in practice; several recurrent models degrade past the training-time $r$ distribution. (iii) *Iso-FLOP is the right control* — questionable, because latent depth is fully parallel over the prompt while CoT is serial, so wall-clock and FLOPs rank the arms differently. (iv) *The latent state is bounded like a CoT scratchpad* — false in the other direction: $s_i \in \mathbb{R}^{n\times d}$ is far larger than $m$ tokens of text.

## 3. State of the Art

**Empirical SOTA.** Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (2025, arXiv:2502.05171) — Huginn-0125, 3.5B parameters, 800B training tokens, $\ell_P{=}2$, $\ell_R{=}4$, $\ell_C{=}2$, $r$ sampled from a log-normal-Poisson with mean 32 during training. *Established:* accuracy rises with $r$ at test time on GSM8K, ARC-C and OpenBookQA and saturates near $r\approx 32$; the model can be trained stably with truncated backprop through the last $k\approx 8$ iterations. *Claimed but unablated:* that this is a substitute for CoT. No arm in the paper is an iso-FLOP CoT model of the same data budget; the comparisons are to open models at different token counts. The headline is a benchmark number, not a controlled contrast.

**Theory SOTA.** Merrill & Sabharwal, *The Expressive Power of Transformers with Chain of Thought* (ICLR 2024): log-precision transformers with $\text{poly}(n)$ CoT steps recognize exactly $\mathsf{P}$; with $O(\log n)$ steps they stay near $\mathsf{TC}^0$. Li et al., *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems* (ICLR 2024): constant-depth, $\text{poly}$-CoT transformers are $\mathsf{P}$-complete. Giannou et al., *Looped Transformers as Programmable Computers* (ICML 2023): a fixed looped transformer with $O(1)$ layers simulates an instruction-set computer, so unbounded looping is also Turing-universal in the limit. The classes coincide asymptotically; the open question is entirely about constants, learnability, and finite $r$.

**Adjacent, established.** Pfau, Merrill & Bowman, *Let's Think Dot by Dot* (COLM 2024): filler tokens give real gains on $3\mathrm{SUM}$-style tasks but require dense parallelizable supervision to learn — evidence that "silent" extra compute is hard to train. Goyal et al., *Pause Tokens* (ICLR 2024): +18% EM on SQuAD for a 1B model, but only when pause tokens are used in pretraining *and* finetuning.

## 4. What Is Known

- **Test-time depth scaling is real at 3.5B.** Huginn's accuracy is monotone in $r$ over $r \in [1, 32]$ on GSM8K/ARC-C and flat to slightly down past $r\approx 32$–$64$. Scale: one 3.5B model, 800B tokens.
- **Looping matches full depth on reasoning, not on memory.** Saunshi et al. (ICLR 2025): a $k$-layer block looped $L$ times is close to a $kL$-layer model on reasoning benchmarks while lagging on knowledge recall, at 1B-scale pretraining.
- **Depth is required for state tracking.** Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024): constant-depth log-precision architectures cannot solve $S_5$ word problems; $\Omega(\log n)$ serial depth is needed. This is the crisp task family where $r$ must grow with $n$.
- **Latent chains work at small scale and degrade at large.** Hao et al., *Coconut* (2024, arXiv:2412.06769): feeding the last hidden state back as the next "thought" embedding beats CoT on ProsQA at GPT-2 scale with fewer generated tokens; no reproduction at $\ge$7B.
- **Latent reasoning is unmonitorable.** Korbak et al., *Chain of Thought Monitorability* (2025, arXiv:2507.11473): externalized reasoning is a safety-relevant observable that latent recurrence removes by construction.

## 5. What Is Not Known

- **Empirically open (the main gap).** No published iso-FLOP, iso-data, iso-recipe comparison of recurrent depth against CoT at $\ge$7B on a serial-reasoning suite. Every ingredient exists; nobody has paid for the paired run.
- **Empirically open.** Whether latent depth composes with RL-trained reasoning (the DeepSeek-R1-style regime) or competes with it. All latent-depth results are pretraining-only.
- **Theoretically open.** Whether $r$ recurrences of a $k$-layer block at $p$-bit precision can simulate $r$ CoT steps of the same block *with only polynomial overhead in $k$ and $p$*. Both are $\mathsf{P}$ in the limit; the finite-$r$ separation is unproved in either direction.
- **Methodologically blocked.** "Serial reasoning depth required by a problem instance" has no accepted measure. Without it, $A(r)$ curves cannot be predicted, only fitted, and "saturation at $r{=}32$" cannot be distinguished from "the benchmark needs no more than 32".

## 6. Why It Is Hard

**The primary obstruction is confounded measurement.** A recurrent model pays $r{\times}$ core cost on every prompt token; a CoT model pays $1{\times}$ on the prompt and $1{\times}$ per thought token. Iso-FLOP therefore depends on the prompt/thought length ratio of the specific benchmark, so the same two models can be declared winner and loser by choosing MATH (long thoughts, short prompts) versus long-context QA. There is no benchmark-independent exchange rate.

Second: **non-identifiability of the mechanism.** A depth-scaling curve $A(r)$ is equally consistent with "iterations carry serial state" and "iterations denoise a fixed point that the coda reads better". $\delta_i$ distinguishes these, but only if one can rule out that the residual drift is task-irrelevant.

Third: **cost.** Deciding the question needs paired pretraining runs at $\ge$7B and $\ge$1T tokens, roughly $10^{22}$–$10^{23}$ FLOPs per arm. That is why the comparison is unrun rather than unrunnable.

## 7. Current Research (as of 2026)

- **Recurrent-depth pretraining.** Geiping and collaborators (Maryland/ELLIS/Tübingen, with ORNL compute) continue on the Huginn line; the reported follow-up direction is per-token adaptive $r$ with an exit criterion on $\delta_i$ *(frontier — verify)*.
- **Looped-transformer theory.** Saunshi, Reddi and colleagues (Google Research) on the looping inductive bias and its separation from parameter count.
- **Weight-tied MoE depth.** Csordás et al., *MoEUT* (NeurIPS 2024) — making Universal-Transformer-style tying compute-competitive by widening each shared layer.
- **Latent-thought decoding.** Coconut-line work on continuous thought vectors; open reproductions at 7B remain thin *(frontier — verify)*.
- **Safety pushback.** Multiple labs have argued against latent reasoning on monitorability grounds (Korbak et al. 2025); this is now a live constraint on whether the substitution is *wanted* even if it works.

## 8. Concrete Next Experiment

**Scale.** Two 7B-class models, identical data (1T tokens), identical tokenizer, optimizer and schedule.

- **Arm A (latent).** $\ell_P{=}2$, $\ell_R{=}4$, $\ell_C{=}2$, $d{=}4096$, $r\sim$ log-normal-Poisson with mean 16, truncated backprop over the last 8 iterations.
- **Arm B (control, CoT).** A standard dense transformer with non-embedding parameter count matched to Arm A's *unrolled mean* cost — i.e. sized so that per-token forward FLOPs equal Arm A at $r{=}16$ — trained on the same corpus, then given a fixed 256-token thought budget at eval.
- **Arm C (cheap control).** Arm A's architecture at $r{=}1$, to separate the recurrence from the tying.

**Eval.** $S_5$ word problems at sequence lengths $n \in \{16, 64, 256\}$ (ground-truth serial depth $\Theta(\log n)$), plus GSM8K and MATH-500, scored under a *shared FLOP meter* that charges attention and prompt processing identically to both arms.

**The deciding number.** The iso-FLOP accuracy gap $\Delta = A_{\text{lat}}(r^\star) - A_{\text{cot}}(256)$ on $S_5$ at $n{=}256$, at the FLOP budget where both arms are at 90% of their own saturation. $\Delta \ge +5$ points means latent depth substitutes for CoT on the task class where serial depth is provably required. $\Delta \le -5$ points means it does not, and the Huginn depth-scaling curve is a capacity effect, not a reasoning effect. $|\Delta| < 5$ leaves the question open and shifts it to the monitorability trade-off.

## 9. Key References

- **[Foundational]** Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser. *Universal Transformers.* ICLR, 2019. — arXiv:1807.03819
- **[Foundational]** Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le, Zhou. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS, 2022. — arXiv:2201.11903
- **[SOTA]** Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* 2025. — arXiv:2502.05171
- **[SOTA]** Saunshi, Dikkala, Li, Kumar, Reddi. *Reasoning with Latent Thoughts: On the Power of Looped Transformers.* ICLR, 2025. — arXiv:2502.17416
- **[Theory]** Merrill, Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024. — arXiv:2310.07923
- **[Theory]** Li, Liu, Zhou, Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR, 2024. — arXiv:2402.12875
- **[Theory]** Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[Theory]** Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos. *Looped Transformers as Programmable Computers.* ICML, 2023. — arXiv:2301.13196
- **[Empirical]** Pfau, Merrill, Bowman. *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models.* COLM, 2024. — arXiv:2404.15758
- **[Empirical]** Goyal, Ji, Rawat, Menon, Kumar, Nagarajan. *Think before you speak: Training Language Models With Pause Tokens.* ICLR, 2024. — arXiv:2310.02226
- **[Empirical]** Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian. *Training Large Language Models to Reason in a Continuous Latent Space.* 2024. — arXiv:2412.06769
- **[Empirical]** Csordás, Irie, Schmidhuber, Potts, Manning. *MoEUT: Mixture-of-Experts Universal Transformers.* NeurIPS, 2024. — arXiv:2405.16039
- **[Position]** Korbak et al. *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.* 2025. — arXiv:2507.11473

## 10. Worked Example

Take Huginn-sized geometry: $d = 4096$, so each transformer layer holds about $12d^2 \approx 201$M parameters. Prelude+coda = 4 layers $\approx 805$M; core = 4 layers $\approx 805$M.

Per-token forward FLOPs (ignoring attention):

$$F_{\text{lat}}(r) = 2\,(0.805 + 0.805r)\times10^9 \;\Rightarrow\; F_{\text{lat}}(32) \approx 53 \text{ GFLOP/token}.$$

A dense 3.5B CoT model costs $F_{\text{cot}} = 2 \times 3.5\times10^9 = 7$ GFLOP/token.

Now price one GSM8K item: 100 prompt tokens, 20 answer tokens, and for the CoT arm 200 thought tokens.

| Arm | Tokens charged | FLOPs |
|---|---|---|
| Latent, $r=32$ | 120 | $120 \times 53\text{G} = 6.4$ TFLOP |
| CoT, 200 thoughts | 320 | $320 \times 7\text{G} = 2.2$ TFLOP |

The latent arm costs **2.9× more** for this item. Huginn's reported GSM8K accuracy at $r{=}32$ (8-shot, CoT prompting) is roughly 40% — in the same band as 7B-class open models that answer this item for 2.2 TFLOP.

The obstruction is now visible and it is arithmetic, not rhetoric. Latent depth is charged on the prompt, so its cost scales with $n \cdot r$ while CoT's scales with $m$. Push the prompt to 1,000 tokens and the latent arm's bill rises 8× while the CoT arm's rises 1.9×; shrink the prompt to 20 tokens and the ranking inverts. The published depth-scaling curve is therefore compatible with latent recurrence being *either* a strict win or a 3× loss, depending entirely on a benchmark property nobody controls for. Until Section 8's paired run fixes the exchange rate, $A(r)$ curves cannot settle the question.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*