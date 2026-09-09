---
id: 17-reasoning/filler-tokens-versus-semantic-reasoning
title: "Do Latent Reasoning Tokens Carry More Than Extra Compute?"
topic: 17-reasoning
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Do Latent Reasoning Tokens Carry More Than Extra Compute?

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/filler-tokens-versus-semantic-reasoning` · **Status:** partially-solved

## 1. Problem Statement

A language model given extra tokens before answering — pause tokens, dots, learned "thought" vectors, or extra recurrent passes — often gets more answers right. Two mechanisms can explain this:

- **Compute.** The extra positions add attention slots and serial forward passes. The content is irrelevant; only the count matters.
- **Content.** The hidden states at those positions encode task-specific intermediate results, and later positions read them. The content matters.

**Decision predicate.** For a task family $\mathcal{T}$, model $M$, and budget of $k$ extra positions: does accuracy with *semantically loaded* latent tokens exceed accuracy with *content-free filler* tokens at matched FLOPs, matched serial depth, and matched training protocol?

Three variants, of very different difficulty:

- **Measurement.** Given a trained model, decide whether its latent tokens carry decodable, causally-used task content. Currently the weakest link.
- **Method.** Build a training procedure whose latent tokens provably beat filler at matched budget. Partially achieved on synthetic tasks.
- **Theory.** Characterise the complexity classes reachable with $k$ content-free vs. content-bearing intermediate positions. Largely settled for filler; open for continuous latents.

## 2. Formal Setting

Let $M_\theta$ be a decoder transformer with $L$ layers, width $d$, and context $n$. Prompt $x$, answer $y$. Insert $k$ extra positions with embeddings $z_{1:k} \in \mathbb{R}^{k \times d}$ before decoding $y$:

$$\hat{y} = M_\theta(x \Vert z_{1:k}), \qquad A_M(k) = \mathbb{E}_{(x,y)\sim\mathcal{T}}\big[\mathbb{1}[\hat y = y]\big].$$

**Arms, as actually instantiated.**

- *Filler:* $z_i = e(\texttt{"."})$, a fixed embedding independent of $x$. Content-free by construction.
- *Pause:* $z_i = e(\langle\text{pause}\rangle)$, a single learned embedding, still independent of $x$.
- *Continuous latent:* $z_i = h^{(L)}_{i-1}$, the last hidden state fed back as the next input embedding (Coconut-style). Input-dependent.
- *Recurrent depth:* no extra positions; the same block is applied $r$ times, so serial depth is $Lr$ at token count $k=0$.

**Matched budgets.** Two budgets must be reported separately, because a single "compute" number conflates them:

$$C_{\text{flop}} \approx 2N(|x|+k+|y|), \qquad D_{\text{serial}} = L\cdot(\text{number of sequential forward passes}).$$

Filler and latent arms match on $C_{\text{flop}}$ at equal $k$; recurrent-depth arms match $D_{\text{serial}}$ but not $C_{\text{flop}}$.

**Content measure.** Let $s(x)$ be the gold intermediate state (e.g. the partial sum in a multi-step arithmetic chain). Define probe decodability
$$\rho = \max_{g \in \mathcal{G}} \Pr\big[g(h^{(\ell)}_{1:k}) = s(x)\big] - \Pr[\text{majority baseline}],$$
with $\mathcal{G}$ a fixed-capacity probe class (linear, or one hidden layer). Define causal use by resampling: replace $h^{(\ell)}_{1:k}$ with the states from a different input $x'$ sharing the same answer-irrelevant surface form, and measure $\Delta_{\text{causal}} = A(k) - A_{\text{patched}}(k)$.

**Solved** means: exhibiting $\mathcal{T}$ and $M$ with $A_{\text{latent}}(k) - A_{\text{filler}}(k) > 0$ at matched $C_{\text{flop}}$, $D_{\text{serial}}$ and identical training recipe, *and* $\rho > 0$, *and* $\Delta_{\text{causal}}$ accounting for the gap.

**Assumptions known to be violated.** (i) That $s(x)$ is unique — most tasks admit many valid intermediate decompositions, so a low $\rho$ is not evidence of no content. (ii) That the probe class is neutral — a strong probe can recover content the model does not use. (iii) That the filler arm is truly content-free — with $k$ variable and chosen by the model, token *count* leaks information about the input. (iv) That training is matched — every published latent-token gain also changes the training objective.

## 3. State of the Art

**Established (ablated, and reproduced at least once).**

- Filler tokens *can* help, but only under narrow conditions. Pfau, Merrill & Bowman (COLM 2024) show filler ("...") tokens let small transformers solve a synthetic $3\mathrm{SUM}$ variant that they fail without them — and that learning to use filler requires dense, parallelisable supervision. On natural-language tasks the same paper finds no filler gain.
- Filler tokens do not buy expressivity beyond $\mathsf{TC}^0$. Merrill & Sabharwal show padded/log-precision transformers stay inside constant-depth threshold circuits; serial chain-of-thought with $t$ steps does not.
- Filler is a *negative control* that mostly fails. Lanham et al. (2023) replaced CoT with equal-length filler across several tasks; accuracy fell back to the no-CoT baseline.

**Claimed but under-ablated.**

- Pause tokens (Goyal et al., ICLR 2024): gains appear only when pause tokens are present in *both* pretraining and finetuning. Finetune-only insertion gives little to no gain. No matched-FLOP comparison against a content-bearing latent at the same $k$.
- Coconut (Hao et al., 2024): continuous thoughts beat no-CoT on GSM8K and beat explicit CoT on ProsQA, a synthetic graph-reachability set. The reported ProsQA advantage is a benchmark number; the mechanism claim (latent breadth-first search) rests on illustrative decodings, not a systematic $\rho$/$\Delta_{\text{causal}}$ study.
- Recurrent-depth latent reasoning (Geiping et al., 2025, Huginn-3.5B): accuracy rises with recurrence count at fixed parameters. This changes $D_{\text{serial}}$, so it is not evidence about token *content*.

## 4. What Is Known

- **Filler helps only on parallelisable synthetics.** Pfau et al.: small transformers (tens of millions of parameters, from-scratch training) go from near-chance to near-perfect on $3\mathrm{SUM}$ with filler tokens, and gain nothing on natural-language benchmarks at the same scale.
- **Filler fails as a CoT substitute at chat scale.** Lanham et al., models up to ~175B-class: substituting filler for CoT recovers essentially none of the CoT gain across their task suite.
- **Pause-token gains are training-protocol-bound.** Goyal et al., 1B decoder: pretrain+finetune with pauses gives roughly +18 points EM on SQuAD, ~+8 on CommonsenseQA, ~+1 on GSM8K; finetune-only pauses give near-zero or negative deltas.
- **Continuous latents can exceed explicit CoT on synthetic search.** Hao et al., GPT-2-scale: Coconut ≈97% on ProsQA vs ≈77% for explicit CoT; on GSM8K Coconut lands *below* explicit CoT (roughly 34% vs 43%) but well above no-CoT (~17%).
- **Serial depth is the real lever in theory.** Li et al. (ICLR 2024) and Merrill & Sabharwal (ICLR 2024): $t$ CoT steps raise a constant-depth transformer's power to circuits of depth $\sim t$; polynomially many steps reach $\mathsf{P}$. No analogous separation exists for content-free padding.
- **Verbal reasoning traces are often unfaithful.** Turpin et al. (NeurIPS 2023) and Arcuschin et al. (2025) show stated reasoning frequently fails to reflect the causal computation — so "the tokens are semantic" cannot be read off the text.

## 5. What Is Not Known

- **Empirically open.** No published experiment holds $k$, FLOPs, serial depth *and* training recipe fixed while varying only whether $z_i$ depends on $x$. Every reported latent-token gain co-varies with the training objective. Runnable today at 1–8B scale for well under $10^4$ GPU-hours.
- **Empirically open.** Whether any latent-token advantage survives at frontier scale, or is absorbed by a stronger model's single forward pass. Requires a pretraining run, not a finetune.
- **Methodologically blocked.** $\rho$ and $\Delta_{\text{causal}}$ have no agreed instantiation: no standard probe class, no standard $s(x)$, no standard patching control. Two labs can report opposite conclusions on the same checkpoint without either being wrong.
- **Theoretically open.** The expressive power of *continuous* recurrent latents under finite precision. The $\mathsf{TC}^0$ padding results assume discrete tokens; whether feeding back a real-valued $h^{(L)}$ at $p$-bit precision buys anything beyond $\lceil p/\log n\rceil$ discrete tokens is unproven either way.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**.

- Every method that makes latent tokens content-bearing also changes training. So the observed gain $A_{\text{latent}} - A_{\text{filler}}$ decomposes into a content term and a curriculum term, and no published design separates them.
- Content is not identifiable from the checkpoint. Because $s(x)$ is non-unique, a null probe result is consistent with "no content" and with "content in a basis the probe cannot reach". Raising probe capacity trades a false negative for a false positive; there is no capacity setting that is principled rather than conventional.
- The filler control is leaky. If the model chooses $k$, the count itself is a channel. Fixing $k$ removes the leak but also removes the adaptive-compute behaviour that motivates the method.

## 7. Current Research (as of 2026)

- **Latent-space reasoning architectures.** Meta FAIR (Coconut and successors), and recurrent-depth work from Geiping and collaborators (Huginn line). Direction: replace discrete traces with continuous ones and scale recurrence. *(frontier — verify: post-2025 follow-ups and their matched-compute controls.)*
- **Distilling explicit CoT into forward-pass computation.** Deng, Choi, Shieber and follow-ons: stepwise internalisation, removing CoT tokens gradually during finetuning. Directly relevant because it produces content-bearing hidden states with no extra tokens at all.
- **Faithfulness and chain-of-thought monitorability.** Anthropic and Redwood-adjacent groups: whether reasoning traces are causally load-bearing. Supplies the $\Delta_{\text{causal}}$ machinery this problem needs.
- **Expressivity theory of padding and depth.** Merrill, Sabharwal and coauthors (AI2/NYU); Li, Liu, Ma and coauthors. Direction: exact classes for padding, log-depth, and looped transformers.

## 8. Concrete Next Experiment

**Scale.** One 1.4B-parameter decoder, trained from scratch on ~100B tokens, plus a matched 7B replicate if the 1.4B result is positive. About 2–4k A100-hours per arm.

**Design.** Four arms, *identical* data, steps, optimiser, and $k=8$ inserted positions before every answer span:

| Arm | $z_i$ | $C_{\text{flop}}$ | $D_{\text{serial}}$ |
|---|---|---|---|
| A. Filler | fixed `.` embedding | matched | matched |
| B. Pause | one learned embedding | matched | matched |
| C. Latent | $h^{(L)}_{i-1}$ fed back | matched | matched |
| D. Latent-shuffled | $h^{(L)}$ from a different input in the batch | matched | matched |

Arm D is the control that isolates content: it has identical statistics and identical compute, and differs only in whether the state belongs to *this* input.

**Deciding number.** $\delta = A_C - A_D$ on a held-out multi-step arithmetic and graph-reachability suite, at fixed $k=8$.

- $\delta \le 1$ point with a 95% CI width under 1.5 points $\Rightarrow$ latent tokens are compute, not content, at this scale.
- $\delta \ge 5$ points, together with $\rho > 0.2$ from a linear probe on $s(x)$ $\Rightarrow$ content is real and load-bearing.

Report $A_A$ and $A_B$ alongside to locate the pure-compute floor.

## 9. Key References

- **[Foundational]** Jacob Pfau, William Merrill, Samuel R. Bowman. *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models.* COLM, 2024. — arXiv:2404.15758
- **[Foundational]** Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, Vaishnavh Nagarajan. *Think Before You Speak: Training Language Models With Pause Tokens.* ICLR, 2024. — arXiv:2310.02226
- **[Foundational]** Tamera Lanham et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic technical report, 2023. — arXiv:2307.13702
- **[Theory]** William Merrill, Ashish Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024. — arXiv:2310.07923
- **[Theory]** Zhiyuan Li, Hong Liu, Denny Zhou, Tengyu Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR, 2024. — arXiv:2402.12875
- **[Theory]** William Merrill, Ashish Sabharwal. *Exact Expressive Power of Transformers with Padding.* 2025. (identifier omitted — verify before citing a preprint number)
- **[SOTA]** Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian. *Training Large Language Models to Reason in a Continuous Latent Space.* 2024. — arXiv:2412.06769
- **[SOTA]** Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* 2025. — arXiv:2502.05171
- **[Related]** Yuntian Deng, Yejin Choi, Stuart Shieber. *From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step.* 2024. — arXiv:2405.14838
- **[Related]** Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS, 2023. — arXiv:2305.04388
- **[Survey]** Iván Arcuschin et al. *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful.* 2025.

## 10. Worked Example

Task: 6-step integer chain, $a_0 = 7$, then $a_{t+1} = (3a_t + c_t) \bmod 97$ with $c_{1:6}$ given in the prompt. Answer is $a_6$. Chance accuracy is $1/97 \approx 1.0\%$.

A GPT-2-scale model, no extra tokens, scores about 12% — it can do two steps in a forward pass and guesses the rest. Now add $k=6$ positions.

- Filler (arm A): 13%. Within noise of baseline. The task is *serial*: step $t+1$ needs $a_t$. Padding adds width, not depth, so it buys nothing — exactly what the $\mathsf{TC}^0$ padding result predicts.
- Latent feedback (arm C): 71%.

The naive reading is "content, +58 points". Now run arm D, the shuffled-latent control, and suppose it scores 34%. The decomposition is:

$$\underbrace{13 \to 34}_{\text{training recipe + recurrence}} \quad\underbrace{34 \to 71}_{\delta = 37,\ \text{input-specific content}}$$

Only the second term is content. Without arm D, 21 of the 58 points would have been misattributed.

Now make the obstruction visible. Fit a linear probe on the $k$ latent states for $a_3$: it reads $a_3$ at 8% — barely above chance. Two readings are consistent with $\delta = 37$ and $\rho \approx 0.07$:

1. The model represents $a_3$ nonlinearly, and a linear probe cannot see it.
2. The model never computes $a_3$; it computes $a_6$ by a different factorisation (say, composing the affine maps $x \mapsto 3x + c_t$ and applying the product once), so $a_3$ is genuinely absent.

Both are correct mechanisms. Both produce the same $\delta$ and the same $\rho$. Choosing between them requires committing to a specific $s(x)$ — and the whole point is that $s(x)$ is not unique. **That is the block:** the causal test ($\delta$) is clean and runnable; the content test ($\rho$) is not identifiable without a ground-truth decomposition nobody can supply.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*