---
id: 16-state-space-models/recurrent-formal-language-hierarchy
title: "Recurrent Models on Formal Language Hierarchies"
topic: 16-state-space-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Recurrent Models on Formal Language Hierarchies

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/recurrent-formal-language-hierarchy` · **Status:** partially-solved

## 1. Problem Statement

Given a recurrent sequence model — LSTM, linear RNN, or a modern state-space model (SSM) such as S4, Mamba, DeltaNet, RWKV — determine **which formal languages it can recognize, and which it can learn from finite samples and generalize to unseen lengths**.

Three variants, routinely conflated:

- **Theory variant.** Fix an architecture family $\mathcal{A}$, a precision model, and a depth/width budget. Characterize $\mathcal{L}(\mathcal{A})$, the class of languages expressible by some parameter setting. Solved is: an exact characterization in terms of a standard class (star-free, regular, $\mathsf{TC}^0$, $\mathsf{NC}^1$).
- **Method variant.** Given a target language $L$ and a training distribution over strings of length $\le n_{\text{train}}$, does gradient descent find parameters that recognize $L$ at lengths $\gg n_{\text{train}}$? Expressible $\ne$ learnable.
- **Measurement variant.** Define a length-generalization metric that separates "learned the automaton" from "fit a length-bounded shortcut". No accepted definition exists.

The theory variant is largely solved for the main families. The method and measurement variants are open.

## 2. Formal Setting

Alphabet $\Sigma$, string $x = x_1 \cdots x_T \in \Sigma^*$. A recurrent model is
$$h_t = f_\theta(h_{t-1}, x_t) \in \mathbb{R}^d, \qquad \hat{y}_t = g_\theta(h_t),$$
with $h_0$ fixed. For linear/SSM families the update is affine in the state:
$$h_t = A_t(x_{\le t})\, h_{t-1} + B_t(x_{\le t}),$$
where $A_t \in \mathbb{R}^{d\times d}$. The **structure of $A_t$ is the whole story**: diagonal with entries in $[0,1]$ (S4, Mamba, GLA), diagonal in $[-1,1]$ (Grazzi et al.), or a rank-one-corrected identity $A_t = I - \beta_t k_t k_t^\top$ (DeltaNet, a Householder reflection when $\beta_t = 2$, $\|k_t\|=1$).

**Measured quantities.**

- *Precision.* $p$ bits per state entry. Measurement: the actual dtype at inference (bf16 $\Rightarrow$ 8 mantissa bits), not the idealized $\mathbb{R}$. Expressivity theorems that assume $p = \Theta(\log T)$ are testable only by re-running at fixed dtype and watching where accuracy breaks.
- *Length generalization score.* Train on $|x| \le n_{\text{tr}}$, test on $n_{\text{tr}} < |x| \le n_{\text{te}}$. Per-token accuracy, scaled against the majority-class baseline $\alpha_0$:
$$\mathrm{LG} = \frac{1}{n_{\text{te}}-n_{\text{tr}}}\sum_{n=n_{\text{tr}}+1}^{n_{\text{te}}} \frac{\mathrm{acc}(n) - \alpha_0}{1-\alpha_0}.$$
Delétang et al. call a task *solved* at $\mathrm{LG} \ge 0.9$.
- *State-tracking difficulty.* The word problem for a finite group $G$: input $g_1\cdots g_T \in G^T$, output the prefix products. $G = S_5$ is the canonical hard case — it is non-solvable, and its word problem is $\mathsf{NC}^1$-complete.

**Assumptions known to be violated in practice.** (i) Unbounded or $\log T$ precision — real runs are bf16/fp16 fixed. (ii) Saturated/hard-threshold activations, used in most RNN automata proofs — real nets are smooth and operate far from saturation. (iii) Uniform sampling over $\Sigma^{\le n}$ — real curricula are length-stratified and heavily skewed. (iv) Infinite training time; separations proved by expressivity say nothing about the optimization path.

## 3. State of the Art

**Theory SOTA (established).**

- Siegelmann & Sontag (1995): RNNs with rational weights and unbounded precision/time are Turing-complete. Vacuous for finite-precision practice.
- Merrill (2019), Merrill et al. (ACL 2020): saturated LSTMs are counter machines; saturated GRUs/simple RNNs are finite-state. The *rational recurrence* hierarchy is a proper hierarchy.
- Merrill, Petty & Sabharwal (ICML 2024), *The Illusion of State in State-Space Models*: S4 and Mamba, at log precision and constant depth, are in $\mathsf{L}$-uniform $\mathsf{TC}^0$ — the same class as transformers. They therefore cannot solve the $S_5$ word problem unless $\mathsf{TC}^0 = \mathsf{NC}^1$.
- Sarrof, Veitsman & Hahn (NeurIPS 2024): diagonal SSMs with non-negative real eigenvalues express exactly the **star-free** regular languages in the regular fragment. Parity ($(\Sigma\Sigma)^*$-style modular counting) is not star-free and is out of reach at any width.

**Empirical SOTA (established by reproduction).**

- Delétang et al. (ICLR 2023) benchmark: RNN/LSTM solve regular and counter tasks; Stack-RNN solves deterministic context-free tasks; Tape-RNN reaches some context-sensitive tasks; transformers fail parity and most non-star-free tasks.
- Grazzi et al. (ICLR 2025): extending the eigenvalue range of Mamba and DeltaNet from $[0,1]$ to $[-1,1]$ makes parity and modular arithmetic solvable, with no loss on language-modeling perplexity at ~370M parameters.

**Claimed but unablated.** (i) That the eigenvalue fix helps *downstream* reasoning, not just synthetic state tracking — reported as scaling-run perplexity and a few benchmark numbers, with no controlled ablation isolating state tracking. (ii) DeltaProduct-style Householder products (Siems et al., 2025) reaching $S_5$: the expressivity claim has a proof, the learnability claim is a benchmark number on a synthetic suite. (iii) RWKV-7's claimed recognition beyond $\mathsf{TC}^0$ — a construction, not a trained-model measurement.

## 4. What Is Known

- **Parity separation is sharp and reproduced.** At $d$ up to 512, non-negative-eigenvalue Mamba stays at chance (50%) on parity beyond training length; sign-extended Mamba reaches >95% at test lengths $8\times$ training. Measured at 100–400M parameters (Grazzi et al., 2025).
- **Transformers fail where LSTMs succeed.** Delétang et al., 2023: over ~20 tasks with $n_{\text{tr}}=40$, $n_{\text{te}}=256$, transformers score $\mathrm{LG} \approx 0$ on parity, cycle navigation and modular arithmetic; LSTMs score $\ge 0.9$ on several. Model scale: ~$10^5$–$10^6$ parameters, deliberately small.
- **LSTMs count; GRUs do not.** Weiss, Goldberg & Yahav (ACL 2018): LSTMs trained on $a^nb^n$ generalize to $n$ far beyond training because the cell state implements an unbounded counter; GRUs, whose state is convex-combined into $[-1,1]$, do not. Reproduced many times at $d \le 100$.
- **Depth buys bounded state tracking.** Liu et al. (ICLR 2023): a constant-depth transformer simulates any solvable-group automaton on length $T$ with depth $O(\log T)$ — a *shortcut*, exact only up to the trained length. This is the mechanism behind "solves it in-distribution, collapses out-of-distribution".
- **The $\mathsf{TC}^0$ ceiling is architecture-agnostic.** Merrill & Sabharwal (TACL 2023) for log-precision transformers, Merrill et al. (ICML 2024) for SSMs: same class, same barrier.

## 5. What Is Not Known

- **Theoretically open.** Whether $\mathsf{TC}^0 \ne \mathsf{NC}^1$. Every "SSMs cannot track state" theorem is conditional on it. Also open: the exact expressivity class of DeltaNet-family models at *fixed* finite precision — proofs use exact-arithmetic Householder reflections, whose products drift in bf16.
- **Empirically open.** Does the expressivity gain from negative eigenvalues or Householder products convert into measurable gains on natural language at $\ge$ 3B parameters and $\ge$ 300B tokens? Runnable today; unrun at that scale with a matched control.
- **Empirically open.** Which of the Delétang separations survive at $10^8$–$10^9$ parameters? The benchmark was run at $10^5$–$10^6$; nobody has repeated it at modern scale with modern SSMs and matched token budgets.
- **Methodologically blocked.** There is no accepted test distinguishing "the model implements the automaton" from "the model implements a depth-$O(\log n_{\text{tr}})$ shortcut that happens to agree on all tested lengths". $\mathrm{LG}$ measured to $n_{\text{te}}=256$ cannot separate them; a shortcut correct to length $2^{16}$ is indistinguishable from the real thing under any feasible test.

## 6. Why It Is Hard

**The primary obstruction is that the evaluation does not measure what it names.** "Length generalization" is operationalized as accuracy on a finite length window. But Liu et al.'s shortcut construction produces models that are exactly correct up to a length determined by depth and precision, then fail. Any finite test window is therefore consistent with both hypotheses, and pushing the window out costs $O(n_{\text{te}})$ inference per example with no bound on how far is far enough.

Two secondary obstructions:

- **Non-identifiability of the mechanism.** Two parameter settings with identical behavior on all tested strings can implement different automata. Probing the state $h_t$ for the automaton's state is circular: a linear probe on a $d=1024$ state recovers a 5-state variable at high accuracy even from models that fail the task, because the information is present but not used by the readout.
- **Confounded architecture comparison.** Mamba-vs-transformer results vary in tokenizer, position encoding, optimizer and token budget. The published parity separations hold under matched settings; almost nothing else does.

## 7. Current Research (as of 2026)

- **Eigenvalue and transition-matrix design.** Grazzi, Siems, Franceschi, Hutter and collaborators (Freiburg / linear-RNN community): negative eigenvalues, then DeltaProduct — $n_h$ Householder reflections per token, trading throughput for state-tracking depth. Reported to reach $S_5$ at modest $n_h$ *(frontier — verify the trained-model result, not the construction)*.
- **Formal-language expressivity of SSMs.** Hahn's group (Saarland) and Merrill/Sabharwal (NYU, AI2): tightening the star-free characterization, extending to input-dependent gating and to chain-of-thought-augmented recurrence.
- **Hybrid and chunked recurrence.** Whether interleaving attention layers with recurrence recovers non-star-free capability, or merely adds a second $\mathsf{TC}^0$ device *(frontier — verify)*.
- **Precision-aware expressivity.** Restating theorems with $p$ fixed rather than $\Theta(\log T)$; this is where the practical separations actually live.

## 8. Concrete Next Experiment

**Question.** Does the expressivity fix (eigenvalue range $[-1,1]$, or $n_h=2$ Householder products) buy anything on real data, or only on synthetics?

- **Scale.** Two 1.3B-parameter models, identical in every respect except the transition-matrix parameterization, trained on the same 100B tokens of a public corpus with the same seed, tokenizer, schedule and data order.
- **Control arm.** Baseline Mamba-2 / DeltaNet with eigenvalues in $[0,1]$ and $n_h=1$. A second control: the fixed model with the state-tracking capability *disabled at inference* by clamping eigenvalues to $[0,1]$, which isolates whether the capability is used.
- **Evaluation.** A held-out state-tracking probe embedded in natural text — e.g. tracking the parity of an even/odd counter described in prose over 2k–32k tokens — plus standard perplexity.
- **The deciding number.** Accuracy on the 32k-token prose parity probe. If the fixed model exceeds the control by $\ge 15$ points absolute while perplexity differs by $< 0.02$ nats, the expressivity class is doing real work at scale. If the gap is $< 3$ points, the synthetic separation does not transfer and the whole line is a benchmark artifact.

## 9. Key References

- **[Foundational]** H. Siegelmann, E. Sontag. *On the Computational Power of Neural Nets.* Journal of Computer and System Sciences, 1995.
- **[Foundational]** G. Weiss, Y. Goldberg, E. Yahav. *On the Practical Computational Power of Finite Precision RNNs for Language Recognition.* ACL, 2018. — arXiv:1805.04908
- **[Foundational]** W. Merrill. *Sequential Neural Networks as Automata.* Workshop on Deep Learning and Formal Languages, ACL, 2019. — arXiv:1906.01615
- **[Foundational]** W. Merrill, G. Weiss, Y. Goldberg, R. Schwartz, N. A. Smith, E. Yahav. *A Formal Hierarchy of RNN Architectures.* ACL, 2020.
- **[Foundational]** M. Hahn. *Theoretical Limitations of Self-Attention in Neural Sequence Models.* TACL, 2020.
- **[SOTA/Benchmark]** G. Delétang, A. Ruoss, J. Grau-Moya, T. Genewein, L. K. Wenliang, E. Catt, C. Cundy, M. Hutter, S. Legg, J. Veness, P. A. Ortega. *Neural Networks and the Chomsky Hierarchy.* ICLR, 2023. — arXiv:2207.02098
- **[SOTA]** B. Liu, J. T. Ash, S. Goel, A. Krishnamurthy, C. Zhang. *Transformers Learn Shortcuts to Automata.* ICLR, 2023.
- **[SOTA]** W. Merrill, A. Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL, 2023.
- **[SOTA]** W. Merrill, J. Petty, A. Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024.
- **[SOTA]** Y. Sarrof, Y. Veitsman, M. Hahn. *The Expressive Capacity of State Space Models: A Formal Language Perspective.* NeurIPS, 2024.
- **[SOTA]** S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS, 2024.
- **[SOTA]** R. Grazzi, J. Siems, J. K. H. Franceschi, T. Brox, F. Hutter et al. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR, 2025.
- **[Frontier]** J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products.* 2025.
- **[Survey]** S. Bhattamishra, K. Ahuja, N. Goyal. *On the Ability and Limitations of Transformers to Recognize Formal Languages.* EMNLP, 2020.

## 10. Worked Example

**Parity in a one-dimensional diagonal SSM.** Target: $y_T = \bigoplus_{t\le T} x_t$, $x_t \in \{0,1\}$.

The exact solution is a sign flip. Encode $h_0 = 1$ and
$$h_t = a_t h_{t-1}, \qquad a_t = 1 - 2x_t \in \{+1,-1\},$$
so $h_T = (-1)^{\sum_t x_t}$ and parity is a threshold on $h_T$. This needs $d=1$ and one bit.

Now impose the S4/Mamba constraint $a_t \in [0,1]$. Then $h_t$ is non-increasing in magnitude, so $h_T$ is a monotone function of the input multiset and cannot distinguish $x=110$ from $x=100$. Sarrof et al.'s theorem is exactly this, generalized to arbitrary $d$: non-negative diagonal transitions give the star-free languages, and parity is not star-free. **No width fixes it.**

**Where the obstruction becomes visible.** A 130M-parameter Mamba trained on parity with $n_{\text{tr}}=40$ nonetheless reaches ~100% training-length accuracy. It does this by *positional bookkeeping*: with $d=1536$ and 24 layers it allocates enough capacity to memorize a lookup over the 40 prefix positions. Test at length 41–256 and accuracy falls to $\approx 50\%$ — chance, since $\alpha_0 = 0.5$ and $\mathrm{LG} \approx 0$.

The sign-extended model gets $\mathrm{LG} > 0.95$ over the same window. So far, so clean. The problem: a *deeper* baseline with $L$ layers implements the Liu et al. shortcut and is exactly correct to some length $n^\star(L, p)$. At $L=48$ and bf16, $n^\star$ is empirically in the low thousands. Testing to $n_{\text{te}} = 256$ scores it as solved; testing to $n_{\text{te}} = 10^4$ scores it as failed. The benchmark's verdict is a function of the window, not of the model — which is why the theory variant is settled and the measurement variant is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*