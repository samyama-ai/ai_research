---
id: 17-reasoning/continuous-versus-discrete-chain-of-thought
title: "Continuous-Vector Chains of Thought Versus Discrete Tokens"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continuous-Vector Chains of Thought Versus Discrete Tokens

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/continuous-versus-discrete-chain-of-thought` · **Status:** empirically-open

## 1. Problem Statement

A discrete chain of thought (CoT) feeds the model's own sampled tokens back as input. A continuous chain of thought replaces that loop: the last hidden state $h_t \in \mathbb{R}^d$ is fed back directly as the next input embedding, skipping the unembedding–sample–embed round trip. The question is whether this buys anything.

- **Measurement variant.** At a *matched serial compute budget* (equal number of forward passes through the network), does continuous feedback beat discrete feedback in task accuracy? This is the version practitioners care about and the version that is empirically open.
- **Method variant.** Is there a training recipe that reliably produces continuous reasoning states without the multi-stage curricula, teacher traces, or distillation targets that current methods require? All working recipes today need a discrete teacher.
- **Theory variant.** Is there a separation — a language or problem family solvable with $O(f(n))$ continuous steps but requiring $\omega(f(n))$ discrete steps for a constant-depth transformer, or a proof that no such separation exists under bounded numeric precision?

Solving it means: a preregistered, compute-matched comparison whose sign is stable across at least two model scales and three task families, plus a theory statement that says what precision the separation needs.

## 2. Formal Setting

Let $M_\theta$ be a decoder transformer with hidden width $d$, vocabulary $V$, embedding matrix $E \in \mathbb{R}^{|V| \times d}$, and unembedding $W_U$. Discrete CoT is the recursion

$$ z_t = M_\theta(e_1,\dots,e_{t-1}), \quad y_t \sim \mathrm{softmax}(W_U z_t / \tau), \quad e_t = E_{y_t}. $$

Continuous CoT replaces the last two steps with $e_t = P(z_t)$ for a projection $P$ (identity in Coconut; a simplex-weighted mixture $e_t = \sum_v p_t(v) E_v$ in "soft thinking" variants).

**Quantities, as measured.**

- **Serial compute** $S$ = number of forward passes to produce the answer. Measured by counting decode steps, *not* by counting emitted characters. Latent steps and token steps each cost one pass; this is the only budget under which the comparison is fair.
- **FLOPs** $F \approx 2NS + $ attention terms, $N$ = non-embedding parameters. Report both $S$ and $F$; they diverge when the latent arm uses a smaller KV cache.
- **Channel capacity of one step.** Discrete: $\log_2|V|$ bits, e.g. $17.0$ bits at $|V|=131{,}072$. Continuous: nominally $16d$ bits at bf16, but the *usable* rate is $I(\text{state};\text{next-step-relevant variable})$ under the noise floor, estimated by a decoder probe $q_\phi$: $\hat{I} = H(Y) - \frac{1}{n}\sum_i -\log q_\phi(y_i \mid h_i)$, in nats, with $Y$ the ground-truth intermediate quantity.
- **Superposition width** $k_t$ = number of distinct reasoning branches simultaneously decodable from $h_t$, measured as the number of gold successor states with probe probability above a preregistered threshold.

**Assumptions and their violations.** (i) *Matched serial compute is the right control.* Violated in most published comparisons, which match token counts or wall-clock instead. (ii) *The latent state is trained end-to-end.* Violated: every strong result uses a discrete teacher trace, so the latent arm inherits the discrete arm's coverage. (iii) *Numeric precision is unbounded.* Violated: bf16 has 8 mantissa bits, so superposition arguments that pack $\Theta(n)$ items into one vector degrade as $n$ grows. (iv) *Greedy decode is representative.* Violated: continuous CoT has no native sampling operator, so self-consistency and pass@$k$ comparisons are not like-for-like.

## 3. State of the Art

**Empirical.**
- **COCONUT** (Hao, Sukhbaatar, Su, Li, Wang, Weston, Tian, 2024/2025). Feeds the last hidden state back as the next embedding, trained by a multi-stage curriculum that progressively replaces textual steps with latent ones. *Established:* on ProsQA (GPT-2 scale) it reaches $97.0\%$ vs $77.5\%$ for CoT with far fewer decoded steps. *Established but negative:* on GSM8K it reaches $34.1\%$ vs $42.9\%$ for CoT (no-CoT baseline $16.5\%$). The headline "latent reasoning works" therefore holds on graph-search tasks and fails on arithmetic word problems at the same scale.
- **Soft Thinking** (Zhang et al., 2025), training-free concept-token mixtures over the embedding table; reported gains of a few pass@1 points with ~20% fewer tokens on QwQ-32B / math benchmarks. *Claimed but unablated:* no matched-serial-compute control, and the gain is confounded with the effective temperature change from mixing.
- **Recurrent-depth latent reasoning** (Geiping et al., 2025, Huginn-0125, 3.5B params, 800B tokens): scales test-time compute by iterating a recurrent block instead of emitting tokens. *Established:* accuracy rises monotonically with recurrence count on reasoning benchmarks. *Not established:* that it beats spending the same FLOPs on discrete CoT.
- **Compression/distillation lines** — CCoT (Cheng & Van Durme, 2024), CODI (Shen et al., 2025), Token Assorted (Su et al., 2025, latent+text hybrids), implicit CoT by stepwise internalization (Deng et al., 2024). All report benchmark numbers; none reports a compute-matched control arm.

**Theory.** Zhu et al. (2025), *Reasoning by Superposition*, give the sharpest positive statement: a 2-layer transformer with continuous thoughts solves directed-graph reachability in $D$ steps ($D$ = graph diameter) by holding a superposition of frontier nodes in one vector, where discrete CoT provably needs to serialize the search. For discrete CoT the reference bounds are Merrill & Sabharwal (ICLR 2024): polynomially many CoT steps give exactly $\mathsf{P}$; logarithmically many stay inside $\mathsf{L}$.

## 4. What Is Known

- Continuous CoT wins on search-shaped tasks at small scale. ProsQA, GPT-2 (124M): $97.0\%$ vs $77.5\%$. ProntoQA: $99.8\%$ vs $98.8\%$ — a ceiling effect, not evidence.
- Continuous CoT loses on GSM8K at GPT-2 scale: $34.1\%$ vs $42.9\%$.
- The superposition mechanism is real and observed, not merely hypothesized: probes on COCONUT-style models recover multiple frontier nodes from a single latent vector, matching the Zhu et al. construction.
- Training instability is reproducible. Removing the curriculum (jumping straight to all-latent) collapses accuracy in the COCONUT ablations; the recipe, not the architecture, carries much of the result.
- Depth substitutes for tokens. Saunshi et al. (2025) show a $k$-layer transformer looped $L$ times matches a $kL$-layer model on reasoning benchmarks while retaining $k$-layer parameter count — latent iteration is not free capacity, it is depth reuse.
- Filler/pause tokens (Goyal et al., ICLR 2024; Pfau et al., COLM 2024) show that *content-free* extra forward passes already help on some tasks. Any latent-CoT gain smaller than the filler-token gain is not evidence for continuous representation.

## 5. What Is Not Known

- **Empirically open.** Whether continuous CoT beats discrete CoT at matched serial compute at $\geq 7$B parameters with RL post-training. Every published head-to-head is at $\leq 3.5$B, or lacks the control arm, or both. The experiment is runnable today for well under $10^{22}$ FLOPs.
- **Theoretically open.** No separation theorem under bounded precision. Zhu et al. assume enough numeric resolution to superpose $\Theta(n)$ states; whether a $p$-bit-mantissa transformer retains any asymptotic advantage is unproven either way. Also open: whether continuous CoT exceeds $\mathsf{P}$-with-poly-steps, i.e. whether it changes the Merrill–Sabharwal picture at all or only saves a polynomial factor.
- **Methodologically blocked.** There is no accepted sampling operator for continuous chains, so pass@$k$, self-consistency, and RL with verifiable rewards — the methods that produce current frontier reasoning results — have no defined analogue. Until that is fixed, the comparison cannot be run in the regime that matters.
- **Methodologically blocked.** Faithfulness and monitorability of latent traces are undefined: there is no ground truth for "what the vector meant," so the safety cost of removing the readable trace (Korbak et al., 2025) cannot be priced.

## 6. Why It Is Hard

The core obstruction is **confounded measurement compounded by non-identifiability of the budget**. Three distinct resources move together when you switch to latent feedback: serial steps, FLOPs, and the entropy injected by sampling. Matching tokens un-matches compute; matching compute un-matches the exploration distribution, because greedy latent decoding removes the sampling noise that discrete CoT relies on. A reported win can therefore be produced by any of: extra depth (filler-token effect), reduced sampling noise, or genuine superposition — and no published protocol separates the three.

Secondary: **absent ground truth** for the latent state. The probe-based capacity estimate $\hat I$ is a lower bound whose tightness depends on probe class, so "the latent carries more information" is not falsifiable as stated. And the training path is a **credit-assignment** problem: with no discrete bottleneck, gradients flow through the whole chain, which is why every working method reintroduces a discrete teacher and thereby caps the latent arm at teacher quality.

## 7. Current Research (as of 2026)

- **Meta (FAIR)** — COCONUT line, continuing on curriculum-free training and hybrid latent/text traces.
- **Maryland / ELLIS (Geiping and collaborators)** — recurrent-depth scaling, whether latent iteration transfers to RL-trained reasoners *(frontier — verify)*.
- **Michigan / Oymak group** — CoT2 and continuous-token analyses of parallel exploration; supervised and policy-gradient training of continuous thoughts.
- **Zhu, Wang and co-authors** — superposition theory, extending reachability separations to precision-bounded models *(frontier — verify)*.
- **Interpretability/safety groups (UK AISI, Anthropic, Redwood co-signatories of the CoT-monitorability position paper)** — arguing the legibility cost of latent traces is a reason to prefer discrete chains even at equal accuracy.

## 8. Concrete Next Experiment

**Scale.** Two model sizes, 1.5B and 8B (e.g. Qwen2.5 base checkpoints). One shared SFT corpus of 300K verified step-by-step traces spanning GSM8K/MATH-style arithmetic, ProsQA-style graph search, and a code-execution trace set. Budget ≈ 4–6 GPU-weeks on 8×H100 per arm.

**Arms (four, identical data and optimizer):**
1. Discrete CoT, sampled at $T=0.7$ — **primary control**.
2. Discrete CoT with the chain truncated so serial steps equal the latent arm, plus filler tokens to pad — **the filler control that most papers omit**.
3. COCONUT-style continuous feedback, $S$ latent steps then a short discrete answer span.
4. Hybrid: latent steps with a discrete token emitted every 4th step.

**Protocol.** Sweep $S \in \{4, 8, 16, 32, 64\}$ and plot accuracy against *forward passes*, not tokens. Report greedy pass@1 for all arms (the only metric defined for arm 3).

**Deciding number.** Greedy pass@1 accuracy on held-out MATH-500 and ProsQA-hard at $S = 32$ forward passes, arm 3 minus arm 2. Preregister an equivalence margin of $\pm 1.5$ percentage points, $n \geq 2{,}000$ items per set (binomial SE ≈ $1.1$ pp, so the margin is detectable). If arm 3 − arm 2 $> +1.5$ pp on both task families at both scales, continuous CoT has a real matched-compute advantage. If it lands inside $\pm 1.5$ pp, the published gains are the filler-token effect and the field should stop attributing them to continuous representation.

## 9. Key References

- **[Foundational]** Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian. *Training Large Language Models to Reason in a Continuous Latent Space.* COLM, 2025 (arXiv 2024). — arXiv:2412.06769
- **[Theory]** William Merrill, Ashish Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024. — arXiv:2310.07923
- **[Theory/SOTA]** Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, Yuandong Tian. *Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought.* 2025. — arXiv:2505.12514
- **[SOTA]** Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* 2025. — arXiv:2502.05171
- **[Control]** Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, Vaishnavh Nagarajan. *Think before you speak: Training Language Models with Pause Tokens.* ICLR, 2024. — arXiv:2310.02226
- **[Control]** Jacob Pfau, William Merrill, Samuel R. Bowman. *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models.* COLM, 2024. — arXiv:2404.15758
- **[Related]** Yuntian Deng, Yejin Choi, Stuart Shieber. *From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step.* 2024. — arXiv:2405.14838
- **[Related]** Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, Sashank J. Reddi. *Reasoning with Latent Thoughts: On the Power of Looped Transformers.* ICLR, 2025. — arXiv:2502.17416
- **[Position]** Tomek Korbak et al. *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.* 2025. — arXiv:2507.11473

## 10. Worked Example

Take $|V| = 131{,}072$ and $d = 4096$, bf16.

- Discrete step capacity: $\log_2 |V| = 17.0$ bits.
- Continuous step nominal capacity: $16 \times 4096 = 65{,}536$ bits — a $3{,}855\times$ ratio. This is the number the field's intuition runs on.

Now measure instead of count. Train a linear probe on a ProsQA-style reachability task with $n = 30$ nodes to read the frontier set out of $h_t$. The maximum recoverable information is $H(\text{frontier}) \le 30$ bits (one bit per node). Observed superposition width in reported COCONUT-style probes is a handful of nodes, so the *usable* rate is on the order of $\hat I \approx 5$–$15$ bits per step — **below or comparable to a single discrete token's 17 bits**, and three to four orders of magnitude below the nominal 65,536.

The gap is not a probe artifact you can define away. Two independent effects cause it: the state must simultaneously carry the residual stream's other jobs (position, syntax, answer format), and bf16's 8-bit mantissa means superposed components separated by less than $\sim 2^{-8}$ relative magnitude are unrecoverable, capping the number of cleanly separable directions far below $d$.

So the arithmetic that motivates continuous CoT ("3,855× more bandwidth") and the arithmetic that predicts its measured behaviour ("roughly one token's worth of usable bits, spent on parallel frontier tracking instead of one serial choice") give opposite answers. The obstruction is visible here: the advantage, where it exists, is **structural** — holding a superposition of branches — not **bandwidth**. That is why continuous CoT wins on breadth-first graph search ($97.0\%$ vs $77.5\%$) and loses on GSM8K ($34.1\%$ vs $42.9\%$), where the computation is a serial chain with no branching frontier to superpose. Any experiment that averages over both task types will report a null and learn nothing.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*