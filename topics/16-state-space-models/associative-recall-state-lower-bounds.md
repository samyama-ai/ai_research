---
id: 16-state-space-models/associative-recall-state-lower-bounds
title: "Associative Recall Lower Bounds for Constant-State Models"
topic: 16-state-space-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Associative Recall Lower Bounds for Constant-State Models

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/associative-recall-state-lower-bounds` · **Status:** partially-solved

## 1. Problem Statement

A constant-state sequence model (linear RNN, SSM, linear attention, gated convolution) compresses an arbitrarily long prefix into a fixed-size state. Attention does not: its KV cache grows with $N$. The question is how much of the recall gap this forces.

Three variants, routinely conflated:

- **Theory.** For the multi-query associative recall task (MQAR) on sequences of length $N$, is there a lower bound on recurrent state size $M$ (in bits) for any model that solves it with error $\le \epsilon$? Answer: yes, $M = \Omega(N)$ in the worst case, one pass, exact. Established.
- **Measurement.** What is the *effective* recall capacity of a deployed architecture — how many key-value pairs it actually retrieves per bit of state — and does it track the information-theoretic floor $K\log_2|V|$ or the worst-case bound $\Omega(N)$? Open; no agreed protocol.
- **Method.** Can an architecture with $M \ll N$ close the recall gap on real text by exploiting non-worst-case key distributions, multiple passes, or a small sparse-attention branch? Empirically open; hybrid models are the current answer, without a matching theory.

Solving the problem means a bound that *predicts the empirical failure point* of a given architecture on a given recall distribution, not merely one that is asymptotically true.

## 2. Formal Setting

**Task.** MQAR (Zoology, Arora et al., ICLR 2024). Vocabulary $V$. Input $u \in V^N$: a prefix of $K$ key-value bigrams $(k_1,v_1),\dots,(k_K,v_K)$ with distinct keys, interleaved with distractor tokens, followed by $Q$ query tokens $q_j \in \{k_i\}$. Target at query position $j$ is $v_{i(j)}$ with $k_{i(j)} = q_j$. Measured as exact-match accuracy over query positions only:
$$\mathrm{Acc} = \frac{1}{Q}\sum_{j=1}^{Q}\mathbf{1}\!\left[\hat y_j = v_{i(j)}\right].$$

**Model.** A recurrent model with per-layer state $h^{(\ell)}_t \in \mathbb{R}^{d_\ell}$ updated causally, $h^{(\ell)}_t = f_\theta(h^{(\ell)}_{t-1}, x_t)$, read out by $g_\theta$. Measured state size, at numeric precision $p$ bits:
$$M = p\sum_{\ell=1}^{L} d_\ell .$$
For Mamba $d_\ell = d_{\text{model}}\cdot d_{\text{state}}\cdot e$ (expand factor $e$); for gated linear attention $d_\ell = d_k d_v$ per head, summed over heads; for a convolution of kernel width $w$ the state includes the $w$-token buffer. $p$ is the *inference* dtype (bf16: $p=16$), not the training dtype.

**Information floor.** Storing $K$ pairs over $|V|$ symbols needs $M \ge K\log_2|V| - O(1)$ bits.

**Worst-case bound.** Reduce from one-way communication complexity of INDEX: Alice holds the $K$ pairs, Bob the query; the recurrent state at the boundary is the only message. $R^{\to}(\mathrm{INDEX}_n) = \Omega(n)$ gives $M = \Omega(K\log|V|)$, and with $K = \Theta(N)$, $M = \Omega(N)$.

**Assumptions, and which break.**
1. *Single pass, causal.* Violated by prefix-scan / "read twice" schemes (JRT), which re-expose the prefix and legitimately beat the one-pass bound.
2. *Uniform, adversarial keys.* Violated hard in natural text: recall targets are Zipfian and often locally repeated, so the worst case is not the operating case.
3. *State is the only channel.* Violated by hybrids with even one attention layer, and by short convolutions whose buffer holds recent tokens verbatim.
4. *Exact recall.* Violated by language modeling, where partial credit in logit space is what perplexity measures.
5. *Fixed precision $p$.* Bounds assuming unbounded-precision reals give no memory constraint at all; $p$ must be pinned to make $M$ meaningful.

## 3. State of the Art

**Theory (established).**
- Arora et al., *Zoology* (ICLR 2024): data-independent gated convolutions require model dimension scaling near-linearly in sequence length to solve MQAR; attention needs $O(\log N)$ width. Proof by communication complexity.
- Jelassi et al., *Repeat After Me: Transformers are Better than State Space Models at Copying* (ICML 2024): copying a length-$n$ string requires $\Omega(n)$ state bits for any recurrent model; a two-layer transformer with hash-based induction heads copies strings exponentially longer than its parameter count.
- Wen, Dang, Lyu, *RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval* (2024): the separation is retrieval-specific, and chain-of-thought closes it for RNNs on some tasks.

**Theory (adjacent, sometimes misapplied).** Merrill, Petty, Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024) places SSMs in $\mathrm{TC}^0$ — a *depth/parallelism* limit on state tracking, not a memory bound on recall. Grazzi et al. (ICLR 2025) show negative eigenvalues in linear RNNs restore some state tracking; this does not affect the recall bound.

**Empirical SOTA.** The frontier is the recall-vs-state Pareto curve, not a single model: Based (Arora et al., ICML 2024), Mamba-2/SSD (Dao & Gu, ICML 2024), DeltaNet (Yang et al., NeurIPS 2024), Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025), RWKV-7 (Peng et al., 2025). Delta-rule updates — which *overwrite* a key's slot rather than accumulate — are the main architectural gain.

**Claimed but unablated.** That delta-rule capacity gains come from better key-slot allocation rather than from optimization effects is asserted more than isolated. Long-context recall claims resting on single benchmark numbers (needle-in-a-haystack pass rates, RULER scores) are benchmark numbers only: they do not control $M$, so they cannot separate architecture from state budget.

## 4. What Is Known

- **The gap is real and quantified at LM scale.** Zoology attributes roughly 82% of the average perplexity gap between attention and gated-convolution models to tokens requiring associative recall ("AR hits"), at 355M parameters on the Pile.
- **The synthetic phase transition is sharp.** On MQAR at $N \le 512$, $|V| = 8192$, attention solves the task at $d_{\text{model}} = 64$; gated-convolution and SSM baselines need $d_{\text{model}}$ growing with $N$ and with the number of pairs, reaching $512$ in the reported sweeps.
- **Copying separates cleanly.** In Jelassi et al., transformers trained on strings up to a fixed length copy substantially longer strings; Mamba-family models degrade at or before their training length. Confirmed both from-scratch (~100M-scale) and with pretrained Pythia vs. Mamba checkpoints of comparable size.
- **Extra passes buy recall.** "Just read twice" (Arora et al., 2024) improves recurrent-model recall at fixed state by re-reading the prefix — evidence the binding constraint is one-pass compression, not raw capacity.
- **Hybrids dominate the curve.** A small fraction of full-attention layers recovers most of the recall gap at a fraction of the cache; this is now the standard production answer (Jamba, Zamba, Samba-style designs).

## 5. What Is Not Known

- **Theoretically open.** A distribution-dependent bound. All existing lower bounds are worst-case over key sets; there is no bound of the form $M \ge \Phi(\mathcal{D}, \epsilon)$ for a realistic key distribution $\mathcal{D}$ with tolerated error $\epsilon$. No approximate-recall bound: allowing $\epsilon = 0.1$ may or may not reduce $\Omega(N)$ to $\Omega(N/\log N)$ or to $O(1)$ under Zipf.
- **Theoretically open.** Bounds for delta-rule / DPLR recurrences that are tight rather than inherited from the generic $\Omega(N)$ argument. Empirically these models beat pure linear attention at equal $M$; no theory explains the constant.
- **Empirically open.** Whether the empirical capacity exponent $\alpha$ in $M^\*(N) \propto N^\alpha$ equals 1. Nobody has run the sweep that varies $M$ *directly* (holding parameter count and depth fixed) across two decades of $N$ for four architecture families.
- **Methodologically blocked.** "Effective state utilization" has no accepted definition. Reported $M$ mixes conv buffers, per-head states and precision inconsistently across papers, so cross-paper Pareto curves are not comparable. Until $M$ is measured the same way, the empirical bound is unmeasurable.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability between capacity and optimization**. Failure on MQAR at state $M$ has two indistinguishable causes: (a) $M$ is below the information the task requires, or (b) $M$ suffices but gradient descent does not find the allocation. The theory only speaks to (a), and (a) is almost never binding — see §10, where a failing model has ~39× the bits its task needs. Every negative empirical result is therefore consistent with the lower bound being vacuous at the tested scale.

Second obstruction: **the evaluation does not measure what it names.** Needle-in-a-haystack and RULER vary $N$ while leaving $M$ fixed and unreported, so they measure a model, not a tradeoff. MQAR does control $M$, but its uniform-random keys are the adversarial case the bound already covers, so it cannot test whether the bound is loose on natural distributions.

## 7. Current Research (as of 2026)

- **Delta-rule and DPLR recurrences** — Yang, Kim, Kim (MIT/Flash Linear Attention), Peng (RWKV), NVIDIA: state-overwrite update rules as capacity multipliers at fixed $M$.
- **Hybrid layer-budget scaling laws** — AI21, Zyphra, Microsoft, Together/Hazy Research: how few attention layers suffice, as a function of $N$ and recall density.
- **Mechanistic accounts of recall in recurrent LMs** — work isolating "gather-and-aggregate" style circuits and showing the recall gap concentrates in a small number of heads *(frontier — verify)*.
- **Approximate / distributional lower bounds** — sketching and streaming theory applied to LM recall; the natural tool is the heavy-hitters lower bound, largely unexploited here *(frontier — verify)*.
- **Test-time-training and long-memory modules** — treating the state as a fast-weight learner, which changes the effective $M$ per token and complicates any static bound.

## 8. Concrete Next Experiment

**Question.** Is the empirical capacity exponent $\alpha$ in $M^\*(N)\propto N^\alpha$ equal to 1 (theory is tight) or below 1 (theory is loose in practice)?

**Scale.** MQAR with $N \in \{64,128,256,512,1024,2048\}$, $|V| = 8192$, $K = N/8$ pairs. Four families at matched depth $L=2$ and matched parameter count: Mamba-2, Gated DeltaNet, Based, and a gated convolution. For each $(N, \text{family})$, sweep $M$ by varying *state width only* — $d_{\text{state}}$ or $d_k d_v$ — over 8 points spanning $2^{10}$ to $2^{17}$ bits at bf16, keeping $d_{\text{model}}$ fixed. 5 seeds. Define $M^\*(N)$ = smallest $M$ reaching $\ge 99\%$ query accuracy in $\ge 4/5$ seeds. Cost: ~1000 runs of $\le 20$ GPU-minutes each, ~300 A100-hours.

**Control arm.** Two controls. (i) Softmax attention at the same depth, which should show $M^\*$ flat in $N$ up to the KV cache. (ii) A **key-distribution arm**: the identical sweep with Zipf($s{=}1.1$) keys instead of uniform. The uniform arm tests the worst-case bound; the Zipf arm tests whether it is loose where it matters.

**The deciding number.** The fitted slope $\alpha$ of $\log M^\*$ on $\log N$, with a bootstrap 95% CI, for uniform vs. Zipf keys. If $\alpha_{\text{unif}} \approx 1$ and $\alpha_{\text{zipf}} \le 0.5$ with non-overlapping CIs, the worst-case bound is confirmed *and* shown irrelevant to text — which redirects the field to distributional bounds. If $\alpha_{\text{zipf}} \approx \alpha_{\text{unif}} \approx 1$, constant-state recall is genuinely capped and hybrids are the only route.

## 9. Key References

- **[Foundational]** S. Arora, S. Eyuboglu, A. Timalsina, I. Johnson, M. Poli, J. Zou, A. Rudra, C. Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[Foundational]** S. Jelassi, D. Brandfonbrener, S. Kakade, E. Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Foundational]** K. Wen, X. Dang, K. Lyu. *RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval.* 2024. — arXiv:2402.18510
- **[SOTA]** S. Arora, S. Eyuboglu, M. Zhang, A. Timalsina, S. Alberti, D. Zinsley, J. Zou, A. Rudra, C. Ré. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML, 2024. — arXiv:2402.18668
- **[SOTA]** S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS, 2024. — arXiv:2406.06484
- **[SOTA]** S. Yang, J. Kautz, A. Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR, 2025. — arXiv:2412.06464
- **[SOTA]** T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[Context]** W. Merrill, J. Petty, A. Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[Context]** R. Grazzi, J. Siems, J. K. H. Franke, A. Zela, F. Hutter, M. Pontil. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR, 2025. — arXiv:2411.12537
- **[Survey]** E. Kushilevitz, N. Nisan. *Communication Complexity.* Cambridge University Press, 1997. (Source of the one-way INDEX bound used by every lower bound above.)

## 10. Worked Example

Take one cell of the sweep: $N = 512$, $|V| = 8192$, $K = 64$ key-value pairs, Mamba-style block, $d_{\text{model}} = 64$, $d_{\text{state}} = 16$, expand $e = 2$, $L = 2$, bf16.

**What the task needs.** Sixty-four values over an 8192-symbol vocabulary, keys given:
$$K\log_2|V| = 64 \times 13 = 832 \text{ bits}.$$

**What the model has.**
$$M = p \cdot L \cdot d_{\text{model}} d_{\text{state}} e = 16 \times 2 \times (64 \times 16 \times 2) = 65{,}536 \text{ bits}.$$

**Ratio: 79× the information floor.** Even charging bf16 an effective 8 usable bits instead of 16, it is ~39×.

**What happens.** In the Zoology sweeps, a model at this width does not reach high MQAR accuracy at $N=512$; attention at the same $d_{\text{model}} = 64$ does. The failure is not an information-theoretic one — the state is two orders of magnitude larger than the content it must hold.

**Where the bound lands.** The worst-case theorem says $M = \Omega(K\log|V|) = \Omega(832)$ bits. The model has 65,536. The bound is satisfied by a factor of 79 and still predicts nothing about the observed failure.

**The obstruction, made visible.** The proven lower bound is not the binding constraint at any scale a practitioner runs. What binds is the model's inability to *allocate* its state as 64 addressable slots under gradient descent — a fact about the update rule's expressivity and its optimization, for which no lower bound exists. That is exactly the gap between "partially solved" and solved: the worst-case theory is done, and it is 79× too weak to be the explanation.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*