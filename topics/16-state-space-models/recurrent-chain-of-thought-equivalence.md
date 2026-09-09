---
id: 16-state-space-models/recurrent-chain-of-thought-equivalence
title: "Recurrent Models and Chain-of-Thought Compute Equivalence"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Recurrent Models and Chain-of-Thought Compute Equivalence

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/recurrent-chain-of-thought-equivalence` · **Status:** open

## 1. Problem Statement

Chain of thought (CoT) — letting a model emit intermediate tokens before its answer — provably enlarges what a Transformer can compute. The open question is whether it does the same for a fixed-state recurrent model.

- **Input.** An architecture family $\mathcal{A}$ (Transformer, SSM/linear RNN, gated linear attention), a per-step precision $p$, a recurrent state size $s(n)$ bits, and a CoT budget $t(n)$ steps on inputs of length $n$.
- **Output.** The language class $\mathsf{CoT}_{\mathcal{A}}[t,s]$ decidable by $\mathcal{A}$ under those budgets.
- **Decision predicate.** Is $\mathsf{CoT}_{\mathrm{SSM}}[t,s] = \mathsf{CoT}_{\mathrm{TF}}[t]$ for $s = O(\mathrm{polylog}\,n)$ and $t$ polynomial — or is there a separation?

Three variants, different difficulty:

- **Theory.** Prove or refute a separation between fixed-state-plus-CoT and growing-state-plus-CoT. The natural conjecture: CoT buys a Transformer *space* (the emitted tape is randomly re-readable by attention) but buys an SSM only *time* (the emitted tape is re-read through a bottleneck of $s$ bits).
- **Method.** Find an architectural patch — hybrid attention layers, state expansion, negative eigenvalues, latent recurrence — that closes any separation at fixed parameter and FLOP budget.
- **Measurement.** Define a compute-matched comparison at all. A Mamba decode step and a Transformer decode step at context $k$ have different FLOP and memory profiles; "same number of CoT tokens" is not "same compute."

Solving it means: a theorem separating or collapsing the classes, plus a matched-compute empirical curve showing whether the separation binds at trained scale.

## 2. Formal Setting

A causal sequence model $M_\theta$ maps a prefix $x_{1:k} \in \Sigma^k$ to a next-token distribution. Under CoT, $M$ generates $y_1,\dots,y_t$ autoregressively and the answer is read from $y_t$. Write $\mathsf{CoT}_{\mathcal{A}}[t(n)]$ for the languages so decided with $t(n)$ generated tokens.

**Recurrent state.** An SSM layer $\ell$ carries $h^{(\ell)}_k \in \mathbb{R}^{d\times N}$ with
$$h_k = A_k \odot h_{k-1} + B_k x_k, \qquad y_k = C_k^\top h_k,$$
$A_k$ diagonal (Mamba/S6) or diagonal-plus-rank-1 (DeltaNet). Measured state size is
$$s = L \cdot d \cdot N \cdot p \ \text{bits},$$
with $p$ the numeric precision *actually used at inference* (bf16 $\Rightarrow p=16$), $L$ layers. For Mamba-2 at $d{=}4096$, $N{=}128$, $L{=}64$: $s \approx 5.4 \times 10^8$ bits. A Transformer's KV cache is $s_{\mathrm{TF}}(k) = 2Ld_{\mathrm{kv}}pk$ — it *grows*. That is the whole distinction, and it is measurable in bytes with `torch.cuda.max_memory_allocated`.

**Compute per emitted token.** $F_{\mathrm{SSM}} \approx 2P$ (parameters $P$, constant in $k$); $F_{\mathrm{TF}}(k) \approx 2P + 4Ld_{\mathrm{kv}}k$. A compute-matched CoT comparison must equalize $\sum_{k} F(k)$, not $t$.

**Circuit setting.** For fixed $t$, one forward pass of a log-precision Transformer or an SSM is uniform $\mathrm{TC}^0$ — constant-depth threshold circuits. CoT composes $t$ such circuits sequentially, giving depth $O(t)$.

**Assumptions, and which are violated.**
- *Log precision* ($p = O(\log n)$). Violated: bf16 is 8 mantissa bits regardless of $n$, so real models are closer to *constant* precision, which is strictly weaker.
- *Uniformity* of the circuit family. Violated: trained weights are not $\mathrm{L}$-uniform in any checked sense; uniformity is an artifact of the proof, not the artifact.
- *CoT tokens are discrete and losslessly re-read.* Violated for latent/continuous CoT (Coconut, recurrent-depth), where the "token" is a real vector.
- *Learnability.* All separation theorems are about expressivity. Nothing guarantees gradient descent finds the construction; this is the single largest gap between Section 4 and practice.

## 3. State of the Art

**Theory (established).**
- Merrill & Sabharwal, *The Expressive Power of Transformers with Chain of Thought* (ICLR 2024): log-precision Transformers with $\Theta(\log n)$ CoT steps characterize $\mathsf{L}$; with polynomially many steps, $\mathsf{P}$. Without CoT they stay in $\mathrm{TC}^0$.
- Li, Liu, Zhou, Ma, *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems* (ICLR 2024): constant-depth Transformers with $T$ CoT steps simulate size-$T$ Boolean circuits; $\mathrm{poly}(n)$ CoT $\Rightarrow$ $\mathsf{P}/\mathrm{poly}$.
- Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024): S4/S6/Mamba layers are in uniform $\mathrm{TC}^0$; they cannot express $S_5$ word problems unless $\mathrm{TC}^0 = \mathrm{NC}^1$. SSMs are *not* more expressive than Transformers, contrary to the RNN intuition.
- Sarrof, Veitsman & Hahn (NeurIPS 2024): SSMs capture exactly the star-free regular languages under standard parameterizations.

**Theory (claimed, not settled).** No published theorem gives a *separation* between $\mathsf{CoT}_{\mathrm{SSM}}$ and $\mathsf{CoT}_{\mathrm{TF}}$ at matched $t$. The folklore argument — fixed state $\Rightarrow$ CoT adds time but not space — is a plausible sketch, not a proof, because the SSM re-reads its own emitted tokens as fresh inputs and no one has shown that the $s$-bit funnel is a genuine bottleneck for a *self-generated* tape.

**Empirical.** Mamba-2 (Dao & Gu, ICML 2024) and Gated Linear Attention (Yang et al., ICML 2024) are the strongest pure-recurrent families. Hybrids are the systems SOTA: Waleffe et al. (2024) trained 8B Mamba, Mamba-2 and a Mamba-2-Hybrid (attention + MLP interleaved) on 3.5T tokens. Nearly every headline recurrent-vs-Transformer reasoning comparison exists **only as a benchmark aggregate**, with no CoT-length ablation and no matched-FLOP control.

## 4. What Is Known

- **Hybrid closes the average gap at 8B/3.5T.** Mamba-2-Hybrid beat the 8B Transformer by **+2.65 points averaged over 12 standard tasks** (Waleffe et al., 2024). Pure Mamba-2 *trailed* on tasks requiring in-context retrieval — notably 5-shot MMLU and phone-book lookup — by margins that grew with context.
- **Copying separates the families at 160M.** Jelassi et al., *Repeat After Me* (ICML 2024): Transformers trained on strings up to length 50 copy strings of several hundred tokens; SSMs of matched size fail to length-generalize. Pretrained Pythia-410M outperformed Mamba-360M on copy and retrieval despite comparable perplexity.
- **The $\mathrm{TC}^0$ ceiling is real and shared.** SSMs cannot do $S_5$ composition in one pass at any length (Merrill et al., ICML 2024). Empirically, Mamba trained on $S_5$ word problems fails to generalize past training length.
- **The ceiling is an artifact of the eigenvalue range, partly.** Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues* (ICLR 2025): extending diagonal eigenvalues from $[0,1]$ to $[-1,1]$ lets Mamba-style models learn parity, and extended DeltaNet learns $S_5$ state tracking — with no loss on 1.3B-scale language-modelling perplexity.
- **Latent recurrence buys measurable reasoning.** Geiping et al. (2025) trained a 3.5B recurrent-depth model on 800B tokens; scaling test-time recurrence to $r{=}32$ (≈132 effective layers) improved GSM8K and ARC monotonically in $r$. This is *depth* recurrence, not *state* recurrence — evidence that serial compute helps, silent on whether fixed state suffices.

## 5. What Is Not Known

- **Theoretically open.** Whether $\mathsf{CoT}_{\mathrm{SSM}}[\mathrm{poly}, \mathrm{polylog}] \subsetneq \mathsf{CoT}_{\mathrm{TF}}[\mathrm{poly}]$. No proof either way. A separation would likely require an unconditional communication-complexity lower bound on the $s$-bit state across the CoT tape; a collapse would require an SSM construction that uses emitted tokens as an external tape with only $O(1)$ re-reads per cell.
- **Empirically open.** Whether a *matched-FLOP* CoT budget erases the retrieval/copy gap. Nobody has run: fix total decode FLOPs, give the SSM proportionally more CoT tokens (it is cheaper per token), and measure. This is runnable today at 1.3B on 8 GPUs.
- **Methodologically blocked.** "Compute equivalence" has no accepted operationalization. Candidates — equal tokens, equal FLOPs, equal wall-clock, equal peak memory — rank the architectures differently and no paper states which it uses. Until one is fixed, cross-paper claims are not comparable.

## 6. Why It Is Hard

The binding obstruction is **confounded measurement compounded by non-identifiability**, not compute cost.

- **The confound.** Every published recurrent-vs-Transformer reasoning comparison varies at least three things at once: architecture, tokens-per-parameter, and CoT length. The +2.65-point hybrid gain is an average over 12 heterogeneous tasks; it cannot be attributed to state capacity, to attention's random access, or to optimization differences.
- **Non-identifiability.** An SSM failing a long-CoT task is consistent with two irreconcilable causes: the $s$-bit state is information-theoretically insufficient (expressivity), or SGD did not find the routing (learnability). No published protocol distinguishes them. Weight-editing the known $S_5$ construction into a trained model and measuring the loss delta would, and has not been done.
- **The evaluation does not measure what it names.** GSM8K and MMLU are called "reasoning" but are dominated by retrieval of memorized facts and short arithmetic — exactly the regime where a $5\times10^8$-bit state is not the constraint. Tasks where state capacity *is* the constraint (copying, phone-book, $S_5$) are the ones where separation appears, and they are dismissed as synthetic.

## 7. Current Research (as of 2026)

- **Hybrid ratio search.** Roughly 1-in-6 to 1-in-8 attention layers is the working recipe (NVIDIA, AI21 Jamba, Falcon-Mamba lineages). Why that ratio, and whether it is a function of CoT length, is unexplained *(frontier — verify)*.
- **Eigenvalue and state-expansion fixes.** Follow-on work to Grazzi et al. on DeltaNet-family products of Householder matrices for richer state tracking; groups at ETH/IDSIA, MIT, Tübingen.
- **Latent CoT.** Coconut (Hao et al., 2024) and recurrent-depth (Geiping et al., 2025) replace discrete tokens with continuous state, which changes the formal question: a real-vector "token" has unbounded nominal capacity, so the $\mathrm{TC}^0$ arguments do not transfer.
- **Log-depth expressivity.** Merrill & Sabharwal's log-depth Transformer results (2025) suggest depth and CoT are partly interchangeable; the SSM analogue is unwritten *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question decided:** does CoT compensate for fixed state at matched decode FLOPs?

- **Scale.** Four models at **1.3B parameters, 100B tokens**, identical data order and tokenizer: (a) Transformer, (b) Mamba-2, (c) Mamba-2 with eigenvalues in $[-1,1]$, (d) Mamba-2-Hybrid at 1-in-6 attention. ≈8×H100 for ~5 days per arm.
- **Task suite.** Three tasks with *known* state-complexity: $S_5$ word problems (length 4–128), string copy (length 50 train / 50–500 test), and multi-hop key–value lookup over 512 pairs. Each admits a verbatim CoT trace, so supervision is exact and grading is not an LLM judge.
- **Control arm.** The Transformer, run at **equal total decode FLOPs**, not equal tokens. Since $F_{\mathrm{SSM}}$ is constant in $k$ and $F_{\mathrm{TF}}$ grows, at context 4096 the SSM gets roughly $1.3$–$2\times$ more CoT tokens for the same FLOPs. Compute the ratio per task from the measured profile, do not assume it.
- **Deciding number.** $\Delta = \mathrm{Acc}_{\mathrm{SSM}}(\text{FLOP-matched CoT}) - \mathrm{Acc}_{\mathrm{TF}}$ on $S_5$ at length 128, with 5 seeds. **If $\Delta > -2$ points, fixed state plus CoT is compute-equivalent in practice** and the hybrid premium is an optimization artifact. **If $\Delta < -10$ points and the gap widens monotonically with problem length while CoT length is scaled to compensate, the separation is real and binding.** Anything between is the honest current prior and should be reported as such.
- **Second reading, free.** Log the per-step state entropy proxy $H \approx \|h_k\|_0$-weighted rank of the state matrix. If accuracy collapses exactly where estimated state rank saturates, the cause is capacity, not optimization — which resolves the non-identifiability of Section 6.

## 9. Key References

- **[Foundational]** Merrill, W., Sabharwal, A. *The Expressive Power of Transformers with Chain of Thought.* ICLR 2024. — arXiv:2310.07923
- **[Foundational]** Merrill, W., Petty, J., Sabharwal, A. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[Foundational]** Li, Z., Liu, H., Zhou, D., Ma, T. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR 2024. — arXiv:2402.12875
- **[Foundational]** Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., Wang, L. *Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective.* NeurIPS 2023. — arXiv:2305.15408
- **[SOTA]** Dao, T., Gu, A. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** Grazzi, R., Siems, J., Franke, J. K. H., Zela, A., Hutter, F., Pontil, M. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025. — arXiv:2411.12537
- **[SOTA]** Waleffe, R., et al. *An Empirical Study of Mamba-based Language Models.* NVIDIA, 2024. — arXiv:2406.07887
- **[SOTA]** Jelassi, S., Brandfonbrener, D., Kakade, S., Malach, E. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[SOTA]** Sarrof, Y., Veitsman, Y., Hahn, M. *The Expressive Capacity of State Space Models: A Formal Language Perspective.* NeurIPS 2024.
- **[SOTA]** Geiping, J., et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* 2025. — arXiv:2502.05171
- **[Survey]** Deletang, G., et al. *Neural Networks and the Chomsky Hierarchy.* ICLR 2023. — arXiv:2207.02098
- **[Survey]** Gu, A., Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024. — arXiv:2312.00752

## 10. Worked Example

**Task.** Copy a 400-token random string after a 4096-token distractor prefix. Models: Mamba-2 1.3B ($d{=}2048$, $N{=}128$, $L{=}48$) vs Transformer 1.3B ($L{=}24$, $d_{\mathrm{kv}}{=}2048$).

**State budget.** Mamba-2 state: $48 \times 2048 \times 128 \times 16 \approx 2.0\times10^8$ bits $= 25$ MB, constant. Transformer KV at $k{=}4496$: $2 \times 24 \times 2048 \times 16 \times 4496 \approx 7.1\times10^9$ bits $= 883$ MB. Ratio **35:1** in the Transformer's favour, growing linearly in $k$.

**Information bound.** 400 tokens over a 50k vocabulary carry $400 \times \log_2 50000 \approx 6240$ bits. That fits in 25 MB by five orders of magnitude — so the naive capacity argument *does not* explain the failure. The SSM fails anyway (Jelassi et al.), because the selective-scan update cannot implement content-addressed lookup: the failure is about *addressing*, not *capacity*.

**Now add CoT.** Decode FLOPs per token: SSM $\approx 2.6$ GFLOP, constant. Transformer at $k{=}4496$: $2.6 + 4{\cdot}24{\cdot}2048{\cdot}4496 \approx 3.5$ GFLOP. So a FLOP-matched budget gives the SSM $1.35\times$ more CoT tokens — it can emit a 540-token scratchpad for the Transformer's 400. That extra 140 tokens is worth about $2.2$ kbit of scratch tape.

**Where the obstruction becomes visible.** Those 540 emitted tokens are re-read by the SSM only through the same 25 MB state, one pass, no random access. The Transformer re-reads its 400 emitted tokens by attention — $O(1)$ depth to any position. So the SSM's CoT is $1.35\times$ more *serial steps* and $0\times$ more *addressable memory*. Extra time does not fix an addressing failure. Any experiment that scores "SSM + longer CoT vs Transformer" on GSM8K, where scratchpads are short and lookups are local, will report parity and conclude equivalence — while the mechanism it claims to test is untouched. That is precisely the confound in Section 6, and why the deciding number in Section 8 is $S_5$ accuracy at length 128, not a benchmark average.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*