---
id: 16-state-space-models/persistent-state-continual-learning
title: "Continual Learning via Persistent Recurrent State"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continual Learning via Persistent Recurrent State

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/persistent-state-continual-learning` · **Status:** open

## 1. Problem Statement

A recurrent or state-space language model carries a fixed-size hidden state $h_t$ that is a causal function of everything it has read. In principle that state is a place to accumulate knowledge across a lifetime of inputs without touching the weights: no gradient step, no replay buffer, no optimizer. The question is whether it actually works.

**The problem.** Given a pretrained recurrent model with state of size $S$ bits and a stream of experience $x_1, x_2, \dots$ that is far longer than any training sequence, does carrying $h_t$ forward — never resetting it — produce durable, retrievable, compositional knowledge? Or does the state decay to a fixed point that encodes only a recency window, making "persistent state" a name for a buffer rather than a memory?

Three variants, different difficulty:

- **Measurement.** Define retention: given a fact injected at token $t_0$ and probed at $t_0 + \Delta$, what is the recovery probability as a function of $\Delta$ and of intervening content? No standard benchmark isolates this from in-context retrieval over a still-visible window.
- **Method.** Build an architecture and an update rule such that state-carried retention decays sub-exponentially in $\Delta$ while throughput stays $O(1)$ per token. Test-time-training layers and delta-rule variants are attempts.
- **Theory.** Prove capacity and interference bounds: how many facts can a state of $S$ bits hold at recall accuracy $\ge 1-\epsilon$ under a given write rule, and is forgetting under a linear-attention write rule necessarily first-in-first-out?

**Solved** would mean: a model whose recall of material seen $10^7$ tokens ago, with state carried and weights frozen, is within a stated margin of a retrieval-augmented control at equal inference FLOPs, with the retention curve reproduced independently.

## 2. Formal Setting

A recurrent layer maps input $x_t \in \mathbb{R}^d$ and state $h_{t-1} \in \mathbb{R}^{n \times d}$ to

$$h_t = A_t \, h_{t-1} + B_t x_t^\top, \qquad y_t = h_t\, C_t,$$

with $A_t$ diagonal or diagonal-plus-low-rank. Mamba-2 and Gated DeltaNet instantiate $A_t = \alpha_t (I - \beta_t k_t k_t^\top)$ (gated delta rule); Mamba uses $A_t = \exp(\Delta_t A)$, elementwise.

**Quantities as measured.**

- **State budget** $S = L \cdot n \cdot d \cdot b$ bits, $L$ layers, $b$ bits per element at inference precision. Measure by the actual size of the serialized state tensor, not the theoretical rank.
- **Retention curve.** Inject a key–value pair $(k^\star, v^\star)$ at position $t_0$ as natural text. At $t_0 + \Delta$, probe with a cloze query and score exact-match. $R(\Delta) = \Pr[\hat{v} = v^\star]$, estimated over $\ge 500$ independent facts with distinct surface forms. Intervening tokens must be drawn from the *same* distribution as the carrier corpus, or $R$ measures distribution shift instead of forgetting.
- **Effective memory horizon** $\Delta_{1/2} = \min\{\Delta : R(\Delta) < \tfrac{1}{2} R(0)\}$.
- **Interference** $I = R_{\text{single}}(\Delta) - R_{m}(\Delta)$, with $m$ facts injected in the same stream. Non-zero $I$ at fixed $\Delta$ is capacity pressure, not decay.
- **Compute parity.** Report FLOPs/token and state bytes. A comparison against a retrieval baseline is meaningless without both.

**Assumptions, and which are violated.**

1. *State is the only carrier of history.* Violated whenever short convolutions, sliding-window attention, or a KV cache coexist with the SSM — as in Jamba, Samba, and most hybrids. Any hybrid's "state persistence" result is confounded by its attention window.
2. *Test-time distribution matches pretraining.* Violated by construction: the point is novel experience.
3. *Length generalization.* Models are trained at $T_{\text{train}} \sim 2$–$32$k tokens and probed at $10^6$+. Position-free recurrence does not guarantee behavioral stability; gates saturate off-distribution.
4. *Recall is the right target.* Continual learning wants skill acquisition and compositional update, not verbatim recall. Retention curves measure the easy proxy.

## 3. State of the Art

**Established.**
- Linear-attention/SSM layers with a delta rule (Schlag et al., ICML 2021; Yang et al., NeurIPS 2024 "Parallelizing Linear Transformers with the Delta Rule") perform an explicit *overwrite* rather than pure addition, which measurably improves associative recall over pure additive linear attention at matched state size.
- Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025) combines a decay gate with the delta rule and beats Mamba-2 and DeltaNet at 1.3B/2.7B scale on in-context recall suites. Established as benchmark improvement; the *retention-versus-distance* decomposition is not reported.
- Mamba-2 (Dao & Gu, ICML 2024) shows state size is the binding constraint: expanding state dimension from 16 to 256 improves recall monotonically with a throughput cost.
- Transformers beat SSMs at copying, with a separation that grows with sequence length (Jelassi et al., ICML 2024).
- SSMs with diagonal transitions are in uniform $\mathrm{TC}^0$ and cannot solve $S_5$ word problems (Merrill, Petty & Sabharwal, ICML 2024) — a hard limit on state *tracking*, distinct from state *storage*.

**Claimed but unablated.** Test-time-training layers (Sun et al., 2024) and Titans (Behrouz, Zhong & Mirrokni, 2024) frame the state as fast weights updated by an inner gradient step, and report favorable long-context numbers. Neither reports a controlled multi-episode continual-learning protocol with a state-reset control arm. Their long-context wins exist as benchmark numbers on needle-style and BABILong-style tasks, which reward retrieval from a still-resident stream, not retention across episode boundaries.

**Absent.** No published result runs a frozen-weight SSM with continuously carried state over $\ge 10^7$ tokens and reports a retention curve.

## 4. What Is Known

- **State size dominates architecture.** Zoology (Arora et al., ICLR 2024) attributes most of the gap between gated-convolution models and attention on associative recall to recurrent-state capacity; at 355M parameters the recall gap closes as state grows, and Based (Arora et al., ICML 2024) trades a linear-attention feature dimension against recall on a measured Pareto frontier.
- **Copying degrades outside the training length.** Jelassi et al. (2024) show SSMs trained to copy at length $n$ fail well before $2n$, while transformers with the right positional scheme extend further. Measured at ~160M scale, synthetic strings.
- **Catastrophic forgetting in weights is well characterized.** EWC (Kirkpatrick et al., PNAS 2017) recovers most of the sequential-task accuracy lost by SGD on permuted MNIST; the parameter-space story does not transfer to state-space memory, and no analogue of the Fisher penalty exists for $h_t$.
- **Recall on long-context suites is not saturated.** RULER (Hsieh et al., COLM 2024) shows most models' effective context is far below their claimed context; recurrent models degrade earliest on multi-hop and aggregation categories.
- **The delta rule bounds interference.** Overwriting $k^\star$'s slot removes the old value exactly in the orthogonal-keys case; with $n$-dimensional state, $n$ orthogonal keys are storable exactly. Real keys are not orthogonal, so realized capacity is empirical and, at $n=128$ per head, is far below $n$ facts.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted protocol separating (a) recall from a state that has been carried across a boundary, from (b) recall from tokens still inside the model's effective window, from (c) knowledge already in the pretrained weights. Every current long-context benchmark conflates all three. Until this is fixed, "persistent state continual learning" cannot be scored.
- **Empirically open.** Whether the retention curve $R(\Delta)$ of a frozen 1–3B recurrent model is exponential or power-law past $10^6$ tokens. The run costs a few thousand GPU-hours; nobody has published it.
- **Theoretically open.** No capacity theorem for the gated delta rule with non-orthogonal keys: the maximum $m$ with $R \ge 1-\epsilon$ given state $S$ under realistic key distributions is unproven either way. Also open: whether persistent state can implement any update that weight-space learning can, or whether $\mathrm{TC}^0$-type limits (Merrill et al., 2024) bound what a fixed state can *become* as opposed to what it can *hold*.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**. When a recurrent model answers a probe about material from $10^6$ tokens ago, three mechanisms are consistent with the observation — carried state, pretraining priors, and surface leakage in the probe — and the hidden state is a single dense tensor with no addressable slots, so no read-out distinguishes them post hoc. Attempts to isolate the first by using synthetic keys create a second problem: out-of-distribution keys break the gate statistics the model was trained with, so the measurement changes the thing measured. Add the compute floor — a $10^7$-token sequential stream cannot be parallelized across the sequence at inference, so wall-clock scales linearly and a single retention curve is a multi-day serial run per arm — and the cheap experiments are all confounded while the clean ones are all expensive.

## 7. Current Research (as of 2026)

- **Test-time memory as fast weights.** Titans, and follow-ups reframing the state as an inner-loop-optimized memory module (Behrouz and colleagues at Google Research), plus TTT-Linear/TTT-MLP (Sun, Li, Dalal et al.). Direction: make the write rule a learned optimizer. *(frontier — verify current benchmark claims.)*
- **Delta-rule family.** Gated DeltaNet and successors (Yang, Kautz, Hatamizadeh; NVIDIA + MIT), and hardware-efficient chunkwise kernels in `flash-linear-attention`.
- **Hybrids that sidestep the question.** Jamba (AI21), Samba (Microsoft), and Nemotron-H mix attention with SSM layers; they win benchmarks but make the persistent-state claim untestable.
- **Continual pretraining with replay** remains the practical alternative and the honest control arm.
- *(frontier — verify)* Work on state compression/eviction for recurrent models, and on writing to state at inference without gradients, is active but unconsolidated.

## 8. Concrete Next Experiment

**Scale.** One 1.3B Gated DeltaNet or Mamba-2, weights frozen, pretrained on ~100B tokens. Build an 8-episode stream, $1.25 \times 10^6$ tokens per episode, $10^7$ tokens total, each episode a distinct domain corpus. Into episode 1 inject 500 natural-language facts with entity names sampled from the corpus's own name distribution (not synthetic tokens). Probe after each subsequent episode: 8 probe points, $\Delta$ from $10^5$ to $8.75 \times 10^6$.

**Arms.**
1. *Persistent:* state carried across all episodes, never reset.
2. *Control — state reset at each episode boundary* (identical compute, identical probes). This is the arm that isolates carried state.
3. *Control — frozen weights, no episode-1 exposure* (measures pretraining leakage; probes must score ≈ chance).
4. *Reference — BM25 retrieval over episode 1 at matched inference FLOPs.*

**Deciding number.** $R_{\text{persist}}(\Delta=8.75\times10^6) - R_{\text{reset}}$, exact-match on 500 facts. If the gap is $\le 2$ points (within noise at $n=500$, s.e. $\approx 2.2$ pts), persistent state carries no durable episodic knowledge at this scale and the research program should move to explicit memory modules. If the gap exceeds 15 points, the retention curve becomes the object of study and the capacity theorem becomes the priority. Cost estimate: ~4 serial inference passes over $10^7$ tokens per arm, on the order of $10^3$ GPU-hours total.

## 9. Key References

- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR, 2022. — arXiv:2111.00396
- **[Foundational]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Foundational]** Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML, 2021. — arXiv:2102.11174
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[SOTA]** Songlin Yang, Jan Kautz, Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR, 2025. — arXiv:2412.06464
- **[SOTA]** Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS, 2024. — arXiv:2406.06484
- **[SOTA]** Yu Sun, Xinhao Li, Karan Dalal, et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* 2024. — arXiv:2407.04620
- **[SOTA]** Ali Behrouz, Peilin Zhong, Vahab Mirrokni. *Titans: Learning to Memorize at Test Time.* 2024. — arXiv:2501.00663
- **[Limits]** Samy Jelassi, David Brandfonbrener, Sham Kakade, Eran Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Limits]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[Measurement]** Simran Arora, Sabri Eyuboglu, Aman Timalsina, et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[Measurement]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Foundational]** James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, et al. *Overcoming catastrophic forgetting in neural networks.* PNAS 114(13), 2017.
- **[Survey]** Liyuan Wang, Xingxing Zhang, Hang Su, Jun Zhu. *A Comprehensive Survey of Continual Learning: Theory, Method and Application.* IEEE TPAMI, 2024. — arXiv:2302.00487

## 10. Worked Example

Take a 1.3B Mamba-2-style model: $L = 48$ layers, 32 heads, head dimension $d_h = 64$, state expansion $n = 128$, bf16.

$$S = 48 \times 32 \times 128 \times 64 \times 2\ \text{bytes} = 25.2\ \text{MB} \approx 2.0 \times 10^8\ \text{bits}.$$

Now the capacity accounting. A single injected fact — "the reactor at Kessel-4 was recommissioned in 2031" — needs the entity, the relation, and the value: call it 40 bits of entropy against a realistic prior. Naively $2.0 \times 10^8 / 40 = 5 \times 10^6$ facts. That number is wrong by orders of magnitude, and seeing why is the point.

The state is not a codebook. Each head writes $h \leftarrow \alpha(I - \beta k k^\top)h + \beta v k^\top$ into a $128 \times 64$ matrix. Exact retrieval requires near-orthogonal keys; the number of $\epsilon$-orthogonal directions usable at recall accuracy $1-\epsilon$ is closer to $n = 128$ per head than to the bit count, and real key vectors from text are strongly anisotropic, so the realized figure is smaller again. Take an optimistic 30 usable slots per head, 32 heads: $\sim 10^3$ retrievable associations per layer-group, several orders below $5\times 10^6$.

Then the decay term. With per-step gate $\alpha = 0.999$ — a plausible value for a token-level forget gate — the contribution of a write survives as $\alpha^{\Delta}$. At $\Delta = 10^4$ tokens, $0.999^{10^4} \approx 4.5 \times 10^{-5}$: the injected fact is $10^{-4}$ of its original magnitude, buried under $10^4$ subsequent writes. Reaching $\Delta = 10^6$ with half the signal intact needs $\alpha \ge 1 - 7\times10^{-7}$, i.e. a gate indistinguishable from 1 in bf16, which means the state never forgets anything and saturates.

**The obstruction, made visible.** The bit budget says $10^6$ facts; the orthogonality constraint says $10^3$; the gate arithmetic says the horizon is $10^4$ tokens unless the gate is numerically 1, at which point capacity collapses instead. These three estimates disagree by four orders of magnitude, and no measurement in the literature adjudicates between them, because every published probe is run inside a window where the tokens are still present. Section 8's reset-control arm exists precisely to break that tie.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*