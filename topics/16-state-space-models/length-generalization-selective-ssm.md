---
id: 16-state-space-models/length-generalization-selective-ssm
title: "Length Generalization of Selective SSMs"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Length Generalization of Selective SSMs

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/length-generalization-selective-ssm` · **Status:** open

## 1. Problem Statement

A selective state-space model (SSM) — Mamba, Mamba-2, and their descendants — has a fixed-size recurrent state and no positional encoding. Both properties suggest it should extrapolate to sequences longer than it was trained on for free. It does not. Trained at context $L$, a Mamba model's per-token loss degrades sharply somewhere past $L$, and retrieval-style accuracy collapses well before the state's nominal capacity is exhausted.

Three distinct variants, usually conflated:

- **Measurement.** What is the right observable for "length generalization" in a constant-memory recurrent model? Per-position NLL, task accuracy, and hidden-state stability give different answers about the same checkpoint. There is no agreed metric that separates *the state ran out of room* from *the state left the distribution it was trained on*.
- **Method.** Find a training recipe or architectural change under which a model trained on $L$ tokens matches, at $\kappa L$ tokens, a model trained natively at $\kappa L$ — at equal parameter count and equal training FLOPs.
- **Theory.** Characterize which functions a selective SSM can realize *uniformly in sequence length*, i.e. with a single parameter vector whose error does not grow with $T$. This is the length-uniform analogue of the expressivity results, and it is open even for depth-2 models.

A solution to the method variant would be: a recipe with $\kappa \ge 8$, a decayed-loss gap $\le 0.02$ nats/token against the native-context control, and no regression on short-context evaluation.

## 2. Formal Setting

A single selective SSM layer maps $x_{1:T}$, $x_t \in \mathbb{R}^{d}$, to $y_{1:T}$ through a per-channel diagonal recurrence:

$$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t x_t, \qquad y_t = C_t^\top h_t + D x_t,$$

with $h_t \in \mathbb{R}^{N}$ per channel, $N \in \{16, 64, 128, 256\}$ in practice. Selectivity means $B_t, C_t, \Delta_t$ are functions of $x_t$:

$$\Delta_t = \mathrm{softplus}(w_\Delta^\top x_t + b_\Delta) \in \mathbb{R}_{>0}, \qquad \bar{A}_t = \exp(\Delta_t A), \quad A = \mathrm{diag}(a_1,\dots,a_N),\ a_n < 0 .$$

**Quantities as measured.**

- *Length generalization gap.* Train on sequences of length $\le L$; evaluate on $L' = \kappa L$. With $\ell_t$ the NLL at position $t$,
 $$G(\kappa) = \frac{1}{L' - L}\sum_{t=L+1}^{L'} \ell_t \;-\; \frac{1}{L'-L}\sum_{t=L+1}^{L'} \ell_t^{\text{native}},$$
 where the native control is the *same architecture and token budget* trained at $L'$. Measuring against $\ell_t$ inside $[1,L]$ instead is the common error: per-position loss falls with $t$ anyway, so an uncontrolled curve conflates extrapolation failure with the ordinary position effect.
- *Effective decay horizon.* $\tau_t^{(n)} = \left(\Delta_t |a_n|\right)^{-1}$, in tokens. The layer's aggregate horizon is measured as the empirical median of $\tau$ over channels and positions on held-out data.
- *Cumulative forgetting.* $S_{s\to t} = \exp\!\big(a_n \sum_{r=s+1}^{t}\Delta_r\big)$. Because $\Delta_r > 0$ and roughly i.i.d. across positions, $\sum_r \Delta_r$ grows $\Theta(t)$, so contributions decay geometrically in *token count* — the model has no mechanism to slow decay as $T$ grows.
- *State drift.* $\rho(t) = \|h_t\|_2 / \mathbb{E}_{t \le L}\|h_t\|_2$. Values $\rho \gg 1$ at $t > L$ are the signature of *state collapse* — the recurrent state leaves the norm regime the readout $C_t$ was fit on.
- *State capacity.* The largest $m$ such that $m$ key–value pairs are recoverable from $h_T$ at $\ge 95\%$ accuracy on synthetic multi-query associative recall. Nominal capacity is $O(N d)$ bits-ish; measured capacity is far smaller.

**Assumptions, and which are violated.** (i) *Test length is the only shift* — violated: long evaluation corpora differ in domain and repetition structure from packed training documents. (ii) *Training documents actually reach length $L$* — violated: packing to $L$ with document-boundary resets means the effective training length is the document-length distribution, typically far below $L$. (iii) *$\Delta_t$ is length-independent* — violated in the interesting cases, since $\Delta_t$ depends on activations that themselves drift with $t$. (iv) *Diagonal $A$ with $a_n<0$*, hence $\bar A_t \in (0,1)$ — this excludes negative eigenvalues and is exactly the restriction that blocks parity-like state tracking.

## 3. State of the Art

**Established.**

- Selective SSMs, in their standard parameterization, are *not* free length-generalizers. Mamba trained at 2k context loses document-retrieval accuracy within a small multiple of the training length; DeciMamba (Ben-Kish et al., ICLR 2025) attributes this to a training-length-bounded *effective receptive field* and recovers extrapolation by pruning tokens between layers.
- *State collapse* is a real and diagnosable failure mode. Chen et al. (2024) show Mamba-2's recurrent state norm grows without bound past the training length; with collapse mitigations, a 370M Mamba-2 trained on short context passes passkey retrieval beyond $10^6$ tokens.
- Hybrids beat pure SSMs on long-context retrieval. Waleffe et al. (NVIDIA, 2024) trained 8B-parameter Mamba, Mamba-2, and Mamba-2-Hybrid models on 3.5T tokens; the pure SSMs lag Transformers on tasks needing copying and in-context retrieval, and the hybrid (a few attention layers interleaved) matches or exceeds the Transformer, including on long-context evaluations.

**Claimed but unablated.** Most "extrapolates to $N\times$" claims are single-task passkey or needle-in-a-haystack numbers, run at one model scale, without the native-context control of §2. Passkey is nearly free for a recurrent model — one salient key, no distractors — and does not certify that language modelling loss extrapolated. Treat these as benchmark numbers, not results.

**Theory SOTA.** Merrill, Petty & Sabharwal (ICML 2024) place fixed-depth SSMs in $\mathrm{TC}^0$-like uniform circuit classes, so they cannot solve $\mathrm{S}_5$ word problems at any length. Sarrof, Veitsman & Hahn (NeurIPS 2024) characterize the formal-language capacity of SSMs, showing they capture star-free languages but not modular counting under the standard positive-$\bar A$ parameterization. Grazzi et al. (ICLR 2025) show that allowing eigenvalues in $[-1,1]$ unlocks parity and richer state tracking. None of these is a *length-uniform learnability* result: they bound what is expressible, not what gradient descent on length-$L$ data finds.

## 4. What Is Known

- **Copying is asymptotically harder.** Jelassi et al. (ICML 2024) prove a fixed-state-size model needs state growing with the string it copies, and show empirically that a 160M-parameter Transformer trained to copy strings up to length 30 generalizes to several hundred, while comparable-size Mamba/GSS models do not.
- **Retrieval saturates at a state-size-dependent point.** Arora et al.'s Zoology line (ICLR 2024) established that multi-query associative recall accuracy for gated-convolution and recurrent models is governed by recurrent state size, with the gap to attention closing as $N$ grows — the trade-off is recall vs. memory, not an incidental bug.
- **State collapse is fixable without retraining.** Chen et al. report that norm-control interventions on Mamba-2 370M/1.3B restore long-context behavior far past training length, evidence that a large part of the failure is *drift*, not *capacity*.
- **Hybrids at 8B/3.5T tokens** (Waleffe et al.) is currently the largest well-controlled comparison; pure Mamba-2's deficit concentrates on tasks requiring verbatim in-context copying.
- **No positional encoding does not imply no position dependence.** $\Delta_t$ and layer-norm statistics are functions of position through the state, so a selective SSM has an implicit, learned, and training-length-tied notion of "how far back".

## 5. What Is Not Known

- **Theoretically open.** Whether any fixed-parameter selective SSM class admits a *length-uniform* approximation guarantee for a nontrivial task family — error bounded independent of $T$ — under gradient training on length-$L$ samples. No proof either way. Also open: whether the negative-eigenvalue extension (Grazzi et al.) changes length-generalization behavior for natural language, as opposed to synthetic state tracking.
- **Empirically open.** Whether state-collapse mitigation plus curriculum closes $G(\kappa)$ at $\kappa \ge 8$ against a native-context control at $\ge 3$B parameters. Every published mitigation is measured at $\le 1.5$B, mostly on synthetic retrieval. The experiment is runnable today; nobody has published it with the control arm.
- **Methodologically blocked.** Separating "capacity exhausted" from "distribution shifted" in a trained model. Both produce rising loss past $L$. There is no accepted probe that reads out how many facts a given $h_T$ still holds, so the two hypotheses are not currently distinguished by any published measurement.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a missing control**. Three things move at once past $t = L$: (a) the state norm drifts out of the trained regime, (b) the cumulative decay $\exp(a_n\sum_r \Delta_r)$ has already erased distant tokens regardless of length generalization, (c) the evaluation corpus changes character at long lengths. Standard long-context suites (passkey, needle-in-a-haystack) score (a)+(c) while being nearly insensitive to (b), so a model can post a $1$M-token passkey number while its per-token language-modelling loss has diverged at $4L$. RULER (Hsieh et al., 2024) partly fixes this for Transformers but was not designed to separate recurrent drift from recurrent forgetting.

Second obstruction: the control arm is expensive. $G(\kappa)$ as defined requires training a native-context model at $L' = \kappa L$ with matched tokens — for $\kappa = 8$ at 3B parameters, that is a second full pretraining run, plus attention-free-but-still-sequential long-context throughput costs. Most papers skip it, which is why the literature is a pile of uncontrolled benchmark numbers.

## 7. Current Research (as of 2026)

- **Norm/state control.** Direct descendants of the state-collapse analysis: decay floors on $\bar A_t$, state normalization, and forgetting schedules applied at inference. Broadly adopted; still measured mostly on synthetic recall.
- **Hybrid stacks.** NVIDIA, Together/Stanford (Zoology, Based) and the Jamba line treat a small number of attention layers as the length-generalization mechanism and the SSM as the throughput mechanism. This is the pragmatic SOTA and sidesteps rather than solves the problem.
- **Eigenvalue-range and non-diagonal recurrences.** Following Grazzi et al., extending $\bar A$ to $[-1,1]$ or to structured non-diagonal forms to recover state tracking. Whether this helps natural-language extrapolation is untested *(frontier — verify)*.
- **Test-time training and chunked recurrence.** Treating the recurrent state as fast weights updated by an inner loss, which changes the length-scaling story entirely *(frontier — verify)*.
- **Length curricula and document-boundary handling.** Under-reported and probably a large fraction of observed effects, given assumption (ii) in §2.

## 8. Concrete Next Experiment

**Question.** Is post-$L$ degradation in selective SSMs drift or capacity?

**Scale.** Mamba-2, 1.3B parameters, $N = 128$, trained on 100B tokens of a public corpus at $L = 4096$, with document boundaries respected (no cross-document state carryover). Roughly $8\times10^{21}$ FLOPs; days on 64 H100s.

**Arms.**
1. Baseline: as trained, evaluated at $L' = 32768$ ($\kappa = 8$).
2. Drift control: identical checkpoint, inference-time state-norm clamp $\|h_t\| \le \max_{t\le L}\|h_t\|$ (the Chen et al. style intervention). No retraining.
3. Native control: same architecture, same 100B tokens, trained at $L' = 32768$. This is the arm everyone omits.
4. Capacity control: baseline architecture with $N = 512$, retrained at $L = 4096$, same tokens.

**Deciding number.** $G(8)$ in nats/token, averaged over positions $4097$–$32768$, each arm against arm 3.

- If arm 2 cuts $G(8)$ by $\ge 70\%$ while arm 4 cuts it by $\le 20\%$: the failure is drift, and it is fixable at inference. Reclassify the problem toward *partially-solved*.
- If arm 4 dominates arm 2: the failure is capacity, and length generalization for constant-state models is bounded by $N$ — a very different research program.
- If neither arm gets $G(8)$ below $0.10$ nats/token: neither hypothesis is sufficient, and §5's methodological block is the real blocker.

Report $\rho(t)$ and median $\tau_t$ alongside, so the mechanism is visible, not inferred.

## 9. Key References

- **[Foundational]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR, 2022. — arXiv:2111.00396
- **[Foundational]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[SOTA]** Assaf Ben-Kish, Itamar Zimerman, Shady Abu-Hussein, Nadav Cohen, Amir Globerson, Lior Wolf, Raja Giryes. *DeciMamba: Exploring the Length Extrapolation Potential of Mamba.* ICLR, 2025. — arXiv:2406.14528
- **[SOTA]** Yingfa Chen, Xinrong Zhang, Shengding Hu, Xu Han, Zhiyuan Liu, Maosong Sun. *Stuffed Mamba: State Collapse and State Capacity of RNN-Based Long-Context Modeling.* 2024. — arXiv:2410.07145
- **[SOTA]** Roger Waleffe et al. *An Empirical Study of Mamba-based Language Models.* NVIDIA, 2024. — arXiv:2406.07887
- **[Theory]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[Theory]** Yash Sarrof, Yana Veitsman, Michael Hahn. *The Expressive Capacity of State Space Models: A Formal Language Perspective.* NeurIPS, 2024. — arXiv:2405.17394
- **[Theory]** Riccardo Grazzi, Julien Siems, Jörg Franke, Arber Zela, Frank Hutter, Massimiliano Pontil. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR, 2025. — arXiv:2411.12537
- **[Theory]** Samy Jelassi, David Brandfonbrener, Sham Kakade, Eran Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Empirical]** Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, Christopher Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[Benchmark]** Cheng-Ping Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Context]** Cedric Anil et al. *Exploring Length Generalization in Large Language Models.* NeurIPS, 2022. — arXiv:2207.04901

## 10. Worked Example

Take one channel of a Mamba-2 layer with $a_n = -1$ and a typical learned $\Delta$ with median $0.05$ per token. The contribution of a token at distance $k$ is $\exp(-0.05k)$. At $k = 100$ that is $6.7\times10^{-3}$; at $k = 500$, $1.4\times10^{-11}$ — below bfloat16 resolution relative to nearby tokens.

Now the model is trained at $L = 4096$ and evaluated at $32768$. Note what the arithmetic says: for this channel, *nothing at distance $>500$ ever mattered, at either length*. Whatever breaks at $t = 8000$ cannot be "it forgot the token at position 1" — that token was already gone at $t = 600$.

So run the two candidate explanations against the same observable.

- **Capacity.** With $N = 128$ and $d_{\text{state}}$ channels, the state holds a bounded number of retrievable items. If capacity were binding, degradation would be flat in $t$ once the state is full — and the state fills within the first few hundred tokens, i.e. well before $L$. It does not degrade there.
- **Drift.** $\rho(t) = \|h_t\|/\mathbb{E}_{t\le L}\|h_t\|$. Because $\Delta_t$ is input-dependent and the softplus is bounded below by roughly $\Delta_{\min}$, a stretch of tokens with small $\Delta$ makes $\bar A_t \to 1$ and the state accumulates $\bar B_t x_t$ without decaying. Over $8\times$ more positions, the chance of a long low-$\Delta$ run grows, and $\rho$ ratchets up. The readout $C_t^\top h_t$ then operates on norms it never saw in training.

The obstruction is that both stories predict the same per-token loss curve — a knee somewhere past $L$, rising with $t$. They are separated only by $\rho(t)$, which is not logged by any standard long-context evaluation, and by the native-context control of §8 arm 3, which is not run because it costs a second pretraining run. That is why the page's status is *open* rather than *empirically open on a known metric*: the cheap experiment exists, and the field has been reporting the expensive-to-interpret one instead.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*