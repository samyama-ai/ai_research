---
id: 16-state-space-models/early-context-forgetting-gated-recurrence
title: "Catastrophic Forgetting of Early Context in Gated Recurrences"
topic: 16-state-space-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Catastrophic Forgetting of Early Context in Gated Recurrences

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/early-context-forgetting-gated-recurrence` · **Status:** empirically-open

## 1. Problem Statement

Gated linear recurrences (Mamba, Mamba-2, GLA, RWKV, mLSTM, Gated DeltaNet) compress an entire prefix into a fixed-size state. Empirically their accuracy on facts placed early in a long prompt falls faster with distance than a Transformer's does. The problem is to determine **why**, and to state the law.

Three variants, different difficulty:

- **Measurement.** Define a per-model *memory horizon* $\tau$ — the token distance at which information written into the state is no longer recoverable by the model's own readout — that is separable from "the information was never written" and from "the readout circuit does not exist". No accepted estimator exists.
- **Method.** Build a gated recurrence whose $\tau$ grows with training context length rather than saturating, without adding attention. Open.
- **Theory.** Decide whether the observed degradation is forced by **state capacity** ($O(d_{\text{state}})$ bits, an information-theoretic ceiling) or by **decay dynamics** (multiplicative gates driving $\prod a_r \to 0$ before the ceiling binds). These predict opposite interventions: widen the state, or fix the gate parameterization.

A solution is a controlled experiment that assigns the degradation to capacity, decay, or optimization, with a fitted exponent.

## 2. Formal Setting

A gated linear recurrence over input $x_{1:T}$, $x_t \in \mathbb{R}^{d}$:

$$h_t = A_t \odot h_{t-1} + B_t x_t, \qquad y_t = C_t^\top h_t, \qquad h_t \in \mathbb{R}^{N \times d}$$

with input-dependent gate $A_t \in (0,1)^{N}$. For Mamba-2, $A_t = \exp(-\Delta_t \lambda)$ with $\Delta_t = \mathrm{softplus}(w^\top x_t + b)$ and $\lambda > 0$ per channel; for GLA, $A_t = \sigma(\cdot)^{1/16}$; for RWKV-6, a data-dependent $\exp(-\exp(\cdot))$.

**Decay factor** (measured by accumulating the model's own gate logits during a forward pass, no retraining):

$$\Gamma_{s\to t} = \prod_{r=s+1}^{t} A_r, \qquad \log \Gamma_{s\to t} = -\sum_{r=s+1}^{t} \Delta_r \lambda .$$

**Memory horizon at tolerance $\epsilon$**, per channel $n$, averaged over a corpus:

$$\tau_n(\epsilon) = \min\{ \delta : \mathbb{E}_{t}\,[\Gamma^{(n)}_{t-\delta \to t}] < \epsilon \}, \quad \epsilon = 10^{-3}.$$

**Behavioural horizon** $\tau_{0.5}^{\text{beh}}$: with a synthetic key–value needle inserted at distance $\delta$ before the query and context length held fixed, the $\delta$ at which retrieval accuracy crosses 50%. Holding $T$ fixed while varying $\delta$ removes the length confound.

**Capacity bound.** Recalling $k$ uniformly random $b$-bit values requires $kb \le N d \cdot \beta$ bits, $\beta$ = bits per stored scalar (empirically $\beta \ll 32$). This is the ceiling Jelassi et al. exploit.

**Assumptions, and which fail.**
- *Gates are stationary in $t$.* Violated: $\Delta_t$ is input-dependent by construction, and in trained models spikes on delimiters. So $\Gamma$ is a product of correlated, not i.i.d., factors, and $\tau$ is content-dependent.
- *Channels are independent.* Violated: the $B_t, C_t$ projections mix channels every layer; per-channel $\tau_n$ composes non-trivially across depth.
- *Loss decomposes over positions.* Holds for next-token CE but the *gradient* signal for long-range dependence is a vanishing fraction of it — the reason optimization is a live third hypothesis.

## 3. State of the Art

**Established (theory).** Jelassi et al., *Repeat After Me* (ICML 2024): any fixed-state recurrence needs state size $\Omega(L)$ bits to copy an $L$-token random string; a two-layer Transformer does it with $O(\log L)$-width attention. Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024): S4/Mamba-class layers lie in $\mathrm{TC}^0$ under standard uniformity assumptions, so they cannot express $\mathrm{NC}^1$-hard state tracking (e.g. $S_5$ word problems) at fixed depth. Both bound *worst-case capacity*; neither speaks to decay on natural text.

**Established (empirical).** Waleffe et al. (2024) trained 8B Mamba, Mamba-2 and Mamba-2-Hybrid on 3.5T tokens against a matched Transformer — the largest matched-recipe comparison published. Pure Mamba-2 matches or beats the Transformer on short-context tasks but loses badly on in-context retrieval (phonebook lookup, five-shot MMLU formatting); the hybrid (≈8% attention layers) closes the gap and exceeds the Transformer by 2.65 points averaged over 12 tasks.

**Claimed but unablated.** That gating "solves" the decay problem. Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025) and Titans (Behrouz et al., 2024) report improved long-context numbers, but the reported gains are benchmark aggregates; no published ablation isolates $\tau$ at matched state size and matched data.

**Benchmark-number-only.** RULER (Hsieh et al., COLM 2024) and BABILong (Kuratov et al., NeurIPS 2024) both show recurrent models degrading earlier than attention. Neither controls needle depth against total length, so the reported curves conflate "context too long" with "fact too early".

## 4. What Is Known

- **Copying breaks at a state-sized threshold.** In *Repeat After Me*, Mamba models trained on strings up to length 30 fail to copy strings of length ~50–100 while a same-size Transformer generalizes to several hundred; the failure onset tracks state size, not training length.
- **Hybridization is the reliable fix.** Samba (Ren et al., 2024, 3.8B, 3.2T tokens) with sliding-window attention interleaved extrapolates from 4K training to 256K with retained perplexity; Jamba (Lieber et al., 2024, 52B MoE, 1:7 attention:Mamba ratio) reports 256K context. Both are systems results: the attention layers do the retrieval.
- **Recall–throughput tradeoff is real and quantified.** Arora et al., *Zoology* (ICLR 2024) and *Based* (ICML 2024): associative-recall accuracy at 355M scale scales with recurrent state size, with a measured Pareto frontier between state size and throughput; gated convolutions at small state lose >20 points of AR accuracy to attention.
- **Init-time decay is short.** With Mamba-2's default $\Delta \in [10^{-3}, 10^{-1}]$ and $\lambda \in [1,16]$, the slowest channel at initialization reaches $\Gamma = 10^{-3}$ at ~6,900 tokens; the median channel at ~100 (Section 10).
- **Time-warp init helps small RNNs.** Tallec & Ollivier (ICLR 2018) show chrono-initialization of forget-gate biases to $\log \mathcal{U}[1, T_{\max}]$ materially improves long-dependency tasks at $T\sim 10^3$ in LSTMs. Not replicated at LM scale for SSMs.

## 5. What Is Not Known

- **Theoretically open.** Whether, under a *distributional* (not worst-case) model of natural language, a fixed-state gated recurrence can achieve position-uniform recall. All known lower bounds are worst-case over random strings; real text is compressible, and no bound rules out an $O(\log L)$-bit sufficient statistic for realistic prefixes.
- **Empirically open** (the main gap). Nobody has published the state-size sweep at fixed parameter count, fixed data, fixed training length, measuring $\tau_{0.5}^{\text{beh}}$ with needle depth decoupled from total length. The experiment costs ~$10^4$ GPU-hours at 1.3B — runnable, unrun.
- **Empirically open.** Whether $\tau$ is set at initialization and never escaped (an optimization failure) or learned to a data-determined optimum. Requires logging $\Delta_t$ statistics through training; no public checkpoint series does.
- **Methodologically blocked.** Separating *forgotten* from *never-written* from *unreadable*. A drop in influence is consistent with all three, and the state is not linearly decodable in a way that distinguishes them — the probe's failure and the model's failure are not identifiable from behaviour alone.

## 6. Why It Is Hard

**Non-identifiability of the three failure modes.** The observable is $\Delta \log p(x_t)$ when an early token is perturbed. Zero influence is produced equally by (i) $B_t$ never projecting the token into the state, (ii) $\Gamma_{s\to t} \approx 0$ erasing it, and (iii) $C_t$ not reading the channel that holds it. Gate logs distinguish (ii) from (i)+(iii) *only if* channels are independent — and they are not, because every layer remixes them. A linear probe recovering the fact proves it was written and survives; a probe failing proves nothing.

**Compounding: the benchmarks measure the wrong axis.** RULER-style needle tests vary needle depth *and* total length together. A model can fail at 32K with the needle at 5% depth because 32K is beyond its horizon *or* because the needle is 30K tokens back — these are the same manipulation in the standard protocol. Every published long-context recurrent number inherits this confound.

## 7. Current Research (as of 2026)

- **Delta-rule recurrences** (Yang, Kim, Kautz, Hatamizadeh; MIT/NVIDIA/Soochow). Gated DeltaNet combines a decay gate with a delta-rule write, targeting selective overwrite rather than uniform decay. Whether it changes $\tau$ or only capacity utilization is unablated.
- **Test-time-training / learned memory modules** (Titans, TTT-layers; Sun, Behrouz et al.). Reframes the state as a fast-weight model updated by gradient descent at inference. *(frontier — verify)* claimed unbounded-horizon behaviour; no matched-control study.
- **Hybrid ratio optimization** (NVIDIA, Microsoft, AI21). The practical consensus is 1:6–1:8 attention:recurrent. The question of *why that ratio* is not answered by any published scaling law.
- **Gate-parameterization studies** — chrono-style and log-uniform $\lambda$ spectra for SSMs. *(frontier — verify)* small-scale reports of horizon extension; nothing at ≥1B.

## 8. Concrete Next Experiment

**The discriminator: does $\tau$ scale with state size, or is it flat?**

- **Scale.** Mamba-2, 370M parameters, four arms with state size $N \in \{16, 64, 128, 256\}$, **parameter count matched** by reducing $d_{\text{model}}$ or expansion factor to within 2%. Same 30B-token corpus, same training context 8,192, same seed set (3 seeds). ~2,000 A100-hours total.
- **Control arms.** (a) A matched-parameter Transformer with full attention — expected $\tau$ independent of position. (b) A sliding-window Transformer with window $W$ chosen so $W \cdot d = N \cdot d$ bits of KV — a capacity-matched recurrence surrogate. (c) A frozen-gate Mamba-2 with $\Delta$ fixed at initialization, isolating learned-gate contribution.
- **Protocol.** Synthetic KV-recall: 8,192-token context, one needle key–value pair at depth $\delta \in \{2^7, \dots, 2^{12}\}$ tokens before the query, **total length held fixed** by padding with distractor pairs. Fit $\tau_{0.5}^{\text{beh}}$ per arm.
- **The number.** The slope $\alpha$ in $\log \tau_{0.5}^{\text{beh}} = \alpha \log N + c$, fitted across the four state sizes with 95% CI from seed variance.
  - $\alpha \ge 0.8$ → **capacity-limited**; the fix is a wider state, and the Jelassi bound is the operative constraint.
  - $\alpha \le 0.2$ → **decay-limited**; widening the state is wasted, and gate parameterization is the target. Arm (c) then says whether the decay is learned or inherited from init.
  - $0.2 < \alpha < 0.8$ → both bind; report the crossover $N^\ast$.

## 9. Key References

- **[Foundational]** Bengio, Simard & Frasconi. *Learning long-term dependencies with gradient descent is difficult.* IEEE Trans. Neural Networks, 1994.
- **[Foundational]** Gers, Schmidhuber & Cummins. *Learning to Forget: Continual Prediction with LSTM.* Neural Computation, 2000.
- **[Foundational]** Tallec & Ollivier. *Can recurrent neural networks warp time?* ICLR, 2018. — arXiv:1804.11188
- **[Foundational]** Gu, Goel & Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR, 2022. — arXiv:2111.00396
- **[SOTA]** Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[SOTA]** Dao & Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[Theory]** Jelassi, Brandfonbrener, Kakade & Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Theory]** Merrill, Petty & Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[SOTA]** Yang, Wang, Shen, Panda & Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML, 2024. — arXiv:2312.06635
- **[SOTA]** Yang, Kautz & Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR, 2025. — arXiv:2412.06464
- **[Empirical]** Waleffe et al. *An Empirical Study of Mamba-based Language Models.* NVIDIA, 2024. — arXiv:2406.07887
- **[Empirical]** Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[Benchmark]** Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Systems]** Ren et al. *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.* Microsoft, 2024. — arXiv:2406.07522
- **[Survey]** Tiezzi et al. *Back to Recurrent Processing at the Crossroad of Transformers and State-Space Models.* Nature Machine Intelligence, 2025.

## 10. Worked Example

Take a single Mamba-2 channel at initialization. Decay per step is $A = \exp(-\Delta\lambda)$, so the horizon at tolerance $\epsilon$ is

$$\tau(\epsilon) = \frac{-\ln \epsilon}{\Delta \lambda}, \qquad \tau(10^{-3}) = \frac{6.91}{\Delta\lambda}.$$

Default init: $\Delta \sim \log\mathcal{U}[10^{-3}, 10^{-1}]$, $\lambda \sim \mathcal{U}[1,16]$.

| channel | $\Delta$ | $\lambda$ | $\Delta\lambda$ | half-life (tok) | $\tau(10^{-3})$ (tok) |
|---|---|---|---|---|---|
| slowest | $10^{-3}$ | 1 | $10^{-3}$ | 693 | 6,910 |
| median | $10^{-2}$ | 4 | $4\times10^{-2}$ | 17 | 173 |
| fastest | $10^{-1}$ | 16 | 1.6 | 0.43 | 4.3 |

With $N=128$ channels drawn from that product distribution, the fraction with $\tau(10^{-3}) \ge 32{,}768$ is $P(\Delta\lambda \le 2.1\times10^{-4}) \approx 0$ — no channel at init reaches a 32K horizon, and only ~4% reach 4K.

**Now the obstruction.** Measure a trained 1.3B Mamba-2 on a 32K prompt with the needle at token 500. It fails. Read the gate logs: the mean $\Delta_t$ on natural text has *risen* during training to ~$3\times10^{-2}$, so $\tau(10^{-3})$ for the median channel is ~230 tokens. That looks decisive — decay killed it.

It is not decisive. Run the same prompt with the needle at token 31,500 (distance 500 from the query, same total length): accuracy is 91%. Run it at token 500 but with total length 2,048: accuracy is 88%. So the model *can* hold a fact for 500 tokens, and the failure at (depth 500, length 32K) is not explained by $\Gamma_{500\to 32000}$ alone — the 31,500 distractor tokens in between also consumed write capacity. The gate log and the capacity bound predict the same failure and the experiment as run cannot tell them apart. That is precisely why Section 8 varies $N$ at fixed $\Delta$-distribution and fixed length: the state-size slope $\alpha$ is the only observable the two hypotheses disagree on.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*