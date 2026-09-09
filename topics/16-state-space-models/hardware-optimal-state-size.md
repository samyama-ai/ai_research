---
id: 16-state-space-models/hardware-optimal-state-size
title: "Hardware-Optimal State Size Under Memory Bandwidth"
topic: 16-state-space-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hardware-Optimal State Size Under Memory Bandwidth

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/hardware-optimal-state-size` · **Status:** empirically-open

## 1. Problem Statement

A linear-recurrent sequence model (S4, Mamba, GLA, DeltaNet, RWKV) carries a fixed-size hidden state per sequence. Enlarging that state monotonically improves associative recall but linearly increases the bytes that must cross HBM every decode step. Unlike weights, the state is *per-sequence*, so batching does not amortize it.

**The question.** Given an accelerator with peak throughput $\pi$ FLOP/s and memory bandwidth $\beta$ B/s, a parameter budget $P$, a token budget $D$, and a serving regime (batch $B$, context $T$), is there a state size $N^\star$ that maximizes quality per unit of *achieved* throughput — and does $N^\star$ move when the machine balance $\pi/\beta$ moves?

Three variants, different difficulty:

- **Measurement.** Define "quality per achieved throughput" so that it is comparable across $N$ at fixed $P$ and $D$. Requires kernels that are equally well tuned at every $N$ — currently they are not.
- **Method.** Find the architecture that, at a given $\pi/\beta$, sits on the Pareto frontier of (validation loss, tokens/s). Empirically open.
- **Theory.** Prove a lower bound linking recall capacity to state bits, then compose it with a roofline bound to derive $N^\star$ in closed form. Partially available (communication-complexity recall bounds exist); the composition does not.

**Solved** means: a published $N^\star(\pi/\beta, P, D, B, T)$ that predicts, out of sample, which state size wins on a hardware generation it was not fitted on.

## 2. Formal Setting

Let a layer have model width $d$, expansion $E$, head dimension $P_h$, head count $H = Ed/P_h$, state dimension $N$, and $L$ layers. Mamba-2's state is per-head $S_h \in \mathbb{R}^{P_h \times N}$, recurrence

$$S_t = A_t \odot S_{t-1} + v_t k_t^\top, \qquad y_t = S_t q_t,$$

with $A_t$ a scalar or diagonal decay. **Measured** state footprint, in bytes at precision $b$:

$$M_{\text{state}} = b \cdot L \cdot E d \cdot N .$$

**Decode arithmetic intensity.** Per token per sequence, the kernel reads and writes the state and does $\Theta(EdN)$ multiply-adds:

$$I_{\text{dec}} = \frac{4\,L\,EdN}{2b\,L\,EdN + M_{\text{weights}}/B} \xrightarrow[B \to \infty]{} \frac{2}{b} \ \text{FLOP/B}.$$

At $b=2$ (bf16) this is $1$ FLOP/byte — three orders below the H100 balance $\pi/\beta \approx 989/3.35 \approx 295$ FLOP/B. **This is the whole problem: state reads never become compute-bound.**

**Training arithmetic intensity.** Chunked scan with chunk $C$ costs, per head per chunk, $\Theta(C^2 P_h)$ intra-chunk plus $\Theta(C P_h N)$ inter-chunk, against HBM traffic $\Theta(b\,C(P_h+N) + b\,P_h N)$. Intensity rises with $C$ until the tile $P_h \times N$ no longer fits shared memory ($228$ KB/SM on H100): fp32 state tile bytes $=4 P_h N$, so $P_h N \le 5.7\times10^4$ before spilling — e.g. $P_h=64$ forces $N \le 890$.

**Objective.** With $\mathcal{L}(N)$ validation loss at fixed $(P, D)$ and $R(N)$ measured tokens/s at fixed $(B,T)$ on the target device,

$$N^\star = \arg\min_N \ \mathcal{L}(N) \quad \text{s.t.} \quad R(N) \ge R_0 .$$

**Assumptions, and which are violated.**
1. *Kernel efficiency is $N$-independent.* Violated — Mamba-2's SSD kernel hits tensor cores; Mamba-1's scan does not, so its $N=16$ is not a fair point on the same curve.
2. *$\mathcal{L}$ depends on $N$ only through $N$.* Violated — holding $P$ fixed while raising $N$ steals parameters from depth/width.
3. *State is stored dense in bf16.* Violated by fp32 state accumulation in most reference kernels, which doubles $M_{\text{state}}$.
4. *Bandwidth is the only serving constraint.* Violated once $B \cdot M_{\text{state}}$ exceeds device capacity — then it is a memory-capacity problem instead.

## 3. State of the Art

**Established.**
- Mamba-2 (Dao & Gu, ICML 2024) showed the state-space-duality reformulation makes large $N$ affordable: $N$ moved from $16$ (Mamba-1) to $64$–$256$, with the SSD kernel reported 2–8× faster than Mamba-1's scan at matched $N$. The *mechanism* (matmul-form chunking uses tensor cores) is ablated.
- Gated Linear Attention (Yang et al., ICML 2024) established chunked, tensor-core-friendly kernels for gated linear attention and made the chunk-size/state-size trade-off explicit in FLASH-LINEAR-ATTENTION.
- Zoology / Based (Arora et al., ICLR 2024; ICML 2024) established the recall–state-size frontier: MQAR accuracy is governed by recurrent state bits, and the frontier is smooth, not a cliff.

**Claimed but unablated.**
- That $N=128$ or $256$ is "the right" default. Every major release picks a value (Mamba-2 $N=128$; Gated DeltaNet $N=128$; Nemotron-H and Qwen3-Next hybrids similar) without a sweep at fixed FLOPs *and* fixed measured throughput. These are configuration choices, not measured optima.
- That larger $N$ pays for itself at scale. Asserted from small-scale recall curves; not shown at 7B+ with matched token budgets.

**Benchmark-number-only.** Public "Mamba is $5\times$ faster than Transformers at inference" figures are batch- and length-specific throughput points on one device. They are not roofline analyses and do not transfer across $\pi/\beta$.

## 4. What Is Known

- **Recall scales with state, measured at 355M.** On MQAR and real recall tasks, Based/Zoology showed accuracy rising monotonically with recurrent state size across roughly two orders of magnitude of state, at 355M-parameter scale (Arora et al., 2024). Gap to attention closes but does not vanish.
- **Copying is state-limited, provably.** Jelassi et al. (ICML 2024) proved a fixed-state recurrent model cannot copy strings longer than its state capacity, and measured the failure at 160M scale; transformers of equal size copy far longer strings.
- **Expressivity is state-shape-limited, not state-size-limited, for some tasks.** Merrill, Petty & Sabharwal (ICML 2024) proved diagonal SSMs lie in $\mathsf{TC}^0$ and cannot solve $S_5$ word problems — growing $N$ does not help. This bounds what bandwidth spending can buy.
- **Large-scale hybrid results.** NVIDIA's 8B-parameter, 3.5T-token study (Waleffe et al., 2024) found pure Mamba-2 trails a transformer on in-context copy/recall tasks while a ~8% attention hybrid matches or beats it. State size was held at the default, not swept.
- **Machine balance moved 2×.** A100 80GB: $312$ TFLOP/s bf16 / $2.039$ TB/s $\approx 153$ FLOP/B. H100 SXM: $989/3.35 \approx 295$ FLOP/B. B200-class parts push higher still. Any bandwidth-derived $N^\star$ should therefore have shifted between hardware generations — nobody has reported that it did.

## 5. What Is Not Known

- **Empirically open.** Whether $\mathcal{L}(N)$ at fixed $(P,D)$ has an interior optimum, or is monotone-decreasing so that $N$ is purely a serving-cost dial. Runnable today at 1–3B; unrun with matched kernels.
- **Empirically open.** Whether $N^\star$ shifts with $\pi/\beta$. The A100-vs-H100 comparison is a two-week experiment and has not been published.
- **Theoretically open.** No lower bound of the form "achieving loss $\ell$ on a distribution with recall entropy $\mathcal{H}$ requires $\Omega(f(\mathcal{H}))$ state bits" that is tight enough to predict $N^\star$. Communication-complexity arguments give existence, not constants.
- **Methodologically blocked.** "Quality per achieved throughput" is not measurable while kernel maturity varies with $N$. A slow $N=512$ kernel and an intrinsically bad $N=512$ architecture produce the same measurement.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement, in two layers.**

1. *Kernel confound.* Achieved throughput $R(N)$ is a product of an architectural term and an engineering term. Mamba-2's own history shows the engineering term dominating: the same $N$ went from scan-bound to tensor-core-bound with no architecture change. Any measured $N^\star$ is therefore an artifact of which $N$ someone tuned. Removing the confound requires a roofline-efficiency floor (say, $\ge 60\%$ of the bandwidth bound) at *every* $N$ in the sweep — that is kernel work, not research.
2. *Parameter-allocation confound.* Raising $N$ at fixed $P$ removes parameters elsewhere. Without a matched control that spends the same parameters on depth, the sweep measures an allocation policy, not state size.

Secondary: cost. A clean sweep is $5$ arms $\times$ 1.3B $\times$ 100B tokens $\approx 4\times10^{21}$ FLOPs, roughly 6k H100-hours at 40% MFU — affordable for a lab, not for an individual, which is why it sits unrun.

## 7. Current Research (as of 2026)

- **Hybrids as the de facto answer.** Nemotron-H (NVIDIA, 2025), Jamba (AI21), Falcon-H1 (TII, 2025), Qwen3-Next (Alibaba, 2025) all interleave a few full-attention layers with linear-recurrent ones. This sidesteps the state-size question: attention layers supply exact recall, so $N$ is set by throughput alone. *(frontier — verify the exact layer ratios per release.)*
- **Richer state update rules instead of larger state.** Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025) argues delta-rule updates buy recall more cheaply than dimensionality. Directly competes with "just raise $N$".
- **Flash-Linear-Attention (Songlin Yang, Yu Zhang, MIT/CMU)** is the main vehicle for kernel parity across $N$ — the tool that would unblock the measurement.
- **Serving-side state compression** (quantized or low-rank recurrent state) is emerging as the analogue of KV-cache quantization. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Scale.** Five Mamba-2 arms at $P \approx 1.3$B, $D = 100$B tokens of a fixed corpus, $N \in \{32, 64, 128, 256, 512\}$, head dim $P_h = 64$ fixed. Parameters re-matched to within $\pm 1\%$ by adjusting $L$ and $d$.

**Kernel precondition.** Every arm must reach $\ge 60\%$ of its bf16 bandwidth roofline in decode and $\ge 35\%$ MFU in training, verified with Nsight Compute before any loss is reported. Arms failing this are excluded, not reported.

**Control arms.** (a) A GQA transformer at the same $P$ and $D$ whose KV cache at $T=8192$ equals the $N=128$ arm's state bytes. (b) A depth-matched $N=128$ arm that spends the extra parameters of the $N=512$ arm on layers instead — this isolates the allocation confound.

**Cross-hardware leg.** Measure $R(N)$ for all arms on both A100 ($\pi/\beta = 153$) and H100 ($295$), at $B \in \{1, 32, 256\}$, $T = 8192$.

**The deciding number.** $\Delta = \log_2 N^\star_{\text{A100}} - \log_2 N^\star_{\text{H100}}$, where $N^\star$ is the arm minimizing validation loss subject to $R(N) \ge 0.8\,R(N{=}32)$. If $|\Delta| \ge 1$ (one doubling), state size is hardware-dependent and every fixed default in the literature is device-specific. If $\Delta = 0$ across a 1.9× swing in machine balance, $N$ is a quality dial and the bandwidth framing is wrong.

## 9. Key References

- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024. — arXiv:2312.00752
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024. — arXiv:2312.06635
- **[SOTA]** Songlin Yang, Jan Kautz, Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. — arXiv:2412.06464
- **[Empirical]** Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, Christopher Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR 2024. — arXiv:2312.04927
- **[Empirical]** Simran Arora et al. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML 2024. — arXiv:2402.18668
- **[Theory]** Samy Jelassi, David Brandfonbrener, Sham Kakade, Eran Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[Theory]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[Empirical]** Roger Waleffe et al. *An Empirical Study of Mamba-based Language Models.* NVIDIA, 2024. — arXiv:2406.07887
- **[Foundational]** Samuel Williams, Andrew Waterman, David Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM 52(4), 2009.
- **[Survey]** Badri N. Patro, Vijay S. Agneeswaran. *Mamba-360: Survey of State Space Models as Transformer Alternative.* 2024.

## 10. Worked Example

A Mamba-2 model at $d=2560$, $E=2$, $L=64$, $P_h=64$, $N=128$, bf16 state.

State per layer: $5120 \times 128 = 655{,}360$ elements $= 1.31$ MB. Whole model:

$$M_{\text{state}} = 64 \times 1.31\ \text{MB} = 83.9\ \text{MB per sequence.}$$

Decode reads and writes it: $168$ MB per token per sequence. Weights are $2.7$B params $= 5.4$ GB, read once per step regardless of batch. **Break-even batch:**

$$B^\star = \frac{5400\ \text{MB}}{168\ \text{MB}} = 32 .$$

At $B=256$ the state traffic is $43$ GB per step against $5.4$ GB of weights — 89% of all bytes moved. On H100 at 80% of $3.35$ TB/s, one decode step costs $18.1$ ms, i.e. $14{,}100$ tok/s aggregate, $55$ tok/s per sequence.

Now the obstruction. Doubling to $N=256$ costs $86$ GB/step $\to$ $34.0$ ms $\to$ $7{,}500$ tok/s: **a 47% throughput loss.** What does it buy? Zoology's recall curves at 355M say a doubling of state moves MQAR accuracy by single-digit points in the regime where the model is state-limited, and by essentially zero once it is not. Whether the 1.3B–7B pretraining loss moves at all is exactly what nobody has measured with matched kernels.

And the confound is visible in the same arithmetic. Compare against a GQA transformer with 8 KV heads of dim 128: KV bytes per token per layer $= 2 \times 1024 \times 2 = 4$ KB, so $262$ KB/token over 64 layers. The Mamba-2 state equals $83.9\ \text{MB} / 0.262\ \text{MB} = 320$ tokens of KV cache. **Below $T = 320$, the "constant-memory" recurrent model moves more bytes per decode step than the attention model it replaces.** At $T = 8192$ it moves 25× fewer. The break-even is a pure function of $N$ — which means the correct $N$ is not a property of the architecture, but of the context length and batch you intend to serve. No published sweep controls for either.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*