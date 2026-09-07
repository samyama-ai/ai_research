---
id: 16-state-space-models/decay-spectrum-parameterization
title: "Optimal Decay Spectrum Parameterization"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Decay Spectrum Parameterization

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/decay-spectrum-parameterization` · **Status:** open

## 1. Problem Statement

Every linear-recurrent sequence layer — S4, S5, LRU, Mamba, Mamba-2, GLA, RetNet, HGRN, xLSTM — carries a set of per-channel eigenvalues $\lambda_n$ inside the unit disk. Their moduli fix how long each channel remembers. The design choice is not the eigenvalues themselves but the **map from unconstrained parameters to eigenvalues**, plus the distribution used to initialize them. Call this pair the *decay spectrum parameterization*.

The problem: given a token budget, a state width, and a target task distribution, choose the parameterization $\lambda = \phi(\theta)$ and initial law $\mathcal{D}_0$ over $\theta$ that minimize final loss.

Three variants, of different difficulty:

- **Measurement.** Given a trained model, recover the decay spectrum it actually uses and the *memory horizon* it exercises. Confounded by non-identifiability (§6).
- **Method.** Find $(\phi, \mathcal{D}_0)$ that dominates on a fixed compute budget. Runnable; run only at small scale.
- **Theory.** Prove that a parameterization achieves a stated approximation or optimization rate for a class of targets with given memory decay. Partially answered for linear systems, open for the gated, input-dependent case.

Solving it means: a rule that takes $(L, N, \text{token budget})$ and emits $(\phi, \mathcal{D}_0)$, beating tuned baselines without per-task search.

## 2. Formal Setting

A diagonal recurrent layer with state width $N$, input $u_t \in \mathbb{R}^{d}$, state $x_t \in \mathbb{C}^{N}$:

$$x_t = \Lambda_t\, x_{t-1} + B_t u_t, \qquad y_t = \Re\!\left(C_t x_t\right) + D u_t,$$

with $\Lambda_t = \mathrm{diag}(\lambda_{t,1},\dots,\lambda_{t,N})$, $|\lambda_{t,n}| < 1$.

**Parameterizations in use.**

| Family | Map $\phi$ | Time-varying? |
|---|---|---|
| S4D / S4 | $\lambda_n = \exp(\Delta_n a_n)$, $\Re a_n<0$, $\Delta_n=\exp(\delta_n)$ | no |
| LRU | $\lambda_n=\exp(-\exp(\nu_n)+i\exp(\theta_n))$ | no |
| Mamba (S6) | $\lambda_{t,n}=\exp(-\Delta_t(u_t)\,\exp(a_n))$, $\Delta_t=\mathrm{softplus}(\cdot)$ | yes |
| Mamba-2 / SSD | $\lambda_{t,n}=a_t$ scalar, shared over $n$ | yes |
| GLA / HGRN | $\lambda_{t,n}=\sigma(\cdot)^{1/\tau}$, optionally lower-bounded per layer | yes |
| RetNet | $\lambda_n=\gamma_h$ fixed per head | no |

**Measured quantities.**

- *Per-channel horizon*: $h_n = -1/\log|\lambda_n|$, in tokens. For $\lambda$ near 1, $h_n \approx (1-|\lambda_n|)^{-1}$.
- *Effective horizon of the layer*: residue-weighted, $\bar h = \sum_n w_n h_n$ with $w_n = |b_n c_n|^2/\sum_m |b_m c_m|^2$. Weighting by residue rather than counting poles is essential: a pole with zero residue is invisible in the transfer function $H(z)=\sum_n b_nc_n/(1-\lambda_n z^{-1})$.
- *Empirical horizon*: $h^{\mathrm{emp}}$ from an intervention — perturb the input at position $t-k$, measure $\|\partial y_t/\partial u_{t-k}\|$, and take the $k$ at which it falls below $e^{-1}$ of its $k=0$ value. This is the operational definition; the algebraic $h_n$ is a proxy for it and diverges from it once $\Lambda_t$ is input-dependent.
- *Spectral density*: $\rho(r) = \frac{1}{N}\sum_n \delta(|\lambda_n| - r)$, measured after training, weighted or unweighted.

**Assumptions, and which fail.**

1. *Diagonalizability / diagonal state.* Holds by construction post-S4D; S4's DPLR is a strict superset, and Gupta et al. (2022) showed the low-rank term is not needed empirically.
2. *Stability $|\lambda|<1$ enforced by $\phi$.* Holds for exponential and sigmoid maps; **violated** by unconstrained-real parameterizations still used in some ablations.
3. *Time-invariance.* **Violated** in every competitive 2024+ model. Then $h_n$ is defined only along a realized token sequence: $h_{n}(t) = -\left(\frac{1}{k}\sum_{s=t-k+1}^{t}\log|\lambda_{s,n}|\right)^{-1}$, and depends on the data.
4. *Linearity of the recurrence.* Holds inside the layer; **violated** across layers, so layer-$\ell$ horizon does not compose to model horizon $\ell \cdot \bar h$.
5. *Task memory is stationary.* **Violated** for natural language, where long-range dependence is heavy-tailed rather than exponential.

## 3. State of the Art

**Established.**

- *Initialization of the spectrum decides whether long-range tasks train at all.* Gu et al. (S4D, NeurIPS 2022) ablated initializations of $a_n$ under a fixed architecture: HiPPO-derived inits (S4D-Lin, S4D-Inv) train Path-X; naive random-diagonal init does not exceed chance (50%). This is a controlled ablation, not a benchmark artifact.
- *Log/exponential parameterization plus a ring initialization recovers S4-class quality from a plain linear RNN.* Orvieto et al. (LRU, ICML 2023) ablate step by step — linear recurrence, then exponential (stable) parameterization, then ring init $|\lambda|\sim\mathcal{U}[r_{\min},r_{\max}]$, then the $\sqrt{1-|\lambda|^2}$ normalization — and show each stage moves LRA. This is the cleanest existing ablation of $\phi$ vs $\mathcal{D}_0$.
- *A scalar decay per head suffices for language at 1–3B.* Dao & Gu (Mamba-2, ICML 2024) replace the per-channel $\Lambda_t$ with $a_t I$ and trade the lost expressivity for $8\times$ larger state; perplexity holds or improves.

**Claimed but unablated.**

- That input-dependent (selective) decay is *necessary* rather than merely sufficient at language scale. Mamba's headline results confound selective $\Delta$, selective $B,C$, and the architecture block.
- That monotone layerwise lower bounds on forget gates (HGRN, NeurIPS 2023) reflect a real hierarchy of timescales rather than an optimization aid. The mechanism claim is not separated from the conditioning claim.
- That the softplus-$\Delta$ + $\log\Delta \sim \mathcal{U}[\log 10^{-3}, \log 10^{-1}]$ init in Mamba is near-optimal. It is inherited from S4D and reused, not re-searched.

**Benchmark-number-only.** Most cross-family comparisons of decay design (RetNet $\gamma$ vs GLA $\sigma$-gates vs Mamba $\Delta$) exist only as leaderboard rows with different tokenizers, data, and widths. No paper varies *only* $\phi$ across these families at matched compute.

## 4. What Is Known

- **Approximation.** Linear-recurrence-plus-nonlinear-projection stacks are universal for continuous causal functionals with *exponentially decaying memory* (Orvieto et al., 2023; Wang & Xue, NeurIPS 2023). Targets with polynomially decaying memory are approximable only at a cost that blows up — the "curse of memory".
- **Reparameterization changes the achievable rate, not just the optimizer path.** Wang & Li (StableSSM, ICML 2024) prove that stable reparameterizations of the form $\lambda = \phi(\theta)$ with $\phi'$ vanishing at the stability boundary lift the curse of memory for a class of linear functionals; the identity map does not.
- **Sharpness near $|\lambda|\to 1$.** As eigenvalues approach the unit circle, the loss Hessian along the eigenvalue direction grows like $h^2$; this is why $\exp(-\exp(\nu))$-style maps work — they make the loss curvature in $\nu$ approximately scale-free in $h$ (Zucchet & Orvieto, NeurIPS 2024).
- **Numbers, with scale.** LRA (Tay et al., ICLR 2021), $\sim$100–600k-parameter models, $L$ up to 16384: S4 and S5 report averages near 86–87 and Path-X above 96; LRU reports an average near 86 and Path-X above 94; all pre-S4D diagonal models with untuned init report Path-X at chance (50%). At language scale, Mamba-2 at 1.3B/2.7B on 300B Pile tokens matches or beats Mamba at equal parameters with $N$ raised from 16 to 64–256.
- **The range that matters is narrow.** Path-X needs $h \gtrsim 10^4$; LRU's Path-X configuration uses $r_{\min}=0.999, r_{\max}=0.9999$, i.e. all initial mass in $h \in [10^3, 10^4]$. Move $r_{\max}$ to $0.99$ and the task fails.

## 5. What Is Not Known

- **Theoretically open.** No approximation or optimization theorem covers *input-dependent* decay. Every rate above assumes $\Lambda$ constant in $t$. Whether selectivity buys a strictly better rate for any natural target class is unproven either way.
- **Theoretically open.** Optimal $\mathcal{D}_0$ as a function of $(N, L)$. Ring-uniform-in-radius, log-uniform-in-$\Delta$, and HiPPO-derived inits induce different $\rho(r)$; no result says which minimizes expected approximation error for a stated prior over target memory kernels.
- **Empirically open.** A matched-compute sweep over $\phi$ at $\geq$1B parameters and $\geq$100B tokens, holding data, tokenizer, block, and $N$ fixed. Every ingredient exists; nobody has published the grid.
- **Methodologically blocked.** "Memory horizon of a trained model" has no agreed measurement. Algebraic $h_n$, residue-weighted $\bar h$, and gradient-based $h^{\mathrm{emp}}$ disagree by more than an order of magnitude on the same checkpoint, and only the last is defined for gated models.

## 6. Why It Is Hard

**Non-identifiability is the specific obstruction.** The layer's input–output map depends on $(\lambda_n, b_nc_n)$ only through the transfer function. Two failure modes follow:

1. *Dead poles.* Channels with $|b_nc_n|\approx 0$ have unconstrained $\lambda_n$ — gradient pressure on them is proportional to their residue. Post-hoc histograms of $|\lambda_n|$ therefore measure the *initialization* in those channels, not anything learned. Published spectra plots almost never residue-weight.
2. *Timescale/scale trade.* In $\lambda = \exp(\Delta a)$ only the product enters; $\Delta$ and $a$ are individually meaningless. Ablations that "tune $\Delta$ init" while holding $a$ fixed are tuning one coordinate of a rank-deficient parameterization.

Secondary: the decisive experiment is expensive in the wrong place. Path-X-style tasks are cheap but reward one narrow horizon band; language is where the answer matters and where a 6-arm sweep at 1.3B/300B tokens is $\sim$10$^5$ GPU-hours. And short-context evaluation cannot distinguish $h=10^3$ from $h=10^6$, so the standard benchmark suite does not measure the thing it is used to argue about.

## 7. Current Research (as of 2026)

- **Gated delta rules.** Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025) couples a scalar decay with a rank-1 delta update, separating "forget everything slowly" from "overwrite one key". Whether the decay parameterization still matters once a delta term exists is the open sub-question. *(frontier — verify)*
- **Decay in attention.** Forgetting Transformer (Lin et al., ICLR 2025) adds a data-dependent forget gate to softmax attention, giving a direct control arm: same decay parameterization, non-recurrent mixer.
- **Hybrid stacks** (Jamba, Zamba, Nemotron-H lines) place a small number of attention layers among many recurrent ones, which may make the recurrent layers' long-horizon spectrum irrelevant. Untested as a stated hypothesis. *(frontier — verify)*
- **Groups:** Gu/Dao (CMU, Princeton, Cartesia); Yang & Kim (MIT); Orvieto & De (ELLIS Tübingen, Google DeepMind); Qin & Zhong (Shanghai AI Lab / TapTap); Hochreiter's group (JKU, xLSTM).

## 8. Concrete Next Experiment

**Question.** Does the *shape* of the initial spectral density $\rho(r)$ affect language-model loss once decay is input-dependent, or does training wash it out?

- **Scale.** 370M-parameter Mamba-2-style model, $N=64$, 8 seeds per arm, 30B tokens of a fixed corpus (e.g. FineWeb-Edu), context 8192. About 5k A100-hours total — affordable, and above the scale where LRA-sized effects vanish.
- **Arms (only $\mathcal{D}_0$ varies; $\phi$, data order, and all else fixed).**
  1. Mamba default: $\log\Delta \sim \mathcal{U}[\log 10^{-3},\log 10^{-1}]$ (**control**).
  2. Horizon-log-uniform over $[10^1, 10^3]$.
  3. Horizon-log-uniform over $[10^1, 10^5]$ — mass beyond context length.
  4. All modes at $h = 10^2$ (degenerate spectrum).
  5. Arm 1 with $\Delta$ frozen at init (decay not learned).
- **Deciding number.** Residue-weighted $\bar h$ measured at the end of training, per arm. If arms 1–4 converge to $\bar h$ within a factor of 2 of each other **and** their validation losses agree within 2 standard errors across seeds, the parameterization debate is settled negatively: init shape is washed out and only the map's conditioning matters. If arm 3 or 4 ends more than $3\times$ away in $\bar h$ *and* loses $\geq 0.02$ nats, initialization is a first-class design variable at scale. Arm 5 bounds how much of any gap is learned versus inherited.
- **Secondary readout.** $h^{\mathrm{emp}}$ from gradient-decay probes at $k \in \{10^2,10^3,10^4\}$, to test whether $\bar h$ and $h^{\mathrm{emp}}$ track each other — which is the methodological blocker in §5.

## 9. Key References

- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR, 2022. — arXiv:2111.00396
- **[Foundational]** Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré. *On the Parameterization and Initialization of Diagonal State Space Models.* NeurIPS, 2022. — arXiv:2206.11893
- **[Foundational]** Ankit Gupta, Albert Gu, Jonathan Berant. *Diagonal State Spaces are as Effective as Structured State Spaces.* NeurIPS, 2022. — arXiv:2203.14343
- **[SOTA]** Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, Soham De. *Resurrecting Recurrent Neural Networks for Long Sequences.* ICML, 2023. — arXiv:2303.06349
- **[SOTA]** Jimmy T.H. Smith, Andrew Warrington, Scott W. Linderman. *Simplified State Space Layers for Sequence Modeling.* ICLR, 2023. — arXiv:2208.04933
- **[SOTA]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[SOTA]** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML, 2024. — arXiv:2312.06635
- **[SOTA]** Songlin Yang, Jan Kautz, Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR, 2025. — arXiv:2412.06464
- **[Theory]** Shida Wang, Qianxiao Li. *StableSSM: Alleviating the Curse of Memory in State-Space Models through Stable Reparameterization.* ICML, 2024.
- **[Theory]** Nicolas Zucchet, Antonio Orvieto. *Recurrent neural networks: vanishing and exploding gradients are not the end of the story.* NeurIPS, 2024.
- **[Related]** Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei. *Retentive Network: A Successor to Transformer for Large Language Models.* 2023. — arXiv:2307.08621
- **[Related]** Zhen Qin, Songlin Yang, Yiran Zhong. *Hierarchically Gated Recurrent Neural Network for Sequence Modeling.* NeurIPS, 2023.
- **[Related]** Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, et al. *xLSTM: Extended Long Short-Term Memory.* NeurIPS, 2024. — arXiv:2405.04517
- **[Survey/Benchmark]** Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler. *Long Range Arena: A Benchmark for Efficient Transformers.* ICLR, 2021. — arXiv:2011.04006

## 10. Worked Example

**Setup.** Path-X: $L = 16384$. Take one LRU layer, $N = 256$, ring init $|\lambda| \sim \mathcal{U}[r_{\min}, r_{\max}]$.

**Step 1 — what horizon the task requires.** A mode contributes signal from position 1 to position $L$ only if $|\lambda|^{L} \not\approx 0$, i.e. $-L\log|\lambda| \lesssim 1$, so

$$|\lambda| \gtrsim \exp(-1/16384) = 1 - 6.1\times 10^{-5}.$$

**Step 2 — how many modes the default init supplies.** With $r_{\min}=0, r_{\max}=1$, the fraction of modes above that threshold is $6.1\times10^{-5}$. With $N=256$: expected count $= 0.016$. With the "safe" setting $r_{\max}=0.99$ ($h_{\max}=100$): expected count $=0$, exactly. The task is unlearnable at init not because the model lacks capacity but because **no channel is initialized in the required band**, and gradients that would move a channel there are proportional to that channel's residue, which is itself small.

**Step 3 — why the parameterization, not just the init, matters.** LRU uses $r = \exp(-\exp(\nu))$. Then

$$\frac{\partial r}{\partial \nu} = -r\,e^{\nu} = r\log(1/r).$$

At $r = 0.9999$ this is $10^{-4}$: a unit step in $\nu$ moves the radius by $10^{-4}$ — it cannot leave the disk. But in horizon terms $h = 1/\log(1/r)$, so

$$\frac{\partial \log h}{\partial \nu} = 1.$$

One optimizer step of size $\eta$ multiplies the horizon by $e^{\eta}$, at every scale. Under the identity parameterization $r = \theta$, $\partial \log h/\partial \theta = 1/((1-r)\log(1/r)) \approx h^2$; at $h = 10^4$ that is $10^{8}$ — a learning rate that moves short modes destroys long ones in one step.

**Step 4 — the obstruction made visible.** Train this layer and histogram $|\lambda_n|$ at the end. Suppose 12 of 256 modes sit above $1-10^{-4}$. Now residue-weight: if those 12 carry $\sum|b_nc_n|^2$ equal to $0.3\%$ of the total, $\bar h$ is dominated by the short modes and is $\approx 10^2$, while the unweighted median of $h_n$ reads $\approx 10^3$ and the max reads $10^4$. Three defensible statistics from one checkpoint, spanning two orders of magnitude, all called "the model's memory". Until the field fixes one — and shows it tracks the interventional $h^{\mathrm{emp}}$ — comparisons between decay parameterizations are comparisons between measurement conventions.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*