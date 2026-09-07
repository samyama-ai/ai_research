---
id: 32-hardware-and-kernels/analog-compute-transformer-tolerance
title: "Analog In-Memory Compute Error Tolerance for Transformer Inference"
topic: 32-hardware-and-kernels
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Analog In-Memory Compute Error Tolerance for Transformer Inference

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/analog-compute-transformer-tolerance` · **Status:** empirically-open

## 1. Problem Statement

Analog in-memory computing (AIMC) performs a matrix-vector product in one step inside a memory crossbar: weights are stored as device conductances, inputs as voltages or pulse durations, and the summed current is the output. It removes the weight-movement cost that dominates transformer decoding, at the price of a matrix-vector product that is *stochastic and drifting* rather than exact.

The problem: **how much analog error can a decoder-only transformer absorb before generation quality degrades, and does that budget fit inside the error of real devices over a real deployment lifetime?**

Three variants, different difficulty:

- **Measurement.** Define an error budget that predicts *generative* degradation, not just next-token perplexity on a static corpus. Currently under-defined.
- **Method.** Train or adapt a model so its accuracy is preserved under a given noise model (hardware-aware training, noise-aware quantization, outlier suppression). Partially solved for CNNs and RNNs; unproven at LLM scale on real silicon.
- **Theory.** Bound end-to-end output perturbation as a function of per-MVM noise, through $L$ layers, softmax attention, and $T$ autoregressive steps. Open.

A solution is: a noise threshold $\sigma^\*$ in device units, validated on hardware, with a stated confidence that models below it lose $<1$ point of downstream accuracy over $\geq 1$ year.

## 2. Formal Setting

Let a linear layer have weights $W \in \mathbb{R}^{m \times d}$ mapped to a differential conductance pair with scale $\alpha$: $G_{ij} = \alpha W_{ij} + \varepsilon_{ij}$, with $|W_{ij}| \le W_{\max}$ and $G \in [0, G_{\max}]$.

The realized product is

$$\hat{y} = \tfrac{1}{\alpha}\,Q_b\!\Big( \sum_{j} G_{ij}(t)\,\phi(x_j) + \eta_i \Big),$$

with the following quantities, each as measured:

- **Programming error** $\varepsilon_{ij} \sim \mathcal{N}(0,\sigma_{\text{prog}}^2)$. Measured as the standard deviation of read-back conductance minus target, averaged over a full array, expressed as a percentage of $G_{\max}$.
- **Drift** $G_{ij}(t) = G_{ij}(t_0)(t/t_0)^{-\nu_{ij}}$, $\nu_{ij}\sim\mathcal{N}(\mu_\nu,\sigma_\nu^2)$. Measured by re-reading the same array at logarithmically spaced times after programming; $\mu_\nu$ is removed by a global output rescale, so only $\sigma_\nu$ is irreducible.
- **Read noise** $\eta$: $1/f$ plus thermal, measured as output-referred current variance at fixed input over a read window.
- **ADC/DAC quantization** $Q_b$ at $b$ bits over clipping range $[-c,c]$; $\phi$ is the input DAC. $c$ is set per tile from calibration activations. Clipping is measured as the fraction of MVM outputs with $|y| > c$.
- **Effective weight error** $\sigma_w^{\text{eff}}(t)^2 = \sigma_{\text{prog}}^2 + \big(\sigma_\nu \ln(t/t_0)\big)^2 \bar{G}^2 + \sigma_{\text{read}}^2$, in units of $W_{\max}$.

Task-level objective: with $\mathcal{M}$ a metric (perplexity, MMLU, pass@1) and $\tau$ a tolerance,

$$\sigma^\*(\tau, t) = \sup\{\sigma_w^{\text{eff}} : \mathbb{E}[\mathcal{M}_{\text{analog}}] - \mathcal{M}_{\text{digital}} \le \tau \ \text{at time } t\}.$$

**Assumptions known to be violated in practice.** (i) Gaussian, i.i.d., zero-mean $\varepsilon$ — real PCM/RRAM error is conductance-dependent, asymmetric, and spatially correlated within a tile. (ii) Independent drift exponents — $\nu$ correlates with programmed state. (iii) Linear crossbar summation — IR drop and sneak paths make the map input-dependent. (iv) Static noise — conductance relaxation is non-stationary in the first seconds after programming. (v) That per-layer perturbations compose additively; attention softmax is not Lipschitz-uniform in its inputs.

## 3. State of the Art

**Systems/empirical SOTA (established, on silicon).**
- Ambrogio et al., *Nature* 2023: a 14 nm analog-AI chip with 35 M PCM devices, 12.4 TOPS/W sustained, MLPerf keyword spotting at 86.14 % and LibriSpeech RNN-T at 9.475 % WER — within measurement noise of software. Real hardware, small models.
- Le Gallo et al., *Nature Electronics* 2023: 64-core PCM chip, 17.6 M devices, ResNet-9/CIFAR-10 at 92.81 % on-chip.
- Wan et al. (NeuRRAM), *Nature* 2022: 48-core RRAM chip, CIFAR-10 85.7 %, Google speech commands 84.7 %, 1.6–2.3$\times$ better energy-delay product than prior CIM at iso-accuracy.

**Method SOTA.** Hardware-aware training (Rasch et al., *Nature Communications* 2023) injects the noise model during fine-tuning and recovers most CNN/RNN accuracy; optimized weight programming (Mackin et al., *Nature Communications* 2022) reduces effective $\sigma_{\text{prog}}$ by iterative program-and-verify.

**Claimed but unablated.** Transformer results on AIMC are almost entirely *simulated*. Spoon et al. (*Frontiers in Computational Neuroscience*, 2021) report near-software accuracy for BERT-class encoders under a PCM noise model — a simulator number, not a chip number. Analog attention proposals (ReTransformer, ICCAD 2020; Leroux et al., analog in-memory attention, 2024–25) report energy and latency projections from device models; the accuracy figures are simulation. **No decoder-only LLM above ~1 B parameters has been run end-to-end on analog hardware and evaluated on a standard generative benchmark.** "Analog foundation model" fine-tuning recipes appeared in 2025 (IBM Research) but report simulator perplexity.

**Theory SOTA.** No end-to-end bound. The closest is generic error-propagation analysis for quantized networks and the noise-robustness results in the CrossSim accuracy analysis (Xiao et al., *IEEE Circuits and Systems Magazine*, 2022), which is layer-local.

## 4. What Is Known

- **PCM programming precision.** Iterative program-and-verify reaches roughly 2–4 % of $G_{\max}$ standard deviation per device; ~4–5 bits of usable weight precision per device pair, measured on IBM 14 nm PCM arrays (Joshi et al., *Nature Communications* 2020).
- **Drift is the dominant long-horizon term.** PCM drift exponents cluster around $\mu_\nu \approx 0.03$–$0.1$. Joshi et al. measured ResNet-32/CIFAR-10 at 93.7 % just after programming, ~92.6 % one day later, with global drift compensation — about 1.1 points lost to drift alone at 10 M parameters.
- **Hardware-aware training buys roughly one bit.** Across CNNs/RNNs, HWA training moves the tolerated weight noise from ~1–2 % to ~4–6 % of $W_{\max}$ (Rasch et al. 2023, models up to ~100 M parameters).
- **Transformers are outlier-heavy.** LLM.int8() (Dettmers et al., NeurIPS 2022) showed systematic activation outliers of 20–100$\times$ the typical channel magnitude emerging above ~6.7 B parameters; SmoothQuant (Xiao et al., ICML 2023) mitigates them by migrating scale into weights. This is a *digital* result that directly sets the analog ADC dynamic-range problem.
- **Attention is the sensitive block.** Ablations in quantization literature consistently find the softmax input and the KV path least tolerant of error; the same ordering appears in AIMC simulations.

## 5. What Is Not Known

- **Empirically open.** Whether a $\geq 7$ B decoder-only model, HWA-fine-tuned, holds within 1 point of MMLU and $+0.1$ nats of perplexity at $\sigma_{\text{prog}} = 3\%$ and $t = 1$ year. The experiment is runnable in simulation today and blocked on hardware only by array capacity. Nobody has published it at that scale.
- **Empirically open.** Whether analog error compounds across autoregressive steps. Perplexity is teacher-forced; a 4096-token generation with per-step noise is a different process, and no published study measures degradation as a function of generation length on analog noise.
- **Methodologically blocked.** There is no validated mapping from simulator noise model to measured chip error. AIHWKit and CrossSim both fit device statistics, but no paper reports the simulator-vs-silicon residual for a transformer. Until that residual is measured, every simulated transformer AIMC number has unquantified error bars.
- **Theoretically open.** No bound of the form $\|\hat{h}_L - h_L\| \le f(\sigma, L, T)$ that survives softmax attention. Whether attention is contracting or amplifying for small weight perturbations, in the regime real models occupy, is unproven either way.

## 6. Why It Is Hard

The specific obstruction is **dynamic range, not noise**. In a digital pipeline an outlier activation channel costs a scale factor — per-channel or per-group scaling is free. In a crossbar, the ADC input range is a physical current budget shared by every column in the tile. A 20$\times$ outlier forces $c$ up by $20\times$, and since ADC bits are fixed by area and energy (each extra bit roughly doubles ADC energy), that outlier consumes $\log_2 20 \approx 4.3$ bits of resolution from every *other* column in the same tile. The property transformers have and CNNs do not — heavy-tailed activation channels — is exactly the property that analog arrays cannot absorb cheaply.

Second obstruction: **confounded measurement**. Reported AIMC accuracy conflates the noise model, the HWA fine-tune, the digital layers kept off-chip (usually LayerNorm, softmax, sometimes attention entirely), and the drift-compensation schedule. Papers rarely state what fraction of the network was actually analog. A "software-equivalent" claim where 30 % of MACs ran in digital measures something other than analog tolerance.

## 7. Current Research (as of 2026)

- **IBM Research Zurich / Almaden** — analog foundation-model fine-tuning, noise-aware LLM training, AIHWKit; PCM chips at 14 nm. Most credible path to a hardware LLM demonstration. *(frontier — verify current array capacity.)*
- **Sandia National Laboratories** — CrossSim, accuracy modelling for analog accelerators, including ADC-aware simulation.
- **Forschungszentrum Jülich / RWTH** — analog attention and KV-cache-in-memory architectures (Leroux, Sebastian and collaborators).
- **Tsinghua / Peking University** — large RRAM arrays and on-chip learning; strongest device-scale integration results.
- **Startups (Mythic, EnCharge, Sagence)** — EnCharge's switched-capacitor analog approach sidesteps device drift by computing in charge domain rather than conductance; the error model differs qualitatively. *(frontier — verify published accuracy data.)*

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B (or Qwen2.5-7B), all linear layers of all 32 blocks mapped to a simulated 512$\times$512 PCM tile array in AIHWKit or CrossSim, with 8-bit ADC and per-tile calibrated clipping. Attention scores and LayerNorm digital, and *report that fraction explicitly*.

**Procedure.** HWA fine-tune for 1 B tokens at the target noise. Sweep $\sigma_{\text{prog}} \in \{1,2,3,4,6\}\%$ of $G_{\max}$; evaluate at $t \in \{1\text{s}, 1\text{h}, 1\text{d}, 1\text{y}\}$ with global drift compensation, $\mu_\nu = 0.06$, $\sigma_\nu = 0.01$.

**Control arms.** (a) W8A8 digital SmoothQuant at iso-throughput. (b) *Iso-noise digital control*: i.i.d. Gaussian weight noise of the same total variance, no ADC clipping, no drift correlation. Arm (b) is the load-bearing control — if it predicts the analog result, the analog-specific structure does not matter and the whole simulator apparatus is unnecessary.

**Deciding number.** $\sigma^\*_{1\text{y}}$: the largest $\sigma_{\text{prog}}$ at which MMLU stays within 1.0 point of FP16 after one simulated year. If $\sigma^\*_{1\text{y}} \geq 3\%$, today's PCM suffices and the problem moves to engineering. If $\sigma^\*_{1\text{y}} < 1.5\%$, device precision must improve by $\geq 2\times$ before LLM AIMC is viable, and the field should say so.

## 9. Key References

- **[Foundational]** Sebastian, Le Gallo, Khaddam-Aljameh, Eleftheriou. *Memory devices and applications for in-memory computing.* Nature Nanotechnology, 2020.
- **[Foundational]** Joshi, Le Gallo, Haefeli, Boybat, Nandakumar, Piveteau, Dazzi, Rajendran, Sebastian, Eleftheriou. *Accurate deep neural network inference using computational phase-change memory.* Nature Communications 11:2473, 2020.
- **[SOTA]** Ambrogio, Narayanan, Okazaki, Fasoli, Mackin, Hosokawa, et al. *An analog-AI chip for energy-efficient speech recognition and transcription.* Nature 620, 2023.
- **[SOTA]** Le Gallo, Khaddam-Aljameh, Stanisavljevic, Vasilopoulos, Kersting, et al. *A 64-core mixed-signal in-memory compute chip based on phase-change memory for deep neural network inference.* Nature Electronics 6, 2023.
- **[SOTA]** Wan, Wu, Hu, Liu, Deiss, et al. *A compute-in-memory chip based on resistive random-access memory.* Nature 608, 2022.
- **[Method]** Rasch, Mackin, Le Gallo, Chen, Fasoli, Odermatt, et al. *Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators.* Nature Communications 14:5282, 2023.
- **[Method]** Mackin, Rasch, Chen, Timcheck, Bruce, et al. *Optimised weight programming for analogue memory-based deep neural networks.* Nature Communications 13:3765, 2022.
- **[Tooling]** Rasch, Moreda, Gokmen, Le Gallo, Carta, et al. *A flexible and fast PyTorch toolkit for simulating training and inference on analog crossbar arrays.* IEEE AICAS, 2021.
- **[Tooling]** Xiao, Feinberg, Bennett, Agrawal, Prabhakar, et al. *On the accuracy of analog neural network inference accelerators.* IEEE Circuits and Systems Magazine, 2022.
- **[Transformer]** Spoon, Tsai, Chen, Rasch, Ambrogio, et al. *Toward software-equivalent accuracy on transformer-based deep neural networks with analog memory devices.* Frontiers in Computational Neuroscience, 2021.
- **[Context]** Dettmers, Lewis, Belkada, Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[Context]** Xiao, Lin, Seznec, Wu, Demouth, Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML, 2023. — arXiv:2211.10438
- **[Survey]** Burr, Shelby, Sebastian, Kim, Kim, et al. *Neuromorphic computing using non-volatile memory.* Advances in Physics: X, 2017.

## 10. Worked Example

One $4096 \times 4096$ projection from a 7 B-class model. Weights approximately $\mathcal{N}(0, 0.02^2)$; clip at $4\sigma$, so $W_{\max} = 0.08$. Inputs unit-variance.

**Signal.** $\mathrm{std}(y_i) = 0.02\sqrt{4096} = 1.28$.

**Analog weight error at programming.** $\sigma_{\text{prog}} = 3\%$ of $W_{\max}$ gives per-weight error $0.0024$. Accumulated: $0.0024\sqrt{4096} = 0.154$. SNR $= 8.3$ (18.4 dB), relative output error 12 %.

**INT8 control.** Step $\Delta = 2 \cdot 0.08 / 255 = 6.3\times10^{-4}$, error std $\Delta/\sqrt{12} = 1.8\times10^{-4}$, accumulated $0.0117$. SNR $= 110$ (40.8 dB).

So a freshly programmed analog tile is $\approx 13\times$ noisier than INT8 — about **5.3 effective weight bits**. Tolerable; HWA training handles this regime for CNNs.

**Now add one day of drift.** With $\sigma_\nu = 0.01$ and $t/t_0 = 8.64\times10^{4}$, $\ln(t/t_0) = 11.4$. After removing the mean drift by a global rescale, the residual relative spread per device is $\sigma_\nu \ln(t/t_0) = 0.114$ — an **11.4 % per-weight error, nearly 4$\times$ the programming error**, and it grows as $\ln t$: at one year it is 0.176.

**Now add one outlier channel.** Set the ADC range from calibration: $c = 4\,\mathrm{std}(y) = 5.12$ at 8 bits, quantization step $0.04$ — negligible. Introduce one activation channel at $20\times$ typical magnitude, as LLM.int8() reports for models above ~6.7 B. That column's output reaches $\approx 25$, so $c$ must rise to $\approx 26$. Step becomes $0.20$, error std $0.059$, accumulated over the tile $0.059\sqrt{4096}/\sqrt{4096}$ per output $= 0.059$ — comparable to a further 1.5 % weight error, and it is charged to **every** column sharing the ADC.

**What the example makes visible.** Programming precision is not the binding constraint. The budget is consumed by (i) drift residual growing logarithmically with deployment time and (ii) a shared, physically fixed dynamic range that transformer outliers exhaust. Both are properties the CNN and RNN demonstrations never had to face — which is why the existing hardware results do not transfer, and why $\sigma^\*_{1\text{y}}$ has to be measured rather than extrapolated.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*