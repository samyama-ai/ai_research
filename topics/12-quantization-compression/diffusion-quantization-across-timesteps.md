---
id: 12-quantization-compression/diffusion-quantization-across-timesteps
title: "Diffusion Model Quantization Across Denoising Steps"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Diffusion Model Quantization Across Denoising Steps

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/diffusion-quantization-across-timesteps` · **Status:** open

## 1. Problem Statement

A diffusion model calls the same network $\epsilon_\theta$ tens to thousands of times, once per denoising step $t$. Post-training quantization (PTQ) methods built for one-shot networks assign a single quantizer to each tensor. But the activation statistics inside $\epsilon_\theta$ change by orders of magnitude across $t$, and the errors made at step $t$ are fed back as input at step $t-1$. The problem: **choose a per-step (or step-shared) quantization policy that minimizes end-of-trajectory sample distribution error under a fixed bit and latency budget.**

Three variants, with different difficulty:

- **Measurement.** Given a quantized sampler, attribute the final degradation to individual steps. There is no accepted per-step error metric that composes to the observed FID gap. Currently blocked, not merely unsolved.
- **Method.** Search over per-step bit-widths, scales, and calibration sets to beat a uniform policy at equal average bits. Runnable today; the search space is $O(B^{T \cdot L})$ for $B$ bit choices, $T$ steps, $L$ layers, and nobody has run it exhaustively even at small scale.
- **Theory.** Prove a bound on the Wasserstein or TV distance between quantized and full-precision sample distributions as a function of per-step quantization error. Open.

Solved would mean: a policy that, at a stated average bit-width, provably (or reproducibly, across ≥3 architectures and ≥2 samplers) matches full-precision sample quality within measurement noise, with a per-step allocation rule derived from measurable statistics rather than fit to a benchmark.

## 2. Formal Setting

Let $x_T \sim \mathcal{N}(0,I)$ and let a deterministic sampler (DDIM, Song et al. 2021) iterate

$$x_{t-1} = \Phi_t\big(x_t, \epsilon_\theta(x_t, t)\big), \qquad t = T, \dots, 1 .$$

A quantizer $Q_{s,z,b}(v) = s\big(\mathrm{clip}(\lfloor v/s \rceil + z, 0, 2^b-1) - z\big)$ has scale $s$, zero-point $z$, bit-width $b$. A **policy** $\pi = \{(b^w_{\ell,t}, b^a_{\ell,t}, s_{\ell,t}, z_{\ell,t})\}$ assigns quantizers to each layer $\ell$ and step $t$. Write the quantized network $\epsilon_{\theta}^{\pi}$ and quantized trajectory $\tilde x_t$.

**Quantities as measured.**

- *Per-step local error*: $e_t = \|\epsilon_\theta^\pi(\tilde x_t,t) - \epsilon_\theta(\tilde x_t,t)\|_2 / \|\epsilon_\theta(\tilde x_t,t)\|_2$, measured on the same input $\tilde x_t$ — this isolates the quantizer from drift.
- *Trajectory drift*: $d_t = \|\tilde x_t - x_t\|_2/\sqrt{n}$, where $x_t$ is the FP trajectory from the same $x_T$ and same noise seeds.
- *Activation range*: $r_{\ell,t} = \max |a_{\ell,t}| $ over a calibration batch, and the dynamic range ratio $\rho_\ell = \max_t r_{\ell,t} / \min_t r_{\ell,t}$.
- *Budget*: average bits $\bar b = \frac{1}{T}\sum_t \sum_\ell n_\ell b_{\ell,t} / \sum_\ell n_\ell$, plus measured latency and peak memory on a named device. Per-step weight bits are only free if weights are re-loaded per step; otherwise the weight policy must be step-shared.
- *Objective*: $\mathcal{D}\big(p_\theta^\pi(x_0),\, p_\theta(x_0)\big)$, in practice FID/sFID against a reference set of 10k–50k samples, plus paired per-seed LPIPS to the FP sample.

**Assumptions and their status.**

1. *Calibration data is drawn from the same distribution as inference-time activations.* Violated: calibration sets are built from FP trajectories, but at inference the model sees quantized trajectories. This is train/test mismatch inside the calibration procedure itself.
2. *Layerwise error is additive and locally independent across steps.* Violated: PTQD (He et al., NeurIPS 2023) shows quantization error has a component strongly correlated with the FP output, i.e. it acts like a systematic gain change, not white noise.
3. *FID measures sample distribution distance.* Weakly violated: FID is Inception-feature Gaussian-moment distance, biased by sample count and insensitive to per-seed trajectory divergence.
4. *Average bits predicts latency.* Violated: mixed-precision kernels have per-precision overhead; a 3.7-bit policy can be slower than a uniform 4-bit one.

## 3. State of the Art

**Established (ablated, and reproduced by later work):**

- **Timestep-aware calibration.** PTQ4DM (Shang et al., CVPR 2023) and Q-Diffusion (Li et al., ICCV 2023) both show that sampling calibration data across the full $t$ range, rather than from one $t$, is what makes W4A8 PTQ work at all. Q-Diffusion additionally splits the shortcut-concatenated UNet layers, whose two input branches have incompatible ranges — this split is independently reproduced.
- **Per-step activation quantizers help.** TDQ (So et al., NeurIPS 2023) predicts the activation step size from $t$ with a small MLP; the gain over a static step size is the paper's core ablation and holds at W4A8 on CIFAR-10 and LSUN.
- **Time-embedding layers are disproportionately sensitive.** TFMQ-DM (Huang et al., CVPR 2024) isolates the temporal-information block and shows that quantizing it alone accounts for a large share of the degradation.
- **Low-rank outlier absorption.** SVDQuant (Li et al., ICLR 2025) moves outliers into a 16-bit low-rank branch so the 4-bit path sees a narrower range, with measured end-to-end speedups on consumer GPUs.

**Claimed but under-ablated:**

- That *bit-width* (as opposed to calibration and scale placement) should vary with $t$. MixDQ (Zhao et al., ECCV 2024) and Q-DiT-style mixed-precision work allocate bits by sensitivity, but the reported ablations mostly separate mixed-precision from uniform at equal average bits without controlling calibration budget.
- That error-correction at inference (PTQD's bias and correlation correction) generalizes past the DDIM/LDM settings it was tuned on.

**Benchmark-number-only results.** Most W4A8 and W4A4 claims on SDXL, PixArt, and FLUX-class models exist as a single FID/CLIP-score pair on COCO prompts at one sampler and one step count. No published grid varies sampler × step count × policy on the same checkpoint.

## 4. What Is Known

- Naive PTQ that ignores $t$ collapses. Reported W4A8 on CIFAR-10 DDIM with single-timestep calibration gives FID in the tens versus FP ≈ 4; timestep-spread calibration recovers most of the gap (PTQ4DM, CVPR 2023; Q-Diffusion, ICCV 2023, at 32M–100M-parameter UNets).
- W4A8 on LDM-class models (LSUN Bedroom/Church, ~270M params, 100–200 DDIM steps) is reported within roughly 0.3–1.0 FID of full precision by Q-Diffusion and PTQD.
- W8A8 is essentially free across every scale measured, from 35M-parameter CIFAR UNets to 12B-parameter FLUX.1 — this is the one uncontested regularity.
- W4A4 is where methods split: SVDQuant reports usable 4-bit weight *and* activation inference on FLUX.1-dev/PixArt-$\Sigma$ (12B / 0.6B) with ~3.5× memory reduction and multi-× latency gain on a 16 GB RTX 4090, only by keeping a 16-bit low-rank branch.
- Activation dynamic range across $t$ in specific UNet layers spans more than an order of magnitude ($\rho_\ell > 10$), measured in TDQ and TFMQ-DM at LDM scale — this is the empirical fact the whole subfield rests on.
- Few-step models are harder, not easier: with 1–4 steps (SDXL-Turbo, LCM) there is no error-averaging over a long trajectory, and MixDQ reports that naive W8A8 already degrades one-step SDXL-Turbo noticeably.

## 5. What Is Not Known

- **Theoretically open.** No bound of the form $\mathcal{D}(p^\pi, p) \le f(\{e_t\}, \text{sampler}, \text{score smoothness})$. Diffusion sampling has convergence guarantees under bounded score error (e.g. Chen et al., ICLR 2023, "Sampling is as easy as learning the score"), but nobody has instantiated those bounds with a quantization error model that matches measured error — which is correlated with the signal, not adversarial and not Gaussian. Whether quantization error contracts or amplifies along the trajectory is unproven in either direction.
- **Empirically open.** The optimal bit schedule $b^\star(t)$ is unknown. Folklore says "early steps (high $t$) matter more for layout, late steps for texture, so spend bits early"; the opposite allocation has never been reported as a controlled ablation at equal $\bar b$ on the same checkpoint. Runnable now; a full grid is a few thousand GPU-hours.
- **Methodologically blocked.** Per-step error attribution. Because $e_t$ is measured on the drifted input $\tilde x_t$, local error and accumulated drift are confounded, and no published decomposition separates them. Without it, "step $t$ contributed $X$ FID" is not a well-defined statement.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** Local quantization error and trajectory drift are entangled by construction: to measure the error a quantizer makes at step 10, you must supply an input, and the only two choices — the FP input (unrealistic) or the quantized input (contaminated by steps 50–11) — measure different things. Teleport-style interventions (quantize step $t$ only, all else FP) cost a full sampling run per step and still perturb the input distribution the later steps see.
2. **Absent ground truth per sample.** There is no reference for "the correct image" of a seed under a different numerical path; FID compares distributions and cannot say whether a specific quantized trajectory failed. Paired LPIPS to the FP sample is a proxy that penalizes benign reparameterization.
3. **Search cost.** Per-step per-layer bit allocation is combinatorial and every objective evaluation is a full sampling run plus a 10k-sample FID. One evaluation on an SDXL-class model is ~1 GPU-hour; a 1000-point search is a small cluster-week — and the resulting policy may not be kernel-realizable.

## 7. Current Research (as of 2026)

- **Diffusion-transformer quantization.** DiT/PixArt/FLUX backbones have different outlier structure than UNets — channel-wise activation outliers resembling LLM behavior. ViDiT-Q (ICLR 2025) extends this to video DiTs, where the token count makes activation quantization the binding constraint. Groups: MIT HAN Lab, Tsinghua NICS-EFC, Infinigence-AI.
- **Rotation/smoothing transplanted from LLMs.** Hadamard rotations and SmoothQuant-style migration applied per-timestep rather than once *(frontier — verify whether the rotation must itself vary with $t$)*.
- **Quantization-aware distillation of few-step samplers**, where the student is trained with the quantizer in the loop and the step schedule is co-designed with the bit schedule *(frontier — verify)*.
- **Kernel co-design.** The practical question has shifted from "what is the best policy" to "what policies have fast kernels", which is quietly narrowing the search space to step-shared weight bits with per-step activation scales.

## 8. Concrete Next Experiment

**Question.** At equal average bits, does a monotone bit schedule over $t$ beat a uniform one — and in which direction?

**Scale.** LDM-4 on LSUN Bedroom (~270M-parameter UNet), DDIM with $T=50$ and $T=20$, plus PixArt-$\Sigma$ (0.6B) on COCO prompts as a second architecture. Weights fixed at W4 and step-shared (kernel-realizable); only *activation* bits vary with $t$.

**Arms**, all at $\bar b_a = 6$ exactly, all with identical calibration budget (1024 samples spread uniformly over $t$) and identical scale-search:
- **Control:** uniform A6 at every step.
- **Front-loaded:** A8 for the first third of steps (high $t$), A6 middle, A4 last third.
- **Back-loaded:** the mirror image.
- **Sensitivity-fit:** bits allocated proportional to measured $\rho_\ell$-weighted per-step Hessian trace, quantized to hit $\bar b_a = 6$.

**Deciding number.** FID-10k gap to full precision, $\Delta\text{FID}$, with 5 seeds per arm and the seed standard deviation reported. The question is settled if one schedule beats the uniform control by more than $3\sigma$ on both models and both step counts. Report paired LPIPS-to-FP as a secondary check that the FID gain is not a distribution-shift artifact. Cost estimate: 4 arms × 2 models × 2 step counts × 5 seeds ≈ 160 sampling+FID runs, on the order of 300–500 A100-hours.

A null result is informative: it would say per-step *bit-width* is a red herring and that per-step *scale* (TDQ-style) already captures the available gain.

## 9. Key References

- **[Foundational]** Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020. — arXiv:2006.11239
- **[Foundational]** Song, Meng, Ermon. *Denoising Diffusion Implicit Models.* ICLR, 2021. — arXiv:2010.02502
- **[Foundational]** Rombach, Blattmann, Lorenz, Esser, Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR, 2022. — arXiv:2112.10752
- **[Foundational]** Nagel, Amjad, van Baalen, Louizos, Blankevoort. *Up or Down? Adaptive Rounding for Post-Training Quantization.* ICML, 2020. — arXiv:2004.10568
- **[Foundational]** Li, Gong, Tan, Yang, Hu, Zhang, Yu, Wang, Gu. *BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction.* ICLR, 2021. — arXiv:2102.05426
- **[SOTA]** Shang, Yuan, Xie, Wu, Yan. *Post-training Quantization on Diffusion Models.* CVPR, 2023. — arXiv:2211.15736
- **[SOTA]** Li, Liu, Du, Zhang, Zhang, Keutzer, Dong, Gholami. *Q-Diffusion: Quantizing Diffusion Models.* ICCV, 2023. — arXiv:2302.04304
- **[SOTA]** He, Liu, Qian, Wang, Hu, Cai, Wang, Ma. *PTQD: Accurate Post-Training Quantization for Diffusion Models.* NeurIPS, 2023. — arXiv:2305.10657
- **[SOTA]** So, Lee, Park, Kim, Chang, Kim. *Temporal Dynamic Quantization for Diffusion Models.* NeurIPS, 2023. — arXiv:2306.02316
- **[SOTA]** Huang, Gong, Liu, Chen, Fu, Zhang, Yu, Doermann, Liu. *TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models.* CVPR, 2024. — arXiv:2311.16503
- **[SOTA]** Zhao, Xu, Zhu, Ning, Wang, Dai, Yang, Wang. *MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed-Precision Quantization.* ECCV, 2024. — arXiv:2405.17873
- **[SOTA]** Li, Lin, Liu, Zhang, Cai, Chen, Han. *SVDQuant: Absorbing Outliers by Low-Rank Component for 4-Bit Diffusion Models.* ICLR, 2025. — arXiv:2411.05007
- **[Theory]** Chen, Chewi, Li, Li, Salim, Zhang. *Sampling is as Easy as Learning the Score: Theory for Diffusion Models with Minimal Data Assumptions.* ICLR, 2023. — arXiv:2209.11215
- **[Survey]** Gholami, Kim, Dong, Yao, Mahoney, Keutzer. *A Survey of Quantization Methods for Efficient Neural Network Inference.* Book chapter / arXiv, 2021. — arXiv:2103.13630

## 10. Worked Example

Take LDM-4 on LSUN Bedroom, DDIM, $T=50$, weights W4, activations A6.

Measure the range of a single mid-block activation across steps. Reported behavior in TDQ/TFMQ-DM for such layers: $r_{\ell,50} \approx 0.9$ at high $t$ and $r_{\ell,1} \approx 12$ at low $t$, so $\rho_\ell \approx 13$.

Fit one static symmetric 6-bit quantizer to the union range $[-12, 12]$. The step size is $s = 24/63 \approx 0.381$. At $t=50$, where activations live in $[-0.9, 0.9]$, the number of codes actually used is $\lceil 1.8/0.381 \rceil \approx 5$ of 64 — an effective precision of $\log_2 5 \approx 2.3$ bits. The relative quantization error there is roughly $s/(2\sqrt{3}\cdot \sigma_a)$; with $\sigma_a \approx 0.3$ this is about $0.37$, i.e. 37% RMS error on that tensor, while the same quantizer at $t=1$ costs about 3%.

So the early steps are being run at an effective 2.3 bits while nominally spending 6. A per-step scale ($s_t = 2 r_{\ell,t}/63$) restores all 64 codes at every $t$ and costs one scalar per layer per step — kilobytes. That is why TDQ-style per-step scales work, and it is settled.

Now the unsettled part. Suppose you fix scales per step and ask whether to *also* shift bits: drop $t \in [1,17]$ to A4 and raise $t \in [34,50]$ to A8 at constant $\bar b_a = 6$. Predicting the sign of $\Delta\text{FID}$ requires knowing how a 37%→9% error reduction at $t=50$ propagates through 50 subsequent steps versus a 3%→6% increase over the last 17. Measuring the first quantity means running the sampler with step 50 quantized and steps 49–1 in FP; the resulting $x_{49}$ then differs from the FP one, so every later measurement is taken on a different input than the FP reference — and $e_t$ and $d_t$ cannot be separated. Both allocations are defensible from the local numbers, and no published experiment distinguishes them. That is the obstruction, in one instance.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*