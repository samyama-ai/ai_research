---
id: 02-attention/massive-activations-attention-outliers
title: "Massive Activations and Outlier Features in Attention"
topic: 02-attention
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Massive Activations and Outlier Features in Attention

> **Topic:** Attention Mechanisms · **ID:** `02-attention/massive-activations-attention-outliers` · **Status:** open

## 1. Problem Statement

Trained transformer language models develop a small number of hidden-state coordinates whose values exceed the median activation magnitude by three to four orders of magnitude, concentrated on a handful of token positions (usually position 1 and delimiters). These co-occur with **attention sinks**: heads that route most of their softmax mass to those same positions. The phenomenon is universal across architectures and training recipes, breaks per-tensor low-bit quantization, and is causally load-bearing — zeroing the outliers destroys the model.

Three variants, of different difficulty:

- **Measurement.** Define an outlier feature and a sink so that the definition is scale-free, basis-justified, and reproducible across models. Currently every paper uses its own threshold.
- **Method.** Train a model with no massive activations that matches a baseline on loss, long-context behaviour, and downstream tasks, and quantizes to W4A4 without rotation tricks. Partially achieved at $\le 1$B; unverified at frontier scale.
- **Theory.** Prove *why* softmax normalization plus a residual stream forces the optimizer into this solution, and whether any bounded-activation attention variant can realize the same function class at equal loss.

Solving it means: a mechanistic account that predicts *which* dimensions and *which* positions become outliers before training finishes, plus a training-time intervention that removes them at no quality cost at $\ge 7$B parameters.

## 2. Formal Setting

Let a decoder-only model of $L$ layers, hidden width $d$, produce residual-stream states $h^{(\ell)} \in \mathbb{R}^{T \times d}$ for a sequence of $T$ tokens. Write $h^{(\ell)}_{t,i}$ for coordinate $i$ at position $t$.

**Massive activation.** As measured: run a fixed corpus $\mathcal{D}$ (e.g. 100 sequences of 4096 tokens from WikiText-2/C4), collect all $|h^{(\ell)}_{t,i}|$, and define the outlier set

$$\mathcal{O}^{(\ell)} = \left\{ (t,i) : |h^{(\ell)}_{t,i}| > \kappa \cdot \operatorname{median}_{t',i'} |h^{(\ell)}_{t',i'}| \right\}, \quad \kappa \approx 10^3 .$$

Sun et al. (2024) use both $\kappa=1000$ and an absolute floor of 100. The threshold is a convention, not a derived quantity — see §6.

**Outlier feature (channel-level).** A coordinate $i$ is an outlier channel if $\max_{t,\ell} |h^{(\ell)}_{t,i}| > \tau$ on at least a fraction $p$ of sequences. Dettmers et al. (2022) use $\tau = 6.0$, $p$ defined over $\ge 6\%$ of layers and $\ge 0.1\%$ of tokens. The quantization-relevant statistic is the per-tensor dynamic range

$$R^{(\ell)} = \frac{\max_{t,i} |h^{(\ell)}_{t,i}|}{\text{RMS}(h^{(\ell)})},$$

since INT8/INT4 per-tensor error scales as $R^{(\ell)}/2^{b}$.

**Attention sink.** For head $(\ell,h)$ with softmax matrix $A^{(\ell,h)} \in \mathbb{R}^{T\times T}$, the sink mass at position $s$ is

$$\sigma^{(\ell,h)}_s = \frac{1}{T-s+1}\sum_{t \ge s} A^{(\ell,h)}_{t,s}.$$

Gu et al. (2025) call position $s$ a sink when $\sigma_s$ exceeds a threshold (they use $0.3$) averaged over heads.

**Bias hypothesis.** Sun et al.'s claim is that massive activations act as a *constant, input-independent bias*: there exist fixed $k', v' \in \mathbb{R}^{d_h}$ such that appending them to every head's key/value set reproduces the model's function without the outliers. The testable form is

$$\operatorname{Attn}(q, [K; k'], [V; v']) \approx \operatorname{Attn}_{\text{orig}}(q, K, V).$$

**Assumptions, and which are violated.** (i) *The residual basis is privileged* — true for post-LN/RMSNorm architectures with elementwise scaling, since normalization and Adam's per-coordinate preconditioning break rotational symmetry (Elhage et al., 2023); false for the rotated models QuaRot/SpinQuant produce, where outliers are basis artifacts. (ii) *Outlier positions are stable across inputs* — mostly true for position 1, violated for delimiter/newline sinks whose identity shifts with domain. (iii) *Softmax mass must sum to 1, forcing a dump site* — the mechanism most cited; violated by design in sigmoid or off-by-one attention, which still sometimes shows outliers.

## 3. State of the Art

**Established (independently reproduced).**
- Outlier features emerge as a phase transition with scale; LLM.int8() (Dettmers et al., NeurIPS 2022) mitigates them by mixed-precision decomposition, keeping ~0.1% of dimensions in FP16.
- SmoothQuant (Xiao et al., ICML 2023) migrates activation outliers into weights via a per-channel diagonal rescale; W8A8 near-lossless up to OPT-175B.
- Rotation-based quantization — QuaRot (Ashkboos et al., NeurIPS 2024) and SpinQuant (Liu et al., ICLR 2025) — applies computational-invariant Hadamard/learned rotations so outliers spread across coordinates. QuaRot reports W4A4KV4 LLaMA-2-70B within ~0.6 WikiText-2 perplexity of FP16.
- Attention sinks are necessary for streaming: keeping the first few KV entries restores long-context perplexity (StreamingLLM, Xiao et al., ICLR 2024).

**Claimed but under-ablated.**
- *Massive activations are attention biases and nothing more.* Supported by the explicit-bias experiment in Sun et al. (2024) at GPT-2 scale; the substitution has not been run as a pretraining intervention at $\ge 7$B.
- *Sinks prevent over-mixing / representational collapse* (Barbero et al., 2025). Mechanistically appealing, argued from Gemma and LLaMA-family measurements; no controlled model pair isolates the effect.
- *Sink formation is a data/optimization artifact, not architectural.* Gu et al. (ICLR 2025) show sinks appear only after enough tokens and vary with LR, weight decay and data mixture, up to 1B.

**Benchmark-number-only.** Most "we removed outliers" claims are reported as post-quantization perplexity or a zero-shot-accuracy average. Neither number measures whether the outlier mechanism was removed or merely relocated.

## 4. What Is Known

- **Magnitude and sparsity.** LLaMA-2-7B: at layer 2, two coordinates (dims 1415 and 2533) at the BOS token and a few delimiters reach $\approx 2.5\times10^3$ against a median activation magnitude of $\approx 0.2$ — roughly $10^4\times$, on fewer than ten of millions of entries (Sun et al., 2024).
- **Causal necessity.** Zeroing those entries collapses the model (perplexity rises by orders of magnitude); replacing them with their *mean* value leaves perplexity essentially unchanged. This is the strongest evidence for the constant-bias reading.
- **Scale phase transition.** In OPT, outlier features are confined to some layers below ~2.7B and appear in all layers at 6.7B, coinciding with the failure of naive INT8 (Dettmers et al., 2022).
- **Pre-LLM precedent.** Outlier dimensions that disrupt performance when clipped exist in BERT-base (Kovaleva et al., Findings of ACL 2021); a few "rogue dimensions" dominate cosine similarity in GPT-2 (Timkey & van Schijndel, EMNLP 2021); they correlate with token frequency (Puccetti et al., Findings of EMNLP 2022).
- **Architectural fixes work small.** Clipped softmax and gated attention remove outliers in BERT-base and OPT-125M/350M, cutting max activation infinity-norm by an order of magnitude and making per-tensor W8A8 near-lossless (Bondarenko et al., NeurIPS 2023).
- **Sinks are trainable away in part.** Gu et al. report that normalization-free sigmoid attention avoids sink formation up to 1B parameters.

## 5. What Is Not Known

- **Theoretically open.** No proof that softmax attention with a residual stream *must* produce unbounded-norm bias features at the loss optimum, nor a lower bound on the dynamic range $R^{(\ell)}$ required to express the functions these models compute. The "no-op attention needs somewhere to point" argument is a story, not a theorem.
- **Empirically open.** Whether an outlier-free pretraining recipe (explicit KV bias, gated attention, or sigmoid attention) matches a matched-compute baseline at $\ge 7$B parameters and $\ge 1$T tokens, on loss *and* 128k-context retrieval. Every clean removal result is $\le 1$B. Runnable today; the compute is the barrier.
- **Empirically open.** Whether rotation methods remove outliers or only hide them: after a Hadamard rotation the coordinate-wise outlier disappears by construction, but the *rank-1 direction* may persist. Nobody has reported the rotation-invariant statistic (top singular value of the activation matrix over the norm) before and after.
- **Methodologically blocked.** There is no basis-free, threshold-free definition of "outlier feature". $\kappa=1000$ vs $\tau=6.0$ are unrelated conventions, so cross-paper counts are not comparable and "we reduced outliers by X%" is not a portable claim.

## 6. Why It Is Hard

- **Non-identifiability of the basis.** Outlier-ness is a coordinate-basis property. Because RMSNorm-then-linear admits exact rotational reparameterizations (the invariance QuaRot exploits), any statement of the form "this model has 4 outlier channels" is a statement about a gauge choice, not about the function. The physically meaningful object — the low-rank, high-norm subspace and the attention mass it absorbs — is measured by almost nobody.
- **Confounded measurement.** Massive activation magnitude, sink attention mass, and quantization error move together across checkpoints. Interventions change all three at once, so "removing outliers improved INT4" does not establish which mechanism mattered.
- **Compute cost of the decisive control.** The only convincing test is a matched-token, matched-data pretraining pair at 7B+. That is $\sim 10^{23}$ FLOPs per arm — outside academic budgets and unreported by labs that can afford it.
- **Evaluation mismatch.** Post-quantization perplexity is the standard proxy, and it is dominated by frequent tokens. Sink-dependent behaviour shows up in long-context retrieval and streaming, which the headline number does not measure.

## 7. Current Research (as of 2026)

- **Rotation and incoherence processing** (ETH/IST Austria for QuaRot; Meta for SpinQuant) is the practical mainline; work continues on making learned rotations cheap and KV-cache-compatible.
- **Sink-aware architectures.** Dedicated sink/register tokens in pretraining (following StreamingLLM and the vision-side "registers" result of Darcet et al., ICLR 2024) are now standard in several open recipes; several 2025 open-weight releases add explicit attention biases or per-head learned sink logits. *(frontier — verify which releases.)*
- **Mechanistic accounts.** Barbero et al. (2025) on sinks as over-mixing control; An et al. (ICLR 2025) on systematic outliers unifying activation, weight and attention outliers; Cancedda (ACL 2024) on spectral filters and "dark signals" feeding sinks.
- **Optimizer-side hypotheses.** Adam's per-coordinate scaling as the privileged-basis source (Anthropic, 2023) is being revisited in the context of Muon and other non-diagonal optimizers, which change the basis privilege story. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Do rotation methods eliminate the outlier mechanism or relocate it?

**Scale.** LLaMA-2-7B and LLaMA-2-70B, inference only, 100 sequences × 4096 tokens of WikiText-2. Cost: a few GPU-hours. No training.

**Arms.** (A) FP16 baseline; (B) QuaRot Hadamard-rotated weights, FP16 activations (rotation only, no quantization — this is the control that isolates the basis change from the bit-width change); (C) QuaRot W4A4.

**Measured quantity.** For each layer, the rotation-invariant concentration

$$\rho^{(\ell)} = \frac{\sigma_1\!\left(h^{(\ell)} - \bar h^{(\ell)}\right)}{\|h^{(\ell)} - \bar h^{(\ell)}\|_F},$$

the fraction of centred activation energy in the top singular direction, alongside the basis-dependent $R^{(\ell)}$.

**Deciding number.** $\rho^{(\ell)}$ at the peak layer, arm B versus arm A. If $\rho$ drops below $0.3$ (say, from $\approx 0.9$), rotation genuinely destroys the rank-1 sink direction and the phenomenon is basis-gauge. If $\rho$ is unchanged within $\pm 0.05$ while $R^{(\ell)}$ falls by $10\times$, then rotation is a coordinate trick: the mechanism is intact and every "outlier removed" claim built on per-channel statistics is measuring the gauge, not the model. That single comparison also supplies the missing basis-free metric for §5.

## 9. Key References

- **[Foundational]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[Foundational]** Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. *Massive Activations in Large Language Models.* COLM, 2024. — arXiv:2402.17762
- **[Foundational]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[SOTA]** Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456
- **[SOTA]** Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort. *SpinQuant: LLM Quantization with Learned Rotations.* ICLR, 2025.
- **[SOTA]** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS, 2023. — arXiv:2306.12929
- **[Method]** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML, 2023. — arXiv:2211.10438
- **[Analysis]** Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, Min Lin. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR, 2025.
- **[Analysis]** Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu. *Why do LLMs attend to the first token?* 2025.
- **[Analysis]** Nicola Cancedda. *Spectral Filters, Dark Signals, and Attention Sinks.* ACL, 2024.
- **[Analysis]** Nelson Elhage et al. *Privileged Bases in the Transformer Residual Stream.* Transformer Circuits Thread, Anthropic, 2023.
- **[Precedent]** Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, Anna Rumshisky. *BERT Busters: Outlier Dimensions that Disrupt Transformers.* Findings of ACL, 2021.
- **[Precedent]** William Timkey, Marten van Schijndel. *All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality.* EMNLP, 2021.
- **[Related]** Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. *Vision Transformers Need Registers.* ICLR, 2024. — arXiv:2309.16588

## 10. Worked Example

Take LLaMA-2-7B, layer 2 residual stream, one 4096-token WikiText-2 sequence.

- Median $|h_{t,i}| \approx 0.2$; RMS over the tensor $\approx 0.9$ excluding outliers.
- Four entries — dims 1415 and 2533 at the BOS token and at the first newline — sit near $2.5\times 10^{3}$.

Per-tensor INT8 with symmetric scaling uses step $\Delta = \max|h| / 127 = 2500/127 \approx 19.7$. Every ordinary activation of magnitude $0.2$ quantizes to bin $0$. The signal-to-quantization-noise ratio for the bulk is

$$\text{SQNR} \approx 20\log_{10}\!\frac{0.9}{19.7/\sqrt{12}} \approx -16\ \text{dB},$$

i.e. the noise is 6× the signal. Four numbers out of $4096 \times 4096 \approx 1.7\times 10^{7}$ — $2\times 10^{-7}$ of the tensor — destroy the layer. That is the entire practical motivation for LLM.int8(), SmoothQuant and QuaRot.

Now the obstruction. Apply a $4096\times4096$ Hadamard rotation $Q$: $\tilde h = hQ$ spreads each massive entry over all coordinates, so $\max|\tilde h| \approx 2500/\sqrt{4096} \approx 39$, $\Delta \approx 0.31$, and SQNR jumps to roughly $+20$ dB. By the per-channel definition the outliers are gone. But $\sigma_1(h - \bar h)$ is unchanged — Hadamard is orthogonal, so every singular value is preserved exactly. The BOS token's activation vector still has norm $\sim 2500$ against a typical token norm of $\sim 60$, and the heads still dump their softmax mass on it. The measurement said "fixed"; the invariant says "identical". This is exactly why §8 measures $\rho$ and not $R$ — and why, seven years into the phenomenon, we still cannot say whether outliers are a fact about transformers or a fact about the coordinates we happen to write them in.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*