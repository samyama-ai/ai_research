---
id: 12-quantization-compression/activation-outlier-emergence-cause
title: "Outlier Emergence Cause in Transformer Activations"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Outlier Emergence Cause in Transformer Activations

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/activation-outlier-emergence-cause` · **Status:** open

## 1. Problem Statement

Trained transformers develop a small number of **activation outliers**: coordinates of the residual stream, or (token, coordinate) pairs, whose magnitude exceeds the median coordinate by two to four orders of magnitude. They break per-tensor and per-token affine quantization, which is why W8A8 and W4A4 post-training quantization fails without countermeasures. The question is **why they exist**.

Three variants, with different difficulty:

- **Measurement.** Given a model $f_\theta$ and a data distribution $\mathcal D$, produce a definition of "outlier" that is scale-free, layer-comparable, and predicts quantization damage. Currently each paper uses a different threshold and the definitions disagree about which models have outliers.
- **Method.** Produce an architecture, initialization, optimizer, or training recipe such that models trained with it reach matched loss on matched tokens **and** have bounded activation kurtosis, so that naive INT8/INT4 activation quantization is lossless. Partially achieved at small scale; unverified at frontier scale.
- **Theory.** Decide whether outliers are (a) *necessary* — any transformer that implements the required "no-op attention" / sink function under softmax must place large mass on privileged directions; (b) *optimizer artifacts* — a consequence of Adam's per-coordinate normalization and $\varepsilon$, removable by changing the optimizer; or (c) *data artifacts* — driven by token frequency and the presence of a fixed first token. Solving means a proof or a decisive intervention experiment that separates (a), (b), (c).

Solved = the theory variant is settled, and the method variant delivers an outlier-free 70B-class model at parity loss.

## 2. Formal Setting

Let the model have depth $L$, width $d$, and residual stream $h^{(\ell)}_t \in \mathbb{R}^d$ for layer $\ell$ and token position $t$ in sequence $x \sim \mathcal{D}$. Measurement is over a fixed calibration set of $N$ sequences of length $T$ (typical practice: $N=128$, $T=2048$, from the pretraining distribution).

**Per-coordinate scale.** For coordinate $j$,
$$m_j^{(\ell)} = \operatorname{median}_{t,x} |h^{(\ell)}_{t,j}|, \qquad M_j^{(\ell)} = \max_{t,x} |h^{(\ell)}_{t,j}|.$$

**Outlier feature (LLM.int8 convention).** Coordinate $j$ is a *systematic outlier feature* if $M_j^{(\ell)} \ge 6.0$ in absolute magnitude, in at least $25\%$ of layers and at least $6\%$ of token positions. This threshold is absolute, not scale-free — it is comparable only across models with the same LayerNorm placement and embedding scale.

**Massive activation (Sun et al. convention).** Entry $(t,j)$ is massive if $|h^{(\ell)}_{t,j}| \ge 100$ and $|h^{(\ell)}_{t,j}| \ge 1000 \cdot \operatorname{median}_{t',j'} |h^{(\ell)}_{t',j'}|$.

**Scale-free surrogate.** Excess kurtosis of the flattened activation tensor,
$$\kappa^{(\ell)} = \frac{\mathbb{E}[(h^{(\ell)} - \mu)^4]}{\sigma^4}, \qquad \text{and} \qquad R^{(\ell)} = \frac{\|h^{(\ell)}\|_\infty}{\|h^{(\ell)}\|_2 / \sqrt{d}} .$$
$R^{(\ell)} \in [1, \sqrt{d}]$; Gaussian activations give $R \approx \sqrt{2\ln(dT)}$.

**What outliers cost.** For symmetric per-tensor INT-$b$ quantization with step $\Delta = 2\|h\|_\infty / (2^b - 1)$, the mean squared error under a uniform-error model is $\Delta^2/12$, so signal-to-quantization-noise is
$$\mathrm{SQNR} = \frac{3(2^b-1)^2}{R^2} \cdot \frac{1}{d}\cdot\frac{\|h\|_2^2}{\sigma^2}\Big/\!\ldots \;\propto\; R^{-2}.$$
The operational consequence: **every factor of 2 in $R$ costs one bit of effective activation precision.** This is the quantity the problem is really about; $\kappa$ and $R$ are the measurable proxies.

**Assumptions, and which are violated.**
1. *Outliers are stationary across data.* Violated: massive activations concentrate on BOS, delimiters (`.`, `\n`), and high-frequency tokens, so $R$ depends on the calibration corpus.
2. *Outliers live in a fixed low-dimensional subspace.* Approximately true within a model, false across seeds — the identity of the outlier coordinates is seed-dependent even when the count is not.
3. *Uniform quantization error.* Violated exactly in the outlier regime, which is why $\mathrm{SQNR}$ estimates understate damage.
4. *Residual stream basis is privileged.* True only because of per-coordinate operations (LayerNorm scale, Adam, GLU elementwise products); rotation-invariance of the rest of the network is what QuaRot exploits.

## 3. State of the Art

**Established (reproduced, ablated).**
- *Mitigation by rescaling.* SmoothQuant (Xiao et al., ICML 2023) migrates activation range into weights via a per-channel diagonal $s$; W8A8 at near-FP16 accuracy on OPT-175B. Ablated over $\alpha$.
- *Mitigation by rotation.* QuaRot (Ashkboos et al., NeurIPS 2024) applies Hadamard rotations that provably preserve the function while destroying basis-alignment of outliers; Llama-2-70B W4A4 within $0.47$ WikiText-2 perplexity of FP16. SpinQuant (Liu et al., 2024) learns the rotation. These *remove the measurement basis*, not the phenomenon: $\|h\|_2$ is unchanged, only $R$ falls.
- *Prevention by attention modification.* Bondarenko, Del Chiaro, Nagel — *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing* (NeurIPS 2023). Clipped softmax and gated attention drive max infinity-norm from $O(10^3)$ to $O(10^1)$ at matched or better perplexity for BERT-base and OPT up to 1.3B. This is the strongest causal evidence for the "no-op attention" account.

**Claimed but unablated / benchmark-only.**
- The *Adam-$\varepsilon$* account of privileged bases (Elhage et al., Anthropic, 2023): a report, not a controlled study across optimizers at scale; the authors themselves state the mechanism is not established.
- The *token-frequency* account (Puccetti et al., Findings of EMNLP 2022) shows correlation between outlier magnitude and token frequency in BERT-scale encoders; it has not been tested as an intervention (frequency-flattened corpus, matched loss).
- Vendor claims that FP8 or MX formats "solve" outliers exist mainly as end-task benchmark numbers, without the $R^{(\ell)}$ profiles that would show whether the outliers went away or were merely absorbed by finer-grained scaling.

## 4. What Is Known

- **Emergence is abrupt in scale.** Dettmers et al. (LLM.int8(), NeurIPS 2022) measured OPT/BLOOM from 125M to 175B: the number of layers containing systematic outliers jumps from ~65% to 100% between 6.0B and 6.7B parameters, and the count of affected coordinates rises from ~6 to ~150 by 13B. The transition tracks **perplexity**, not parameter count, when models of different families are aligned.
- **Magnitudes.** Sun et al. (*Massive Activations in Large Language Models*, COLM 2024): LLaMA-2-7B has exactly 4 massive activations, at coordinates 1415 and 2533, with magnitude ~2500 versus a median absolute activation of ~0.2 — a ratio of $10^4$. They appear at layer 2, persist to layer 30, and vanish at the last layer.
- **They are functionally load-bearing.** Zeroing the massive activations raises LLaMA-2-7B WikiText perplexity from ~5.5 to $>10^3$; replacing them with their mean value is nearly harmless. They act as fixed biases, not as data-dependent features.
- **They implement attention sinks.** The same tokens carry the sink mass in StreamingLLM (Xiao et al., ICLR 2024); removing the first four tokens' KV collapses streaming perplexity. Gu et al. (ICLR 2025) show the sink appears within the first few thousand optimizer steps and is suppressed by replacing softmax with normalization-free sigmoid attention, at 1B scale.
- **Vision transformers show the same thing.** Darcet et al. (*Vision Transformers Need Registers*, ICLR 2024): high-norm artifact tokens in low-information patches, removable by adding register tokens — a cross-modal replication of the "model needs a place to put nothing" account.
- **Small-scale prevention works.** Beyond Bondarenko et al., encoder results (Kovaleva et al., Findings ACL 2021; Timkey & van Schijndel, EMNLP 2021) show a handful of rogue dimensions dominate cosine similarity in BERT/GPT-2 and can be clipped with negligible task loss.

## 5. What Is Not Known

- **Theoretically open.** Whether softmax attention *necessarily* produces unbounded-norm sink states. No theorem gives a lower bound on $\|h\|_\infty$ for any network computing an approximately-identity attention operation under a normalized attention map. Conversely, no proof that a bounded-$R$ transformer can match the loss of an unbounded one at fixed parameter count.
- **Empirically open.** The optimizer intervention at scale: train a $\ge 7$B model with a non-per-coordinate-normalizing optimizer (SGD-momentum, Muon, or Adam with $\varepsilon = 0$ plus gradient clipping) and measure $R^{(\ell)}$ at matched loss. Runnable — roughly $10^{22}$–$10^{23}$ FLOPs — but not reported with matched controls.
- **Empirically open.** Whether clipped-softmax / gated attention / registers survive to 7B–70B and 15T tokens. Every prevention result is at $\le$ 1.3B.
- **Methodologically blocked.** "Outlier" has no scale-free definition. Rotation-based methods reduce $R$ to near-Gaussian without changing $\|h\|_2$, so under the LLM.int8 threshold QuaRot models "have no outliers" while computing the identical function. Until the field agrees whether the object of study is a basis-aligned coordinate or a basis-free heavy-tailed direction, cross-paper claims are not comparable.

## 6. Why It Is Hard

**Non-identifiability under confounded interventions.** Every proposed cause — softmax normalization, Adam's per-coordinate step, LayerNorm's learned per-channel gain, the fixed BOS token, token frequency — is present simultaneously in every production run, and each intervention perturbs the others. Removing Adam changes the loss trajectory, so a lower $R$ may reflect an undertrained model rather than an outlier-free one; matching loss then requires re-tuning the learning-rate schedule, which itself changes $R$. There is **no ground-truth counterfactual**: no pair of models identical except in the hypothesized cause exists at frontier scale, because each such pair costs a full pretraining run.

Second obstruction: **the evaluation does not measure the thing it names.** Papers report downstream accuracy after quantization, which conflates (i) how heavy-tailed the activations are, (ii) how well the quantizer handles heavy tails, and (iii) how robust the task is. A method that improves (ii) is scored as if it explained (i).

## 7. Current Research (as of 2026)

- **Rotation/incoherence processing** as the default production answer: QuaRot, SpinQuant, and their successors in llama.cpp / TensorRT-LLM kernels. Industrial direction; treats the cause as unnecessary to know.
- **Attention-sink mechanism** — Gu et al. and follow-ups on sigmoid/normalization-free attention, plus register tokens migrating from ViTs to LLMs. *(frontier — verify: several 2025–2026 open-weight releases reportedly train with learned sink logits; check the model card before citing.)*
- **Optimizer geometry** — Muon and other non-diagonal preconditioners are being adopted for large runs; their effect on activation kurtosis is an available free measurement that few reports include. *(frontier — verify)*
- **Outlier-aware training for FP4/FP8** — NVIDIA, Microsoft, and academic groups studying whether microscaling block formats make prevention moot.

## 8. Concrete Next Experiment

**Question.** Is the massive-activation/sink phenomenon caused by softmax normalization, by Adam's per-coordinate normalization, or by the fixed prefix token?

**Scale.** Five 1.4B-parameter Llama-architecture models, 100B tokens each of a fixed public corpus (e.g. FineWeb-Edu), identical data order, identical seed, ~$1.7\times10^{21}$ FLOPs each — about 2k A100-hours per arm, 10k total.

**Arms.**
- **A (control).** Standard: softmax attention, AdamW ($\varepsilon=10^{-8}$), BOS prepended.
- **B.** Softmax replaced by sigmoid attention with $1/n$ normalization removed (Gu et al. variant).
- **C.** AdamW replaced by Muon on all 2-D parameters, LR retuned to match arm A's loss within 0.5%.
- **D.** No BOS; documents packed without any fixed prefix; 4 learned register tokens instead.
- **E.** Frequency-flattened corpus: subsample the top-100 tokens so no token exceeds 0.5% of the stream, matched total tokens.

**Deciding number.** $R^{(\ell)} = \|h^{(\ell)}\|_\infty / (\|h^{(\ell)}\|_2/\sqrt{d})$ at the layer of maximum kurtosis, on a held-out 128×2048-token calibration set, **conditional on matched validation loss within 1%**. Arm A is expected at $R \approx 40$–$120$. An arm is judged to have removed the cause if $R \le 8$ (within $2\times$ of the Gaussian value $\sqrt{2\ln(dT)} \approx 5.5$ for $d=2048$, $T=2048$) at matched loss. Secondary readout: W8A8 per-tensor PTQ perplexity gap, which should fall below 0.1 for any arm reaching $R \le 8$.

Any arm that reaches $R\le 8$ at parity loss falsifies the necessity claim for its factor's competitors; all arms failing supports the "necessary" theory branch.

## 9. Key References

- **[Foundational]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[Foundational]** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. *Understanding and Overcoming the Challenges of Efficient Transformer Quantization.* EMNLP, 2021. — arXiv:2109.12948
- **[SOTA / mechanism]** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS, 2023. — arXiv:2306.12929
- **[SOTA / measurement]** Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. *Massive Activations in Large Language Models.* COLM, 2024. — arXiv:2402.17762
- **[SOTA / mitigation]** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML, 2023. — arXiv:2211.10438
- **[SOTA / mitigation]** Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456
- **[Mechanism]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Mechanism]** Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, Min Lin. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR, 2025. — arXiv:2410.10781
- **[Cross-modal]** Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. *Vision Transformers Need Registers.* ICLR, 2024. — arXiv:2309.16588
- **[Earlier evidence]** Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, Anna Rumshisky. *BERT Busters: Outlier Dimensions that Disrupt Transformers.* Findings of ACL, 2021.
- **[Earlier evidence]** William Timkey, Marten van Schijndel. *All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality.* EMNLP, 2021.
- **[Data account]** Giovanni Puccetti, Anna Rogers, Aleksandr Drozd, Felice Dell'Orletta. *Outlier Dimensions that Disrupt Transformers Are Driven by Frequency.* Findings of EMNLP, 2022.
- **[Mitigation]** Xiuying Wei et al. *Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models.* NeurIPS, 2022. — arXiv:2209.13325
- **[Note]** Nelson Elhage et al. *Privileged Bases in the Transformer Residual Stream.* Anthropic Transformer Circuits Thread, 2023.

## 10. Worked Example

Take LLaMA-2-7B, $d = 4096$, layer 20 residual stream, on a 2048-token WikiText sequence.

Measured (Sun et al.): median $|h| \approx 0.2$; the four massive entries at coordinates 1415 and 2533 on the BOS and first-`.` tokens reach $|h| \approx 2500$.

Estimate $\|h\|_2$ for one such token: the 4094 non-massive coordinates contribute roughly $\sqrt{4094}\times 0.3 \approx 19$ (using ~0.3 RMS); the two massive coordinates contribute $\approx \sqrt{2}\times 2500 \approx 3536$. So $\|h\|_2 \approx 3536$, $\|h\|_\infty = 2500$, and
$$R = \frac{2500}{3536/\sqrt{4096}} = \frac{2500}{55.3} \approx 45.2 .$$
Gaussian reference: $\sqrt{2\ln(4096\cdot 2048)} \approx 4.7$. The model is $\approx 10\times$ over.

Per-tensor INT8 on this tensor: $\Delta = 2\cdot 2500/255 = 19.6$. The median coordinate, $|h| = 0.2$, quantizes to **zero**. Effectively all non-outlier information in that token's activation is destroyed — 8 bits buys $\log_2(255/45.2) \approx 2.5$ usable bits. Per-token scaling rescues the ordinary tokens but not the sink tokens, and the sink tokens are the ones whose removal costs $>10^3$ perplexity.

**Where the obstruction becomes visible.** Apply a random Hadamard rotation $Q$ ($h \mapsto Qh$, exactly function-preserving when folded into adjacent weights). $\|Qh\|_2 = 3536$ is unchanged, but the mass spreads: $\|Qh\|_\infty \approx 3536\cdot\sqrt{2\ln(4096)/4096} \approx 3536 \times 0.0645 \approx 228$, giving $R \approx 4.1$. Under the LLM.int8 threshold this model now has **no outliers at all**, and INT8 per-tensor error drops by $\approx 11\times$ — while every activation vector, every logit, and every prediction is bit-for-bit the same function. The heavy-tailed *direction* is untouched; only its coordinate expression moved.

This is the catalog's core point: the field's outlier statistic is a property of the basis, the causal question is about the function, and no current experiment separates them. Any explanation of "why outliers emerge" that a rotation can erase is an explanation of the basis, not of the phenomenon.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*