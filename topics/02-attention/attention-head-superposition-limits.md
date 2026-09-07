---
id: 02-attention/attention-head-superposition-limits
title: "Superposition Limits in Attention Head Feature Coding"
topic: 02-attention
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Superposition Limits in Attention Head Feature Coding

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-head-superposition-limits` · **Status:** open

## 1. Problem Statement

An attention head writes into the residual stream through a rank-$d_h$ output map ($d_h = 64$ in GPT-2 small and Llama-2-7B). Empirically, a single head appears to participate in far more than 64 distinguishable behaviours. The problem: **how many features can one head encode before interference degrades loss, and what is the exchange rate between head count $H$, head width $d_h$, and encodable features?**

Three variants, with different difficulty:

- **Measurement.** Given a trained model and a head $h$, produce a number $\hat m_h$ — the count of features that head reads or writes — with a stated identification criterion and an error bar. Currently there is no agreed estimator.
- **Method.** Train or edit a model so that a chosen head is monosemantic (one feature) at fixed loss, or show that the loss cost of doing so is bounded.
- **Theory.** Prove a capacity theorem: for a loss $\mathcal{L}$, feature sparsity $s$, and per-head width $d_h$, bound the number of features recoverable from the head's output with interference below $\epsilon$.

Solving it means: a capacity law $m^*(d_h, s, \epsilon)$ that predicts, within a factor of 2, the measured feature count of real heads at $\ge 1$B parameters, plus the loss penalty of forcing $m \le d_h$.

## 2. Formal Setting

Residual stream $x_t \in \mathbb{R}^d$. Head $h$ in layer $\ell$ has $W_Q^h, W_K^h, W_V^h \in \mathbb{R}^{d \times d_h}$ and $W_O^h \in \mathbb{R}^{d_h \times d}$, giving the two circuits of Elhage et al. (2021): $W_{QK}^h = W_Q^h W_K^{h\top} \in \mathbb{R}^{d\times d}$ and $W_{OV}^h = W_V^h W_O^h \in \mathbb{R}^{d \times d}$, both of rank $\le d_h$.

**Feature model.** Assume a dictionary $\{f_i\}_{i=1}^m$, $f_i \in \mathbb{R}^d$, unit norm, with activations $a_i(t) \ge 0$ that are $s$-sparse: $\Pr[a_i > 0] = s$, typically $s \in [10^{-3}, 10^{-1}]$. The head's output at position $t$ is
$$o_t^h = \sum_{t'} \alpha_{tt'} W_{OV}^{h\top} x_{t'}, \qquad \alpha = \mathrm{softmax}\!\left(x_t^\top W_{QK}^h X^\top/\sqrt{d_h}\right).$$

**Measured quantities.**

- *Head feature count* $\hat m_h$: train a sparse autoencoder (SAE) on the per-head pre-$W_O$ activations $z_t^h \in \mathbb{R}^{d_h}$, with $\|\cdot\|_1$ penalty $\lambda$; $\hat m_h$ is the number of dictionary atoms alive (firing on $>10^{-6}$ of tokens) at a fixed reconstruction fidelity, e.g. **loss recovered $\ge 0.90$** where loss-recovered $= 1 - (\mathcal{L}_{\text{splice}} - \mathcal{L}_{\text{orig}})/(\mathcal{L}_{\text{ablate}} - \mathcal{L}_{\text{orig}})$.
- *Interference* between recovered atoms $\hat f_i$: $\epsilon = \max_{i\neq j} |\langle \hat f_i, \hat f_j\rangle|$; the Johnson–Lindenstrauss packing bound gives $m \le \exp(O(\epsilon^2 d_h))$, so $d_h = 64$ admits $\sim 10^3$–$10^4$ atoms at $\epsilon \approx 0.3$ — the whole question is whether the network uses that room.
- *Capacity* (Scherlis et al. 2022): $C_i = \langle \hat W_i, \hat W_i\rangle^2 / \sum_j \langle \hat W_i, \hat W_j\rangle^2$ for embedding direction $W_i$, with $\sum_i C_i \le d_h$. Monosemantic $\Rightarrow C_i = 1$; a fully superposed head has $C_i \ll 1$ for all $i$.
- *Loss cost of decompression*: $\Delta\mathcal{L} = \mathcal{L}(\text{model with } d_h \to k d_h, H \to H/k) - \mathcal{L}(\text{baseline})$ at matched parameters and tokens.

**Assumptions, and which are violated.** (i) Linear feature superposition — partially violated: attention outputs contain position- and token-conditional structure that is not a fixed linear dictionary. (ii) Independent Bernoulli sparsity — violated; real features are strongly correlated and hierarchically organised, and correlation is exactly what changes the optimal packing geometry in Elhage et al. (2022). (iii) Head-local coding — violated: heads compose, so the same feature is spread across heads (Wang et al. 2023, IOI). (iv) The SAE dictionary is identified — not established; SAE atom counts scale with dictionary size and $\lambda$, which is the central measurement problem in §6.

## 3. State of the Art

**Theory SOTA (established).** Elhage et al., *Toy Models of Superposition* (Transformer Circuits, 2022): in a ReLU autoencoder with $d$ dims and $m \gg d$ sparse features, the optimal solution stores $\Theta(1/s)$-scaled numbers of features in geometric arrangements (antipodal pairs, triangles, pentagons, tetrahedra) as sparsity rises; phase transitions in $s$ are exact in the toy model. Scherlis et al., *Polysemanticity and Capacity in Neural Networks* (2022), give the capacity constraint $\sum_i C_i \le d$ and show capacity allocation is a constrained optimisation with discrete phase changes. Neither result is proved for attention: both concern a fixed linear-plus-ReLU write, not a softmax-gated, rank-constrained bilinear read.

**Empirical SOTA (established).** Attention-output SAEs work: Kissane et al., *Interpreting Attention Layer Outputs with Sparse Autoencoders* (ICML 2024 MI workshop, arXiv:2406.17759) train SAEs on the concatenated $z$ of GPT-2 small and recover interpretable features attributable to individual heads; they report heads that are *polysemantic* at the head level — a single head carries several unrelated feature families. Gemma Scope (Lieberum et al., 2024, arXiv:2408.05147) released attention-output SAEs for Gemma-2 2B and 9B at multiple widths, making the width-sweep runnable off the shelf.

**Claimed but unablated.** That head-level polysemanticity is *superposition* (capacity-limited packing) rather than *feature composition* (one computation used in many contexts) — this is asserted in blog-form write-ups and not separated by any published intervention. That SAE atom counts on head activations estimate a true $m_h$: no paper establishes that the count is stable under dictionary size, and the known scaling is that alive-atom count grows roughly with dictionary width up to the compute budget. That is a benchmark number, not a capacity measurement.

## 4. What Is Known

- **Heads are redundant at inference.** Michel et al. (NeurIPS 2019, arXiv:1905.10650): in WMT Transformer and BERT, most layers can be reduced to a single head at test time with small BLEU/accuracy loss; ablating 20% of heads by importance score costs little. Voita et al. (ACL 2019, arXiv:1905.09418): 38 of 48 encoder heads pruned in an EN-RU Transformer with a 0.15 BLEU drop. Scale: 6-layer, 8-head encoder-decoders and BERT-base.
- **Heads implement identifiable algorithms.** Olsson et al. (2022) locate induction heads and tie them to a loss bump at $\sim 2$–$4 \times 10^9$ tokens across models from 2M to 13B parameters. Wang et al. (ICLR 2023, arXiv:2211.00593) map the IOI circuit in GPT-2 small to 26 heads in 7 classes — i.e. 26 heads for *one* task, evidence that heads are shared rather than dedicated.
- **MLP-side superposition is real and measurable.** Gurnee et al. (TMLR 2023, arXiv:2305.01610) show sparse probing recovers features from $k$-sparse neuron sets in Pythia models, with many features distributed across neurons. Bricken et al. (2023) and Templeton et al. (2024, Claude 3 Sonnet) scale SAEs to $10^7$ features on residual streams with interpretable atoms.
- **Head width matters less than head count, up to a point.** The standard result that $d_h = d/H$ with $H$ scaled up gives near-flat loss holds in published architecture sweeps; but no published sweep isolates $d_h$ at fixed $H$ and fixed parameters over a decade of scale.

## 5. What Is Not Known

- **Theoretically open.** No capacity theorem for the softmax-gated head. The rank-$d_h$ $W_{OV}$ bounds the *output* subspace, but the number of *behaviours* is not bounded by rank, since $\alpha$ is input-dependent — the head is a data-dependent linear map, and the right capacity object (something like the covering number of $\{W_{OV}^\top \Pi_\alpha\}$) is undefined in the literature.
- **Methodologically blocked.** $\hat m_h$ has no dictionary-size-independent definition. Atom counts rise with SAE width and fall with $\lambda$; no published criterion fixes the operating point from first principles.
- **Empirically open.** The $d_h$ vs. $H$ exchange rate at matched compute: nobody has trained the $\{(H, d_h)\}$ grid at $\ge 1$B parameters and measured whether widening heads reduces measured polysemanticity at equal loss. This is runnable today for a few hundred GPU-days.

## 6. Why It Is Hard

**Non-identifiability of the feature count.** The measurement is a free parameter dressed as an observable: an SAE with $2^{14}$ atoms on a 64-dim head activation will report $\sim 10^3$ alive atoms, and one with $2^{16}$ will report more, at similar loss-recovered. Without ground truth there is no calibration point, so "this head encodes 900 features" is a statement about the probe.

**Confounded measurement, second source.** Polysemanticity at the head level has two causes that current interventions do not separate: capacity pressure (superposition proper) and context-conditional reuse of one operation (an induction head fires on any repeated bigram — that is one algorithm, not $10^3$ features). Any $\hat m_h$ counts both.

**Compute.** The decisive experiment is an architecture sweep, not a probe. Training the $(H, d_h)$ grid at 1B scale with matched tokens is 6–12 runs, each $\sim 10^{21}$ FLOPs, plus SAEs on every head of every run.

## 7. Current Research (as of 2026)

- Attention-output and head-resolved SAEs: Anthropic interpretability, Google DeepMind (Gemma Scope), and the EleutherAI/independent SAE community. Direction: per-head dictionaries with shared decoders across heads.
- Computation-in-superposition theory: Vaintrob, Mendel and Hänni, *Mathematical Models of Computation in Superposition* (2024, arXiv:2408.05451) — bounds on how many boolean operations a layer can carry out in superposition; not yet specialised to attention. *(frontier — verify extension to QK circuits)*
- Attention-head taxonomies: Zheng et al., *Attention Heads of Large Language Models: A Survey* (2024, arXiv:2409.03752) collates head roles across models — descriptive, not capacity-theoretic.
- Transcoder/cross-layer approaches that replace per-component dictionaries with circuit-level ones, which would sidestep the head-local coding assumption. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does head width $d_h$ bind feature coding, or is head polysemanticity independent of width?

**Scale.** Train 4 models at 1.3B parameters, 100B tokens (Chinchilla-ish), identical data order, identical $d = 2048$ and total attention parameters: $(H, d_h) \in \{(32, 64), (16, 128), (8, 256), (64, 32)\}$.

**Control arm.** $(H, d_h) = (32, 64)$ — the standard configuration — plus a *shuffled-feature control*: for each trained model, a re-run with the same architecture on token-level-shuffled text, which destroys long-range features while preserving unigram statistics. The control fixes what atom counts look like when there is little to encode.

**Measurement.** Train per-head SAEs on $z^h$ at three dictionary widths ($8 d_h$, $32 d_h$, $128 d_h$) with $\lambda$ tuned to loss-recovered $= 0.90 \pm 0.01$. Report $\hat m_h$ per head and the mean capacity $\bar C$.

**The deciding number.** The slope
$$\beta = \frac{d \log \bar{\hat m}}{d \log d_h}$$
measured at fixed loss-recovered and fixed dictionary-to-$d_h$ ratio. **$\beta \approx 1$** means feature count tracks width — heads are capacity-bound, superposition is the binding constraint, and widening heads decompresses. **$\beta \approx 0$** means the count is set by the task, not the head, and head polysemanticity is reuse, not packing. Anything in between, with error bars from the three dictionary widths, quantifies the exchange rate. Secondary number: $\Delta\mathcal{L}$ across the grid — if all four configurations land within 0.01 nats, width is free and the capacity question is decoupled from performance.

## 9. Key References

- **[Foundational]** Elhage, Nanda, Olsson, et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021.
- **[Foundational]** Elhage, Hume, Olsson, et al. *Toy Models of Superposition.* Transformer Circuits Thread, 2022. — arXiv:2209.10652
- **[Foundational]** Olsson, Elhage, Nanda, et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, 2022. — arXiv:2209.11895
- **[Theory]** Scherlis, Sachan, Jermyn, Benton, Shlegeris. *Polysemanticity and Capacity in Neural Networks.* 2022. — arXiv:2210.01892
- **[Theory]** Vaintrob, Mendel, Hänni. *Mathematical Models of Computation in Superposition.* 2024. — arXiv:2408.05451
- **[SOTA]** Kissane, Krzyzanowski, Bloom, Conmy, Nanda. *Interpreting Attention Layer Outputs with Sparse Autoencoders.* ICML 2024 Mechanistic Interpretability Workshop. — arXiv:2406.17759
- **[SOTA]** Lieberum, Rajamanoharan, Conmy, et al. *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.* BlackboxNLP 2024. — arXiv:2408.05147
- **[SOTA]** Templeton, Conerly, Marcus, et al. *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Transformer Circuits Thread, 2024.
- **[Empirical]** Michel, Levy, Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS 2019. — arXiv:1905.10650
- **[Empirical]** Voita, Talbot, Moiseev, Sennrich, Titov. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL 2019. — arXiv:1905.09418
- **[Empirical]** Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* ICLR 2023. — arXiv:2211.00593
- **[Empirical]** Gurnee, Nanda, Pauly, Harvey, Troitskii, Bertsimas. *Finding Neurons in a Haystack: Case Studies with Sparse Probing.* TMLR 2023. — arXiv:2305.01610
- **[Survey]** Zheng, Wang, Chen, et al. *Attention Heads of Large Language Models: A Survey.* 2024. — arXiv:2409.03752

## 10. Worked Example

Take GPT-2 small: $d = 768$, $H = 12$, $L = 12$, $d_h = 64$.

**Packing budget.** Almost-orthogonal packing in $\mathbb{R}^{64}$ at interference $\epsilon = 0.3$ admits roughly $m \approx \exp(\epsilon^2 d_h / 8) \cdot \text{poly} \approx e^{0.72}\cdot\text{poly}$ under the crude JL constant, and empirically $\sim 10^3$ vectors with $\max|\langle f_i,f_j\rangle| \le 0.3$ fit comfortably in 64 dims. So geometry permits $\mathcal{O}(10^3)$ atoms per head.

**What the probe reports.** Train an SAE on head L5H1's $z \in \mathbb{R}^{64}$ with dictionary size 512 at loss-recovered 0.90: suppose 310 atoms alive. Re-train at dictionary size 2048, same fidelity target: 940 alive. Same head, same model, same fidelity — the reported feature count triples with the probe's width. Fit the slope: $\log(940/310)/\log(2048/512) = 1.11/1.39 = 0.80$. The atom count is growing nearly linearly in dictionary size, which is the signature of a probe that has not saturated.

**Where the obstruction becomes visible.** Suppose 40% of L5H1's atoms are "previous token is X, attend back" variants. Those are not 400 independent features stored in 64 dimensions; they are one operation — the QK circuit's token-matching pattern — instantiated on 400 different token identities that live in the *residual stream's* dictionary, not the head's. The SAE has no way to tell the two apart, because both produce many sparse, interpretable, distinct atoms in $z$. So the measured $\hat m_h \approx 940$ is not evidence of superposition at capacity $\sim 10^3$; it is consistent with a head of effective capacity $\approx 3$ reading a rich upstream dictionary.

That ambiguity is why §8 makes the *slope* $\beta$ the deciding number rather than any single count: reuse of one operation should be indifferent to $d_h$, while true packing must decompress when the head gets wider.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*