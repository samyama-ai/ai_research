---
id: 24-multimodal/cross-modal-routing-circuits
title: "Mechanistic Circuit for Cross-Modal Information Routing"
topic: 24-multimodal
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Mechanistic Circuit for Cross-Modal Information Routing

> **Topic:** Multimodal Models · **ID:** `24-multimodal/cross-modal-routing-circuits` · **Status:** open

## 1. Problem Statement

In a decoder-only vision-language model (VLM) — image encoder, projector, LLM backbone — visual evidence must reach the token position where an answer is emitted. **Which components carry it, in what order, and does the answer generalize across prompts, models, and modalities?**

Three variants, of very different difficulty:

- **Measurement.** Given a model $M$, a task distribution $\mathcal{D}$, and a faithfulness metric, produce a subgraph $C$ of the computational graph that reproduces $M$'s behavior on $\mathcal{D}$ to within tolerance $\epsilon$ while containing a small fraction of edges. Solving = a circuit that survives *causal scrubbing* (resampling ablation consistent with the claimed hypothesis), not just mean-ablation.
- **Method.** Find such circuits automatically at 7B–70B scale with vision tokens numbering $10^2$–$10^4$, where existing patching methods cost $O(|\text{nodes}|)$ forward passes.
- **Theory.** Is cross-modal routing *modality-specific* — a mechanism absent from text-only models — or is it the pre-existing text retrieval circuitry (induction heads, attribute-extraction heads, mover heads) applied to tokens that happen to originate in a projector? A proof-shaped answer would state conditions on the projector and the training objective under which the visually-conditioned circuit is a graph isomorphism of the text-only one.

## 2. Formal Setting

Let the model compute residual streams $h^{(\ell)}_i \in \mathbb{R}^d$ at layer $\ell \in \{1..L\}$, position $i \in \{1..T\}$. Positions split into $P_v$ (projected visual tokens), $P_t$ (prompt text), $P_a$ (answer position). Write the computational graph $G = (V, E)$ where $V$ = attention heads $\{a^{(\ell)}_h\}$, MLPs $\{m^{(\ell)}\}$, and embedding/unembedding nodes; edges are residual-stream writes read by downstream nodes.

**Task.** Paired inputs $(x, x')$ where $x'$ differs only in the visual fact queried (a *counterfactual image*), with logit-difference metric
$$\mathcal{M}(x) = \log p(y \mid x) - \log p(y' \mid x).$$

**Edge attribution.** For edge $e$, the patched effect is
$$\mathrm{IE}(e) = \mathcal{M}\big(M_{\,e \leftarrow x'}(x)\big) - \mathcal{M}(M(x)),$$
where $e \leftarrow x'$ means the activation flowing along $e$ is replaced by its value on $x'$. Measured either exactly (one forward pass per edge) or by attribution patching, the first-order estimate $\mathrm{IE}(e) \approx (a_e' - a_e)^\top \partial \mathcal{M} / \partial a_e$, which costs two forwards and one backward for all edges at once but is known to be inaccurate where the loss is locally flat or saturated.

**Circuit.** $C \subseteq E$ with **faithfulness** $F(C) = \mathbb{E}_x[\mathcal{M}(M_{\bar{C}\leftarrow x'}(x))] / \mathbb{E}_x[\mathcal{M}(M(x))]$ (all edges *outside* $C$ resampled), **completeness** (no $K \subseteq C$ whose removal has an effect unexplained by $C$), and **sparsity** $|C|/|E|$.

**Modality-routing score.** For a matched text-only rendering $x_{\text{txt}}$ of the same fact (caption instead of image), define
$$\rho = \frac{|C_{\text{img}} \cap C_{\text{txt}}|}{|C_{\text{img}} \cup C_{\text{txt}}|}.$$
$\rho \to 1$ means routing reuses text machinery; $\rho \to 0$ means a modality-specific circuit exists.

**Assumptions, and which are violated.** (i) *Positions are comparable across $x$ and $x'$* — violated whenever image tokenization is resolution-dependent or uses tiling/AnyRes, so $|P_v|$ changes. (ii) *Resampling from $x'$ stays on-distribution* — violated; patched activations are off-manifold, which is exactly what causal scrubbing's hypothesis-consistent resampling tries to repair, at high sample cost. (iii) *Linear, additive residual stream* — approximately holds, but LayerNorm scaling makes edge effects non-additive. (iv) *No self-repair* — violated: the Hydra effect (McGrath et al., 2023) shows downstream components compensate for ablated ones, so $\mathrm{IE}$ underestimates necessity.

## 3. State of the Art

**Established (with ablations).**
- *Text-based decomposition of CLIP* (Gandelsman, Efros, Steinhardt, ICLR 2024): CLIP ViT's image representation decomposes across heads and positions; a small number of late-layer heads carry most direct effect on the output, and individual heads admit text descriptions. Ablation-verified; used to remove spurious cues.
- *Causal tracing in VLMs* (Basu, Grayson, Cohen, Rossi, Feizi et al., "Understanding Information Storage and Transfer in Multi-modal LLMs", NeurIPS 2024): in LLaVA-style models, the causal sites for visual-fact retrieval sit in **early-to-mid MLP layers at the last text position**, and consistently earlier than the corresponding sites in the text-only backbone. Includes an editing check (MEMIT-style edits transfer).
- *Automatic circuit discovery* (Conmy et al., ACDC, NeurIPS 2023) and *IOI* (Wang et al., ICLR 2023) fix the methodology — edge-level pruning against a faithfulness threshold — but were validated on text-only GPT-2 small (117M).

**Claimed but not fully ablated.**
- *Cross-modal information flow* (Zhang et al., 2024/2025): visual information is first broadcast into text positions in lower layers, then concentrated at the final position in higher layers. The two-stage picture is supported by attention-knockout curves, not by a scrubbed circuit.
- *Visual information processing in VLMs* (Neo, Ong, Torr, Jiang, Shin, Nanda, 2024/2025): visual tokens at object locations become linearly decodable in the LLM's own vocabulary space by mid layers — i.e. the LLM refines visual tokens in place rather than only pulling from them. Logit-lens evidence; logit lens is known to be unreliable in mid-stack.

**Benchmark-number-only.** Most VLM interpretability claims are reported as accuracy drop under attention knockout on VQA-style sets. A drop is not a circuit: it names no edges and does not distinguish routing from representation degradation.

## 4. What Is Known

- **Multimodal neurons exist.** CLIP RN50x4: single units respond to a concept across photo, sketch and rendered text (Goh et al., Distill 2021). Text-only transformers reached through a linear projector also contain units that translate visual features into related text tokens (Schwettmann et al., 2023, on 6B-scale GPT-J with a linear image adapter).
- **Register/sink behavior.** ViTs allocate high-norm "register" tokens in low-information patches (Darcet, Oquab, Mairal, Bojanowski, ICLR 2024); LLM backbones concentrate attention on sinks. Any routing circuit measured without controlling for sinks attributes mass to positions that carry no task content.
- **Most visual tokens are removable.** Token-pruning work at LLaVA-1.5-7B scale (576 visual tokens) shows large fractions can be dropped with small accuracy loss — evidence that routing is sparse in $P_v$, and a warning that edge-importance is highly redundant.
- **Method fidelity.** Attribution patching agrees with activation patching on high-effect edges and diverges on low-effect ones (Syed, Rager, Conmy, 2023); at VLM scale the circuit is mostly made of low-effect edges, so the cheap method is weakest where it is most needed.

## 5. What Is Not Known

- **Theoretically open.** Whether $\rho$ (Section 2) must be near 1 for any model trained with a frozen LLM and a learned projector. No result rules out either regime. Also open: identifiability — whether two disjoint edge sets can both be faithful and complete on the same $\mathcal{D}$.
- **Empirically open.** No published, scrubbed, edge-level circuit for a single cross-modal routing task at $\geq$7B with $\geq$576 visual tokens. The experiment is runnable; the cost (Section 8) is why it has not been run.
- **Methodologically blocked.** The counterfactual image. Text circuits get clean minimal pairs ("John gave Mary" / "Mary gave John"). Changing one visual fact changes thousands of pixels and hence every visual token, so the "minimal" intervention is not minimal, and $\mathrm{IE}$ mixes routing with encoder-side representation change.

## 6. Why It Is Hard

Four named obstructions, in order of bite:

1. **No minimal counterfactual (absent ground truth for the intervention).** Patching from $x'$ perturbs all of $P_v$ at once. Synthetic renders (one attribute changed, pixel-identical elsewhere) fix this but are off-distribution for the encoder.
2. **Non-identifiability under redundancy.** Visual evidence is duplicated across many patch tokens and many heads. Ablating any one changes little; the greedy edge-pruning that ACDC performs then returns a circuit that is faithful but not unique — a different random seed returns a different, equally faithful set.
3. **Self-repair.** The Hydra effect means necessity measured by ablation systematically undershoots. Circuits found by thresholding $|\mathrm{IE}|$ therefore omit genuinely used components.
4. **Compute.** Exact edge patching is $O(|E|)$ forward passes. For a 32-layer, 32-head backbone with $T \approx 1200$, edge-level attribution over head-to-head connections is $\sim 10^6$ candidate edges; exhaustive activation patching at ~0.2 s/pass is ~$10^5$ GPU-hours per task.

## 7. Current Research (as of 2026)

- **Sparse-feature circuits.** Replacing neurons with SAE/transcoder features as circuit nodes, then patching in feature space (Marks, Rager, Michaud, Belinkov, Bau, Mueller, 2024). Extension to VLM projector outputs is active *(frontier — verify)*.
- **Attention-knockout maps of modality flow** (Zhang, Gu and collaborators; groups at Amsterdam and CMU) — moving from layer-band claims to head-level claims.
- **Editing as validation** — Basu et al.'s line: if the circuit is right, editing its MLP sites should change visual-fact answers and nothing else. Strongest available falsifier.
- **Native-multimodal models** (interleaved-trained, no frozen backbone) — whether $\rho$ falls when the LLM is not frozen is being probed *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Is cross-modal routing modality-specific, or reused text machinery? Decide by measuring $\rho$.

- **Scale.** LLaVA-1.5-7B (Vicuna-7B backbone, CLIP ViT-L/14-336, 576 visual tokens) plus one native-multimodal control at comparable size. Task: single-attribute retrieval ("What color is the mug?") over 2,000 *procedurally rendered* scenes (Blender/CLEVR-style) where the counterfactual differs in exactly one object attribute and is otherwise pixel-identical — this buys the minimal pair that natural images cannot.
- **Arms.** (a) Image condition. (b) **Control arm:** identical questions with the scene given as a text caption, same answer distribution — yields $C_{\text{txt}}$. (c) Shuffled-image placebo to calibrate the noise floor of edge attribution.
- **Procedure.** Attribution patching to rank $\sim10^6$ edges; exact activation patching on the top 3,000; ACDC-style pruning to the smallest $C$ with $F(C) \geq 0.8$; causal-scrubbing check on the final $C$ with 100 resamples per node.
- **Budget.** ~2,500 A100-hours. This is the smallest version that is not a toy.
- **Deciding number.** $\rho = |C_{\text{img}} \cap C_{\text{txt}}| / |C_{\text{img}} \cup C_{\text{txt}}|$ over attention heads. **$\rho \geq 0.7$** (against a placebo-arm floor that should sit near 0.1–0.2) supports reuse: no modality-specific circuit, and interpretability of VLMs reduces to interpretability of the backbone plus the projector. **$\rho \leq 0.3$** establishes a distinct routing mechanism worth naming and cataloguing. Report $\rho$ with a seed-variability band across 5 pruning seeds; if the band exceeds $\pm 0.15$, the result is non-identifiability (obstruction 2), not an answer.

## 9. Key References

- **[Foundational]** Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, Chris Olah. *Multimodal Neurons in Artificial Neural Networks.* Distill, 2021.
- **[Foundational]** Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, Jacob Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* ICLR, 2023. — arXiv:2211.00593
- **[SOTA]** Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS, 2023. — arXiv:2304.14997
- **[SOTA]** Yossi Gandelsman, Alexei A. Efros, Jacob Steinhardt. *Interpreting CLIP's Image Representation via Text-Based Decomposition.* ICLR, 2024. — arXiv:2310.05916
- **[SOTA]** Samyadeep Basu, Martin Grayson, Cecily Morrison, Besmira Nushi, Soheil Feizi, Daniela Massiceti. *Understanding Information Storage and Transfer in Multi-modal Large Language Models.* NeurIPS, 2024. — arXiv:2406.04236
- **[SOTA]** Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, Aaron Mueller. *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.* 2024. — arXiv:2403.19647
- **[Method]** Aaquib Syed, Can Rager, Arthur Conmy. *Attribution Patching Outperforms Automated Circuit Discovery.* 2023. — arXiv:2310.10348
- **[Method]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Context]** Thomas McGrath, Matthew Rahtz, János Kramár, Vladimir Mikulik, Shane Legg. *The Hydra Effect: Emergent Self-repair in Language Model Computations.* 2023. — arXiv:2307.15771
- **[Context]** Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. *Vision Transformers Need Registers.* ICLR, 2024. — arXiv:2309.16588
- **[Context]** Sarah Schwettmann, Neil Chowdhury, Samuel Klein, David Bau, Antonio Torralba. *Multimodal Neurons in Pretrained Text-Only Transformers.* ICCV Workshops, 2023. — arXiv:2308.01544
- **[Survey]** Haoyi Qiu et al. / see also Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, Fazl Barez. *Towards Interpreting Visual Information Processing in Vision-Language Models.* 2024. — arXiv:2410.07149

## 10. Worked Example

**Setup.** LLaVA-1.5-7B. Prompt: `USER: <image> What color is the mug? ASSISTANT:`. $x$ renders a red mug, $x'$ a blue mug, identical camera, lighting and geometry. $\mathcal{M} = \log p(\text{"red"}) - \log p(\text{"blue"})$; unpatched $\mathcal{M}(x) \approx +6$ nats.

**Step 1 — where is the information?** Patch all 576 visual-token residual streams at layer $\ell$ from $x'$. Expect $\mathcal{M}$ to flip sign for $\ell \lesssim 15$ and to stop flipping by $\ell \approx 20$: the answer has already been moved to the last position. That reproduces the two-stage picture — and settles nothing, because "layers 1–15 at 576 positions" is $8{,}640$ nodes, not a circuit.

**Step 2 — which visual tokens?** Patch one visual token at a time. With the mug covering ~9 patches of 576, expect each single-token patch to move $\mathcal{M}$ by well under 1 nat, while patching all 9 together flips it. Sum of individual effects $\ll$ joint effect: the redundancy is superadditive. Greedy top-$k$ selection by individual effect therefore returns tokens in an order barely distinguishable from the noise floor set by the shuffled-image placebo.

**Step 3 — the obstruction, made numeric.** Suppose per-token effects are $\{0.31, 0.28, 0.27, 0.26, 0.24, 0.22, 0.21, 0.19, 0.18\}$ nats for the 9 mug patches, and the placebo arm's 95th-percentile single-token effect is $0.17$ nats. The separation is $\approx 1.1\times$. Two pruning seeds that break ties differently return circuits sharing perhaps 4 of 9 tokens — $\rho$-style overlap $0.29$ *within the same condition*. Any cross-modal overlap measured against the text arm is then swamped by seed variance.

**What this shows.** The blocker is not that the model is opaque. It is that the intervention has no minimal unit: visual evidence is spread redundantly across tokens and heads, so ablation-based importance produces a faithful-but-non-unique circuit. Progress requires either an intervention with a genuinely minimal unit (rendered counterfactuals plus feature-space, not token-space, nodes) or a faithfulness metric that scores *sets* rather than ranking edges — which is why Section 8 makes seed variability, not just $\rho$, part of the readout.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*