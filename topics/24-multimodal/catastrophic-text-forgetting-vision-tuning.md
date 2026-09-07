---
id: 24-multimodal/catastrophic-text-forgetting-vision-tuning
title: "Catastrophic Text Forgetting from Vision Instruction Tuning"
topic: 24-multimodal
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Catastrophic Text Forgetting from Vision Instruction Tuning

> **Topic:** Multimodal Models · **ID:** `24-multimodal/catastrophic-text-forgetting-vision-tuning` · **Status:** empirically-open

## 1. Problem Statement

Take a language model $M_0$ with known text-only capability. Attach a vision encoder and projector, then instruction-tune on image–text pairs to get a vision-language model $M_1$. **How much text-only capability is lost, why, and what is the cheapest intervention that recovers it without degrading vision performance?**

Three variants, routinely conflated:

- **Measurement.** Given $M_0$ and $M_1$, produce a number $\Delta$ that isolates *forgetting caused by the vision data* from (a) forgetting caused by any further optimization at all, (b) prompt-format shift, (c) chat-template mismatch. This is the blocked variant — most published "text degradation" numbers do not have a compute-matched text-only control arm.
- **Method.** Find a training recipe minimizing $\Delta$ subject to a vision-score floor and a parameter/compute budget. Candidates: text replay, frozen-LM cross-attention adapters, LoRA, mixture-of-experts routing, model merging.
- **Theory.** Predict $\Delta$ from data mixture ratio, adapter rank, and learning rate before training. No such predictive law exists.

Solving it means: a recipe that hits full-fine-tune vision quality with $\Delta$ within measurement noise of the text-only control, plus a scaling law that says how much text replay is needed at a given model size.

## 2. Formal Setting

Let $\theta_0 \in \mathbb{R}^d$ be the base LM weights, $\phi$ the vision encoder + projector parameters. Vision-tuning minimizes

$$\mathcal{L}(\theta,\phi) = \mathbb{E}_{(x_v, x_t, y)\sim \mathcal{D}_V}\big[-\log p_{\theta,\phi}(y \mid x_v, x_t)\big] + \lambda\, \mathbb{E}_{(x_t,y)\sim \mathcal{D}_T}\big[-\log p_\theta(y \mid x_t)\big],$$

where $\mathcal{D}_V$ is the vision instruction set, $\mathcal{D}_T$ the text replay set, and $\lambda \ge 0$ the replay weight. Define the **replay ratio** $r = |\mathcal{D}_T| / (|\mathcal{D}_T| + |\mathcal{D}_V|)$, measured in *tokens*, not examples — image-heavy examples have very different token counts and example-level ratios are not comparable across papers.

Text capability is a benchmark suite score $S_T(\cdot) \in [0,1]$ (MMLU, GSM8K, HumanEval, MT-Bench, IFEval). Naïve forgetting:

$$\Delta_{\text{naive}} = S_T(M_0) - S_T(M_1).$$

This is confounded. The identified quantity requires a **compute-matched control** $M_1^{\text{ctrl}}$: same optimizer, same steps, same token count, same chat template, trained only on $\mathcal{D}_T$-like text instruction data. Then

$$\Delta_{\text{vision}} = S_T(M_1^{\text{ctrl}}) - S_T(M_1),$$

the degradation attributable to the vision data rather than to fine-tuning per se. Because $S_T$ is chance-floored, report the normalized form $\tilde\Delta = \Delta_{\text{vision}} / (S_T(M_1^{\text{ctrl}}) - c)$ with $c$ the chance rate (0.25 for MMLU).

Weight-space proxy: drift $\|\theta_1 - \theta_0\|_2 / \|\theta_0\|_2$, and the Fisher-weighted drift $\sum_i F_i (\theta_{1,i}-\theta_{0,i})^2$ with $F_i$ the diagonal Fisher of $\theta_0$ on text data — the quantity EWC (Kirkpatrick et al., PNAS 2017) penalizes.

**Assumptions, and which fail.**
1. *$S_T$ measures capability, not format compliance.* Violated: much of the observed drop on MMLU/GSM8K is answer-extraction failure under a changed chat template, not lost knowledge. Log-likelihood-scored MMLU and generation-scored MMLU move differently on the same checkpoint.
2. *Base-model text scores are known.* Violated for most open VLMs: the released base is often an already-instruction-tuned chat model whose scores were measured under a different harness version and prompt.
3. *Vision and text losses share a parameter basis.* Partly violated — cross-attention-adapter designs (Flamingo-style) leave $\theta_0$ frozen by construction, so $\Delta_{\text{vision}} \equiv 0$ and the problem is only about the vision-quality cost.
4. *Contamination is equal across arms.* Unverified everywhere.

## 3. State of the Art

**Established (ablated, with a control arm or frozen-LM guarantee):**
- **Frozen-LM cross-attention** (Flamingo, Alayrac et al., NeurIPS 2022; carried into Llama 3's vision adapters, Grattafiori et al., 2024) makes text forgetting exactly zero because $\theta_0$ is untouched. Llama 3 explicitly reports this as the reason for the cross-attention choice. Cost: weaker image-text integration than full fine-tuning on some interleaved and OCR-heavy tasks.
- **Text replay in the SFT mixture.** VILA (Lin et al., CVPR 2024) ablates joint SFT and shows that blending text-only instruction data back into visual SFT recovers most of the text degradation while preserving vision scores. This is the closest thing to a controlled ablation in the literature, and it is the recipe most subsequent VLMs copied.
- **LoRA forgets less than full fine-tuning.** Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024) establish the trade-off frontier in the code/math domain: LoRA loses less source-domain capability at equal target-domain gain, but caps target-domain gain. Not measured on the vision axis.

**Claimed but unablated:** Wings (Zhang et al., NeurIPS 2024) introduces modality-specific low-rank residual experts and reports reduced text-only forgetting on its own benchmark — the comparison arms are not compute-matched text-only controls. Model-merging and MoE routing fixes are reported as benchmark deltas without weight-space or Fisher analysis.

**Benchmark-number-only:** the widely repeated claim that LLaVA-style full fine-tuning "destroys" text ability rests largely on scattered MMLU/MT-Bench comparisons against the base chat model, i.e. $\Delta_{\text{naive}}$, not $\Delta_{\text{vision}}$.

## 4. What Is Known

- Continual fine-tuning of decoder LMs at **1B–7B** measurably degrades held-out capability, and degradation grows with model scale in the 1B→7B range for general knowledge and reasoning (Luo et al., *An Empirical Study of Catastrophic Forgetting in LLMs During Continual Instruction Tuning*, 2023). Measured on text-only continual tuning, not vision.
- Multimodal instruction tuning degrades the model's own **visual** classification ability relative to the frozen CLIP encoder — the EMT evaluation (Zhai et al., 2023) shows fine-tuned LLaVA underperforming its encoder's linear probe on CIFAR-10/100-style tasks at 7B/13B. Direct evidence that vision SFT overwrites pre-existing competence, in the visual direction.
- **Data mixture dominates architecture** at fixed scale: Idefics2 (Laurençon et al., NeurIPS 2024) and Cambrian-1 (Tong et al., NeurIPS 2024) both find SFT mixture composition to be the largest lever on final scores at 7B–13B; Cambrian-1's mixture explicitly includes text-only data for this reason.
- Vision instruction sets are small relative to LM pre-training: LLaVA-1.5 uses ~665K SFT examples (Liu et al., CVPR 2024), roughly $10^{-4}$ of the base model's pre-training tokens — so the forgetting is caused by a small, highly non-i.i.d. update, not by data volume.
- EWC-style Fisher regularization bounds drift in the classic continual-learning setting (Kirkpatrick et al., PNAS 2017). It is essentially untested at VLM scale.

## 5. What Is Not Known

- **Empirically open (dominant).** No published VLM training run reports $\Delta_{\text{vision}}$ against a compute-matched text-only control arm at $\ge$7B. The experiment is a few thousand GPU-hours and entirely runnable; nobody has run it as a clean 2×2.
- **Empirically open.** The scaling of required replay ratio $r^\*(N)$ with parameter count $N$. Recipes use $r \approx 0.05$–$0.2$ by folklore. Whether $r^\*$ rises or falls with $N$ is unmeasured.
- **Methodologically blocked.** Separating *knowledge loss* from *format/extraction loss*. Until every text score is reported under both log-likelihood and generation scoring with the VLM's own chat template applied to both arms, $\Delta$ is not identified.
- **Theoretically open.** Whether there exists a parameterization with zero text forgetting and no vision-quality penalty — i.e. whether the LoRA-style capacity/retention trade-off is fundamental or an artifact of low-rank updates.
- **Theoretically open.** No bound relating Fisher-weighted drift to downstream benchmark loss for autoregressive LMs; the quadratic EWC approximation is known to be poor far from $\theta_0$.

## 6. Why It Is Hard

The obstruction is **confounded measurement**, not compute.

Three effects are superimposed in every reported number: (i) genuine parameter overwriting, (ii) chat-template and answer-format shift that breaks the eval harness's answer extraction, (iii) ordinary fine-tuning drift that any additional SFT — text or vision — would produce. Only (i) is the object of interest, and no standard reporting protocol separates them. A 4-point MMLU drop can be entirely (ii): re-running the same weights with the base model's template often recovers most of it.

Second obstruction: **absent ground truth for the control arm.** "The same training run without images" is under-specified — matched on steps, tokens, examples, or optimizer state? Each choice gives a different $\Delta$, and papers do not state which they used.

Third: **non-identifiability of the mechanism.** Weight drift, Fisher-weighted drift, and benchmark drop are only loosely correlated; a run can drift a lot in $L_2$ and lose nothing measurable, so the weight-space proxy cannot arbitrate.

## 7. Current Research (as of 2026)

- **Frozen-LM adapters as the default for frontier text models.** Meta's Llama vision line keeps $\theta_0$ frozen precisely to guarantee text parity; the open trade-off is how much vision quality that costs on OCR/document tasks.
- **Mixture-composition science.** Cambrian-1 (NYU), Idefics2/HuggingFace M4, Molmo (AI2, Deitke et al., 2024) treat the text fraction of SFT as a tuned hyperparameter. Published as ablation tables, not as a scaling law. *(frontier — verify: whether any 2025–26 release publishes $r^\*$ vs $N$.)*
- **Modality-routed parameters.** Wings-style residual experts, modality-conditioned MoE, and per-modality LoRA. *(frontier — verify: independent reproduction of Wings' forgetting reduction.)*
- **Continual multimodal benchmarks.** CoIN (NeurIPS 2024 Datasets & Benchmarks) supplies a task-sequence protocol; it measures cross-task forgetting inside vision, and extending it to a text-retention axis is an open slot.
- **Merging and task arithmetic** as a post-hoc repair: interpolate $\theta_1$ back toward $\theta_0$ and sweep the coefficient. Cheap, under-reported.

## 8. Concrete Next Experiment

**The 2×2 with a compute-matched control.**

- **Scale.** One base model, 8B parameters, open weights, known pre-training corpus (e.g. a Llama-3.1-8B or Qwen-2.5-7B *base*, not chat — this removes assumption 2's violation). Four arms, each 665K-example-equivalent, **token-matched** and step-matched:
  - **A (control):** text-only instruction SFT, same template, same token budget.
  - **B:** vision SFT only ($r=0$), LLaVA-1.5 mixture.
  - **C:** vision SFT with $r = 0.10$ text replay.
  - **D:** vision SFT, LM frozen, cross-attention adapter only.
- **Evaluation.** MMLU, GSM8K, HumanEval, IFEult (IFEval), MT-Bench — each scored **twice**, log-likelihood and generation, under each arm's own chat template *and* under arm A's template. Vision: MMMU, TextVQA, DocVQA.
- **Cost.** ~4 × 500 A100-hours plus eval. Under 3,000 GPU-hours total.
- **The deciding number.** $\tilde\Delta_{\text{vision}}(B) = [S_T(A) - S_T(B)] / [S_T(A) - 0.25]$ on log-likelihood-scored MMLU, with a bootstrap CI over the 14K MMLU items.
  - $\tilde\Delta_{\text{vision}}(B) < 0.02$ → text forgetting from vision SFT is a measurement artifact; the field's premise is wrong and the effort belongs on template robustness.
  - $\tilde\Delta_{\text{vision}}(B) > 0.10$ → real overwriting; then $\tilde\Delta_{\text{vision}}(C)$ tells whether 10% replay closes it, and arm D prices the vision cost of the zero-forgetting guarantee.
- **Secondary readout.** The gap between generation-scored and log-likelihood-scored $\Delta$ on arm B is a direct estimate of the format-confound magnitude — the first such number in the literature.

## 9. Key References

- **[Foundational]** Kirkpatrick, Pascanu, Rabinowitz, et al. *Overcoming catastrophic forgetting in neural networks.* PNAS 114(13), 2017.
- **[Foundational]** McCloskey & Cohen. *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* Psychology of Learning and Motivation, 1989.
- **[Foundational]** Alayrac, Donahue, Luc, et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS 2022. — arXiv:2204.14198
- **[Foundational]** Liu, Li, Wu, Lee. *Visual Instruction Tuning.* NeurIPS 2023. — arXiv:2304.08485
- **[SOTA]** Liu, Li, Li, Lee. *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5). CVPR 2024. — arXiv:2310.03744
- **[SOTA]** Lin, Yin, Ping, et al. *VILA: On Pre-training for Visual Language Models.* CVPR 2024. — arXiv:2312.07533
- **[SOTA]** Biderman, Ortiz, Portes, et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024.
- **[SOTA]** Zhang, Yao, Ye, et al. *Wings: Learning Multimodal LLMs without Text-only Forgetting.* NeurIPS 2024.
- **[Empirical]** Zhai, Tong, Li, et al. *Investigating the Catastrophic Forgetting in Multimodal Large Language Models.* 2023. — arXiv:2309.10313
- **[Empirical]** Luo, Yang, Meng, et al. *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.* 2023.
- **[Survey/Recipe]** Tong, Brown, Wu, et al. *Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs.* NeurIPS 2024. — arXiv:2406.16860
- **[Survey/Recipe]** Laurençon, Tronchon, Cord, Sanh. *What matters when building vision-language models?* (Idefics2). NeurIPS 2024. — arXiv:2405.02246
- **[Benchmark]** Chen, Zhu, Chen, et al. *CoIN: A Benchmark of Continual Instruction Tuning for Multimodal Large Language Models.* NeurIPS 2024 Datasets & Benchmarks.

## 10. Worked Example

Take the canonical case: Vicuna-1.5-13B → LLaVA-1.5-13B, full fine-tune on the 665K mixture.

- Base text score: Vicuna-1.5-13B reports MMLU ≈ **55.8** (5-shot, LMSYS release notes).
- LLaVA-1.5-13B's paper reports **no text-only benchmark at all**. Every table is vision. So the headline claim "vision tuning costs ~4 MMLU points" is assembled from third-party re-evaluations under unstated harnesses.

Suppose a re-evaluation gives LLaVA-1.5-13B MMLU = 51.9, generation-scored, under LLaVA's own template. Then:

$$\Delta_{\text{naive}} = 55.8 - 51.9 = 3.9,\qquad \tilde\Delta_{\text{naive}} = \frac{3.9}{55.8-25} = 0.127.$$

A 12.7% relative loss — apparently large. Now decompose it:

- Re-score the *same weights* by log-likelihood over the four options (no answer extraction): 54.1. Format confound = **2.2 points**, 56% of the effect.
- Train the control arm A — 665K text-only instruction examples, same steps, same template — and it scores 54.6.

$$\Delta_{\text{vision}} = 54.6 - 54.1 = 0.5, \qquad \tilde\Delta_{\text{vision}} = \frac{0.5}{54.6-25} \approx 0.017.$$

The 3.9-point "catastrophic forgetting" resolves into 2.2 points of answer-format brittleness, 1.2 points of generic SFT drift that text-only tuning also causes, and **0.5 points** attributable to the images. With a 14K-item MMLU bootstrap the standard error is roughly 0.4 points, so 0.5 is not distinguishable from zero.

The obstruction is now visible: the reported effect and the identified effect differ by nearly an order of magnitude, the sign of the identified effect is not resolvable at this eval size, and the arithmetic above cannot be checked against the literature because **arm A has never been trained and $S_T$ for LLaVA-1.5-13B was never published by its authors**. The numbers past line two are what the experiment in §8 would supply; the point is that the current evidence base cannot tell 0.017 from 0.127.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*