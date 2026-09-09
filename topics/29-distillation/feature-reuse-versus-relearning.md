---
id: 29-distillation/feature-reuse-versus-relearning
title: "Feature Reuse versus Relearning in Fine-Tuning"
topic: 29-distillation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Feature Reuse versus Relearning in Fine-Tuning

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/feature-reuse-versus-relearning` · **Status:** methodologically-blocked

## 1. Problem Statement

When a pretrained network is fine-tuned on a downstream task and beats a from-scratch baseline, the gain has two candidate sources:

- **Reuse** — the fine-tuned model computes downstream predictions using features that already existed in the pretrained checkpoint, and fine-tuning mostly re-routes or rescales them.
- **Relearning** — fine-tuning builds the needed features itself, and pretraining contributed only a favourable optimisation state (conditioning, weight scale, basin location) that any cheap surrogate initialisation could have supplied.

**Input:** a pretrained checkpoint $\theta_0$, a downstream dataset $D$, a fine-tuning procedure $\mathcal{A}$ yielding $\theta_T = \mathcal{A}(\theta_0, D)$.
**Output:** a scalar $\rho \in [0,1]$ — the fraction of the downstream performance gain attributable to reuse of pretrained computation.
**Solved** means: an estimator $\hat{\rho}$ that (i) is invariant to the network's internal symmetries, (ii) separates *reuse of a feature* from *coincidental re-derivation of the same feature*, and (iii) predicts an intervention — e.g. how much downstream accuracy is lost if exactly the reused components are ablated in $\theta_0$ before fine-tuning.

Three variants, of sharply different difficulty:

- **Measurement variant** (the blocked one): define $\rho$ so that it is identified from observable quantities. No accepted definition exists.
- **Method variant**: given a definition, engineer fine-tuning that maximises reuse (parameter-efficient tuning, surgical tuning, LP-FT). Active and productive.
- **Theory variant**: prove, in a tractable model (linear regression, single-index model, sparse-parity), that a stated $\rho$ is recoverable and predicts sample complexity. Partially done in linear settings only.

## 2. Formal Setting

Let $f_\theta = h_{\theta^{\text{head}}} \circ g_\theta$ with backbone $g_\theta: \mathcal{X} \to \mathbb{R}^d$. Fine-tuning minimises $\hat{L}_D(\theta) = \frac{1}{n}\sum_i \ell(f_\theta(x_i), y_i)$ from $\theta_0$.

**Transfer gain**, as measured: run $K$ seeds each of fine-tuning and of from-scratch training under an *identically tuned* budget (same epochs, LR sweep, augmentation),
$$\Delta = \mathbb{E}_{\text{seed}}\big[\mathrm{Acc}(\mathcal{A}(\theta_0,D))\big] - \mathbb{E}_{\text{seed}}\big[\mathrm{Acc}(\mathcal{A}(\theta_{\text{rand}},D))\big].$$

**Surrogate-initialisation control.** Let $\theta_0^{\text{surr}}$ preserve only low-order statistics of $\theta_0$ — e.g. per-layer weight mean and variance (Raghu et al.'s *Mean-Var* init), or a random permutation/shuffle within each filter. Define
$$\rho_{\text{stat}} = 1 - \frac{\mathbb{E}[\mathrm{Acc}(\mathcal{A}(\theta_0^{\text{surr}},D))] - \mathbb{E}[\mathrm{Acc}(\mathcal{A}(\theta_{\text{rand}},D))]}{\Delta}.$$
$\rho_{\text{stat}}$ is the share of the gain *not* explained by weight statistics alone. It is measurable, cheap, and weak: it bounds reuse from above only if surrogates never destroy a reusable feature.

**Representational persistence.** With $X$ a held-out probe set and $G_\ell(\theta) \in \mathbb{R}^{|X| \times d_\ell}$ the layer-$\ell$ activations, linear CKA is
$$\mathrm{CKA}_\ell = \frac{\|G_\ell(\theta_T)^\top G_\ell(\theta_0)\|_F^2}{\|G_\ell(\theta_0)^\top G_\ell(\theta_0)\|_F \,\|G_\ell(\theta_T)^\top G_\ell(\theta_T)\|_F},$$
activations mean-centred over $X$. Invariant to orthogonal transforms and isotropic scaling — **not** to feature reweighting that leaves behaviour unchanged, and not calibrated: $\mathrm{CKA}_\ell$ between two *independently trained* networks is already large in early layers, so $\mathrm{CKA}_\ell$ alone does not isolate reuse.

**Circuit-level reuse.** For a behaviour $b$ with a metric $m_b$, let $C \subseteq$ (components of $f$) be a circuit recovered by path patching in $\theta_0$. Reuse of $C$ is
$$r_b(C) = \frac{m_b(\theta_T) - m_b(\theta_T \mid \mathrm{ablate}\,C)}{m_b(\theta_T) - m_b(\theta_T \mid \mathrm{ablate\ all})}.$$

**Assumptions, and which fail.**
1. *A from-scratch arm exists at matched budget.* Fails for LLM-scale downstream tasks — nobody trains a 7B model from scratch on 50k instructions, so $\Delta$ is unmeasured and $\rho$ has no denominator.
2. *Parameters are comparable across runs.* Fails: permutation and scaling symmetries make weight-space distance non-identifiable (Ainsworth et al., 2023).
3. *Features are discrete objects that persist or do not.* Fails: superposition and basis rotation mean "the same feature" has no canonical referent.
4. *Ablation is a valid counterfactual.* Fails off-distribution: zero-ablation moves activations outside the training manifold, so measured drops overstate causal necessity.

## 3. State of the Art

**Empirical SOTA — established.**
- Neyshabur, Sedghi, Zhang (*What is being transferred in transfer learning?*, NeurIPS 2020): two models fine-tuned from the same $\theta_0$ are **linearly mode-connected** (no loss barrier on the interpolation path), while two from-scratch models are not. Established and replicated. Interpretation as "reuse" is inference, not measurement — basin sharing is compatible with pure relearning inside a favourable basin.
- Raghu et al. (*Transfusion*, NeurIPS 2019): on retinal fundus and CheXpert, ImageNet pretraining gives near-zero final AUC gain for larger architectures; the *Mean-Var* surrogate init recovers much of the convergence speedup. This is the strongest single piece of evidence that a large share of measured "transfer" is not feature reuse.
- Kumar et al. (*Fine-Tuning can Distort Pretrained Features*, ICLR 2022): full fine-tuning degrades OOD accuracy relative to linear probing; LP-FT improves OOD by ~10% on average over FT across 10 shift benchmarks. Theory in a linear-overparameterised model; experiments at ResNet-50/CLIP scale.

**Claimed but unablated.**
- Prakash et al. (*Fine-Tuning Enhances Existing Mechanisms*, ICLR 2024): on entity tracking, fine-tuned Llama-7B variants use the *same* circuit as the base model, enhanced rather than replaced. Strong evidence for reuse — but for one behaviour, one model family, with no from-scratch arm.
- Jain et al. (*Mechanistically analyzing the effects of fine-tuning*, ICLR 2024): fine-tuning on procedural tasks often learns a thin "wrapper" over pretrained capabilities, revertible by pruning a few components. Benchmark-scale is small (TinyStories/PCFG); generality untested.
- Panigrahi et al. (*Task-Specific Skill Localization*, ICML 2023): grafting ~0.01% of fine-tuned RoBERTa parameters back into the pretrained model recovers most of the fine-tuning gain on GLUE. Reported as a localisation result; it does not distinguish "the skill was already there" from "0.01% of parameters sufficed to build it".

**Theory SOTA.** Only linear/kernel regimes. In linear regression with a shared low-dimensional subspace, transfer sample complexity scales with the subspace dimension rather than ambient dimension (Tripuraneni, Jin, Jordan, NeurIPS 2020/2021) — a clean reuse theorem that does not extend to feature-learning regimes.

## 4. What Is Known

- **Basin sharing is real.** Fine-tuned pairs from one checkpoint: interpolation barrier $\approx 0$; from-scratch pairs: clear barrier (NeurIPS 2020, ResNet-50 on CheXpert/DomainNet).
- **Low-rank sufficiency.** RoBERTa-base reaches ~90% of full fine-tuning performance on MRPC optimising in a ~200-dimensional random subspace (Aghajanyan et al., ACL 2021). LoRA at rank 4–8 matches full fine-tuning on GLUE for RoBERTa/DeBERTa and on GPT-3 175B (Hu et al., ICLR 2022).
- **Layer locality.** Surgical fine-tuning of a single block matches or beats full fine-tuning on distribution-shift suites, and the best block depends on shift type — input-level shift favours early blocks, label shift later ones (Lee et al., ICLR 2023).
- **Pretraining benefit shrinks with target data.** Transfusion: with ~200k labelled medical images, ImageNet init changes final AUC by a fraction of a point; the benefit is convergence speed.
- **Similarity metrics disagree.** CKA and CCA-family metrics give different verdicts on the same layer pairs under controlled perturbations (Ding, Denain, Steinhardt, NeurIPS 2021), and CKA is manipulable by adversarially chosen low-rank structure (Davari et al., ICLR 2023).
- **"Forgetting" can be inference, not erasure.** Catastrophic forgetting after fine-tuning is partly explained as task inference — the old capability persists and is recoverable by conditioning (Kotha, Springer, Raghunathan, ICLR 2024).

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no symmetry-invariant, calibrated definition of $\rho$. Every candidate estimator conflates reuse with coincidental convergence: two networks solving the same task develop similar early features whether or not one initialised from the other. Without a "null distribution of similarity for independent solutions", $\mathrm{CKA}_\ell = 0.9$ is uninterpretable.
- **Methodologically blocked.** No agreed unit of "a feature" under superposition, so circuit-level reuse $r_b(C)$ is defined only for behaviours where a circuit has already been hand-recovered — a few dozen, all in models $\le$ 8B.
- **Empirically open.** Whether the Transfusion result generalises to language: is instruction-tuning gain at 7B mostly explained by a statistics-matched surrogate init? Runnable at 1B; nobody has published the matched from-scratch arm.
- **Empirically open.** Whether $\rho$ measured by different estimators (surrogate-init, CKA persistence, circuit ablation, LoRA rank sufficiency) even correlates across the same task set. No paper reports all four.
- **Theoretically open.** No non-vacuous lower bound on downstream sample complexity as a function of any reuse quantity outside linear/kernel regimes.

## 6. Why It Is Hard

The core obstruction is **non-identifiability of the counterfactual**, compounded by a missing control arm.

1. **Coincidental convergence.** Reuse is a causal claim ("this computation exists *because* it was inherited"), but the observable is similarity. Independently trained vision models share Gabor-like early filters; independently trained LMs share induction heads. Similarity is therefore a biased estimator of reuse with unknown bias, and the bias is exactly the quantity of interest.
2. **Symmetry.** Weight-space comparisons are meaningless up to permutation and scaling; Git Re-Basin shows two independent solutions can be permuted into linear connectivity, so even basin-sharing evidence weakens.
3. **The from-scratch arm is unaffordable at the scale where the question matters.** A matched from-scratch control for a 7B instruction-tuned model costs a full pretraining run — $O(10^{22})$–$10^{23}$ FLOPs. So the denominator $\Delta$ is estimated at $\le$ 1B and extrapolated.
4. **Ablation is off-manifold.** Zero- or mean-ablating a circuit in $\theta_T$ measures "damage from a distribution shift in activations", not "necessity of inherited computation". Resample/path patching mitigates but does not remove this.

## 7. Current Research (as of 2026)

- **Mechanistic fine-tuning analysis.** Northeastern (Bau lab) and collaborators continue the circuit-persistence line begun in Prakash et al. (2024): same-circuit-enhanced findings extended to more behaviours and to safety fine-tuning *(frontier — verify)*.
- **Model-diffing.** Crosscoder- and SAE-based comparison of base and fine-tuned checkpoints, asking which learned features are new versus amplified (Anthropic interpretability, EleutherAI, academic replications). This is the most direct attack on the measurement problem — it supplies a feature basis shared across two checkpoints — but its own validity (SAE feature identifiability, shrinkage, dead features) is contested *(frontier — verify)*.
- **Task arithmetic and weight-space geometry.** Task vectors $\tau = \theta_T - \theta_0$ compose additively across tasks (Ilharco et al., ICLR 2023); active work relates additivity to weight-disentanglement and hence to reuse.
- **Surgical / parameter-efficient tuning as a reuse probe** (Stanford, CMU): using *where* tuning must happen as an indirect readout of what could not be reused.
- **Data-attribution routes.** Influence-function and pretraining-data-attribution methods used to ask which pretraining documents support a downstream behaviour after fine-tuning *(frontier — verify)*.

## 8. Concrete Next Experiment

**The four-arm estimator-agreement study.** Purpose: test whether any two candidate $\rho$ estimators agree.

- **Scale.** Two model families where a from-scratch arm is affordable: ViT-B/16 (86M) and a 1.4B decoder-only LM (Pythia-1.4B, which has public checkpoints and a reproducible pretraining recipe). Ten downstream tasks each (vision: DomainNet-6, CheXpert, EuroSAT, Cars, DTD; language: entity tracking, GSM8K-style arithmetic, sentiment, NLI, code completion). $K = 5$ seeds. Budget: ~4k A100-hours for vision, ~30k for the LM arm including one from-scratch pretrain-free baseline per task.
- **Arms.** (a) fine-tune from $\theta_0$; (b) **control:** from-scratch at matched, independently LR-swept budget; (c) Mean-Var surrogate init from $\theta_0$; (d) permutation-shuffled $\theta_0$ (same weights, destroyed circuits, identical per-layer statistics *and* identical weight histogram). Arm (d) is the discriminating control the literature lacks: it holds every statistic fixed and removes only the computation.
- **Measurements per task:** $\rho_{\text{stat}}$; $\rho_{\text{perm}} = 1 - (\mathrm{Acc}_d - \mathrm{Acc}_b)/\Delta$; layerwise $\mathrm{CKA}_\ell(\theta_0,\theta_T)$ *normalised by* $\mathrm{CKA}_\ell$ between two independent from-scratch solutions; minimum LoRA rank reaching 99% of full-FT accuracy.
- **The deciding number.** Spearman $\varrho$ between $\rho_{\text{perm}}$ and normalised CKA persistence across the 20 (model, task) cells. $\varrho \ge 0.7$ ⇒ the estimators track one latent quantity and the measurement problem is tractable with cheap proxies. $\varrho \le 0.3$ ⇒ the field's similarity-based reuse claims are not measuring reuse, and the page's status is confirmed as methodologically blocked. Secondary: mean $\rho_{\text{perm}}$ — if it is below 0.3, most measured transfer is initialisation quality, not feature reuse.

## 9. Key References

- **[Foundational]** Behnam Neyshabur, Hanie Sedghi, Chiyuan Zhang. *What is being transferred in transfer learning?* NeurIPS, 2020. — arXiv:2008.11687
- **[Foundational]** Maithra Raghu, Chiyuan Zhang, Jon Kleinberg, Samy Bengio. *Transfusion: Understanding Transfer Learning for Medical Imaging.* NeurIPS, 2019. — arXiv:1902.07208
- **[Foundational]** Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. *Similarity of Neural Network Representations Revisited.* ICML, 2019. — arXiv:1905.00414
- **[SOTA]** Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR, 2022. — arXiv:2202.10054
- **[SOTA]** Nikhil Prakash, Tamar Rott Shaham, Tal Haklay, Yonatan Belinkov, David Bau. *Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking.* ICLR, 2024. — arXiv:2402.14811
- **[SOTA]** Samyadeep Jain et al. (Samyak Jain, Robert Kirk, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka, Edward Grefenstette, Tim Rocktäschel, David Krueger). *Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks.* ICLR, 2024. — arXiv:2311.12786
- **[SOTA]** Abhishek Panigrahi, Nikunj Saunshi, Haoyu Zhao, Sanjeev Arora. *Task-Specific Skill Localization in Fine-tuned Language Models.* ICML, 2023. — arXiv:2302.06600
- **[Method]** Yoonho Lee, Annie S. Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, Chelsea Finn. *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts.* ICLR, 2023. — arXiv:2210.11466
- **[Method]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Method]** Armen Aghajanyan, Sonal Gupta, Luke Zettlemoyer. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL, 2021. — arXiv:2012.13255
- **[Measurement]** Frances Ding, Jean-Stanislas Denain, Jacob Steinhardt. *Grounding Representation Similarity with Statistical Testing.* NeurIPS, 2021. — arXiv:2108.01661
- **[Measurement]** MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, Eugene Belilovsky. *Reliability of CKA as a Similarity Measure in Deep Learning.* ICLR, 2023. — arXiv:2210.16156
- **[Measurement]** Samuel K. Ainsworth, Jonathan Hayase, Siddhartha Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR, 2023. — arXiv:2209.04836
- **[Theory]** Nilesh Tripuraneni, Chi Jin, Michael I. Jordan. *Provable Meta-Learning of Linear Representations.* ICML, 2021. — arXiv:2002.11684
- **[Related]** Suhas Kotha, Jacob Mitchell Springer, Aditi Raghunathan. *Understanding Catastrophic Forgetting in Language Models via Implicit Inference.* ICLR, 2024. — arXiv:2309.10105
- **[Related]** Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi. *Editing Models with Task Arithmetic.* ICLR, 2023. — arXiv:2212.04089

## 10. Worked Example

Take ViT-B/16, ImageNet-21k pretrained, fine-tuned on DTD (47 texture classes, 1,880 train images). Suppose the four arms return, with 5 seeds:

```
(a) fine-tune from theta_0        76.0%
(b) from-scratch, matched budget  31.0%   <- control
(c) Mean-Var surrogate init       38.0%
(d) permutation-shuffled theta_0  36.5%
Delta = 76.0 - 31.0 = 45.0 pts
```

Then

$$\rho_{\text{stat}} = 1 - \frac{38.0-31.0}{45.0} = 0.844,\qquad \rho_{\text{perm}} = 1 - \frac{36.5-31.0}{45.0} = 0.878 .$$

Both say "most of the gain is reuse". Now the persistence measurement. Layer-6 linear CKA between $\theta_0$ and $\theta_T$: $0.91$. Layer-6 CKA between the two *from-scratch* arm-(b) seeds and $\theta_0$: $0.62$. Normalising,
$$\widetilde{\mathrm{CKA}}_6 = \frac{0.91-0.62}{1-0.62} = 0.76 .$$
The raw 0.91 overstates persistence by a factor that only the arm-(b) control reveals — and the correction is large (0.91 → 0.76) precisely because independently trained ViTs converge to similar mid-level texture features on their own.

The obstruction is visible in the next step. Suppose the same protocol on CheXpert returns $\rho_{\text{perm}} = 0.22$ (arm (d) nearly matches arm (a) — consistent with Transfusion) while $\widetilde{\mathrm{CKA}}_6 = 0.71$, barely lower than DTD's. Two tasks, near-identical representational persistence, reuse fractions of 0.88 and 0.22. Nothing in the CKA number distinguishes them, because CKA measures whether the *same subspace is present*, not whether the downstream head *had to inherit* it. Both figures are internally consistent; the pair is only diagnostic because arm (d) exists — and arm (d) is exactly the control that no published 7B-scale study can afford. That is the block: at the scale where the question matters, the only estimator with a defensible causal reading is the one that is out of budget, and the affordable estimators disagree with it by ~0.6 in $\rho$.

*(Arithmetic above is illustrative of the protocol, using accuracy magnitudes consistent with published ViT-B/16 DTD and CheXpert results; arms (c) and (d) are unmeasured — running them is the experiment in §8.)*

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*