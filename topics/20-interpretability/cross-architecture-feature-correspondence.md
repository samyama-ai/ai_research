---
id: 20-interpretability/cross-architecture-feature-correspondence
title: "Cross-Architecture Feature Correspondence"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Architecture Feature Correspondence

> **Topic:** Interpretability · **ID:** `20-interpretability/cross-architecture-feature-correspondence` · **Status:** open

## 1. Problem Statement

Given two trained networks $A$ and $B$ that differ in **architecture** — not just seed — decide which internal features of $A$ correspond to which features of $B$, and certify the answer.

- **Input:** two models, a shared input distribution, and a decomposition of each model's activations into features (neurons, SAE latents, attention heads, subspaces).
- **Output:** a partial matching $\pi \subseteq F_A \times F_B$ plus a confidence per pair.
- **Decision predicate:** for a claimed pair $(f_A, f_B)$, does intervening on $f_B$ produce the causal consequence that intervening on $f_A$ produces, on held-out inputs?

Three variants, routinely conflated:

- **Measurement:** define "same feature" across models with different widths, depths, tokenizers, and state dynamics (attention vs. SSM vs. convolution). This is the blocked variant.
- **Method:** given the definition, find the matching efficiently over $10^5$–$10^7$ candidate features per model.
- **Theory:** prove when correspondence must exist. The universality hypothesis (Olah et al., Distill 2020) predicts convergent features; the Platonic Representation Hypothesis (Huh et al., ICML 2024) predicts convergent *representations*, which is weaker and does not imply a feature-level matching.

Solving it means: a matcher that, on models with known shared structure, recovers the planted correspondence at high precision, and whose pairs survive causal transfer tests on models where no ground truth exists.

## 2. Formal Setting

Let $x \sim \mathcal{D}$ over $N$ held-out inputs. Model $A$ has hidden state $h_A(x) \in \mathbb{R}^{d_A}$ at a chosen site (residual stream at layer $\ell$, or MLP post-activation). Feature dictionaries:

$$h_A(x) \approx b_A + \sum_{i=1}^{m_A} a_i^A(x)\, d_i^A, \qquad a^A(x) = \mathrm{ReLU}(W_A^{\mathrm{enc}}(h_A(x)-b_A)+b^{\mathrm{enc}}_A)$$

trained by $\min \mathbb{E}\|h_A - \hat h_A\|_2^2 + \lambda\|a^A\|_1$ (Cunningham et al., ICLR 2024). **Measured as:** $a^A \in \mathbb{R}^{N \times m_A}$, a sparse matrix, $\|a^A(x)\|_0$ typically 20–100 with $m_A/d_A \in [8,64]$.

**Correlational score.** $\rho_{ij} = \mathrm{corr}_N(a_i^A, a_j^B)$; matching by rectangular linear assignment $\max_\pi \sum \rho_{ij}$, or Hungarian on $-\rho$. Cost $O(m_A m_B)$ to build, $O(m^3)$ to solve — the binding constraint at $m = 10^6$.

**Representational score.** Linear CKA (Kornblith et al., ICML 2019) on centered Gram matrices $K = h_Ah_A^\top$, $L = h_Bh_B^\top$:

$$\mathrm{CKA}(K,L)=\frac{\|h_B^\top h_A\|_F^2}{\|h_A^\top h_A\|_F\,\|h_B^\top h_B\|_F}.$$

This scores *spaces*, not features; it cannot return $\pi$.

**Causal score — the one that matters.** For pair $(i,j)$ and intervention $\mathrm{do}(a_i^A \!\leftarrow\! a_i^A + \alpha)$, define the transfer ratio on next-token distributions:

$$T_{ij}(\alpha) = \frac{\big\langle \Delta_A(\alpha),\, \Delta_B(\alpha)\big\rangle}{\|\Delta_A(\alpha)\|\,\|\Delta_B(\alpha)\|}, \quad \Delta_M(\alpha) = \mathbb{E}_x\big[\log p_M^{\mathrm{do}}(\cdot|x) - \log p_M(\cdot|x)\big].$$

A pair is **certified** if $T_{ij} > \tau$ on inputs disjoint from those used to fit $\pi$, and $T_{ij'} $ for random $j'$ is near 0. With different tokenizers, $\Delta$ must be pushed through a shared output vocabulary — an unsolved coupling step.

**Assumptions, with violation status:**
1. *A shared input distribution exists.* Holds for text/image models; violated when pretraining corpora differ, which shifts feature inventories.
2. *Features are directions, and superposition is linear.* Known violated in part — nonlinear and multi-dimensional features are documented (e.g. circular day-of-week features, Engels et al., 2024).
3. *Dictionaries are identifiable.* Violated: SAEs trained on the same model with different seeds recover overlapping but non-identical dictionaries.
4. *Sites are comparable.* Violated across architectures — layer $\ell$ of a 12-layer transformer and layer $\ell$ of a 48-layer SSM are not the same computational depth.

## 3. State of the Art

**Established.**
- **Stitching** (Lenc & Vedaldi, CVPR 2015; Bansal, Nakkiran & Barak, NeurIPS 2021): insert a trained affine map between layer $\ell$ of $A$ and layer $\ell+1$ of $B$; measure the accuracy penalty. Well-ablated, gives a scalar per layer pair, gives no per-feature matching.
- **CKA / SVCCA / PWCCA** (Raghu et al., NeurIPS 2017; Morcos et al., NeurIPS 2018; Kornblith et al., ICML 2019): space-level similarity, cheap, widely reproduced — and shown fragile (Ding et al., NeurIPS 2021; Davari et al., ICLR 2023).
- **One-to-one neuron matching** (Li et al., ICLR 2016, "Convergent Learning"): correlation matching between independently trained AlexNets. Established that some units match one-to-one and many require one-to-many maps.
- **Relative representations** (Moschella et al., ICLR 2023): re-encode each activation by its similarities to a set of anchor inputs; enables zero-shot stitching across models and even modalities. Established at small scale; the anchor set is a hyperparameter with no principled choice.

**Claimed but unablated.**
- **Crosscoders** (Lindsey, Templeton et al., Transformer Circuits 2024) train one dictionary jointly on both models' activations, yielding shared and model-specific latents. Demonstrated for base-vs-finetuned diffing; the cross-*architecture* case is asserted as a natural extension, not shown. Subsequent work notes "model-specific" latents can be artifacts of the $L_1$ penalty, not real differences.
- **Cross-architecture universality** (Wang et al., 2024, on Mamba vs. transformer) reports analogous induction-like mechanisms in SSMs. Interesting, but reported as case studies, not a matcher with precision/recall.
- SAE quality on benchmarks (SAEBench, Karvonen et al., 2025) exists only as benchmark numbers; no benchmark yet scores *cross-model* matching.

## 4. What Is Known

- **Same-seed universality is partial.** Gurnee et al. (2024) find that across five GPT-2 Small seeds (124M), only roughly 1–5% of neurons are "universal" by their correlation threshold — and those are disproportionately interpretable (entropy neurons, position, token-frequency).
- **Stitching penalties are small but nonzero.** Bansal et al. (NeurIPS 2021) show ResNets on CIFAR-10/ImageNet stitch across seeds and across differing widths with modest accuracy loss, and that stitching penalty is not predicted by CKA.
- **CKA is not a reliable arbiter.** Ding et al. (NeurIPS 2021) show CKA can be insensitive to changes that alter functional behaviour and sensitive to ones that do not; Davari et al. (ICLR 2023) show CKA is dominated by a few high-variance directions and can be driven to arbitrary values by translation.
- **SAE dictionaries are seed-unstable.** Reported feature-level overlap between SAEs trained with different seeds on the same model is well below 100% at standard widths (16k–1M latents on models from GPT-2 Small to Claude 3 Sonnet); the exact figure is metric-dependent and no consensus number exists.
- **Representational convergence grows with scale.** Huh et al. (ICML 2024) show mutual-nearest-neighbour alignment between vision and language models increases with model capability — measured on off-the-shelf checkpoints, not causally tested.

## 5. What Is Not Known

- **Methodologically blocked:** what "the same feature" means when $d_A \neq d_B$, tokenizers differ, and features are multi-dimensional. No accepted cross-model analogue of the causal $T_{ij}$ above with a validated $\tau$. This is the primary blocker.
- **Empirically open:** the crosscoder-vs-independent-SAE-plus-Hungarian comparison at $\geq$1B parameters across a transformer and a non-transformer, scored by causal transfer. Runnable today; unrun.
- **Theoretically open:** no theorem gives conditions under which two architectures trained on the same distribution must share a feature basis up to permutation. Linear-mode-connectivity-modulo-permutation results apply within an architecture, not across.

## 6. Why It Is Hard

**Non-identifiability compounded by absent ground truth.** The dictionary is not unique: $\lambda$, width, and seed each change which latents appear, so a mismatch between $A$'s and $B$'s dictionaries is unattributable — it could be a real architectural difference or a decoder artifact. There is no dataset where the true cross-architecture correspondence is known, so precision cannot be measured, only proxied.

Second: **the evaluation does not measure what it names.** Correlation matching scores co-activation; the claim being made is causal-role identity. Two features can correlate at $\rho=0.9$ because both track token frequency, while ablating them has orthogonal downstream effects.

Third: **cost.** Matching $10^6 \times 10^6$ latents requires either approximate nearest neighbours or dense $10^{12}$-entry score matrices, and each *causal* check costs a forward pass pair per intervention, so certifying $10^4$ pairs at 512 held-out prompts is $\sim10^7$ forward passes.

## 7. Current Research (as of 2026)

- **Crosscoders and model diffing** — Anthropic's interpretability team; extensions to cross-architecture diffing are the natural next step *(frontier — verify)*.
- **Attribution-based and transcoder-style dictionaries** replacing plain SAEs, which changes what "a feature" is and therefore what is being matched (Ameisen, Lindsey et al., 2025, circuit tracing).
- **SAE evaluation infrastructure** — SAEBench (Karvonen, Rager, Bloom et al., 2025) and Neuronpedia; a cross-model matching task is an obvious missing track.
- **Representational similarity, formalised** — Klabunde et al.'s survey and the "representational alignment" community (Sucholutsky et al., 2023) push toward measures with statistical tests rather than single scalars.
- **Open problems framing** — Sharkey et al. (2025) list cross-model universality as unresolved.

## 8. Concrete Next Experiment

**Planted-correspondence benchmark, then causal certification.**

- **Scale:** two 1.4B-parameter models on identical data, identical tokenizer, identical token order: one transformer (Pythia-1.4B recipe), one Mamba-style SSM. Train SAEs at $m=32{,}768$ ($\sim$16× expansion) on the mid-depth residual stream of each, matched $L_0 \approx 40$.
- **Ground truth arm:** train a third model — a transformer with the *same* architecture as $A$ but a different seed. The seed-pair gives an upper bound on achievable matching; a randomly permuted dictionary gives the floor.
- **Method arms:** (i) Hungarian on activation correlation; (ii) crosscoder trained jointly on $A$ and $B$; (iii) relative representations with 4,096 anchors.
- **Deciding number:** the **certified-pair rate** — the fraction of the top-1,000 proposed pairs with causal transfer $T_{ij} > 0.5$ on 512 held-out prompts, against a null of random pairs. If the cross-architecture rate is $\geq 50\%$ of the same-architecture-different-seed rate, cross-architecture correspondence is real and findable; if it is $\leq 10\%$, current dictionaries do not transport across architectures and the measurement variant is confirmed blocked.

Cost estimate: $\sim$3 model trainings, 3 SAE trainings, and $\approx 10^6$ intervened forward passes — days on 8 A100s, not months.

## 9. Key References

- **[Foundational]** Karel Lenc, Andrea Vedaldi. *Understanding Image Representations by Measuring Their Equivariance and Equivalence.* CVPR, 2015.
- **[Foundational]** Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, John Hopcroft. *Convergent Learning: Do Different Neural Networks Learn the Same Representations?* ICLR, 2016. — arXiv:1511.07543
- **[Foundational]** Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, Shan Carter. *Zoom In: An Introduction to Circuits.* Distill, 2020.
- **[SOTA]** Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. *Similarity of Neural Network Representations Revisited.* ICML, 2019. — arXiv:1905.00414
- **[SOTA]** Yamini Bansal, Preetum Nakkiran, Boaz Barak. *Revisiting Model Stitching to Compare Neural Representations.* NeurIPS, 2021. — arXiv:2106.07682
- **[SOTA]** Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey. *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* ICLR, 2024. — arXiv:2309.08600
- **[SOTA]** Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, Emanuele Rodolà. *Relative Representations Enable Zero-Shot Latent Space Communication.* ICLR, 2023. — arXiv:2209.15430
- **[SOTA]** Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, Joshua Batson, Chris Olah. *Sparse Crosscoders for Cross-Layer Features and Model Diffing.* Transformer Circuits Thread, 2024.
- **[SOTA]** Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola. *The Platonic Representation Hypothesis.* ICML, 2024. — arXiv:2405.07987
- **[Empirical]** Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, Dimitris Bertsimas. *Universal Neurons in GPT2 Language Models.* TMLR, 2024. — arXiv:2401.12181
- **[Critique]** Frances Ding, Jean-Stanislas Denain, Jacob Steinhardt. *Grounding Representation Similarity with Statistical Testing.* NeurIPS, 2021. — arXiv:2108.01661
- **[Critique]** MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, Eugene Belilovsky. *Reliability of CKA as a Similarity Measure in Deep Learning.* ICLR, 2023.
- **[Survey]** Max Klabunde, Tobias Schumacher, Markus Strohmaier, Florian Lemmerich. *Similarity of Neural Network Models: A Survey of Functional and Representational Measures.* ACM Computing Surveys, 2025.
- **[Survey]** Lee Sharkey et al. *Open Problems in Mechanistic Interpretability.* 2025. — arXiv:2501.16496

## 10. Worked Example

Take a single well-studied feature: the **induction** behaviour, "having seen `[A][B]` earlier, predict `[B]` after `[A]`".

In GPT-2 Small this is carried by a small set of induction heads at layers 5–6. In a Mamba-style SSM there are no heads; the analogous computation is spread over channels of the state-space recurrence.

Run the matcher naively:

1. Score 512 prompts with repeated random token pairs. GPT-2's induction head output projection direction $d^A$ fires on the second occurrence of `[B]`'s predecessor with mean activation, say, 3.2 vs. 0.1 baseline — a clean 32× contrast.
2. Compute $\rho$ between that activation trace and all 32,768 SSM SAE latents. Suppose the best match is $\rho = 0.71$.
3. Now check the null: shuffle the SSM latents' identity and re-run. Because *every* induction-relevant latent fires on exactly the same token positions — the repeated-token positions — the 20th-best match still scores $\rho = 0.63$.

The correlation signal is almost entirely positional coincidence, not feature identity. The gap between rank-1 and rank-20 is 0.08, well inside the seed-to-seed variation of SAE training.

Now apply the causal test. Ablate $d^A$ in GPT-2: induction accuracy on held-out repeats drops from 0.86 to 0.11. Ablate the rank-1 SSM latent: accuracy drops from 0.84 to 0.79. Transfer ratio $T \approx 0.1$.

The obstruction is visible: the correlational matcher confidently returns a pair, the causal test refutes it, and there is no ground truth telling us whether the failure is (a) the SSM genuinely distributing induction over many latents, (b) the SAE splitting one feature into many, or (c) the wrong layer being compared. All three predict the same measurement. Until an experiment separates them — which is what §8 is for — the matching is unfalsifiable.

*(Numbers in steps 1–3 are illustrative of the regime the experiment in §8 would measure, not reported results; the GPT-2 induction-head ablation effect is qualitatively established, the SSM comparison is not.)*

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*