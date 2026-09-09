---
id: 20-interpretability/feature-universality-across-seeds
title: "Universality of Features Across Random Seeds"
topic: 20-interpretability
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Universality of Features Across Random Seeds

> **Topic:** Interpretability · **ID:** `20-interpretability/feature-universality-across-seeds` · **Status:** empirically-open

## 1. Problem Statement

Two networks with identical architecture, data, and hyperparameters, differing only in random seed (init, data order, dropout, kernel nondeterminism), converge to different parameter vectors. The universality question: **do they represent the same features?**

Three variants, routinely conflated:

- **Measurement variant.** Given two trained models, output a matching between their features and a universality rate $U \in [0,1]$ — the fraction of features in model $A$ with a counterpart in $B$ above a stated similarity threshold, scored against a stated null. Open because no matching procedure has an agreed null model.
- **Method variant.** Produce a decomposition (neurons, SAE latents, directions) whose $U$ is high *and* whose units are causally interchangeable: patching $B$'s matched feature into $A$ reproduces $A$'s own feature's causal effect.
- **Theory variant.** Prove or refute that the SGD posterior over solutions concentrates, up to a symmetry group $G$ (permutation, scaling, rotation within degenerate subspaces), on a single functional decomposition.

Solving it means: a procedure that, for a fixed model class and data distribution, reports $U$ with a calibrated null, and a demonstration that the features it calls "shared" are causally substitutable across seeds.

## 2. Formal Setting

Let $f_\theta: \mathcal{X} \to \mathbb{R}^{|V|}$ be a transformer, $\theta \sim \mathcal{A}(s)$ the output of training algorithm $\mathcal{A}$ under seed $s$. Fix a site $\ell$ (residual stream, MLP hidden layer) with activation map $h^\ell_\theta: \mathcal{X} \to \mathbb{R}^{d}$.

**Feature dictionary.** A decomposition is a matrix $D \in \mathbb{R}^{d \times m}$ with columns $d_i$ (unit norm) and an encoder $a: \mathbb{R}^d \to \mathbb{R}^m_{\ge 0}$, so $h \approx b + \sum_i a_i(h)\, d_i$. For SAEs, $a(h) = \sigma(W_e(h-b) + b_e)$ trained to minimise $\mathbb{E}\|h - \hat h\|_2^2 + \lambda \|a\|_1$ (or under TopK/JumpReLU, an explicit $L_0$).

**Two similarity statistics, both measured on a held-out corpus $\mathcal{D}$ of $N$ tokens.**

$$s^{\text{geo}}_{ij} = \langle d^A_i, d^B_j \rangle, \qquad s^{\text{fun}}_{ij} = \frac{\mathrm{Cov}_{x\sim\mathcal{D}}\!\left[a^A_i(x), a^B_j(x)\right]}{\sigma_i^A \sigma_j^B}$$

$s^{\text{geo}}$ requires $A$ and $B$ to share a basis — true for two SAEs on the *same* model, false across model seeds. $s^{\text{fun}}$ is basis-free but needs paired token streams and is dominated by the base rate: features firing on $10^{-4}$ of tokens have correlation estimates with standard error $\approx (N p)^{-1/2}$, so $N \ge 10^8$ tokens for stable tail estimates.

**Universality rate.** With matching $\pi$ (Hungarian on $-s$, or greedy max) and threshold $\tau$:

$$U(\tau) = \frac{1}{m}\left|\{i : s_{i\pi(i)} \ge \tau\}\right|, \qquad U_{\text{net}}(\tau) = U(\tau) - \mathbb{E}\left[U^{\text{null}}(\tau)\right]$$

The null $U^{\text{null}}$ is where the problem lives (§10).

**Causal universality.** For feature $i$ in $A$ matched to $j$ in $B$, with $E_i^A(x) = f_{\theta_A}(x) - f_{\theta_A}(x \mid a_i \!\leftarrow\! 0)$ the ablation effect in logit space, define transfer fidelity

$$\rho_{ij} = \frac{\langle E_i^A, E_j^B\rangle_{\mathcal{D}}}{\|E_i^A\|\,\|E_j^B\|}.$$

**Assumptions, and which are violated.** (i) *Same data distribution* — holds by construction, but data *order* is itself part of the seed and is rarely varied separately; (ii) *one-to-one correspondence* — violated: feature splitting means one $A$-feature maps to $k$ $B$-features as dictionary size grows (Bricken et al. 2023); (iii) *linear, sparse features* — assumed, not established; circular/multi-dimensional features exist (Engels et al. 2024); (iv) *symmetry group is permutation only* — violated wherever activations have degenerate subspaces, so the true group includes rotations and the matching is not identifiable.

## 3. State of the Art

**Empirical SOTA (established).**
- *Universal neurons.* Gurnee et al., "Universal Neurons in GPT2 Language Models," TMLR 2024: five GPT-2 small models, same data, different seeds; only **1–5% of neurons** are universal under a pairwise activation-correlation criterion. Universal neurons are disproportionately interpretable (unigram, position, syntax, entropy-modulating neurons). This is a reproduced, ablated result.
- *Convergent learning.* Li et al., ICLR 2016: in AlexNet conv1, most filters have a high-correlation partner across seeds; a minority ("rare" features) have none.
- *Representational similarity.* Kornblith et al., ICML 2019: linear CKA reliably matches corresponding layers across seeds where CCA/SVCCA fail. Establishes *layerwise* correspondence, not *feature-level* correspondence.
- *Algorithmic universality.* Chughtai, Chan, Nanda, ICML 2023: networks trained on group composition consistently implement the same algorithm via irreducible representations — but which irreps, and their neuron-basis embedding, vary by seed. Universality holds at the algorithm level and fails at the unit level.

**Claimed but unablated.**
- SAE feature universality across seeds. Bricken et al. (Transformer Circuits, 2023) report that most features from two SAEs trained with different seeds on the same one-layer model have close analogues, with a long unmatched tail. Paulo & Belrose, "Sparse Autoencoders Trained on the Same Data Learn Different Features" (2025, arXiv:2501.16615), report the opposite emphasis: a large fraction of SAE latents have no cross-seed counterpart. Both are single-setting benchmark numbers under different thresholds and different matching rules; neither reports a matched null.
- Lan et al., "Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models" (2024, arXiv:2410.06981): SVCCA/RSA similarity between SAE feature *spaces* across different LLMs. A space-level statistic, not a per-feature matching.
- Huh et al., "The Platonic Representation Hypothesis," ICML 2024: convergence of representations across models and modalities. Kernel-alignment evidence; does not claim feature-level identity.

**Theory SOTA.** Entezari et al. (ICLR 2022) conjecture that SGD solutions are permutation-equivalent modulo a single basin. Ainsworth, Hayase, Srinivasa, "Git Re-Basin" (ICLR 2023), drive the loss barrier to near zero for wide ResNets on CIFAR-10 after permutation alignment; barriers persist at ImageNet scale and for narrow models. No theorem covers transformers.

## 4. What Is Known

- **1–5%** neuron universality, GPT-2 small (124M), 5 seeds, correlation threshold on 100M+ tokens (Gurnee et al. 2024). Scale: small.
- **Permutation alignment removes most of the barrier** for ResNet-20×16 on CIFAR-10; test-loss barrier drops from $\gtrsim 0.5$ to $\approx 0$ (Ainsworth et al. 2023). Width matters: narrow models retain barriers.
- **Linear mode connectivity is decided early.** Frankle et al., ICML 2020: after roughly 1–5% of training, two branches with different SGD noise are linearly connected. So seed-induced divergence is concentrated in a short early window.
- **CKA is not a safe adjudicator.** Ding, Denain, Steinhardt (NeurIPS 2021) and Davari et al. (ICLR 2023) show CKA is dominated by a few high-variance directions and can be driven to arbitrary values by translation/outlier manipulation.
- **Feature splitting is real.** Bricken et al. 2023: increasing dictionary size from 512 to 16,384 on a one-layer model splits a single "base64" feature into many specialised children — so $U$ is a function of $m$, not a property of the model.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted null distribution for $s^{\text{geo}}$ or $s^{\text{fun}}$. Every reported $U$ is a raw match rate at a hand-picked $\tau$, so cross-paper numbers (Bricken vs. Paulo & Belrose) are not comparable. Nor is there an accepted treatment of one-to-many matching under feature splitting.
- **Empirically open.** Nobody has run the seed sweep that separates *model* nondeterminism from *dictionary-learning* nondeterminism: $\ge 8$ model seeds × $\ge 3$ SAE seeds each, at $\ge 1$B parameters, with causal transfer fidelity $\rho$ reported alongside $U$. Cost is the only barrier.
- **Theoretically open.** No proof that the SGD posterior for transformers concentrates modulo permutation. No characterisation of which features are seed-stable — the "1–5%" set has no predictive theory.

## 6. Why It Is Hard

**Non-identifiability plus an uncalibrated null.** Where activations concentrate in a low effective dimension, high cosine similarity between dictionary directions is *forced* by geometry, not by shared computation. The null $\mathbb{E}[U^{\text{null}}]$ must be computed in the effective dimension of the activation distribution, not the ambient dimension $d$ — and effective dimension is itself estimator-dependent (§10). Consequence: the same data support "SAE features are universal" and "SAE features are seed-specific."

Compounding: (a) feature splitting makes $U$ depend on $m$, so $U$ is a property of the (model, dictionary size, sparsity) triple; (b) the SAE training objective is non-convex and its own seed variance is unmeasured as a control, so $1 - U$ mixes model divergence with dictionary-learning noise; (c) causal ground truth is absent — there is no independent labelling of "the same feature," so every evaluation is circular unless it uses causal transfer.

## 7. Current Research (as of 2026)

- **Neuron-level universality at scale**, extending Gurnee et al. to multi-billion-parameter seed families — blocked on the absence of public multi-seed pretraining runs above ~1B (Pythia and OLMo release limited seed variation).
- **Cross-model SAE alignment** via Procrustes/CCA on decoder matrices (Lan et al. 2024 line, Oxford/Torr and Krueger groups). *(frontier — verify)*
- **Crosscoders** — dictionaries trained jointly on activations from several models to force a shared latent basis (Anthropic, 2024–2025). Directly targets the matching problem but makes universality partly definitional. *(frontier — verify)*
- **Seed-robustness of circuits** rather than features: whether IOI-style circuits (Wang et al., ICLR 2023) recur across seeds. *(frontier — verify)*
- **Permutation-symmetry theory** applied to attention heads and MLP neurons jointly, extending Git Re-Basin to transformer blocks.

## 8. Concrete Next Experiment

**Scale.** Pretrain $8$ models, 410M params (Pythia architecture), on the same 20B-token corpus. Four seeds vary init + data order; four vary init only (fixed order). Train $3$ SAEs per model at the layer-12 residual stream, $m = 32{,}768$, TopK $k=32$, each with a distinct SAE seed. Cost: roughly 8 × 400 A100-hours pretraining plus 24 SAE runs — under $5\times10^3$ GPU-hours.

**Arms.**
1. *Treatment:* SAE from model $A$ vs SAE from model $B$ (different model seeds), matched by $s^{\text{fun}}$ on $2\times10^8$ held-out tokens.
2. *Control (upper bound):* two SAEs with different SAE seeds on the **same** model. This isolates dictionary-learning noise.
3. *Control (null):* SAE on model $A$ vs SAE trained on a randomly-initialised model's activations at the same layer, plus a rotation-shuffled null in the measured effective dimension.

**Deciding number.**
$$\Delta = U_{\text{same-model}}(\tau^\star) - U_{\text{diff-model}}(\tau^\star)$$
with $\tau^\star$ fixed by requiring the arm-3 null to give $U^{\text{null}} = 0.01$. **If $\Delta < 0.05$, seed-level model nondeterminism contributes essentially nothing beyond SAE training noise, and features are universal in the operational sense. If $\Delta > 0.25$, they are not.** Report mean causal transfer fidelity $\bar\rho$ over matched pairs as the secondary check; $U$ high with $\bar\rho < 0.5$ means the matching is geometric coincidence.

## 9. Key References

- **[Foundational]** Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, John Hopcroft. *Convergent Learning: Do Different Neural Networks Learn the Same Representations?* ICLR, 2016. — arXiv:1511.07543
- **[Foundational]** Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, Shan Carter. *Zoom In: An Introduction to Circuits.* Distill, 2020.
- **[SOTA]** Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, Dimitris Bertsimas. *Universal Neurons in GPT2 Language Models.* TMLR, 2024. — arXiv:2401.12181
- **[SOTA]** Bilal Chughtai, Lawrence Chan, Neel Nanda. *A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations.* ICML, 2023. — arXiv:2302.03025
- **[SOTA]** Trenton Bricken et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread, 2023.
- **[SOTA]** Gonçalo Paulo, Nora Belrose. *Sparse Autoencoders Trained on the Same Data Learn Different Features.* 2025. — arXiv:2501.16615
- **[SOTA]** Samuel Ainsworth, Jonathan Hayase, Siddhartha Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR, 2023. — arXiv:2209.04836
- **[Method]** Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. *Similarity of Neural Network Representations Revisited.* ICML, 2019. — arXiv:1905.00414
- **[Critique]** Frances Ding, Jean-Stanislas Denain, Jacob Steinhardt. *Grounding Representation Similarity with Statistical Testing.* NeurIPS, 2021.
- **[Survey]** Max Klabunde, Tobias Schumacher, Markus Strohmaier, Florian Lemmerich. *Similarity of Neural Network Models: A Survey of Functional and Representational Measures.* 2023. — arXiv:2305.06329

## 10. Worked Example

Take two SAEs, $m = 32{,}768$ latents, decoder directions in $d = 768$. A feature is called "shared" if its best cosine match exceeds $\tau = 0.7$ — the threshold used informally across the SAE literature.

**Ambient-dimension null.** For a unit vector $u$ and $m$ i.i.d. uniform unit vectors in $\mathbb{R}^d$, the max cosine concentrates at

$$\mathbb{E}\!\left[\max_j \langle u, v_j\rangle\right] \approx \sqrt{\frac{2\ln m}{d}} = \sqrt{\frac{2 \times 10.40}{768}} = 0.165.$$

At this null, an observed $U(0.7) = 0.62$ looks decisive: $0.70 \gg 0.165$.

**Effective-dimension null.** Both SAEs see the same activation covariance $\Sigma$. Residual-stream activations are anisotropic; participation ratio $\mathrm{PR} = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$ is commonly one to two orders below $d$. Substituting $d_{\text{eff}} = \mathrm{PR}$:

| $d_{\text{eff}}$ | null $\mathbb{E}[\max\cos]$ | verdict at $\tau=0.7$ |
|---|---|---|
| 768 | 0.165 | strong evidence of sharing |
| 128 | 0.403 | weak evidence |
| 64 | 0.570 | near chance |
| 30 | 0.833 | **$\tau$ is below chance** |

**The obstruction, made visible.** The reported universality rate flips sign as a function of an estimator ($\mathrm{PR}$) that no published SAE-universality result reports. Two SAEs trained on white noise with the same covariance would produce $U(0.7)$ near 1 at $d_{\text{eff}} = 30$ while sharing no computation at all. This is why $\Delta$ in §8 — a *difference* between arms that share the same activation geometry — is the decidable quantity, and raw $U$ is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*