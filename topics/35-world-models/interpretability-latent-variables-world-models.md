---
id: 35-world-models/interpretability-latent-variables-world-models
title: "Interpretability of Latent Variables in World Models"
topic: 35-world-models
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Interpretability of Latent Variables in World Models

> **Topic:** World Models & Planning · **ID:** `35-world-models/interpretability-latent-variables-world-models` · **Status:** methodologically-blocked

## 1. Problem Statement

A learned world model compresses observations into a latent state $z_t$ and predicts forward in that state. The question: **does $z_t$ carry identifiable, human-nameable variables of the environment (position, occlusion, object identity, agent intent, physical law), and can we certify that a claimed correspondence is real rather than an artefact of the decoder we used to read it out?**

Three variants, of very different difficulty:

- **Measurement.** Given a trained model and a candidate ground-truth variable $g$, produce a score that is high iff the model *uses* a representation of $g$ in its own dynamics — not merely that $g$ is linearly decodable from $z_t$. This variant is the blocked one: no accepted score exists.
- **Method.** Train world models whose latents are identifiable up to a known equivalence class (permutation, elementwise reparametrisation) by construction, via sparsity, interventions, or temporal structure.
- **Theory.** Characterise which environment factors are recoverable from observation-action sequences alone, and under which assumptions the recovery is unique.

Solving it means: a procedure that, applied to a model, returns a set of latent directions with named referents and a falsifiable guarantee — intervening on that direction changes the model's rollout in exactly the way changing the referent changes the environment, and does not change anything else.

## 2. Formal Setting

Environment: a POMDP $(\mathcal{S},\mathcal{A},\mathcal{O},T,\Omega,\gamma)$ with true state $s_t\in\mathcal{S}$ and observation $o_t\sim\Omega(\cdot\mid s_t)$. A ground-truth factorisation $s_t=(g_t^1,\dots,g_t^K)$ is assumed available only in simulators.

Model: encoder $q_\phi(z_t\mid o_{\le t},a_{<t})$, latent $z_t\in\mathbb{R}^d$, transition $p_\theta(z_{t+1}\mid z_t,a_t)$, decoder $p_\theta(o_t\mid z_t)$, trained on the ELBO
$$\mathcal{L}(\theta,\phi)=\mathbb{E}_q\Big[\sum_t \log p_\theta(o_t\mid z_t)-\beta\,\mathrm{KL}\big(q_\phi(z_t\mid\cdot)\,\|\,p_\theta(z_t\mid z_{t-1},a_{t-1})\big)\Big].$$

**Quantities, as measured.**

- *Probe accuracy.* $\mathrm{Acc}(f)=\mathbb{E}[\mathbf{1}\{f(z_t)=g_t^k\}]$ over a held-out trajectory set, $f$ from a fixed class $\mathcal{F}$ (linear, or MLP with stated width). Measured on $\ge 10^5$ held-out timesteps; report per-class, since board/grid targets are heavily imbalanced.
- *Probe selectivity* (Hewitt & Liang, EMNLP 2019). $\mathrm{Sel}(f)=\mathrm{Acc}(f)-\mathrm{Acc}_{\text{ctrl}}(f)$, where the control task relabels each $g^k$ value with a fixed random label. Non-trivial only if $\mathcal{F}$ is held constant across arms.
- *Causal use.* Pick a direction/subspace $P$. Patch $z_t\mapsto z_t+\alpha(P v'-Pv)$ to set the read-out of $g^k$ to a counterfactual value $g'^k$, roll the model forward $H$ steps, and compare to the simulator rolled forward from the intervened true state:
$$\mathrm{IIA}=\mathbb{E}\big[\mathbf{1}\{\text{model rollout} \equiv \text{simulator rollout under } g^k\!\to\! g'^k\}\big],$$
interchange-intervention accuracy in the sense of Geiger et al. (NeurIPS 2021; CLeaR 2024).
- *Specificity.* $\Delta_{\neg k}$ = change in decoded values of all $g^{j\neq k}$ under the same patch. A direction is "the $g^k$ direction" only if $\mathrm{IIA}$ is high **and** $\Delta_{\neg k}\approx 0$.
- *Identifiability.* Latents $z$ and $\tilde z$ are equivalent if $\tilde z=h(z)$ with $h$ a permutation composed with elementwise diffeomorphisms. Success is recovery up to $\sim$.

**Assumptions, and which fail.**

1. *A ground-truth factorisation exists and is known.* Holds in Atari/MuJoCo/Othello; fails for video, robotics from pixels, and language-conditioned models — the main setting of interest. **Violated in practice.**
2. *The probe class $\mathcal{F}$ matches the model's own read-out.* Unverified; a nonlinear probe can extract information the network never uses (Belinkov, *Computational Linguistics* 2022). **Routinely violated.**
3. *Latent factors are marginally independent.* False in almost every environment — position and velocity, object identity and appearance are coupled. **Violated.**
4. *The intervened latent is on-distribution.* Patching moves $z$ off the training manifold, so a low $\mathrm{IIA}$ may indicate a broken model rather than an absent variable. **Violated by construction.**

## 3. State of the Art

**Theory SOTA (established).** Locatello et al. (ICML 2019, best paper) prove unsupervised disentanglement is impossible without inductive bias: for any generative model with independent latents there exist infinitely many entangled $\tilde z$ with the same marginal — a corollary of Hyvärinen & Pajunen's nonlinear-ICA non-uniqueness (*Neural Networks*, 1999). Identifiability returns under auxiliary structure: iVAE (Khemakhem et al., AISTATS 2020) with an observed auxiliary variable; temporal contrastive learning (Hyvärinen & Morioka, NeurIPS 2016); interventions (Ahuja et al., ICML 2023; Brehmer et al., NeurIPS 2022); temporal interventions with known targets, CITRIS (Lippe et al., ICML 2022). These are genuine theorems with stated conditions, not benchmark claims.

**Empirical SOTA (established).** Linear probes recover board state from sequence models trained only on move strings: Othello-GPT (Li et al., ICLR 2023) and its linear reframing (Nanda et al., BlackboxNLP 2023); chess board state and player skill (Karvonen, COLM 2024). Vafa et al. (NeurIPS 2024) give the first non-probe measurement — Myhill–Nerode-based sequence-compression and distinction metrics — and show a model can have near-perfect next-token accuracy while its recovered map is not a valid map.

**Claimed but unablated.** That sparse autoencoders (Cunningham/Huben et al., ICLR 2024; Templeton et al., Anthropic 2024) recover the model's *own* features rather than a sparse basis of the activation cloud. Counter-evidence is now on the record: SAEs trained on the same data with different seeds learn substantially different feature sets (Paulo & Belrose, 2025), SAEs produce interpretable-looking features on *randomly initialised* transformers (Heap et al., 2025), and SAE features do not beat simple baselines on sparse probing (Kantamneni et al., 2025). Also unablated: the claim that DreamerV3-class latents (Hafner et al., *Nature* 2025) contain object-factored structure — the reported result is task return, not a representation measurement.

## 4. What Is Known

- **Numbers, Othello-GPT (8 layers, ~25M params, 20M synthetic games).** Li et al. report nonlinear-probe error $\approx 1.7\%$ vs linear-probe error $\approx 20.4\%$ for absolute black/white board state; Nanda et al. re-coordinatise to *mine/theirs* and get linear error $\approx 1\%$. Same network, same information — the difficulty of the read-out was a property of the chosen coordinate frame, not of the model. This is the single most instructive result in the area.
- **Chess (Karvonen, COLM 2024, ~25M-param GPT on PGN).** Linear board-state probes exceed 99% square accuracy; activation patching along a "skill" direction changes played move quality measurably.
- **Locatello et al.:** 12,800 models across 6 unsupervised methods, 6 datasets, 6 metrics. Disentanglement scores are uncorrelated with downstream task performance, and random seed accounts for more variance than method choice.
- **Vafa et al. (2024):** on New York taxi trajectories, a transformer with $>$95% next-turn validity yields a reconstructed street map containing streets that do not exist; small detours collapse route validity. High predictive accuracy does not imply a coherent latent map.
- **Vafa et al. (ICML 2025):** a model trained to fit orbital trajectories to high accuracy, when probed for the force law by inductive-bias transfer, does not encode Newtonian gravitation.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no accepted, sound definition of "latent $z$ represents variable $g$". Probe accuracy is confounded by probe capacity; SAE features are seed-dependent and fire on random networks; patching-based $\mathrm{IIA}$ is confounded by off-manifold inputs. No score today separates *decodable* from *used*, and no benchmark scores existing methods against a known-answer key at scale.
- **Theoretically open.** Whether identifiability up to permutation is achievable from observation-action sequences alone under sparse-mechanism assumptions with *unknown* intervention targets and dependent latents. CITRIS assumes known targets; the unknown-target, non-independent case has no theorem either way.
- **Empirically open.** Whether identifiability-motivated objectives (mechanism sparsity, temporal contrastive) survive at video-world-model scale ($10^9$ params, Genie-class; Bruce et al., ICML 2024). Runnable; expensive; unrun.

## 6. Why It Is Hard

Three specific obstructions, in order of bite.

1. **Non-identifiability is a theorem, not a difficulty.** By Hyvärinen–Pajunen/Locatello, the loss cannot distinguish $z$ from $h(z)$ for measure-preserving $h$. Any interpretation is a claim about the *inductive bias of the optimiser*, not about the objective.
2. **The measurement is confounded.** Probe accuracy grows with probe capacity independently of the model. The Othello case shows the flip side: a change of coordinate frame moved linear error from 20% to 1% with the network fixed. So "is it linearly represented?" is not a question about the network alone; it is a question about network-plus-frame, and the frame is chosen by the interpreter.
3. **Absent ground truth off-simulator.** Every clean result above is on a synthetic environment with an enumerable state (Othello, chess, grid mazes). For video and robot world models — the deployment case — no $g^k$ key exists, so the measurement cannot even be scored.

## 7. Current Research (as of 2026)

- **Causal abstraction / distributed alignment search** (Stanford, Geiger, Potts, Icard) — $\mathrm{IIA}$ under learned orthogonal rotations; the most principled available criterion, still limited by off-manifold patching.
- **Identifiable causal representation learning** (Amsterdam/Lippe & Gavves; Mila/Lachapelle & Lacoste-Julien; MPI-IS/von Kügelgen, Schölkopf) — sparse-mechanism and intervention-based identifiability, moving toward unknown targets.
- **SAE reassessment** (DeepMind, EleutherAI, MIT/Tegmark) — the 2025 negative results have shifted the field from "scale SAEs" to "validate SAEs against baselines". *(frontier — verify current consensus.)*
- **World-model evaluation without probes** (Harvard/Vafa, Rambachan, Mullainathan) — Myhill–Nerode and inductive-bias probes as model-free coherence tests.
- **Interpretability of large video world models** (DeepMind Genie line, and open replications) — largely aspirational; no published latent-identifiability measurement at that scale. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Does any current interpretability method recover latent variables that the model actually uses, when the answer key is known and the confounds are controlled?

**Scale.** Train four world models on a procedurally generated 2D physics environment with $K=8$ enumerable factors (agent $x,y$; two object positions; object colour; gravity sign; friction coefficient; goal index), $10^8$ frames, $\sim$300M-parameter recurrent latent model (DreamerV3 recipe). Cheap: order 1–2 GPU-weeks per arm on 8×H100.

**Arms.**
- A: standard RSSM.
- B: RSSM + mechanism-sparsity regulariser.
- C: RSSM + observed intervention targets (CITRIS-style) — *upper-bound arm*.
- D (**control arm**): randomly initialised, untrained RSSM of identical architecture, plus a probe-capacity-matched control task in the Hewitt–Liang sense.

**Read-outs.** For each of {linear probe, MLP probe, SAE feature, DAS rotation}, find the best subspace for each $g^k$, then compute $\mathrm{IIA}$ and specificity $\Delta_{\neg k}$ against the simulator under 1,000 counterfactual interventions per factor.

**The deciding number.** $\bar{\mathrm{IIA}}_{\text{trained}}-\bar{\mathrm{IIA}}_{\text{random}}$, averaged over the 8 factors at specificity $\Delta_{\neg k}<0.05$. If this gap is $<0.10$ for SAEs and probes while arm C exceeds 0.60, the measurement problem is confirmed as the binding constraint and unsupervised read-outs should not be used to make representational claims. If probes on arm A reach $\ge 0.60$ at that specificity, the field has a working measurement and the problem downgrades from methodologically blocked to empirically open.

## 9. Key References

- **[Foundational]** Ha, D., Schmidhuber, J. *World Models.* NeurIPS, 2018. — arXiv:1803.10122
- **[Foundational]** Hyvärinen, A., Pajunen, P. *Nonlinear independent component analysis: Existence and uniqueness results.* Neural Networks, 1999.
- **[Foundational]** Locatello, F., Bauer, S., Lucic, M., Rätsch, G., Gelly, S., Schölkopf, B., Bachem, O. *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML, 2019. — arXiv:1811.12359
- **[Theory]** Khemakhem, I., Kingma, D. P., Monti, R. P., Hyvärinen, A. *Variational Autoencoders and Nonlinear ICA: A Unifying Framework.* AISTATS, 2020. — arXiv:1907.04809
- **[Theory]** Lippe, P., Magliacane, S., Löwe, S., Asano, Y. M., Cohen, T., Gavves, E. *CITRIS: Causal Identifiability from Temporal Intervened Sequences.* ICML, 2022.
- **[Theory]** Geiger, A., Lu, H., Icard, T., Potts, C. *Causal Abstractions of Neural Networks.* NeurIPS, 2021.
- **[SOTA]** Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., Wattenberg, M. *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task.* ICLR, 2023. — arXiv:2210.13382
- **[SOTA]** Nanda, N., Lee, A., Wattenberg, M. *Emergent Linear Representations in World Models of Self-Supervised Sequence Models.* BlackboxNLP @ EMNLP, 2023. — arXiv:2309.00941
- **[SOTA]** Vafa, K., Chen, J. Y., Rambachan, A., Kleinberg, J., Mullainathan, S. *Evaluating the World Model Implicit in a Generative Model.* NeurIPS, 2024. — arXiv:2406.03689
- **[SOTA]** Karvonen, A. *Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.* COLM, 2024.
- **[SOTA]** Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. *Mastering diverse control tasks through world models.* Nature, 2025.
- **[Method]** Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L. *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* ICLR, 2024. — arXiv:2309.08600
- **[Method]** Bruce, J. et al. *Genie: Generative Interactive Environments.* ICML, 2024. — arXiv:2402.15391
- **[Critique]** Hewitt, J., Liang, P. *Designing and Interpreting Probes with Control Tasks.* EMNLP, 2019.
- **[Survey]** Belinkov, Y. *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics, 2022.
- **[Survey]** Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., Bengio, Y. *Toward Causal Representation Learning.* Proceedings of the IEEE, 2021.

## 10. Worked Example

Take Othello-GPT: 8 layers, ~25M parameters, trained only on legal move sequences. Ground truth $g_t$ = 64 squares, each in {empty, black, white}.

- Frame 1, absolute colour. Linear probe on layer-6 residual stream: reported error $\approx 20.4\%$ over 60 squares. Conclusion an interpreter would draw: "the board is not linearly represented."
- Frame 2, relative colour (mine/theirs, i.e. relabel by side to move). Same activations, same probe class, same data. Reported error $\approx 1\%$. Conclusion: "the board is linearly represented, cleanly."

Arithmetic that makes the obstruction visible. Nothing about the network changed between the two rows. The measured quantity moved by a factor of $\sim 20$ in error. So the estimator
$$\hat{R}(g) = \max_{f\in\mathcal{F}} \mathrm{Acc}(f(z);g)$$
is not a function of $(z,g)$ alone — it is a function of $(z, \psi(g))$ for an interpreter-chosen relabelling $\psi$. Since $\psi$ ranges over a combinatorially large set ($3^{64}$ possible square-value recodings, before considering nonlinear factor changes), $\hat R$ has no well-defined maximum an experimenter can certify having reached. A negative probe result therefore carries almost no information: it is always consistent with "wrong frame".

Now the second half. Li et al. show the representation is *causal* — patching the probe direction to flip a square changes the model's legal-move predictions in the expected way. That is the right kind of evidence. But the patch moves $z$ to a state the encoder never produced, and there is no control arm establishing what $\mathrm{IIA}$ an untrained network of the same architecture would score under the same patching procedure. Both halves of the standard argument — decodability and patchability — are missing a baseline. That is why the status here is **methodologically blocked** and not merely open: the field's headline positive result is real, and we still cannot say by how much it beats chance.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*