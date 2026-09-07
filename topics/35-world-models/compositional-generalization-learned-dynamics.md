---
id: 35-world-models/compositional-generalization-learned-dynamics
title: "Compositional Generalization of Learned Dynamics"
topic: 35-world-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compositional Generalization of Learned Dynamics

> **Topic:** World Models & Planning · **ID:** `35-world-models/compositional-generalization-learned-dynamics` · **Status:** empirically-open

## 1. Problem Statement

A learned world model is trained on transitions drawn from a set of environment configurations. The question is whether it predicts correctly on configurations built from the *same* primitive parts in *unseen combinations* — three objects when it saw two, a ramp plus a fluid when it saw each alone, an object with mass $m_1$ and material $c_2$ when the training set paired $m_1$ only with $c_1$.

- **Input:** observation/state history $o_{\le t}$ and action $a_t$.
- **Output:** predicted next observation $\hat{o}_{t+1}$, or an $H$-step rollout $\hat{o}_{t+1:t+H}$.
- **Decision predicate:** does rollout error stay bounded on the combinatorially-held-out test set, at a level useful for control?

Three variants, of very different difficulty:

- **Measurement:** define a split of environment configurations that is genuinely combinatorial (no test factor combination appears in training) *and* a rollout-error metric whose degradation is attributable to composition rather than to distribution shift in pixel statistics, contact frequency, or episode length.
- **Method:** build an architecture or training recipe whose combinatorial-split error is within a stated factor of its i.i.d.-split error.
- **Theory:** state conditions on the data distribution and the model class under which zero training error implies bounded error on unseen factor combinations.

Solved means: for a stated primitive vocabulary, error on held-out combinations is within a small constant factor of i.i.d. error, and the result survives ablation of the shortcut hypotheses in §6.

## 2. Formal Setting

Let the environment be a controlled Markov process with state $s \in \mathcal{S}$, action $a \in \mathcal{A}$, transition kernel $T(s' \mid s, a)$. Assume the state factorizes into $K$ entities with attributes:
$$s = (e_1, \dots, e_K), \qquad e_k \in \mathcal{E} = \mathcal{Z}_1 \times \cdots \times \mathcal{Z}_M,$$
where $\mathcal{Z}_m$ is the $m$-th attribute vocabulary (shape, material, mass bin, initial velocity bin). A *configuration* is $c = (K, \{e_k\}, \text{scene params})$. The support of training configurations is $\mathcal{C}_{\text{tr}} \subseteq \mathcal{C}$; the test support is $\mathcal{C}_{\text{te}}$.

**Combinatorial split (measured).** Require *primitive coverage* and *combination disjointness*:
$$\bigcup_{c \in \mathcal{C}_{\text{te}}} \text{prims}(c) \subseteq \bigcup_{c \in \mathcal{C}_{\text{tr}}} \text{prims}(c), \qquad \mathcal{C}_{\text{tr}} \cap \mathcal{C}_{\text{te}} = \emptyset,$$
with $\text{prims}(c)$ the multiset of attribute values and pairwise interaction types present. In practice, generate $\mathcal{C}$ by a procedural generator and hold out a fixed sub-lattice of the attribute product.

**Error (measured).** For model $f_\theta$ and horizon $H$:
$$R_H(\mathcal{C}') = \mathbb{E}_{c \sim \mathcal{C}'}\ \mathbb{E}_{\tau \sim \pi, c}\ \frac{1}{H}\sum_{h=1}^{H} d\big(\hat{o}_{t+h}, o_{t+h}\big),$$
$d$ being per-particle/per-object position MSE for state models, LPIPS or $\ell_2$ for pixel models. Report the **compositionality gap**
$$G_H = \frac{R_H(\mathcal{C}_{\text{te}})}{R_H(\mathcal{C}_{\text{tr}}^{\text{held-out-seeds}})},$$
where the denominator is an i.i.d. split of *unseen seeds from seen configurations* — this is the control that separates composition failure from ordinary generalization failure. Also report the **task gap**: success rate of a fixed planner (e.g. CEM, 1000 samples, horizon 20) using the model, on training vs. held-out configurations.

**Assumptions, and which are violated.**
1. *State factorizes into entities with independent attributes.* Violated in pixel-space models, where entities are not given and slot assignment is unidentifiable.
2. *Dynamics are compositional*: $T$ decomposes into per-entity and pairwise terms. Violated by contact/friction, multi-body constraints, fluids, and any global field.
3. *Attribute values are marginally covered in training.* Usually true by construction; but *interaction-type* coverage (e.g. "sphere-on-ramp contact") is often silently violated.
4. *Error is comparable across splits.* Violated whenever held-out configurations have more objects — more objects means more contacts means higher error for reasons unrelated to composition.

## 3. State of the Art

**Structured state-space models (established).** Interaction Networks (Battaglia et al., NeurIPS 2016) and Graph Network Simulators (Sanchez-Gonzalez et al., ICML 2020) build the compositional prior into the architecture: per-node and per-edge functions shared across entities. GNS trained on scenes of order $10^3$ particles produces stable rollouts on scenes roughly an order of magnitude larger, and MeshGraphNets (Pfaff et al., ICLR 2021) generalize across mesh resolutions. This is *established* for **cardinality** generalization with ground-truth state.

**Attribute-recombination generalization (claimed but under-ablated).** Papers that report "compositional generalization" usually vary object count, not the attribute lattice. Held-out *attribute pairs* under matched contact statistics are rarely reported.

**Pixel-space video/world models (benchmark numbers only).** DreamerV3 (Hafner et al., Nature 2025) is SOTA for learned-model control breadth, but its evaluation is per-domain reward, not a combinatorial split. Large action-conditioned video world models (Genie-family, 2024–2025) demonstrate qualitative novel-scene rollout; no published combinatorial-split error numbers exist *(frontier — verify)*.

**Theory SOTA.** Wiedemer et al. (NeurIPS 2023) give sufficient conditions — compositional support plus a compositional decoder — under which a model that fits the training distribution provably extends to unseen combinations. This is for generative/autoencoding settings, not multi-step controlled dynamics.

## 4. What Is Known

- **Sequence models fail hard on combinatorial splits.** SCAN "add-jump" (Lake & Baroni, ICML 2018): best seq2seq reaches ~1% exact-match versus ~99% on a random split, at $10^4$-example scale. CFQ MCD splits (Keysers et al., ICLR 2020): LSTM+attention drops from ~98% (random split) to ~15% mean MCD accuracy, at $10^5$ examples. These are language-to-logical-form, not dynamics, but they establish that the gap is architectural, not a data-quantity artifact.
- **Structure buys cardinality, not attributes.** GNS-style models extrapolate from ~$10^3$ to ~$10^4$–$10^5$ particles with stable rollouts (ICML 2020), while pure CNN/RNN video models degrade quickly outside training object counts.
- **Physical prediction from vision lags state-based prediction.** Physion (Bear et al., NeurIPS 2021 D&B): human accuracy averages roughly 75% on the "will they contact" predicate; vision-based models sit well below humans, while a model given ground-truth object state is closest. Scale: ~$10^4$ trials per scenario.
- **Disentanglement is not identifiable without inductive bias or supervision** (Locatello et al., ICML 2019, best paper). Since factor recovery underpins attribute recombination, this bounds what unsupervised pixel world models can be *guaranteed* to compose.
- **Within-domain factor extrapolation already fails** for representation learners (Schott et al., ICLR 2022) — the gap is not specific to dynamics.

## 5. What Is Not Known

- **Empirically open.** No published measurement of $G_H$ for a frontier-scale action-conditioned video world model ($\ge 10^9$ parameters, $\ge 10^4$ hours of video) on a *procedurally-controlled* combinatorial split with matched contact statistics. The experiment is runnable today; nobody has run it at that scale with the right control arm.
- **Empirically open.** Whether $G_H$ decreases with model/data scale, stays flat, or is a scaling-immune architectural deficit. SCAN-style evidence suggests near-flat; no dynamics-domain scaling curve exists.
- **Theoretically open.** No theorem giving conditions on $T$ and $f_\theta$ under which $H$-step rollout error on unseen attribute combinations is bounded. Wiedemer et al. cover single-step generative composition; error compounding over $H$ steps is unaddressed.
- **Methodologically blocked.** There is no agreed definition of "primitive" for pixel-space dynamics. Without a slot/entity ground truth, "unseen combination" is not well posed, so a claimed combinatorial split cannot be audited.

## 6. Why It Is Hard

**The evaluation does not measure what it names — confounded measurement.** Reported "compositional generalization" for dynamics models almost always varies object count or scene size. Adding objects raises contact density, collision frequency, and chaotic divergence; rollout error rises for reasons that have nothing to do with recombining primitives. Unless test configurations are matched on contact count, Lyapunov-style divergence rate, and episode length, $G_H$ measures difficulty, not compositionality.

**Non-identifiability compounds it.** In pixel space the entity decomposition is latent and provably unidentifiable without bias or supervision (Locatello et al. 2019). A model may compose correctly with respect to *its* factorization while failing the human-labelled one, and no measurement currently distinguishes the two cases.

**Compounding error hides the effect.** Rollout MSE grows superlinearly in $H$ for chaotic systems, so at large $H$ both splits saturate and $G_H \to 1$ for trivial reasons; at small $H$ single-step smoothness masks the failure. The signal lives in a narrow $H$ band that has to be chosen per system.

## 7. Current Research (as of 2026)

- **Object-centric world models** — slot-based latents with per-slot dynamics, aiming to make the factorization explicit and hence auditable (groups at DeepMind, MPI-IS Tübingen, Vector/Mila).
- **Graph and mesh simulators for engineering domains** — DeepMind and academic CFD groups; the strongest existing evidence of dynamics composition, but with ground-truth state.
- **Frontier interactive video world models** (Genie-class, and open replications) — the open question is whether internet-scale video pretraining buys attribute recombination for free *(frontier — verify: no combinatorial-split numbers published)*.
- **Theory of compositional generalization** — extending Wiedemer-style sufficient conditions to sequential/controlled settings *(frontier — verify)*.
- **Benchmark construction** — procedural generators (Procgen-style, Cobbe et al., ICML 2020) repurposed to expose an explicit attribute lattice with declarable held-out cells.

## 8. Concrete Next Experiment

**Setup.** A procedural rigid-body/fluid simulator with an explicit lattice: $M=3$ attributes — shape $\in$ {sphere, cube, cylinder, cone}, material $\in$ {rigid, elastic, granular, viscous}, ramp angle $\in$ {0°, 15°, 30°, 45°} — giving 64 cells. Hold out 16 cells forming a Latin-square pattern so every attribute value appears in training but no held-out pair does. **Force matched statistics:** resample training and test episodes so that mean contact events per episode and mean kinetic energy match to within 5%.

**Scale.** Train three model families at three sizes each ($3\times10^7$, $3\times10^8$, $3\times10^9$ parameters) on $10^7$ transitions: (a) GNS with ground-truth state, (b) slot-based latent dynamics model, (c) monolithic action-conditioned video transformer.

**Control arm.** Same models, same budget, trained on an i.i.d. split covering all 64 cells with identical episode count — matched contact statistics, matched horizon.

**Deciding number.** $G_{20}$ = ratio of 20-step position error on held-out cells to i.i.d. control error, per family per size. Decision rule: if $G_{20} > 3$ at all three sizes for the video transformer and the slope $\partial \log G_{20} / \partial \log N_{\text{params}}$ is flat within $\pm 0.05$, compositional generalization is an architectural deficit that scale does not fix. If $G_{20}$ falls below 1.5 at $3\times10^9$, it is a data/scale problem.

## 9. Key References

- **[Foundational]** Battaglia, Pascanu, Lai, Rezende, Kavukcuoglu. *Interaction Networks for Learning about Objects, Relations and Physics.* NeurIPS, 2016. — arXiv:1612.00222
- **[Foundational]** Lake, Baroni. *Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks.* ICML, 2018. — arXiv:1711.00350
- **[Foundational]** Ha, Schmidhuber. *Recurrent World Models Facilitate Policy Evolution.* NeurIPS, 2018. — arXiv:1803.10122
- **[SOTA]** Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec, Battaglia. *Learning to Simulate Complex Physics with Graph Networks.* ICML, 2020. — arXiv:2002.09405
- **[SOTA]** Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia. *Learning Mesh-Based Simulation with Graph Networks.* ICLR, 2021. — arXiv:2010.03409
- **[SOTA]** Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Control Tasks through World Models.* Nature, 2025. (DreamerV3; preprint arXiv:2301.04104)
- **[Benchmark]** Keysers et al. *Measuring Compositional Generalization: A Comprehensive Method on Realistic Data.* ICLR, 2020. — arXiv:1912.09713
- **[Benchmark]** Bear et al. *Physion: Evaluating Physical Prediction from Vision in Humans and Machines.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2106.08261
- **[Benchmark]** Yi, Gan, Li, Kohli, Wu, Torralba, Tenenbaum. *CLEVRER: Collision Events for Video Representation and Reasoning.* ICLR, 2020. — arXiv:1910.01442
- **[Benchmark]** Cobbe, Hesse, Hilton, Schulman. *Leveraging Procedural Generation to Benchmark Reinforcement Learning.* ICML, 2020. — arXiv:1912.01588
- **[Theory]** Wiedemer, Mahner, Brady, von Kügelgen, Brendel. *Compositional Generalization from First Principles.* NeurIPS, 2023. — arXiv:2307.05596
- **[Theory]** Locatello et al. *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML, 2019. — arXiv:1811.12359
- **[Survey]** Hupkes, Dankers, Mul, Bruni. *Compositionality Decomposed: How do Neural Networks Generalise?* JAIR, 2020. — arXiv:1908.08351
- **[Related]** Schott et al. *Visual Representation Learning Does Not Generalize Strongly Within the Same Domain.* ICLR, 2022. — arXiv:2107.08221

## 10. Worked Example

Two attributes, two values each: material $\in$ {rigid, elastic}, ramp $\in$ {flat, 30°}. Training covers three cells: (rigid, flat), (rigid, 30°), (elastic, flat). Held out: (elastic, 30°). Each cell gets 250k transitions; a 300M-parameter latent video model is trained to convergence.

Measured 20-step centroid error (arbitrary length units, mean over 2k episodes):

| Split | Error | Contacts/episode |
|---|---|---|
| i.i.d. seeds, seen cells | 0.041 | 3.1 |
| (elastic, 30°) held out | 0.163 | 5.4 |

Naive read: $G_{20} = 0.163/0.041 = 4.0$ — a large compositional failure.

Now the control. Retrain on all four cells, same budget. Error on (elastic, 30°) is 0.118, not 0.041 — that cell is intrinsically harder, because elastic bodies on a slope bounce more (5.4 contacts versus 3.1). The composition-attributable ratio is $0.163/0.118 = 1.38$, not 4.0.

This is the obstruction in one number. Nearly two-thirds of the apparent gap ($4.0 \to 1.38$) is intrinsic cell difficulty, not failure to recombine primitives. Any paper that reports the 4.0 without the all-cells control arm — and most do — is reporting difficulty shift under the name "compositional generalization". The residual 1.38 is the real quantity of interest, it is small enough to be confused with seed noise at 2k episodes, and pinning it down requires either far more episodes or contact-matched resampling. That is why the problem is empirically open rather than settled: the experiment is cheap, but the control arm and the matching are almost never run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*