---
id: 07-embeddings/curvature-selection-mixed-geometry
title: "Curvature Selection for Mixed-Geometry Embeddings"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Curvature Selection for Mixed-Geometry Embeddings

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/curvature-selection-mixed-geometry` · **Status:** open

## 1. Problem Statement

Given a dataset with a target metric or relational structure and a total dimension budget $d$, choose the **signature** of the embedding space: how many factors, of what type (hyperbolic, Euclidean, spherical), of what dimension, at what curvature. Product ("mixed-curvature") spaces $\mathbb{H}^{d_1}_{K_1}\times\cdots\times\mathbb{S}^{d_k}_{K_k}\times\mathbb{R}^{d_0}$ strictly generalize each single-curvature space, so the question is not *whether* they help but *how to pick one without fitting all of them*.

Three variants, with different difficulty:

- **Measurement.** Given data, estimate a curvature profile that predicts which signature will win. Requires an estimator that is well-defined on finite, noisy, non-metric data.
- **Method.** Learn the signature jointly with the embedding — curvatures by gradient descent, dimension allocation by search or a differentiable relaxation — at cost comparable to a single fit.
- **Theory.** Prove distortion separations: exhibit a family of metric spaces where a product signature achieves distortion $\varepsilon$ in $d$ dimensions and every single-curvature space of dimension $d$ needs $\gg\varepsilon$.

Solved would mean: a procedure that reads the data once, emits a signature, and matches within noise the best signature found by exhaustive search — with a proof or a broad reproduction that the estimator's ranking correlates with realized distortion.

## 2. Formal Setting

**Input.** A finite metric space $(X, d_X)$, $|X| = n$, usually the shortest-path metric of a graph $G$, or a similarity matrix. Measured as: all-pairs BFS distances for graphs ($O(nm)$), or a sampled subset of pairs when $n > 10^5$.

**Model space.** A product $\mathcal{M} = \prod_{i=1}^{k} M_i^{d_i}(K_i)$ with $\sum_i d_i = d$, each $M_i$ a constant-curvature model of curvature $K_i$ ($<0$ hyperbolic, $0$ Euclidean, $>0$ spherical). Distances combine in $\ell_2$:
$$d_\mathcal{M}(x,y)^2 = \sum_{i=1}^{k} d_{M_i}\!\left(x^{(i)}, y^{(i)}\right)^2 .$$
For the hyperboloid of curvature $K<0$, $d_{M}(x,y) = |K|^{-1/2}\operatorname{arcosh}(-K\langle x,y\rangle_{\mathcal L})$; for the sphere of curvature $K>0$, $d_M(x,y)=K^{-1/2}\arccos(K\langle x,y\rangle)$.

**Objective.** Average distortion and mean average precision (MAP) are the two reported quantities:
$$D_{\mathrm{avg}} = \frac{2}{n(n-1)}\sum_{x<y}\frac{\left|d_\mathcal{M}(f(x),f(y)) - d_X(x,y)\right|}{d_X(x,y)},\qquad
D_{\mathrm{wc}} = \frac{\max_{x,y} \frac{d_\mathcal{M}}{d_X}}{\min_{x,y}\frac{d_\mathcal{M}}{d_X}} .$$
MAP is rank-based and scale-free; $D_{\mathrm{avg}}$ is not. **The two disagree**: a signature can win on MAP and lose on $D_{\mathrm{avg}}$, so "best signature" is objective-dependent.

**Curvature estimators, as actually computed.**
- *Gromov $\delta$-hyperbolicity*: the four-point condition, exact cost $\Theta(n^4)$, in practice estimated from $10^5$–$10^6$ sampled quadruples; reported normalized as $\delta/\mathrm{diam}(X)$.
- *Discrete sectional curvature* (Gu et al., ICLR 2019): for a node $m$ with neighbours $b,c$ and a reference $a$, a Toponogov-style comparison $\xi(m;b,c;a)$ built from graph distances; averaged per node to give $\bar\xi(m)$, then histogrammed.
- *Ollivier–Ricci* $\kappa(u,v) = 1 - W_1(\mu_u,\mu_v)/d(u,v)$, one optimal-transport solve per edge; *Balanced Forman* curvature (Topping et al., ICLR 2022) is a closed-form $O(\deg^2)$ proxy.

**Assumptions and where they break.** (i) The data is metric — violated for asymmetric or intransitive similarity data. (ii) Curvature is spatially constant within a factor — violated for every real graph, which is why the estimators return distributions, not scalars. (iii) The $\ell_2$ product combination is the right one — it is a modelling choice, not derived. (iv) Curvature magnitude is identifiable — **it is not**, see §6.

## 3. State of the Art

**Established.**
- Product-space embeddings with per-factor learned curvature (Gu, Sala, Gunel, Ré, ICLR 2019) reduce $D_{\mathrm{avg}}$ relative to single-curvature spaces at equal total dimension on graphs that mix tree-like and cyclic structure. The signature itself is chosen by grid search over a handful of hand-picked candidates, not by an estimator.
- Hyperbolic space dominates Euclidean at low dimension on hierarchies: WordNet noun reconstruction, MAP $\approx 0.86$ at $d=5$ in the Poincaré ball versus far lower for Euclidean at the same $d$ (Nickel & Kiela, NeurIPS 2017); the Lorentz model raises this and improves optimization stability (Nickel & Kiela, ICML 2018).
- Combinatorial constructions give explicit trade-offs between distortion, dimension and *bits of precision* (Sala et al., ICML 2018): matching a tree's metric to distortion $\varepsilon$ in $\mathbb{H}^2$ requires coordinate precision growing with $\mathrm{diam}/\varepsilon$. Established, with proofs.

**Claimed but unablated.** That the empirical curvature estimator *selects* the winning signature. Gu et al. report the estimator and the wins, but the paper does not present a controlled study in which the estimator's ranking is scored against exhaustive-search ground truth across many graphs. Later work inherits the claim.

**Benchmark-number-only.** Mixed-curvature VAEs (Skopek, Ganea, Bécigneul, ICLR 2020) report likelihood gains for product latents on MNIST-class data; the gains are small relative to seed variance and the signature is again chosen from a short list. Constant-curvature GCNs ($\kappa$-GCN, Bachmann, Bécigneul, Ganea, ICML 2020) report node-classification numbers where the learned curvature often drifts toward $0$ without hurting accuracy — a null result for curvature *mattering* in that setting that is rarely quoted as such.

## 4. What Is Known

- **Trees embed into $\mathbb{H}^2$ with arbitrarily low distortion**, dimension-independent (Sarkar's construction, GD 2011); the cost is numerical precision, not dimension. Verified at $n\sim10^4$ nodes.
- **Cycles do not**: $C_n$ has $\Omega(\log n)$-type lower bounds against tree-metric approximation, and embeds isometrically into $\mathbb{S}^1$ of the matched radius.
- **Low-dimensional hyperbolic KG embeddings match high-dimensional Euclidean ones**: WN18RR MRR $\approx 0.52$ at $d=32$ (Chami et al., ACL 2020), against Euclidean baselines needing $d\ge 200$ for comparable MRR. Scale: $\sim4\times10^4$ entities.
- **Curvature learned by gradient descent is poorly conditioned.** Multiple papers report per-factor curvature converging to near-zero or to the initialization-dependent value; $\kappa$-GCN (ICML 2020) documents insensitivity of accuracy to the learned $\kappa$.
- **Graph curvature predicts a different thing well**: negative Balanced Forman curvature localizes over-squashing bottlenecks in GNNs (Topping et al., ICLR 2022), at $n\sim10^3$–$10^4$. This is established for *message passing*, not for *embedding distortion*.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed scalar summary of a curvature *distribution* that is predictive of signature quality. $\delta$-hyperbolicity, mean sectional curvature and Ollivier-Ricci mean disagree in sign on the same graph; no paper fixes one and validates it against exhaustive search.
- **Theoretically open.** No separation theorem of the form: a family $\{X_n\}$ where $\mathbb{H}^{d/2}\times\mathbb{S}^{d/2}$ achieves $D_{\mathrm{avg}}=O(\varepsilon)$ while every $d$-dimensional constant-curvature space suffers $\Omega(1)$. Lower bounds for products of symmetric spaces are largely absent.
- **Empirically open.** Whether signature choice survives scale. Every reported product-space win is at $d \le 100$ and $n \le 10^5$. Whether it persists at $d=768$ and $n=10^7$ — where Euclidean capacity is abundant — is runnable and unrun.
- **Theoretically open.** Identifiability: under what loss functions is $(K_i, d_i)$ recoverable at all (§6).

## 6. Why It Is Hard

**Non-identifiability of curvature magnitude.** For a single factor, $d_{M(K)}(x,y) = |K|^{-1/2} \cdot \rho(\text{unit-curvature distance})$. Rescaling $K \mapsto \lambda^2 K$ and the embedding radii inversely leaves all distances fixed. So any loss that is invariant to a global rescale of $d_\mathcal{M}$ — every rank-based loss, every loss with a learnable temperature, contrastive InfoNCE — cannot identify $|K|$. Only the *sign* and the *ratios* between factors are determined. Reported "learned curvatures" are therefore partly artifacts of the parameterization and the distance normalization.

**The estimator averages away the signal.** A graph that is half tree and half cycle has a bimodal curvature histogram whose mean is near zero. A mean-based selector picks Euclidean, which is worse than either factor alone. Distributional matching is the right move but has no established decision rule.

**Confounded measurement.** Product-space runs change three things at once relative to a Euclidean baseline: geometry, optimizer (Riemannian Adam), and initialization scale. Optimization difficulty in hyperbolic space near the boundary — where gradients vanish in float32 — is a known confounder for negative results.

Compute is *not* the obstruction: full grid search over signatures at $d=100$, $n=10^4$ costs GPU-hours.

## 7. Current Research (as of 2026)

- **Heterogeneous and learned-manifold embeddings.** Moving beyond constant-curvature factors to symmetric spaces of non-constant curvature and to per-point curvature. Groups around Ganea/Bécigneul's line (ETH) and Sala/Ré's line (Wisconsin/Stanford) originated this; work continues in the graph-representation community. *(frontier — verify current group activity.)*
- **Curvature-aware routing / mixture-of-geometries**, where a gate assigns points to factors rather than embedding every point in every factor. Reported gains remain small and seed-sensitive. *(frontier — verify.)*
- **Hyperbolic components in large vision-language and LLM embedding stacks**, motivated by hierarchy in image–text data. Whether measured gains exceed tuning variance at production dimension is the open point. *(frontier — verify.)*
- **Discrete curvature for GNN rewiring** is the healthiest sub-line, because it has a falsifiable target (over-squashing) rather than a distortion number.

## 8. Concrete Next Experiment

**Question.** Does any curvature estimator predict the exhaustive-search-optimal signature better than a constant baseline?

**Scale.** 60 graphs, $n \in [2\times10^3, 5\times10^4]$: 20 synthetic with known ground-truth geometry (trees, ring lattices, tree-of-cycles, torus grids), 40 real (citation, road, protein-interaction, web, product co-purchase). Fixed budget $d=32$. Signature grid: all $(k \le 3)$ partitions of 32 into factors from $\{\mathbb{H},\mathbb{E},\mathbb{S}\}$ with dimensions in $\{8,16,24,32\}$ — 43 signatures. 5 seeds each. Total $\approx 13{,}000$ fits, each minutes on one GPU.

**Control arm.** (a) Always-Euclidean $\mathbb{R}^{32}$; (b) always-$\mathbb{H}^{32}$; (c) a random signature; (d) an oracle that sees the best signature. Same Riemannian Adam, same initialization scale, same epoch count for every arm — this removes the optimizer confounder.

**Deciding number.** *Regret*: $R = D_{\mathrm{avg}}(\text{selected}) - D_{\mathrm{avg}}(\text{oracle})$, averaged over the 60 graphs, for each of four selectors (mean sectional curvature, $\delta$-hyperbolicity, Ollivier-Ricci histogram + nearest-neighbour rule, learned-curvature-from-one-fit). **If no estimator achieves mean regret below that of the always-Euclidean control by more than 2 standard errors, curvature selection is empirically unsolved and the estimator literature is not doing what it claims.** Report the same table under MAP to expose objective-dependence.

## 9. Key References

- **[Foundational]** Maximilian Nickel, Douwe Kiela. *Poincaré Embeddings for Learning Hierarchical Representations.* NeurIPS, 2017. — arXiv:1705.08039
- **[Foundational]** Maximilian Nickel, Douwe Kiela. *Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry.* ICML, 2018.
- **[Foundational]** Rik Sarkar. *Low Distortion Delaunay Embedding of Trees in Hyperbolic Plane.* Graph Drawing, 2011.
- **[Foundational]** Frederic Sala, Christopher De Sa, Albert Gu, Christopher Ré. *Representation Tradeoffs for Hyperbolic Embeddings.* ICML, 2018.
- **[SOTA]** Albert Gu, Frederic Sala, Beliz Gunel, Christopher Ré. *Learning Mixed-Curvature Representations in Product Spaces.* ICLR, 2019.
- **[SOTA]** Ondrej Skopek, Octavian-Eugen Ganea, Gary Bécigneul. *Mixed-curvature Variational Autoencoders.* ICLR, 2020.
- **[SOTA]** Gregor Bachmann, Gary Bécigneul, Octavian-Eugen Ganea. *Constant Curvature Graph Convolutional Networks.* ICML, 2020.
- **[SOTA]** Ines Chami, Adva Wolf, Da-Cheng Juan, Frederic Sala, Sujith Ravi, Christopher Ré. *Low-Dimensional Hyperbolic Knowledge Graph Embeddings.* ACL, 2020.
- **[Related]** Jake Topping, Francesco Di Giovanni, Benjamin Chamberlain, Xiaowen Dong, Michael Bronstein. *Understanding Over-Squashing and Bottlenecks on Graphs via Curvature.* ICLR, 2022.
- **[Survey]** Wei Peng, Tuomas Varanka, Abdelrahman Mostafa, Henglin Shi, Guoying Zhao. *Hyperbolic Deep Neural Networks: A Survey.* IEEE TPAMI, 2022.

## 10. Worked Example

Take the 4-cycle $C_4$ with unit edges: adjacent distance $1$, diagonal distance $2$.

**Euclidean $\mathbb{R}^2$.** Best symmetric placement is $(\pm a, 0), (0, \pm a)$. Diagonals $=2a$, adjacent $=a\sqrt{2}$. Matching diagonals gives $a=1$, adjacent $=1.414$, so $D_{\mathrm{wc}} = 1.414/1 = 1.414$ and $D_{\mathrm{avg}} = \tfrac{4}{6}(0.414) = 0.276$.

**Sphere $\mathbb{S}^1_K$.** Four equally spaced points on a circle of circumference $8$: adjacent arc $=1$, opposite arc $=2$. **Exactly isometric**, $D_{\mathrm{avg}}=0$. The radius is $r = 8/2\pi = 1.273$, so $K = 1/r^2 = 0.617$.

The obstruction is visible in that last number. $K=0.617$ is not a property of $C_4$; it is a property of $C_4$ *with unit edge length*. Rescale the target metric by $\lambda$ and the required curvature becomes $K/\lambda^2$. If the loss is rank-based (MAP) or has a learnable temperature, $\lambda$ is free, and every $K>0$ fits equally well — the fitted curvature reports the optimizer's starting point, not the data.

Now the second failure. Build a **tree-of-cycles**: a depth-4 binary tree, each node replaced by a $C_4$ gadget, $n = 60$. The sectional-curvature histogram is bimodal — negative on tree edges, positive inside gadgets — and the mean sits near $-0.05$, close enough to zero that a mean-based selector emits $\mathbb{R}^{32}$. Exhaustive search on this family instead favors $\mathbb{H}^{16}\times\mathbb{S}^{16}$, which recovers the two structures separately. A single scalar summary of curvature cannot express "half of each," which is exactly what the product space is for. That mismatch — between the statistic computed and the decision it is used for — is the live problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*