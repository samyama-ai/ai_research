---
id: 07-embeddings/hyperbolic-embedding-numerical-stability
title: "Numerical Stability Limits of Hyperbolic Embeddings"
topic: 07-embeddings
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Numerical Stability Limits of Hyperbolic Embeddings

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/hyperbolic-embedding-numerical-stability` · **Status:** solved-but-impractical

## 1. Problem Statement

Hyperbolic space embeds trees with arbitrarily low distortion in two dimensions, which Euclidean space cannot do at any dimension. The catch is arithmetic: the volume that makes the embedding possible grows exponentially with radius, so the coordinates that encode a deep hierarchy sit within a rounding error of each other in any fixed-precision float.

Three variants, with different difficulty:

- **Measurement.** Given a trained hyperbolic embedding, decide which reported gains come from the geometry and which from precision loss. No standard diagnostic exists; papers report task metrics, not distance error.
- **Method.** Train a hyperbolic embedding of a hierarchy of depth $\ell$ and branching factor $b$ at low distortion, on GPU, at a wall-clock cost within a small constant of float64 Riemannian SGD. Unsolved in practice.
- **Theory.** Characterize the exact trade-off between mantissa bits $p$, embedding dimension $n$, graph diameter, and achievable distortion. Partially settled (Sala et al. 2018); the lower-bound side is not tight.

Solving it means: a method that reaches reference-precision distortion on a depth-20 tree at float64-comparable throughput, plus a diagnostic that flags precision-limited results before publication.

## 2. Formal Setting

**Models.** The Poincaré ball $\mathbb{B}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$ with
$$d_{\mathbb{B}}(x,y) = \operatorname{arcosh}\!\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right),$$
and the Lorentz (hyperboloid) model $\mathbb{H}^n = \{x \in \mathbb{R}^{n+1} : \langle x,x\rangle_{\mathcal{L}} = -1,\ x_0 > 0\}$ with $\langle x,y\rangle_{\mathcal{L}} = -x_0y_0 + \sum_{i\ge1} x_iy_i$ and $d_{\mathbb{H}}(x,y) = \operatorname{arcosh}(-\langle x,y\rangle_{\mathcal{L}})$.

**Precision.** Let $p$ be mantissa bits ($p=24$ for float32, $53$ for float64, $106$ for double-double, $113$ for float128), $\epsilon_m = 2^{-p}$.

**Measured quantities.**
- *Representable radius.* $R_{\max}(p) = \sup\{d(0,x) : \operatorname{fl}(x) \neq \operatorname{fl}(\partial)\}$. For the Poincaré ball, $\|x\| = \tanh(d/2)$, so $1-\|x\| \approx 2e^{-d}$ and representability requires $2e^{-d} > \epsilon_m$:
$$R_{\max}(p) \approx p\ln 2 + \ln 2 \approx 0.69\,p.$$
- *Distance error.* $\eta(x,y) = |{\hat d}(x,y) - d(x,y)| / d(x,y)$, where $\hat d$ is the float computation and $d$ the value from a 512-bit MPFR reference on the same stored coordinates. Measured over a sample of $10^6$ node pairs.
- *Distortion.* $D = \max_{u \neq v} \frac{d(f(u),f(v))}{d_G(u,v)} \cdot \max_{u\neq v}\frac{d_G(u,v)}{d(f(u),f(v))}$; average distortion $D_{\mathrm{avg}}$ is the mean of $|d(f(u),f(v))/d_G(u,v) - 1|$.
- *Cost.* Wall-clock seconds per epoch on one A100 at fixed batch size, plus bytes per embedding vector.

**Assumptions, and which fail.**
1. *Coordinates carry the information.* False near the boundary: at $d(0,x) = 40$ the Poincaré coordinate stores $\approx 3.6\times10^{-35}$ of gap, below float64 resolution, so distinct leaves become bit-identical.
2. *The distance formula is backward-stable.* False. $-\langle x,y\rangle_{\mathcal{L}}$ subtracts two large near-equal terms — catastrophic cancellation — and $\operatorname{arcosh}(1+u)$ has derivative $\sim u^{-1/2}$ near $u=0$, amplifying it.
3. *Riemannian SGD retains the manifold constraint.* Approximately false: retraction/projection after each step re-clips $\|x\| \le 1-\delta$, and the standard library default $\delta = 10^{-5}$ caps radius at $2\operatorname{artanh}(1-10^{-5}) \approx 12.2$ — one third of the $36.7$ float64 allows.
4. *Distortion is the objective.* Not measured in most applied papers; task accuracy is.

## 3. State of the Art

**Theory SOTA (established).** Sala, De Sa, Gu, Ré, *Representation Tradeoffs for Hyperbolic Embeddings*, ICML 2018: an explicit combinatorial construction (a generalization of Sarkar's 2011 two-dimensional tree construction) achieving distortion $1+\varepsilon$, with an accompanying precision analysis showing the required number of bits grows linearly in the longest path length $\ell$ and logarithmically in max degree $d_{\max}$, and inversely in $\varepsilon$. This is the reference statement of the trade-off, with proof.

**Exact-arithmetic SOTA (established, narrow).** Yu and De Sa, *Numerically Accurate Hyperbolic Embeddings Using Tiling-Based Models*, NeurIPS 2019: parameterize points by integer coordinates in a regular tiling of $\mathbb{H}^2$, making distance computation exact and sidestepping $R_{\max}$ entirely. Follow-up: Yu and De Sa, *Representing Hyperbolic Space Accurately using Multi-Component Floats*, NeurIPS 2021, which uses multi-component float (unevaluated float sums, $\approx 106$-bit) arithmetic and is dimension-general.

**Stability analysis SOTA (established).** Mishne, Wan, Wang, Yang, *The Numerical Stability of Hyperbolic Representation Learning*, ICML 2023: analyzes error growth in the Poincaré ball versus the Lorentz model, finds the two fail in different regimes (Poincaré degrades near the boundary; Lorentz degrades in a different, larger-coordinate regime), and proposes a Euclidean-space reparameterization to avoid both.

**Claimed but unablated.** Applied hyperbolic pipelines — hyperbolic neural networks (Ganea, Bécigneul, Hofmann, NeurIPS 2018), HGCN (Chami, Ying, Ré, Leskovec, NeurIPS 2019), hyperbolic image embeddings (Khrulkov et al., CVPR 2020), Poincaré ResNet (van Spengler, Berkhout, Mettes, ICCV 2023) — report task gains but almost never report distance error or distortion against a high-precision reference, and run in float32 with clipping. Whether the reported gains survive at reference precision is a benchmark number, not an ablation.

Guo et al., *Clipped Hyperbolic Classifiers Are Super-Hyperbolic Classifiers*, CVPR 2022, is the partial exception: it shows a numerical artifact (vanishing gradients from boundary saturation) is the cause of a training failure, and that clipping the feature norm fixes it — evidence that measured hyperbolic gains are entangled with numerics.

## 4. What Is Known

- **Radius is linear in mantissa bits.** $R_{\max} \approx 0.69p$: $\approx 16.6$ for float32, $\approx 36.7$ for float64, $\approx 73$ for double-double, $\approx 78$ for float128. Derivation in §2; consistent with the boundary-saturation behavior reported by Sala et al. (2018).
- **The precision requirement is real, not an implementation defect.** Sala et al. (2018) prove it for the combinatorial construction — for fixed $\varepsilon$, bits scale $\Theta(\ell)$ in path length. Doubling tree depth doubles the required mantissa.
- **Exact tiling works in 2D.** Yu and De Sa (2019) recover the theoretical distortion of the combinatorial construction on tree benchmarks where float64 Poincaré training does not, using integer-parameterized tilings.
- **Lorentz is not uniformly better than Poincaré.** Nickel and Kiela (ICML 2018) reported better optimization behavior for the Lorentz model on WordNet; Mishne et al. (ICML 2023) showed the improvement is regime-dependent, not universal, with explicit error bounds per model.
- **Low-dimensional hyperbolic embeddings match high-dimensional Euclidean ones on shallow hierarchies.** Nickel and Kiela (NeurIPS 2017) reported mean average precision near $0.86$ on WordNet noun reconstruction (82,115 nodes, 743,241 transitive-closure edges) at dimension 10, versus much larger Euclidean dimension for comparable quality. WordNet's depth is under 20 and its effective radius stays inside float64's $36.7$, which is why float64 suffices there — this is the scale at which the field's headline result was measured, and it is a scale where the problem does not bite.

## 5. What Is Not Known

- **Empirically open.** Whether reported gains in applied hyperbolic deep learning (image embeddings, hyperbolic GCNs, hyperbolic transformers) survive at 106-bit reference precision. The experiment is runnable today; nobody has run it across a benchmark suite. This is the largest gap and the cheapest to close.
- **Empirically open.** Wall-clock cost of multi-component-float hyperbolic training against a float64 GPU baseline at $10^6$+ nodes. No fused CUDA kernel for multi-component hyperbolic operations is publicly available, so the honest current answer is "unmeasured at scale."
- **Theoretically open.** A matching lower bound: is $\Omega(\ell)$ bits *necessary* for *any* $n$-dimensional embedding at distortion $1+\varepsilon$, or only for the known constructions? Sala et al. bound their construction, not the problem.
- **Theoretically open.** Whether an alternative chart of $\mathbb{H}^n$ (log-polar, Klein, or a learned reparameterization) has representable radius growing faster than linearly in $p$. §2's derivation is chart-specific; no chart-independent impossibility result exists.
- **Methodologically blocked.** "Numerical stability of a hyperbolic model" has no agreed operational definition. Candidates — worst-case $\eta$, fraction of gradient steps producing NaN, distortion gap to reference — are not interconvertible and give different model rankings.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus a missing reference**. Every applied hyperbolic result mixes at least four knobs: curvature $c$, clipping radius $\delta$, float width, and the model (Poincaré/Lorentz/Klein). All four move distortion, and none of the four is reported as an ablation in most papers. Because there is no cheap high-precision reference run, "hyperbolic beat Euclidean" and "the clipping happened to regularize" are observationally equivalent — Guo et al. (CVPR 2022) is the case where they were separated, and the answer was partly the artifact.

Second obstruction: **cost asymmetry**. The fix is known — more bits — but 106-bit arithmetic has no tensor-core path. The gap is not intellectual; it is that a correct method costs an order of magnitude more per FLOP than an incorrect one that scores well on shallow benchmarks. Hence *solved-but-impractical*.

## 7. Current Research (as of 2026)

- **Reparameterization over higher precision.** The Mishne/Wan/Wang/Yang line (UC San Diego) — keep parameters in Euclidean coordinates, map to the manifold only where needed. Cheap, and the dominant direction.
- **Exact and multi-component arithmetic.** Yu and De Sa (Cornell) — tiling models and multi-component floats. Correct; adoption limited by kernel support.
- **Fully hyperbolic layers.** Chen et al., *Fully Hyperbolic Neural Networks*, ACL 2022 — Lorentz-native linear layers avoiding repeated exp/log map round-trips, which removes one error-amplifying step per layer.
- **Curvature learning and mixed-curvature products** as an implicit numerical fix: learning small $|c|$ shrinks the working radius. Whether this is a geometry gain or a precision workaround is unresolved. *(frontier — verify)*
- **Hyperbolic components inside large pretrained models** (hierarchy-aware retrieval heads, hyperbolic attention). Reported gains are small and, so far, not precision-ablated. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** At what hierarchy depth does float64 hyperbolic training start losing accuracy to arithmetic rather than to optimization?

**Scale.** Complete binary trees of depth $\ell \in \{8, 12, 16, 20, 24\}$ ($2^{\ell+1}-1$ nodes, up to $3.4\times10^7$ at $\ell=24$; run $\ell \le 20$, $2.1\times10^6$ nodes, if compute-limited). Dimension $n=10$. Riemannian SGD, 500 epochs, identical seeds and hyperparameters across arms. Plus WordNet nouns (82,115 nodes) as a real-data control.

**Arms.**
1. Poincaré, float32, clip $\delta = 10^{-5}$ (the common default).
2. Poincaré, float64, clip $\delta = 10^{-15}$.
3. Lorentz, float64.
4. **Control arm:** the same optimizer at 106-bit double-double precision (or MPFR at 128 bits, CPU, on a subsampled node set if needed). This is the reference; all error is measured against it.

**Deciding number.** $\ell^*$ = the smallest depth at which mean distortion $D_{\mathrm{avg}}$ of arm 3 exceeds the control arm's by more than $0.01$ absolute. Report $\ell^*$ per arm.

**Prediction.** From $R_{\max} \approx 0.69p$ and a Sarkar scaling $\tau \approx 2$–$4$ per edge, arm 1 breaks at $\ell^* \approx 5$–$8$ and arm 3 at $\ell^* \approx 12$–$18$. If arm 3's $\ell^*$ exceeds 24, the applied field's float64 practice is safe at realistic depths and the problem is confined to synthetic deep trees. If $\ell^* < 16$, a large fraction of published hyperbolic GCN results on deep hierarchies are precision-limited.

## 9. Key References

- **[Foundational]** Maximilian Nickel, Douwe Kiela. *Poincaré Embeddings for Learning Hierarchical Representations.* NeurIPS, 2017. — arXiv:1705.08039
- **[Foundational]** Rik Sarkar. *Low Distortion Delaunay Embedding of Trees in Hyperbolic Plane.* Graph Drawing (GD), 2011.
- **[SOTA — theory]** Frederic Sala, Christopher De Sa, Albert Gu, Christopher Ré. *Representation Tradeoffs for Hyperbolic Embeddings.* ICML, 2018. — arXiv:1804.03329
- **[SOTA — exact arithmetic]** Tao Yu, Christopher De Sa. *Numerically Accurate Hyperbolic Embeddings Using Tiling-Based Models.* NeurIPS, 2019.
- **[SOTA — precision]** Tao Yu, Christopher De Sa. *Representing Hyperbolic Space Accurately using Multi-Component Floats.* NeurIPS, 2021.
- **[SOTA — stability analysis]** Gal Mishne, Zhengchao Wan, Yusu Wang, Sheng Yang. *The Numerical Stability of Hyperbolic Representation Learning.* ICML, 2023. — arXiv:2211.00181
- **[Method]** Maximilian Nickel, Douwe Kiela. *Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry.* ICML, 2018. — arXiv:1806.03417
- **[Method]** Octavian-Eugen Ganea, Gary Bécigneul, Thomas Hofmann. *Hyperbolic Neural Networks.* NeurIPS, 2018. — arXiv:1805.09112
- **[Method]** Ines Chami, Rex Ying, Christopher Ré, Jure Leskovec. *Hyperbolic Graph Convolutional Neural Networks.* NeurIPS, 2019. — arXiv:1910.12933
- **[Evidence of confound]** Yunhui Guo, Xudong Wang, Yubei Chen, Stella X. Yu. *Clipped Hyperbolic Classifiers Are Super-Hyperbolic Classifiers.* CVPR, 2022.
- **[Method]** Weize Chen, Xu Han, Yankai Lin, Hexu Zhao, Zhiyuan Liu, Peng Li, Maosong Sun, Jie Zhou. *Fully Hyperbolic Neural Networks.* ACL, 2022.
- **[Survey]** Wenjie Peng, Tobias Varanka, Abdelrahman Mostafa, Henglin Shi, Guoying Zhao. *Hyperbolic Deep Neural Networks: A Survey.* IEEE TPAMI, 2022.

## 10. Worked Example

Embed a complete binary tree of depth $\ell = 20$ ($2{,}097{,}151$ nodes) by Sarkar's construction with edge scaling $\tau = 4$ — a modest value; smaller $\tau$ increases distortion because sibling subtrees crowd angularly.

**Step 1 — required radius.** A leaf sits at $d(0,x) = \ell\tau = 80$.

**Step 2 — Poincaré coordinate.** $\|x\| = \tanh(40)$, so
$$1 - \|x\| \approx 2e^{-80} = 3.6\times10^{-35}.$$

**Step 3 — compare to float64.** $\epsilon_m = 2^{-53} = 1.11\times10^{-16}$. The gap the coordinate must encode is $3.2\times10^{19}$ times smaller than the smallest gap float64 can hold near $\|x\|=1$. Every one of the $2^{20} = 1{,}048{,}576$ leaves rounds to the same stored value $\|x\| = 1.0$. Computed pairwise leaf distance: $\operatorname{arcosh}(1 + 0/0) \to$ NaN, or $0$ after the standard $\delta$-clip. True distance between two leaves in different halves: $160$.

**Step 4 — precision actually needed.** $p \ge 80/\ln 2 \approx 116$ bits. float128 (113 bits) is marginally short; double-double (106) is short; 128-bit MPFR suffices.

**Step 5 — what the default clip does.** With geoopt's default $\delta = 10^{-5}$, the maximum representable radius is $12.2$, i.e. depth $12.2/4 \approx 3$ levels of this tree. Levels 4 through 20 — $99.99\%$ of the nodes — are projected onto a shell of radius $12.2$ and become mutually near-equidistant.

**The obstruction made visible.** Training this embedding in float64 runs to completion, emits no warning, and produces a loss curve that looks converged, because the clip removes the NaNs. Reconstruction mAP restricted to the top 3 levels will look excellent. Only a comparison against the 128-bit control arm of §8 reveals that the bottom 17 levels carry no information. The failure is silent, and the standard metric does not name it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*