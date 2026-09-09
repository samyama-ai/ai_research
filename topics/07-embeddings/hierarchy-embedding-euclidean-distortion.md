---
id: 07-embeddings/hierarchy-embedding-euclidean-distortion
title: "Embedding Geometry of Hierarchies in Euclidean Space"
topic: 07-embeddings
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Embedding Geometry of Hierarchies in Euclidean Space

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/hierarchy-embedding-euclidean-distortion` · **Status:** partially-solved

## 1. Problem Statement

**Input.** A hierarchy given as a tree or DAG $T=(V,E)$ with $|V|=n$, carrying the shortest-path metric $d_T$.

**Output.** A map $f: V \to \mathbb{R}^d$ for a fixed budget $d$ (typically $d \in [2, 512]$, far below $n$).

**Objective.** Minimise distortion of $d_T$ under the Euclidean metric $\|f(u)-f(v)\|_2$.

Three variants that are routinely conflated:

- **Theory variant.** For a given tree family and a given $d$, what is $\inf_f D(f)$, the optimal worst-case distortion? Solved for $d$ unbounded, open for fixed small $d$.
- **Method variant.** Can a *trainable* Euclidean parametrisation reach the optimum that exists, using SGD from random init? This is where the observed hyperbolic-vs-Euclidean gap lives.
- **Measurement variant.** Is "hierarchy is preserved" the same predicate as "the metric is preserved"? Ranking metrics (MAP, mean rank) are invariant to any monotone transform of distance, so they do not measure distortion. A page-level claim that Euclidean space "fails at hierarchy" is usually an MAP claim, not a distortion claim.

**Solved** means: a distortion lower bound for fixed $d$ that matches a constructive upper bound, plus an ablation showing whether the empirical Euclidean deficit is geometric or optimisational.

## 2. Formal Setting

Let $f: V \to \mathbb{R}^d$ be non-contracting, i.e. $\|f(u)-f(v)\| \ge d_T(u,v)$. Define expansion and worst-case distortion:

$$D(f) = \max_{u \neq v} \frac{\|f(u)-f(v)\|_2}{d_T(u,v)}, \qquad D_{\mathrm{opt}}(T,d) = \inf_{f: V \to \mathbb{R}^d} D(f).$$

**Average distortion** (the quantity actually reported in the ML literature, Sala et al. 2018):

$$D_{\mathrm{avg}}(f) = \frac{1}{\binom{n}{2}} \sum_{u<v} \frac{\bigl|\,\|f(u)-f(v)\|_2 - d_T(u,v)\,\bigr|}{d_T(u,v)}.$$

Measured by enumerating all $\binom{n}{2}$ pairs (feasible to $n \approx 10^5$; sampled above that, which introduces variance nobody reports).

**Ranking score.** For $u$ with parent set $\mathcal{P}(u)$, MAP is computed from the rank of each $v \in \mathcal{P}(u)$ in the list of all nodes sorted by $\|f(u)-f(\cdot)\|$. Note $\mathrm{MAP}(f) = \mathrm{MAP}(\phi \circ f)$ for any strictly increasing $\phi$ applied to distances — MAP is a *topology* measure, $D_{\mathrm{avg}}$ is a *metric* measure.

**Hyperbolicity.** Gromov $\delta$ measured by the four-point condition, sampled over quadruples; $\delta=0$ for trees.

**Assumptions known to be violated in practice.**
- *The hierarchy is a tree.* WordNet's noun hypernym graph is a DAG with multiple inheritance; $\delta > 0$.
- *Precision is exact.* Hyperbolic constructions need $\Theta(\ell \log n)$ bits to realise their claimed distortion ($\ell$ = tree depth); float32 silently truncates this (Sala et al. 2018).
- *Non-contraction.* SGD-trained embeddings are not non-contracting, so reported $D_{\mathrm{avg}}$ mixes scale error with shape error unless a global scale is fitted first. Many papers do not fit it.

## 3. State of the Art

**Theory SOTA (established).**
- Bourgain (1985): every $n$-point metric embeds in $\ell_2$ with distortion $O(\log n)$; tight up to constants via expanders (Linial–London–Rabinovich, Combinatorica 1995).
- Bourgain (1986): the complete binary tree of depth $k$ needs distortion $\Omega(\sqrt{\log k})$ in any Hilbert space, i.e. $\Omega(\sqrt{\log\log n})$.
- Matoušek (1999): matching $O(\sqrt{\log\log n})$ upper bound for $n$-point trees into $\ell_2$. **In unbounded dimension the tree problem is closed, and the answer is "almost free".**
- Gupta (DCG 2000): trees into $\mathbb{R}^d$ at distortion polynomial in $n^{1/(d-1)}$ — the dimension-bounded regime, and the only regime that matters for ML. Bounds here are *not* tight.
- Sarkar (GD 2011): trees embed in $\mathbb{H}^2$ at distortion $1+\varepsilon$ for any $\varepsilon$, with bit-precision cost.

**Empirical SOTA (established).** Poincaré embeddings (Nickel & Kiela, NeurIPS 2017) and the combinatorial construction of Sala et al. (ICML 2018) dominate Euclidean baselines on WordNet nouns at $d \le 10$ by roughly an order of magnitude in MAP.

**Claimed but unablated.** That the gap is *geometric*. No published study fixes the Euclidean optimiser quality — same loss, same initialisation quality, same precision, global-method init — and re-measures. The comparison is between a well-engineered hyperbolic pipeline and a naive Euclidean one.

**Benchmark-number-only.** Downstream claims ("hyperbolic GNNs improve link prediction by $x$ points") exist as leaderboard deltas on Disease/Airport/PubMed (Chami et al. 2019) with $\delta$-hyperbolicity correlations, not as controlled tests of the embedding-geometry hypothesis.

## 4. What Is Known

- **Distortion for unbounded $d$:** $\Theta(\sqrt{\log\log n})$ for trees in $\ell_2$. At $n=10^6$ this is $\approx 1.7$ — a constant. Scale: asymptotic theorem, not a measurement.
- **WordNet nouns, $n = 82{,}115$** (Nickel & Kiela 2017): Poincaré reconstruction MAP rises from $\approx 0.02$–$0.06$ for Euclidean baselines at $d \in \{5, \ldots, 200\}$ to $\approx 0.82$ at $d=5$ and $\approx 0.87$ at $d=200$ for Poincaré. Mean rank drops from hundreds to single digits. This is the single most-cited number in the field.
- **Same graph** (Sala et al. 2018): a *non-learned* combinatorial construction reaches $\mathrm{MAP} \approx 0.99$ and $D_{\mathrm{avg}} \approx 0.08$ — beating SGD-trained Poincaré. Scale: full 82k-node hypernym closure. Establishes that on this benchmark the learning procedure, not the geometry, was the binding constraint for hyperbolic space too.
- **Precision:** the same work shows the required bits grow with tree depth, so a $d=2$ hyperbolic embedding at float64 caps achievable distortion for deep trees.
- **Product spaces** (Gu et al., ICLR 2019): mixed-curvature $\mathbb{H} \times \mathbb{S} \times \mathbb{E}$ products beat single-curvature spaces on graphs with mixed $\delta$ — evidence that "hierarchy" is not a single geometric signature.
- **Euclidean LLM spaces encode hierarchy without low distortion** (Park et al. 2024): categorical and hierarchical concepts in LLM representation spaces appear as orthogonal directions and polytopes, i.e. hierarchy carried by *direction* and *norm*, not by pairwise distance. Scale: Gemma-2B, LLaMA-3-8B, WordNet-derived concept sets.

## 5. What Is Not Known

- **Theoretically open.** $D_{\mathrm{opt}}(T, d)$ for the complete binary tree at fixed small $d$ (say $d \in [5,100]$). Gupta's upper bound and packing lower bounds are separated by large polynomial factors. No proof that $\mathbb{R}^{10}$ *cannot* embed WordNet's hierarchy at $D_{\mathrm{avg}} < 0.1$.
- **Empirically open.** Whether a strongly-optimised Euclidean embedding at $d=10$ closes the WordNet gap. Runnable today on one GPU-day; nobody has published it. The hyperbolic side already showed a $10\times$ MAP jump from better optimisation alone (Sala et al.), so the prior that the Euclidean side has similar headroom is not idle.
- **Methodologically blocked.** "Preserves hierarchy" has no agreed measurement. MAP, $D_{\mathrm{avg}}$, worst-case $D$, and ancestor-descendant classification accuracy disagree in ordering, and MAP is provably blind to monotone distance rescaling — precisely the degree of freedom that separates $\mathbb{H}^d$ from $\mathbb{R}^d$ (the exponential radial map). Until a metric is fixed, the field cannot say what it is measuring.

## 6. Why It Is Hard

**Named obstruction: the headline evaluation does not measure the named quantity, and this hides a non-identifiability.**

$\mathbb{H}^d$ differs from $\mathbb{R}^d$ mainly by the exponential map $r \mapsto \sinh r$ on the radial coordinate. A Euclidean embedding composed with a monotone radial rescaling has *identical MAP* and *very different distortion*. So a MAP gap is consistent with two incompatible causes — genuinely insufficient Euclidean volume, or a Euclidean parametrisation whose gradients vanish for the exponentially many far-apart leaves — and MAP cannot distinguish them. Compounding factors:

- **Absent ground truth.** $D_{\mathrm{opt}}(T,d)$ is not computable at $n=10^5$; there is no reference against which to score a learned embedding, only against other learned embeddings.
- **Confounded optimisation.** Riemannian SGD, burn-in learning rates, negative-sample counts, and initialisation scale all differ between the arms in published comparisons.
- **Weak lower bounds.** Packing arguments (Section 10) go vacuous by $d \approx 8$, so no theory currently forbids the Euclidean result that experiments fail to find.

## 7. Current Research (as of 2026)

- **Mixed-curvature and learned-curvature spaces.** Descendants of Gu/Sala/Ré (Stanford) and the hyperbolic-GNN line (Chami, Leskovec); curvature treated as a learned parameter per factor.
- **Hierarchy in LLM representation spaces.** Veitch's group (Chicago) and the linear-representation-hypothesis literature: hierarchy as orthogonality and polytope geometry in Euclidean activation space, which sidesteps distortion entirely. This is the most likely source of a reframing.
- **Hyperbolic vision/multimodal encoders** (MERU-style entailment-cone image-text models). Reported gains are benchmark deltas; controlled Euclidean-with-equal-tuning arms are still rare *(frontier — verify)*.
- **Critiques of the tree premise.** Sonthalia & Gilbert (NeurIPS 2020) showed hyperbolic embeddings can be recovered as low-dimensional structure without an explicit tree, weakening the "hierarchy ⇒ hyperbolic" inference.
- **Precision-aware embedding.** Fixed-point and higher-precision arithmetic for deep-tree hyperbolic embeddings *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Is the Euclidean deficit at low $d$ geometric or optimisational?

**Scale.** WordNet noun hypernym transitive closure, $n = 82{,}115$, $d = 10$ fixed. Report full $\binom{n}{2} \approx 3.4\times10^9$ pair distortion (chunked; ~1 GPU-day).

**Arms.**
1. *Reference:* Sala et al. combinatorial hyperbolic construction, $d=10$, float64.
2. *Naive Euclidean control:* the standard baseline — random init, Euclidean SGD on the same soft-ranking loss. Reproduces the historical $\mathrm{MAP} \approx 0.05$ number.
3. *Strong Euclidean arm:* classical MDS on $d_T$ for init, then Adam directly on $D_{\mathrm{avg}}$ (not the ranking loss), float64, 20 restarts, learned global scale, and a radially-warped readout $f(v) \mapsto \psi(\|f(v)\|)\hat f(v)$ with $\psi$ monotone and learned.
4. *Ablation:* arm 3 without the radial warp, to isolate the exponential-map degree of freedom.

**Deciding number.** The best $D_{\mathrm{avg}}$ from arm 3, against the hyperbolic reference $D_{\mathrm{avg}}^{\mathbb{H}} \approx 0.08$.

- $D_{\mathrm{avg}}^{\mathbb{R}} < 0.16$ (within $2\times$): the gap is optimisational; $\mathbb{R}^{10}$ has the capacity and the literature's headline comparison is unsound.
- $D_{\mathrm{avg}}^{\mathbb{R}} > 0.5$ across all restarts: the gap is geometric, and the result becomes the first empirical lower-bound evidence for $D_{\mathrm{opt}}(T, 10)$.

## 9. Key References

- **[Foundational]** J. Bourgain. *On Lipschitz embedding of finite metric spaces in Hilbert space.* Israel Journal of Mathematics 52, 1985.
- **[Foundational]** J. Bourgain. *The metrical interpretation of superreflexivity in Banach spaces.* Israel Journal of Mathematics 56, 1986.
- **[Foundational]** N. Linial, E. London, Y. Rabinovich. *The geometry of graphs and some of its algorithmic applications.* Combinatorica 15(2), 1995.
- **[Foundational]** J. Matoušek. *On embedding trees into uniformly convex Banach spaces.* Israel Journal of Mathematics 114, 1999.
- **[Foundational]** A. Gupta. *Embedding tree metrics into low-dimensional Euclidean spaces.* Discrete & Computational Geometry 24, 2000 (STOC 1999).
- **[Foundational]** R. Sarkar. *Low distortion Delaunay embedding of trees in hyperbolic plane.* Graph Drawing, 2011.
- **[SOTA]** M. Nickel, D. Kiela. *Poincaré embeddings for learning hierarchical representations.* NeurIPS 2017. — arXiv:1705.08039
- **[SOTA]** F. Sala, C. De Sa, A. Gu, C. Ré. *Representation tradeoffs for hyperbolic embeddings.* ICML 2018. — arXiv:1804.03329
- **[SOTA]** O. Ganea, G. Bécigneul, T. Hofmann. *Hyperbolic entailment cones for learning hierarchical embeddings.* ICML 2018. — arXiv:1804.01882
- **[SOTA]** M. Nickel, D. Kiela. *Learning continuous hierarchies in the Lorentz model of hyperbolic geometry.* ICML 2018. — arXiv:1806.03417
- **[SOTA]** A. Gu, F. Sala, B. Gunel, C. Ré. *Learning mixed-curvature representations in product spaces.* ICLR 2019.
- **[SOTA]** I. Chami, R. Ying, C. Ré, J. Leskovec. *Hyperbolic graph convolutional neural networks.* NeurIPS 2019. — arXiv:1910.12933
- **[SOTA]** K. Park, Y. J. Choe, Y. Jiang, V. Veitch. *The geometry of categorical and hierarchical concepts in large language models.* 2024. — arXiv:2406.01506
- **[Contrarian]** R. Sonthalia, A. C. Gilbert. *Tree! I am no Tree! I am a low dimensional hyperbolic embedding.* NeurIPS 2020.
- **[Survey]** P. Indyk, J. Matoušek. *Low-distortion embeddings of finite metric spaces.* Handbook of Discrete and Computational Geometry, 2nd ed., 2004.
- **[Survey]** W. Peng, T. Varanka, A. Mostafa, H. Shi, G. Zhao. *Hyperbolic deep neural networks: a survey.* IEEE TPAMI 44(12), 2022.

## 10. Worked Example

**Instance.** Star $K_{1,m}$ with $m=1024$ leaves: centre-to-leaf $=1$, leaf-to-leaf $=2$.

Take $f$ non-contracting with distortion $D$. Place the centre at $c$. Every leaf lies in the ball $B(c, D)$, and any two leaves are $\ge 2$ apart. Volume-packing in $\mathbb{R}^d$ gives

$$m \le \left(\frac{D + 1}{1}\right)^d \;\Longrightarrow\; D \ge m^{1/d} - 1.$$

- $d = 2$: $D \ge 1024^{1/2} - 1 = 31$. Strong.
- $d = 10$: $D \ge 1024^{0.1} - 1 = 2 - 1 = 1.0$. **Vacuous** — distortion $1$ is a perfect embedding.

**Now the complete binary tree of depth $k=10$** ($n = 2047$, $1024$ leaves, diameter $20$). Leaves are $\ge 2$ apart and lie in a ball of radius $kD = 10D$. Packing gives $1024 \le (1+10D)^d$, so $D \ge (2^{10/d}-1)/10$. At $d=10$ that is $D \ge 0.1$ — again vacuous.

**The obstruction, made visible.** At $d=10$ no known lower bound rules out a near-isometric Euclidean embedding of a 1024-leaf hierarchy. Yet the measured Euclidean baseline on WordNet at $d=5$–$10$ sits at $\mathrm{MAP}\approx 0.05$ against $0.82$ for Poincaré. Theory permits what practice fails to find, by a wide margin. So the observed gap is *not currently explained by geometry* — it is explained by nothing at all, which is exactly why the field's central empirical claim ("hierarchies need hyperbolic space") remains unproven rather than false. Section 8 is the cheapest way to find out which.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*